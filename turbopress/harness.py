"""Measurement harness: does unbiased quantization pay for itself?

Two experiments, both against exact full-precision references:

1. ``single-layer``: for one linear layer, decompose the output error of each
   quantization config into *bias* (the component that survives averaging
   over the method's randomness -- sketch seeds for QJL, rounding draws for
   stochastic rounding) and *noise* (the rest). The QJL theory predicts
   rel_bias -> 0 as trials grow for corrected configs, while nearest rounding
   without correction has rel_bias == rel_rmse (its error is deterministic).

2. ``compounding``: a depth-L ReLU MLP chain plus a classifier head, fully
   quantized, measuring hidden-state error growth with depth, final softmax
   KL to the full-precision model, and top-1 agreement. This is the test of
   the core hypothesis: deterministic per-layer bias should compound faster
   than the zero-mean noise left after QJL correction, at matched bit budgets
   (e.g. 2 bits + k=d sketch  vs  3 bits uncorrected, both ~3 bits/weight).

Activations come in two flavors: ``isotropic`` Gaussian, and ``aniso`` --
a fixed mean direction plus a dominant low-rank subspace plus noise, which
mimics real LLM activations (massive/sink activations, anisotropic
covariance). The fixed mean is where deterministic quantization bias hurts
most: it shifts every token's output the same way.

Run:  python -m turbopress.harness [--quick] [--out results/harness_results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from turbopress.hadamard import RandomizedOrthogonal
from turbopress.linear import QJLCorrectedLinear
from turbopress.qjl import build_qjl_sketch, correction_matrix
from turbopress.quantizer import quantize_rows

__all__ = ["QuantConfig", "make_activations", "run_single_layer", "run_compounding", "main"]


@dataclass(frozen=True)
class QuantConfig:
    """One quantization configuration under test."""

    label: str
    bits: int
    sketch_k: int
    rounding: str  # "nearest" | "stochastic"

    def effective_bits(self, dim: int) -> float:
        """Bits/weight including per-row scale, sketch bits, and norms (fp16)."""
        overhead = 16 + (self.sketch_k + 16 if self.sketch_k > 0 else 0)
        return self.bits + overhead / dim


def make_activations(
    batch: int,
    dim: int,
    mode: str,
    seed: int,
    mean_frac: float = 0.3,
    subspace_frac: float = 0.5,
    subspace_rank: int = 8,
) -> Tensor:
    """Sample activations with per-coordinate RMS ~= 1.

    ``isotropic``: iid N(0, 1).
    ``aniso``: fixed mean direction (``mean_frac`` of the energy, identical
    for every sample) + dominant low-rank subspace (``subspace_frac``) +
    isotropic noise (the remainder). Mimics LLM activation structure.
    """
    gen = torch.Generator().manual_seed(seed)
    if mode == "isotropic":
        return torch.randn(batch, dim, generator=gen)
    if mode != "aniso":
        raise ValueError(f"unknown activation mode {mode!r}")

    noise_frac = 1.0 - mean_frac - subspace_frac
    if noise_frac < 0:
        raise ValueError("mean_frac + subspace_frac must be <= 1")

    mu = torch.randn(dim, generator=gen)
    mu = mu / mu.norm()
    basis = torch.linalg.qr(torch.randn(dim, subspace_rank, generator=gen))[0].T

    mean_part = math.sqrt(mean_frac * dim) * mu
    coeffs = torch.randn(batch, subspace_rank, generator=gen)
    subspace_part = math.sqrt(subspace_frac * dim / subspace_rank) * (coeffs @ basis)
    noise_part = math.sqrt(noise_frac) * torch.randn(batch, dim, generator=gen)
    return mean_part + subspace_part + noise_part


def _rel_fro(err: Tensor, ref: Tensor) -> float:
    return float(err.norm() / ref.norm())


def run_single_layer(
    configs: list[QuantConfig],
    dim: int = 1024,
    n_out: int = 1024,
    batch: int = 512,
    n_trials: int = 24,
    act_mode: str = "isotropic",
    seed: int = 0,
) -> list[dict]:
    """Bias/variance decomposition of one quantized layer's output error.

    For each config, ``n_trials`` outputs are produced varying only the
    config's own randomness (sketch seed, or rounding draws); the rotation
    and, for nearest rounding, the code assignment stay fixed. Reported:

      rel_rmse  -- sqrt(mean_t ||y_t - y||^2) / ||y||          (total error)
      rel_bias  -- ||mean_t y_t - y|| / ||y||                  (systematic part)

    Deterministic configs (nearest, k=0) have one trial and rel_bias == rel_rmse.
    """
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(n_out, dim, generator=gen) / math.sqrt(dim)
    x = make_activations(batch, dim, act_mode, seed=seed + 1)
    y_ref = x @ w.T

    rotation = RandomizedOrthogonal(dim, seed=seed + 2)
    w_rot = rotation(w)
    z = rotation(x)

    results = []
    for cfg in configs:
        outputs: list[Tensor] = []
        if cfg.rounding == "nearest":
            quantized = quantize_rows(w_rot, bits=cfg.bits, rounding="nearest")
            w_hat = quantized.dequantize()
            base = z @ w_hat.T
            if cfg.sketch_k == 0:
                outputs.append(base)
            else:
                residual = w_rot - w_hat
                for t in range(n_trials):
                    sketch = build_qjl_sketch(residual, cfg.sketch_k, seed=seed + 100 + t)
                    corr = (z @ sketch.proj.T) @ correction_matrix(sketch).T
                    outputs.append(base + corr)
        elif cfg.rounding == "stochastic":
            for t in range(n_trials):
                qgen = torch.Generator().manual_seed(seed + 200 + t)
                quantized = quantize_rows(
                    w_rot, bits=cfg.bits, rounding="stochastic", generator=qgen
                )
                outputs.append(z @ quantized.dequantize().T)
        else:
            raise ValueError(f"unknown rounding {cfg.rounding!r}")

        stack = torch.stack(outputs)
        errs = stack - y_ref
        rel_rmse = float(errs.pow(2).mean(dim=0).sum().sqrt() / y_ref.norm())
        rel_bias = _rel_fro(stack.mean(dim=0) - y_ref, y_ref)
        results.append(
            {
                "label": cfg.label,
                "act_mode": act_mode,
                "eff_bits": round(cfg.effective_bits(dim), 4),
                "rel_rmse": rel_rmse,
                "rel_bias": rel_bias,
                "n_trials": len(outputs),
            }
        )
    return results


class _Chain(nn.Module):
    """Depth-L ReLU MLP with a classifier head, float reference version."""

    def __init__(self, depth: int, dim: int, vocab: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            layer = nn.Linear(dim, dim, bias=False)
            # He-scaled so ReLU activations keep unit variance with depth.
            layer.weight.data = torch.randn(dim, dim, generator=gen) * math.sqrt(2.0 / dim)
            self.layers.append(layer)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight.data = torch.randn(vocab, dim, generator=gen) / math.sqrt(dim)

    def forward_with_states(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        states = []
        h = x
        for layer in self.layers:
            h = F.relu(layer(h))
            states.append(h)
        return self.head(h), states


def _quantize_chain(chain: _Chain, cfg: QuantConfig, seed: int) -> nn.Module:
    """Return a copy of ``chain`` with every linear (incl. head) quantized."""
    q = nn.Module()
    q.layers = nn.ModuleList(
        QJLCorrectedLinear.from_linear(
            layer, bits=cfg.bits, sketch_k=cfg.sketch_k, seed=seed + i, rounding=cfg.rounding
        )
        for i, layer in enumerate(chain.layers)
    )
    head_k = min(cfg.sketch_k, chain.head.in_features)
    q.head = QJLCorrectedLinear.from_linear(
        chain.head, bits=cfg.bits, sketch_k=head_k, seed=seed + len(chain.layers),
        rounding=cfg.rounding,
    )
    return q


def _forward_quant(q: nn.Module, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    states = []
    h = x
    for layer in q.layers:
        h = F.relu(layer(h))
        states.append(h)
    return q.head(h), states


def run_compounding(
    configs: list[QuantConfig],
    depth: int = 16,
    dim: int = 512,
    batch: int = 256,
    vocab: int = 256,
    act_mode: str = "aniso",
    seed: int = 0,
    checkpoints: tuple[int, ...] = (1, 2, 4, 8, 16),
) -> list[dict]:
    """Error growth with depth for fully quantized chains.

    Reports the relative hidden-state error at each checkpoint depth, the
    mean KL divergence KL(p_fp || p_quant) of the final softmax, and top-1
    agreement with the full-precision chain.
    """
    checkpoints = tuple(c for c in checkpoints if c <= depth)
    chain = _Chain(depth, dim, vocab, seed=seed)
    x = make_activations(batch, dim, act_mode, seed=seed + 1)
    with torch.no_grad():
        logits_ref, states_ref = chain.forward_with_states(x)
    log_p = F.log_softmax(logits_ref, dim=-1)
    p = log_p.exp()

    results = []
    for cfg in configs:
        q = _quantize_chain(chain, cfg, seed=seed + 10_000)
        with torch.no_grad():
            logits_q, states_q = _forward_quant(q, x)
        depth_err = {
            str(c): _rel_fro(states_q[c - 1] - states_ref[c - 1], states_ref[c - 1])
            for c in checkpoints
        }
        log_q = F.log_softmax(logits_q, dim=-1)
        kl = float((p * (log_p - log_q)).sum(dim=-1).mean())
        top1 = float((logits_q.argmax(-1) == logits_ref.argmax(-1)).float().mean())
        results.append(
            {
                "label": cfg.label,
                "act_mode": act_mode,
                "eff_bits": round(cfg.effective_bits(dim), 4),
                "depth_rel_err": depth_err,
                "final_kl": kl,
                "top1_agreement": top1,
            }
        )
    return results


def _print_table(rows: list[dict], columns: list[tuple[str, str]], title: str) -> None:
    print(f"\n=== {title} ===")
    widths = [
        max(len(header), *(len(_fmt(r.get(key))) for r in rows)) for key, header in columns
    ]
    header_line = "  ".join(h.ljust(w) for (_, h), w in zip(columns, widths))
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(_fmt(r.get(k)).ljust(w) for (k, _), w in zip(columns, widths)))


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def _default_configs(dim: int) -> list[QuantConfig]:
    return [
        QuantConfig("nearest 4b (anchor)", bits=4, sketch_k=0, rounding="nearest"),
        QuantConfig("nearest 3b", bits=3, sketch_k=0, rounding="nearest"),
        QuantConfig("nearest 2b", bits=2, sketch_k=0, rounding="nearest"),
        QuantConfig("stochastic 2b", bits=2, sketch_k=0, rounding="stochastic"),
        QuantConfig("2b + QJL k=d/8", bits=2, sketch_k=dim // 8, rounding="nearest"),
        QuantConfig("2b + QJL k=d (=3b)", bits=2, sketch_k=dim, rounding="nearest"),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small sizes for a fast smoke run")
    parser.add_argument("--out", default="results/harness_results.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    if args.quick:
        sl_kwargs = dict(dim=256, n_out=256, batch=128, n_trials=8)
        cp_kwargs = dict(depth=8, dim=128, batch=64, vocab=64, checkpoints=(1, 2, 4, 8))
    else:
        sl_kwargs = dict(dim=1024, n_out=1024, batch=512, n_trials=24)
        cp_kwargs = dict(depth=16, dim=512, batch=256, vocab=256)

    t0 = time.time()
    all_results: dict[str, list[dict]] = {"single_layer": [], "compounding": []}

    sl_configs = _default_configs(sl_kwargs["dim"])
    for mode in ("isotropic", "aniso"):
        all_results["single_layer"] += run_single_layer(
            sl_configs, act_mode=mode, seed=args.seed, **sl_kwargs
        )
    _print_table(
        all_results["single_layer"],
        [
            ("label", "config"),
            ("act_mode", "activations"),
            ("eff_bits", "bits/w"),
            ("rel_rmse", "rel RMSE"),
            ("rel_bias", "rel bias"),
            ("n_trials", "trials"),
        ],
        "Single layer: output error decomposition (lower is better)",
    )

    cp_configs = _default_configs(cp_kwargs["dim"])
    for mode in ("isotropic", "aniso"):
        all_results["compounding"] += run_compounding(
            cp_configs, act_mode=mode, seed=args.seed, **cp_kwargs
        )
    rows = []
    for r in all_results["compounding"]:
        flat = {k: v for k, v in r.items() if k != "depth_rel_err"}
        for c, e in r["depth_rel_err"].items():
            flat[f"err@L{c}"] = e
        rows.append(flat)
    depth_cols = [(f"err@L{c}", f"err@L{c}") for c in all_results["compounding"][0]["depth_rel_err"]]
    _print_table(
        rows,
        [("label", "config"), ("act_mode", "activations"), ("eff_bits", "bits/w")]
        + depth_cols
        + [("final_kl", "KL(fp||q)"), ("top1_agreement", "top1 agree")],
        "Depth compounding: fully quantized chain vs full precision",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "quick": args.quick,
            "seed": args.seed,
            "single_layer": sl_kwargs,
            "compounding": cp_kwargs,
        },
        "results": all_results,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved results to {out_path} ({payload['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
