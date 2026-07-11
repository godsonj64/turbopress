"""Real-model A/B: does QJL bias correction pay off on a pretrained LLM?

The synthetic depth-compounding experiment (harness.py) refuted the QJL
hypothesis, but it was a worst case: no residual connections, no LayerNorm.
This experiment repeats the A/B on a real pretrained causal LM (Llama-family
via transformers): every ``nn.Linear`` inside the decoder blocks (attention
and MLP projections) is quantized; embeddings, lm_head, and norms stay in
float. Metrics against the full-precision model on real text:

  * mean token-level KL( p_fp || p_quant ) -- the "visible loss" metric
  * top-1 next-token agreement with the fp model
  * perplexity of both models on the eval text

Sketch sizes scale with each layer's in_features (``sketch_ratio``: k =
ratio * d), since real layers have mixed dimensions. Reported bits/weight are
the exact storage-weighted average over all quantized layers.

Run:  python -m turbopress.real_model [--model HuggingFaceTB/SmolLM2-135M]
      [--seqs 16] [--seqlen 256] [--out results/real_model_results.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from turbopress.linear import QJLCorrectedLinear

__all__ = [
    "RealConfig",
    "collect_input_scales",
    "quantize_model_copy",
    "evaluate_pair",
    "main",
]

_TEXT_URLS = [
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",
    "https://www.gutenberg.org/files/11/11-0.txt",
]


@dataclass(frozen=True)
class RealConfig:
    """One quantization configuration; k = round(sketch_ratio * in_features)."""

    label: str
    bits: int
    sketch_ratio: float = 0.0
    rounding: str = "nearest"
    method: str = "scalar"
    n_states: int = 8
    equilibrate: bool = False
    equil_mode: str = "quarter"
    equil_alpha: float = 0.25
    error_feedback: bool = False


DEFAULT_CONFIGS = [
    # Baselines (TurboQuant-style rotated scalar quantizer, biased rounding).
    RealConfig("nearest 2b", bits=2),
    RealConfig("nearest 3b", bits=3),
    RealConfig("nearest 4b", bits=4),
    # Round-2 winners: trellis coding + rotation-aware equilibration.
    RealConfig("tcq 2b S=64 +eq", bits=2, method="tcq", n_states=64, equilibrate=True),
    RealConfig("tcq 3b S=64 +eq", bits=3, method="tcq", n_states=64, equilibrate=True),
    RealConfig("tcq 4b S=64 +eq", bits=4, method="tcq", n_states=64, equilibrate=True),
    # New: GPTQ/LDLQ error feedback on the rotated scalar quantizer.
    RealConfig("gptq 2b +eq", bits=2, equilibrate=True, error_feedback=True),
    RealConfig("gptq 3b +eq", bits=3, equilibrate=True, error_feedback=True),
    RealConfig("gptq 4b +eq", bits=4, equilibrate=True, error_feedback=True),
    # Round 4: block-LDLQ error feedback OVER the trellis (QTIP-style) --
    # the trellis codes each column block jointly while LDLQ feeds the block's
    # Hessian-weighted error forward. Same rate and stored form as tcq.
    RealConfig("tcq+ef 2b S=64 +eq", bits=2, method="tcq", n_states=64,
               equilibrate=True, error_feedback=True),
    RealConfig("tcq+ef 3b S=64 +eq", bits=3, method="tcq", n_states=64,
               equilibrate=True, error_feedback=True),
    RealConfig("tcq+ef 4b S=64 +eq", bits=4, method="tcq", n_states=64,
               equilibrate=True, error_feedback=True),
]

# Focused head-to-head for the round-4 combination: the two prior winners vs
# the combined method at the bit-widths where they differ most.
EF_CONFIGS = [
    RealConfig("tcq 2b S=64 +eq", bits=2, method="tcq", n_states=64, equilibrate=True),
    RealConfig("gptq 2b +eq", bits=2, equilibrate=True, error_feedback=True),
    RealConfig("tcq+ef 2b S=64 +eq", bits=2, method="tcq", n_states=64,
               equilibrate=True, error_feedback=True),
    RealConfig("tcq 3b S=64 +eq", bits=3, method="tcq", n_states=64, equilibrate=True),
    RealConfig("gptq 3b +eq", bits=3, equilibrate=True, error_feedback=True),
    RealConfig("tcq+ef 3b S=64 +eq", bits=3, method="tcq", n_states=64,
               equilibrate=True, error_feedback=True),
    RealConfig("tcq 4b S=64 +eq", bits=4, method="tcq", n_states=64, equilibrate=True),
    RealConfig("tcq+ef 4b S=64 +eq", bits=4, method="tcq", n_states=64,
               equilibrate=True, error_feedback=True),
]

# Controlled equilibration-exponent comparison (Proposition 1): identical
# rotated scalar quantizer, the ONLY difference is the activation fold ---
# AWQ/SmoothQuant square root (equil_mode="awq") vs the rotation-aware quarter
# power (equil_mode="quarter"). "no-eq" is the unequilibrated floor.
EQUIL_CONFIGS = [
    RealConfig("scalar 2b no-eq", bits=2),
    RealConfig("scalar 2b awq-sqrt", bits=2, equilibrate=True, equil_mode="awq"),
    RealConfig("scalar 2b quarter", bits=2, equilibrate=True, equil_mode="quarter"),
    RealConfig("scalar 3b no-eq", bits=3),
    RealConfig("scalar 3b awq-sqrt", bits=3, equilibrate=True, equil_mode="awq"),
    RealConfig("scalar 3b quarter", bits=3, equilibrate=True, equil_mode="quarter"),
    RealConfig("scalar 4b no-eq", bits=4),
    RealConfig("scalar 4b awq-sqrt", bits=4, equilibrate=True, equil_mode="awq"),
    RealConfig("scalar 4b quarter", bits=4, equilibrate=True, equil_mode="quarter"),
]

# Proposition 2 sweep: does error feedback move the optimal equilibration
# exponent? s_j = m_j^a / c_j^(1/2-a). Theory: without EF the optimum is
# a = 1/4 (Prop 1); with ideal EF the activation term drops out and the
# optimum slides to a = 0 (pure column normalization). The no-EF rows are
# the control; their optimum should stay at 1/4.
PROP2_CONFIGS = [
    cfg
    for a in (0.0, 0.125, 0.25)
    for cfg in (
        RealConfig(f"tcq 3b a={a:g}", bits=3, method="tcq", n_states=64,
                   equilibrate=True, equil_alpha=a),
        RealConfig(f"tcq+ef 3b a={a:g}", bits=3, method="tcq", n_states=64,
                   equilibrate=True, equil_alpha=a, error_feedback=True),
        RealConfig(f"tcq+ef 2b a={a:g}", bits=2, method="tcq", n_states=64,
                   equilibrate=True, equil_alpha=a, error_feedback=True),
    )
]

# The QJL/stochastic configs from the first A/B (verdict: refuted; see README)
# are kept runnable for reproduction but excluded from the default sweep.
LEGACY_CONFIGS = [
    RealConfig("stochastic 2b", bits=2, rounding="stochastic"),
    RealConfig("2b + QJL k=d/8", bits=2, sketch_ratio=0.125),
    RealConfig("2b + QJL k=d (=3b)", bits=2, sketch_ratio=1.0),
]


def _decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder block list of a Llama-family causal LM."""
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    raise ValueError(
        f"could not locate decoder layers on {type(model).__name__}; "
        "expected model.layers / transformer.h / model.decoder.layers"
    )


@torch.no_grad()
def collect_input_scales(
    model: nn.Module, batches: list[Tensor], floor_frac: float = 0.05
) -> dict[str, Tensor]:
    """Per-channel activation scales sqrt(E[x_j^2]) for every decoder linear.

    Runs the calibration ``batches`` through ``model`` with forward hooks on
    each nn.Linear inside the decoder blocks, keyed by ``"{layer_idx}.{name}"``.
    Scales are floored at ``floor_frac`` of their RMS so near-dead channels
    cannot blow up the inverse scaling at inference.
    """
    layers = _decoder_layers(model)
    sums: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(key: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().to(torch.float32)
            flat = x.reshape(-1, x.shape[-1])
            if key in sums:
                sums[key] += flat.pow(2).sum(dim=0)
            else:
                sums[key] = flat.pow(2).sum(dim=0)
            counts[key] = counts.get(key, 0) + flat.shape[0]

        return hook

    for layer_idx, block in enumerate(layers):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(make_hook(f"{layer_idx}.{name}")))
    try:
        model.eval()
        for ids in batches:
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    scales = {}
    for key, s in sums.items():
        rms = (s / counts[key]).sqrt()
        floor = floor_frac * float(rms.pow(2).mean().sqrt())
        scales[key] = rms.clamp_min(max(floor, 1e-8))
    return scales


@torch.no_grad()
def collect_hessians(
    model: nn.Module, batches: list[Tensor]
) -> dict[str, Tensor]:
    """Per-layer input second-moment ``E[x x^T]`` for every decoder linear.

    Same hook plumbing as :func:`collect_input_scales`, but accumulates the
    full [in, in] Gram matrix required by GPTQ/LDLQ error feedback. Sums are
    kept on the activation device during the pass, then returned on CPU (one
    d*d float32 matrix per layer) so they do not compete with the two models
    held on the GPU during quantization + evaluation.
    """
    layers = _decoder_layers(model)
    grams: dict[str, Tensor] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(key: str):
        def hook(module, inputs, _output):
            x = inputs[0].detach().to(torch.float32)
            flat = x.reshape(-1, x.shape[-1])
            gram = flat.T @ flat
            if key in grams:
                grams[key] += gram
            else:
                grams[key] = gram
            counts[key] = counts.get(key, 0) + flat.shape[0]

        return hook

    for layer_idx, block in enumerate(layers):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(make_hook(f"{layer_idx}.{name}")))
    try:
        model.eval()
        for ids in batches:
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    return {key: (g / counts[key]).cpu() for key, g in grams.items()}


def quantize_model_copy(
    model: nn.Module,
    cfg: RealConfig,
    seed: int = 0,
    col_scales: dict[str, Tensor] | None = None,
    hessians: dict[str, Tensor] | None = None,
) -> tuple[nn.Module, dict]:
    """Deep-copy ``model`` and quantize every nn.Linear inside its decoder blocks.

    ``col_scales`` (from :func:`collect_input_scales`) is required when
    ``cfg.equilibrate`` is set. Returns ``(quantized_model, stats)`` where
    stats include the number of replaced layers and the exact
    storage-weighted bits/weight.
    """
    if cfg.equilibrate and col_scales is None:
        raise ValueError(f"config {cfg.label!r} requires col_scales")
    if cfg.error_feedback and hessians is None:
        raise ValueError(f"config {cfg.label!r} requires hessians")
    q_model = copy.deepcopy(model).eval()
    layers = _decoder_layers(q_model)

    n_replaced = 0
    total_bits = 0.0
    total_weights = 0
    for layer_idx, block in enumerate(layers):
        # Snapshot the list first: we mutate the module tree while iterating.
        linears = [
            (name, mod)
            for name, mod in block.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        for j, (name, mod) in enumerate(linears):
            sketch_k = round(cfg.sketch_ratio * mod.in_features)
            key = f"{layer_idx}.{name}"
            col_scale = None
            if cfg.equilibrate:
                if key not in col_scales:
                    raise KeyError(f"no calibration scales collected for {key}")
                col_scale = col_scales[key]
            hessian = None
            if cfg.error_feedback:
                if key not in hessians:
                    raise KeyError(f"no hessian collected for {key}")
                hessian = hessians[key]
            qlin = QJLCorrectedLinear.from_linear(
                mod,
                bits=cfg.bits,
                sketch_k=sketch_k,
                seed=seed + 1000 * layer_idx + j,
                rounding=cfg.rounding,
                method=cfg.method,
                n_states=cfg.n_states,
                col_scale=col_scale,
                equil_mode=cfg.equil_mode,
                equil_alpha=cfg.equil_alpha,
                error_feedback=cfg.error_feedback,
                hessian=hessian,
            )
            parent = block
            *parents, leaf = name.split(".")
            for p in parents:
                parent = getattr(parent, p)
            setattr(parent, leaf, qlin)
            report = qlin.storage_report()
            n_weights = mod.in_features * mod.out_features
            total_bits += report["bits_per_weight_total"] * n_weights
            total_weights += n_weights
            n_replaced += 1

    return q_model, {
        "n_replaced": n_replaced,
        "bits_per_weight": total_bits / max(total_weights, 1),
        "quantized_weights": total_weights,
    }


@torch.no_grad()
def evaluate_pair(
    fp_model: nn.Module, q_model: nn.Module, batches: list[Tensor]
) -> dict:
    """Compare quantized vs full-precision logits over token batches.

    Each batch is int64 [B, L]. KL and agreement use positions < L-1 (the
    positions that predict a next token); perplexity uses the shifted labels.
    """
    fp_model.eval()
    q_model.eval()
    kl_sum = 0.0
    top1_sum = 0.0
    ce_fp_sum = 0.0
    ce_q_sum = 0.0
    n_tokens = 0
    for ids in batches:
        logits_fp = fp_model(input_ids=ids).logits.float()[:, :-1]
        logits_q = q_model(input_ids=ids).logits.float()[:, :-1]
        labels = ids[:, 1:]

        log_p = F.log_softmax(logits_fp, dim=-1)
        log_q = F.log_softmax(logits_q, dim=-1)
        kl = (log_p.exp() * (log_p - log_q)).sum(-1)
        top1 = (logits_fp.argmax(-1) == logits_q.argmax(-1)).float()

        flat_p = log_p.reshape(-1, log_p.shape[-1])
        flat_q = log_q.reshape(-1, log_q.shape[-1])
        flat_labels = labels.reshape(-1)
        ce_fp = F.nll_loss(flat_p, flat_labels, reduction="sum")
        ce_q = F.nll_loss(flat_q, flat_labels, reduction="sum")

        t = kl.numel()
        kl_sum += float(kl.sum())
        top1_sum += float(top1.sum())
        ce_fp_sum += float(ce_fp)
        ce_q_sum += float(ce_q)
        n_tokens += t

    return {
        "mean_kl": kl_sum / n_tokens,
        "top1_agreement": top1_sum / n_tokens,
        "ppl_fp": math.exp(ce_fp_sum / n_tokens),
        "ppl_q": math.exp(ce_q_sum / n_tokens),
        "n_tokens": n_tokens,
    }


def _fetch_eval_text(cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    last_err: Exception | None = None
    for url in _TEXT_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "turbopress/0.1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(text, encoding="utf-8")
            return text
        except Exception as exc:  # noqa: BLE001 - report the last URL failure
            last_err = exc
    raise RuntimeError(f"could not download eval text: {last_err}")


def load_eval_batches(
    tokenizer,
    n_seqs: int,
    seq_len: int,
    batch_size: int,
    cache_dir: Path,
    offset_tokens: int = 0,
) -> list[Tensor]:
    """Tokenize real text into contiguous [batch, seq_len] blocks (no padding).

    ``offset_tokens`` skips the first tokens, so calibration and evaluation
    can use disjoint slices of the same text.
    """
    text = _fetch_eval_text(cache_dir / "eval_text.txt")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    needed = offset_tokens + n_seqs * seq_len
    if ids.numel() < needed:
        raise ValueError(
            f"eval text has {ids.numel()} tokens, need {needed}; "
            "reduce --seqs/--seqlen"
        )
    chunks = ids[offset_tokens:needed].reshape(n_seqs, seq_len)
    return [chunks[i : i + batch_size] for i in range(0, n_seqs, batch_size)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seqs", type=int, default=16)
    parser.add_argument("--calib-seqs", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/real_model_results.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", default="float16", choices=["float16", "bfloat16", "float32"]
    )
    parser.add_argument(
        "--config-set", default="default",
        choices=["default", "equil", "ef", "prop2"],
        help="'default' = full method sweep; 'equil' = AWQ-sqrt vs quarter-power "
        "equilibration baseline (Proposition 1); 'ef' = focused tcq vs gptq vs "
        "tcq+ef head-to-head (round 4); 'prop2' = equilibration-exponent sweep "
        "with/without error feedback (Proposition 2).",
    )
    args = parser.parse_args(argv)
    configs = {
        "default": DEFAULT_CONFIGS,
        "equil": EQUIL_CONFIGS,
        "ef": EF_CONFIGS,
        "prop2": PROP2_CONFIGS,
    }[args.config_set]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    print(f"Loading {args.model} ({args.dtype}, {device})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp_model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device).eval()
    )

    batches = [
        b.to(device)
        for b in load_eval_batches(
            tokenizer, args.seqs, args.seqlen, args.batch, Path(args.data_dir)
        )
    ]
    print(f"Eval set: {args.seqs} x {args.seqlen} tokens of real text", flush=True)

    need_calib = any(cfg.equilibrate for cfg in configs)
    need_hessian = any(cfg.error_feedback for cfg in configs)
    col_scales = None
    hessians = None
    if need_calib or need_hessian:
        calib = [
            b.to(device)
            for b in load_eval_batches(
                tokenizer,
                args.calib_seqs,
                args.seqlen,
                args.batch,
                Path(args.data_dir),
                offset_tokens=args.seqs * args.seqlen,  # disjoint from eval slice
            )
        ]
        print(
            f"Calibrating on {args.calib_seqs} x {args.seqlen} held-out tokens "
            f"(scales={need_calib}, hessians={need_hessian})...",
            flush=True,
        )
        if need_calib:
            col_scales = collect_input_scales(fp_model, calib)
        if need_hessian:
            hessians = collect_hessians(fp_model, calib)

    results = []
    for cfg in configs:
        t0 = time.time()
        q_model, stats = quantize_model_copy(
            fp_model, cfg, seed=args.seed, col_scales=col_scales, hessians=hessians
        )
        metrics = evaluate_pair(fp_model, q_model, batches)
        del q_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = {
            "label": cfg.label,
            "bits_per_weight": round(stats["bits_per_weight"], 4),
            "n_replaced": stats["n_replaced"],
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()},
            "seconds": round(time.time() - t0, 1),
        }
        results.append(row)
        print(
            f"  {cfg.label:<22} bits/w={row['bits_per_weight']:<7} "
            f"KL={row['mean_kl']:<10} top1={row['top1_agreement']:<8} "
            f"ppl={row['ppl_q']:<10} ({row['seconds']}s)",
            flush=True,
        )

    print(f"\nfp32 reference perplexity: {results[0]['ppl_fp']:.4f}")
    header = f"{'config':<22} {'bits/w':>7} {'KL(fp||q)':>10} {'top1':>7} {'ppl':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<22} {r['bits_per_weight']:>7.3f} {r['mean_kl']:>10.4f} "
            f"{r['top1_agreement']:>7.3f} {r['ppl_q']:>10.3f}"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "settings": vars(args),
                "ppl_fp": results[0]["ppl_fp"],
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
