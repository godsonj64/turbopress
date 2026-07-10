"""Per-layer bit allocation by measured KL sensitivity.

A uniform bit-width spends the same budget on every layer, but layers are not
equally sensitive: quantizing an early attention projection may barely move the
output distribution while quantizing a late MLP down-projection wrecks it. This
module measures each layer's sensitivity empirically and then spends a fixed
average budget unevenly to minimize total predicted damage.

Two-phase procedure:

1. **Sensitivity probe.** For every quantized linear and every candidate
   bit-width b, quantize *only that layer* to b bits (all others full
   precision) and measure the mean token-level KL(fp || quant) on a probe set.
   ``cost[layer][b]`` is that KL -- a direct, model-faithful sensitivity
   measurement, exactly the "measured KL sensitivity" signal we want to
   allocate against. Cheap: one layer is swapped at a time, so the cost is one
   forward pass per (layer, bit) rather than a combinatorial search.

2. **Allocation.** Choose a bit-width per layer to minimize the summed KL cost
   subject to a target average bits/weight, weighting each layer by its
   parameter count (a wide layer at +1 bit costs more budget than a narrow
   one). Solved greedily: start everyone at the minimum bit-width, then
   repeatedly spend the next unit of budget on the upgrade with the best
   KL-reduction-per-extra-bit ratio. Greedy is optimal here because each
   layer's KL is (empirically, and by construction) convex-decreasing in bits,
   so marginal returns are diminishing.

The resulting ``{key: bits}`` map is consumed by ``real_model`` /
``run_allocation`` to build a mixed-precision model whose measured average
bits/weight matches the budget but whose KL is far below the uniform model's.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from turbopress.linear import QJLCorrectedLinear
from turbopress.real_model import _decoder_layers, evaluate_pair

__all__ = [
    "LayerSlot",
    "measure_sensitivity",
    "allocate_bits",
    "build_mixed_model",
    "main",
]


@dataclass
class LayerSlot:
    """One quantizable linear: where it lives and how big it is."""

    key: str  # "{layer_idx}.{name}"
    layer_idx: int
    name: str
    n_weights: int


def _iter_slots(model: nn.Module):
    for layer_idx, block in enumerate(_decoder_layers(model)):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                yield layer_idx, name, mod


def _replace(block: nn.Module, name: str, new: nn.Module) -> None:
    parent = block
    *parents, leaf = name.split(".")
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


def _quantize_one(
    mod: nn.Linear,
    bits: int,
    seed: int,
    method: str,
    n_states: int,
    col_scale: Tensor | None,
    error_feedback: bool,
    hessian: Tensor | None,
) -> QJLCorrectedLinear:
    return QJLCorrectedLinear.from_linear(
        mod,
        bits=bits,
        seed=seed,
        method=method,
        n_states=n_states,
        col_scale=col_scale,
        error_feedback=error_feedback,
        hessian=hessian,
    )


@torch.no_grad()
def measure_sensitivity(
    fp_model: nn.Module,
    batches: list[Tensor],
    bit_choices: tuple[int, ...],
    method: str = "tcq",
    n_states: int = 64,
    col_scales: dict[str, Tensor] | None = None,
    hessians: dict[str, Tensor] | None = None,
    error_feedback: bool = False,
    seed: int = 0,
) -> tuple[list[LayerSlot], dict[str, dict[int, float]]]:
    """Measure per-layer KL(fp||quant) for each candidate bit-width.

    Quantizes one layer at a time (all others fp) and evaluates KL against
    ``fp_model`` on ``batches``. Returns the layer slots and a nested
    ``cost[key][bits] -> mean_kl`` table.
    """
    slots: list[LayerSlot] = []
    cost: dict[str, dict[int, float]] = {}
    # One working copy is mutated a single layer at a time and the original
    # layer restored afterwards, so only two models are ever resident (no
    # per-(layer, bit) deepcopy of the whole network).
    work = copy.deepcopy(fp_model).eval()
    for layer_idx, name, _ in _iter_slots(fp_model):
        key = f"{layer_idx}.{name}"
        block = _decoder_layers(work)[layer_idx]
        orig = dict(block.named_modules())[name]
        slots.append(
            LayerSlot(key, layer_idx, name, orig.in_features * orig.out_features)
        )
        cost[key] = {}
        col_scale = col_scales.get(key) if col_scales else None
        hessian = hessians.get(key) if hessians else None
        for bits in bit_choices:
            qlin = _quantize_one(
                orig, bits, seed + layer_idx, method, n_states,
                col_scale, error_feedback, hessian,
            )
            _replace(block, name, qlin)
            metrics = evaluate_pair(fp_model, work, batches)
            cost[key][bits] = metrics["mean_kl"]
        _replace(block, name, orig)  # restore full precision for the next probe
        if batches[0].device.type == "cuda":
            torch.cuda.empty_cache()
    del work
    return slots, cost


def allocate_bits(
    slots: list[LayerSlot],
    cost: dict[str, dict[int, float]],
    target_bits: float,
) -> dict[str, int]:
    """Greedy KL-minimizing bit allocation under an average-bits budget.

    Every layer starts at the smallest candidate width; each step applies the
    upgrade with the largest KL drop per extra *bit-weight* (KL saved divided
    by added storage), until the parameter-weighted average bits/weight would
    exceed ``target_bits``. Returns ``{key: bits}``.
    """
    choices = sorted(next(iter(cost.values())).keys())
    total_weights = sum(s.n_weights for s in slots)
    alloc = {s.key: choices[0] for s in slots}
    by_key = {s.key: s for s in slots}

    def avg_bits(a: dict[str, int]) -> float:
        return sum(by_key[k].n_weights * b for k, b in a.items()) / total_weights

    while True:
        best = None  # (gain_per_bit, key, next_bits, new_avg)
        for key, cur in alloc.items():
            higher = [c for c in choices if c > cur]
            if not higher:
                continue
            nxt = higher[0]
            slot = by_key[key]
            trial = dict(alloc)
            trial[key] = nxt
            new_avg = avg_bits(trial)
            if new_avg > target_bits:
                continue
            kl_saved = cost[key][cur] - cost[key][nxt]
            added_bits = (nxt - cur) * slot.n_weights
            gain = kl_saved / added_bits
            if best is None or gain > best[0]:
                best = (gain, key, nxt, new_avg)
        if best is None:
            break
        alloc[best[1]] = best[2]
    return alloc


def build_mixed_model(
    fp_model: nn.Module,
    alloc: dict[str, int],
    method: str = "tcq",
    n_states: int = 64,
    col_scales: dict[str, Tensor] | None = None,
    hessians: dict[str, Tensor] | None = None,
    error_feedback: bool = False,
    seed: int = 0,
) -> tuple[nn.Module, dict]:
    """Build a mixed-precision quantized copy following ``alloc`` ({key: bits}).

    Returns ``(model, stats)`` with the exact parameter-weighted average
    bits/weight (matching ``real_model.quantize_model_copy``'s accounting).
    """
    q_model = copy.deepcopy(fp_model).eval()
    total_bits = 0.0
    total_weights = 0
    for layer_idx, block in enumerate(_decoder_layers(q_model)):
        linears = [
            (name, mod)
            for name, mod in block.named_modules()
            if isinstance(mod, nn.Linear)
        ]
        for name, mod in linears:
            key = f"{layer_idx}.{name}"
            bits = alloc[key]
            col_scale = col_scales.get(key) if col_scales else None
            hessian = hessians.get(key) if hessians else None
            qlin = _quantize_one(
                mod, bits, seed + layer_idx, method, n_states,
                col_scale, error_feedback, hessian,
            )
            _replace(block, name, qlin)
            n_weights = mod.in_features * mod.out_features
            total_bits += qlin.storage_report()["bits_per_weight_total"] * n_weights
            total_weights += n_weights
    return q_model, {
        "bits_per_weight": total_bits / max(total_weights, 1),
        "quantized_weights": total_weights,
    }


def main(argv: list[str] | None = None) -> None:
    """Probe KL sensitivity, allocate bits at target budgets, and compare.

    Reuses ``real_model``'s loaders/calibration. For each target budget it
    builds a mixed-precision model and reports it against the uniform model at
    the nearest integer bit-width, on held-out eval text.
    """
    import argparse
    import json
    import time
    from pathlib import Path

    from turbopress import real_model as rm

    parser = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--seqs", type=int, default=16)
    parser.add_argument("--calib-seqs", type=int, default=8)
    parser.add_argument("--probe-seqs", type=int, default=8)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", default="tcq", choices=["tcq", "scalar"])
    parser.add_argument("--n-states", type=int, default=64)
    parser.add_argument("--error-feedback", action="store_true")
    parser.add_argument("--bit-choices", default="2,3,4")
    parser.add_argument("--targets", default="2.5,3.0")
    parser.add_argument("--out", default="results/allocation.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", default="float16")
    args = parser.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    bit_choices = tuple(int(b) for b in args.bit_choices.split(","))
    targets = [float(t) for t in args.targets.split(",")]

    print(f"Loading {args.model} ({args.dtype}, {device})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp_model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        .to(device)
        .eval()
    )

    data_dir = Path(args.data_dir)
    eval_batches = [
        b.to(device)
        for b in rm.load_eval_batches(tokenizer, args.seqs, args.seqlen, args.batch, data_dir)
    ]
    calib = [
        b.to(device)
        for b in rm.load_eval_batches(
            tokenizer, args.calib_seqs, args.seqlen, args.batch, data_dir,
            offset_tokens=args.seqs * args.seqlen,
        )
    ]
    probe = [
        b.to(device)
        for b in rm.load_eval_batches(
            tokenizer, args.probe_seqs, args.seqlen, args.batch, data_dir,
            offset_tokens=(args.seqs + args.calib_seqs) * args.seqlen,
        )
    ]

    print("Collecting calibration scales + hessians...", flush=True)
    col_scales = rm.collect_input_scales(fp_model, calib)
    hessians = rm.collect_hessians(fp_model, calib) if args.error_feedback else None

    print(
        f"Probing per-layer KL sensitivity at bits {bit_choices} "
        f"(method={args.method})...",
        flush=True,
    )
    t0 = time.time()
    slots, cost = measure_sensitivity(
        fp_model, probe, bit_choices, method=args.method, n_states=args.n_states,
        col_scales=col_scales, hessians=hessians, error_feedback=args.error_feedback,
        seed=args.seed,
    )
    print(f"  probed {len(slots)} layers in {time.time() - t0:.0f}s", flush=True)

    results = []
    for target in targets:
        alloc = allocate_bits(slots, cost, target)
        q_model, stats = build_mixed_model(
            fp_model, alloc, method=args.method, n_states=args.n_states,
            col_scales=col_scales, hessians=hessians,
            error_feedback=args.error_feedback, seed=args.seed,
        )
        mixed = rm.evaluate_pair(fp_model, q_model, eval_batches)
        del q_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Uniform baseline at the nearest achievable integer width.
        uni_bits = max(bit_choices[0], min(bit_choices[-1], round(target)))
        uni_alloc = {s.key: uni_bits for s in slots}
        u_model, u_stats = build_mixed_model(
            fp_model, uni_alloc, method=args.method, n_states=args.n_states,
            col_scales=col_scales, hessians=hessians,
            error_feedback=args.error_feedback, seed=args.seed,
        )
        uniform = rm.evaluate_pair(fp_model, u_model, eval_batches)
        del u_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        hist = {b: sum(1 for v in alloc.values() if v == b) for b in bit_choices}
        row = {
            "target_bits": target,
            "mixed_bits_per_weight": round(stats["bits_per_weight"], 4),
            "mixed_kl": round(mixed["mean_kl"], 6),
            "mixed_top1": round(mixed["top1_agreement"], 6),
            "mixed_ppl": round(mixed["ppl_q"], 4),
            "uniform_bits": uni_bits,
            "uniform_bits_per_weight": round(u_stats["bits_per_weight"], 4),
            "uniform_kl": round(uniform["mean_kl"], 6),
            "uniform_top1": round(uniform["top1_agreement"], 6),
            "uniform_ppl": round(uniform["ppl_q"], 4),
            "bit_histogram": hist,
        }
        results.append(row)
        print(
            f"target {target}: mixed {row['mixed_bits_per_weight']}b "
            f"KL={row['mixed_kl']} ppl={row['mixed_ppl']} {hist}  |  "
            f"uniform {uni_bits}b KL={row['uniform_kl']} ppl={row['uniform_ppl']}",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "settings": vars(args),
                "ppl_fp": round(mixed["ppl_fp"], 4),
                "sensitivity": cost,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved allocation results to {out_path}")


if __name__ == "__main__":
    main()
