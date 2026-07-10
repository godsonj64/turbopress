"""Controlled LDLQ-vs-nearest microbenchmark, saved as a results JSON.

Reproduces the synthetic setup of ``tests/test_gptq.py::test_ldlq_reduces_
hessian_loss`` -- a random weight matrix quantized against a *correlated*
activation Hessian H = E[x x^T] -- and measures the Hessian-weighted output
loss  tr[(W - W_hat) H (W - W_hat)^T] / (n d)  for nearest rounding vs LDLQ
error feedback, averaged over several seeds.

This exists so that paper figures quoting the LDLQ gain are generated from a
checked-in, re-runnable measurement (``results/ldlq_micro.json``) instead of
hand-typed numbers.

Run:  python -m turbopress.ldlq_micro [--out results/ldlq_micro.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from turbopress.gptq import ldlq_quantize_rows
from turbopress.quantizer import quantize_rows

__all__ = ["run_micro", "main"]


def _hessian_loss(w: torch.Tensor, what: torch.Tensor, h: torch.Tensor) -> float:
    e = w - what
    return float(torch.einsum("nd,de,ne->", e, h, e) / w.numel())


def run_micro(
    bits_list: tuple[int, ...] = (2, 3, 4),
    d: int = 128,
    n: int = 64,
    n_samples: int = 4096,
    n_seeds: int = 5,
) -> list[dict]:
    """Hessian-weighted loss of nearest vs LDLQ per bit-width (mean/std over seeds)."""
    results = []
    for bits in bits_list:
        near_losses, ldlq_losses = [], []
        for seed in range(n_seeds):
            gen = torch.Generator().manual_seed(seed)
            # Correlated activations: x = A z with a well-conditioned random A,
            # so H has real off-diagonal structure for LDLQ to exploit.
            a = torch.randn(d, d, generator=gen) * 0.3 + torch.eye(d)
            x = torch.randn(n_samples, d, generator=gen) @ a.T
            h = (x.T @ x) / n_samples
            w = torch.randn(n, d, generator=gen)
            near = quantize_rows(w, bits=bits).dequantize()
            ldlq = ldlq_quantize_rows(w, h, bits=bits).dequantize()
            near_losses.append(_hessian_loss(w, near, h))
            ldlq_losses.append(_hessian_loss(w, ldlq, h))
        near_t = torch.tensor(near_losses)
        ldlq_t = torch.tensor(ldlq_losses)
        results.append(
            {
                "bits": bits,
                "nearest_loss_mean": float(near_t.mean()),
                "nearest_loss_std": float(near_t.std()),
                "ldlq_loss_mean": float(ldlq_t.mean()),
                "ldlq_loss_std": float(ldlq_t.std()),
                "gain": float(near_t.mean() / ldlq_t.mean()),
            }
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/ldlq_micro.json")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args(argv)

    results = run_micro(
        d=args.dim, n=args.rows, n_samples=args.samples, n_seeds=args.seeds
    )
    for r in results:
        print(
            f"bits={r['bits']}: nearest {r['nearest_loss_mean']:.5f} "
            f"vs LDLQ {r['ldlq_loss_mean']:.5f}  ({r['gain']:.2f}x lower)"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "settings": {
                    "dim": args.dim,
                    "rows": args.rows,
                    "samples": args.samples,
                    "seeds": args.seeds,
                },
                "results": results,
            },
            indent=2,
        )
    )
    print(f"Saved results to {out}")


if __name__ == "__main__":
    main()
