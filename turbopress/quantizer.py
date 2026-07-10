"""Row-wise low-bit scalar quantization with Lloyd-Max Gaussian codebooks.

Each row of the (already rotated) weight matrix is modeled as
``scale_row * N(0, 1)`` and quantized against the shared Lloyd-Max codebook.
Two rounding modes are provided:

  * ``"nearest"``: MSE-optimal deterministic rounding, plus a few alternating
    refinement steps between code assignment and the per-row scale (given the
    assigned codewords q, the least-squares scale is <w, q> / <q, q>). The
    resulting error is a *deterministic* function of the weights: this is the
    biased baseline the QJL correction (qjl.py) is designed to fix.
  * ``"stochastic"``: stochastic rounding between the two bracketing
    codewords. Unbiased per weight for values inside the codebook range
    (clipped values at the tails are biased toward the extreme codewords),
    at the cost of roughly doubling the per-weight error variance versus
    nearest rounding. Serves as the classical debiasing baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from turbopress.codebooks import lloyd_max_gaussian

__all__ = ["RowQuantized", "quantize_rows"]


@dataclass
class RowQuantized:
    """Quantized rows: ``w ~= scales[:, None] * codebook[codes]``."""

    codes: Tensor  # uint8, shape [n_rows, dim]
    scales: Tensor  # float32, shape [n_rows]
    codebook: Tensor  # float32, sorted, shape [2**bits]
    bits: int

    def dequantize(self) -> Tensor:
        """Reconstruct the float32 approximation of the original rows."""
        return self.scales[:, None] * self.codebook[self.codes.long()]


def _nearest_codes(z: Tensor, codebook: Tensor) -> Tensor:
    """Nearest-codeword assignment via midpoint thresholds (codebook sorted)."""
    thresholds = (codebook[:-1] + codebook[1:]) / 2.0
    return torch.bucketize(z.contiguous(), thresholds)


def _stochastic_codes(z: Tensor, codebook: Tensor, generator: torch.Generator) -> Tensor:
    """Stochastic rounding to one of the two bracketing codewords.

    For z in [codebook[i], codebook[i+1]] the upper codeword is chosen with
    probability (z - c_i) / (c_{i+1} - c_i), so E[codebook[code]] = z for
    in-range z. Out-of-range z is clipped to the extreme codewords first.
    """
    lo, hi = codebook[0], codebook[-1]
    zc = z.clamp(lo, hi)
    lower = (torch.bucketize(zc.contiguous(), codebook, right=True) - 1).clamp(
        0, codebook.numel() - 2
    )
    c_lo = codebook[lower]
    c_hi = codebook[lower + 1]
    p_up = ((zc - c_lo) / (c_hi - c_lo)).clamp(0.0, 1.0)
    up = torch.bernoulli(p_up, generator=generator)
    return lower + up.to(lower.dtype)


def quantize_rows(
    w: Tensor,
    bits: int,
    rounding: str = "nearest",
    scale_iters: int = 2,
    generator: torch.Generator | None = None,
) -> RowQuantized:
    """Quantize each row of ``w`` (shape [n_rows, dim]) to ``bits`` bits/weight.

    Rows that are exactly zero get scale 0 and dequantize to exactly zero.
    ``generator`` is required only for ``rounding="stochastic"``.
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2D tensor [n_rows, dim], got shape {tuple(w.shape)}")
    if not torch.is_floating_point(w):
        raise TypeError(f"expected a floating-point tensor, got {w.dtype}")
    if rounding not in ("nearest", "stochastic"):
        raise ValueError(f"rounding must be 'nearest' or 'stochastic', got {rounding!r}")
    if scale_iters < 0:
        raise ValueError(f"scale_iters must be >= 0, got {scale_iters}")

    levels64, _ = lloyd_max_gaussian(bits)
    codebook = levels64.to(torch.float32).to(w.device)
    w32 = w.to(torch.float32)

    scales = w32.pow(2).mean(dim=1).sqrt()
    nonzero = scales > 0
    safe = torch.where(nonzero, scales, torch.ones_like(scales))

    if rounding == "stochastic":
        if generator is None:
            raise ValueError("stochastic rounding requires a torch.Generator")
        codes = _stochastic_codes(w32 / safe[:, None], codebook, generator)
    else:
        codes = _nearest_codes(w32 / safe[:, None], codebook)
        for _ in range(scale_iters):
            q = codebook[codes]
            num = (w32 * q).sum(dim=1)
            den = (q * q).sum(dim=1)
            new_scale = torch.where(den > 0, num / den.clamp_min(1e-30), safe)
            # A non-positive least-squares scale means the assignment is
            # degenerate; keep the previous scale for that row.
            safe = torch.where(new_scale > 0, new_scale, safe)
            codes = _nearest_codes(w32 / safe[:, None], codebook)

    scales = torch.where(nonzero, safe, torch.zeros_like(safe))
    return RowQuantized(
        codes=codes.to(torch.uint8),
        scales=scales,
        codebook=codebook,
        bits=bits,
    )
