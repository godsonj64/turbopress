"""Lloyd-Max optimal scalar quantizer codebooks for the standard normal.

After the randomized orthogonal transform (hadamard.py), the coordinates of
each weight row are approximately N(0, sigma_row^2). Dividing by the row
scale reduces scalar quantization to quantizing N(0, 1) samples, for which
the Lloyd-Max codebook is the MSE-optimal fixed-rate scalar quantizer.

The fixed point of the Lloyd iteration satisfies:
  * thresholds are midpoints of adjacent codewords, and
  * each codeword is the conditional mean of Z ~ N(0,1) on its cell:
      E[Z | a < Z < b] = (phi(a) - phi(b)) / (Phi(b) - Phi(a)).

At the fixed point the distortion is  MSE = E[Z^2] - sum_i m_i c_i^2
where m_i is the cell mass and c_i the codeword. Known values used as test
anchors: 1 bit -> levels +/- sqrt(2/pi) ~= 0.7979, MSE = 1 - 2/pi ~= 0.3634;
2 bits -> levels ~= +/-0.4528, +/-1.5104, MSE ~= 0.1175.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["lloyd_max_gaussian"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)

# Cache of (levels float64 tensor, predicted MSE) keyed by bit-width.
_CACHE: dict[int, tuple[Tensor, float]] = {}


def _phi(t: Tensor) -> Tensor:
    """Standard normal pdf; maps +/-inf to 0 exactly."""
    return torch.exp(-0.5 * t * t) / _SQRT_2PI


def _big_phi(t: Tensor) -> Tensor:
    """Standard normal cdf; maps -inf to 0 and +inf to 1 exactly."""
    return 0.5 * (1.0 + torch.erf(t / _SQRT_2))


def lloyd_max_gaussian(
    bits: int, max_iters: int = 100_000, tol: float = 1e-12
) -> tuple[Tensor, float]:
    """Return ``(levels, mse)`` for the optimal ``2**bits``-level quantizer of N(0,1).

    ``levels`` is a sorted float64 tensor of codewords; ``mse`` is the
    predicted distortion E[(Z - q(Z))^2] at the fixed point. Results are
    cached; the returned tensor is a defensive clone.
    """
    if not isinstance(bits, int) or not 1 <= bits <= 8:
        raise ValueError(f"bits must be an int in [1, 8], got {bits!r}")
    if bits in _CACHE:
        levels, mse = _CACHE[bits]
        return levels.clone(), mse

    n = 1 << bits
    # Initialize codewords at the cell-center quantiles of N(0,1).
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    c = _SQRT_2 * torch.erfinv(2.0 * p - 1.0)

    inf = torch.tensor([math.inf], dtype=torch.float64)
    mse = math.inf
    for _ in range(max_iters):
        t = torch.cat([-inf, (c[:-1] + c[1:]) / 2.0, inf])
        mass = _big_phi(t[1:]) - _big_phi(t[:-1])
        # Cell masses are strictly positive for bits <= 8 in float64.
        c_new = (_phi(t[:-1]) - _phi(t[1:])) / mass
        mse = 1.0 - float((mass * c_new * c_new).sum())
        # Lloyd's MSE improvement per step shrinks long before the fixed
        # point, so converge on codeword movement, not on the MSE delta.
        moved = float((c_new - c).abs().max())
        c = c_new
        if moved < tol:
            break

    _CACHE[bits] = (c, mse)
    return c.clone(), mse
