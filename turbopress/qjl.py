"""1-bit Quantized-JL sketch of quantization residuals (bias correction).

Let r = w - w_hat be the quantization residual of one weight row and x an
activation vector. The layer's uncorrected output error is the *deterministic*
quantity <r, x>. This module stores, per row, only the residual norm ||r||
and k sign bits  sign(S r)  for a shared seeded Gaussian sketch S in R^{k x d},
and estimates <r, x> at inference time as

    est(x) = sqrt(pi/2) * ||r|| / k * sum_i sign(<s_i, r>) * <s_i, x>.

Unbiasedness: for s ~ N(0, I_d), (s^T r, s^T x) is jointly Gaussian with
E[sign(s^T r) * s^T x] = sqrt(2/pi) * <r, x> / ||r||   (Grothendieck identity /
Stein's lemma), so E[est(x)] = <r, x> exactly, for every fixed x and r != 0.
Adding est(x) to the quantized layer output therefore makes it an unbiased
estimate of the full-precision output, over the sketch randomness.

Variance: per sketch row, Var[sign(s^T r) s^T x] = ||x||^2 - (2/pi) <r_hat, x>^2
with r_hat = r / ||r||, hence

    Var[est(x)] = (pi/2) * ||r||^2 * (||x||^2 - (2/pi)<r_hat, x>^2) / k
                <= (pi/2) * ||r||^2 * ||x||^2 / k.

So the correction trades the deterministic error <r, x> (bias) for zero-mean
noise with standard deviation O(||r|| ||x|| / sqrt(k)). Storage cost is
k bits + one norm per row, i.e. k/d bits per weight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["QJLSketch", "build_qjl_sketch", "correction_matrix", "estimate_inner_products"]


@dataclass
class QJLSketch:
    """QJL sketch of ``n_rows`` residual vectors in R^dim.

    ``proj`` is fully determined by ``(k, dim, seed)``; only the seed, the
    sign bits, and the norms need to be persisted.
    """

    proj: Tensor  # float32 [k, dim], seeded N(0, 1)
    signs: Tensor  # int8 in {-1, +1}, shape [n_rows, k]
    norms: Tensor  # float32 [n_rows], residual L2 norms
    seed: int

    @property
    def k(self) -> int:
        return self.proj.shape[0]

    @property
    def dim(self) -> int:
        return self.proj.shape[1]


def build_qjl_sketch(residual: Tensor, k: int, seed: int) -> QJLSketch:
    """Sketch each row of ``residual`` (shape [n_rows, dim]) with k sign bits.

    Rows with zero norm contribute exactly zero to every estimate (their
    stored norm is 0), so sign(0) handling is inconsequential; we map it to +1
    for determinism.
    """
    if residual.ndim != 2:
        raise ValueError(
            f"expected a 2D residual [n_rows, dim], got shape {tuple(residual.shape)}"
        )
    if k < 1:
        raise ValueError(f"sketch size k must be >= 1, got {k}")
    r32 = residual.to(torch.float32)
    gen = torch.Generator().manual_seed(seed)
    proj = torch.randn(k, r32.shape[1], generator=gen).to(r32.device)
    t = r32 @ proj.T
    signs = torch.where(t >= 0, 1, -1).to(torch.int8)
    return QJLSketch(proj=proj, signs=signs, norms=r32.norm(dim=1), seed=seed)


def correction_matrix(sketch: QJLSketch) -> Tensor:
    """Return D in R^{n_rows x k} such that est = (x @ proj^T) @ D^T.

    D_{j,i} = sqrt(pi/2) / k * ||r_j|| * sign(<s_i, r_j>).
    """
    coeff = math.sqrt(math.pi / 2.0) / sketch.k
    return coeff * sketch.signs.to(torch.float32) * sketch.norms[:, None]


def estimate_inner_products(sketch: QJLSketch, x: Tensor) -> Tensor:
    """Unbiased estimates of <r_j, x_b> for all rows j and batch vectors x_b.

    ``x`` has shape [..., dim]; the result has shape [..., n_rows].
    """
    if x.shape[-1] != sketch.dim:
        raise ValueError(f"expected last dimension {sketch.dim}, got {x.shape[-1]}")
    x32 = x.to(torch.float32)
    return (x32 @ sketch.proj.T) @ correction_matrix(sketch).T
