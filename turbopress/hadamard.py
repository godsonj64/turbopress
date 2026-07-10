"""Seeded randomized orthogonal transforms used to precondition weights.

The transform R is orthogonal (R^T R = I), so for any weight matrix W and
input x:  W x = (W R^T)(R x).  Quantizing W R^T instead of W is lossless in
exact arithmetic, while the random rotation spreads outliers so that every
coordinate of a rotated row is approximately Gaussian -- the regime in which
the Lloyd-Max Gaussian codebook (codebooks.py) is the optimal scalar
quantizer.

Construction:
  * dim with a power-of-2 factor: R = H_block . diag(signs), where H_block is
    a block-diagonal orthonormal Walsh-Hadamard transform over blocks of size
    ``largest power-of-2 divisor of dim`` and ``signs`` is a seeded Rademacher
    vector. H_block is symmetric, so R^T = diag(signs) . H_block.
  * odd dim (no power-of-2 factor): a dense seeded orthogonal matrix from the
    QR decomposition of a Gaussian matrix (with a deterministic sign
    convention), which is exactly orthogonal but costs O(dim^2) memory.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = ["fwht", "largest_pow2_divisor", "RandomizedOrthogonal"]


def largest_pow2_divisor(n: int) -> int:
    """Return the largest power of 2 that divides ``n`` (n >= 1)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    return n & (-n)


def fwht(x: Tensor) -> Tensor:
    """Orthonormal fast Walsh-Hadamard transform along the last dimension.

    The last dimension must be a power of 2. The transform matrix H is
    symmetric and orthonormal (H = H^T = H^-1), so ``fwht(fwht(x)) == x``.
    Complexity is O(d log d) per vector.
    """
    d = x.shape[-1]
    if d & (d - 1):
        raise ValueError(f"fwht requires a power-of-2 last dimension, got {d}")
    orig_shape = x.shape
    y = x.reshape(-1, d)
    h = 1
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        even = y[:, :, 0, :]
        odd = y[:, :, 1, :]
        y = torch.stack((even + odd, even - odd), dim=2).reshape(-1, d)
        h *= 2
    return (y / math.sqrt(d)).reshape(orig_shape)


class RandomizedOrthogonal(nn.Module):
    """Seeded orthogonal map applied along the last dimension of a tensor.

    ``forward(x)`` computes R x per vector and ``inverse(x)`` computes R^T x;
    the two compose to the identity up to floating-point error. The transform
    is fully determined by ``(dim, seed)``, so only the seed needs to be
    stored to reproduce it.
    """

    q: Tensor | None

    def __init__(self, dim: int, seed: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.seed = seed
        self.block = largest_pow2_divisor(dim)

        gen = torch.Generator().manual_seed(seed)
        signs = torch.randint(0, 2, (dim,), generator=gen, dtype=torch.int64)
        self.register_buffer("signs", (signs * 2 - 1).to(torch.float32))

        if self.block == 1 and dim > 1:
            # Odd dimension: exact dense orthogonal matrix, deterministic
            # across runs via a sign convention on the QR factors.
            g = torch.randn(dim, dim, generator=gen, dtype=torch.float64)
            q, r = torch.linalg.qr(g)
            q = q * torch.sign(torch.diagonal(r))
            self.register_buffer("q", q.to(torch.float32))
        else:
            self.register_buffer("q", None)

    def _check(self, x: Tensor) -> None:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"expected last dimension {self.dim}, got {x.shape[-1]}"
            )
        if not torch.is_floating_point(x):
            raise TypeError(f"expected a floating-point tensor, got {x.dtype}")

    def _block_fwht(self, x: Tensor) -> Tensor:
        if self.block == self.dim:
            return fwht(x)
        m = self.dim // self.block
        shape = x.shape
        y = fwht(x.reshape(*shape[:-1], m, self.block))
        return y.reshape(shape)

    def forward(self, x: Tensor) -> Tensor:
        """Apply R along the last dimension: x -> H_block (signs * x)."""
        self._check(x)
        z = x * self.signs.to(x.dtype)
        if self.q is not None:
            return z @ self.q.T.to(x.dtype)
        return self._block_fwht(z)

    def inverse(self, x: Tensor) -> Tensor:
        """Apply R^T along the last dimension: x -> signs * (H_block x)."""
        self._check(x)
        if self.q is not None:
            z = x @ self.q.to(x.dtype)
        else:
            z = self._block_fwht(x)
        return z * self.signs.to(x.dtype)

    def extra_repr(self) -> str:
        kind = "dense-qr" if self.q is not None else f"hadamard(block={self.block})"
        return f"dim={self.dim}, seed={self.seed}, kind={kind}"
