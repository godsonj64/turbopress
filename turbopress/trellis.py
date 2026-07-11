"""Trellis-coded quantization (TCQ) of rows against an analytic codebook.

Classical TCQ (Marcellin & Fischer, 1990) closes much of the gap between
scalar quantization and the rate-distortion bound at a fixed rate R:

  * Use a *doubled* codebook of 2^(R+1) levels -- here the Lloyd-Max optimal
    codebook for N(0,1) at R+1 bits (codebooks.py), so the design stays
    fully analytic / calibration-free (the rotation in hadamard.py makes
    the coordinate distribution known in advance).
  * Partition the sorted levels into 4 subsets by index mod 4 (Ungerboeck set
    partitioning), so each subset is a coarse quantizer with 4x spacing.
  * A rate-1/2 convolutional code defines a trellis over S states; each
    transition is labeled with one subset. The encoder runs Viterbi over the
    coordinates of a row, choosing the level *sequence* with minimum total
    squared error among all trellis paths.

Rate accounting: each sample is coded by 1 path bit (the trellis input bit)
plus R-1 bits selecting the member within the subset determined by the path,
i.e. exactly R bits/sample. ``decode_levels`` reconstructs the chosen levels
from those bits alone; a round-trip test proves the storage claim.

Reference points for N(0,1) at R = 2: scalar Lloyd-Max MSE 0.1175; 4/8-state
TCQ ~0.089-0.093; distortion-rate bound 0.0625.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from turbopress.codebooks import lloyd_max_gaussian

__all__ = [
    "Trellis",
    "TCQQuantized",
    "tcq_quantize_rows",
    "tcq_optimized_codebook",
    "decode_levels",
]

_N_SUBSETS = 4

# Best-known rate-1/2 convolutional generators (octal) per state count;
# constraint length K = log2(S) + 1.
_GENERATORS: dict[int, tuple[int, int]] = {
    4: (0o5, 0o7),
    8: (0o15, 0o17),
    16: (0o23, 0o35),
    64: (0o133, 0o171),
}


class Trellis:
    """Shift-register trellis of a rate-1/2 convolutional code.

    State s holds the last m = log2(S) input bits. On input bit b the next
    state is ``((s << 1) | b) & (S - 1)`` and the emitted subset label is the
    2-bit output ``(parity((x & g1)) << 1) | parity(x & g0)`` of the code,
    with ``x = (b << m) | s``.
    """

    def __init__(self, n_states: int) -> None:
        if n_states not in _GENERATORS:
            raise ValueError(
                f"n_states must be one of {sorted(_GENERATORS)}, got {n_states}"
            )
        self.n_states = n_states
        m = n_states.bit_length() - 1
        g0, g1 = _GENERATORS[n_states]

        subset = torch.empty(n_states, 2, dtype=torch.int64)
        for s in range(n_states):
            for b in (0, 1):
                x = (b << m) | s
                subset[s, b] = (((x & g1).bit_count() & 1) << 1) | (
                    (x & g0).bit_count() & 1
                )
        self.subset_table = subset  # [S, 2]

        ns = torch.arange(n_states)
        self.prev0 = ns >> 1  # predecessor with top history bit 0
        self.prev1 = (ns >> 1) | (n_states >> 1)  # ... with top history bit 1
        bit = ns & 1  # the input bit consumed on any transition into ns
        self.sub0 = subset[self.prev0, bit]  # subset on branch prev0 -> ns
        self.sub1 = subset[self.prev1, bit]  # subset on branch prev1 -> ns


@dataclass
class TCQQuantized:
    """TCQ-coded rows: ``w ~= scales[:, None] * codebook[level_codes]``.

    ``codebook`` has 2^(bits+1) levels but each sample costs only ``bits``
    bits to store: ``path_bits`` (1/sample) + ``member_codes`` (bits-1 per
    sample); ``level_codes`` is derivable from them via ``decode_levels``.
    """

    level_codes: Tensor  # uint8 [n, d], index into the doubled codebook
    path_bits: Tensor  # uint8 [n, d], trellis input bits
    member_codes: Tensor  # uint8 [n, d], member index within the subset
    scales: Tensor  # float32 [n]
    codebook: Tensor  # float32 [2**(bits+1)], sorted
    bits: int
    n_states: int

    def dequantize(self) -> Tensor:
        return self.scales[:, None] * self.codebook[self.level_codes.long()]


def _subset_costs(z: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor]:
    """Nearest member and squared cost per subset: [n, d, 4] each."""
    n, d = z.shape
    members = torch.empty(n, d, _N_SUBSETS, dtype=torch.int64, device=z.device)
    costs = torch.empty(n, d, _N_SUBSETS, dtype=torch.float32, device=z.device)
    for j in range(_N_SUBSETS):
        sub_cb = codebook[j::_N_SUBSETS]  # sorted slice of the sorted codebook
        if sub_cb.numel() == 1:
            m = torch.zeros(n, d, dtype=torch.int64)
        else:
            thresholds = (sub_cb[:-1] + sub_cb[1:]) / 2.0
            m = torch.bucketize(z.contiguous(), thresholds)
        members[:, :, j] = m
        costs[:, :, j] = (z - sub_cb[m]).pow(2)
    return members, costs


def _viterbi(
    z: Tensor,
    codebook: Tensor,
    trellis: Trellis,
    start_state: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Minimum-distortion trellis path per row.

    Each row's path begins in ``start_state`` (int64 [n]; default state 0,
    the stored-stream convention). Returns ``(level_codes, path_bits,
    member_codes, end_state)``; the first three are [n, d] uint8. Feeding a
    call's ``end_state`` into the next call encodes one continuous trellis
    path, so concatenated path bits decode with the standard sequential walk
    from state 0 -- the storage format is unchanged. Runs entirely on
    ``z.device`` (the trellis branch tables are moved there).
    """
    n, d = z.shape
    dev = z.device
    prev0, prev1 = trellis.prev0.to(dev), trellis.prev1.to(dev)
    sub0, sub1 = trellis.sub0.to(dev), trellis.sub1.to(dev)
    members, costs = _subset_costs(z, codebook)

    alpha = torch.full((n, trellis.n_states), torch.inf, device=dev)
    if start_state is None:
        alpha[:, 0] = 0.0
    else:
        alpha[torch.arange(n, device=dev), start_state.to(dev).long()] = 0.0
    bp = torch.empty(n, d, trellis.n_states, dtype=torch.bool, device=dev)
    for t in range(d):
        c = costs[:, t, :]
        cand0 = alpha[:, prev0] + c[:, sub0]
        cand1 = alpha[:, prev1] + c[:, sub1]
        take1 = cand1 < cand0
        alpha = torch.where(take1, cand1, cand0)
        bp[:, t, :] = take1

    level_codes = torch.empty(n, d, dtype=torch.uint8, device=dev)
    path_bits = torch.empty(n, d, dtype=torch.uint8, device=dev)
    member_codes = torch.empty(n, d, dtype=torch.uint8, device=dev)
    rows = torch.arange(n, device=dev)
    state = alpha.argmin(dim=1)
    end_state = state.clone()
    for t in range(d - 1, -1, -1):
        take1 = bp[rows, t, state]
        prev = torch.where(take1, prev1[state], prev0[state])
        subset = torch.where(take1, sub1[state], sub0[state])
        member = members[rows, t, subset]
        path_bits[:, t] = (state & 1).to(torch.uint8)
        member_codes[:, t] = member.to(torch.uint8)
        level_codes[:, t] = (member * _N_SUBSETS + subset).to(torch.uint8)
        state = prev
    return level_codes, path_bits, member_codes, end_state


def decode_levels(
    path_bits: Tensor, member_codes: Tensor, trellis: Trellis
) -> Tensor:
    """Reconstruct level codes from the stored R-bit stream (start state 0)."""
    n, d = path_bits.shape
    dev = path_bits.device
    level_codes = torch.empty(n, d, dtype=torch.uint8, device=dev)
    state = torch.zeros(n, dtype=torch.int64, device=dev)
    subset_table = trellis.subset_table.to(dev)
    mask = trellis.n_states - 1
    for t in range(d):
        b = path_bits[:, t].long()
        subset = subset_table[state, b]
        level_codes[:, t] = (member_codes[:, t].long() * _N_SUBSETS + subset).to(
            torch.uint8
        )
        state = ((state << 1) | b) & mask
    return level_codes


# Cache of trellis-optimized codebooks keyed by every design parameter.
_OPT_CACHE: dict[tuple[int, int, int, int, int], Tensor] = {}


def tcq_optimized_codebook(
    bits: int,
    n_states: int,
    n_samples: int = 1 << 16,
    iters: int = 8,
    seed: int = 0,
) -> Tensor:
    """Codebook re-optimized *under the trellis* on synthetic N(0,1) samples.

    The doubled Lloyd-Max codebook is optimal for memoryless scalar
    quantization, not for trellis-constrained assignment. This runs
    generalized Lloyd with Viterbi assignments: encode a large Gaussian
    sample, then move each level to the mean of the samples the trellis
    actually assigned to it. Because the source distribution is known
    (post-rotation coordinates are ~N(0,1)), this stays fully data-free:
    no model calibration data is involved, and the result is cached and
    deterministic given the arguments.

    Within-subset order is restored after each update (the per-subset
    nearest-member search requires sorted subset slices); global order
    across subsets is irrelevant to the code.
    """
    key = (bits, n_states, n_samples, iters, seed)
    if key in _OPT_CACHE:
        return _OPT_CACHE[key].clone()

    trellis = Trellis(n_states)
    levels64, _ = lloyd_max_gaussian(bits + 1)
    codebook = levels64.to(torch.float32)
    gen = torch.Generator().manual_seed(seed)
    n_rows = 8
    z = torch.randn(n_rows, n_samples // n_rows, generator=gen)

    for _ in range(iters):
        level_codes, _, _, _ = _viterbi(z, codebook, trellis)
        flat_codes = level_codes.long().flatten()
        flat_z = z.flatten()
        n_levels = codebook.numel()
        sums = torch.zeros(n_levels).index_add_(0, flat_codes, flat_z)
        counts = torch.zeros(n_levels).index_add_(
            0, flat_codes, torch.ones_like(flat_z)
        )
        codebook = torch.where(counts > 0, sums / counts.clamp_min(1.0), codebook)
        for j in range(_N_SUBSETS):
            codebook[j::_N_SUBSETS] = codebook[j::_N_SUBSETS].sort().values

    _OPT_CACHE[key] = codebook
    return codebook.clone()


def tcq_quantize_rows(
    w: Tensor,
    bits: int,
    n_states: int = 8,
    scale_iters: int = 1,
    codebook_mode: str = "optimized",
) -> TCQQuantized:
    """TCQ-quantize each row of ``w`` (shape [n_rows, dim]) at ``bits`` bits/weight.

    The doubled codebook (2^(bits+1) levels) must fit uint8 indices, so
    ``bits`` is limited to [1, 7]. Zero rows get scale 0 and dequantize to
    exactly zero. ``scale_iters`` re-runs Viterbi after a least-squares
    refit of the per-row scale. ``codebook_mode`` is ``"optimized"``
    (generalized-Lloyd-under-the-trellis, see ``tcq_optimized_codebook``)
    or ``"lloyd"`` (plain doubled Lloyd-Max).
    """
    if w.ndim != 2:
        raise ValueError(f"expected a 2D tensor [n_rows, dim], got shape {tuple(w.shape)}")
    if not torch.is_floating_point(w):
        raise TypeError(f"expected a floating-point tensor, got {w.dtype}")
    if not 1 <= bits <= 7:
        raise ValueError(f"tcq bits must be in [1, 7], got {bits}")
    if scale_iters < 0:
        raise ValueError(f"scale_iters must be >= 0, got {scale_iters}")
    if codebook_mode not in ("optimized", "lloyd"):
        raise ValueError(
            f"codebook_mode must be 'optimized' or 'lloyd', got {codebook_mode!r}"
        )

    trellis = Trellis(n_states)
    if codebook_mode == "optimized":
        codebook = tcq_optimized_codebook(bits, n_states)
    else:
        levels64, _ = lloyd_max_gaussian(bits + 1)
        codebook = levels64.to(torch.float32)
    w32 = w.to(torch.float32)
    codebook = codebook.to(w32.device)

    scales = w32.pow(2).mean(dim=1).sqrt()
    nonzero = scales > 0
    safe = torch.where(nonzero, scales, torch.ones_like(scales))

    level_codes, path_bits, member_codes, _ = _viterbi(w32 / safe[:, None], codebook, trellis)
    for _ in range(scale_iters):
        q = codebook[level_codes.long()]
        num = (w32 * q).sum(dim=1)
        den = (q * q).sum(dim=1)
        new_scale = torch.where(den > 0, num / den.clamp_min(1e-30), safe)
        safe = torch.where(new_scale > 0, new_scale, safe)
        level_codes, path_bits, member_codes, _ = _viterbi(
            w32 / safe[:, None], codebook, trellis
        )

    scales = torch.where(nonzero, safe, torch.zeros_like(safe))
    return TCQQuantized(
        level_codes=level_codes.to(w.device),
        path_bits=path_bits.to(w.device),
        member_codes=member_codes.to(w.device),
        scales=scales.to(w.device),
        codebook=codebook.to(w.device),
        bits=bits,
        n_states=n_states,
    )
