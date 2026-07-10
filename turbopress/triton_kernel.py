"""Fused packed-TCQ GEMV kernel (Triton): decode-inside-matmul.

The fp16 weight matrix never materializes: each program decodes a
(BLOCK_N x BLOCK_D) tile of trellis-coded weights straight from the packed
bit-streams (window-parallel decode, see runtime.py) and accumulates the
matvec on the fly. For single-token generation the GEMV is memory-bandwidth
bound, so reading ``bits`` bits per weight instead of 16 is not just a
memory win -- it is the speedup, with a theoretical ceiling of 16/bits x
over an fp16 GEMV.

Status: written for server GPUs (A10/A100/H100); Triton has no supported
wheel on this Windows dev box, so tests exercise the torch reference paths
and skip the kernel unless triton imports. Batch > 1 falls back to looping
the batch dimension; use mode="cached" for large-batch prefill.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - dev box has no triton wheel
    HAS_TRITON = False

__all__ = ["HAS_TRITON", "packed_gemv"]


if HAS_TRITON:

    _CONFIGS = [
        triton.Config({"BLOCK_N": bn, "BLOCK_D": bd}, num_warps=w)
        for bn in (32, 64, 128)
        for bd in (128, 256, 512)
        for w in (2, 4, 8)
    ]

    @triton.autotune(configs=_CONFIGS, key=["D", "N"])
    @triton.jit
    def _tcq_gemv(
        z_ptr,          # [D]  activation row (already rotated/equilibrated)
        path_ptr,       # [N, PATH_BYTES] packed path bits (LE, row-aligned)
        memb_ptr,       # [N, MEMB_BYTES] packed member bits (LE, row-aligned)
        scales_ptr,     # [N]  fp16 per-row scales
        cb_ptr,         # [2**(BITS+1)] fp16 codebook
        lut_ptr,        # [2**(M+1)] uint8 window->subset LUT
        y_ptr,          # [N]  fp32 output
        D, N,
        PATH_BYTES, MEMB_BYTES,
        M: tl.constexpr,          # log2(n_states)
        MEMBER_W: tl.constexpr,   # bits - 1 (member width; 0 handled on host)
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_b = tl.program_id(1)  # batch row: one launch covers the batch
        z_ptr = z_ptr + pid_b * D
        y_ptr = y_ptr + pid_b * N
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = rn < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        scales = tl.load(scales_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)

        for d0 in range(0, D, BLOCK_D):
            rc = d0 + tl.arange(0, BLOCK_D)
            d_mask = rc < D
            zv = tl.load(z_ptr + rc, mask=d_mask, other=0.0).to(tl.float32)

            # ---- path window: bits rc-M .. rc of each row's path stream ----
            # stream bit i lives at byte i>>3, bit i&7 (little-endian). Load
            # two consecutive bytes so any (M+1)<=8 bit window is covered.
            start = rc - M
            start = tl.where(start < 0, 0, start)  # clamp; low bits masked below
            byte0 = start >> 3
            shift = start & 7
            offs = rn[:, None] * PATH_BYTES + byte0[None, :]
            b_lo = tl.load(path_ptr + offs, mask=n_mask[:, None] & d_mask[None, :],
                           other=0).to(tl.int32)
            b_hi = tl.load(path_ptr + offs + 1,
                           mask=n_mask[:, None] & (byte0[None, :] + 1 < PATH_BYTES),
                           other=0).to(tl.int32)
            win = ((b_lo | (b_hi << 8)) >> shift[None, :]) & ((1 << (M + 1)) - 1)
            # zero-pad the start-of-row window (encoder starts in state 0):
            # for rc < M the clamped window contains bits 0..rc; shift them up
            # so the current bit lands at position M.
            deficit = tl.where(rc < M, M - rc, 0)
            win = (win << deficit[None, :]) & ((1 << (M + 1)) - 1)
            subset = tl.load(lut_ptr + win).to(tl.int32)

            # ---- member bits ------------------------------------------------
            mstart = rc * MEMBER_W
            mbyte = mstart >> 3
            mshift = mstart & 7
            moffs = rn[:, None] * MEMB_BYTES + mbyte[None, :]
            mm = n_mask[:, None] & d_mask[None, :]
            m_lo = tl.load(memb_ptr + moffs, mask=mm, other=0).to(tl.int32)
            m_hi = tl.load(memb_ptr + moffs + 1,
                           mask=n_mask[:, None] & (mbyte[None, :] + 1 < MEMB_BYTES),
                           other=0).to(tl.int32)
            member = ((m_lo | (m_hi << 8)) >> mshift[None, :]) & ((1 << MEMBER_W) - 1)

            level = member * 4 + subset
            w = tl.load(cb_ptr + level).to(tl.float32) * scales[:, None]
            acc += tl.sum(tl.where(mm, w * zv[None, :], 0.0), axis=1)

        tl.store(y_ptr + rn, acc, mask=n_mask)


def packed_gemv(z: Tensor, layer) -> Tensor:
    """y = z @ W_packed^T for a PackedTCQLinear, decoding inside the kernel.

    ``z`` is the already-rotated activation [..., in_features]. Batch rows
    are looped (generation workloads have batch ~1); prefill should use
    mode="cached".
    """
    if not HAS_TRITON:  # pragma: no cover
        raise RuntimeError("triton is not available")
    if layer.bits < 2:
        raise ValueError("packed_gemv requires bits >= 2 (member width >= 1)")
    flat = z.reshape(-1, layer.in_features).contiguous()
    n, d = layer.out_features, layer.in_features
    m = layer.n_states.bit_length() - 1
    out = torch.empty(flat.shape[0], n, dtype=torch.float32, device=z.device)
    grid = lambda meta: ((n + meta["BLOCK_N"] - 1) // meta["BLOCK_N"], flat.shape[0])  # noqa: E731
    _tcq_gemv[grid](
        flat, layer.path_packed, layer.member_packed,
        layer.scales, layer.codebook, layer.lut, out,
        d, n, layer.path_packed.shape[1], layer.member_packed.shape[1],
        M=m, MEMBER_W=layer.bits - 1,
    )
    return out.reshape(*z.shape[:-1], n).to(z.dtype)
