"""Fused packed-TCQ GEMV kernel (Triton): decode-inside-matmul.

The fp16 weight matrix never materializes. The runtime builds a GPU decode
stream at load time (level codes packed at bits+1 bits/weight -- see
runtime.py; the trellis window decode runs once, offline), so the hot loop
per weight is a single aligned bit-field extract plus a small codebook
gather, then an FMA. Runtime weight memory is (bits+1)/16 of fp16; the
on-disk artifact stays at the true trellis rate.

Autotuned over block shape and warps; one launch covers the whole batch.
Validated on RTX 5050 (Blackwell) via the triton-windows wheel.
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - no wheel on some platforms
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
        z_ptr,         # [B, D] rotated/equilibrated activations
        lv_ptr,        # [N, LV_BYTES] packed level codes (LE, row-aligned)
        scales_ptr,    # [N] fp16 per-row scales
        cb_ptr,        # [2**W] fp16 codebook
        y_ptr,         # [B, N] fp32 output
        B, D, N, LV_BYTES,
        W: tl.constexpr,       # bits + 1 (level-code width)
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_b = tl.program_id(1)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rb = pid_b * 16 + tl.arange(0, 16)  # batch tile: tl.dot needs >=16
        n_mask = rn < N
        b_mask = rb < B
        acc = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
        scales = tl.load(scales_ptr + rn, mask=n_mask, other=0.0).to(tl.float16)

        for d0 in range(0, D, BLOCK_D):
            rc = d0 + tl.arange(0, BLOCK_D)
            d_mask = rc < D
            # activations tile [BLOCK_D, 16]
            zv = tl.load(z_ptr + rb[None, :] * D + rc[:, None],
                         mask=d_mask[:, None] & b_mask[None, :], other=0.0)
            # decode the weight tile [BLOCK_N, BLOCK_D]: one aligned field
            # extract + one small codebook gather per weight. For W = 4
            # (3-bit models) the field is nibble-aligned.
            start = rc * W
            byte0 = start >> 3
            shift = start & 7
            offs = rn[:, None] * LV_BYTES + byte0[None, :]
            mm = n_mask[:, None] & d_mask[None, :]
            b_lo = tl.load(lv_ptr + offs, mask=mm, other=0).to(tl.int32)
            b_hi = tl.load(lv_ptr + offs + 1,
                           mask=n_mask[:, None] & (byte0[None, :] + 1 < LV_BYTES),
                           other=0).to(tl.int32)
            level = ((b_lo | (b_hi << 8)) >> shift[None, :]) & ((1 << W) - 1)
            w = tl.load(cb_ptr + level) * scales[:, None]
            w = tl.where(mm, w, 0.0).to(tl.float16)
            # tensor-core matmul: [BLOCK_N, BLOCK_D] @ [BLOCK_D, 16]
            acc = tl.dot(w, zv.to(tl.float16), acc)

        y_offs = rb[None, :] * N + rn[:, None]
        tl.store(y_ptr + y_offs, acc, mask=n_mask[:, None] & b_mask[None, :])


def packed_gemv(z: Tensor, layer) -> Tensor:
    """y = z @ W_packed^T for a PackedTCQLinear in mode="triton"."""
    if not HAS_TRITON:  # pragma: no cover
        raise RuntimeError("triton is not available")
    if layer.levels_packed is None:
        raise RuntimeError("layer was not built with mode='triton'")
    flat = z.reshape(-1, layer.in_features).contiguous()
    n, d = layer.out_features, layer.in_features
    out = torch.empty(flat.shape[0], n, dtype=torch.float32, device=z.device)
    grid = lambda meta: ((n + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],  # noqa: E731
                         (flat.shape[0] + 15) // 16)
    _tcq_gemv[grid](
        flat, layer.levels_packed, layer.scales, layer.codebook, out,
        flat.shape[0], d, n, layer.levels_packed.shape[1],
        W=layer.bits + 1,
    )
    return out.reshape(*z.shape[:-1], n).to(z.dtype)
