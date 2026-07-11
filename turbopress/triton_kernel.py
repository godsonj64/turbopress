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

__all__ = ["HAS_TRITON", "packed_gemv", "fwht_scale"]


if HAS_TRITON:

    _CONFIGS = [
        triton.Config({"BLOCK_N": bn, "BLOCK_D": bd, "SPLIT_D": sp},
                      num_warps=w, num_stages=ns)
        for bn in (32, 64, 128)
        for bd in (512, 1024)
        for w in (4, 8)
        for sp in (1, 4, 8)
        for ns in (2, 3)
    ]

    @triton.autotune(configs=_CONFIGS, key=["D", "N"], reset_to_zero=["y_ptr"])
    @triton.jit
    def _tcq_gemv(
        z_ptr,         # [B, D] rotated/equilibrated activations
        lv_ptr,        # [N, LV_BYTES] packed level codes (LE, row 4B-padded)
        lv32_ptr,      # the same buffer viewed as int32 words
        scales_ptr,    # [N] fp16 per-row scales
        cb_ptr,        # [2**W] fp16 codebook
        y_ptr,         # [B, N] fp32 output (zero-initialized: split-D atomics)
        B, D, N, LV_BYTES, LV_WORDS,
        W: tl.constexpr,         # bits + 1 (level-code width)
        WORDPACK: tl.constexpr,  # W divides 32: coalesced word-load fast path
        CPW: tl.constexpr,       # codes per 32-bit word (32 // W)
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SPLIT_D: tl.constexpr,   # D-axis parallelism (GEMV grids starve SMs)
    ):
        pid = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_s = tl.program_id(2)
        rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        rb = pid_b * 16 + tl.arange(0, 16)  # batch tile: tl.dot needs >=16
        n_mask = rn < N
        b_mask = rb < B
        # Note: an explicit batch-1 FMA-reduction path (no dot tile) was
        # tried and measured SLOWER (tl.sum reduction trees lose to masked
        # tensor-core dots on Blackwell: 1536x576 8.2us -> 21us), so the
        # dot path serves every batch size.
        acc = tl.zeros((BLOCK_N, 16), dtype=tl.float32)
        # Row scales are constant along D: factor them out of the reduction
        # and multiply once at the end.
        scales = tl.load(scales_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)

        # this split's D range, rounded to whole BLOCK_D tiles
        chunk = tl.cdiv(tl.cdiv(D, BLOCK_D), SPLIT_D) * BLOCK_D
        d_lo = pid_s * chunk
        d_hi = tl.minimum(d_lo + chunk, D)

        for d0 in range(d_lo, d_hi, BLOCK_D):
            if WORDPACK:
                # Coalesced fast path (W = 2/4/8): one u32 load per lane
                # yields CPW codes; each warp transaction moves 128 useful
                # bytes instead of the ~16 a byte-gather achieves. The codes
                # are bit-sliced in registers into CPW column slices; the
                # matching activation slices are strided but tiny (z is
                # L1-resident), and each slice feeds one tensor-core dot.
                rw = (d0 // CPW) + tl.arange(0, BLOCK_D // CPW)
                w_mask = rw < LV_WORDS
                words = tl.load(
                    lv32_ptr + rn[:, None] * LV_WORDS + rw[None, :],
                    mask=n_mask[:, None] & w_mask[None, :], other=0,
                )
                for k in tl.static_range(CPW):
                    level = (words >> (k * W)) & ((1 << W) - 1)
                    wk = tl.load(cb_ptr + level).to(tl.float16)
                    rc_k = rw * CPW + k
                    zv_k = tl.load(
                        z_ptr + rb[None, :] * D + rc_k[:, None],
                        mask=(rc_k < D)[:, None] & b_mask[None, :], other=0.0,
                    )
                    acc = tl.dot(wk, zv_k.to(tl.float16), acc)
            else:
                # General path (W = 3/5/6/7): fields cross byte boundaries.
                rc = d0 + tl.arange(0, BLOCK_D)
                d_mask = rc < D
                zv = tl.load(z_ptr + rb[None, :] * D + rc[:, None],
                             mask=d_mask[:, None] & b_mask[None, :], other=0.0)
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
                w = tl.load(cb_ptr + level)
                w = tl.where(mm, w, 0.0).to(tl.float16)
                acc = tl.dot(w, zv.to(tl.float16), acc)

        out = acc * scales[:, None]
        y_offs = rb[None, :] * N + rn[:, None]
        y_mask = n_mask[:, None] & b_mask[None, :]
        if SPLIT_D == 1:
            tl.store(y_ptr + y_offs, out, mask=y_mask)
        else:
            tl.atomic_add(y_ptr + y_offs, out, mask=y_mask)


def packed_gemv(z: Tensor, layer) -> Tensor:
    """y = z @ W_packed^T for a PackedTCQLinear in mode="triton"."""
    if not HAS_TRITON:  # pragma: no cover
        raise RuntimeError("triton is not available")
    if layer.levels_packed is None:
        raise RuntimeError("layer was not built with mode='triton'")
    flat = z.reshape(-1, layer.in_features).contiguous()
    n, d = layer.out_features, layer.in_features
    # zeros, not empty: split-D configs accumulate with atomic adds.
    out = torch.zeros(flat.shape[0], n, dtype=torch.float32, device=z.device)
    grid = lambda meta: ((n + meta["BLOCK_N"] - 1) // meta["BLOCK_N"],  # noqa: E731
                         (flat.shape[0] + 15) // 16,
                         meta["SPLIT_D"])
    w_bits = layer.bits + 1
    lv = layer.levels_packed
    wordpack = 32 % w_bits == 0 and lv.shape[1] % 4 == 0
    # the WORDPACK=False specialization never dereferences lv32_ptr (the
    # branch is compiled out), so any valid pointer stands in for it there.
    lv32 = lv.view(torch.int32) if wordpack else lv
    _tcq_gemv[grid](
        flat, lv, lv32, layer.scales, layer.codebook, out,
        flat.shape[0], d, n, lv.shape[1],
        lv.shape[1] // 4 if wordpack else lv.shape[1],
        W=w_bits, WORDPACK=wordpack, CPW=max(32 // w_bits, 1),
    )
    return out.reshape(*z.shape[:-1], n).to(z.dtype)


# ----------------------------------------------------------------------------
# fused rotation: z = blockFWHT(x * rot_scale) in ONE kernel launch
# ----------------------------------------------------------------------------
# The torch fallback runs the butterfly as ~4 elementwise kernels per stage
# (~30 launches for a 1024-point block-transform) and was measured to cost
# ~28 us/layer inside a CUDA-graphed decode step -- 2-9x the fused GEMV it
# feeds. The Walsh-Hadamard transform factorizes over any bit split of the
# index (H_{AB} = H_A (x) H_B), so an s-point transform on the row-major
# reshape [A, B] is just  H_A @ V @ H_B  -- two tensor-core dots.

if HAS_TRITON:

    @triton.jit
    def _fwht_scale(
        x_ptr,        # [B, D] fp16/fp32 input
        rs_ptr,       # [D] fp16 per-channel scale (signs * 1/equil)
        ha_ptr,       # [A, A] fp16 normalized Hadamard factor
        hb_ptr,       # [BB, BB] fp16 normalized Hadamard factor
        y_ptr,        # [B, D] fp16 output
        B, D,
        BS: tl.constexpr,        # transform block size = A * BB
        A: tl.constexpr,
        BB: tl.constexpr,
        TWO_STAGE: tl.constexpr,  # False: single H_BS dot over a batch tile
    ):
        pid_m = tl.program_id(0)  # which transform block along D
        pid_b = tl.program_id(1)  # row (TWO_STAGE) or batch tile (single)
        base = pid_m * BS
        if TWO_STAGE:
            ra = tl.arange(0, A)
            rb = tl.arange(0, BB)
            offs = base + ra[:, None] * BB + rb[None, :]
            x = tl.load(x_ptr + pid_b * D + offs).to(tl.float32)
            rs = tl.load(rs_ptr + offs).to(tl.float32)
            v = (x * rs).to(tl.float16)
            ha = tl.load(ha_ptr + ra[:, None] * A + ra[None, :])
            hb = tl.load(hb_ptr + rb[:, None] * BB + rb[None, :])
            y = tl.dot(ha, v)                      # [A, BB] fp32
            y = tl.dot(y.to(tl.float16), hb)       # [A, BB] fp32
            tl.store(y_ptr + pid_b * D + offs, y.to(tl.float16))
        else:
            # small blocks (< 256): one H_BS dot across a 16-row batch tile
            rows = pid_b * 16 + tl.arange(0, 16)
            r_mask = rows < B
            rc = tl.arange(0, BS)
            x = tl.load(x_ptr + rows[None, :] * D + (base + rc)[:, None],
                        mask=r_mask[None, :], other=0.0).to(tl.float32)
            rs = tl.load(rs_ptr + base + rc).to(tl.float32)
            v = (x * rs[:, None]).to(tl.float16)
            h = tl.load(ha_ptr + rc[:, None] * BS + rc[None, :])
            y = tl.dot(h, v)                       # [BS, 16] fp32
            tl.store(y_ptr + rows[None, :] * D + (base + rc)[:, None],
                     y.to(tl.float16), mask=r_mask[None, :])


# normalized Hadamard factor matrices, cached per (size, device)
_H_CACHE: dict[tuple[int, torch.device], Tensor] = {}


def _hadamard(size: int, device) -> Tensor:
    key = (size, device)
    if key not in _H_CACHE:
        from turbopress.runtime import _fwht
        _H_CACHE[key] = _fwht(torch.eye(size)).to(device, torch.float16).contiguous()
    return _H_CACHE[key]


def _factor(bs: int) -> tuple[int, int]:
    """Split bs = A * B with both factors >= 16 and as square as possible."""
    a = 1 << ((bs.bit_length() - 1) // 2)
    while a * a < bs:
        a <<= 1
    return a, bs // a


def fwht_scale(x: Tensor, rot_scale: Tensor, block: int) -> Tensor:
    """``blockFWHT(x * rot_scale)`` fused into one kernel launch.

    Bit-exactness note: fp16 inputs with fp32 dot accumulation -- matches
    the torch path to fp16 tolerance (validated in tests), not bitwise.
    """
    if not HAS_TRITON:  # pragma: no cover
        raise RuntimeError("triton is not available")
    d = x.shape[-1]
    flat = x.reshape(-1, d).to(torch.float16).contiguous()
    b = flat.shape[0]
    out = torch.empty_like(flat)
    m = d // block
    if block >= 256:
        a, bb = _factor(block)
        _fwht_scale[(m, b)](
            flat, rot_scale, _hadamard(a, x.device), _hadamard(bb, x.device),
            out, b, d, BS=block, A=a, BB=bb, TWO_STAGE=True,
        )
    else:
        h = _hadamard(block, x.device)
        _fwht_scale[(m, (b + 15) // 16)](
            flat, rot_scale, h, h,
            out, b, d, BS=block, A=block, BB=block, TWO_STAGE=False,
        )
    return out.reshape(x.shape)
