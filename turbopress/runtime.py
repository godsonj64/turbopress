"""TurboPress packed runtime: run models directly from trellis-coded bits.

The storage format everywhere else in this repo needs a decode step before
inference. This module is the runtime analogue of GGUF/llama.cpp for the
TurboPress format: weights stay packed at their true bit-width in memory and
are decoded on the fly, so runtime weight memory is ~bits/16 of fp16.

Why this is possible -- the window-decode property
--------------------------------------------------
Our trellis is a shift register: the state after step t is literally the
last m = log2(S) path bits. Therefore the subset chosen at position t is a
pure function of the (m+1)-bit window ``b[t-m..t]`` of the path stream
(zero-padded at the start, because encoding begins in state 0):

    subset_t = LUT[ window(b, t) ],   LUT has 2^(m+1) entries.

Decoding is embarrassingly parallel -- no sequential trellis walk. This is
the same property QTIP engineers deliberately ("bitshift trellis"); here it
falls out of the encoder construction. It enables (a) O(1)-depth vectorized
decode (used below; also makes artifact loading ~100x faster than the
sequential walk) and (b) fused decode-inside-GEMV kernels
(triton_kernel.py) where the fp16 weight matrix never exists in memory.

Format v2 (kernel-friendly, differs from the one-cell artifact's v1):
  * little-endian bit order within bytes, each row padded to a whole byte,
    so element (r, c)'s bits live at predictable offsets;
  * weights kept in the *rotated, equilibrated* basis; the activation-side
    transform (divide by s, multiply signs, block-FWHT) is applied at
    forward time, exactly as the algebra requires: W x = W'(R D^-1 x).

Execution modes for :class:`PackedTCQLinear`:
  * ``"cached"``: decode once at load, fold the rotation/equilibration back
    into the cached fp16 weights, and free the packed buffers. Forward is a
    plain F.linear -- fp16 speed and fp16 memory ("load-time decompression").
  * ``"tiled"``:  weights stay packed; row tiles decode per forward
    (memory = packed + one tile; the low-memory pure-PyTorch path --
    decode cost per call makes it slow, use for memory-bound loads).
  * ``"triton"``: fused packed GEMV -- packed memory AND bandwidth-bound
    speed (see triton_kernel.py; server GPUs).
"""

from __future__ import annotations

import math
import zlib
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from turbopress.hadamard import RandomizedOrthogonal
from turbopress.trellis import Trellis, tcq_quantize_rows

__all__ = [
    "pack_le",
    "unpack_le",
    "subset_lut",
    "window_decode_levels",
    "pack_linear",
    "PackedTCQLinear",
    "pack_model",
]

_MODES = ("cached", "tiled", "triton")


# ----------------------------------------------------------------------------
# little-endian, row-aligned bit packing
# ----------------------------------------------------------------------------
def pack_le(vals: Tensor, width: int) -> Tensor:
    """Pack uint8 values (< 2**width) little-endian, each row byte-aligned.

    [n, d] -> [n, ceil(d * width / 8)] uint8. Bit j of element c sits at
    stream position c*width + j; stream bit i occupies byte i>>3, bit i&7.
    """
    if vals.ndim != 2 or vals.dtype != torch.uint8:
        raise ValueError("expected a 2D uint8 tensor")
    if not 1 <= width <= 8:
        raise ValueError(f"width must be in [1, 8], got {width}")
    n, d = vals.shape
    bits = (vals[:, :, None] >> torch.arange(width, device=vals.device)) & 1
    stream = bits.reshape(n, d * width)
    pad = (-stream.shape[1]) % 8
    if pad:
        stream = F.pad(stream, (0, pad))
    weights = (1 << torch.arange(8, device=vals.device, dtype=torch.int32))
    return (stream.reshape(n, -1, 8).to(torch.int32) * weights).sum(-1).to(torch.uint8)


def unpack_le(packed: Tensor, d: int, width: int) -> Tensor:
    """Inverse of :func:`pack_le` -> [n, d] uint8."""
    n = packed.shape[0]
    shifts = torch.arange(8, device=packed.device, dtype=torch.uint8)
    bits = (packed[:, :, None] >> shifts) & 1
    stream = bits.reshape(n, -1)[:, : d * width]
    weights = (1 << torch.arange(width, device=packed.device, dtype=torch.int32))
    return (stream.reshape(n, d, width).to(torch.int32) * weights).sum(-1).to(torch.uint8)


# ----------------------------------------------------------------------------
# parallel window decode
# ----------------------------------------------------------------------------
def subset_lut(trellis: Trellis) -> Tensor:
    """LUT over the natural (m+1)-bit window integer -> subset index.

    Window integer x has b_{t-m+j} at bit j (stream order), i.e. the current
    bit b_t at bit m and the state (b_{t-1-k} at bit k) as the reverse of the
    low m bits -- the LUT absorbs that reindexing so callers never reverse.
    """
    m = trellis.n_states.bit_length() - 1
    xs = torch.arange(1 << (m + 1))
    b_cur = (xs >> m) & 1
    state = torch.zeros_like(xs)
    for k in range(m):
        state |= ((xs >> (m - 1 - k)) & 1) << k
    return trellis.subset_table[state, b_cur].to(torch.uint8)


def window_decode_levels(path_bits: Tensor, member_codes: Tensor, trellis: Trellis) -> Tensor:
    """Exact, walk-free equivalent of ``trellis.decode_levels``.

    All positions decode independently from an (m+1)-bit sliding window of
    the path stream (zero-padded: the encoder starts in state 0).
    """
    n, d = path_bits.shape
    m = trellis.n_states.bit_length() - 1
    lut = subset_lut(trellis).to(path_bits.device)
    padded = F.pad(path_bits.to(torch.int32), (m, 0))
    # Stream-order window: bit j of x is b_{t-m+j} (so b_t sits at bit m).
    # This matches both the LUT convention and the packed byte stream the
    # Triton kernel reads, keeping a single LUT for every consumer.
    x = padded[:, m:] << m
    for j in range(m):
        x = x | (padded[:, j : j + d] << j)
    subsets = lut[x.long()]
    return (member_codes.to(torch.int32) * 4 + subsets).to(torch.uint8)


# ----------------------------------------------------------------------------
# quantize an nn.Linear into a packed payload (format v2)
# ----------------------------------------------------------------------------
def pack_linear(
    linear: nn.Linear,
    act_rms: Tensor,
    bits: int,
    n_states: int = 64,
    seed: int = 0,
) -> dict[str, Any]:
    """TurboPress-quantize ``linear`` and return a packed format-v2 payload.

    Pipeline: quarter-power equilibration (all factors rounded to their
    stored fp16 precision before use, so the packed model is bit-identical
    to the validated one) -> seeded rotation (signs stored) -> TCQ.
    """
    w = linear.weight.detach().to(torch.float32).cpu()
    n, d = w.shape
    act = act_rms.to(torch.float32).cpu().clamp_min(1e-8)
    act = act.clamp_min(0.05 * float(act.pow(2).mean().sqrt()))
    cn = w.norm(dim=0)
    cn = cn.clamp_min(max(0.05 * float(cn.pow(2).mean().sqrt()), 1e-12))
    s = (act / cn).sqrt().to(torch.float16).float()

    rot = RandomizedOrthogonal(d, seed=seed)
    w_rot = rot(w * s[None, :])
    q = tcq_quantize_rows(w_rot, bits=bits, n_states=n_states)
    signs01 = (rot.signs > 0).to(torch.uint8)[None, :]

    return {
        "format": 2,
        "n": n,
        "d": d,
        "bits": bits,
        "n_states": n_states,
        "block": rot.block,
        "path_packed": pack_le(q.path_bits, 1),
        "member_packed": pack_le(q.member_codes, bits - 1) if bits > 1 else None,
        "scales": q.scales.to(torch.float16),
        "signs_packed": pack_le(signs01, 1),
        "inv_equil": (1.0 / s).to(torch.float16),
        "codebook": q.codebook.to(torch.float16),
        "bias": None if linear.bias is None else linear.bias.detach().to(torch.float16),
        "seed": seed,
    }


# ----------------------------------------------------------------------------
# the packed linear module
# ----------------------------------------------------------------------------
def _fwht(x: Tensor) -> Tensor:
    d = x.shape[-1]
    y = x.reshape(-1, d)
    h = 1
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        even, odd = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((even + odd, even - odd), dim=2).reshape(-1, d)
        h *= 2
    return (y / math.sqrt(d)).reshape(x.shape)


def _block_fwht(x: Tensor, block: int) -> Tensor:
    if block == x.shape[-1]:
        return _fwht(x)
    m = x.shape[-1] // block
    return _fwht(x.reshape(*x.shape[:-1], m, block)).reshape(x.shape)


class PackedTCQLinear(nn.Module):
    """Linear layer that runs from packed trellis-coded weights.

    ``mode="tiled"`` keeps only the packed bits resident (plus one decoded
    row tile during the matmul); ``"cached"`` trades memory for speed by
    decoding once; ``"triton"`` uses the fused GPU kernel when available.
    """

    def __init__(self, payload: dict[str, Any], mode: str = "cached",
                 tile_rows: int = 1024) -> None:
        super().__init__()
        if payload.get("format") != 2:
            raise ValueError("expected a format-v2 payload from pack_linear()")
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.out_features: int = payload["n"]
        self.in_features: int = payload["d"]
        self.bits: int = payload["bits"]
        self.n_states: int = payload["n_states"]
        self.block: int = payload["block"]
        self.mode = mode
        self.tile_rows = tile_rows
        self._trellis = Trellis(self.n_states)

        self.register_buffer("path_packed", payload["path_packed"])
        self.register_buffer("member_packed", payload["member_packed"])
        self.register_buffer("scales", payload["scales"])
        self.register_buffer("signs_packed", payload["signs_packed"])
        self.register_buffer("inv_equil", payload["inv_equil"])
        self.register_buffer("codebook", payload["codebook"])
        self.register_buffer("bias", payload["bias"])
        self.register_buffer("lut", subset_lut(self._trellis))
        # Combined activation-side transform for the packed modes: one fused
        # multiply (signs * 1/s) instead of an unpack + two multiplies per call.
        signs = unpack_le(payload["signs_packed"], self.in_features, 1)[0]
        self.register_buffer(
            "rot_scale",
            ((signs.float() * 2 - 1) * payload["inv_equil"].float()).to(torch.float16),
        )
        self.register_buffer("_w_cache", None)

        self.register_buffer("levels_packed", None)
        if mode == "triton":
            from turbopress.triton_kernel import HAS_TRITON
            if not HAS_TRITON:
                raise RuntimeError("mode='triton' requires the triton package")
            # Build the GPU decode stream once: window-decode to level codes
            # packed at (bits+1) bits/weight (nibble-aligned for 3-bit), so
            # the kernel does one aligned field extract + one small codebook
            # gather per weight -- no window math or LUT in the hot loop.
            # Runtime memory is (bits+1)/16 of fp16; disk stays at true rate.
            d = self.in_features
            pathb = unpack_le(self.path_packed, d, 1)
            if self.member_packed is not None:
                memb = unpack_le(self.member_packed, d, self.bits - 1)
            else:
                memb = torch.zeros_like(pathb)
            levels = window_decode_levels(pathb, memb, self._trellis)
            packed = pack_le(levels, self.bits + 1)
            # Pad rows to a 4-byte multiple: the kernel reads the stream as
            # coalesced u32 words (8 codes per load at 3-bit), not bytes.
            pad = (-packed.shape[1]) % 4
            if pad:
                packed = F.pad(packed, (0, pad))
            self.levels_packed = packed.contiguous()
            for name in ("path_packed", "member_packed", "lut"):
                setattr(self, name, None)
        elif mode == "cached":
            # Load-time decompression: decode once, rotate the weights back
            # to the original basis (W = W' R D^-1 row-wise), free the packed
            # streams. Forward becomes a plain F.linear at fp16 speed.
            w_rot = self.decode_rows(0, self.out_features).float()
            signs_f = signs.float() * 2 - 1
            w = _block_fwht(w_rot, self.block) * signs_f[None, :]
            w = w * payload["inv_equil"].float()[None, :]
            self._w_cache = w.to(torch.float16)
            for name in ("path_packed", "member_packed", "signs_packed", "lut"):
                setattr(self, name, None)

    # -- decode -----------------------------------------------------------
    def decode_rows(self, r0: int, r1: int) -> Tensor:
        """Decode rows [r0, r1) of the rotated weight matrix to fp16."""
        d = self.in_features
        pathb = unpack_le(self.path_packed[r0:r1], d, 1)
        if self.member_packed is not None:
            memb = unpack_le(self.member_packed[r0:r1], d, self.bits - 1)
        else:
            memb = torch.zeros_like(pathb)
        levels = window_decode_levels(pathb, memb, self._trellis)
        w = self.codebook.float()[levels.long()] * self.scales[r0:r1].float()[:, None]
        return w.to(torch.float16)

    # -- forward ----------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {x.shape[-1]}")
        compute_dtype = torch.float16 if x.is_cuda else torch.float32
        if self.mode == "cached":
            # Rotation already folded into the cached weights at load time.
            y = F.linear(x.to(compute_dtype), self._w_cache.to(compute_dtype))
        else:
            if self.mode == "triton" and x.is_cuda and self.block >= 16:
                # fused single-launch rotation: ~30 butterfly kernels -> 1
                from turbopress.triton_kernel import fwht_scale, packed_gemv
                z = fwht_scale(x, self.rot_scale, self.block)
                y = packed_gemv(z, self)
            elif self.mode == "triton":
                from turbopress.triton_kernel import packed_gemv
                z = _block_fwht(x.to(compute_dtype) * self.rot_scale.to(compute_dtype),
                                self.block)
                y = packed_gemv(z, self)
            else:  # tiled
                z = _block_fwht(x.to(compute_dtype) * self.rot_scale.to(compute_dtype),
                                self.block)
                flat = z.reshape(-1, self.in_features)
                y = torch.empty(flat.shape[0], self.out_features,
                                dtype=compute_dtype, device=z.device)
                for r0 in range(0, self.out_features, self.tile_rows):
                    r1 = min(r0 + self.tile_rows, self.out_features)
                    y[:, r0:r1] = flat @ self.decode_rows(r0, r1).to(compute_dtype).T
                y = y.reshape(*z.shape[:-1], self.out_features)
        if self.bias is not None:
            y = y + self.bias.to(y.dtype)
        return y.to(x.dtype)

    # -- accounting ---------------------------------------------------------
    def packed_bytes(self) -> int:
        """Resident weight bytes under the current mode.

        ``cached`` frees the packed streams and holds fp16 weights; the
        packed modes hold the bit-streams plus small fp16 side tensors.
        """
        if self._w_cache is not None:
            total = self._w_cache.numel() * self._w_cache.element_size()
        elif self.levels_packed is not None:  # triton decode stream
            total = self.levels_packed.numel() + self.signs_packed.numel()
            total += 2 * (self.inv_equil.numel() + self.rot_scale.numel())
        else:
            total = self.path_packed.numel() + self.signs_packed.numel()
            if self.member_packed is not None:
                total += self.member_packed.numel()
            total += 2 * (self.inv_equil.numel() + self.rot_scale.numel())
        total += 2 * (self.scales.numel() + self.codebook.numel())
        if self.bias is not None:
            total += 2 * self.bias.numel()
        return total

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits={self.bits}, n_states={self.n_states}, mode={self.mode!r}, "
                f"resident={self.packed_bytes() / 2**20:.1f} MiB "
                f"(fp16 {2 * self.in_features * self.out_features / 2**20:.1f} MiB)")


# ----------------------------------------------------------------------------
# whole-model conversion
# ----------------------------------------------------------------------------
@torch.no_grad()
def pack_model(
    model: nn.Module,
    stats: dict[str, Tensor],
    bits: int,
    n_states: int = 64,
    mode: str = "cached",
    seed: int = 0,
    log=None,
) -> dict[str, int]:
    """Replace every decoder nn.Linear with a PackedTCQLinear, in place.

    ``stats`` maps ``"{layer_idx}.{name}"`` to per-channel activation RMS
    (see ``real_model.collect_input_scales``). Returns byte accounting.
    """
    from turbopress.real_model import _decoder_layers

    layers = _decoder_layers(model)
    packed_total = fp16_total = 0
    for i, block in enumerate(layers):
        replacements = []
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                payload = pack_linear(
                    mod, stats[f"{i}.{name}"], bits=bits, n_states=n_states,
                    seed=seed + 7919 * i + zlib.crc32(name.encode()) % 1000,
                )
                replacements.append((name, PackedTCQLinear(payload, mode=mode)))
        for name, qlin in replacements:
            parent = block
            *parents, leaf = name.split(".")
            for p in parents:
                parent = getattr(parent, p)
            device = getattr(parent, leaf).weight.device
            setattr(parent, leaf, qlin.to(device))
            packed_total += qlin.packed_bytes()
            fp16_total += 2 * qlin.in_features * qlin.out_features
        if log is not None:
            log.info(f"packed block {i + 1}/{len(layers)}")
    return {
        "packed_bytes": packed_total,
        "fp16_bytes": fp16_total,
        "compression": fp16_total / max(packed_total, 1),
    }
