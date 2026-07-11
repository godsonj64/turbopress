"""QJLCorrectedLinear: a drop-in quantized replacement for nn.Linear.

Forward path (exact in exact arithmetic before quantization):

    y = W x + b = (W D R^T)(R D^-1 x) + b
      ~= W_hat (R D^-1 x) + QJL_correction(R D^-1 x) + b

where D = diag(col_scale) is an optional activation-aware equilibration
(identity when ``col_scale`` is None), R is a seeded randomized orthogonal
transform (hadamard.py), W_hat is the low-bit reconstruction of W D R^T
(quantizer.py or trellis.py), and the QJL correction (qjl.py) is an unbiased
estimate of the residual inner products, making the whole output unbiased
over the sketch randomness.

Rotation-aware equilibration: after the rotation, quantization error is
spread ~uniformly over rotated coordinates, so the expected output error is

    E ||Delta y||^2  ~  eps^2 / d * (sum_j c_j s_j^2) * (sum_j m_j / s_j^2),

with c_j = ||W_{:,j}||^2 (weight column energy) and m_j = E[x_j^2]
(calibrated activation energy). By Cauchy-Schwarz this is minimized at
s_j = (m_j / c_j)^(1/4) -- NOT the AWQ/SmoothQuant choice s_j = m_j^(1/2),
which is optimal only for quantizers whose error stays column-aligned. The
caller passes the activation RMS sqrt(m_j) as ``col_scale``; the optimal
fold against the weight column norms is computed here. This is the one
data-dependent stage; everything else is calibration-free.

This is a *measurement* implementation: codes are stored at their true
bit-widths for accounting (see ``storage_report``), but the forward pass
materializes W_hat in float32 for speed and clarity rather than using packed
low-bit kernels. Numerical results are identical to what a packed kernel
would produce; only memory/latency of the prototype are not representative.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from turbopress.gptq import ldlq_quantize_rows, ldlq_tcq_quantize_rows, rotated_hessian
from turbopress.hadamard import RandomizedOrthogonal
from turbopress.qjl import QJLSketch, build_qjl_sketch, correction_matrix
from turbopress.quantizer import quantize_rows
from turbopress.trellis import tcq_quantize_rows

__all__ = ["QJLCorrectedLinear"]

# Seed offsets so the rotation, quantizer, and sketch draw independent streams
# from a single user-provided seed.
_ROT_SEED_OFFSET = 0x9E3779B1
_QUANT_SEED_OFFSET = 0x85EBCA6B
_SKETCH_SEED_OFFSET = 0xC2B2AE35


class QJLCorrectedLinear(nn.Module):
    """Low-bit linear layer with optional unbiased QJL output correction.

    Build with :meth:`from_linear`. ``sketch_k=0`` disables the correction
    (biased nearest-rounding baseline); ``sketch_k=in_features`` costs exactly
    1 extra bit per weight and matches the TurboQuant inner-product design.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int,
        sketch_k: int,
        seed: int,
        rounding: str,
        method: str,
        rotation: RandomizedOrthogonal,
        weight_hat: Tensor,
        codes: Tensor,
        scales: Tensor,
        codebook: Tensor,
        sketch: QJLSketch | None,
        bias: Tensor | None,
        inv_col_scale: Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.sketch_k = sketch_k
        self.seed = seed
        self.rounding = rounding
        self.method = method
        self.rotation = rotation

        self.register_buffer("weight_hat", weight_hat)
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales)
        self.register_buffer("codebook", codebook)
        self.register_buffer("bias", bias)
        self.register_buffer("inv_col_scale", inv_col_scale)
        if sketch is not None:
            self.register_buffer("sketch_proj", sketch.proj)
            self.register_buffer("sketch_signs", sketch.signs)
            self.register_buffer("sketch_norms", sketch.norms)
            self.register_buffer("sketch_corr", correction_matrix(sketch))
        else:
            self.register_buffer("sketch_proj", None)
            self.register_buffer("sketch_signs", None)
            self.register_buffer("sketch_norms", None)
            self.register_buffer("sketch_corr", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 2,
        sketch_k: int = 0,
        seed: int = 0,
        rounding: str = "nearest",
        scale_iters: int = 2,
        method: str = "scalar",
        n_states: int = 8,
        col_scale: Tensor | None = None,
        equil_mode: str = "quarter",
        equil_alpha: float = 0.25,
        error_feedback: bool = False,
        hessian: Tensor | None = None,
    ) -> "QJLCorrectedLinear":
        """Quantize an ``nn.Linear`` in place of its float weights.

        ``sketch_k`` is the number of QJL sign bits per output row (0 disables
        the correction; each unit costs ``1/in_features`` bits per weight).
        ``method="scalar"`` uses the Lloyd-Max scalar quantizer with the given
        ``rounding``; ``method="tcq"`` uses trellis-coded quantization with
        ``n_states`` trellis states (rounding is ignored: TCQ is deterministic).
        ``col_scale`` is an optional positive [in_features] vector of
        per-input-channel activation RMS values sqrt(E[x_j^2]) from a
        calibration set. The rotation-aware optimal equilibration
        s_j = (sqrt(E[x_j^2]) / ||W_{:,j}||)^(1/2) is derived from it (see
        module docstring), folded into the weights before quantization, and
        divided out of the activations at inference, exactly.
        """
        if not isinstance(linear, nn.Linear):
            raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
        if sketch_k < 0:
            raise ValueError(f"sketch_k must be >= 0, got {sketch_k}")
        if method not in ("scalar", "tcq"):
            raise ValueError(f"method must be 'scalar' or 'tcq', got {method!r}")
        if equil_mode not in ("quarter", "awq"):
            raise ValueError(f"equil_mode must be 'quarter' or 'awq', got {equil_mode!r}")
        if error_feedback and hessian is None:
            raise ValueError("error_feedback=True requires a hessian")

        in_features = linear.in_features
        out_features = linear.out_features
        w = linear.weight.detach().to(torch.float32)
        device = w.device

        inv_col_scale = None
        if col_scale is not None:
            col_scale = col_scale.detach().to(torch.float32).to(device)
            if col_scale.shape != (in_features,):
                raise ValueError(
                    f"col_scale must have shape ({in_features},), "
                    f"got {tuple(col_scale.shape)}"
                )
            if not torch.all(torch.isfinite(col_scale)) or torch.any(col_scale <= 0):
                raise ValueError("col_scale entries must be finite and positive")
            if equil_mode == "quarter":
                # Generalized rotation-aware fold: s_j = m_j^a / c_j^(1/2 - a)
                # with m_j = E[x_j^2] (col_scale = sqrt(m_j)), c_j = ||W_{:,j}||^2
                # and a = equil_alpha. a = 1/4 is the Prop-1 optimum for a
                # feedback-free rotated quantizer, s_j = (m_j/c_j)^(1/4);
                # a = 0 is the Prop-2 ideal-error-feedback optimum, pure
                # column normalization s_j = 1/||W_{:,j}|| (the determinant
                # bound makes activation statistics drop out when LDLQ
                # absorbs all linearly-predictable structure).
                # Column norms are floored to keep the fold bounded on dead cols.
                if not 0.0 <= equil_alpha <= 0.5:
                    raise ValueError(
                        f"equil_alpha must be in [0, 0.5], got {equil_alpha}"
                    )
                col_norm = w.norm(dim=0)
                floor = max(0.05 * float(col_norm.pow(2).mean().sqrt()), 1e-12)
                col_norm = col_norm.clamp_min(floor)
                s = col_scale.pow(2 * equil_alpha) / col_norm.pow(1 - 2 * equil_alpha)
            else:  # "awq": SmoothQuant/AWQ square-root activation fold, s_j =
                # sqrt(E[x_j^2]) (ignores the weight column norm), normalized to
                # unit geometric mean. This is the baseline Prop. 1 argues against.
                s = col_scale / col_scale.log().mean().exp()
            w = w * s[None, :]
            inv_col_scale = 1.0 / s

        mask = (1 << 31) - 1
        rotation = RandomizedOrthogonal(in_features, (seed + _ROT_SEED_OFFSET) & mask)
        rotation = rotation.to(device)
        w_rot = rotation(w)  # rows of W R^T, i.e. each row rotated by R

        if method == "tcq":
            if error_feedback:
                # Block-LDLQ over the trellis (QTIP-style): Hessian-aware
                # error feedback with the trellis as the block quantizer.
                hz = rotated_hessian(hessian.to(device), rotation, inv_col_scale)
                quantized = ldlq_tcq_quantize_rows(
                    w_rot, hz, bits=bits, n_states=n_states
                )
            else:
                quantized = tcq_quantize_rows(w_rot, bits=bits, n_states=n_states)
            codes = quantized.level_codes
        elif error_feedback:
            hz = rotated_hessian(hessian.to(device), rotation, inv_col_scale)
            quantized = ldlq_quantize_rows(w_rot, hz, bits=bits)
            codes = quantized.codes
        else:
            gen = torch.Generator().manual_seed((seed + _QUANT_SEED_OFFSET) & mask)
            quantized = quantize_rows(
                w_rot, bits=bits, rounding=rounding, scale_iters=scale_iters, generator=gen
            )
            codes = quantized.codes
        weight_hat = quantized.dequantize()

        sketch = None
        if sketch_k > 0:
            residual = w_rot - weight_hat
            sketch = build_qjl_sketch(
                residual, k=sketch_k, seed=(seed + _SKETCH_SEED_OFFSET) & mask
            )

        bias = None
        if linear.bias is not None:
            bias = linear.bias.detach().to(torch.float32).clone()

        return cls(
            in_features=in_features,
            out_features=out_features,
            bits=bits,
            sketch_k=sketch_k,
            seed=seed,
            rounding=rounding,
            method=method,
            rotation=rotation,
            weight_hat=weight_hat,
            codes=codes,
            scales=quantized.scales,
            codebook=quantized.codebook,
            sketch=sketch,
            bias=bias,
            inv_col_scale=inv_col_scale,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        x32 = x.to(torch.float32)
        if self.inv_col_scale is not None:
            x32 = x32 * self.inv_col_scale
        z = self.rotation(x32)
        y = F.linear(z, self.weight_hat)
        if self.sketch_corr is not None:
            y = y + (z @ self.sketch_proj.T) @ self.sketch_corr.T
        if self.bias is not None:
            y = y + self.bias
        return y.to(x.dtype)

    def storage_report(self) -> dict[str, Any]:
        """Bit accounting for the canonical stored form (not the fp32 cache).

        Per-row overheads: one float16 scale, plus (if sketched) one float16
        residual norm and ``sketch_k`` sign bits. The rotation and the sketch
        projection are seeded and cost O(1). The shared codebook is counted
        once. ``weight_hat`` and ``sketch_corr``/``sketch_proj`` are runtime
        caches derivable from the stored fields, so they are excluded.
        """
        n, d = self.out_features, self.in_features
        n_weights = n * d
        # For method="tcq" the stored stream is 1 path bit + (bits-1) member
        # bits per weight = bits/weight exactly (round-trip proven in tests),
        # even though ``codes`` caches the doubled-codebook level index.
        code_bits = self.bits * n_weights
        scale_bits = 16 * n
        codebook_bits = 16 * self.codebook.numel()
        sketch_sign_bits = self.sketch_k * n
        sketch_norm_bits = 16 * n if self.sketch_k > 0 else 0
        equil_bits = 16 * d if self.inv_col_scale is not None else 0
        total = (
            code_bits
            + scale_bits
            + codebook_bits
            + sketch_sign_bits
            + sketch_norm_bits
            + equil_bits
        )
        return {
            "bits_per_weight_codes": self.bits,
            "bits_per_weight_scales": scale_bits / n_weights,
            "bits_per_weight_sketch": (sketch_sign_bits + sketch_norm_bits) / n_weights,
            "bits_per_weight_equil": equil_bits / n_weights,
            "bits_per_weight_total": total / n_weights,
            "total_bytes": math.ceil(total / 8),
            "fp16_bytes": 2 * n_weights,
            "compression_vs_fp16": (16 * n_weights) / total,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, method={self.method!r}, sketch_k={self.sketch_k}, "
            f"rounding={self.rounding!r}, bias={self.bias is not None}, seed={self.seed}"
        )
