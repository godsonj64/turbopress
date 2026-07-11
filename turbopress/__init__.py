"""turbopress: bias-free low-bit weight quantization for linear layers.

Implements "stage 4" of the TurboPress proposal: an MSE-optimal scalar
quantizer applied after a seeded randomized orthogonal transform, plus a
1-bit Quantized-JL (QJL) sketch of the quantization residual that makes
every layer output an *unbiased* estimate of the full-precision output
(unbiased over the sketch randomness).

Reference: TurboQuant (arXiv:2504.19874) for the rotation + optimal scalar
quantizer + QJL residual construction; this package applies it to weight
matrices and measures the bias/variance trade-off it buys.

Top-level names are imported lazily (PEP 562) so lightweight consumers - e.g.
``from turbopress.certificate import verify_certificate`` - don't pull in torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "0.5.1"

# name -> submodule that defines it (imported on first access).
_LAZY = {
    "allocate_bits": "turbopress.allocate",
    "build_mixed_model": "turbopress.allocate",
    "measure_sensitivity": "turbopress.allocate",
    "lloyd_max_gaussian": "turbopress.codebooks",
    "ldlq_quantize_rows": "turbopress.gptq",
    "ldlq_tcq_quantize_rows": "turbopress.gptq",
    "rotated_hessian": "turbopress.gptq",
    "RandomizedOrthogonal": "turbopress.hadamard",
    "fwht": "turbopress.hadamard",
    "QJLCorrectedLinear": "turbopress.linear",
    "QJLSketch": "turbopress.qjl",
    "build_qjl_sketch": "turbopress.qjl",
    "estimate_inner_products": "turbopress.qjl",
    "RowQuantized": "turbopress.quantizer",
    "quantize_rows": "turbopress.quantizer",
    "PackedTCQLinear": "turbopress.runtime",
    "pack_linear": "turbopress.runtime",
    "pack_model": "turbopress.runtime",
    "load_packed_model": "turbopress.runtime",
    "window_decode_levels": "turbopress.runtime",
    "TCQQuantized": "turbopress.trellis",
    "Trellis": "turbopress.trellis",
    "tcq_quantize_rows": "turbopress.trellis",
}

__all__ = [*sorted(_LAZY), "__version__"]


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'turbopress' has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


if TYPE_CHECKING:  # help type checkers / IDEs resolve the lazy names
    from turbopress.allocate import allocate_bits, build_mixed_model, measure_sensitivity
    from turbopress.codebooks import lloyd_max_gaussian
    from turbopress.gptq import (
        ldlq_quantize_rows,
        ldlq_tcq_quantize_rows,
        rotated_hessian,
    )
    from turbopress.hadamard import RandomizedOrthogonal, fwht
    from turbopress.linear import QJLCorrectedLinear
    from turbopress.qjl import QJLSketch, build_qjl_sketch, estimate_inner_products
    from turbopress.quantizer import RowQuantized, quantize_rows
    from turbopress.runtime import (
        PackedTCQLinear,
        load_packed_model,
        pack_linear,
        pack_model,
        window_decode_levels,
    )
    from turbopress.trellis import TCQQuantized, Trellis, tcq_quantize_rows
