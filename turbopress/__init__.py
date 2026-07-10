"""turbopress: bias-free low-bit weight quantization for linear layers.

Implements "stage 4" of the TurboPress proposal: an MSE-optimal scalar
quantizer applied after a seeded randomized orthogonal transform, plus a
1-bit Quantized-JL (QJL) sketch of the quantization residual that makes
every layer output an *unbiased* estimate of the full-precision output
(unbiased over the sketch randomness).

Reference: TurboQuant (arXiv:2504.19874) for the rotation + optimal scalar
quantizer + QJL residual construction; this package applies it to weight
matrices and measures the bias/variance trade-off it buys.
"""

from turbopress.allocate import allocate_bits, build_mixed_model, measure_sensitivity
from turbopress.codebooks import lloyd_max_gaussian
from turbopress.gptq import ldlq_quantize_rows, rotated_hessian
from turbopress.hadamard import RandomizedOrthogonal, fwht
from turbopress.linear import QJLCorrectedLinear
from turbopress.qjl import QJLSketch, build_qjl_sketch, estimate_inner_products
from turbopress.quantizer import RowQuantized, quantize_rows
from turbopress.trellis import TCQQuantized, Trellis, tcq_quantize_rows

__version__ = "0.3.0"

__all__ = [
    "QJLCorrectedLinear",
    "QJLSketch",
    "RandomizedOrthogonal",
    "RowQuantized",
    "TCQQuantized",
    "Trellis",
    "allocate_bits",
    "build_mixed_model",
    "build_qjl_sketch",
    "estimate_inner_products",
    "fwht",
    "ldlq_quantize_rows",
    "lloyd_max_gaussian",
    "measure_sensitivity",
    "quantize_rows",
    "rotated_hessian",
    "tcq_quantize_rows",
    "__version__",
]
