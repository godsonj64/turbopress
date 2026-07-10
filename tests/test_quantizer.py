import math

import pytest
import torch

from turbopress.codebooks import lloyd_max_gaussian
from turbopress.quantizer import _stochastic_codes, quantize_rows


def _rel_err(w, quantized):
    return float((quantized.dequantize() - w).norm() / w.norm())


def test_reconstruction_improves_with_bits():
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(64, 512, generator=gen)
    errs = [_rel_err(w, quantize_rows(w, bits=b)) for b in (1, 2, 3, 4, 8)]
    assert all(a > b for a, b in zip(errs, errs[1:]))
    # Gaussian rows at 8 bits: rel error ~ sqrt(3.5e-4) ~ 1.9%.
    assert errs[-1] < 0.03
    # 2-bit Lloyd-Max on Gaussian data: rel error ~ sqrt(0.1175) ~ 34%.
    assert 0.25 < errs[1] < 0.45


def test_scale_refinement_does_not_hurt():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(32, 256, generator=gen)
    base = _rel_err(w, quantize_rows(w, bits=2, scale_iters=0))
    refined = _rel_err(w, quantize_rows(w, bits=2, scale_iters=3))
    assert refined <= base + 1e-6


def test_zero_row_dequantizes_to_zero():
    w = torch.randn(4, 64)
    w[2] = 0.0
    q = quantize_rows(w, bits=2)
    assert q.scales[2] == 0.0
    assert torch.all(q.dequantize()[2] == 0.0)
    assert _rel_err(w, q) < 0.5


def test_codes_fit_bit_width():
    w = torch.randn(16, 128)
    for bits in (1, 2, 3, 4):
        q = quantize_rows(w, bits=bits)
        assert q.codes.dtype == torch.uint8
        assert int(q.codes.max()) < (1 << bits)


def test_stochastic_codes_are_unbiased_in_range():
    # For z strictly inside the codebook range, E[codebook[code]] == z.
    levels, _ = lloyd_max_gaussian(2)
    codebook = levels.to(torch.float32)
    gen = torch.Generator().manual_seed(0)
    z = torch.empty(256).uniform_(
        float(codebook[0]) + 0.05, float(codebook[-1]) - 0.05, generator=gen
    )
    n_draws = 4000
    acc = torch.zeros_like(z)
    for _ in range(n_draws):
        acc += codebook[_stochastic_codes(z, codebook, gen)]
    mean = acc / n_draws
    # Per-draw std is bounded by half the largest codeword gap (~1.06/2);
    # allow 5 standard errors.
    max_gap = float((codebook[1:] - codebook[:-1]).max())
    tol = 5 * (max_gap / 2) / math.sqrt(n_draws)
    assert float((mean - z).abs().max()) < tol


def test_stochastic_has_higher_variance_than_nearest():
    gen = torch.Generator().manual_seed(2)
    w = torch.randn(64, 512, generator=gen)
    nearest = _rel_err(w, quantize_rows(w, bits=2))
    stochastic = _rel_err(
        w, quantize_rows(w, bits=2, rounding="stochastic", generator=gen)
    )
    assert stochastic > nearest


def test_input_validation():
    with pytest.raises(ValueError):
        quantize_rows(torch.randn(2, 3, 4), bits=2)
    with pytest.raises(TypeError):
        quantize_rows(torch.ones(2, 4, dtype=torch.int32), bits=2)
    with pytest.raises(ValueError):
        quantize_rows(torch.randn(2, 4), bits=2, rounding="banker")
    with pytest.raises(ValueError):
        quantize_rows(torch.randn(2, 4), bits=2, rounding="stochastic")
