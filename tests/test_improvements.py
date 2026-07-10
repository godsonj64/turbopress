"""Tests for the two beyond-TurboQuant methods: trellis-optimized codebooks
and activation-aware equilibration."""

import pytest
import torch
from torch import nn

from turbopress.codebooks import lloyd_max_gaussian
from turbopress.linear import QJLCorrectedLinear
from turbopress.trellis import (
    Trellis,
    decode_levels,
    tcq_optimized_codebook,
    tcq_quantize_rows,
)


def _tcq_mse(z, **kwargs):
    q = tcq_quantize_rows(z, scale_iters=0, **kwargs)
    recon = q.codebook[q.level_codes.long()] * q.scales[:, None]
    return float((recon - z).pow(2).mean())


def test_optimized_codebook_beats_doubled_lloyd_max():
    # Held-out Gaussian sample, different seed than the design sample.
    gen = torch.Generator().manual_seed(99)
    z = torch.randn(4, 8192, generator=gen)
    mse_lloyd = _tcq_mse(z, bits=2, n_states=8, codebook_mode="lloyd")
    mse_opt = _tcq_mse(z, bits=2, n_states=8, codebook_mode="optimized")
    assert mse_opt < mse_lloyd
    # And still beats the scalar-optimal quantizer by a clear margin.
    _, scalar = lloyd_max_gaussian(2)
    assert mse_opt < 0.90 * scalar


def test_optimized_codebook_is_cached_and_deterministic():
    a = tcq_optimized_codebook(2, 8)
    b = tcq_optimized_codebook(2, 8)
    torch.testing.assert_close(a, b)
    a += 100.0  # defensive clone: mutating must not poison the cache
    c = tcq_optimized_codebook(2, 8)
    assert float(c.abs().max()) < 10.0
    # Subset slices stay sorted (required by the per-subset nearest search).
    for j in range(4):
        s = c[j::4]
        assert torch.all(s[1:] >= s[:-1])


def test_roundtrip_still_holds_with_optimized_codebook():
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(4, 64, generator=gen)
    q = tcq_quantize_rows(w, bits=2, n_states=8, codebook_mode="optimized")
    rebuilt = decode_levels(q.path_bits, q.member_codes, Trellis(8))
    assert torch.equal(rebuilt, q.level_codes)


def test_tcq_rejects_bad_codebook_mode():
    with pytest.raises(ValueError):
        tcq_quantize_rows(torch.randn(2, 8), bits=2, codebook_mode="learned")


def _make_aniso_layer_and_input(in_f=256, out_f=128, seed=0):
    """Linear layer + activations whose channel scales span 3 orders of magnitude."""
    gen = torch.Generator().manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=False)
    linear.weight.data = torch.randn(out_f, in_f, generator=gen) / in_f**0.5
    channel_std = torch.logspace(-1.5, 1.5, in_f)
    x = torch.randn(512, in_f, generator=gen) * channel_std
    return linear, x, channel_std


def test_equilibration_reduces_output_error():
    linear, x, channel_std = _make_aniso_layer_and_input()
    col_scale = x.pow(2).mean(dim=0).sqrt()
    with torch.no_grad():
        y_ref = linear(x)
        y_plain = QJLCorrectedLinear.from_linear(linear, bits=2, seed=0)(x)
        y_eq = QJLCorrectedLinear.from_linear(
            linear, bits=2, seed=0, col_scale=col_scale
        )(x)
    err_plain = float((y_plain - y_ref).norm() / y_ref.norm())
    err_eq = float((y_eq - y_ref).norm() / y_ref.norm())
    assert err_eq < 0.8 * err_plain


def test_equilibration_is_exact_at_high_bits():
    linear, x, _ = _make_aniso_layer_and_input(seed=1)
    col_scale = x.pow(2).mean(dim=0).sqrt()
    q = QJLCorrectedLinear.from_linear(linear, bits=8, seed=0, col_scale=col_scale)
    with torch.no_grad():
        y_ref = linear(x)
        y_q = q(x)
    assert float((y_q - y_ref).norm() / y_ref.norm()) < 0.03


def test_equilibration_storage_and_validation():
    linear, x, _ = _make_aniso_layer_and_input()
    col_scale = x.pow(2).mean(dim=0).sqrt()
    q = QJLCorrectedLinear.from_linear(linear, bits=2, col_scale=col_scale)
    report = q.storage_report()
    assert report["bits_per_weight_equil"] == pytest.approx(16 * 256 / (128 * 256))
    plain = QJLCorrectedLinear.from_linear(linear, bits=2).storage_report()
    assert report["bits_per_weight_total"] > plain["bits_per_weight_total"]

    with pytest.raises(ValueError):
        QJLCorrectedLinear.from_linear(linear, bits=2, col_scale=torch.ones(5))
    with pytest.raises(ValueError):
        QJLCorrectedLinear.from_linear(linear, bits=2, col_scale=torch.zeros(256))


def test_equilibration_composes_with_tcq():
    linear, x, _ = _make_aniso_layer_and_input(seed=2)
    col_scale = x.pow(2).mean(dim=0).sqrt()
    with torch.no_grad():
        y_ref = linear(x)
        y_tcq = QJLCorrectedLinear.from_linear(linear, bits=2, method="tcq", seed=0)(x)
        y_both = QJLCorrectedLinear.from_linear(
            linear, bits=2, method="tcq", seed=0, col_scale=col_scale
        )(x)
    err_tcq = float((y_tcq - y_ref).norm() / y_ref.norm())
    err_both = float((y_both - y_ref).norm() / y_ref.norm())
    assert err_both < 0.8 * err_tcq


def test_quarter_fold_beats_awq_sqrt_fold():
    # Proposition 1: under the rotated quantizer the quarter-power fold should
    # beat the AWQ/SmoothQuant square-root fold when weight column norms vary.
    torch.manual_seed(0)
    d, n, N = 256, 128, 4096
    # Column energies that vary independently of activation energies, so the
    # weight-norm term c_j matters (the regime where the exponent differs).
    col_gain = torch.linspace(0.2, 3.0, d)
    linear = nn.Linear(d, n, bias=False)
    with torch.no_grad():
        linear.weight.mul_(col_gain[None, :])
    x = torch.randn(N, d) * torch.linspace(3.0, 0.2, d)[None, :]
    col_scale = x.pow(2).mean(dim=0).sqrt()
    with torch.no_grad():
        y_ref = linear(x)
        y_awq = QJLCorrectedLinear.from_linear(
            linear, bits=2, seed=0, col_scale=col_scale, equil_mode="awq")(x)
        y_qtr = QJLCorrectedLinear.from_linear(
            linear, bits=2, seed=0, col_scale=col_scale, equil_mode="quarter")(x)
    err_awq = float((y_awq - y_ref).norm() / y_ref.norm())
    err_qtr = float((y_qtr - y_ref).norm() / y_ref.norm())
    assert err_qtr < err_awq


def test_invalid_equil_mode_rejected():
    linear = nn.Linear(64, 32, bias=False)
    with pytest.raises(ValueError):
        QJLCorrectedLinear.from_linear(
            linear, bits=2, col_scale=torch.ones(64), equil_mode="nope")
