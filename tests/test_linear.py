import pytest
import torch
from torch import nn

from turbopress.linear import QJLCorrectedLinear


def _make_linear(in_f=128, out_f=64, bias=True, seed=0):
    gen = torch.Generator().manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=bias)
    linear.weight.data = torch.randn(out_f, in_f, generator=gen) / in_f**0.5
    if bias:
        linear.bias.data = torch.randn(out_f, generator=gen) * 0.1
    return linear


def test_high_bit_is_near_exact():
    linear = _make_linear()
    q = QJLCorrectedLinear.from_linear(linear, bits=8, sketch_k=0, seed=0)
    x = torch.randn(32, 128)
    with torch.no_grad():
        y_ref = linear(x)
        y_q = q(x)
    assert float((y_q - y_ref).norm() / y_ref.norm()) < 0.03


def test_shapes_dtype_and_batch_dims():
    linear = _make_linear()
    q = QJLCorrectedLinear.from_linear(linear, bits=4, sketch_k=16, seed=0)
    x = torch.randn(2, 5, 128, dtype=torch.float16)
    y = q(x)
    assert y.shape == (2, 5, 64)
    assert y.dtype == torch.float16


def test_bias_is_preserved():
    linear = _make_linear(bias=True)
    q = QJLCorrectedLinear.from_linear(linear, bits=8, sketch_k=0, seed=0)
    x = torch.zeros(1, 128)
    torch.testing.assert_close(q(x)[0], linear.bias.data, rtol=1e-3, atol=1e-4)


def test_determinism():
    linear = _make_linear()
    x = torch.randn(8, 128)
    a = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=32, seed=5)(x)
    b = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=32, seed=5)(x)
    torch.testing.assert_close(a, b)
    c = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=32, seed=6)(x)
    assert not torch.allclose(a, c)


def test_non_power_of_two_in_features():
    linear = _make_linear(in_f=96, out_f=48)
    q = QJLCorrectedLinear.from_linear(linear, bits=8, sketch_k=0, seed=0)
    x = torch.randn(4, 96)
    with torch.no_grad():
        rel = float((q(x) - linear(x)).norm() / linear(x).norm())
    assert rel < 0.03


def test_qjl_correction_removes_bias():
    """Averaged over sketch seeds, corrected outputs converge to the exact
    ones, while the uncorrected 2-bit output has a fixed deterministic error."""
    linear = _make_linear(in_f=256, out_f=128, bias=False, seed=1)
    x = torch.randn(64, 256)
    with torch.no_grad():
        y_ref = linear(x)

        base = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=0, seed=0)(x)
        err_uncorrected = float((base - y_ref).norm() / y_ref.norm())

        n_seeds = 48
        mean = torch.zeros_like(y_ref)
        for s in range(n_seeds):
            q = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=256, seed=1000 + s)
            mean += q(x)
        mean /= n_seeds
    # Different seeds change the rotation too, so this averages over the full
    # method randomness; unbiasedness still holds (each rotation's sketch is
    # unbiased conditionally on the rotation).
    rel_bias = float((mean - y_ref).norm() / y_ref.norm())
    assert rel_bias < 0.4 * err_uncorrected


def test_storage_report_accounting():
    linear = _make_linear(in_f=256, out_f=128, bias=False)
    q = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=256, seed=0)
    report = q.storage_report()
    assert report["bits_per_weight_codes"] == 2
    # scales: 16 bits / 256 cols; sketch: (256 signs + 16 norm) / 256 cols.
    assert abs(report["bits_per_weight_scales"] - 16 / 256) < 1e-9
    assert abs(report["bits_per_weight_sketch"] - (256 + 16) / 256) < 1e-9
    expected_total = 2 + 16 / 256 + (256 + 16) / 256 + 16 * 4 / (128 * 256)
    assert abs(report["bits_per_weight_total"] - expected_total) < 1e-9
    assert report["compression_vs_fp16"] == pytest.approx(
        16 / report["bits_per_weight_total"]
    )


def test_input_validation():
    linear = _make_linear()
    q = QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=0, seed=0)
    with pytest.raises(ValueError):
        q(torch.randn(3, 64))
    with pytest.raises(TypeError):
        QJLCorrectedLinear.from_linear(nn.Conv1d(4, 4, 1), bits=2)
    with pytest.raises(ValueError):
        QJLCorrectedLinear.from_linear(linear, bits=2, sketch_k=-1)
