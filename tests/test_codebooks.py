import math

import pytest
import torch

from turbopress.codebooks import lloyd_max_gaussian


def test_one_bit_closed_form():
    # Optimal 1-bit quantizer of N(0,1): levels +/- sqrt(2/pi), MSE 1 - 2/pi.
    levels, mse = lloyd_max_gaussian(1)
    expected = math.sqrt(2.0 / math.pi)
    torch.testing.assert_close(
        levels, torch.tensor([-expected, expected], dtype=torch.float64)
    )
    assert abs(mse - (1.0 - 2.0 / math.pi)) < 1e-10


def test_two_bit_known_values():
    # Classical Lloyd-Max values for 4 levels: +/-0.4528, +/-1.510, MSE ~0.1175.
    levels, mse = lloyd_max_gaussian(2)
    assert abs(levels[2].item() - 0.4528) < 2e-3
    assert abs(levels[3].item() - 1.510) < 2e-3
    assert abs(mse - 0.1175) < 1e-3


@pytest.mark.parametrize("bits", [1, 2, 3, 4, 6, 8])
def test_levels_sorted_and_symmetric(bits):
    levels, mse = lloyd_max_gaussian(bits)
    assert levels.shape == (1 << bits,)
    assert torch.all(levels[1:] > levels[:-1])
    torch.testing.assert_close(levels, -levels.flip(0), rtol=1e-8, atol=1e-8)
    assert 0.0 < mse < 1.0


def test_mse_decreases_with_bits():
    mses = [lloyd_max_gaussian(b)[1] for b in range(1, 9)]
    assert all(a > b for a, b in zip(mses, mses[1:]))
    # High-rate regime: each extra bit reduces MSE ~4x (within slack).
    assert mses[6] / mses[7] > 3.0


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_predicted_mse_matches_empirical(bits):
    levels, predicted = lloyd_max_gaussian(bits)
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(1_000_000, generator=gen, dtype=torch.float64)
    thresholds = (levels[:-1] + levels[1:]) / 2
    q = levels[torch.bucketize(z, thresholds)]
    empirical = float((z - q).pow(2).mean())
    assert abs(empirical - predicted) / predicted < 0.01


def test_cache_returns_defensive_copy():
    a, _ = lloyd_max_gaussian(3)
    a += 100.0
    b, _ = lloyd_max_gaussian(3)
    assert b.abs().max() < 10.0


@pytest.mark.parametrize("bits", [0, 9, 2.5])
def test_rejects_bad_bits(bits):
    with pytest.raises(ValueError):
        lloyd_max_gaussian(bits)
