import math

import pytest
import torch

from turbopress.hadamard import RandomizedOrthogonal, fwht, largest_pow2_divisor


def test_largest_pow2_divisor():
    assert largest_pow2_divisor(1) == 1
    assert largest_pow2_divisor(96) == 32
    assert largest_pow2_divisor(1024) == 1024
    assert largest_pow2_divisor(7) == 1
    with pytest.raises(ValueError):
        largest_pow2_divisor(0)


def test_fwht_matches_dense_hadamard():
    d = 8
    # Dense orthonormal Hadamard via Sylvester construction.
    h = torch.tensor([[1.0]])
    while h.shape[0] < d:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    h = h / math.sqrt(d)
    x = torch.randn(5, d)
    torch.testing.assert_close(fwht(x), x @ h.T, rtol=1e-5, atol=1e-5)


def test_fwht_is_orthonormal_involution():
    x = torch.randn(3, 4, 64)
    y = fwht(x)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fwht(y), x, rtol=1e-5, atol=1e-5)


def test_fwht_rejects_non_power_of_two():
    with pytest.raises(ValueError):
        fwht(torch.randn(2, 6))


@pytest.mark.parametrize("dim", [64, 96, 7, 1])
def test_randomized_orthogonal_roundtrip_and_isometry(dim):
    rot = RandomizedOrthogonal(dim, seed=123)
    x = torch.randn(10, dim)
    y = rot(x)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(rot.inverse(y), x, rtol=1e-4, atol=1e-5)


def test_rotation_preserves_matmul():
    # (W R^T)(R x) == W x: rotating weight rows and inputs is lossless.
    dim, out, batch = 96, 32, 8
    rot = RandomizedOrthogonal(dim, seed=7)
    w = torch.randn(out, dim)
    x = torch.randn(batch, dim)
    y_ref = x @ w.T
    y_rot = rot(x) @ rot(w).T
    torch.testing.assert_close(y_rot, y_ref, rtol=1e-4, atol=1e-4)


def test_seed_determinism_and_distinctness():
    x = torch.randn(4, 64)
    a = RandomizedOrthogonal(64, seed=1)(x)
    b = RandomizedOrthogonal(64, seed=1)(x)
    c = RandomizedOrthogonal(64, seed=2)(x)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


def test_rejects_wrong_dim_and_dtype():
    rot = RandomizedOrthogonal(16, seed=0)
    with pytest.raises(ValueError):
        rot(torch.randn(3, 8))
    with pytest.raises(TypeError):
        rot(torch.ones(3, 16, dtype=torch.int64))
