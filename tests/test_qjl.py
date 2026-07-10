import math

import pytest
import torch

from turbopress.qjl import build_qjl_sketch, estimate_inner_products


def test_shapes_and_seeding():
    r = torch.randn(6, 128)
    sk = build_qjl_sketch(r, k=16, seed=0)
    assert sk.proj.shape == (16, 128)
    assert sk.signs.shape == (6, 16)
    assert sk.norms.shape == (6,)
    assert set(sk.signs.unique().tolist()) <= {-1, 1}
    sk2 = build_qjl_sketch(r, k=16, seed=0)
    torch.testing.assert_close(sk.proj, sk2.proj)
    assert torch.equal(sk.signs, sk2.signs)

    x = torch.randn(3, 5, 128)
    est = estimate_inner_products(sk, x)
    assert est.shape == (3, 5, 6)


def test_estimator_is_unbiased():
    """Average over many independent sketches converges to <r, x>."""
    d, k, n_seeds = 128, 8, 1200
    gen = torch.Generator().manual_seed(0)
    r = torch.randn(3, d, generator=gen)
    x = torch.randn(10, d, generator=gen)
    truth = x @ r.T  # [10, 3]

    estimates = torch.stack(
        [estimate_inner_products(build_qjl_sketch(r, k, seed=s), x) for s in range(n_seeds)]
    )
    mean = estimates.mean(dim=0)
    stderr = estimates.std(dim=0) / math.sqrt(n_seeds)
    # Every entry within 5 standard errors of the truth.
    assert torch.all((mean - truth).abs() < 5 * stderr + 1e-6)


def test_variance_scales_as_one_over_k():
    d, n_seeds = 128, 300
    gen = torch.Generator().manual_seed(1)
    r = torch.randn(1, d, generator=gen)
    x = torch.randn(4, d, generator=gen)

    def emp_var(k):
        est = torch.stack(
            [
                estimate_inner_products(build_qjl_sketch(r, k, seed=1000 + s), x)
                for s in range(n_seeds)
            ]
        )
        return float(est.var(dim=0).mean())

    ratio = emp_var(4) / emp_var(64)
    assert 8.0 < ratio < 32.0  # ideal 16, generous CI


def test_variance_bound_holds():
    # Var[est] <= (pi/2) ||r||^2 ||x||^2 / k  (per theory in qjl.py).
    d, k, n_seeds = 64, 8, 400
    gen = torch.Generator().manual_seed(2)
    r = torch.randn(1, d, generator=gen)
    x = torch.randn(1, d, generator=gen)
    est = torch.stack(
        [
            estimate_inner_products(build_qjl_sketch(r, k, seed=2000 + s), x)
            for s in range(n_seeds)
        ]
    )
    bound = (math.pi / 2) * float(r.norm() ** 2) * float(x.norm() ** 2) / k
    assert float(est.var(dim=0)) < 1.3 * bound


def test_zero_residual_contributes_exactly_zero():
    r = torch.zeros(2, 32)
    r[1] = torch.randn(32)
    sk = build_qjl_sketch(r, k=8, seed=0)
    est = estimate_inner_products(sk, torch.randn(5, 32))
    assert torch.all(est[:, 0] == 0.0)
    assert torch.any(est[:, 1] != 0.0)


def test_input_validation():
    with pytest.raises(ValueError):
        build_qjl_sketch(torch.randn(4, 8), k=0, seed=0)
    with pytest.raises(ValueError):
        build_qjl_sketch(torch.randn(4, 8, 2), k=4, seed=0)
    sk = build_qjl_sketch(torch.randn(4, 8), k=4, seed=0)
    with pytest.raises(ValueError):
        estimate_inner_products(sk, torch.randn(3, 16))
