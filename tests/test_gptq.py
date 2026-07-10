import pytest
import torch

from turbopress.gptq import ldlq_quantize_rows, rotated_hessian
from turbopress.hadamard import RandomizedOrthogonal
from turbopress.quantizer import quantize_rows


def _hessian_loss(w, what, h):
    e = w - what
    return float(torch.einsum("nd,de,ne->", e, h, e) / w.numel())


def test_rotated_hessian_matches_dense():
    torch.manual_seed(0)
    d = 96  # non-power-of-2 factor still handled by the fast transform path
    x = torch.randn(2048, d)
    h = (x.T @ x) / x.shape[0]
    rot = RandomizedOrthogonal(d, 7)
    hz = rotated_hessian(h, rot)
    # Reference: rotate the samples, then form the Gram matrix.
    z = rot(x)
    hz_ref = (z.T @ z) / x.shape[0]
    assert torch.allclose(hz, hz_ref, atol=1e-4, rtol=1e-4)


def test_rotated_hessian_with_equilibration():
    torch.manual_seed(1)
    d = 64
    x = torch.randn(1024, d)
    h = (x.T @ x) / x.shape[0]
    rot = RandomizedOrthogonal(d, 3)
    inv_s = torch.rand(d) + 0.5
    hz = rotated_hessian(h, rot, inv_col_scale=inv_s)
    z = rot(x * inv_s)  # D^-1 x then rotate
    hz_ref = (z.T @ z) / x.shape[0]
    assert torch.allclose(hz, hz_ref, atol=1e-4, rtol=1e-4)


def test_ldlq_reduces_hessian_loss():
    torch.manual_seed(0)
    d, n, N = 128, 64, 4096
    a = torch.randn(d, d) * 0.3 + torch.eye(d)
    x = torch.randn(N, d) @ a.T
    hz = (x.T @ x) / N
    w = torch.randn(n, d)
    for bits in (2, 3):
        near = quantize_rows(w, bits=bits).dequantize()
        ldlq = ldlq_quantize_rows(w, hz, bits=bits).dequantize()
        assert _hessian_loss(w, ldlq, hz) < _hessian_loss(w, near, hz)


def test_ldlq_reduces_to_nearest_on_identity_hessian():
    # With H = I the inverse-Hessian Cholesky is diagonal, so no error is fed
    # between channels and LDLQ must coincide with independent nearest rounding.
    torch.manual_seed(2)
    d, n = 64, 32
    w = torch.randn(n, d)
    hz = torch.eye(d)
    q_ldlq = ldlq_quantize_rows(w, hz, bits=3, percdamp=0.0)
    q_near = quantize_rows(w, bits=3, scale_iters=0)
    assert torch.equal(q_ldlq.codes, q_near.codes)


def test_ldlq_storage_and_code_range():
    torch.manual_seed(3)
    d, n = 96, 48
    w = torch.randn(n, d)
    hz = torch.eye(d)
    q = ldlq_quantize_rows(w, hz, bits=2)
    assert q.bits == 2
    assert q.codes.shape == (n, d)
    assert int(q.codes.max()) < 2**2
    assert q.codebook.numel() == 2**2


def test_ldlq_zero_row_stays_zero():
    torch.manual_seed(4)
    d, n = 64, 8
    w = torch.randn(n, d)
    w[3] = 0.0
    q = ldlq_quantize_rows(w, torch.eye(d), bits=3)
    assert float(q.scales[3]) == 0.0
    assert torch.all(q.dequantize()[3] == 0.0)


def test_ldlq_matches_across_devices():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    torch.manual_seed(5)
    d, n, N = 128, 64, 2048
    w = torch.randn(n, d)
    x = torch.randn(N, d)
    hz = (x.T @ x) / N
    q_cpu = ldlq_quantize_rows(w, hz, bits=3)
    q_gpu = ldlq_quantize_rows(w.cuda(), hz.cuda(), bits=3)
    assert torch.equal(q_cpu.codes, q_gpu.codes.cpu())
