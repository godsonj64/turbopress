"""Block-LDLQ error feedback over the trellis (ldlq_tcq_quantize_rows).

The three invariants that make the combination sound:

1. **Storage format unchanged**: chaining Viterbi state across column blocks
   must still produce one continuous trellis path -- the concatenated
   path/member bits decode with the standard sequential walk from state 0.
2. **Reduction**: with a diagonal Hessian (no cross-channel feedback) and a
   single block, the method IS plain TCQ with fixed row-RMS scales.
3. **The point of it**: on a correlated Hessian, the Hessian-weighted layer
   loss tr(E H E^T) must drop vs feedback-free TCQ at the same rate.
"""

import pytest
import torch

from turbopress.gptq import ldlq_tcq_quantize_rows
from turbopress.trellis import Trellis, decode_levels, tcq_quantize_rows


def _correlated_hessian(d: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(d, 2 * d, generator=gen)
    h = a @ a.T / (2 * d)
    # Strengthen off-diagonal structure so feedback has something to exploit.
    v = torch.randn(d, 1, generator=gen)
    return h + 0.5 * (v @ v.T)


def test_chained_blocks_round_trip_exactly():
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(6, 96, generator=gen)
    h = _correlated_hessian(96)
    for n_states in (8, 64):
        q = ldlq_tcq_quantize_rows(w, h, bits=3, n_states=n_states, block_size=16)
        rebuilt = decode_levels(q.path_bits, q.member_codes, Trellis(n_states))
        assert torch.equal(rebuilt, q.level_codes)


def test_diagonal_hessian_single_block_reduces_to_plain_tcq():
    gen = torch.Generator().manual_seed(1)
    w = torch.randn(5, 64, generator=gen)
    for iters in (0, 1):
        q_ef = ldlq_tcq_quantize_rows(
            w, torch.eye(64), bits=2, n_states=8, block_size=64, scale_iters=iters
        )
        q_plain = tcq_quantize_rows(w, bits=2, n_states=8, scale_iters=iters)
        assert torch.equal(q_ef.level_codes, q_plain.level_codes)
        assert torch.equal(q_ef.path_bits, q_plain.path_bits)
        assert torch.allclose(q_ef.scales, q_plain.scales)


def test_diagonal_hessian_multi_block_stays_near_plain_tcq():
    # Chained-state greedy blocks are slightly suboptimal vs one global
    # Viterbi pass. Measured on this seed the boundary loss halves with the
    # block size: +14.4% MSE at block_size=16, +6.5% at 32, +2.1% at 64,
    # 0% at 128 -- which is why the production default is 128. Guard the
    # realistic regime.
    gen = torch.Generator().manual_seed(2)
    w = torch.randn(8, 128, generator=gen)
    q_blk = ldlq_tcq_quantize_rows(w, torch.eye(128), bits=2, n_states=8,
                                   block_size=64, scale_iters=0)
    q_one = tcq_quantize_rows(w, bits=2, n_states=8, scale_iters=0)
    mse_blk = float((w - q_blk.dequantize()).pow(2).mean())
    mse_one = float((w - q_one.dequantize()).pow(2).mean())
    assert mse_blk <= mse_one * 1.05


def test_feedback_reduces_hessian_weighted_loss():
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(8, 128, generator=gen)
    h = _correlated_hessian(128, seed=3)
    q_ef = ldlq_tcq_quantize_rows(w, h, bits=2, n_states=8, block_size=32)
    q_plain = tcq_quantize_rows(w, bits=2, n_states=8)

    def hloss(q):
        e = w - q.dequantize()
        return float(torch.einsum("ni,ij,nj->", e, h, e))

    assert hloss(q_ef) < hloss(q_plain)


def test_zero_rows_and_custom_codebook():
    gen = torch.Generator().manual_seed(4)
    w = torch.randn(4, 32, generator=gen)
    w[2] = 0.0
    cb = tcq_quantize_rows(torch.randn(2, 32, generator=gen), bits=2, n_states=8).codebook
    cb16 = cb.to(torch.float16).float()  # stored-precision codebook override
    q = ldlq_tcq_quantize_rows(
        w, _correlated_hessian(32, seed=4), bits=2, n_states=8,
        block_size=8, codebook=cb16,
    )
    assert torch.equal(q.codebook, cb16)
    assert float(q.scales[2]) == 0.0
    assert torch.equal(q.dequantize()[2], torch.zeros(32))


def test_pipeline_quantize_matrix_ef_payload_decodes_exactly():
    # The artifact contract: unpacking the payload and reconstructing must
    # equal the returned w_hat bit-for-bit, with and without error feedback.
    from turbopress import pipeline as P

    gen = torch.Generator().manual_seed(6)
    w = torch.randn(16, 64, generator=gen)
    act = torch.rand(64, generator=gen) + 0.5
    h = _correlated_hessian(64, seed=6)
    for hessian in (None, h):
        payload, w_hat = P.quantize_matrix(
            w, act, bits=3, n_states=16, seed=0, hessian=hessian, ef_block=16
        )
        n, d = payload["n"], payload["d"]
        pb = torch.from_numpy(P.unpack_bits(payload["path_bits"], n * d, 1)).reshape(n, d)
        mb = torch.from_numpy(P.unpack_bits(payload["member_bits"], n * d, 2)).reshape(n, d)
        lv = P.decode_levels(pb, mb, P.Trellis(16))
        w_chk = P.rotate_inv(
            payload["scales"].float()[:, None] * payload["codebook"].float()[lv.long()],
            torch.from_numpy(P.unpack_bits(payload["signs"], d, 1)).float() * 2 - 1,
            payload["block"],
        ) / payload["equil"].float()[None, :]
        assert torch.allclose(w_chk, w_hat, atol=1e-6)
    # Error feedback must actually change the codes on a correlated Hessian.
    p_plain, _ = P.quantize_matrix(w, act, bits=3, n_states=16, seed=0)
    p_ef, _ = P.quantize_matrix(w, act, bits=3, n_states=16, seed=0,
                                hessian=h, ef_block=16)
    assert not torch.equal(p_plain["path_bits"], p_ef["path_bits"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_fwht_scale_matches_torch_path():
    triton_kernel = pytest.importorskip("turbopress.triton_kernel")
    if not triton_kernel.HAS_TRITON:
        pytest.skip("triton not available")
    from turbopress.runtime import _block_fwht

    gen = torch.Generator().manual_seed(7)
    # (d, block): all block-FWHT regimes -- small single-dot (64, 128) and
    # two-stage Kronecker (512, 1024) -- at batch 1 and a prefill-like batch.
    for d, block in ((576, 64), (1536, 512), (1024, 1024), (3072, 1024), (128, 128)):
        rs = (torch.randn(d, generator=gen).sign() *
              (torch.rand(d, generator=gen) + 0.5)).to(torch.float16).cuda()
        for b in (1, 5):
            x = torch.randn(b, d, generator=gen).to(torch.float16).cuda()
            ref = _block_fwht(x * rs, block)
            got = triton_kernel.fwht_scale(x, rs, block)
            assert got.shape == ref.shape
            err = float((got.float() - ref.float()).abs().max())
            scale = float(ref.float().abs().max())
            assert err <= 2e-3 * max(scale, 1.0), (d, block, b, err, scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_storage_contract():
    gen = torch.Generator().manual_seed(5)
    w = torch.randn(4, 64, generator=gen)
    h = _correlated_hessian(64, seed=5)
    q = ldlq_tcq_quantize_rows(w.cuda(), h.cuda(), bits=3, n_states=16, block_size=16)
    rebuilt = decode_levels(q.path_bits.cpu(), q.member_codes.cpu(), Trellis(16))
    assert torch.equal(rebuilt, q.level_codes.cpu())
