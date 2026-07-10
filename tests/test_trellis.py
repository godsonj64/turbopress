import pytest
import torch
from torch import nn

from turbopress.codebooks import lloyd_max_gaussian
from turbopress.linear import QJLCorrectedLinear
from turbopress.trellis import Trellis, decode_levels, tcq_quantize_rows


@pytest.mark.parametrize("n_states", [4, 8, 16, 64])
def test_trellis_structure(n_states):
    tr = Trellis(n_states)
    assert tr.subset_table.shape == (n_states, 2)
    # Every state has exactly two incoming branches, and predecessors differ.
    assert torch.all(tr.prev0 != tr.prev1)
    assert torch.all((tr.prev0 >= 0) & (tr.prev0 < n_states))
    assert torch.all((tr.prev1 >= 0) & (tr.prev1 < n_states))
    # All four subsets are used somewhere in the trellis.
    assert set(tr.subset_table.flatten().tolist()) == {0, 1, 2, 3}
    # The two branches out of any state carry different subsets.
    assert torch.all(tr.subset_table[:, 0] != tr.subset_table[:, 1])


def test_trellis_rejects_unknown_state_count():
    with pytest.raises(ValueError):
        Trellis(6)


def test_roundtrip_proves_bit_stream_storage():
    """level_codes must be exactly recoverable from path_bits + member_codes:
    this is what justifies charging only `bits` bits/weight for TCQ."""
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(8, 96, generator=gen)
    for bits in (1, 2, 3):
        q = tcq_quantize_rows(w, bits=bits, n_states=8)
        rebuilt = decode_levels(q.path_bits, q.member_codes, Trellis(8))
        assert torch.equal(rebuilt, q.level_codes)
        assert int(q.path_bits.max()) <= 1
        assert int(q.member_codes.max()) < max(1, 1 << (bits - 1))


def test_tcq_beats_scalar_lloyd_max_on_gaussian():
    """The whole point of TCQ: distortion below the scalar-quantizer optimum.

    N(0,1) @ 2 bits: scalar Lloyd-Max MSE = 0.1175, 8-state TCQ should land
    around 0.089-0.100, above the rate-distortion bound 0.0625.
    """
    gen = torch.Generator().manual_seed(0)
    z = torch.randn(4, 8192, generator=gen)
    q = tcq_quantize_rows(z, bits=2, n_states=8, scale_iters=0)
    # Use unit scale explicitly: measure pure codebook distortion.
    recon = q.codebook[q.level_codes.long()] * q.scales[:, None]
    mse = float((recon - z).pow(2).mean())
    _, scalar_mse = lloyd_max_gaussian(2)
    assert mse < 0.96 * scalar_mse  # clearly better than scalar optimum
    assert mse > 0.0625  # cannot beat the rate-distortion bound


def test_more_states_do_not_hurt():
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(2, 8192, generator=gen)

    def mse(n_states):
        q = tcq_quantize_rows(z, bits=2, n_states=n_states, scale_iters=0)
        recon = q.codebook[q.level_codes.long()] * q.scales[:, None]
        return float((recon - z).pow(2).mean())

    assert mse(64) <= mse(4) * 1.02


def test_scale_refinement_and_zero_rows():
    gen = torch.Generator().manual_seed(2)
    w = torch.randn(6, 128, generator=gen) * 3.7
    w[3] = 0.0
    q = tcq_quantize_rows(w, bits=2, n_states=8)
    assert q.scales[3] == 0.0
    assert torch.all(q.dequantize()[3] == 0.0)
    rel = float((q.dequantize() - w).norm() / w.norm())
    assert rel < 0.34  # better than the scalar 2-bit ~0.34


def test_input_validation():
    with pytest.raises(ValueError):
        tcq_quantize_rows(torch.randn(2, 3, 4), bits=2)
    with pytest.raises(ValueError):
        tcq_quantize_rows(torch.randn(2, 8), bits=8)
    with pytest.raises(TypeError):
        tcq_quantize_rows(torch.ones(2, 8, dtype=torch.int64), bits=2)


def test_linear_integration_tcq_beats_scalar():
    gen = torch.Generator().manual_seed(3)
    linear = nn.Linear(256, 128, bias=False)
    linear.weight.data = torch.randn(128, 256, generator=gen) / 16.0
    x = torch.randn(64, 256, generator=gen)
    with torch.no_grad():
        y_ref = linear(x)
        y_scalar = QJLCorrectedLinear.from_linear(linear, bits=2, seed=0)(x)
        q_tcq = QJLCorrectedLinear.from_linear(linear, bits=2, seed=0, method="tcq")
        y_tcq = q_tcq(x)
    err_scalar = float((y_scalar - y_ref).norm() / y_ref.norm())
    err_tcq = float((y_tcq - y_ref).norm() / y_ref.norm())
    assert err_tcq < err_scalar
    # Storage: TCQ still charges exactly `bits` per weight for codes.
    report = q_tcq.storage_report()
    assert report["bits_per_weight_codes"] == 2

    with pytest.raises(ValueError):
        QJLCorrectedLinear.from_linear(linear, bits=2, method="vector")
