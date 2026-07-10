import pytest
import torch
from torch import nn

from turbopress.hadamard import RandomizedOrthogonal
from turbopress.runtime import (
    PackedTCQLinear,
    pack_le,
    pack_linear,
    pack_model,
    subset_lut,
    unpack_le,
    window_decode_levels,
)
from turbopress.trellis import Trellis, decode_levels, tcq_quantize_rows


@pytest.mark.parametrize("width", [1, 2, 3, 5, 8])
def test_pack_le_roundtrip(width):
    gen = torch.Generator().manual_seed(0)
    vals = torch.randint(0, 1 << width, (5, 77), generator=gen, dtype=torch.uint8)
    packed = pack_le(vals, width)
    assert packed.dtype == torch.uint8
    assert packed.shape == (5, -(-77 * width // 8))
    assert torch.equal(unpack_le(packed, 77, width), vals)


@pytest.mark.parametrize("n_states", [8, 16, 64])
@pytest.mark.parametrize("bits", [1, 2, 3])
def test_window_decode_matches_sequential_walk(n_states, bits):
    """The load-bearing claim of the runtime: decoding is window-parallel.

    The parallel window decode must reproduce the sequential trellis walk
    bit-for-bit on real encoder output (including the zero-state start)."""
    gen = torch.Generator().manual_seed(0)
    w = torch.randn(6, 96, generator=gen)
    q = tcq_quantize_rows(w, bits=bits, n_states=n_states)
    tr = Trellis(n_states)
    sequential = decode_levels(q.path_bits, q.member_codes, tr)
    parallel = window_decode_levels(q.path_bits, q.member_codes, tr)
    assert torch.equal(parallel, sequential)
    assert torch.equal(parallel, q.level_codes)


def test_subset_lut_size_and_range():
    tr = Trellis(64)
    lut = subset_lut(tr)
    assert lut.shape == (128,)
    assert set(lut.unique().tolist()) == {0, 1, 2, 3}


def _make_layer(in_f=128, out_f=64, seed=0, bias=False):
    gen = torch.Generator().manual_seed(seed)
    linear = nn.Linear(in_f, out_f, bias=bias)
    linear.weight.data = torch.randn(out_f, in_f, generator=gen) / in_f**0.5
    if bias:
        linear.bias.data = torch.randn(out_f, generator=gen) * 0.1
    act_rms = torch.rand(in_f, generator=gen) + 0.5
    return linear, act_rms


def _reference_forward(payload, linear, act_rms, x, bits, n_states, seed):
    """Reconstruct in the original basis (the validated math) and apply."""
    w = linear.weight.detach().float()
    act = act_rms.float().clamp_min(1e-8)
    act = act.clamp_min(0.05 * float(act.pow(2).mean().sqrt()))
    cn = w.norm(dim=0)
    cn = cn.clamp_min(max(0.05 * float(cn.pow(2).mean().sqrt()), 1e-12))
    s = (act / cn).sqrt().to(torch.float16).float()
    rot = RandomizedOrthogonal(w.shape[1], seed=seed)
    q = tcq_quantize_rows(rot(w * s[None, :]), bits=bits, n_states=n_states)
    w_rot_hat = (q.scales.to(torch.float16).float()[:, None]
                 * q.codebook.to(torch.float16).float()[q.level_codes.long()])
    w_hat = rot.inverse(w_rot_hat) / s[None, :]
    y = x @ w_hat.T
    if linear.bias is not None:
        y = y + linear.bias.detach().to(torch.float16).float()
    return y


@pytest.mark.parametrize("bits", [2, 3])
def test_packed_forward_matches_reference(bits):
    linear, act_rms = _make_layer(bias=True)
    payload = pack_linear(linear, act_rms, bits=bits, n_states=16, seed=3)
    packed = PackedTCQLinear(payload, mode="cached")
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(32, 128, generator=gen)
    with torch.no_grad():
        y = packed(x)
    y_ref = _reference_forward(payload, linear, act_rms, x, bits, 16, seed=3)
    rel = float((y - y_ref).norm() / y_ref.norm())
    assert rel < 2e-3  # same math, different (rotated) evaluation order


def test_cached_and_tiled_agree():
    # Same decoded weights, two evaluation orders: cached folds the rotation
    # into fp16 weights at load (one extra fp16 rounding, the standard fp16
    # checkpoint representation); tiled rotates activations in fp32. Compare
    # norm-relative, not elementwise (near-zero outputs make relative
    # elementwise noise meaningless).
    linear, act_rms = _make_layer(in_f=96, out_f=200)
    payload = pack_linear(linear, act_rms, bits=2, n_states=8, seed=0)
    cached = PackedTCQLinear(payload, mode="cached")
    tiled = PackedTCQLinear(payload, mode="tiled", tile_rows=64)
    x = torch.randn(4, 7, 96)
    with torch.no_grad():
        a, b = cached(x), tiled(x)
    assert float((a - b).norm() / b.norm()) < 2e-3


def test_packed_bytes_scale_with_bits():
    linear, act_rms = _make_layer(in_f=256, out_f=256)
    sizes = {}
    for bits in (2, 4):
        payload = pack_linear(linear, act_rms, bits=bits, n_states=8)
        sizes[bits] = PackedTCQLinear(payload, mode="tiled").packed_bytes()
    fp16 = 2 * 256 * 256
    assert sizes[2] < fp16 / 6  # ~2.3 bits/weight incl. overheads
    assert 1.6 < sizes[4] / sizes[2] < 2.1
    # cached mode trades memory for speed: resident ~ fp16.
    cached = PackedTCQLinear(pack_linear(linear, act_rms, bits=2, n_states=8),
                             mode="cached")
    assert cached.packed_bytes() >= fp16
    assert cached.path_packed is None  # packed streams freed after folding


def test_mode_and_payload_validation():
    linear, act_rms = _make_layer()
    payload = pack_linear(linear, act_rms, bits=2, n_states=8)
    with pytest.raises(ValueError):
        PackedTCQLinear(payload, mode="warp")
    with pytest.raises(ValueError):
        PackedTCQLinear({"format": 1})
    packed = PackedTCQLinear(payload)
    with pytest.raises(ValueError):
        packed(torch.randn(2, 64))


def test_pack_model_end_to_end():
    transformers = pytest.importorskip("transformers")
    from turbopress.real_model import collect_input_scales

    config = transformers.LlamaConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    gen = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 256, (2, 32), generator=gen)]
    stats = collect_input_scales(model, batches)

    report = pack_model(model, stats, bits=2, n_states=8, mode="tiled")
    n_packed = sum(isinstance(m, PackedTCQLinear) for m in model.modules())
    assert n_packed == 14
    assert report["compression"] > 5  # ~2.4 bits vs 16
    with torch.no_grad():
        out = model(input_ids=batches[0]).logits
    assert out.shape == (2, 32, 256)
    assert torch.all(torch.isfinite(out))


def test_triton_kernel_matches_torch_paths():
    from turbopress.triton_kernel import HAS_TRITON
    if not HAS_TRITON:
        pytest.skip("triton not available on this platform")
    linear, act_rms = _make_layer(in_f=256, out_f=192)
    payload = pack_linear(linear, act_rms, bits=3, n_states=64)
    cached = PackedTCQLinear(payload, mode="cached").cuda()
    fused = PackedTCQLinear(payload, mode="triton").cuda()
    x = torch.randn(2, 256, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        torch.testing.assert_close(fused(x), cached(x), rtol=2e-2, atol=2e-2)
