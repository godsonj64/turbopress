import pytest
import torch

transformers = pytest.importorskip("transformers")

from turbopress.real_model import (
    RealConfig,
    collect_input_scales,
    evaluate_pair,
    quantize_model_copy,
)


@pytest.fixture(scope="module")
def tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).eval()


@pytest.fixture(scope="module")
def batches():
    gen = torch.Generator().manual_seed(0)
    return [torch.randint(0, 256, (2, 32), generator=gen) for _ in range(2)]


def test_collect_input_scales_shapes(tiny_llama, batches):
    scales = collect_input_scales(tiny_llama, batches)
    # 2 layers x 7 linears each.
    assert len(scales) == 14
    for s in scales.values():
        assert torch.all(s > 0)
        assert torch.all(torch.isfinite(s))
    # q/k/v projections read the hidden stream (64); down_proj reads MLP (128).
    assert scales["0.self_attn.q_proj"].shape == (64,)
    assert scales["0.mlp.down_proj"].shape == (128,)


def test_equilibrated_quantization_end_to_end(tiny_llama, batches):
    scales = collect_input_scales(tiny_llama, batches)
    cfg = RealConfig("2b tcq +eq", bits=2, method="tcq", equilibrate=True)
    q_model, stats = quantize_model_copy(tiny_llama, cfg, seed=0, col_scales=scales)
    assert stats["n_replaced"] == 14
    # Equilibration vectors add storage.
    plain_cfg = RealConfig("2b tcq", bits=2, method="tcq")
    _, plain_stats = quantize_model_copy(tiny_llama, plain_cfg, seed=0)
    assert stats["bits_per_weight"] > plain_stats["bits_per_weight"]
    metrics = evaluate_pair(tiny_llama, q_model, batches)
    assert metrics["mean_kl"] >= 0


def test_equilibrate_requires_scales(tiny_llama):
    cfg = RealConfig("2b +eq", bits=2, equilibrate=True)
    with pytest.raises(ValueError):
        quantize_model_copy(tiny_llama, cfg, seed=0, col_scales=None)
