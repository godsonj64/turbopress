import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from turbopress.linear import QJLCorrectedLinear
from turbopress.real_model import RealConfig, evaluate_pair, quantize_model_copy


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


def _count_linears(module):
    return sum(isinstance(m, nn.Linear) for m in module.modules())


def test_quantize_replaces_only_decoder_linears(tiny_llama):
    cfg = RealConfig("2b", bits=2, sketch_ratio=1.0)
    q_model, stats = quantize_model_copy(tiny_llama, cfg, seed=0)
    # Llama block: q, k, v, o + gate, up, down = 7 linears per layer.
    assert stats["n_replaced"] == 14
    n_q = sum(isinstance(m, QJLCorrectedLinear) for m in q_model.modules())
    assert n_q == 14
    # lm_head must remain a plain Linear.
    assert isinstance(q_model.lm_head, nn.Linear)
    # Original model untouched.
    assert _count_linears(tiny_llama) == _count_linears(q_model.lm_head) + 14
    assert not any(isinstance(m, QJLCorrectedLinear) for m in tiny_llama.modules())
    # bits/weight = 2 (codes) + ~1 (k=d signs) + scales/norms overhead, which
    # is proportionally large at these tiny dims (16+16 bits over d=64 cols).
    assert 3.0 < stats["bits_per_weight"] < 3.6


def test_evaluate_pair_metrics_sane(tiny_llama):
    gen = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 256, (2, 32), generator=gen) for _ in range(2)]

    q8, _ = quantize_model_copy(tiny_llama, RealConfig("8b", bits=8), seed=0)
    q2, _ = quantize_model_copy(tiny_llama, RealConfig("2b", bits=2), seed=0)
    m8 = evaluate_pair(tiny_llama, q8, batches)
    m2 = evaluate_pair(tiny_llama, q2, batches)

    for m in (m8, m2):
        assert m["mean_kl"] >= 0
        assert 0 <= m["top1_agreement"] <= 1
        assert m["ppl_fp"] > 0 and m["ppl_q"] > 0
        assert m["n_tokens"] == 2 * 2 * 31
    # Same fp reference in both evaluations.
    assert m8["ppl_fp"] == pytest.approx(m2["ppl_fp"])
    # Finer quantization tracks the fp model more closely.
    assert m8["mean_kl"] < m2["mean_kl"]
    assert m8["top1_agreement"] > m2["top1_agreement"]


def test_quantized_model_forward_matches_at_high_bits(tiny_llama):
    q8, _ = quantize_model_copy(tiny_llama, RealConfig("8b", bits=8), seed=0)
    ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = tiny_llama(input_ids=ids).logits
        out = q8(input_ids=ids).logits
    assert float((out - ref).norm() / ref.norm()) < 0.05
