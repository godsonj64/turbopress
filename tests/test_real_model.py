import pytest
import torch
from torch import nn

transformers = pytest.importorskip("transformers")

from turbopress.linear import QJLCorrectedLinear
from turbopress.real_model import (
    RealConfig,
    ablation_configs,
    evaluate_pair,
    quantize_model_copy,
    run_ablation,
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


def test_ablation_configs_ladder():
    cfgs = ablation_configs(("nearest", "scalar", "gptq", "tcq"), bits=3, n_states=64)
    by_label = {c.label: c for c in cfgs}
    assert not by_label["nearest 3b"].equilibrate
    assert by_label["scalar 3b +eq"].equilibrate
    assert by_label["scalar 3b +eq"].method == "scalar"
    assert by_label["gptq 3b +eq"].error_feedback
    tcq = by_label["tcq 3b S=64 +eq"]
    assert tcq.method == "tcq" and tcq.n_states == 64 and tcq.equilibrate
    assert all(c.bits == 3 for c in cfgs)
    with pytest.raises(ValueError):
        ablation_configs(("bogus",), bits=3)
    # TCQ's doubled codebook caps at 7 bits; scalar methods still allow 8.
    with pytest.raises(ValueError, match="tcq"):
        ablation_configs(("tcq",), bits=8)
    assert ablation_configs(("gptq",), bits=8)[0].bits == 8


def test_run_ablation_shares_calibration_and_ranks_by_bits(tiny_llama):
    gen = torch.Generator().manual_seed(0)
    eval_batches = [torch.randint(0, 256, (2, 32), generator=gen) for _ in range(2)]
    calib_batches = [torch.randint(0, 256, (2, 32), generator=gen) for _ in range(2)]
    # Exercise every path that needs calibration/hessians end to end: a coarse
    # 2-bit gptq (equilibration + LDLQ error feedback) vs a fine 6-bit tcq
    # (TCQ's doubled codebook caps bits at 7).
    configs = [
        RealConfig("gptq 2b +eq", bits=2, equilibrate=True, error_feedback=True),
        RealConfig("tcq 6b +eq", bits=6, method="tcq", n_states=8, equilibrate=True),
    ]
    rows = run_ablation(tiny_llama, configs, eval_batches, calib_batches, seed=0)
    assert [r["label"] for r in rows] == ["gptq 2b +eq", "tcq 6b +eq"]
    for r in rows:
        assert r["mean_kl"] >= 0
        assert 0 <= r["top1_agreement"] <= 1
        assert r["bits_per_weight"] > 0
        assert r["n_replaced"] == 14
    # Same fp reference; finer quantization tracks it more closely.
    assert rows[0]["ppl_fp"] == pytest.approx(rows[1]["ppl_fp"])
    assert rows[1]["mean_kl"] < rows[0]["mean_kl"]


def test_run_ablation_requires_calibration_when_needed(tiny_llama):
    gen = torch.Generator().manual_seed(0)
    eval_batches = [torch.randint(0, 256, (2, 16), generator=gen)]
    configs = [RealConfig("scalar 3b +eq", bits=3, equilibrate=True)]
    with pytest.raises(ValueError, match="calib"):
        run_ablation(tiny_llama, configs, eval_batches, calib_batches=None)


def test_quantized_model_forward_matches_at_high_bits(tiny_llama):
    q8, _ = quantize_model_copy(tiny_llama, RealConfig("8b", bits=8), seed=0)
    ids = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        ref = tiny_llama(input_ids=ids).logits
        out = q8(input_ids=ids).logits
    assert float((out - ref).norm() / ref.norm()) < 0.05
