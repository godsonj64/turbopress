"""extra_state must not store tied embeddings twice (tie_word_embeddings)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from turbopress.pipeline import extra_state


def _tiny_llama(tie: bool):
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=tie,
    )
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(config).eval()


def test_tied_lm_head_dropped_and_restored_on_load():
    model = _tiny_llama(tie=True)
    extra = extra_state(model, quant_keys=set())
    assert "lm_head.weight" not in extra
    assert "model.embed_tokens.weight" in extra

    # The artifact loader path: fresh model from config (post-init ties the
    # weights), then a strict=False load of extra_state.
    model2 = transformers.LlamaForCausalLM(model.config)
    missing, unexpected = model2.load_state_dict(extra, strict=False)
    assert not unexpected
    sd2 = model2.state_dict()
    assert torch.equal(sd2["lm_head.weight"], sd2["model.embed_tokens.weight"])
    assert torch.equal(
        sd2["lm_head.weight"].half(), extra["model.embed_tokens.weight"]
    )


def test_untied_lm_head_kept():
    model = _tiny_llama(tie=False)
    extra = extra_state(model, quant_keys=set())
    assert "lm_head.weight" in extra
    assert "model.embed_tokens.weight" in extra


def test_quant_keys_excluded():
    model = _tiny_llama(tie=True)
    key = "model.layers.0.self_attn.q_proj.weight"
    extra = extra_state(model, quant_keys={key})
    assert key not in extra
