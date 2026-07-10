import pytest
import torch

from turbopress.allocate import (
    LayerSlot,
    allocate_bits,
    build_mixed_model,
    measure_sensitivity,
)

transformers = pytest.importorskip("transformers")


def _slots(sizes):
    return [LayerSlot(f"L{i}", i, "proj", n) for i, n in enumerate(sizes)]


def test_allocate_respects_budget_and_minimum():
    slots = _slots([100, 100, 100])
    # Layer 1 is highly sensitive; others barely move KL.
    cost = {
        "L0": {2: 0.10, 3: 0.09, 4: 0.08},
        "L1": {2: 5.00, 3: 1.00, 4: 0.20},
        "L2": {2: 0.10, 3: 0.09, 4: 0.08},
    }
    alloc = allocate_bits(slots, cost, target_bits=2.5)
    assert all(2 <= b <= 4 for b in alloc.values())
    avg = sum(100 * b for b in alloc.values()) / 300
    assert avg <= 2.5 + 1e-9
    # The sensitive layer should get the extra bits.
    assert alloc["L1"] >= alloc["L0"]
    assert alloc["L1"] >= alloc["L2"]


def test_allocate_full_budget_hits_max():
    slots = _slots([50, 50])
    cost = {"L0": {2: 1.0, 4: 0.1}, "L1": {2: 1.0, 4: 0.1}}
    alloc = allocate_bits(slots, cost, target_bits=4.0)
    assert alloc == {"L0": 4, "L1": 4}


def test_allocate_tight_budget_stays_at_floor():
    slots = _slots([50, 50])
    cost = {"L0": {2: 1.0, 3: 0.5}, "L1": {2: 1.0, 3: 0.5}}
    alloc = allocate_bits(slots, cost, target_bits=2.0)
    assert alloc == {"L0": 2, "L1": 2}


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


def test_measure_and_build_mixed(tiny_llama):
    gen = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 256, (2, 24), generator=gen) for _ in range(2)]
    slots, cost = measure_sensitivity(
        tiny_llama, batches, bit_choices=(2, 4), method="scalar"
    )
    assert len(slots) == 14  # 7 linears x 2 layers
    for c in cost.values():
        assert set(c.keys()) == {2, 4}
        assert all(v >= -1e-4 for v in c.values())  # KL >= 0 up to fp error
        # More bits should not increase KL by much; usually it drops.
        assert c[4] <= c[2] + 1e-3

    alloc = allocate_bits(slots, cost, target_bits=3.0)
    q_model, stats = build_mixed_model(tiny_llama, alloc, method="scalar")
    assert stats["bits_per_weight"] <= 3.0 + 1.0  # + per-row scale overhead
    n_q = sum(1 for m in q_model.modules() if type(m).__name__ == "QJLCorrectedLinear")
    assert n_q == 14
    # The original model must be untouched.
    assert not any(
        type(m).__name__ == "QJLCorrectedLinear" for m in tiny_llama.modules()
    )
