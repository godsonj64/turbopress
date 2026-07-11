"""Tests for the packed-artifact loader and the ``turbopress run`` subcommand.

The end-to-end load + Triton/CUDA-graph paths need a real artifact and a GPU,
so these cover what runs offline: the v1->v2 conversion is exact (the core
correctness lock), argument parsing, and the loader's error guards.
"""

import pytest
import torch

from turbopress.cli import build_parser
from turbopress.pipeline import quantize_matrix
from turbopress.runtime import (
    PackedTCQLinear,
    load_packed_model,
    v1_payload_to_packed,
)


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_v1_to_packed_reproduces_artifact_weights(bits):
    """A PackedTCQLinear built from the converted payload decodes to the same
    weights the pipeline reconstructed -- i.e. no quantization is lost."""
    torch.manual_seed(0)
    n, d = 96, 256
    w = torch.randn(n, d) * 0.05
    act_rms = w.abs().mean(0).clamp_min(1e-4)
    payload, w_hat = quantize_matrix(w, act_rms, bits, 16, seed=0, device="cpu")

    packed = PackedTCQLinear(v1_payload_to_packed(payload, None, "cpu"), mode="cached")
    assert torch.allclose(packed._w_cache.float(), w_hat.float(), atol=1e-2)

    # and a forward matches x @ w_hat^T to fp16 tolerance
    x = torch.randn(4, d)
    y = packed(x.half()).float()
    assert torch.allclose(y, x @ w_hat.float().T, atol=5e-2)


def test_v1_to_packed_carries_bias():
    torch.manual_seed(1)
    n, d = 64, 128
    w = torch.randn(n, d) * 0.05
    bias = torch.randn(n)
    payload, _ = quantize_matrix(w, w.abs().mean(0).clamp_min(1e-4), 3, 16, seed=0, device="cpu")
    packed = PackedTCQLinear(v1_payload_to_packed(payload, bias, "cpu"), mode="cached")
    assert packed.bias is not None
    assert torch.allclose(packed.bias.float(), bias, atol=1e-2)


def test_run_args_parse():
    args = build_parser().parse_args(
        ["run", "art_dir", "--mode", "triton", "--cuda-graph",
         "--max-new", "32", "--prompt", "hi"]
    )
    assert args.command == "run"
    assert args.artifact == "art_dir"
    assert args.mode == "triton"
    assert args.cuda_graph is True
    assert args.max_new == 32
    assert args.prompt == "hi"
    assert callable(args.func)


def test_run_defaults():
    args = build_parser().parse_args(["run", "art_dir"])
    assert args.mode == "cached"          # always-works default
    assert args.cuda_graph is False
    assert args.device is None


def test_run_rejects_bad_mode():
    with pytest.raises(SystemExit):  # argparse choices
        build_parser().parse_args(["run", "art", "--mode", "bogus"])


def test_load_packed_model_bad_mode_raises():
    with pytest.raises(ValueError):
        load_packed_model("anywhere", device="cpu", mode="nope")


def test_load_packed_model_missing_artifact_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_packed_model(str(tmp_path), device="cpu", mode="cached")
