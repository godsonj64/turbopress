"""Offline tests for validate's model-source routing.

These exercise the artifact-vs-HF dispatch without downloading any model: a
fake artifact directory carries the two sentinel files and a stub
``run_quantized.py`` whose ``load_quantized_model`` records how it was called.
"""

import torch

from turbopress import validate

_STUB_RUNTIME = '''
def load_quantized_model(artifact_dir=None, device="cuda", dtype=None):
    print("decoded chatter that should be swallowed")
    return {"loaded_from": str(artifact_dir), "device": device, "dtype": dtype}
'''


def _make_fake_artifact(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "turbopress_weights.pt").write_bytes(b"stub")
    (root / "run_quantized.py").write_text(_STUB_RUNTIME, encoding="utf-8")
    (root / "tokenizer").mkdir(exist_ok=True)
    return root


def test_is_artifact_true_for_artifact_dir(tmp_path):
    art = _make_fake_artifact(tmp_path / "Model-turbopress-3bit")
    assert validate._is_turbopress_artifact(str(art))


def test_is_artifact_false_for_plain_dir(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text("{}", encoding="utf-8")
    assert not validate._is_turbopress_artifact(str(plain))


def test_is_artifact_false_for_hf_id():
    # A bare HF id is not a local directory.
    assert not validate._is_turbopress_artifact("Qwen/Qwen3-4B")


def test_tokenizer_source_routes_artifact_to_subdir(tmp_path):
    art = _make_fake_artifact(tmp_path / "Model-turbopress-3bit")
    assert validate._tokenizer_source(str(art)) == str(art / "tokenizer")


def test_tokenizer_source_passthrough_for_hf_id():
    assert validate._tokenizer_source("Qwen/Qwen3-4B") == "Qwen/Qwen3-4B"


def test_load_model_uses_bundled_loader_for_artifact(tmp_path, capsys):
    art = _make_fake_artifact(tmp_path / "Model-turbopress-3bit")
    model = validate._load_model(str(art), "cpu", torch.float16)
    # Routed through the artifact's own run_quantized.py loader with our args.
    assert model["loaded_from"] == str(art)
    assert model["device"] == "cpu"
    assert model["dtype"] == torch.float16
    # The loader's per-matrix chatter must not leak into the validate report.
    assert "decoded chatter" not in capsys.readouterr().out
