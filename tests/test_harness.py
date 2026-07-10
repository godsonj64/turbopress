import json

import torch

from turbopress.harness import (
    QuantConfig,
    main,
    make_activations,
    run_compounding,
    run_single_layer,
)


def test_make_activations_energy_and_determinism():
    x1 = make_activations(64, 128, "isotropic", seed=0)
    x2 = make_activations(64, 128, "isotropic", seed=0)
    torch.testing.assert_close(x1, x2)
    a = make_activations(512, 128, "aniso", seed=0)
    # Per-coordinate RMS ~ 1 by construction.
    rms = float(a.pow(2).mean().sqrt())
    assert 0.8 < rms < 1.2
    # The fixed mean direction is actually fixed across samples.
    assert float(a.mean(0).norm()) > 0.4 * float(a[0].norm())


def test_effective_bits():
    assert QuantConfig("a", 2, 0, "nearest").effective_bits(256) == 2 + 16 / 256
    assert QuantConfig("a", 2, 256, "nearest").effective_bits(256) == 2 + (16 + 272) / 256


def test_single_layer_bias_separation():
    configs = [
        QuantConfig("nearest 2b", 2, 0, "nearest"),
        QuantConfig("2b + QJL k=d", 2, 128, "nearest"),
    ]
    res = run_single_layer(configs, dim=128, n_out=128, batch=64, n_trials=16, seed=0)
    uncorrected = next(r for r in res if r["label"] == "nearest 2b")
    corrected = next(r for r in res if r["label"] == "2b + QJL k=d")
    # Deterministic config: bias equals total error.
    assert uncorrected["rel_bias"] == uncorrected["rel_rmse"]
    # QJL correction: systematic error is a small fraction of total error.
    assert corrected["rel_bias"] < 0.5 * corrected["rel_rmse"]
    assert corrected["rel_bias"] < 0.5 * uncorrected["rel_bias"]


def test_compounding_runs_and_reports():
    configs = [QuantConfig("nearest 2b", 2, 0, "nearest")]
    res = run_compounding(
        configs, depth=4, dim=64, batch=32, vocab=32, act_mode="isotropic",
        seed=0, checkpoints=(1, 2, 4),
    )
    r = res[0]
    assert set(r["depth_rel_err"]) == {"1", "2", "4"}
    assert all(v > 0 for v in r["depth_rel_err"].values())
    assert r["final_kl"] >= 0
    assert 0 <= r["top1_agreement"] <= 1


def test_main_quick_smoke(tmp_path, capsys):
    out = tmp_path / "res.json"
    main(["--quick", "--out", str(out)])
    payload = json.loads(out.read_text())
    assert payload["settings"]["quick"] is True
    assert len(payload["results"]["single_layer"]) > 0
    assert len(payload["results"]["compounding"]) > 0
    printed = capsys.readouterr().out
    assert "Single layer" in printed and "Depth compounding" in printed
