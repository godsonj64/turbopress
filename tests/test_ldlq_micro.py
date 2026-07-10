import json

from turbopress.ldlq_micro import main, run_micro


def test_run_micro_ldlq_wins_at_every_bit_width():
    results = run_micro(bits_list=(2, 3), d=64, n=16, n_samples=1024, n_seeds=2)
    assert [r["bits"] for r in results] == [2, 3]
    for r in results:
        assert r["ldlq_loss_mean"] < r["nearest_loss_mean"]
        assert r["gain"] > 1.0
        assert r["nearest_loss_std"] >= 0.0


def test_main_writes_json(tmp_path, capsys):
    out = tmp_path / "micro.json"
    main(["--out", str(out), "--dim", "64", "--rows", "16",
          "--samples", "1024", "--seeds", "2"])
    payload = json.loads(out.read_text())
    assert payload["settings"]["seeds"] == 2
    assert len(payload["results"]) == 3
    assert "LDLQ" in capsys.readouterr().out
