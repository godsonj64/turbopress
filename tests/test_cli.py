"""Offline tests for the CLI wiring (no model downloads, no torch heavy paths)."""

import pytest

from turbopress import __version__
from turbopress.cli import build_parser, main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_prints_help_and_returns_1(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_compress_args_parse():
    args = build_parser().parse_args(["compress", "Qwen/Qwen3-4B", "--bits", "2"])
    assert args.command == "compress"
    assert args.model == "Qwen/Qwen3-4B"
    assert args.bits == 2
    assert args.n_states == 64
    assert callable(args.func)


def test_validate_args_parse():
    args = build_parser().parse_args(
        ["validate", "org/fp16", "org/quant", "--seqs", "8", "--out", "r.json"]
    )
    assert args.command == "validate"
    assert args.reference == "org/fp16"
    assert args.candidate == "org/quant"
    assert args.seqs == 8
    assert args.out == "r.json"


def test_bits_bounds_rejected_by_parser_choices():
    # n_states is constrained by argparse choices; invalid values exit(2).
    with pytest.raises(SystemExit):
        build_parser().parse_args(["compress", "m", "--n-states", "7"])


def test_ablate_args_parse():
    args = build_parser().parse_args(
        ["ablate", "Qwen/Qwen3-4B", "--bits", "3", "--methods", "nearest,tcq"]
    )
    assert args.command == "ablate"
    assert args.model == "Qwen/Qwen3-4B"
    assert args.bits == 3
    assert args.methods == "nearest,tcq"
    assert args.n_states == 64
    assert callable(args.func)


def test_ablate_rejects_bad_n_states():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ablate", "m", "--n-states", "7"])
