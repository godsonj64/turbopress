"""Command-line interface for TurboPress.

Subcommands:
    turbopress compress <model> --bits 3        run the full quantization pipeline
    turbopress validate <ref> <candidate>       measure KL / top-1 / perplexity

Heavy dependencies (torch, transformers) are imported lazily inside each
handler so ``turbopress --version`` and ``turbopress --help`` stay fast and
work without a full install.
"""

from __future__ import annotations

import argparse
import json
import sys

from turbopress import __version__


def _add_compress(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compress",
        help="quantize every decoder linear of a Hugging Face causal LM",
        description="Run the TurboPress pipeline (rotate -> quarter-power "
        "equilibration -> trellis-coded quantization) and write a certified, "
        "self-contained artifact.",
    )
    p.add_argument("model", help="Hugging Face model id or local path")
    p.add_argument("--bits", type=int, default=3, help="bits/weight, 2..6 (default: 3)")
    p.add_argument("--n-states", type=int, default=64, choices=[4, 8, 16, 64],
                   help="trellis states: 16 = faster, 64 = best (default)")
    p.add_argument("--out", default="turbopress_out", help="output directory")
    p.add_argument("--seqlen", type=int, default=512, help="tokens per sequence")
    p.add_argument("--eval-seqs", type=int, default=8, help="eval sequences")
    p.add_argument("--calib-seqs", type=int, default=8, help="calibration sequences")
    p.add_argument("--batch", type=int, default=2, help="sequences per forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--no-self-test", action="store_true",
                   help="skip reloading the artifact to verify it round-trips")
    p.set_defaults(func=_run_compress)


def _add_validate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "validate",
        help="measure KL / top-1 agreement / perplexity between two models",
        description="Compare a candidate model against a reference (they must "
        "share a tokenizer/vocabulary) on held-out text. Method-agnostic: works "
        "for TurboPress, GPTQ, AWQ, or any HF-loadable checkpoint.",
    )
    p.add_argument("reference", help="reference model id/path (e.g. the fp16 model)")
    p.add_argument("candidate", help="candidate model id/path (e.g. the quantized copy)")
    p.add_argument("--seqs", type=int, default=16, help="eval sequences (default: 16)")
    p.add_argument("--seqlen", type=int, default=256, help="tokens per sequence")
    p.add_argument("--batch", type=int, default=4, help="sequences per forward")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--data-dir", default="data", help="cache dir for eval text")
    p.add_argument("--out", default=None, help="write the report to this JSON file")
    p.set_defaults(func=_run_validate)


def _run_compress(args: argparse.Namespace) -> int:
    from turbopress.pipeline import compress

    cfg = {
        "MODEL_ID": args.model,
        "BITS": args.bits,
        "N_STATES": args.n_states,
        "OUT_DIR": args.out,
        "SEQLEN": args.seqlen,
        "EVAL_SEQS": args.eval_seqs,
        "CALIB_SEQS": args.calib_seqs,
        "BATCH": args.batch,
        "SEED": args.seed,
        "SELF_TEST": not args.no_self_test,
    }
    if args.device:
        cfg["DEVICE"] = args.device
    result = compress(cfg)
    m = result["metrics"]
    print(
        f"\nartifact: {result['artifact']}\n"
        f"bits/weight: {result['bits_per_weight']:.3f}  "
        f"KL(fp||q): {m['mean_kl']:.4f}  top-1: {m['top1_agreement']:.3f}  "
        f"ppl: {m['ppl_fp']:.2f} -> {m['ppl_q']:.2f}"
    )
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    from turbopress.validate import validate_models

    result = validate_models(
        args.reference,
        args.candidate,
        seqs=args.seqs,
        seqlen=args.seqlen,
        batch=args.batch,
        device=args.device,
        dtype=args.dtype,
        data_dir=args.data_dir,
        out=args.out,
    )
    m = result["metrics"]
    print(
        f"\nreference: {result['reference']}\n"
        f"candidate: {result['candidate']}\n"
        f"KL(ref||cand): {m['mean_kl']:.4f} nats   "
        f"top-1 agreement: {m['top1_agreement']:.3f}\n"
        f"perplexity: {m['ppl_fp']:.3f} (ref) -> {m['ppl_q']:.3f} (cand)   "
        f"[{m['n_tokens']} tokens]"
    )
    if args.out:
        print(f"report written to {args.out}")
    else:
        print(json.dumps(m, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turbopress",
        description="Certified low-bit LLM weight quantization.",
    )
    parser.add_argument("--version", action="version",
                        version=f"turbopress {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="{compress,validate}")
    _add_compress(sub)
    _add_validate(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
