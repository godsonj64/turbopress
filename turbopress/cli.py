"""Command-line interface for TurboPress.

Subcommands:
    turbopress compress <model> --bits 3        run the full quantization pipeline
    turbopress ablate <model> --bits 3          compare methods (equilibration /
                                                error feedback / trellis) on one model
    turbopress validate <ref> <candidate>       measure KL / top-1 / perplexity
    turbopress run <artifact>                   generate text from a packed artifact

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


def _add_run(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "run",
        help="generate text from a TurboPress artifact, running from the packed bits",
        description="Load a TurboPress artifact directory and generate greedily. "
        "--mode 'cached' decodes to fp16 at load (works on CPU or GPU); 'tiled' and "
        "'triton' keep the weights trellis-coded in memory (~bits/16 of fp16). "
        "--cuda-graph captures one decode step and replays it -- the fast packed "
        "path (needs a CUDA device).",
    )
    p.add_argument("artifact", help="path to the TurboPress artifact directory")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-new", type=int, default=64, help="tokens to generate")
    p.add_argument("--mode", default="cached", choices=["cached", "tiled", "triton"],
                   help="cached = fp16 at load; tiled/triton keep weights packed")
    p.add_argument("--cuda-graph", action="store_true",
                   help="CUDA-graph decode: the fast packed path (needs a CUDA device)")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.set_defaults(func=_run_run)


def _add_ablate(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ablate",
        help="measure how much equilibration / error feedback / trellis coding "
        "help on YOUR model",
        description="Quantize a model several ways on ONE shared calibration and "
        "eval slice and report KL / top-1 / perplexity for each, so you can see "
        "which methods actually help before committing. Methods: 'nearest' "
        "(rotated scalar, no equilibration), 'scalar' (+ quarter-power "
        "equilibration), 'gptq' (scalar + equilibration + GPTQ/LDLQ Hessian error "
        "feedback), 'tcq' (trellis coding + equilibration -- the path `compress` "
        "ships). This is a measurement: no packed artifact is written.",
    )
    p.add_argument("model", help="Hugging Face model id or local path")
    p.add_argument("--bits", type=int, default=3, help="bits/weight, 2..8 (default: 3)")
    p.add_argument(
        "--methods", default="nearest,scalar,gptq,tcq",
        help="comma-separated subset of {nearest,scalar,gptq,tcq} "
        "(default: nearest,scalar,gptq,tcq)",
    )
    p.add_argument("--n-states", type=int, default=64, choices=[4, 8, 16, 64],
                   help="trellis states for the tcq method (default: 64)")
    p.add_argument("--seqs", type=int, default=16, help="eval sequences")
    p.add_argument("--calib-seqs", type=int, default=8, help="calibration sequences")
    p.add_argument("--seqlen", type=int, default=256, help="tokens per sequence")
    p.add_argument("--batch", type=int, default=4, help="sequences per forward")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--dtype", default="float16",
                   choices=["float16", "bfloat16", "float32"])
    p.add_argument("--data-dir", default="data", help="cache dir for eval text")
    p.add_argument("--out", default=None, help="write the report to this JSON file")
    p.set_defaults(func=_run_ablate)


def _run_ablate(args: argparse.Namespace) -> int:
    import json as _json
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turbopress.real_model import ablation_configs, load_eval_batches, run_ablation

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    configs = ablation_configs(tuple(methods), bits=args.bits, n_states=args.n_states)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(args.seed)
    print(f"Loading {args.model} ({args.dtype}, {device})...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp_model = (
        AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
        .to(device)
        .eval()
    )

    data_dir = Path(args.data_dir)
    eval_batches = [
        b.to(device)
        for b in load_eval_batches(tokenizer, args.seqs, args.seqlen, args.batch, data_dir)
    ]
    calib_batches = None
    if any(c.equilibrate or c.error_feedback for c in configs):
        calib_batches = [
            b.to(device)
            for b in load_eval_batches(
                tokenizer, args.calib_seqs, args.seqlen, args.batch, data_dir,
                offset_tokens=args.seqs * args.seqlen,  # disjoint from eval slice
            )
        ]
    print(f"Eval: {args.seqs}x{args.seqlen} tokens | methods: {', '.join(methods)}",
          flush=True)

    results = run_ablation(
        fp_model, configs, eval_batches, calib_batches=calib_batches,
        seed=args.seed, verbose=True,
    )

    header = f"{'config':<22} {'bits/w':>7} {'KL(fp||q)':>10} {'top1':>7} {'ppl':>10}"
    print(f"\nfp16 reference perplexity: {results[0]['ppl_fp']:.4f}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['label']:<22} {r['bits_per_weight']:>7.3f} {r['mean_kl']:>10.4f} "
            f"{r['top1_agreement']:>7.3f} {r['ppl_q']:>10.3f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_json.dumps(
            {"model": args.model, "bits": args.bits, "methods": methods,
             "ppl_fp": results[0]["ppl_fp"], "results": results}, indent=2))
        print(f"\nreport written to {out_path}")
    return 0


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


def _cuda_graph_generate(model, input_ids, max_new: int):
    """Greedy decode by capturing one decode step in a CUDA graph and replaying it.

    Prefill runs eagerly; the single-token step is captured once against a
    fixed-size StaticCache, so Triton's per-launch dispatch is paid at capture
    instead of every step. Greedy, so it matches the eager loop token-for-token.
    """
    import torch
    from transformers import StaticCache

    dev = model.device
    prompt_len = input_ids.shape[1]
    kw = dict(max_cache_len=prompt_len + max_new + 16, device=dev, dtype=model.dtype)
    try:  # StaticCache kwarg name changed across transformers versions
        cache = StaticCache(config=model.config, max_batch_size=1, **kw)
    except TypeError:
        cache = StaticCache(config=model.config, batch_size=1, **kw)

    logits = model(
        input_ids=input_ids, past_key_values=cache, use_cache=True,
        cache_position=torch.arange(prompt_len, device=dev),
    ).logits
    tok = logits[:, -1:].argmax(-1).clone()
    pos = torch.tensor([prompt_len], device=dev, dtype=torch.long)
    gen = [tok.clone()]

    def step():
        return model(
            input_ids=tok, past_key_values=cache, use_cache=True, cache_position=pos,
        ).logits

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            tok.copy_(step()[:, -1:].argmax(-1)); pos.add_(1); gen.append(tok.clone())
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_logits = step()
    torch.cuda.synchronize()

    while len(gen) < max_new:
        graph.replay()
        tok.copy_(graph_logits[:, -1:].argmax(-1)); pos.add_(1); gen.append(tok.clone())
    torch.cuda.synchronize()
    return torch.cat([input_ids] + gen[:max_new], dim=1).cpu()


def _run_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    import torch
    from transformers import AutoTokenizer

    from turbopress.runtime import load_packed_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.cuda_graph and not device.startswith("cuda"):
        print("error: --cuda-graph requires a CUDA device", file=sys.stderr)
        return 2

    tokenizer = AutoTokenizer.from_pretrained(Path(args.artifact) / "tokenizer")
    model, meta = load_packed_model(args.artifact, device=device, mode=args.mode)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    if args.cuda_graph:
        out = _cuda_graph_generate(model, ids, args.max_new)
    else:
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)

    decode = "cuda-graph" if args.cuda_graph else "eager"
    print(f"\n{meta['model_id']} @ {meta['bits']}b | mode={args.mode} | {decode} decode\n")
    print(tokenizer.decode(out[0], skip_special_tokens=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turbopress",
        description="Certified low-bit LLM weight quantization.",
    )
    parser.add_argument("--version", action="version",
                        version=f"turbopress {__version__}")
    sub = parser.add_subparsers(dest="command",
                                metavar="{compress,ablate,validate,run}")
    _add_compress(sub)
    _add_ablate(sub)
    _add_validate(sub)
    _add_run(sub)
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
