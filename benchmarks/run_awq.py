"""Real AWQ via AutoAWQ — INT4 (its native regime) + eval.

AWQ is a 4-bit group-wise method; it does not target 2/3-bit, so this runs 4-bit
only. NOTE: autoawq is archived upstream and is version-sensitive with recent
transformers / newer architectures — if it fails to install or quantize Qwen3,
run it in its own fresh environment, or drop it from the comparison.

    pip install "transformers>=4.51" datasets accelerate autoawq
    python run_awq.py --model Qwen/Qwen3-1.7B --group-size 128
"""

from __future__ import annotations

import argparse
import gc
import time

import common_eval as ce
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--save-dir", default="artifacts/awq_int4")
    args = ap.parse_args()

    from awq import AutoAWQForCausalLM

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, args.batch)

    print("\n########## AWQ INT4 g%d ##########" % args.group_size, flush=True)
    t0 = time.time()
    q = AutoAWQForCausalLM.from_pretrained(args.model)
    q.quantize(tok, quant_config={
        "zero_point": True, "q_group_size": args.group_size,
        "w_bit": 4, "version": "GEMM",
    })
    q.save_quantized(args.save_dir)
    tok.save_pretrained(args.save_dir)
    seconds = time.time() - t0
    del q
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()

    def load_quant():
        return AutoModelForCausalLM.from_pretrained(
            args.save_dir, device_map=args.device
        ).eval()

    metrics = ce.run_eval(args.model, load_quant, batches, args.device)
    ce.save_result(args.results_dir, "awq", 4, args.model, metrics,
                   size_mb=ce.dir_size_mb(args.save_dir), seconds=seconds)


if __name__ == "__main__":
    main()
