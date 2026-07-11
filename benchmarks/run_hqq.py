"""HQQ (Half-Quadratic Quantization) — aggressive 2/3/4-bit + eval.

HQQ is calibration-free and supports very low bit-widths, so it's a strong
aggressive baseline. group_size defaults to 64 (smaller groups = better low-bit
accuracy at some size cost).

    pip install "transformers>=4.51" datasets accelerate hqq
    python run_hqq.py --model Qwen/Qwen3-1.7B --bits 2,3,4 --group-size 64
"""

from __future__ import annotations

import argparse
import time
import traceback

import common_eval as ce
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HqqConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--bits", default="2,3,4", help="comma list from {1,2,3,4,8}")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--save-dir", default="artifacts/hqq")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, args.batch)

    for bits in [int(b) for b in args.bits.split(",")]:
        print(f"\n########## HQQ {bits}-bit g{args.group_size} ##########", flush=True)
        save_dir = f"{args.save_dir}_{bits}bit"

        def load_quant(bits=bits, save_dir=save_dir):
            t0 = time.time()
            cfg = HqqConfig(nbits=bits, group_size=args.group_size)
            m = AutoModelForCausalLM.from_pretrained(
                args.model, quantization_config=cfg, dtype=torch.float16,
                device_map=args.device,
            ).eval()
            load_quant.seconds = time.time() - t0
            try:
                m.save_pretrained(save_dir)
            except Exception as e:  # noqa: BLE001
                print("  (could not save_pretrained for size accounting:", e, ")")
            return m

        try:
            metrics = ce.run_eval(args.model, load_quant, batches, args.device)
            ce.save_result(args.results_dir, "hqq", bits, args.model, metrics,
                           size_mb=ce.dir_size_mb(save_dir),
                           seconds=getattr(load_quant, "seconds", None))
        except Exception as e:  # noqa: BLE001 - one bad width shouldn't drop the rest
            traceback.print_exc()
            print(f"  [hqq {bits}-bit] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
