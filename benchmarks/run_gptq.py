"""Real GPTQ via GPTQModel — aggressive 2/3/4-bit + eval.

GPTQModel (the maintained successor to AutoGPTQ) supports 2/3/4/8-bit and modern
architectures including Qwen3. Calibration uses a disjoint WikiText-2 train slice.

    pip install "transformers>=4.51" datasets accelerate gptqmodel
    python run_gptq.py --model Qwen/Qwen3-1.7B --bits 2,3,4 --group-size 128
"""

from __future__ import annotations

import argparse
import gc
import time
import traceback

import torch
from transformers import AutoTokenizer

import common_eval as ce


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--bits", default="2,3,4", help="comma list from {2,3,4,8}")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib", type=int, default=128, help="calibration samples")
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--save-dir", default="artifacts/gptq")
    args = ap.parse_args()

    from gptqmodel import GPTQModel, QuantizeConfig

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, args.batch)
    calib = ce.calib_texts(args.calib)

    for bits in [int(b) for b in args.bits.split(",")]:
        print(f"\n########## GPTQ {bits}-bit g{args.group_size} ##########", flush=True)
        save_dir = f"{args.save_dir}_{bits}bit"
        try:
            t0 = time.time()
            qc = QuantizeConfig(bits=bits, group_size=args.group_size)
            m = GPTQModel.load(args.model, qc)
            m.quantize(calib)
            m.save(save_dir)
            seconds = time.time() - t0
            del m
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()

            def load_quant(save_dir=save_dir):
                return GPTQModel.load(save_dir).model.eval()

            metrics = ce.run_eval(args.model, load_quant, batches, args.device)
            ce.save_result(args.results_dir, "gptq", bits, args.model, metrics,
                           size_mb=ce.dir_size_mb(save_dir), seconds=seconds)
        except Exception as e:  # noqa: BLE001 - one bad width shouldn't drop the rest
            traceback.print_exc()
            print(f"  [gptq {bits}-bit] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
