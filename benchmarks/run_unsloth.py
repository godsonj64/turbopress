"""Unsloth Dynamic 4-bit (pre-quantized bnb) baseline + eval.

Unsloth publishes ready-made "Dynamic" bitsandbytes 4-bit checkpoints
(their Dynamic method keeps sensitive layers in higher precision). This
loads the pre-quantized repo directly through transformers, so it is scored
by the exact same evaluator as every other method in this suite.

    pip install "transformers>=4.51" datasets accelerate bitsandbytes
    python run_unsloth.py --model Qwen/Qwen3-0.6B \
        --quant unsloth/Qwen3-0.6B-unsloth-bnb-4bit
"""

from __future__ import annotations

import argparse
import time

import common_eval as ce
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="fp16 reference repo (the base model)")
    ap.add_argument("--quant", default="unsloth/Qwen3-0.6B-unsloth-bnb-4bit",
                    help="pre-quantized Unsloth Dynamic bnb-4bit repo")
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--save-dir", default="artifacts/unsloth_dynamic_4bit")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, args.batch)

    print(f"\n########## Unsloth Dynamic 4-bit ({args.quant}) ##########", flush=True)

    def load_quant():
        t0 = time.time()
        m = AutoModelForCausalLM.from_pretrained(
            args.quant, device_map=args.device, dtype=torch.float16
        ).eval()
        load_quant.seconds = time.time() - t0
        try:  # serialize so we report a real on-disk size
            m.save_pretrained(args.save_dir)
        except Exception as e:  # noqa: BLE001
            print("  (could not save_pretrained for size accounting:", e, ")")
        return m

    metrics = ce.run_eval(args.model, load_quant, batches, args.device)
    ce.save_result(args.results_dir, "unsloth-dynamic-bnb", 4, args.model, metrics,
                   size_mb=ce.dir_size_mb(args.save_dir),
                   seconds=getattr(load_quant, "seconds", None))


if __name__ == "__main__":
    main()
