"""bitsandbytes NF4 (4-bit) baseline + eval.

bitsandbytes is weight-only 4-bit (NF4) — its aggressive floor. Double
quantization is enabled to shave the block-scale overhead.

    pip install "transformers>=4.51" datasets accelerate bitsandbytes
    python run_bnb.py --model Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
import time

import common_eval as ce
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--save-dir", default="artifacts/bnb_nf4")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, args.batch)

    print("\n########## bitsandbytes NF4 (4-bit) ##########", flush=True)

    def load_quant():
        t0 = time.time()
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        m = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=cfg, device_map=args.device
        ).eval()
        load_quant.seconds = time.time() - t0
        try:  # serialize the 4-bit weights so we can report a real on-disk size
            m.save_pretrained(args.save_dir)
        except Exception as e:  # noqa: BLE001
            print("  (could not save_pretrained for size accounting:", e, ")")
        return m

    metrics = ce.run_eval(args.model, load_quant, batches, args.device)
    ce.save_result(args.results_dir, "bitsandbytes-nf4", 4, args.model, metrics,
                   size_mb=ce.dir_size_mb(args.save_dir),
                   seconds=getattr(load_quant, "seconds", None))


if __name__ == "__main__":
    main()
