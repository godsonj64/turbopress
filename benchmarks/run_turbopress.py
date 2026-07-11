"""TurboPress (real TCQ pipeline) — aggressive low-bit compression + eval.

Runs `turbopress compress` at each requested bit-width, then scores the reloaded
artifact with the shared evaluator.

    pip install "turbopress[llm]>=0.4.3" "transformers>=4.51" datasets accelerate
    python run_turbopress.py --model Qwen/Qwen3-1.7B --bits 2,3,4
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from transformers import AutoTokenizer

import common_eval as ce


def load_turbopress_artifact(art_dir: str, device: str):
    loader_path = str(Path(art_dir) / "run_quantized.py")
    spec = importlib.util.spec_from_file_location("tp_runtime", loader_path)
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    return rt.load_quantized_model(art_dir, device=device).eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--bits", default="2,3,4", help="comma list, each in [2,6]")
    ap.add_argument("--eval-seqs", type=int, default=32)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--work-dir", default="artifacts")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, args.batch)

    for bits in [int(b) for b in args.bits.split(",")]:
        try:
            art_root = Path(args.work_dir) / f"turbopress_{bits}bit"
            print(f"\n########## TurboPress TCQ {bits}-bit ##########", flush=True)
            t0 = time.time()
            subprocess.run(
                [sys.executable, "-m", "turbopress.cli", "compress", args.model,
                 "--bits", str(bits), "--no-self-test", "--out", str(art_root)],
                check=True,
            )
            art_dir = glob.glob(str(art_root / "*-turbopress-*"))[0]
            seconds = time.time() - t0
            metrics = ce.run_eval(
                args.model, lambda d=art_dir: load_turbopress_artifact(d, args.device),
                batches, args.device,
            )
            ce.save_result(args.results_dir, "turbopress", bits, args.model, metrics,
                           size_mb=ce.dir_size_mb(art_dir), seconds=seconds)
        except Exception as e:  # noqa: BLE001 - one bad width shouldn't drop the rest
            traceback.print_exc()
            print(f"  [turbopress {bits}-bit] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
