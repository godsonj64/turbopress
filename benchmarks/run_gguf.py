"""GGUF (llama.cpp) quality eval against the shared fp16 reference.

Feeds the SAME HF-tokenized WikiText-2 eval tokens into llama.cpp (its
tokenizer is bypassed entirely), so KL / top-1 / perplexity are computed on
identical positions to every other method in this suite. Include the BF16
GGUF as a control row: its KL vs the HF fp16 reference isolates cross-engine
numeric noise (RoPE/kernel differences), which every quantized row inherits.

    pip install llama-cpp-python huggingface_hub
    python run_gguf.py --repo unsloth/Qwen3-0.6B-GGUF \
        --files Qwen3-0.6B-BF16.gguf,Qwen3-0.6B-UD-Q4_K_XL.gguf,Qwen3-0.6B-UD-IQ2_M.gguf
"""

from __future__ import annotations

import argparse
import math
import re
import time
import traceback

import common_eval as ce
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def ref_logits(model_id: str, seqs: torch.Tensor, device: str) -> torch.Tensor:
    """fp16 reference logits [n_seqs, L, V], stored fp16 on CPU."""
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16)
    m = m.to(device).eval()
    outs = []
    with torch.no_grad():
        for i in range(seqs.shape[0]):
            lg = m(input_ids=seqs[i : i + 1].to(device)).logits
            outs.append(lg.to(torch.float16).cpu())
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(outs, 0)


def nominal_bits(fname: str) -> str:
    m = re.search(r"(?:UD-)?(?:I?Q|BF|F)(\d+)", fname)
    return m.group(1) if m else "x"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="HF repo for the fp16 reference + tokenizer")
    ap.add_argument("--repo", default="unsloth/Qwen3-0.6B-GGUF")
    ap.add_argument("--files", required=True, help="comma list of .gguf filenames")
    ap.add_argument("--eval-seqs", type=int, default=16)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="offload layers to GPU (needs a CUDA llama.cpp build)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    tok = AutoTokenizer.from_pretrained(args.model)
    batches = ce.build_eval(tok, args.eval_seqs, args.seqlen, batch=1)
    seqs = torch.cat(batches, 0)  # [n_seqs, L]
    print(f"reference logits: {args.model} on {args.device} "
          f"({seqs.shape[0]}x{seqs.shape[1]} tokens)", flush=True)
    ref = ref_logits(args.model, seqs, args.device)  # [n, L, V] fp16 cpu

    for fname in args.files.split(","):
        fname = fname.strip()
        try:
            print(f"\n########## GGUF {fname} ##########", flush=True)
            path = hf_hub_download(args.repo, fname)
            t0 = time.time()
            llm = Llama(model_path=path, n_ctx=args.seqlen, logits_all=True,
                        n_gpu_layers=args.n_gpu_layers, verbose=False)
            kl = t1 = ce_fp = ce_q = 0.0
            n = 0
            for i in range(seqs.shape[0]):
                tokens = seqs[i].tolist()
                llm.reset()
                llm.eval(tokens)
                q = torch.from_numpy(
                    np.asarray(llm.scores[: len(tokens)], dtype=np.float32).copy()
                )
                r = ref[i].float()
                v = min(q.shape[-1], r.shape[-1])
                pf = r[:-1, :v].log_softmax(-1)
                pq = q[:-1, :v].log_softmax(-1)
                lab = torch.tensor(tokens[1:])
                kl += float((pf.exp() * (pf - pq)).sum(-1).sum())
                t1 += float((pf.argmax(-1) == pq.argmax(-1)).sum())
                ce_fp += float(F.nll_loss(pf, lab, reduction="sum"))
                ce_q += float(F.nll_loss(pq, lab, reduction="sum"))
                n += len(tokens) - 1
            del llm
            metrics = {
                "mean_kl": kl / n,
                "top1": t1 / n,
                "ppl_fp": math.exp(ce_fp / n),
                "ppl_q": math.exp(ce_q / n),
                "tokens": n,
            }
            method = fname.replace(".gguf", "").replace("Qwen3-0.6B-", "gguf-")
            ce.save_result(args.results_dir, method, nominal_bits(fname),
                           args.model, metrics, size_mb=ce.dir_size_mb(path),
                           seconds=time.time() - t0)
        except Exception as e:  # noqa: BLE001 - one bad file shouldn't drop the rest
            traceback.print_exc()
            print(f"  [{fname}] FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
