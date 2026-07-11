"""Shared, dependency-light evaluation for the quantization benchmark.

Every method script imports THIS module so all methods are scored identically:
the same eval slice, the same fp16 reference, the same metrics (KL, top-1
agreement, perplexity). Only depends on torch + transformers + datasets, so it
installs cleanly alongside any single quantization backend.

Metrics (all vs the fp16 model, on held-out WikiText-2 test text):
  * mean_kl      : mean token-level KL(fp || quant) in nats  (lower = better)
  * top1         : fraction of positions where argmax matches fp16
  * ppl_q/ppl_fp : perplexity of the quantized / fp16 model on the same tokens
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def build_eval(tokenizer, n_seqs: int = 32, seqlen: int = 512, batch: int = 4):
    """Contiguous [batch, seqlen] blocks from WikiText-2 *test* (the eval slice)."""
    import datasets

    wt = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in wt["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    need = n_seqs * seqlen
    if ids.numel() < need:
        raise ValueError(f"eval text too short: {ids.numel()} < {need}")
    ids = ids[:need].reshape(n_seqs, seqlen)
    return [ids[i : i + batch] for i in range(0, n_seqs, batch)]


def calib_texts(n: int = 128, min_words: int = 24, split: str = "train") -> list[str]:
    """Disjoint calibration strings from WikiText-2 *train* (never the eval slice)."""
    import datasets

    wt = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    out = [t.strip() for t in wt["text"] if len(t.split()) >= min_words]
    return out[:n]


@torch.no_grad()
def evaluate_vs_fp16(fp_model, q_model, batches, device: str = "cuda") -> dict:
    """KL / top-1 / perplexity of ``q_model`` against ``fp_model`` on ``batches``."""
    fp_model.eval()
    q_model.eval()
    kl = t1 = ce_fp = ce_q = 0.0
    n = 0
    for b in batches:
        b = b.to(device)
        lf = fp_model(input_ids=b).logits.float()[:, :-1]
        lq = q_model(input_ids=b).logits.float()[:, :-1]
        pf = lf.log_softmax(-1)
        pq = lq.log_softmax(-1)
        lab = b[:, 1:]
        kl += (pf.exp() * (pf - pq)).sum(-1).sum().item()
        t1 += (lf.argmax(-1) == lq.argmax(-1)).sum().item()
        ce_fp += F.nll_loss(pf.reshape(-1, pf.shape[-1]), lab.reshape(-1), reduction="sum").item()
        ce_q += F.nll_loss(pq.reshape(-1, pq.shape[-1]), lab.reshape(-1), reduction="sum").item()
        n += lab.numel()
    return {
        "mean_kl": kl / n,
        "top1": t1 / n,
        "ppl_fp": math.exp(ce_fp / n),
        "ppl_q": math.exp(ce_q / n),
        "tokens": n,
    }


def run_eval(model_id: str, load_quant, batches, device: str = "cuda") -> dict:
    """Load a *fresh* fp16 reference, build the quant model via ``load_quant()``,
    evaluate, then free both. Keeps peak memory to one fp16 + one quant model."""
    from transformers import AutoModelForCausalLM

    fp = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    q = load_quant()
    metrics = evaluate_vs_fp16(fp, q, batches, device)
    del fp, q
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return metrics


def dir_size_mb(path) -> float | None:
    """On-disk size (MiB) of a file or directory, or None if it does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    if p.is_file():
        return round(p.stat().st_size / 2**20, 1)
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20, 1)


def save_result(results_dir, method: str, nominal_bits, model_id: str,
                metrics: dict, size_mb=None, seconds=None) -> dict:
    """Write one method's result to ``results_dir/<method>_<bits>bit.json``."""
    rec = {
        "method": method,
        "nominal_bits": nominal_bits,
        "model": model_id,
        "size_mb": size_mb,
        "seconds": None if seconds is None else round(seconds, 1),
        "mean_kl": round(metrics["mean_kl"], 4),
        "top1": round(metrics["top1"], 4),
        "ppl_q": round(metrics["ppl_q"], 3),
        "ppl_fp": round(metrics["ppl_fp"], 3),
        "tokens": metrics["tokens"],
    }
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{method}_{nominal_bits}bit.json"
    f.write_text(json.dumps(rec, indent=2))
    print(f"\n[saved] {f}\n{json.dumps(rec, indent=2)}", flush=True)
    return rec
