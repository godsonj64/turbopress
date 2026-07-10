"""Fidelity harness: measure how close one model's outputs are to another's.

``turbopress validate <reference> <candidate>`` loads two ``transformers``
causal LMs that share a tokenizer/vocabulary (typically a full-precision model
and a quantized copy of it -- TurboPress, GPTQ, AWQ, or any HF-loadable
checkpoint) and reports token-level KL(reference || candidate), top-1
next-token agreement, and the perplexity of both on held-out real text.

This is deliberately method-agnostic: the reference and candidate can come from
any source, so the same numbers can be quoted across quantizers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from turbopress.real_model import evaluate_pair, load_eval_batches

__all__ = ["validate_models"]


def validate_models(
    reference: str,
    candidate: str,
    *,
    seqs: int = 16,
    seqlen: int = 256,
    batch: int = 4,
    device: str | None = None,
    dtype: str = "float16",
    data_dir: str = "data",
    out: str | None = None,
) -> dict:
    """Compare ``candidate`` against ``reference`` on held-out text.

    Both are loaded with ``AutoModelForCausalLM`` and must share the reference's
    tokenizer and vocabulary. Returns the metrics dict (mean_kl, top1_agreement,
    ppl_fp, ppl_q, n_tokens); if ``out`` is given, also writes it as JSON.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype)
    dev = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(reference)
    ref_model = (
        AutoModelForCausalLM.from_pretrained(reference, dtype=torch_dtype).to(dev).eval()
    )
    cand_model = (
        AutoModelForCausalLM.from_pretrained(candidate, dtype=torch_dtype).to(dev).eval()
    )

    batches = [
        b.to(dev)
        for b in load_eval_batches(tokenizer, seqs, seqlen, batch, Path(data_dir))
    ]
    metrics = evaluate_pair(ref_model, cand_model, batches)
    result = {
        "reference": reference,
        "candidate": candidate,
        "eval": {"seqs": seqs, "seqlen": seqlen, "n_tokens": metrics["n_tokens"]},
        "metrics": metrics,
    }
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2))
    return result
