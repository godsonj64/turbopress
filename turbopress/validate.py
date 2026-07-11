"""Fidelity harness: measure how close one model's outputs are to another's.

``turbopress validate <reference> <candidate>`` loads two causal LMs that share
a tokenizer/vocabulary (typically a full-precision model and a quantized copy of
it -- TurboPress, GPTQ, AWQ, or any HF-loadable checkpoint) and reports
token-level KL(reference || candidate), top-1 next-token agreement, and the
perplexity of both on held-out real text.

Each side is loaded independently: a plain HF id/path goes through
``AutoModelForCausalLM``, and a TurboPress artifact directory goes through the
standalone ``run_quantized.py`` loader bundled inside it -- the same loader the
``compress`` self-test certifies, so the numbers here match the packed model.
This keeps the harness method-agnostic: reference and candidate can come from
any source, so the same metrics can be quoted across quantizers.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path

import torch

from turbopress.real_model import evaluate_pair, load_eval_batches

__all__ = ["validate_models"]


def _is_turbopress_artifact(path: str) -> bool:
    """True if ``path`` is a TurboPress artifact directory.

    An artifact carries packed weights (``turbopress_weights.pt``) plus its own
    self-contained loader (``run_quantized.py``); it deliberately has no
    top-level ``config.json``, so ``AutoModelForCausalLM`` cannot load it.
    """
    p = Path(path)
    return (
        p.is_dir()
        and (p / "turbopress_weights.pt").exists()
        and (p / "run_quantized.py").exists()
    )


def _load_artifact_model(path: str, device: str, torch_dtype: torch.dtype):
    """Load a TurboPress artifact via its bundled ``run_quantized.py``.

    Imports the artifact's own loader (rather than duplicating the decode here)
    so the decoded weights are bit-identical to what ``compress`` self-tested.
    The loader prints a line per decoded matrix; that chatter is swallowed so
    the validate report stays readable.
    """
    art_dir = Path(path)
    module_name = f"tp_artifact_loader_{abs(hash(str(art_dir.resolve())))}"
    spec = importlib.util.spec_from_file_location(
        module_name, art_dir / "run_quantized.py"
    )
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)
    with contextlib.redirect_stdout(io.StringIO()):
        return rt.load_quantized_model(art_dir, device=device, dtype=torch_dtype)


def _load_model(path: str, device: str, torch_dtype: torch.dtype):
    """Load ``path`` as a causal LM: TurboPress artifact or any HF checkpoint."""
    if _is_turbopress_artifact(path):
        return _load_artifact_model(path, device, torch_dtype)
    from transformers import AutoModelForCausalLM

    return (
        AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype).to(device).eval()
    )


def _tokenizer_source(path: str) -> str:
    """Where to load the shared tokenizer from.

    A TurboPress artifact stores its tokenizer in a ``tokenizer/`` subdir; a
    plain HF id/path serves the tokenizer directly.
    """
    if _is_turbopress_artifact(path):
        return str(Path(path) / "tokenizer")
    return path


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

    Each model is loaded independently -- an ``AutoModelForCausalLM`` checkpoint
    or a TurboPress artifact directory -- and both must share the reference's
    tokenizer and vocabulary. Returns the metrics dict (mean_kl, top1_agreement,
    ppl_fp, ppl_q, n_tokens); if ``out`` is given, also writes it as JSON.
    """
    from transformers import AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = getattr(torch, dtype)
    dev = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_source(reference))
    ref_model = _load_model(reference, device, torch_dtype)
    cand_model = _load_model(candidate, device, torch_dtype)

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
