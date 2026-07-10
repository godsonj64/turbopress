"""TurboPress end-to-end LLM weight-quantization pipeline (packaged).

Takes any medium Hugging Face causal LM (Llama / Qwen / Mistral family),
quantizes every decoder linear to ``BITS`` bits/weight with the TurboPress
pipeline -- seeded randomized rotation -> quarter-power activation
equilibration -> trellis-coded quantization (Viterbi, analytic data-free
codebook) -- then validates against the full-precision model (KL, top-1,
perplexity), logs everything, and writes a downloadable artifact::

    <model>-turbopress-<bits>bit/
      turbopress_weights.pt     packed codes at true bit-width
      run_quantized.py          standalone loader / demo (no repo needed)
      quantization_config.json  settings + measured metrics
      hf_config/  tokenizer/    everything needed to run offline
      turbopress.log            full run log
    + the same folder zipped for download.

This is the importable form of ``scripts/turbopress_onecell.py``; the CLI
(``turbopress compress``) and the one-cell script both call :func:`compress`,
so there is a single validated code path. Requires a single CUDA GPU with
enough VRAM for the model in fp16 (a 4B model needs ~10 GB; validation briefly
holds eval logits on CPU). ``transformers`` + ``torch`` are required;
``datasets`` is auto-installed (falls back to a public-domain text offline).

Config: pass a dict to :func:`compress`, or set env vars TP_MODEL_ID, TP_BITS,
... and call :func:`config_from_env`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["compress", "config_from_env", "DEFAULT_CONFIG", "main"]


def _cfg(name, default):
    v = os.environ.get(f"TP_{name}")
    if v is None:
        return default
    if isinstance(default, bool):
        return v.lower() in ("1", "true", "yes")
    if isinstance(default, int):
        return int(v)
    if isinstance(default, float):
        return float(v)
    return v


def config_from_env() -> dict:
    """Build a config dict from TP_* environment variables (with defaults)."""
    return {
        "MODEL_ID": _cfg("MODEL_ID", "Qwen/Qwen3-4B"),  # any Llama/Qwen/Mistral-family LM
        "BITS": _cfg("BITS", 3),                # 2..6 bits per weight
        "N_STATES": _cfg("N_STATES", 64),       # trellis states: 16 = faster, 64 = best
        "SEQLEN": _cfg("SEQLEN", 512),          # tokens per eval/calibration sequence
        "EVAL_SEQS": _cfg("EVAL_SEQS", 8),      # eval set = EVAL_SEQS x SEQLEN tokens
        "CALIB_SEQS": _cfg("CALIB_SEQS", 8),    # disjoint calibration slice
        "BATCH": _cfg("BATCH", 2),              # sequences per forward
        "SEED": _cfg("SEED", 0),
        "OUT_DIR": _cfg("OUT_DIR", "turbopress_out"),
        "DEVICE": _cfg("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "SELF_TEST": _cfg("SELF_TEST", True),   # reload artifact & verify logits match
        "DEMO_PROMPT": _cfg("DEMO_PROMPT", "The three most important ideas in physics are"),
    }


DEFAULT_CONFIG = config_from_env()

# ----------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------
log = logging.getLogger("turbopress")
log.setLevel(logging.INFO)
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S")
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(_fmt)
    log.addHandler(_h)
else:
    _fmt = log.handlers[0].formatter


# ----------------------------------------------------------------------------
# bit packing (true-rate storage)
# ----------------------------------------------------------------------------
def pack_bits(codes: np.ndarray, width: int) -> np.ndarray:
    """Pack uint8 values (< 2**width) into a flat uint8 buffer, width bits each."""
    assert codes.dtype == np.uint8 and 1 <= width <= 8
    flat = codes.reshape(-1)
    bits = (flat[:, None] >> np.arange(width, dtype=np.uint8)) & 1
    return np.packbits(bits.reshape(-1))


def unpack_bits(buf: np.ndarray, count: int, width: int) -> np.ndarray:
    """Inverse of pack_bits: recover `count` uint8 values of `width` bits."""
    bits = np.unpackbits(buf)[: count * width].reshape(count, width)
    return (bits << np.arange(width, dtype=np.uint8)).sum(axis=1).astype(np.uint8)


# ----------------------------------------------------------------------------
# randomized orthogonal rotation (block fast Walsh-Hadamard x stored signs)
# ----------------------------------------------------------------------------
def fwht(x: Tensor) -> Tensor:
    """Orthonormal FWHT along the last dim (power of 2). H = H^T = H^-1."""
    d = x.shape[-1]
    y = x.reshape(-1, d)
    h = 1
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        even, odd = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((even + odd, even - odd), dim=2).reshape(-1, d)
        h *= 2
    return (y / math.sqrt(d)).reshape(x.shape)


def _block_of(dim: int) -> int:
    block = dim & (-dim)  # largest power-of-2 divisor
    if block < 2:
        raise ValueError(f"in_features={dim} has no power-of-2 factor; unsupported")
    return block


def rotate(x: Tensor, signs: Tensor, block: int) -> Tensor:
    """R x with R = H_block . diag(signs), applied along the last dim."""
    z = x * signs
    if block == x.shape[-1]:
        return fwht(z)
    m = x.shape[-1] // block
    return fwht(z.reshape(*x.shape[:-1], m, block)).reshape(x.shape)


def rotate_inv(x: Tensor, signs: Tensor, block: int) -> Tensor:
    """R^T x (H_block is symmetric, so R^T = diag(signs) . H_block)."""
    if block == x.shape[-1]:
        z = fwht(x)
    else:
        m = x.shape[-1] // block
        z = fwht(x.reshape(*x.shape[:-1], m, block)).reshape(x.shape)
    return z * signs


# ----------------------------------------------------------------------------
# Lloyd-Max codebook for N(0,1) + trellis-coded quantization
# ----------------------------------------------------------------------------
def lloyd_max_gaussian(bits: int, iters: int = 100_000, tol: float = 1e-12) -> Tensor:
    n = 1 << bits
    p = (torch.arange(n, dtype=torch.float64) + 0.5) / n
    c = math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)
    inf = torch.tensor([math.inf], dtype=torch.float64)
    sq2pi, sq2 = math.sqrt(2 * math.pi), math.sqrt(2.0)
    for _ in range(iters):
        t = torch.cat([-inf, (c[:-1] + c[1:]) / 2, inf])
        mass = 0.5 * (torch.erf(t[1:] / sq2) - torch.erf(t[:-1] / sq2))
        phi = torch.exp(-0.5 * t * t) / sq2pi
        c_new = (phi[:-1] - phi[1:]) / mass
        moved = float((c_new - c).abs().max())
        c = c_new
        if moved < tol:
            break
    return c


_TRELLIS_GENERATORS = {4: (0o5, 0o7), 8: (0o15, 0o17), 16: (0o23, 0o35), 64: (0o133, 0o171)}


class Trellis:
    """Shift-register trellis of a rate-1/2 convolutional code (4 subsets)."""

    def __init__(self, n_states: int) -> None:
        if n_states not in _TRELLIS_GENERATORS:
            raise ValueError(f"n_states must be one of {sorted(_TRELLIS_GENERATORS)}")
        self.n_states = n_states
        m = n_states.bit_length() - 1
        g0, g1 = _TRELLIS_GENERATORS[n_states]
        sub = torch.empty(n_states, 2, dtype=torch.int64)
        for s in range(n_states):
            for b in (0, 1):
                x = (b << m) | s
                sub[s, b] = (((x & g1).bit_count() & 1) << 1) | ((x & g0).bit_count() & 1)
        self.subset_table = sub
        ns = torch.arange(n_states)
        self.prev0 = ns >> 1
        self.prev1 = (ns >> 1) | (n_states >> 1)
        bit = ns & 1
        self.sub0 = sub[self.prev0, bit]
        self.sub1 = sub[self.prev1, bit]


def _viterbi(z: Tensor, codebook: Tensor, trellis: Trellis):
    """Min-distortion trellis path per row (start state 0). z: [m, d] fp32 CPU.

    Returns (level_codes, path_bits, member_codes), each [m, d] uint8.
    """
    m, d = z.shape
    S = trellis.n_states
    members = torch.empty(m, d, 4, dtype=torch.int64)
    costs = torch.empty(m, d, 4, dtype=torch.float32)
    for j in range(4):
        sub_cb = codebook[j::4]
        if sub_cb.numel() == 1:
            mem = torch.zeros(m, d, dtype=torch.int64)
        else:
            thr = (sub_cb[:-1] + sub_cb[1:]) / 2
            mem = torch.bucketize(z.contiguous(), thr)
        members[:, :, j] = mem
        costs[:, :, j] = (z - sub_cb[mem]).pow(2)
    alpha = torch.full((m, S), math.inf)
    alpha[:, 0] = 0.0
    bp = torch.empty(m, d, S, dtype=torch.bool)
    for t in range(d):
        c = costs[:, t, :]
        c0 = alpha[:, trellis.prev0] + c[:, trellis.sub0]
        c1 = alpha[:, trellis.prev1] + c[:, trellis.sub1]
        take1 = c1 < c0
        alpha = torch.where(take1, c1, c0)
        bp[:, t, :] = take1
    rows = torch.arange(m)
    state = alpha.argmin(dim=1)
    level = torch.empty(m, d, dtype=torch.uint8)
    pathb = torch.empty(m, d, dtype=torch.uint8)
    memb = torch.empty(m, d, dtype=torch.uint8)
    for t in range(d - 1, -1, -1):
        take1 = bp[rows, t, state]
        prev = torch.where(take1, trellis.prev1[state], trellis.prev0[state])
        sub = torch.where(take1, trellis.sub1[state], trellis.sub0[state])
        mem = members[rows, t, sub]
        pathb[:, t] = (state & 1).to(torch.uint8)
        memb[:, t] = mem.to(torch.uint8)
        level[:, t] = (mem * 4 + sub).to(torch.uint8)
        state = prev
    return level, pathb, memb


def viterbi_chunked(z: Tensor, codebook: Tensor, trellis: Trellis, budget_bytes=2 << 30):
    """Row-chunked Viterbi so backpointer memory stays under `budget_bytes`."""
    m, d = z.shape
    per_row = d * (trellis.n_states + 48)  # bp bool + members/costs
    chunk = max(16, min(m, budget_bytes // max(per_row, 1)))
    outs = [
        _viterbi(z[i : i + chunk], codebook, trellis) for i in range(0, m, chunk)
    ]
    return tuple(torch.cat(parts, dim=0) for parts in zip(*outs))


_OPT_CB_CACHE: dict[tuple[int, int], Tensor] = {}


def tcq_codebook(bits: int, n_states: int, iters: int = 6, seed: int = 0) -> Tensor:
    """Doubled Lloyd-Max codebook re-optimized *under the trellis* on synthetic
    N(0,1) samples (generalized Lloyd with Viterbi assignments). Data-free."""
    key = (bits, n_states)
    if key in _OPT_CB_CACHE:
        return _OPT_CB_CACHE[key].clone()
    trellis = Trellis(n_states)
    cb = lloyd_max_gaussian(bits + 1).to(torch.float32)
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(8, 8192, generator=gen)
    for _ in range(iters):
        lc, _, _ = _viterbi(z, cb, trellis)
        flat_c, flat_z = lc.long().flatten(), z.flatten()
        sums = torch.zeros(cb.numel()).index_add_(0, flat_c, flat_z)
        cnts = torch.zeros(cb.numel()).index_add_(0, flat_c, torch.ones_like(flat_z))
        cb = torch.where(cnts > 0, sums / cnts.clamp_min(1.0), cb)
        for j in range(4):
            cb[j::4] = cb[j::4].sort().values
    _OPT_CB_CACHE[key] = cb
    return cb.clone()


def decode_levels(path_bits: Tensor, member_codes: Tensor, trellis: Trellis) -> Tensor:
    """Reconstruct level codes from the stored bit-stream (start state 0)."""
    n, d = path_bits.shape
    out = torch.empty(n, d, dtype=torch.uint8)
    state = torch.zeros(n, dtype=torch.int64)
    mask = trellis.n_states - 1
    for t in range(d):
        b = path_bits[:, t].long()
        sub = trellis.subset_table[state, b]
        out[:, t] = (member_codes[:, t].long() * 4 + sub).to(torch.uint8)
        state = ((state << 1) | b) & mask
    return out


# ----------------------------------------------------------------------------
# per-matrix quantization: equilibrate -> rotate -> TCQ -> pack + reconstruct
# ----------------------------------------------------------------------------
def quantize_matrix(w: Tensor, act_rms: Tensor, bits: int, n_states: int, seed: int):
    """Returns (payload dict for the artifact, reconstructed fp32 weight)."""
    w32 = w.detach().to(torch.float32).cpu()
    n, d = w32.shape
    # quarter-power equilibration: s_j = (E[x_j^2] / ||W_:,j||^2)^(1/4).
    # s is rounded to its stored fp16 precision *before* use, so the validated
    # model is bit-identical to what the artifact reconstructs.
    act = act_rms.to(torch.float32).cpu()
    act = act.clamp_min(max(0.05 * float(act.pow(2).mean().sqrt()), 1e-8))
    col_norm = w32.norm(dim=0)
    col_norm = col_norm.clamp_min(max(0.05 * float(col_norm.pow(2).mean().sqrt()), 1e-12))
    s = (act / col_norm).sqrt().to(torch.float16).float()
    w_eq = w32 * s[None, :]
    # seeded rotation (signs are STORED, so reproducibility never depends on RNG)
    block = _block_of(d)
    gen = torch.Generator().manual_seed(seed)
    signs = (torch.randint(0, 2, (d,), generator=gen, dtype=torch.int64) * 2 - 1).float()
    w_rot = rotate(w_eq, signs, block)
    # per-row scale + trellis encode + one least-squares scale refit
    scales = w_rot.pow(2).mean(dim=1).sqrt()
    nz = scales > 0
    safe = torch.where(nz, scales, torch.ones_like(scales))
    cb = tcq_codebook(bits, n_states).to(torch.float16).float()  # stored precision
    level, pathb, memb = viterbi_chunked(w_rot / safe[:, None], cb, Trellis(n_states))
    q = cb[level.long()]
    num, den = (w_rot * q).sum(dim=1), (q * q).sum(dim=1)
    refit = torch.where(den > 0, num / den.clamp_min(1e-30), safe)
    safe = torch.where(refit > 0, refit, safe)
    scales = torch.where(nz, safe, torch.zeros_like(safe)).to(torch.float16).float()
    # reconstruct in the original basis: W ~= (scale * levels) rotated back, / s.
    # All factors are already at stored fp16 precision, so this equals the
    # loader's reconstruction exactly.
    w_hat = rotate_inv(scales[:, None] * q, signs, block) / s[None, :]
    payload = {
        "n": n, "d": d, "block": block, "bits": bits, "n_states": n_states,
        "path_bits": torch.from_numpy(pack_bits(pathb.numpy(), 1)),
        "member_bits": (
            torch.from_numpy(pack_bits(memb.numpy(), bits - 1)) if bits > 1 else None
        ),
        "scales": scales.to(torch.float16),
        "signs": torch.from_numpy(pack_bits(((signs > 0).to(torch.uint8)).numpy(), 1)),
        "equil": s.to(torch.float16),
        "codebook": cb.to(torch.float16),
    }
    return payload, w_hat


# ----------------------------------------------------------------------------
# model plumbing: decoder layers, calibration hooks, evaluation
# ----------------------------------------------------------------------------
def decoder_layers(model: nn.Module):
    for path in ("model.layers", "transformer.h", "model.decoder.layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj, path
    raise ValueError(f"cannot locate decoder layers on {type(model).__name__}")


@torch.no_grad()
def collect_channel_stats(model, layers, batches, device):
    """Per-channel sqrt(E[x_j^2]) for every decoder nn.Linear, via hooks."""
    sums, counts, handles = {}, {}, []

    def mk(key):
        def hook(_m, inputs, _o):
            x = inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1])
            if key in sums:
                sums[key] += x.pow(2).sum(dim=0)
            else:
                sums[key] = x.pow(2).sum(dim=0)
            counts[key] = counts.get(key, 0) + x.shape[0]
        return hook

    for i, block in enumerate(layers):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(mk(f"{i}.{name}")))
    try:
        for ids in batches:
            model(input_ids=ids.to(device))
    finally:
        for h in handles:
            h.remove()
    return {k: (v / counts[k]).sqrt().cpu() for k, v in sums.items()}


@torch.no_grad()
def eval_logits(model, batches, device):
    """Next-token logits (fp16, CPU) and labels for the eval set."""
    outs, labels = [], []
    for ids in batches:
        ids = ids.to(device)
        lg = model(input_ids=ids).logits[:, :-1]
        outs.append(lg.reshape(-1, lg.shape[-1]).to(torch.float16).cpu())
        labels.append(ids[:, 1:].reshape(-1).cpu())
    return torch.cat(outs), torch.cat(labels)


def metrics(fp_logits, q_logits, labels, chunk=1024):
    kl = top1 = ce_fp = ce_q = 0.0
    T = fp_logits.shape[0]
    for i in range(0, T, chunk):
        a = fp_logits[i : i + chunk].float()
        b = q_logits[i : i + chunk].float()
        lab = labels[i : i + chunk]
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        kl += float((la.exp() * (la - lb)).sum(-1).sum())
        top1 += float((a.argmax(-1) == b.argmax(-1)).sum())
        ce_fp += float(F.nll_loss(la, lab, reduction="sum"))
        ce_q += float(F.nll_loss(lb, lab, reduction="sum"))
    return {
        "mean_kl": kl / T,
        "top1_agreement": top1 / T,
        "ppl_fp": math.exp(ce_fp / T),
        "ppl_q": math.exp(ce_q / T),
        "n_tokens": T,
    }


def get_eval_text() -> str:
    """Real text for eval/calibration.

    Tries WikiText-2 from the HF Hub under its canonical id first (works in
    HF-only-egress environments like Colab/managed notebooks; the bare id
    'wikitext' is rejected by newer datasets/huggingface_hub versions), then
    the legacy id, then Project Gutenberg on the open internet.
    """
    errors = []
    try:
        try:
            from datasets import load_dataset
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "datasets"]
            )
            from datasets import load_dataset
        for repo in ("Salesforce/wikitext", "wikitext"):
            try:
                ds_test = load_dataset(repo, "wikitext-2-raw-v1", split="test")
                ds_train = load_dataset(repo, "wikitext-2-raw-v1", split="train[:2%]")
                log.info(f"eval/calibration text: {repo} (wikitext-2-raw-v1)")
                return "\n".join(ds_test["text"]) + "\n" + "\n".join(ds_train["text"])
            except Exception as exc:  # noqa: BLE001 - try the next source
                errors.append(f"{repo}: {str(exc)[:200]}")
    except Exception as exc:  # noqa: BLE001 - datasets install/import failed
        errors.append(f"datasets: {str(exc)[:200]}")
    log.warning("wikitext unavailable; falling back to Gutenberg text. Tried: "
                + " | ".join(errors))
    import urllib.request
    req = urllib.request.Request(
        "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        headers={"User-Agent": "turbopress/1.0"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


# ----------------------------------------------------------------------------
# standalone loader written into the artifact (self-contained, repo-free)
# ----------------------------------------------------------------------------
RUNTIME_PY = r'''"""Standalone loader for a TurboPress-quantized model artifact.

Usage:
    python run_quantized.py --prompt "Hello" [--device cuda] [--max-new 64]
    python run_quantized.py --export-hf ./exported_fp16   # plain HF checkpoint

The artifact stores trellis-coded weights at their true bit-width; this
script decodes them, undoes the rotation/equilibration exactly, and loads a
standard `transformers` model (weights materialize as fp16 in memory).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent


def unpack_bits(buf, count, width):
    bits = np.unpackbits(buf.numpy())[: count * width].reshape(count, width)
    return (bits << np.arange(width, dtype=np.uint8)).sum(axis=1).astype(np.uint8)


def fwht(x):
    d = x.shape[-1]
    y = x.reshape(-1, d)
    h = 1
    while h < d:
        y = y.reshape(-1, d // (2 * h), 2, h)
        even, odd = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((even + odd, even - odd), dim=2).reshape(-1, d)
        h *= 2
    return (y / math.sqrt(d)).reshape(x.shape)


def rotate_inv(x, signs, block):
    if block == x.shape[-1]:
        z = fwht(x)
    else:
        m = x.shape[-1] // block
        z = fwht(x.reshape(*x.shape[:-1], m, block)).reshape(x.shape)
    return z * signs


def trellis_tables(n_states, g0, g1):
    m = n_states.bit_length() - 1
    sub = torch.empty(n_states, 2, dtype=torch.int64)
    for s in range(n_states):
        for b in (0, 1):
            x = (b << m) | s
            sub[s, b] = (((x & g1).bit_count() & 1) << 1) | ((x & g0).bit_count() & 1)
    return sub


def decode_weight(p, generators):
    n, d, bits = p["n"], p["d"], p["bits"]
    pathb = torch.from_numpy(unpack_bits(p["path_bits"], n * d, 1)).reshape(n, d)
    if bits > 1:
        memb = torch.from_numpy(unpack_bits(p["member_bits"], n * d, bits - 1)).reshape(n, d)
    else:
        memb = torch.zeros(n, d, dtype=torch.uint8)
    sub_table = trellis_tables(p["n_states"], *generators)
    level = torch.empty(n, d, dtype=torch.int64)
    state = torch.zeros(n, dtype=torch.int64)
    mask = p["n_states"] - 1
    for t in range(d):
        b = pathb[:, t].long()
        level[:, t] = memb[:, t].long() * 4 + sub_table[state, b]
        state = ((state << 1) | b) & mask
    cb = p["codebook"].float()
    scales = p["scales"].float()
    w_rot = scales[:, None] * cb[level]
    signs = torch.from_numpy(unpack_bits(p["signs"], d, 1)).float() * 2 - 1
    s = p["equil"].float()
    return rotate_inv(w_rot, signs, p["block"]) / s[None, :]


def load_quantized_model(artifact_dir=None, device="cuda", dtype=torch.float16):
    from transformers import AutoConfig, AutoModelForCausalLM
    root = Path(artifact_dir) if artifact_dir else HERE
    blob = torch.load(root / "turbopress_weights.pt", map_location="cpu",
                      weights_only=False)
    meta = blob["meta"]
    config = AutoConfig.from_pretrained(root / "hf_config")
    model = AutoModelForCausalLM.from_config(config)
    missing, unexpected = model.load_state_dict(blob["extra_state"], strict=False)
    unexpected = [k for k in unexpected if not k.endswith(".weight")]
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    sd = {}
    for key, p in blob["quantized"].items():
        w = decode_weight(p, tuple(meta["generators"]))
        sd[key] = w.to(dtype)
        print(f"  decoded {key}  [{p['n']}x{p['d']}] @ {p['bits']}b")
    model.load_state_dict(sd, strict=False)
    return model.to(device=device, dtype=dtype).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--export-hf", default=None,
                    help="also save a plain fp16 HF checkpoint to this dir")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HERE / "tokenizer")
    model = load_quantized_model(HERE, device=args.device)
    if args.export_hf:
        model.save_pretrained(args.export_hf)
        tok.save_pretrained(args.export_hf)
        print(f"exported plain HF checkpoint -> {args.export_hf}")
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
    print("\n--- generation ---")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
'''


# ----------------------------------------------------------------------------
# artifact helpers
# ----------------------------------------------------------------------------
def extra_state(model, quant_keys) -> dict[str, Tensor]:
    """Non-quantized tensors to store alongside the packed codes.

    With ``tie_word_embeddings`` the state dict exposes the shared embedding
    under both ``model.embed_tokens.weight`` and ``lm_head.weight``; saving the
    per-key CPU copies would store it twice, so the ``lm_head.weight`` alias is
    dropped (the loader's ``from_config`` re-ties it before ``load_state_dict``).
    """
    tied = getattr(model.config, "tie_word_embeddings", False)
    extra = {}
    for k, v in model.state_dict().items():
        if k in quant_keys or (tied and k == "lm_head.weight"):
            continue
        extra[k] = v.to(torch.float16).cpu() if v.is_floating_point() else v.cpu()
    return extra


# ----------------------------------------------------------------------------
# main flow
# ----------------------------------------------------------------------------
def compress(cfg: dict | None = None) -> dict:
    """Run the full TurboPress pipeline for one model/bit-width config.

    ``cfg`` is a dict with the keys in :data:`DEFAULT_CONFIG` (defaults are used
    for any that are missing). Returns ``{"artifact", "zip", "metrics",
    "bits_per_weight"}``.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    torch.manual_seed(cfg["SEED"])
    device = cfg["DEVICE"]
    bits, n_states = int(cfg["BITS"]), int(cfg["N_STATES"])
    if not 2 <= bits <= 6:
        raise ValueError(f"BITS must be in [2, 6], got {bits}")

    name = cfg["MODEL_ID"].split("/")[-1]
    art_dir = Path(cfg["OUT_DIR"]) / f"{name}-turbopress-{bits}bit"
    art_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(art_dir / "turbopress.log", mode="w", encoding="utf-8")
    fh.setFormatter(_fmt)
    log.addHandler(fh)

    log.info(f"TurboPress pipeline | model={cfg['MODEL_ID']} "
             f"bits={bits} n_states={n_states} device={device}")
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        log.info(f"GPU: {torch.cuda.get_device_name(0)} "
                 f"({free / 2**30:.1f} / {total / 2**30:.1f} GiB free)")

    # -- load model + tokenizer ------------------------------------------------
    log.info("loading model (fp16)...")
    tok = AutoTokenizer.from_pretrained(cfg["MODEL_ID"])
    model = AutoModelForCausalLM.from_pretrained(cfg["MODEL_ID"], dtype=torch.float16)
    model = model.to(device).eval()
    layers, layers_path = decoder_layers(model)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"loaded: {n_params / 1e9:.2f}B params, {len(layers)} decoder blocks")

    # -- data ------------------------------------------------------------------
    text = get_eval_text()
    ids = tok(text, return_tensors="pt").input_ids[0]
    seqlen, ev, cal, bsz = cfg["SEQLEN"], cfg["EVAL_SEQS"], cfg["CALIB_SEQS"], cfg["BATCH"]
    need = (ev + cal) * seqlen
    if ids.numel() < need:
        raise ValueError(f"eval text too short: {ids.numel()} tokens < {need}")
    eval_chunks = ids[: ev * seqlen].reshape(ev, seqlen)
    calib_chunks = ids[ev * seqlen : need].reshape(cal, seqlen)
    eval_batches = [eval_chunks[i : i + bsz] for i in range(0, ev, bsz)]
    calib_batches = [calib_chunks[i : i + bsz] for i in range(0, cal, bsz)]
    log.info(f"eval: {ev}x{seqlen} tokens | calibration (disjoint): {cal}x{seqlen}")

    # -- fp16 reference logits ---------------------------------------------------
    log.info("computing fp16 reference logits...")
    fp_logits, labels = eval_logits(model, eval_batches, device)

    # -- calibration: per-channel activation RMS --------------------------------
    log.info("collecting per-channel activation statistics...")
    stats = collect_channel_stats(model, layers, calib_batches, device)
    log.info(f"calibrated {len(stats)} linears")

    # -- pre-compute the data-free trellis codebook -----------------------------
    log.info(f"designing analytic TCQ codebook ({bits}b, S={n_states})...")
    tcq_codebook(bits, n_states)

    # -- quantize every decoder linear, in place --------------------------------
    quantized: dict[str, dict] = {}
    quant_keys = set()
    total_w = total_bits = 0
    n_mats = sum(1 for b in layers for _, m_ in b.named_modules() if isinstance(m_, nn.Linear))
    done, t_q = 0, time.time()
    integrity_checked = False
    for i, block in enumerate(layers):
        for lname, mod in list(block.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            key = f"{layers_path}.{i}.{lname}.weight"
            payload, w_hat = quantize_matrix(
                mod.weight, stats[f"{i}.{lname}"], bits, n_states,
                seed=cfg["SEED"] + 7919 * i + zlib.crc32(lname.encode()) % 1000,
            )
            if not integrity_checked:  # decode-from-packed must round-trip exactly
                n_, d_ = payload["n"], payload["d"]
                pb = torch.from_numpy(
                    unpack_bits(payload["path_bits"], n_ * d_, 1)).reshape(n_, d_)
                mb = torch.from_numpy(
                    unpack_bits(payload["member_bits"], n_ * d_, bits - 1)).reshape(n_, d_)
                lv = decode_levels(pb, mb, Trellis(n_states))
                w_chk = rotate_inv(
                    payload["scales"].float()[:, None]
                    * payload["codebook"].float()[lv.long()],
                    torch.from_numpy(unpack_bits(payload["signs"], d_, 1)).float() * 2 - 1,
                    payload["block"],
                ) / payload["equil"].float()[None, :]
                assert torch.allclose(w_chk, w_hat, atol=1e-6), "pack/decode mismatch"
                integrity_checked = True
                log.info("integrity check: packed bit-stream round-trips exactly")
            mod.weight.data = w_hat.to(dtype=torch.float16, device=device)
            quantized[key] = payload
            quant_keys.add(key)
            nw = payload["n"] * payload["d"]
            total_w += nw
            total_bits += (
                bits * nw + 16 * payload["n"]                     # codes + scales
                + 16 * (payload["d"] + payload["codebook"].numel())  # equil + codebook
                + payload["d"]                                     # rotation signs
            )
            done += 1
            if done % 25 == 0 or done == n_mats:
                el = time.time() - t_q
                log.info(f"quantized {done}/{n_mats} matrices "
                         f"({el:.0f}s, eta {el / done * (n_mats - done):.0f}s)")
    bpw = total_bits / total_w
    log.info(f"quantized {done} linears | {total_w / 1e9:.2f}B weights @ "
             f"{bpw:.3f} bits/weight (incl. all overheads)")

    # -- validation --------------------------------------------------------------
    log.info("evaluating quantized model...")
    q_logits, _ = eval_logits(model, eval_batches, device)
    m = metrics(fp_logits, q_logits, labels)
    log.info(f"RESULTS | KL(fp||q) = {m['mean_kl']:.4f} nats | "
             f"top-1 agreement = {m['top1_agreement']:.3f} | "
             f"ppl fp16 = {m['ppl_fp']:.3f} -> quant = {m['ppl_q']:.3f} "
             f"({m['n_tokens']} tokens)")

    log.info("demo generation (quantized model):")
    ids_p = tok(cfg["DEMO_PROMPT"], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(ids_p, max_new_tokens=48, do_sample=False)
    log.info("  " + tok.decode(out[0], skip_special_tokens=True).replace("\n", " "))

    # -- artifact ----------------------------------------------------------------
    log.info("writing artifact...")
    extra = extra_state(model, quant_keys)
    meta = {
        "format_version": 1,
        "pipeline": "rotate -> quarter-power equilibration -> TCQ (analytic codebook)",
        "model_id": cfg["MODEL_ID"], "bits": bits, "n_states": n_states,
        "generators": list(_TRELLIS_GENERATORS[n_states]),
        "bits_per_weight_total": round(bpw, 4), "seed": cfg["SEED"],
    }
    torch.save(
        {"meta": meta, "quantized": quantized, "extra_state": extra},
        art_dir / "turbopress_weights.pt",
    )
    model.config.save_pretrained(art_dir / "hf_config")
    tok.save_pretrained(art_dir / "tokenizer")
    (art_dir / "run_quantized.py").write_text(RUNTIME_PY, encoding="utf-8")
    (art_dir / "quantization_config.json").write_text(json.dumps(
        {"meta": meta, "config": {k: v for k, v in cfg.items()}, "metrics": m,
         "quantized_layers": sorted(quantized)}, indent=2))
    (art_dir / "README.md").write_text(
        f"# {name} -- TurboPress {bits}-bit\n\n"
        f"Quantized with the TurboPress pipeline ({meta['pipeline']}).\n"
        f"Measured vs fp16 on {m['n_tokens']} held-out tokens: "
        f"KL {m['mean_kl']:.4f}, top-1 agreement {m['top1_agreement']:.3f}, "
        f"perplexity {m['ppl_fp']:.2f} -> {m['ppl_q']:.2f}.\n\n"
        f"Run: `python run_quantized.py --prompt \"...\"` "
        f"(needs torch + transformers; weights decode to fp16 in memory).\n"
        f"Export a plain HF checkpoint: `python run_quantized.py --export-hf out/`.\n",
        encoding="utf-8")

    # -- self-test: reload from the artifact and compare logits ------------------
    if cfg["SELF_TEST"]:
        log.info("self-test: reloading model from the artifact...")
        spec = importlib.util.spec_from_file_location("tp_runtime", art_dir / "run_quantized.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)
        model2 = rt.load_quantized_model(art_dir, device=device)
        # 1) exact check: every parameter/buffer must match the validated model.
        sd1, sd2 = model.state_dict(), model2.state_dict()
        assert set(sd1) == set(sd2), (
            f"state dict key mismatch: only-in-original={sorted(set(sd1) - set(sd2))[:5]} "
            f"only-in-reloaded={sorted(set(sd2) - set(sd1))[:5]}"
        )
        diffs = {
            k: float((sd1[k].float() - sd2[k].float()).abs().max()) for k in sd1
        }
        worst = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
        assert worst[0][1] < 1e-6, f"artifact weight mismatch, worst tensors: {worst}"
        log.info(f"self-test: all {len(diffs)} tensors identical "
                 f"(max diff {worst[0][1]:.1e})")
        # 2) sanity check on logits (loose: fp16 kernel scheduling may differ).
        with torch.no_grad():
            a = model(input_ids=eval_batches[0].to(device)).logits.float()
            b = model2(input_ids=eval_batches[0].to(device)).logits.float()
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        kl = float((la.exp() * (la - lb)).sum(-1).mean())
        log.info(f"self-test: reload KL {kl:.2e}, max logit diff "
                 f"{float((a - b).abs().max()):.2e} (fp16 kernel noise)")
        assert kl < 1e-2, f"artifact reload produces different outputs: KL {kl}"
        log.info("self-test OK")
        del model2
        if device == "cuda":
            torch.cuda.empty_cache()

    zip_path = shutil.make_archive(str(art_dir), "zip", root_dir=art_dir.parent,
                                   base_dir=art_dir.name)
    art_mb = sum(f.stat().st_size for f in art_dir.rglob("*") if f.is_file()) / 2**20
    fp16_mb = 2 * n_params / 2**20
    log.info(f"artifact: {art_dir}  ({art_mb:.0f} MiB on disk; fp16 model would be "
             f"{fp16_mb:.0f} MiB -> {fp16_mb / art_mb:.1f}x smaller)")
    log.info(f"download: {zip_path}")
    log.info(f"total wall time: {(time.time() - t_start) / 60:.1f} min")
    log.removeHandler(fh)
    fh.close()
    return {"artifact": str(art_dir), "zip": zip_path, "metrics": m,
            "bits_per_weight": bpw, "n_params": n_params}


def main(cfg: dict | None = None) -> dict:
    """Entry point for ``python -m turbopress.pipeline`` (uses env config)."""
    return compress(cfg)


if __name__ == "__main__":
    main()
