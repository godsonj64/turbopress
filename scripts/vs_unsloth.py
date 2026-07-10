"""TurboPress vs Unsloth Dynamic GGUF -- same model, same tokens, same harness.

Contenders, all evaluated against the same fp16 reference with identical
token-level metrics (KL(fp||q), top-1 agreement, perplexity):

  * TurboPress: rotate -> quarter-power equilibration -> analytic TCQ (S=64),
    reconstructed in place at 2/3/4 bits (reuses the tested turbopress package).
  * Unsloth Dynamic 2.0 GGUFs (UD-Q2_K_XL / UD-Q3_K_XL / UD-Q4_K_XL, or the
    closest available), dequantized through transformers' GGUF loader so the
    evaluation code path is byte-for-byte the same.

Bit accounting: GGUF sizes include embeddings/head, TurboPress keeps them
fp16, so the table reports BOTH a decoder-weights bits/weight (TurboPress
only) and a whole-model bits/param computed the same way for both sides
(total stored bytes * 8 / total params). Compare the whole-model column.

Run:  python scripts/vs_unsloth.py   (env: VS_MODEL_ID, VS_GGUF_REPO, ...)
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turbopress.hadamard import RandomizedOrthogonal          # noqa: E402
from turbopress.real_model import _decoder_layers, _fetch_eval_text  # noqa: E402
from turbopress.trellis import Trellis, tcq_optimized_codebook  # noqa: E402

MODEL_ID = os.environ.get("VS_MODEL_ID", "Qwen/Qwen3-1.7B")
GGUF_REPO = os.environ.get("VS_GGUF_REPO", "unsloth/Qwen3-1.7B-GGUF")
BITS = [int(b) for b in os.environ.get("VS_BITS", "2,3,4").split(",")]
N_STATES = int(os.environ.get("VS_N_STATES", "64"))
SEQLEN = int(os.environ.get("VS_SEQLEN", "512"))
EVAL_SEQS = int(os.environ.get("VS_EVAL_SEQS", "8"))
CALIB_SEQS = int(os.environ.get("VS_CALIB_SEQS", "8"))
BATCH = int(os.environ.get("VS_BATCH", "1"))
DEVICE = os.environ.get("VS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("VS_SEED", "0"))
OUT = os.environ.get("VS_OUT", "results/vs_unsloth_qwen17b.json")

log = logging.getLogger("vs_unsloth")
log.setLevel(logging.INFO)
log.handlers.clear()
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
log.addHandler(_h)


def get_text() -> str:
    try:
        from datasets import load_dataset
        ds_test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ds_train = load_dataset(
            "Salesforce/wikitext", "wikitext-2-raw-v1", split="train[:2%]"
        )
        log.info("text: Salesforce/wikitext (wikitext-2-raw-v1)")
        return "\n".join(ds_test["text"]) + "\n" + "\n".join(ds_train["text"])
    except Exception as exc:  # noqa: BLE001
        log.info(f"datasets unavailable ({str(exc)[:120]}); using cached Gutenberg text")
        return _fetch_eval_text(Path("data") / "eval_text.txt")


@torch.no_grad()
def eval_logits(model, batches):
    outs, labels = [], []
    for ids in batches:
        ids = ids.to(DEVICE)
        lg = model(input_ids=ids).logits[:, :-1]
        outs.append(lg.reshape(-1, lg.shape[-1]).to(torch.float16).cpu())
        labels.append(ids[:, 1:].reshape(-1).cpu())
    return torch.cat(outs), torch.cat(labels)


def metrics(fp_logits, q_logits, labels, chunk=512):
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
    return {"mean_kl": kl / T, "top1_agreement": top1 / T,
            "ppl_fp": math.exp(ce_fp / T), "ppl_q": math.exp(ce_q / T)}


@torch.no_grad()
def collect_stats(model, layers, batches):
    sums, counts, handles = {}, {}, []

    def mk(key):
        def hook(_m, inputs, _o):
            x = inputs[0].detach().to(torch.float32).reshape(-1, inputs[0].shape[-1])
            sums[key] = sums.get(key, 0) + x.pow(2).sum(dim=0)
            counts[key] = counts.get(key, 0) + x.shape[0]
        return hook

    for i, block in enumerate(layers):
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                handles.append(mod.register_forward_hook(mk(f"{i}.{name}")))
    try:
        for ids in batches:
            model(input_ids=ids.to(DEVICE))
    finally:
        for h in handles:
            h.remove()
    return {k: (v / counts[k]).sqrt().cpu() for k, v in sums.items()}


def _trellis_on(device: str, n_states: int) -> Trellis:
    tr = Trellis(n_states)
    for attr in ("prev0", "prev1", "sub0", "sub1", "subset_table"):
        setattr(tr, attr, getattr(tr, attr).to(device))
    return tr


def _viterbi_dev(z, cb, tr):
    """Device-agnostic Viterbi (runs on the GPU when z lives there)."""
    m, d = z.shape
    S, dev = tr.n_states, z.device
    members = torch.empty(m, d, 4, dtype=torch.int64, device=dev)
    costs = torch.empty(m, d, 4, dtype=torch.float32, device=dev)
    for j in range(4):
        sub = cb[j::4]
        if sub.numel() == 1:
            mem = torch.zeros(m, d, dtype=torch.int64, device=dev)
        else:
            thr = (sub[:-1] + sub[1:]) / 2
            mem = torch.bucketize(z.contiguous(), thr)
        members[:, :, j] = mem
        costs[:, :, j] = (z - sub[mem]).pow(2)
    alpha = torch.full((m, S), math.inf, device=dev)
    alpha[:, 0] = 0.0
    bp = torch.empty(m, d, S, dtype=torch.bool, device=dev)
    for t in range(d):
        c = costs[:, t, :]
        c0 = alpha[:, tr.prev0] + c[:, tr.sub0]
        c1 = alpha[:, tr.prev1] + c[:, tr.sub1]
        take1 = c1 < c0
        alpha = torch.where(take1, c1, c0)
        bp[:, t, :] = take1
    rows = torch.arange(m, device=dev)
    state = alpha.argmin(dim=1)
    level = torch.empty(m, d, dtype=torch.int64, device=dev)
    for t in range(d - 1, -1, -1):
        take1 = bp[rows, t, state]
        prev = torch.where(take1, tr.prev1[state], tr.prev0[state])
        sub = torch.where(take1, tr.sub1[state], tr.sub0[state])
        level[:, t] = members[rows, t, sub] * 4 + sub
        state = prev
    return level


def _viterbi_chunked(z, cb, tr, budget_bytes=1_000_000_000):
    """Row-chunked so backpointer VRAM stays under ~1 GB."""
    m, d = z.shape
    per_row = d * (tr.n_states + 48)
    chunk = max(16, min(m, budget_bytes // max(per_row, 1)))
    return torch.cat([_viterbi_dev(z[i : i + chunk], cb, tr) for i in range(0, m, chunk)])


@torch.no_grad()
def turbopress_in_place(model, stats, bits):
    """Quantize every decoder linear in place, entirely on DEVICE."""
    layers = _decoder_layers(model)
    tr = _trellis_on(DEVICE, N_STATES)
    cb = tcq_optimized_codebook(bits, N_STATES).to(DEVICE)  # designed offline, cached
    total_bits = total_w = 0
    n_mats = sum(1 for b in layers for _, m in b.named_modules() if isinstance(m, nn.Linear))
    done, t0 = 0, time.time()
    for i, block in enumerate(layers):
        for name, mod in list(block.named_modules()):
            if not isinstance(mod, nn.Linear):
                continue
            w = mod.weight.detach().to(torch.float32)  # stays on DEVICE
            n, d = w.shape
            act = stats[f"{i}.{name}"].to(w.device).clamp_min(1e-8)
            act = act.clamp_min(0.05 * float(act.pow(2).mean().sqrt()))
            cn = w.norm(dim=0)
            cn = cn.clamp_min(max(0.05 * float(cn.pow(2).mean().sqrt()), 1e-12))
            s = (act / cn).sqrt().to(torch.float16).float()
            rot = RandomizedOrthogonal(
                d, seed=SEED + 7919 * i + zlib.crc32(name.encode()) % 1000
            ).to(w.device)
            w_rot = rot(w * s[None, :])
            scales = w_rot.pow(2).mean(dim=1).sqrt()
            nzr = scales > 0
            safe = torch.where(nzr, scales, torch.ones_like(scales))
            level = _viterbi_chunked(w_rot / safe[:, None], cb, tr)
            q = cb[level]
            num, den = (w_rot * q).sum(dim=1), (q * q).sum(dim=1)
            refit = torch.where(den > 0, num / den.clamp_min(1e-30), safe)
            safe = torch.where(refit > 0, refit, safe)
            scales = torch.where(nzr, safe, torch.zeros_like(safe))
            w_hat = rot.inverse(scales[:, None] * q) / s[None, :]
            mod.weight.data = w_hat.to(torch.float16)
            total_bits += bits * n * d + 16 * n + 17 * d + 16 * cb.numel()
            total_w += n * d
            done += 1
            if done % 28 == 0 or done == n_mats:
                el = time.time() - t0
                log.info(f"  quantized {done}/{n_mats} ({el:.0f}s, "
                         f"eta {el / done * (n_mats - done):.0f}s)")
    return total_bits, total_w


def pick_gguf_files(repo):
    from huggingface_hub import HfApi
    files = [f for f in HfApi().list_repo_files(repo) if f.endswith(".gguf")]
    picks = []
    for prefs in (("UD-Q2_K_XL", "Q2_K_L", "Q2_K"),
                  ("UD-Q3_K_XL", "Q3_K_M", "Q3_K_S"),
                  ("UD-Q4_K_XL", "Q4_K_M", "Q4_K_S")):
        for p in prefs:
            hit = [f for f in files if p in f]
            if hit:
                picks.append(sorted(hit)[0])
                break
    if not picks:
        raise RuntimeError(f"no matching .gguf files in {repo}; found: {files[:20]}")
    return picks


def free(model):
    del model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Workaround: the gguf wheel's import name is missing from
    # packages_distributions() on Python 3.10, so transformers resolves its
    # version to 'N/A' and crashes in is_gguf_available(). Point the mapping
    # at the distribution explicitly.
    import transformers.utils.import_utils as _iu
    _iu.PACKAGE_DISTRIBUTION_MAPPING.setdefault("gguf", ["gguf"])

    torch.manual_seed(SEED)
    t_all = time.time()
    log.info(f"model={MODEL_ID} vs {GGUF_REPO} | bits={BITS} S={N_STATES} "
             f"device={DEVICE}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(get_text(), return_tensors="pt").input_ids[0]
    need = (EVAL_SEQS + CALIB_SEQS) * SEQLEN
    assert ids.numel() >= need, f"text too short: {ids.numel()} < {need}"
    ev = ids[: EVAL_SEQS * SEQLEN].reshape(EVAL_SEQS, SEQLEN)
    ca = ids[EVAL_SEQS * SEQLEN : need].reshape(CALIB_SEQS, SEQLEN)
    eval_batches = [ev[i : i + BATCH] for i in range(0, EVAL_SEQS, BATCH)]
    calib_batches = [ca[i : i + BATCH] for i in range(0, CALIB_SEQS, BATCH)]

    log.info("fp16 reference pass...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model = model.to(DEVICE).eval()
    n_params = sum(p.numel() for p in model.parameters())
    fp_logits, labels = eval_logits(model, eval_batches)
    layers = _decoder_layers(model)
    stats = collect_stats(model, layers, calib_batches)
    log.info(f"{n_params / 1e9:.2f}B params, calibrated {len(stats)} linears")
    free(model)

    results = []
    for bits in BITS:
        log.info(f"TurboPress {bits}b (TCQ S={N_STATES} + quarter equil)...")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
        model = model.to(DEVICE).eval()
        dec_bits, dec_w = turbopress_in_place(model, stats, bits)
        q_logits, _ = eval_logits(model, eval_batches)
        m = metrics(fp_logits, q_logits, labels)
        whole_bits = dec_bits + 16 * (n_params - dec_w)
        row = {"method": f"TurboPress {bits}b", "family": "turbopress",
               "decoder_bpw": round(dec_bits / dec_w, 3),
               "whole_model_bpw": round(whole_bits / n_params, 3), **m}
        results.append(row)
        log.info(f"  -> KL {m['mean_kl']:.4f} top1 {m['top1_agreement']:.3f} "
                 f"ppl {m['ppl_q']:.2f} | whole-model {row['whole_model_bpw']} b/param")
        free(model)

    for fname in pick_gguf_files(GGUF_REPO):
        log.info(f"Unsloth {fname}...")
        path = hf_hub_download(GGUF_REPO, fname)
        size_bits = Path(path).stat().st_size * 8
        model = AutoModelForCausalLM.from_pretrained(
            GGUF_REPO, gguf_file=fname, dtype=torch.float16
        ).to(DEVICE).eval()
        q_logits, _ = eval_logits(model, eval_batches)
        m = metrics(fp_logits, q_logits, labels)
        row = {"method": fname, "family": "unsloth", "decoder_bpw": None,
               "whole_model_bpw": round(size_bits / n_params, 3), **m}
        results.append(row)
        log.info(f"  -> KL {m['mean_kl']:.4f} top1 {m['top1_agreement']:.3f} "
                 f"ppl {m['ppl_q']:.2f} | whole-model {row['whole_model_bpw']} b/param")
        free(model)

    hdr = (f"{'method':<34} {'whole b/p':>9} {'KL(fp||q)':>10} {'top1':>7} {'ppl':>10}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in sorted(results, key=lambda r: r["whole_model_bpw"]):
        print(f"{r['method']:<34} {r['whole_model_bpw']:>9.3f} {r['mean_kl']:>10.4f} "
              f"{r['top1_agreement']:>7.3f} {r['ppl_q']:>10.3f}")
    print(f"\nfp16 reference perplexity: {results[0]['ppl_fp']:.4f} "
          f"({fp_logits.shape[0]} tokens)")

    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "settings": {"model": MODEL_ID, "gguf_repo": GGUF_REPO, "bits": BITS,
                     "n_states": N_STATES, "seqlen": SEQLEN, "eval_seqs": EVAL_SEQS,
                     "calib_seqs": CALIB_SEQS, "seed": SEED, "device": DEVICE},
        "n_params": n_params, "results": results}, indent=2))
    log.info(f"saved {out} | total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
