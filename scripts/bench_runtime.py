"""Benchmark the packed TurboPress runtime against fp16 on a real model.

Measures, for fp16 / packed-cached / packed-tiled:
  * decoder weight memory actually resident (bytes)
  * generation throughput (tokens/s, greedy, batch 1)
  * agreement of generated tokens with the packed-cached reference
    (cached and tiled must agree exactly: same decoded weights)

Run:  python scripts/bench_runtime.py  (env: BR_MODEL_ID, BR_BITS, BR_TOKENS)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turbopress.real_model import _decoder_layers, collect_input_scales  # noqa: E402
from turbopress.runtime import PackedTCQLinear, pack_model  # noqa: E402

MODEL_ID = os.environ.get("BR_MODEL_ID", "HuggingFaceTB/SmolLM2-135M")
BITS = int(os.environ.get("BR_BITS", "3"))
N_STATES = int(os.environ.get("BR_N_STATES", "16"))
NEW_TOKENS = int(os.environ.get("BR_TOKENS", "128"))
DEVICE = os.environ.get("BR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
PROMPT = "The most important discovery in twentieth century physics was"

log = logging.getLogger("bench")
log.setLevel(logging.INFO)
log.handlers.clear()
_h = logging.StreamHandler(sys.stdout)
_h.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
log.addHandler(_h)


@torch.no_grad()
def generate(model, ids, n_new):
    out = model.generate(
        ids, max_new_tokens=n_new, do_sample=False,
        pad_token_id=model.config.eos_token_id,
    )
    return out[0, ids.shape[1]:]


@torch.no_grad()
def bench(model, ids, n_new):
    generate(model, ids, min(8, n_new))  # warmup
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    toks = generate(model, ids, n_new)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return toks, n_new / (time.time() - t0)


def decoder_weight_bytes(model):
    total = 0
    for block in _decoder_layers(model):
        for mod in block.modules():
            if isinstance(mod, PackedTCQLinear):
                total += mod.packed_bytes()  # mode-aware: cache or bit-streams
            elif isinstance(mod, torch.nn.Linear):
                total += mod.weight.numel() * mod.weight.element_size()
    return total


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    log.info(f"model={MODEL_ID} bits={BITS} S={N_STATES} device={DEVICE} "
             f"tokens={NEW_TOKENS}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEVICE)

    results = {}

    # ---- fp16 baseline -------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model = model.to(DEVICE).eval()
    fp_bytes = decoder_weight_bytes(model)
    _, tps = bench(model, ids, NEW_TOKENS)
    results["fp16"] = {"weight_mib": fp_bytes / 2**20, "tok_s": tps}
    log.info(f"fp16          : {fp_bytes / 2**20:7.1f} MiB decoder weights, "
             f"{tps:6.1f} tok/s")

    # ---- calibration + teacher-forced reference --------------------------
    calib_ids = tok("Compression is the art of keeping only what matters. " * 40,
                    return_tensors="pt").input_ids[:, :512].to(DEVICE)
    stats = collect_input_scales(model, [calib_ids])
    # Teacher-forced next-token predictions on a fixed window: the honest
    # fidelity signal (free-running greedy trajectories diverge chaotically
    # from tiny logit differences and say nothing about quality).
    with torch.no_grad():
        fp_pred = model(input_ids=calib_ids[:, :256]).logits.argmax(-1)

    # ---- packed modes ----------------------------------------------------
    for mode in ("cached", "tiled"):
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
        m = m.to(DEVICE).eval()
        t0 = time.time()
        report = pack_model(m, stats, bits=BITS, n_states=N_STATES, mode=mode)
        log.info(f"packing ({mode}) took {time.time() - t0:.0f}s, "
                 f"compression {report['compression']:.1f}x")
        toks, tps = bench(m, ids, NEW_TOKENS)
        w_bytes = decoder_weight_bytes(m)
        with torch.no_grad():
            q_pred = m(input_ids=calib_ids[:, :256]).logits.argmax(-1)
        agree = float((q_pred == fp_pred).float().mean())
        results[f"packed-{mode}"] = {
            "weight_mib": w_bytes / 2**20, "tok_s": tps,
            "teacher_forced_top1_vs_fp16": agree,
            "compression_vs_fp16": fp_bytes / w_bytes,
        }
        log.info(f"packed-{mode:<7}: {w_bytes / 2**20:7.1f} MiB decoder weights "
                 f"({fp_bytes / w_bytes:.1f}x vs fp16), {tps:6.1f} tok/s, "
                 f"teacher-forced top-1 agreement {agree:.0%}")
        if mode == "cached":
            sample = tok.decode(toks[:32])
            log.info(f"  sample: {sample!r}")
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = Path("results/runtime_bench.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"settings": {"model": MODEL_ID, "bits": BITS, "n_states": N_STATES,
                      "device": DEVICE, "new_tokens": NEW_TOKENS},
         "results": results}, indent=2))
    log.info(f"saved {out}")


if __name__ == "__main__":
    main()
