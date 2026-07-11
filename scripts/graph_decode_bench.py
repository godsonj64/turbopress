"""CUDA-graph decode-step benchmark: packed-triton runtime vs fp16.

Eager per-token decoding pays ~50-80 us of Python dispatch per *layer* call
(Triton's launch wrapper; measured on Windows), which buries the fused
kernel's actual speed. The fix is the vLLM/TensorRT recipe: prefill once,
then capture ONE full decode step (static input ids + static KV cache +
device-side cache position) in a CUDA graph and replay it per token --
zero Python between kernels.

Measures tokens/s for {fp16, packed-triton} x {eager, graph} and verifies
the graph replays produce exactly the greedy tokens of the eager loop.

Run:  python scripts/graph_decode_bench.py
      (env: GB_MODEL_ID, GB_BITS, GB_N_STATES, GB_TOKENS)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from turbopress.real_model import collect_input_scales  # noqa: E402
from turbopress.runtime import pack_model  # noqa: E402

MODEL_ID = os.environ.get("GB_MODEL_ID", "HuggingFaceTB/SmolLM2-135M")
BITS = int(os.environ.get("GB_BITS", "3"))
N_STATES = int(os.environ.get("GB_N_STATES", "16"))
NEW_TOKENS = int(os.environ.get("GB_TOKENS", "128"))
PROMPT = "The most important discovery in twentieth century physics was"


def build_static_cache(model, max_len):
    from transformers import StaticCache

    kwargs = dict(max_cache_len=max_len, device=model.device, dtype=model.dtype)
    try:
        return StaticCache(config=model.config, max_batch_size=1, **kwargs)
    except TypeError:  # older signature
        return StaticCache(config=model.config, batch_size=1, **kwargs)


@torch.inference_mode()
def decode_eager(model, prompt_ids, n_new, max_len):
    cache = build_static_cache(model, max_len)
    L = prompt_ids.shape[1]
    logits = model(
        input_ids=prompt_ids, past_key_values=cache, use_cache=True,
        cache_position=torch.arange(L, device=model.device),
    ).logits
    tok = logits[:, -1:].argmax(-1)
    out = [int(tok)]
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(1, n_new):
        logits = model(
            input_ids=tok, past_key_values=cache, use_cache=True,
            cache_position=torch.tensor([L + step - 1], device=model.device),
        ).logits
        tok = logits[:, -1:].argmax(-1)
        out.append(int(tok))
    torch.cuda.synchronize()
    return out, (n_new - 1) / (time.time() - t0)


@torch.inference_mode()
def decode_graph(model, prompt_ids, n_new, max_len):
    cache = build_static_cache(model, max_len)
    dev = model.device
    L = prompt_ids.shape[1]
    logits = model(
        input_ids=prompt_ids, past_key_values=cache, use_cache=True,
        cache_position=torch.arange(L, device=dev),
    ).logits
    static_tok = logits[:, -1:].argmax(-1).clone()
    static_pos = torch.tensor([L], device=dev)
    out = [int(static_tok)]

    def step():
        return model(
            input_ids=static_tok, past_key_values=cache, use_cache=True,
            cache_position=static_pos,
        ).logits

    # warmup on a side stream (resolves triton autotune, cuDNN plans, ...)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            logits = step()
            static_tok.copy_(logits[:, -1:].argmax(-1))
            static_pos += 1
            out.append(int(static_tok))
    torch.cuda.current_stream().wait_stream(s)

    # Capture RECORDS the step without executing it (the KV write and the
    # logits are not computed until the first replay), so the static state
    # must not be advanced here and no token may be consumed from capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_logits = step()

    torch.cuda.synchronize()
    t0 = time.time()
    n_timed = 0
    for _ in range(n_new - len(out)):
        graph.replay()  # executes at the current static_tok/static_pos
        static_tok.copy_(g_logits[:, -1:].argmax(-1))
        static_pos += 1
        n_timed += 1
        out.append(int(static_tok))
    torch.cuda.synchronize()
    tps = n_timed / (time.time() - t0)
    return out, tps


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(dev)
    max_len = ids.shape[1] + NEW_TOKENS + 8
    print(f"model={MODEL_ID} bits={BITS} S={N_STATES} new_tokens={NEW_TOKENS}")

    results = {}
    for mode in ("fp16", "triton"):
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
        model = model.to(dev).eval()
        if mode == "triton":
            calib = tok("Compression is the art of keeping only what matters. " * 40,
                        return_tensors="pt").input_ids[:, :512].to(dev)
            stats = collect_input_scales(model, [calib])
            report = pack_model(model, stats, bits=BITS, n_states=N_STATES,
                                mode="triton")
            print(f"packed (triton): compression {report['compression']:.1f}x")
        toks_e, tps_e = decode_eager(model, ids, NEW_TOKENS, max_len)
        toks_g, tps_g = decode_graph(model, ids, NEW_TOKENS, max_len)
        match = toks_e[: len(toks_g)] == toks_g
        results[mode] = (tps_e, tps_g)
        print(f"{mode:>7}: eager {tps_e:6.1f} tok/s | graph {tps_g:6.1f} tok/s "
              f"({tps_g / tps_e:.1f}x from graph capture) | "
              f"greedy tokens match eager: {match}")
        del model
        torch.cuda.empty_cache()

    fp_e, fp_g = results["fp16"]
    pk_e, pk_g = results["triton"]
    print(f"\npacked-vs-fp16: eager {pk_e / fp_e:.2f}x | graph {pk_g / fp_g:.2f}x")


if __name__ == "__main__":
    main()
