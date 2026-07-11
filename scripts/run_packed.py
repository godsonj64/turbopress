"""Run a TurboPress artifact *packed* -- weights stay trellis-coded in VRAM.

Unlike the artifact's bundled ``run_quantized.py`` (which decodes every weight
to a full fp16 matrix), this loads each decoder linear into a
``PackedTCQLinear`` so resident weight memory is ~bits/16 of fp16 and the fp16
matrix is never materialized.

    python scripts/run_packed.py ./out/Qwen3-0.6B-turbopress-4bit \
        --prompt "The capital of France is" --max-new 128 --mode triton

Modes (see turbopress/runtime.py):
  triton  fused packed GEMV -- packed memory + fast (needs triton / triton-windows)
  tiled   pure-PyTorch, packed memory, slow (no triton needed)
  cached  decode once to fp16 at load (fp16 memory/speed; the run_quantized.py path)

Works with an installed turbopress >= 0.4; needs torch + transformers, and for
--mode triton a working `triton` (HAS_TRITON True) and a CUDA GPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from turbopress.pipeline import _TRELLIS_GENERATORS, unpack_bits
from turbopress.real_model import _decoder_layers
from turbopress.runtime import PackedTCQLinear, pack_le


def _unpack(t: torch.Tensor, count: int, width: int) -> torch.Tensor:
    """format-v1 bit-unpack -> uint8 [count] (numpy bridge, as the artifact stores)."""
    arr = t.detach().cpu().contiguous().numpy()
    return torch.from_numpy(unpack_bits(arr, count, width)).to(torch.uint8)


def _v1_to_v2(payload: dict, bias: torch.Tensor | None, device: str) -> dict:
    """Convert a pipeline format-v1 matrix into the format-v2 PackedTCQLinear payload.

    Preserves the exact stored trellis decisions, scales, codebook, signs and
    equilibration -- no re-quantization, so outputs match the compressed model.
    """
    n, d, bits = int(payload["n"]), int(payload["d"]), int(payload["bits"])
    path = _unpack(payload["path_bits"], n * d, 1).reshape(n, d)
    member = (
        _unpack(payload["member_bits"], n * d, bits - 1).reshape(n, d)
        if bits > 1
        else None
    )
    signs = _unpack(payload["signs"], d, 1).reshape(1, d)
    equil = payload["equil"].detach().float().clamp_min(1e-12)
    return {
        "format": 2,
        "n": n,
        "d": d,
        "bits": bits,
        "n_states": int(payload["n_states"]),
        "block": int(payload["block"]),
        "path_packed": pack_le(path.to(device), 1),
        "member_packed": (
            pack_le(member.to(device), bits - 1) if member is not None else None
        ),
        "scales": payload["scales"].detach().to(device=device, dtype=torch.float16),
        "signs_packed": pack_le(signs.to(device), 1),
        "inv_equil": (1.0 / equil).to(device=device, dtype=torch.float16),
        "codebook": payload["codebook"].detach().to(device=device, dtype=torch.float16),
        "bias": None if bias is None else bias.detach().to(device=device, dtype=torch.float16),
        "seed": 0,
    }


@torch.inference_mode()
def load_packed_model(artifact: Path, device: str = "cuda", mode: str = "triton"):
    """Load an artifact directory into a packed (PackedTCQLinear) causal LM."""
    from transformers import AutoConfig, AutoModelForCausalLM

    blob = torch.load(
        artifact / "turbopress_weights.pt", map_location="cpu", weights_only=False
    )
    meta, quantized, extra = blob["meta"], blob.pop("quantized"), blob.pop("extra_state")
    if int(meta.get("format_version", -1)) != 1:
        raise RuntimeError("expected a TurboPress format_version=1 artifact")
    n_states = int(meta["n_states"])
    if tuple(meta["generators"]) != tuple(_TRELLIS_GENERATORS[n_states]):
        raise RuntimeError("trellis generator mismatch")

    # fp16 skeleton on CPU (random weights; the real ones are loaded below).
    config = AutoConfig.from_pretrained(artifact / "hf_config")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = AutoModelForCausalLM.from_config(config)
    finally:
        torch.set_default_dtype(prev)
    model = model.to(dtype=torch.float16)

    # non-quantized tensors (embeddings, norms, ...); quant keys + tied lm_head
    # come back as "missing", which is expected.
    model.load_state_dict(extra, strict=False)
    model.tie_weights()  # restore lm_head <- embed_tokens (dropped in the artifact)

    for key in list(quantized):
        payload = quantized.pop(key)
        module_path = key[: -len(".weight")]
        lin = model.get_submodule(module_path)
        if not isinstance(lin, nn.Linear):
            raise RuntimeError(f"{module_path} is not nn.Linear")
        bias = None if lin.bias is None else lin.bias.detach().clone()
        packed = PackedTCQLinear(_v1_to_v2(payload, bias, device), mode=mode)
        parent_path, leaf = module_path.rsplit(".", 1)
        setattr(model.get_submodule(parent_path), leaf, packed)

    model = model.to(device=device, dtype=torch.float16).eval()
    model.config.use_cache = True
    model.tie_weights()
    return model, meta


def _weight_mib(model) -> tuple[float, float]:
    """(resident packed weight MiB, equivalent fp16 MiB) over decoder linears."""
    packed = fp16 = 0
    for block in _decoder_layers(model):
        for m in block.modules():
            if isinstance(m, PackedTCQLinear):
                packed += m.packed_bytes()
                fp16 += 2 * m.in_features * m.out_features
            elif isinstance(m, nn.Linear):
                packed += m.weight.numel() * m.weight.element_size()
                fp16 += 2 * m.weight.numel()
    return packed / 2**20, fp16 / 2**20


def _static_cache(model, max_len: int):
    from transformers import StaticCache

    kw = dict(max_cache_len=max_len, device=model.device, dtype=model.dtype)
    try:  # kwarg name changed across transformers versions
        return StaticCache(config=model.config, max_batch_size=1, **kw)
    except TypeError:
        return StaticCache(config=model.config, batch_size=1, **kw)


@torch.inference_mode()
def graph_generate(model, input_ids, max_new: int):
    """Greedy decode via CUDA graphs -- the fast packed path.

    Prefill runs eagerly (dynamic length); the single-token decode step is
    captured once against a fixed-size StaticCache and replayed for every
    subsequent token, so Triton's per-launch Python dispatch is paid once at
    capture instead of on every layer of every step. Ported from the validated
    benchmark harness; greedy, so tokens match the eager loop.
    """
    dev = model.device
    prompt_len = input_ids.shape[1]
    cache = _static_cache(model, prompt_len + max_new + 16)

    logits = model(
        input_ids=input_ids, past_key_values=cache, use_cache=True,
        cache_position=torch.arange(prompt_len, device=dev),
    ).logits
    tok = logits[:, -1:].argmax(-1).clone()
    pos = torch.tensor([prompt_len], device=dev, dtype=torch.long)
    gen = [tok.clone()]

    def step():
        return model(
            input_ids=tok, past_key_values=cache, use_cache=True, cache_position=pos,
        ).logits

    # Warm up on a side stream before capture (required for CUDA graphs).
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            nxt = step()[:, -1:].argmax(-1)
            tok.copy_(nxt); pos.add_(1); gen.append(tok.clone())
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_logits = step()
    torch.cuda.synchronize()

    while len(gen) < max_new:
        graph.replay()
        tok.copy_(graph_logits[:, -1:].argmax(-1)); pos.add_(1); gen.append(tok.clone())
    torch.cuda.synchronize()

    generated = torch.cat(gen[:max_new], dim=1)
    return torch.cat([input_ids, generated], dim=1).cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", help="path to the TurboPress artifact directory")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--mode", default="triton", choices=["triton", "tiled", "cached"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="capture one decode step in a CUDA graph and replay it -- the "
                         "fast packed path (needs --device cuda). Without it, uses "
                         "eager model.generate(), which pays Triton dispatch every step.")
    args = ap.parse_args()

    art = Path(args.artifact)
    if args.mode == "triton":
        from turbopress.triton_kernel import HAS_TRITON

        if not HAS_TRITON:
            raise SystemExit("--mode triton needs a working `triton` (HAS_TRITON is False)")
        if args.device != "cuda":
            raise SystemExit("--mode triton requires --device cuda")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(art / "tokenizer")
    t0 = time.perf_counter()
    model, meta = load_packed_model(art, device=args.device, mode=args.mode)
    load_s = time.perf_counter() - t0
    packed_mib, fp16_mib = _weight_mib(model)
    print(
        f"loaded {meta['model_id']} @ {meta['bits']}b, mode={args.mode} in {load_s:.1f}s\n"
        f"decoder weight memory: {packed_mib:.1f} MiB packed "
        f"(fp16 would be {fp16_mib:.1f} MiB -> {fp16_mib / max(packed_mib, 1e-9):.1f}x less)"
    )

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    if args.cuda_graph:
        if args.device != "cuda":
            raise SystemExit("--cuda-graph requires --device cuda")
        t0 = time.perf_counter()
        out = graph_generate(model, ids, args.max_new)
        label = "cuda-graph decode"
    else:
        t0 = time.perf_counter()
        out = model.generate(ids, max_new_tokens=args.max_new, do_sample=False)
        label = "eager generate"
    gen_s = time.perf_counter() - t0
    n_new = out.shape[1] - ids.shape[1]
    print(f"\n--- {label}: {n_new} tokens, {n_new / gen_s:.1f} tok/s ---")
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
