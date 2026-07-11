"""TurboPress native packed Triton runtime -- benchmark (laptop / local edition).

Loads a TurboPress artifact into the packed Triton runtime (weights stay
trellis-coded in VRAM), and compares eager vs CUDA-graph decode against an
fp16 baseline. Adapted from the Colab cell to run locally through the
pip-installed `turbopress` -- no pip install here (uses whatever you have
installed), no /content paths, and the version guard is a warning, not a hard
error (the packed runtime is compatible across 0.5.x).

Usage:
    python scripts/bench_packed.py [ARTIFACT_DIR]

ARTIFACT_DIR defaults to the Qwen3-0.6B/4bit artifact below; pass a path to
benchmark a different one (e.g. the 4B/3bit artifact -- but set
RUN_FP16_BASELINE = False for that, its fp16 copy won't fit in 8 GB).

Requires: torch + transformers, a CUDA GPU, and a working `triton`
(HAS_TRITON True -- `triton-windows` on Windows).
"""

import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# (expandable_segments is a Linux-only allocator feature -- omitted so it doesn't
#  print a harmless "not supported on this platform" warning on Windows.)

import torch
import triton  # noqa: F401  (import validates the wheel is importable)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    StaticCache,
)

from turbopress import __version__ as tp_version
from turbopress.pipeline import _TRELLIS_GENERATORS, unpack_bits
from turbopress.real_model import _decoder_layers
from turbopress.runtime import PackedTCQLinear, pack_le
from turbopress.triton_kernel import HAS_TRITON

# ------------------------------ configuration ------------------------------

_DEFAULT_ARTIFACT = r"C:\Users\godso\out\Qwen3-0.6B-turbopress-4bit"
ARTIFACT_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_ARTIFACT)
# NB: build the zip name by appending, not with_suffix() -- the model name
# contains a dot ("0.6B"), which with_suffix() would truncate.
ARTIFACT_ZIP = ARTIFACT_DIR.parent / (ARTIFACT_DIR.name + ".zip")
REPORT_PATH = ARTIFACT_DIR.parent / "turbopress_native_report.json"

PROMPT = "The three most important ideas in theoretical physics are"
NEW_TOKENS = 128
REPEATS = 3
RUN_FP16_BASELINE = True
VERSION = "0.5.1"


# ------------------------------- utilities ---------------------------------


def sync() -> None:
    torch.cuda.synchronize()


def clean() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    sync()


def gib(byte_count) -> float:
    return float(byte_count) / 2**30


def decoder_bytes(model) -> int:
    total = 0
    for block in _decoder_layers(model):
        for module in block.modules():
            if isinstance(module, PackedTCQLinear):
                total += module.packed_bytes()
            elif isinstance(module, torch.nn.Linear):
                total += module.weight.numel() * module.weight.element_size()
    return int(total)


# ---------------------- artifact format-v1 conversion ----------------------


def unpack_v1(tensor, count, width):
    array = tensor.detach().cpu().contiguous().numpy()
    return torch.from_numpy(unpack_bits(array, count, width)).to(torch.uint8)


def v1_to_v2(payload, bias=None, device="cuda"):
    """Convert a pipeline format-v1 artifact matrix into the format-v2
    representation expected by PackedTCQLinear (no re-quantization)."""
    n = int(payload["n"])
    d = int(payload["d"])
    bits = int(payload["bits"])

    path = unpack_v1(payload["path_bits"], n * d, 1).reshape(n, d)
    member = (
        unpack_v1(payload["member_bits"], n * d, bits - 1).reshape(n, d)
        if bits > 1
        else None
    )
    signs = unpack_v1(payload["signs"], d, 1).reshape(1, d)
    equilibration = payload["equil"].detach().float().clamp_min(1e-12)

    converted = {
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
        "inv_equil": (1.0 / equilibration).to(device=device, dtype=torch.float16),
        "codebook": payload["codebook"].detach().to(device=device, dtype=torch.float16),
        "bias": None if bias is None else bias.detach().to(device=device, dtype=torch.float16),
        "seed": 0,
    }
    del path, member, signs, equilibration
    return converted


# --------------------------- tied-weight handling --------------------------


def tied_patterns(model) -> list:
    """Model-declared tied-weight aliases (Qwen3 ties lm_head <- embed_tokens)."""
    spec = getattr(model, "_tied_weights_keys", None)
    patterns: list = []
    if isinstance(spec, dict):
        patterns.extend(str(k) for k in spec.keys())
    elif isinstance(spec, (list, tuple, set)):
        patterns.extend(str(x) for x in spec)
    elif isinstance(spec, str):
        patterns.append(spec)
    if getattr(model.config, "tie_word_embeddings", False):
        patterns.append("lm_head.weight")
    return patterns


def allowed_tied_key(key, patterns) -> bool:
    import re

    for pattern in patterns:
        if key == pattern:
            return True
        try:
            if re.fullmatch(pattern, key):
                return True
        except re.error:
            pass
    return False


def verify_tie(model) -> None:
    """Restore tied parameters and verify the head shares storage with embed."""
    model.tie_weights()
    if not getattr(model.config, "tie_word_embeddings", False):
        return
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if inp is None or out is None:
        raise RuntimeError("tie_word_embeddings=True but embedding module is missing.")
    same = inp.weight is out.weight or inp.weight.data_ptr() == out.weight.data_ptr()
    if not same:
        raise RuntimeError("lm_head.weight is not tied to model.embed_tokens.weight.")
    print("Verified tied weights: lm_head.weight -> model.embed_tokens.weight")


# --------------------------- native model loader ---------------------------


@torch.inference_mode()
def load_native(artifact: Path, device: str = "cuda"):
    total_start = time.perf_counter()

    blob = torch.load(
        artifact / "turbopress_weights.pt", map_location="cpu", weights_only=False
    )
    metadata = dict(blob["meta"])
    quantized = blob.pop("quantized")
    extra_state = blob.pop("extra_state")
    del blob

    if int(metadata.get("format_version", -1)) != 1:
        raise RuntimeError("Expected a TurboPress pipeline format_version=1 artifact.")

    n_states = int(metadata["n_states"])
    if tuple(metadata["generators"]) != tuple(_TRELLIS_GENERATORS[n_states]):
        raise RuntimeError("Trellis generator mismatch.")

    print(f"Source model:        {metadata['model_id']}")
    print(f"Artifact bits:       {metadata['bits']}")
    print(f"Trellis states:      {metadata['n_states']}")
    print(f"Error feedback:      {metadata.get('error_feedback')}")
    print(f"Equilibration alpha: {metadata.get('equil_alpha')}")
    print(f"Pipeline:            {metadata.get('pipeline')}")

    config = AutoConfig.from_pretrained(artifact / "hf_config")
    config._attn_implementation = "sdpa"

    old_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model = AutoModelForCausalLM.from_config(config)
    finally:
        torch.set_default_dtype(old_default)
    model = model.to(dtype=torch.float16)

    missing, unexpected = model.load_state_dict(extra_state, strict=False)
    del extra_state

    quantized_keys = set(quantized.keys())
    tie_aliases = tied_patterns(model)
    invalid_missing = [
        k for k in missing
        if k not in quantized_keys and not allowed_tied_key(k, tie_aliases)
    ]
    invalid_unexpected = [k for k in unexpected if k not in quantized_keys]
    if invalid_missing:
        raise RuntimeError(f"Unexpected missing model keys: {invalid_missing[:10]}")
    if invalid_unexpected:
        raise RuntimeError(f"Unexpected artifact keys: {invalid_unexpected[:10]}")

    verify_tie(model)  # restore the omitted duplicate Qwen lm_head

    matrix_count = len(quantized)
    print(f"Converting {matrix_count} matrices to PackedTCQLinear(mode='triton')...")

    torch.cuda.reset_peak_memory_stats()
    conversion_start = time.perf_counter()

    for index, weight_key in enumerate(list(quantized), start=1):
        payload_v1 = quantized.pop(weight_key)
        if not weight_key.endswith(".weight"):
            raise RuntimeError(f"Invalid weight key: {weight_key}")
        module_path = weight_key[: -len(".weight")]
        original_linear = model.get_submodule(module_path)
        if not isinstance(original_linear, torch.nn.Linear):
            raise RuntimeError(f"{module_path} is not nn.Linear.")
        bias = (
            None if original_linear.bias is None else original_linear.bias.detach().clone()
        )
        payload_v2 = v1_to_v2(payload_v1, bias=bias, device=device)
        packed_linear = PackedTCQLinear(payload_v2, mode="triton")
        parent_path, leaf_name = module_path.rsplit(".", 1)
        setattr(model.get_submodule(parent_path), leaf_name, packed_linear)
        del original_linear, payload_v1, payload_v2, bias

        if index % 25 == 0 or index == matrix_count:
            sync()
            elapsed = time.perf_counter() - conversion_start
            eta = elapsed / index * (matrix_count - index)
            print(f"  {index:3d}/{matrix_count} | {elapsed:7.1f}s | ETA {eta:7.1f}s")
            gc.collect()
            torch.cuda.empty_cache()

    del quantized

    model = model.to(device=device, dtype=torch.float16).eval()
    model.config.use_cache = True
    verify_tie(model)  # tie must survive the device transfer
    sync()

    packed_count = 0
    residual_linears = []
    for layer_index, block in enumerate(_decoder_layers(model)):
        for name, module in block.named_modules():
            full_name = f"model.layers.{layer_index}.{name}"
            if isinstance(module, PackedTCQLinear):
                packed_count += 1
                if module.mode != "triton":
                    raise RuntimeError(f"{full_name} is not in Triton mode.")
                if module.levels_packed is None:
                    raise RuntimeError(f"{full_name} has no packed level stream.")
                if module._w_cache is not None:
                    raise RuntimeError(f"{full_name} contains an FP16 weight cache.")
            elif isinstance(module, torch.nn.Linear):
                residual_linears.append(full_name)
    if residual_linears:
        raise RuntimeError(f"Unconverted decoder linears: {residual_linears[:10]}")

    return (
        model,
        metadata,
        {
            "load_seconds": time.perf_counter() - total_start,
            "conversion_seconds": time.perf_counter() - conversion_start,
            "conversion_peak_vram_gib": gib(torch.cuda.max_memory_allocated()),
            "packed_layer_count": packed_count,
        },
    )


# ----------------------------- decode helpers ------------------------------


def make_cache(model, max_length: int):
    args = {"max_cache_len": max_length, "device": model.device, "dtype": model.dtype}
    try:
        return StaticCache(config=model.config, max_batch_size=1, **args)
    except TypeError:
        return StaticCache(config=model.config, batch_size=1, **args)


@torch.inference_mode()
def eager_tokens(model, input_ids, token_count: int, max_length: int):
    cache = make_cache(model, max_length)
    prompt_length = input_ids.shape[1]
    logits = model(
        input_ids=input_ids, past_key_values=cache, use_cache=True,
        cache_position=torch.arange(prompt_length, device=model.device),
    ).logits
    token = logits[:, -1:].argmax(dim=-1)
    output = [token.clone()]
    for step in range(1, token_count):
        position = torch.tensor(
            [prompt_length + step - 1], device=model.device, dtype=torch.long
        )
        logits = model(
            input_ids=token, past_key_values=cache, use_cache=True, cache_position=position
        ).logits
        token = logits[:, -1:].argmax(dim=-1)
        output.append(token.clone())
    sync()
    return torch.cat(output, dim=1).cpu()


@torch.inference_mode()
def eager_speed(model, input_ids, token_count: int, max_length: int) -> float:
    measurements = []
    for _ in range(REPEATS):
        cache = make_cache(model, max_length)
        prompt_length = input_ids.shape[1]
        logits = model(
            input_ids=input_ids, past_key_values=cache, use_cache=True,
            cache_position=torch.arange(prompt_length, device=model.device),
        ).logits
        token = logits[:, -1:].argmax(dim=-1)
        sync()
        start = time.perf_counter()
        for step in range(1, token_count):
            position = torch.tensor(
                [prompt_length + step - 1], device=model.device, dtype=torch.long
            )
            logits = model(
                input_ids=token, past_key_values=cache, use_cache=True,
                cache_position=position,
            ).logits
            token = logits[:, -1:].argmax(dim=-1)
        sync()
        measurements.append((token_count - 1) / (time.perf_counter() - start))
    return float(statistics.median(measurements))


@torch.inference_mode()
def prepare_graph(model, input_ids, max_length: int):
    cache = make_cache(model, max_length)
    prompt_length = input_ids.shape[1]
    logits = model(
        input_ids=input_ids, past_key_values=cache, use_cache=True,
        cache_position=torch.arange(prompt_length, device=model.device),
    ).logits
    static_token = logits[:, -1:].argmax(dim=-1).clone()
    static_position = torch.tensor([prompt_length], device=model.device, dtype=torch.long)
    warmup_tokens = [static_token.clone()]

    def step():
        return model(
            input_ids=static_token, past_key_values=cache, use_cache=True,
            cache_position=static_position,
        ).logits

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            warmup_logits = step()
            static_token.copy_(warmup_logits[:, -1:].argmax(dim=-1))
            static_position.add_(1)
            warmup_tokens.append(static_token.clone())
    torch.cuda.current_stream().wait_stream(side_stream)
    sync()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_logits = step()
    sync()

    return {
        "graph": graph,
        "logits": graph_logits,
        "token": static_token,
        "position": static_position,
        "warmup": warmup_tokens,
        "cache": cache,
    }


@torch.inference_mode()
def graph_tokens(model, input_ids, token_count: int, max_length: int):
    state = prepare_graph(model, input_ids, max_length)
    output = list(state["warmup"])
    remaining = max(token_count - len(output), 0)
    for _ in range(remaining):
        state["graph"].replay()
        state["token"].copy_(state["logits"][:, -1:].argmax(dim=-1))
        state["position"].add_(1)
        output.append(state["token"].clone())
    sync()
    return torch.cat(output[:token_count], dim=1).cpu()


@torch.inference_mode()
def graph_speed(model, input_ids, token_count: int, max_length: int) -> float:
    measurements = []
    for _ in range(REPEATS):
        state = prepare_graph(model, input_ids, max_length)
        measured = max(token_count - len(state["warmup"]), 1)
        sync()
        start = time.perf_counter()
        for _ in range(measured):
            state["graph"].replay()
            state["token"].copy_(state["logits"][:, -1:].argmax(dim=-1))
            state["position"].add_(1)
        sync()
        measurements.append(measured / (time.perf_counter() - start))
        del state
    return float(statistics.median(measurements))


def benchmark(model, tokenizer) -> dict:
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)
    max_length = input_ids.shape[1] + NEW_TOKENS + 16
    eager_output = eager_tokens(model, input_ids, NEW_TOKENS, max_length)
    graph_output = graph_tokens(model, input_ids, NEW_TOKENS, max_length)
    return {
        "eager_tokens_per_second": eager_speed(model, input_ids, NEW_TOKENS, max_length),
        "graph_tokens_per_second": graph_speed(model, input_ids, NEW_TOKENS, max_length),
        "graph_matches_eager": bool(torch.equal(eager_output, graph_output)),
        "generated_text": tokenizer.decode(eager_output[0], skip_special_tokens=True),
    }


# ------------------------------ driver -------------------------------------


def main() -> None:
    if tp_version != VERSION:
        print(
            f"Note: TurboPress {tp_version} installed (script targets {VERSION}). "
            "It runs, but the fast v5 Triton kernel + CUDA-graph runtime landed in "
            "0.5.0 -- `pip install -U --no-deps turbopress` for the best packed speed."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This benchmark needs a GPU.")
    if not HAS_TRITON:
        raise RuntimeError("TurboPress Triton support is unavailable (HAS_TRITON is False).")

    # Optional Hugging Face token from Colab Secrets -- harmless no-op locally.
    try:
        from google.colab import userdata

        tok_secret = userdata.get("HF_TOKEN")
        if tok_secret:
            os.environ["HF_TOKEN"] = tok_secret
            os.environ["HUGGING_FACE_HUB_TOKEN"] = tok_secret
    except Exception:
        pass

    if not ARTIFACT_DIR.exists():
        if ARTIFACT_ZIP.exists():
            print(f"Extracting {ARTIFACT_ZIP} ...")
            ARTIFACT_DIR.parent.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.unpack_archive(str(ARTIFACT_ZIP), str(ARTIFACT_DIR.parent))
        else:
            raise FileNotFoundError(
                "TurboPress artifact not found.\n"
                f"Expected directory: {ARTIFACT_DIR}\nor ZIP archive: {ARTIFACT_ZIP}"
            )
    for path in (ARTIFACT_DIR / "turbopress_weights.pt", ARTIFACT_DIR / "hf_config",
                 ARTIFACT_DIR / "tokenizer"):
        if not path.exists():
            raise FileNotFoundError(f"Incomplete artifact: missing {path}")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 80)
    print("TURBOPRESS NATIVE PACKED RUNTIME")
    print("=" * 80)
    print(f"TurboPress:         {tp_version}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"Triton:             {triton.__version__}")
    print(f"CUDA:               {torch.version.cuda}")
    print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
    print(f"GPU VRAM:           {gib(torch.cuda.get_device_properties(0).total_memory):.2f} GiB")

    tokenizer = AutoTokenizer.from_pretrained(ARTIFACT_DIR / "tokenizer")
    report = {
        "environment": {
            "gpu": torch.cuda.get_device_name(0),
            "pytorch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "turbopress": tp_version,
        }
    }

    # ----------------------------- FP16 baseline ---------------------------
    if RUN_FP16_BASELINE:
        print("\n" + "=" * 80 + "\nFP16 BASELINE\n" + "=" * 80)
        clean()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()

        qcfg_path = ARTIFACT_DIR / "quantization_config.json"
        if qcfg_path.exists():
            source_model_id = json.loads(qcfg_path.read_text(encoding="utf-8"))["meta"]["model_id"]
        else:
            mb = torch.load(
                ARTIFACT_DIR / "turbopress_weights.pt", map_location="cpu", weights_only=False
            )
            source_model_id = mb["meta"]["model_id"]
            del mb

        fp16_model = (
            AutoModelForCausalLM.from_pretrained(
                source_model_id, dtype=torch.float16, low_cpu_mem_usage=True,
                attn_implementation="sdpa",
            )
            .to("cuda")
            .eval()
        )
        fp16_model.config.use_cache = True
        sync()
        load_seconds = time.perf_counter() - start
        fp16_decoder_memory = gib(decoder_bytes(fp16_model))
        fp16_loaded_vram = gib(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        fp16_runtime = benchmark(fp16_model, tokenizer)
        fp16_peak_vram = gib(torch.cuda.max_memory_allocated())

        report["fp16"] = {
            "load_seconds": load_seconds,
            "decoder_weight_gib": fp16_decoder_memory,
            "loaded_vram_gib": fp16_loaded_vram,
            "runtime_peak_vram_gib": fp16_peak_vram,
            **fp16_runtime,
        }
        print(f"Load time:             {load_seconds:.2f} s")
        print(f"Decoder weight memory: {fp16_decoder_memory:.3f} GiB")
        print(f"Loaded VRAM:           {fp16_loaded_vram:.3f} GiB")
        print(f"Runtime peak VRAM:     {fp16_peak_vram:.3f} GiB")
        print(f"Eager decode:          {fp16_runtime['eager_tokens_per_second']:.2f} tok/s")
        print(f"CUDA-graph decode:     {fp16_runtime['graph_tokens_per_second']:.2f} tok/s")
        print(f"Graph/eager match:     {fp16_runtime['graph_matches_eager']}")
        del fp16_model
        clean()

    # ------------------------ packed Triton benchmark ----------------------
    print("\n" + "=" * 80 + "\nTURBOPRESS PACKED TRITON RUNTIME\n" + "=" * 80)
    clean()
    packed_model, metadata, load_report = load_native(ARTIFACT_DIR)
    packed_decoder_memory = gib(decoder_bytes(packed_model))
    packed_loaded_vram = gib(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    packed_runtime = benchmark(packed_model, tokenizer)
    packed_peak_vram = gib(torch.cuda.max_memory_allocated())

    report["packed_triton"] = {
        **load_report,
        "decoder_weight_gib": packed_decoder_memory,
        "loaded_vram_gib": packed_loaded_vram,
        "runtime_peak_vram_gib": packed_peak_vram,
        **packed_runtime,
    }
    report["artifact"] = metadata

    print(f"Packed decoder memory: {packed_decoder_memory:.3f} GiB")
    print(f"Loaded VRAM:           {packed_loaded_vram:.3f} GiB")
    print(f"Runtime peak VRAM:     {packed_peak_vram:.3f} GiB")
    print(f"Eager decode:          {packed_runtime['eager_tokens_per_second']:.2f} tok/s")
    print(f"CUDA-graph decode:     {packed_runtime['graph_tokens_per_second']:.2f} tok/s")
    print(f"Graph/eager match:     {packed_runtime['graph_matches_eager']}")

    # ------------------------------- comparison ----------------------------
    if "fp16" in report:
        fp16_result, packed_result = report["fp16"], report["packed_triton"]
        report["comparison"] = {
            "decoder_memory_compression": (
                fp16_result["decoder_weight_gib"] / packed_result["decoder_weight_gib"]
            ),
            "loaded_vram_reduction": (
                fp16_result["loaded_vram_gib"] / packed_result["loaded_vram_gib"]
            ),
            "packed_to_fp16_eager_speed": (
                packed_result["eager_tokens_per_second"]
                / fp16_result["eager_tokens_per_second"]
            ),
            "packed_to_fp16_graph_speed": (
                packed_result["graph_tokens_per_second"]
                / fp16_result["graph_tokens_per_second"]
            ),
        }
        print("\n" + "=" * 80 + "\nFINAL COMPARISON\n" + "=" * 80)
        for name, value in report["comparison"].items():
            print(f"{name}: {value:.3f}x")

    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nPacked continuation:")
    print(PROMPT + packed_runtime["generated_text"])
    print(f"\nJSON report: {REPORT_PATH}")
    print("Completed successfully.")


if __name__ == "__main__":
    main()
