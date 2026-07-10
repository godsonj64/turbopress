# ============================================================================
#  TurboPress -- one-cell LLM weight quantizer  (paste this whole cell & run)
# ============================================================================
#  Takes any medium Hugging Face causal LM (Llama / Qwen / Mistral family),
#  quantizes every decoder linear to BITS bits/weight with the TurboPress
#  pipeline -- seeded randomized rotation -> quarter-power activation
#  equilibration -> trellis-coded quantization (Viterbi, analytic data-free
#  codebook) -- then validates against the full-precision model (KL, top-1,
#  perplexity), logs everything, and writes a downloadable artifact:
#
#      <model>-turbopress-<bits>bit/
#        turbopress_weights.pt     packed codes at true bit-width
#        run_quantized.py          standalone loader / demo (no repo needed)
#        quantization_config.json  settings + measured metrics
#        hf_config/  tokenizer/    everything needed to run offline
#        turbopress.log            full run log
#      + the same folder zipped for download.
#
#  Requirements: single CUDA GPU with enough VRAM for the model in fp16
#  (a 4B model needs ~10 GB; validation briefly holds eval logits on CPU).
#
#  The full pipeline now lives in the installable package so the notebook cell,
#  the `turbopress compress` CLI, and CI all share one validated code path:
#
#      pip install "turbopress[llm]"        # or: pip install -e ".[llm]"
#
#  Config: edit the env vars below (or set them before importing), or run the
#  CLI directly:  turbopress compress Qwen/Qwen3-4B --bits 3
# ============================================================================

import os

# --- config: override any TP_* env var before the import below --------------
os.environ.setdefault("TP_MODEL_ID", "Qwen/Qwen3-4B")  # Llama/Qwen/Mistral-family LM
os.environ.setdefault("TP_BITS", "3")                  # 2..6 bits per weight
os.environ.setdefault("TP_N_STATES", "64")             # 16 = faster, 64 = best
os.environ.setdefault("TP_OUT_DIR", "turbopress_out")

from turbopress.pipeline import compress, config_from_env  # noqa: E402

RESULT = compress(config_from_env())
print(RESULT["metrics"])
