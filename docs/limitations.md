# Limitations

TurboPress is a research prototype. Known limitations, stated plainly:

- **Model families.** Tested on Llama / Qwen / Mistral-family causal LMs. The
  decoder-layer locator handles `model.layers`, `transformer.h`, and
  `model.decoder.layers`; other architectures are unsupported.
- **Dimensions.** Each linear's `in_features` must have a power-of-2 factor for
  the block Hadamard rotation. Layers without one are unsupported.
- **Storage, not yet runtime.** The artifact stores weights at the true low
  bit-width, but `run_quantized.py` decodes them to **fp16 in memory** to run on
  stock `transformers`. There is no packed-bit inference kernel yet, so inference
  VRAM is fp16-sized. Export to GGUF/FP8 for serving.
- **Evaluation scale.** Reported KL / top-1 / perplexity use a few thousand
  tokens of WikiText-2 (or a public-domain fallback). These are consistent,
  reproducible relative metrics — not a substitute for full downstream
  benchmark suites.
- **Certificates are unsigned.** `quantization_config.json` records settings and
  measured metrics but is not yet cryptographically signed. Re-verify with
  `turbopress validate`.
- **Scale of validation.** Method comparisons here are on small models
  (≤~0.6B). Behavior at 7B+ is expected to follow the same trends but is not yet
  measured end-to-end in this repo.
- **Codebook assumes near-Gaussian rotated weights.** The data-free codebook is
  optimized for `N(0,1)`; the rotation makes rotated weights approximately
  Gaussian, but heavy-tailed layers can deviate.
