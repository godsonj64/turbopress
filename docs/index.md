# TurboPress

**Certified low-bit LLM weight quantization.** TurboPress compresses the
decoder linears of a Hugging Face causal LM (Llama / Qwen / Mistral family) to
2–6 bits per weight with a measured, reproducible pipeline:

1. **Seeded randomized rotation** (block fast Walsh–Hadamard × stored signs) to
   decorrelate weight residuals from activation structure.
2. **Quarter-power activation equilibration** — the rotation-aware optimal fold
   `s_j = (E[x_j²] / ‖W_:,j‖²)^(1/4)` (not the AWQ square-root rule).
3. **Trellis-coded quantization** with a data-free, trellis-optimized codebook
   (Viterbi search over a rate-1/2 convolutional code).

Every run **validates against the full-precision model** and reports KL, top-1
next-token agreement, and perplexity, then writes a self-contained artifact
that reloads without the repo.

```bash
pip install "turbopress[llm]"
turbopress compress Qwen/Qwen3-4B --bits 3
turbopress validate Qwen/Qwen3-4B ./turbopress_out/Qwen3-4B-turbopress-3bit
```

See the [Quickstart](quickstart.md) to get running, the [CLI](cli.md) reference
for every flag, and [Research](research.md) for what was measured — including
what was **refuted** (unbiased QJL residual correction of weights) as well as
what worked.

!!! note "Research prototype"
    TurboPress is an actively developed research prototype. Read the
    [Limitations](limitations.md) before relying on it in production.
