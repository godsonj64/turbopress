# TurboPress

**Certified low-bit LLM weight quantization.** TurboPress compresses the
decoder linears of a Hugging Face causal LM (Llama / Qwen / Mistral family) to
2–6 bits per weight with a measured, reproducible pipeline:

1. **Seeded randomized rotation** (block fast Walsh–Hadamard × stored signs) to
   decorrelate weight residuals from activation structure.
2. **Rotation-aware activation equilibration** — `s_j = m_j^α / c_j^(1/2−α)`
   with the derived optimal exponent: α = ¼ for a feedback-free quantizer
   (Proposition 1; not the AWQ square-root rule) and α = ⅛ with error
   feedback on (Proposition 2 — feedback absorbs activation structure).
3. **Trellis-coded quantization** with a data-free, trellis-optimized codebook
   (Viterbi search over a rate-1/2 convolutional code).
4. **Block-LDLQ error feedback over the trellis** (v0.5.0, on by default) —
   each column block's Hessian-weighted error feeds forward while the packed
   format stays identical. On SmolLM2-135M at 3 bits this cut KL 14.6% and
   perplexity 8.1% vs v0.4.2 at the same artifact size.

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
