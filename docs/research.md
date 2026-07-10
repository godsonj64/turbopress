# Research

TurboPress is measurement-driven. It builds on the primitives of
[TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874) — randomized
rotation, an MSE-optimal scalar quantizer, and a Quantized-JL residual sketch —
and reports what did and did **not** pay off.

## What was refuted

The original hypothesis was that TurboQuant's *unbiased* QJL residual correction,
applied to weights, would beat deterministic rounding at an equal bit budget.
It does **not**. Both in synthetic depth-compounding chains and on a real
pretrained model, equal-budget nearest-rounding won: the variance the correction
injects exceeds the bias it removes, and the random rotation already decorrelates
the residual from activation structure. The QJL-for-weights idea is measured
dead; the code path is kept runnable for reproduction only.

## What worked

Three additions each beat the rotated scalar baseline at equal bits:

- **Trellis-coded quantization (TCQ)** with a data-free, trellis-optimized
  Gaussian codebook — the clear winner at 2 bits.
- **Quarter-power activation equilibration** — the rotation-aware optimal fold
  `s_j = (E[x_j²] / ‖W_:,j‖²)^(1/4)`, a different exponent from the AWQ/BASE-Q
  square-root rule because the objective (weight-only, rotation-isotropic error)
  is different.
- **GPTQ/LDLQ error feedback** — Hessian-aware sequential rounding; a large, cheap
  win that roughly matches TCQ at 3–4 bits and is complementary at 2 bits.

Plus **per-layer bit allocation** by measured KL sensitivity: at an equal 3.0
bits/weight, mixed precision beats uniform.

## Measured results (Qwen3-0.6B, all 196 block linears, 4,096 tokens)

| config | bits/w | KL(fp‖q) | top-1 | ppl |
|--------|-------:|---------:|------:|----:|
| nearest 2b (TurboQuant-style) | 2.01 | 10.28 | 0.02 | 564,874 |
| nearest 4b | 4.01 | 0.399 | 0.670 | 43.9 |
| tcq 2b S=64 +eq | 2.02 | 1.215 | 0.450 | 91.4 |
| **tcq 4b S=64 +eq** | 4.02 | **0.079** | 0.843 | **30.6** |
| gptq 4b +eq (error feedback) | 4.02 | 0.095 | 0.844 | 30.5 |

fp16 reference perplexity: 29.11. Reproduce with
`python -m turbopress.real_model --model Qwen/Qwen3-0.6B`.

See the repository `README.md` for the full sweeps and the equilibration-exponent
ablation.
