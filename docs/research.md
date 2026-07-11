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

## Round 4: error feedback *over* the trellis, and Proposition 2

Round 3 left the two strongest methods separate: TCQ (joint trellis coding,
wins at 2 bits) and LDLQ (Hessian-aware error feedback, large cheap win).
Round 4 composes them, QTIP-style: columns are trellis-coded in blocks (the
Viterbi encoder chains its state across blocks, so the packed bit-stream is
format-identical to plain TCQ) and each block's Hessian-weighted error feeds
forward through the block-LDL factors of `H_z⁻¹`
(`turbopress.gptq.ldlq_tcq_quantize_rows`, `TP_ERROR_FEEDBACK` in the
pipeline).

Two honest measurement notes from the first head-to-head (SmolLM2-135M):

- **tcq+ef vs tcq is a statistical tie on KL** (slightly worse) while top-1
  agreement is consistently ~2 points *better* at every bit-width, at half
  the encode cost (one sweep instead of two). Scalar GPTQ loses to both.
- **Scale refit interacts destructively with error feedback at 2 bits**
  (KL 2.24 → 4.19 when the refit re-sweep was added): the refit fits scales
  to codes that encode feedback-*compensated* rows, the mis-fit scale clips
  the small codebook, and the feedback loop amplifies rather than cancels
  the error. The EF path therefore defaults to no refit.

### Proposition 2 (error feedback changes the optimal equilibration exponent)

Proposition 1 minimizes `(Σ_j c_j s_j²) · tr(H_ζ)` over the fold `s`
(`c_j = ‖W_:,j‖²`, `m_j = E[x_j²]`, `tr(H_ζ) = Σ_j m_j/s_j²`), giving
`s_j = (m_j/c_j)^{1/4}`. Under LDLQ the proxy loss replaces `tr(H_ζ)` with
`tr(D_L)` — the trace of the diagonal factor of the LDL decomposition of
`H_ζ`, i.e. only the *innovation* variance that error feedback cannot
exploit. Two exact facts: `tr(D_L) ≥ d · det(H_ζ)^{1/d}` (AM–GM, tight when
the incoherence rotation flattens the pivots) and
`det(H_ζ) = det(H) / Π_j s_j²` (rotation-independent). In that ideal-EF
limit the objective becomes

```
F(s) = (Σ_j c_j s_j²) · d · det(H)^{1/d} · Π_j s_j^{-2/d}
```

whose stationarity condition is `c_j s_j² = const`:
**`s_j ∝ 1/‖W_:,j‖` — pure weight-column normalization; the activation
statistics drop out entirely**, because error feedback already absorbs all
linearly-predictable activation structure through the Hessian. The three
regimes lie on the line `α + β = ½` for `s_j = m_j^α · c_j^{-β}`:

| quantizer regime | optimal fold | α (on m) | β (on c) |
|---|---|---:|---:|
| no rotation (AWQ/SmoothQuant objective) | `m^{1/2}` | ½ | 0 |
| rotation, no EF (**Prop 1**) | `(m/c)^{1/4}` | ¼ | ¼ |
| rotation + ideal EF (**Prop 2**) | `c^{-1/2}` | 0 | ½ |

Real feedback is imperfect, so the falsifiable prediction is: *with EF on,
the optimal α slides from ¼ toward 0; without EF it stays at ¼.*

**Confirmed** (SmolLM2-135M, all 210 block linears, KL(fp‖q) on 4,080
held-out tokens):

| KL(fp‖q) | α = 0 | α = ⅛ | α = ¼ | optimum |
|---|---:|---:|---:|---|
| tcq 3b (control, no EF) | 0.758 | 0.358 | **0.321** | ¼ — stays put (Prop 1) |
| tcq+ef 3b | 0.431 | **0.276** | 0.325 | ⅛ — shifted toward 0 |
| tcq+ef 2b | 2.139 | **1.312** | 1.785 | ⅛ — shifted toward 0 |

The control's optimum stays at ¼ and error feedback moves it to ≈⅛ — midway
to the ideal-EF limit, i.e. real LDLQ absorbs roughly half the activation
structure. The correction is not academic: at the shifted exponent, tcq+ef
beats the best feedback-free configuration by **14% KL at 3 bits** and
**36% KL at 2 bits** at the same rate (the earlier apparent tie between
tcq and tcq+ef was an artifact of running the feedback at the feedback-free
exponent). The pipeline therefore defaults to α = ⅛ when error feedback is
on and ¼ otherwise (`TP_EQUIL_ALPHA` overrides). Reproduce with
`python -m turbopress.real_model --config-set prop2` (the fold is exposed as
`equil_alpha` on `QJLCorrectedLinear.from_linear`).
