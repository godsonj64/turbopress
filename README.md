# turbopress

[![CI](https://github.com/godsonj64/turbopress/actions/workflows/ci.yml/badge.svg)](https://github.com/godsonj64/turbopress/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/turbopress.svg)](https://pypi.org/project/turbopress/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://godsonj64.github.io/turbopress/)

A production-quality research prototype of **TurboPress**: a post-training
LLM weight-quantization pipeline built on, and measured against, the
primitives of [TurboQuant (arXiv:2504.19874)](https://arxiv.org/abs/2504.19874)
(randomized rotation -> optimal scalar quantization -> QJL residual sketch).

Two development rounds, both driven by measurement:

* **Round 1 (stage 4 test):** does TurboQuant's unbiased QJL residual
  correction pay off for *weights*? **No** — refuted synthetically and on a
  real pretrained model (details below).
* **Round 2 (beyond TurboQuant):** three methods that beat the TurboQuant-style
  scalar baseline at equal bits — trellis-coded quantization (TCQ) with
  **data-free trellis-optimized codebooks**, and **rotation-aware
  equilibration** with a corrected optimal scaling rule. Verdict tables below.
* **Round 3 (scale-up + error feedback + mixed precision):** the pipeline moved
  to the GPU and a bigger model (Qwen3-0.6B); added **GPTQ/LDLQ error feedback**
  (Hessian-aware sequential rounding, the biggest known PTQ gain the pipeline
  was missing); and added **per-layer bit allocation** by measured KL
  sensitivity. Results immediately below.

## Paper

**TurboPress: A Measurement-Driven Study of Rotated Low-Bit Weight Quantization
— Equilibration Exponents, Error Feedback, and Per-Layer Bit Allocation.**
Godson Johnson. Preprint — read it here:
**[paper/turbopress_preprint.pdf](paper/turbopress_preprint.pdf)**
(figures regenerate from the raw result logs via
[`paper/make_figures.py`](paper/make_figures.py)).

> **Abstract.** We present TurboPress, an instrumented post-training
> quantization (PTQ) pipeline for LLM weights, with controlled A/B measurements
> that isolate the contribution of each design choice at *exactly matched* bit
> budgets. Building on incoherence processing (QuIP/QuaRot/QuIP#/QTIP) and on the
> rotation + optimal-quantizer construction of TurboQuant, TurboPress combines a
> seeded randomized orthogonal rotation, trellis-coded quantization with
> data-free trellis-optimized codebooks, GPTQ/LDLQ error feedback, and per-layer
> bit allocation. The contributions are primarily empirical and analytical
> rather than a new state of the art: (i) a corrected rotation-aware
> *equilibration exponent* — in the weight-only, post-rotation regime the optimal
> per-channel fold is a **quarter power** `s_j = (E[x_j²]/‖W_:,j‖²)^(1/4)`,
> distinct from the AWQ/SmoothQuant square root and from BASE-Q's
> weight–activation balance; (ii) a rigorous **negative result** that unbiased
> (QJL-style) residual correction, while provably debiasing, loses to a finer
> deterministic quantizer at equal budget for weights; and (iii) an end-to-end
> measurement on Qwen3-0.6B showing error feedback and trellis coding are
> complementary (trellis wins at 2 bits; both near-lossless at 4 bits), and that
> KL-sensitivity-driven mixed precision beats uniform precision at an equal
> 3.0-bit budget.

**Contributions**

1. **Corrected equilibration exponent (Proposition 1).** Under a rotation-based
   quantizer whose error is isotropic in the rotated basis, the expected output
   error factorizes as `(Σⱼ cⱼ sⱼ²)(Σⱼ mⱼ/sⱼ²)` with `cⱼ = ‖W_:,j‖²`,
   `mⱼ = E[x_j²]`; Cauchy–Schwarz gives the optimum `sⱼ = (mⱼ/cⱼ)^(1/4)` — a
   *quarter* power, not the square root used by AWQ/SmoothQuant, arising from a
   different objective than the 1/2 power BASE-Q derives for the joint
   weight+activation setting.
2. **Error feedback in rotated coordinates.** GPTQ/LDLQ implemented in the
   rotated, equilibrated basis via a fast transform of the activation Hessian
   `H_z = R D⁻¹ H D⁻¹ Rᵀ`; cuts 2-bit KL 5.0× and 3-bit KL 5.3× over nearest at
   ~3× lower cost than TCQ, and `gptq 3b` beats `nearest 4b` (a full bit saved).
3. **Per-layer bit allocation.** A measured KL-sensitivity probe plus a greedy
   budget knapsack beats uniform precision at an equal 3.0-bit budget
   (KL 0.238 vs 0.276).
4. **A refuted hypothesis.** Unbiased QJL residual correction for weights loses
   to deterministic quantization at equal budget, both synthetically and on a
   real model; the correction's variance exceeds the bias it removes.

## Install & CLI

```bash
pip install "turbopress[llm]"          # torch + transformers + datasets
# or, from a clone:  pip install -e ".[dev,llm]"
```

```bash
# quantize every decoder linear and write a certified, self-contained artifact
turbopress compress Qwen/Qwen3-4B --bits 3 --out ./out

# measure KL / top-1 agreement / perplexity between any two HF-loadable models
turbopress validate Qwen/Qwen3-4B ./out/Qwen3-4B-turbopress-3bit --out report.json
```

The full pipeline lives in [`turbopress/pipeline.py`](turbopress/pipeline.py)
(`compress()`); the notebook cell [`scripts/turbopress_onecell.py`](scripts/turbopress_onecell.py)
and the `turbopress compress` CLI share that one validated code path. Docs:
`mkdocs serve` (see [`docs/`](docs/)).

## Packed runtime (v0.4): run models *from* the bits

`turbopress/runtime.py` is the runtime analogue of GGUF/llama.cpp for the
TurboPress format: weights stay trellis-coded in memory and decode on the
fly, so resident weight memory is ~bits/16 of fp16 — not just the download.

The enabling property: **our shift-register trellis decodes window-parallel.**
The subset at position `t` is a pure LUT lookup on the path-bit window
`b[t-m..t]` (the encoder starts in state 0, so the window zero-pads) — no
sequential trellis walk. This is the property QTIP engineers deliberately
("bitshift trellis"); here it falls out of the encoder construction, and it
is enforced by a test that checks window decode against the sequential walk
bit-for-bit.

```python
from turbopress import pack_model
from turbopress.real_model import collect_input_scales

stats = collect_input_scales(model, calib_batches)
pack_model(model, stats, bits=3, mode="tiled")   # linears now run from packed bits
```

Three execution modes for `PackedTCQLinear`, measured on SmolLM2-135M
(3-bit, RTX 5050, greedy generation; all modes: 98% teacher-forced top-1
agreement with fp16):

| mode | decoder weight memory | tok/s | what it is |
|---|---:|---:|---|
| fp16 baseline | 202.5 MiB | 26.4 | — |
| `cached` | 202.8 MiB (1.0x) | 25.6 | load-time decompression: decode once, fold the rotation into fp16 weights, free the packed streams |
| `tiled` | **38.9 MiB (5.2x less)** | 2.5 | weights stay packed; row tiles decode per forward (pure PyTorch memory mode) |
| `triton` | 38.9 MiB (5.2x less) | — | fused decode-inside-GEMV (`turbopress/triton_kernel.py`): the fp16 matrix never exists; batch-1 GEMV is bandwidth-bound, so the theoretical ceiling is **16/bits x faster** than fp16 (server GPUs; no Triton wheel on this Windows dev box, so it ships kernel-complete but locally untested — the torch paths above are the tested reference) |

Reproduce: `python scripts/bench_runtime.py` -> `results/runtime_bench.json`.

## Round 3 results (Qwen3-0.6B, GPU, all 196 block linears quantized)

Token-level metrics vs the fp16 model (perplexity **29.11**) on 4,096 tokens of
real text; equilibration + Hessians calibrated on a disjoint 2,048-token slice.
Run: `python -m turbopress.real_model --model Qwen/Qwen3-0.6B` (RTX 5050, 8 GB).

| config                        | bits/w | KL(fp\|\|q) | top-1 | ppl |
|-------------------------------|-------:|---------:|------:|----------:|
| nearest 2b (TurboQuant-style) |  2.012 |   10.282 | 0.020 | 564,874 |
| nearest 3b                    |  3.013 |    1.641 | 0.381 |     138.7 |
| nearest 4b                    |  4.013 |    0.399 | 0.670 |      43.9 |
| tcq 2b S=64 +eq               |  2.023 |    1.215 | 0.450 |      91.4 |
| tcq 3b S=64 +eq               |  3.023 |    0.289 | 0.710 |      38.6 |
| **tcq 4b S=64 +eq**           |  4.023 | **0.079**| 0.843 |    **30.6** |
| gptq 2b +eq (error feedback)  |  2.023 |    2.072 | 0.383 |     177.1 |
| gptq 3b +eq (error feedback)  |  3.023 |    0.312 | 0.707 |      36.3 |
| **gptq 4b +eq (error feedback)** | 4.023 | 0.095 | 0.844 |    **30.5** |

Headlines:

* **The 4-bit "invisible" claim sharpens with scale, as predicted.** On 135M the
  best 4-bit model was +8.1% perplexity over fp (15.2 vs 14.06); on 0.6B it is
  **+4.6–5.2%** (30.5–30.6 vs 29.11) at KL ≈ 0.08–0.10. Bigger model, smaller
  relative loss — 4-bit is effectively lossless here.
* **2-bit improves meaningfully with scale.** The 2-bit perplexity *ratio* to fp
  drops from **8.3×** on 135M (116 / 14.06) to **3.1×** on 0.6B (91.4 / 29.11)
  for the same tcq+eq method.
* **Error feedback is a large, cheap win over the biased baseline.** GPTQ/LDLQ
  cuts 2-bit KL from 10.28 → **2.07** (5.0×) and 3-bit KL from 1.641 → **0.312**
  (5.3×) vs plain nearest rounding, at ~3× *lower* quantization cost than TCQ
  (79 s vs 246 s for the whole model). **gptq 3b beats nearest 4b** (KL 0.312 vs
  0.399) — a full bit saved from error feedback alone.
* **Trellis vs error feedback are complementary.** At 3–4 bits they are a
  statistical tie (tcq slightly better KL, gptq slightly better ppl); at the
  hardest **2-bit** setting the joint trellis code still wins clearly (KL 1.22
  vs 2.07), because that is where scalar codebook granularity hurts most. The
  obvious next step is to combine them (block-LDLQ *over* the trellis, QuIP#/
  QTIP-style) rather than pick one.

### GPTQ / LDLQ error feedback (`gptq.py`)

Nearest rounding leaves each layer's output error as a fixed bias with no
cross-channel cancellation. LDLQ (QuIP) / GPTQ quantize the input channels
sequentially and feed each channel's rounding error forward into the
not-yet-quantized channels, weighted by the layer's input Hessian
`H = E[x x^T]`, minimizing `tr((W − Ŵ) H (W − Ŵ)^T)` rather than raw weight MSE.
Crucially this runs in the pipeline's *rotated, equilibrated* coordinates — the
random rotation is exactly the incoherence preprocessing that makes LDLQ
effective — via `rotated_hessian`, which forms `H_z = R D^-1 H D^-1 R^T` with
the same fast transform R and equilibration folds D as the quantized layer. The
stored form and bit accounting are identical to the plain scalar path; only the
codeword each weight rounds to changes.

### Per-layer bit allocation by measured KL sensitivity (`allocate.py`)

Layers are not equally sensitive, so a uniform bit-width wastes budget. For every
quantized linear and each candidate width (2/3/4 bits) we quantize *only that
layer* and measure the resulting KL(fp‖quant) — a direct, model-faithful
sensitivity signal — then greedily spend a fixed average budget on the upgrades
with the best KL-drop-per-added-bit. Run:
`python -m turbopress.allocate --model Qwen/Qwen3-0.6B --targets 2.5,3.0`.

Mean single-layer 2-bit KL by projection type (why allocation helps): the MLP
projections dominate sensitivity, attention q/k are nearly free.

| down_proj | up_proj | gate_proj | o_proj | v_proj | q_proj | k_proj |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0088 | 0.0080 | 0.0062 | 0.0047 | 0.0047 | 0.0025 | 0.0016 |

Allocation vs uniform (TCQ+eq, Qwen3-0.6B, fp ppl 29.11):

| budget | model | bits/w | KL(fp\|\|q) | top-1 | ppl | mix (2/3/4 bit layers) |
|--------|-------|-------:|---------:|------:|-----:|------------------------|
| 3.0 (equal) | **mixed** | 3.023 | **0.238** | **0.723** | **34.8** | 24 / 149 / 23 |
| 3.0 (equal) | uniform 3b | 3.023 | 0.276 | 0.700 | 36.6 | 0 / 196 / 0 |
| 2.5 | mixed | 2.523 | 0.494 | 0.609 | 44.8 | 101 / 91 / 4 |
| — | uniform 2b | 2.023 | 1.386 | 0.416 | 105.2 | 196 / 0 / 0 |

* **Equal-budget win:** at exactly 3.0 bits/weight, allocating unevenly by
  measured KL sensitivity beats uniform 3-bit — **14% lower KL** (0.238 vs 0.276)
  and 5% lower perplexity — by moving 24 insensitive layers down to 2-bit and 23
  sensitive ones up to 4-bit.
* **Fractional budgets:** 2.5 bits/weight (impossible with a uniform integer
  width) lands KL 0.494, roughly *one third* of the way from uniform-2b (1.386)
  to uniform-3b — i.e. a half-bit spent well recovers most of the 2→3-bit gain.

Mixed precision is an established idea; here it is a measured system feature, and
the sensitivity table above is a reusable diagnostic (MLP-heavy sensitivity is
consistent with the wider literature).

## TurboPress v2 results (SmolLM2-135M, all 210 block linears quantized)

Token-level metrics vs the fp32 model (perplexity 14.06) on 4,096 tokens of
real text; equilibration is calibrated on a disjoint 2,048-token slice:

| config                  | bits/w | KL(fp\|\|q) | top-1 agree |        ppl |
|-------------------------|-------:|---------:|------------:|-----------:|
| nearest 3b (TurboQuant-style) | 3.024 | 1.647 |       0.381 |       74.6 |
| tcq 3b S=64             |  3.024 |    1.048 |       0.471 |       39.4 |
| **tcq 3b S=64 + equil** |  3.046 | **0.319** |   **0.700** |   **18.5** |
| nearest 2b (TurboQuant-style) | 2.024 | 9.188 |       0.011 |    145,393 |
| tcq 2b S=64             |  2.024 |    7.322 |       0.032 |     24,293 |
| **tcq 2b S=64 + equil** |  2.046 | **2.052** |   **0.325** |  **116.1** |
| **tcq 4b S=64 + equil** |  4.047 | **0.079** |   **0.844** | **15.2** |

Headlines, all at (essentially) equal bit budgets:

* **2 bits:** perplexity 145,393 -> 116 (a ~1250x improvement over the
  TurboQuant-style scalar quantizer), KL 4.5x lower.
* **3 bits:** KL 5.2x lower; the 3-bit combined method **beats the plain
  4-bit baseline** (KL 0.32 vs 0.40, ppl 18.5 vs 20.5) — a full bit saved.
* **4 bits:** ppl 15.2 vs fp32's 14.06 on a 135M model (the hardest size to
  quantize) — approaching "no visible loss" territory before any error
  feedback or recovery distillation.

### The three beyond-TurboQuant methods

1. **TCQ over rotated coordinates** (`trellis.py`): TurboQuant's scalar
   quantizer is near-optimal only in the *memoryless* class; trellis-coded
   quantization (Marcellin & Fischer 1990; QTIP 2024) codes whole rows
   jointly via Viterbi over an Ungerboeck-partitioned doubled codebook,
   attacking the ~2.7x gap to the rate-distortion bound. 8.8-27% lower
   distortion on N(0,1) at 2-3 bits; storage is exactly `bits`/weight
   (1 path bit + bits-1 member bits; proven by an encode->decode round-trip
   test).
2. **Data-free trellis-optimized codebooks**: the Lloyd-Max codebook is not
   optimal *under a trellis*. Because the rotation makes the coordinate
   distribution known (~N(0,1)), generalized Lloyd with Viterbi assignments
   runs offline on synthetic Gaussians — no calibration data, deterministic,
   cached.
3. **Rotation-aware equilibration** (`linear.py`): the AWQ/SmoothQuant fold
   `s_j = sqrt(E[x_j^2])` is provably suboptimal under a rotation-based
   quantizer. Rotation spreads quantization error uniformly, so the output
   error factorizes as `(sum_j c_j s_j^2)(sum_j m_j / s_j^2)` with
   `c_j = ||W_col_j||^2`, `m_j = E[x_j^2]`; Cauchy-Schwarz gives the optimum
   `s_j = (m_j / c_j)^(1/4)`. Empirically ~3x more effective than the sqrt
   fold; this is the pipeline's only data-dependent stage.

---

## Round 1: the QJL-for-weights hypothesis (refuted)

Weight PTQ methods (GPTQ, QuIP#, QTIP, ...) round deterministically, so each
layer's output error is a fixed *bias*. The hypothesis: biases compound through
depth worse than zero-mean noise, so spending bits on a QJL sketch that makes
each layer's output **unbiased** should beat spending the same bits on a finer
deterministic quantizer.

### What the experiments say (seed 0, `python -m turbopress.harness`)

### Single layer (d = 1024, per-output-row sketches, 24 trials)

| config             | bits/w | rel RMSE | rel bias |
|--------------------|-------:|---------:|---------:|
| nearest 4b         |  4.016 |    0.097 |    0.097 |
| nearest 3b         |  3.016 |    0.185 |    0.185 |
| nearest 2b         |  2.016 |    0.342 |    0.342 |
| stochastic 2b      |  2.016 |    0.435 |    0.224 |
| 2b + QJL k=d/8     |  2.156 |    1.211 |    0.247 |
| 2b + QJL k=d (=3b) |  3.031 |    0.428 |    0.087 |

The theory is confirmed exactly: at an equal ~3-bit budget the QJL correction
cuts systematic error by 2.1x (0.185 -> 0.087), and the measured noise matches
the analytical variance `(pi/2)(d/k) * MSE_2b` to within a few percent.

### Depth compounding (16-layer ReLU chain + classifier head, fully quantized)

| config             | bits/w | err@L1 | err@L16 | KL(fp\|\|q) | top-1 agree |
|--------------------|-------:|-------:|--------:|---------:|------------:|
| nearest 3b         |  3.031 |  0.181 |    0.50 |    0.096 |        0.98 |
| nearest 2b         |  2.031 |  0.328 |    0.76 |    0.216 |        0.36 |
| stochastic 2b      |  2.031 |  0.413 |    0.72 |    0.199 |        0.19 |
| 2b + QJL k=d (=3b) |  3.063 |  0.407 |    2.29 |    2.581 |        0.01 |
| 2b + QJL k=d/8     |  2.188 |  1.121 | 1681.79 |  6023.61 |        0.00 |

(isotropic activations shown; the anisotropic run is nearly identical.)

### Real pretrained model, round-1 sweep (SmolLM2-135M)

Token-level metrics vs the fp32 model on 4,096 tokens of real text (fp32
perplexity 13.85 in this run; the QJL/stochastic configs remain runnable via
`LEGACY_CONFIGS` in `real_model.py`):

| config             | bits/w | KL(fp\|\|q) | top-1 agree |        ppl |
|--------------------|-------:|---------:|------------:|-----------:|
| nearest 4b         |  4.024 |    0.431 |       0.639 |       21.2 |
| nearest 3b         |  3.024 |    1.717 |       0.364 |       82.9 |
| nearest 2b         |  2.024 |    9.094 |       0.010 |     138336 |
| stochastic 2b      |  2.024 |   11.427 |       0.028 |    1454608 |
| 2b + QJL k=d/8     |  2.172 |   14.444 |       0.003 |   29663051 |
| 2b + QJL k=d (=3b) |  3.047 |    9.750 |       0.018 |     293934 |

Residual connections and RMSNorm do damp the compounding (no synthetic-style
blow-up: KL ~10, not ~6000), **but the verdict does not flip**: at an equal
~3-bit budget the QJL-corrected model (KL 9.75) is ~5.7x worse than plain
nearest 3-bit (KL 1.72), and the correction does not even beat uncorrected
2-bit. The absolute numbers also show that rotation + optimal scalar
quantization alone is far from usable at <=3 bits on a 135M model (small
models are the hardest to quantize) — which is precisely why stages 1, 3, 5,
and 6 of the plan (activation-aware scaling, trellis coding, error feedback,
recovery distillation) carry the real weight.

### Findings — the hypothesis is refuted in this setting

1. **The QJL correction removes bias exactly as the theory predicts** (unit
   tests verify unbiasedness and the variance bound statistically), **but the
   variance it adds exceeds the bias it removes.** The estimator's noise std is
   `sqrt(pi/2) * ||r|| * ||x|| / sqrt(k)`; even at k = d (a full extra
   bit/weight) that is ~1.25x the typical uncorrected error `<r, x>`, and the
   deterministic 3-bit quantizer at the same total budget has 2.3x lower output
   RMSE. Exactly this 2.3x appears in the measurements.
2. **Variance compounds worse than bias, not better.** The correction noise
   scales with the norm of the incoming (already-corrupted) activations, which
   creates multiplicative error feedback through depth: k=d/8 explodes; even
   k=d loses badly to plain 3-bit at equal budget. Stochastic rounding — the
   classical unbiased baseline — also fails to beat nearest rounding end-to-end.
3. **The random rotation already neutralizes activation structure.** Adding a
   dominant fixed mean direction and low-rank structure to the activations
   (mimicking LLM massive activations) barely changes any number: the seeded
   rotation decorrelates quantization residuals from activation structure, so
   the "systematic bias times a repeated activation pattern" failure mode the
   correction was designed for does not survive the rotation.

**Implication for the TurboPress research plan:** the bit budget for weight
quantization is better spent on stronger *deterministic* quantizers (trellis /
vector codebooks, i.e. stage 3) plus cross-layer error feedback (stage 5) and
light recovery training (stage 6). The QJL correction remains the right tool
where TurboQuant proved it — KV-cache / attention inner products, where errors
aggregate across many queries and the error path is shallow. A Wiener-shrunk
correction (`corr * B^2/(B^2+V)`) is MSE-optimal but by the same algebra still
loses to nearest 3-bit at equal budget; measuring it is a one-line extension.

The synthetic chain is a worst case (no skips, no LayerNorm), so the A/B was
escalated to a real pretrained model (table above): compounding is indeed
damped by the architecture, but the equal-budget ranking is unchanged. The
negative result stands in both settings.

## Package layout

- `turbopress/hadamard.py` — seeded randomized orthogonal transform R
  (block-diagonal fast Walsh-Hadamard x Rademacher signs; exact dense QR
  fallback for odd dims). Guarantees `(W R^T)(R x) = W x`.
- `turbopress/codebooks.py` — Lloyd-Max optimal scalar codebooks for N(0,1),
  computed to fixed point in float64 and validated against closed forms.
- `turbopress/quantizer.py` — row-wise quantization with nearest rounding
  (+ alternating scale refinement) or stochastic rounding.
- `turbopress/trellis.py` — trellis-coded quantization: Ungerboeck trellises
  from standard rate-1/2 convolutional generators, vectorized Viterbi
  encoding, data-free trellis-optimized codebooks, and `decode_levels` for
  the bit-stream round-trip proof.
- `turbopress/qjl.py` — 1-bit QJL residual sketch; unbiased inner-product
  estimator with proof and variance bound in the docstring.
- `turbopress/gptq.py` — GPTQ/LDLQ error feedback for the rotated scalar
  quantizer: `rotated_hessian` maps a raw activation Hessian into the
  rotated/equilibrated coordinates, and `ldlq_quantize_rows` runs the
  Hessian-weighted sequential rounding. Reduces to plain nearest rounding when
  the Hessian is diagonal (proven in tests).
- `turbopress/linear.py` — `QJLCorrectedLinear.from_linear(nn.Linear, bits,
  sketch_k, seed, rounding, method="scalar"|"tcq", n_states, col_scale,
  error_feedback, hessian)`, a drop-in quantized layer with rotation-aware
  equilibration, optional LDLQ error feedback, and exact bit accounting via
  `storage_report()`. Device-agnostic (CPU/CUDA).
- `turbopress/allocate.py` — per-layer bit allocation: `measure_sensitivity`
  probes each layer's KL at each candidate width, `allocate_bits` does the
  greedy knapsack under an average-bit budget, `build_mixed_model` assembles
  the mixed-precision network; `python -m turbopress.allocate` runs the full
  compare vs uniform.
- `turbopress/harness.py` — the two synthetic experiments; `--quick` for a
  smoke run; results saved to `results/harness_results.json`.
- `turbopress/real_model.py` — the real-model sweep: quantizes every linear in
  the decoder blocks of a Hugging Face causal LM (embeddings/head/norms stay
  fp) and measures KL / top-1 / perplexity against the reference model on real
  text. Runs on GPU (`--device cuda --dtype float16`) and any Llama/Qwen-family
  model; collects calibration scales + Hessians as needed.
- `tests/` — 98 tests: exact math identities, statistical unbiasedness and
  variance-scaling tests with explicit confidence bounds, trellis round-trip
  and rate-distortion checks (CPU + CUDA), equilibration optimality checks,
  LDLQ Hessian-loss-reduction and diagonal-Hessian equivalence, greedy
  allocation properties, module contracts, harness smoke test, and real-model
  plumbing tests on a tiny in-memory Llama (no download needed).

## Usage

```python
import torch
from torch import nn
from turbopress import QJLCorrectedLinear

layer = nn.Linear(4096, 4096)
act_rms = torch.ones(4096)  # sqrt(E[x_j^2]) from your calibration set
qlayer = QJLCorrectedLinear.from_linear(
    layer, bits=2, method="tcq", n_states=64, col_scale=act_rms, seed=0
)
y = qlayer(torch.randn(8, 4096))
print(qlayer.storage_report())  # true bits/weight incl. scales + equil
```

```bash
python -m pytest tests                                    # full suite (98 tests)
python -m turbopress.harness                              # synthetic experiments, ~4 s on CPU
python -m turbopress.real_model --model Qwen/Qwen3-0.6B   # GPU sweep incl. TCQ + GPTQ, ~20 min
python -m turbopress.allocate  --model Qwen/Qwen3-0.6B    # per-layer bit allocation vs uniform
python -m turbopress.ldlq_micro                           # LDLQ vs nearest microbenchmark
```

### One-cell quantizer (Jupyter / server GPU)

`scripts/turbopress_onecell.py` is a thin notebook cell over the packaged
pipeline (`from turbopress.pipeline import compress`): after `pip install
"turbopress[llm]"` you can paste it into a single cell on a GPU box, or just run
`turbopress compress <model> --bits 3`. Either way it runs the same validated
code path and quantizes any Llama/Qwen/Mistral-family HF model with the
TurboPress pipeline
(rotate -> quarter-power equilibration -> analytic TCQ), validates KL / top-1
/ perplexity against fp16 on WikiText-2, logs to console + file, and writes a
downloadable artifact: bit-packed weights at true rate, a standalone
`run_quantized.py` loader/demo (with `--export-hf` to emit a plain fp16 HF
checkpoint), tokenizer/config, and measured metrics — plus a zip of the whole
folder. Configure via the `os.environ.setdefault(...)` lines at the top, any
`TP_*` env var, or the `turbopress compress` flags.
Verified end-to-end: the packed artifact reloads to a bitwise-identical model
(state-dict compared tensor by tensor in the built-in self-test).

## Scope notes

This is a *measurement* implementation: codes/signs are stored at true widths
for accounting, but the forward pass materializes the dequantized weight (in the
model dtype) rather than using packed low-bit kernels — numerics are identical to
a packed kernel; only prototype memory/latency are not representative. Runs on
CPU or a single GPU (`--device`/`--dtype`); everything is seeded and
deterministic on a given device. Model sizes are bounded by holding the
reference and quantized copies together in memory (Qwen3-0.6B fp16 fits an 8 GB
card comfortably).

## Hosted service (compression as a service)

Beyond the library, the repo contains a hosted MVP so a stranger can pay and
compress a private model with no human in the loop, and gate CI on measured
fidelity:

- **[`service/`](service/)** — FastAPI control plane (API-key auth, `/jobs`,
  signed `/certificates`, Stripe metered billing, R2-presigned artifacts). Runs
  locally with a GPU-free inline runner: `pytest service/tests`.
- **[`worker/`](worker/)** — Modal serverless GPU function that runs the real
  pipeline, uploads artifacts to Cloudflare R2, signs a certificate, and calls
  the control plane back.
- **[`action/`](action/)** — the `compress-action` GitHub Action that fails the
  build when fidelity gates (`mean_kl_max`, `top1_agreement_min`, …) are not met.

Certificates are Ed25519-signed; verify with `turbopress.certificate.verify_certificate`
(`pip install "turbopress[sign]"`). Architecture and deploy steps are in
[`service/README.md`](service/README.md) and the [hosted docs](docs/hosted.md).

## Related work & positioning

TurboPress does not claim a new frontier method; the pipeline shape (rotation →
strong codebook → error feedback → mixed precision) overlaps substantially with
recent work. What is defensible here are the two analytical points — the
quarter-power equilibration exponent and the QJL-for-weights negative result —
plus the controlled, equal-budget measurements.

- **Rotation / incoherence processing:** QuIP, QuIP#, QuaRot, SpinQuant — random
  or learned rotations make weights incoherent and render rotated coordinates
  approximately Gaussian (the regime where the Lloyd–Max scalar quantizer is
  optimal).
- **Quantizers:** trellis-coded quantization (Marcellin–Fischer; Ungerboeck set
  partitioning) applied to LLM weights by QTIP; PVQ is another route.
- **Error feedback:** OBQ/GPTQ and LDLQ (QuIP) quantize channels sequentially
  through the inverse Hessian; Qronos generalizes the correction step.
- **Activation-aware scaling:** SmoothQuant and AWQ rescale by a power of
  activation magnitude (AWQ grid-searches α∈[0.25, 0.75]); BASE-Q derives a
  square-root scale for the *weight-and-activation* setting. Our quarter power is
  the weight-only, post-rotation optimum — a different objective — and it
  explains why the empirically favored AWQ exponents and QAM-W's frozen α=0.3
  cluster near 1/4. Concurrent recipes (QAM-W, OrbitQuant, PiSO) cover adjacent
  ground; the exponent for this specific regime is, to our knowledge, unstated
  elsewhere. Full references are in [`paper/references.bib`](paper/references.bib).

**Limitations** (see the paper for detail): evidence is at ≤0.6B parameters (the
2-bit ranking may shift at 7B+); comparisons are against our own controlled
configurations, not released GPTQ/AWQ/QuIP#/QTIP checkpoints on WikiText-2 and
zero-shot suites; the equilibration result rests on stated isotropic-error
modeling assumptions (an approximation, not an exact theorem); and this is a
measurement implementation — dequantized weights are materialized in the model
dtype, so numerics but not latency match a packed kernel.

## Citation

If you use TurboPress, please cite:

```bibtex
@misc{johnson2026turbopress,
  title  = {TurboPress: A Measurement-Driven Study of Rotated Low-Bit Weight
            Quantization -- Equilibration Exponents, Error Feedback, and
            Per-Layer Bit Allocation},
  author = {Johnson, Godson},
  year   = {2026},
  note   = {Preprint},
  url    = {https://github.com/godsonj64/turbopress}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
