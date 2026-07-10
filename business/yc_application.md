# YC Application Draft — TurboPress

> Draft for the standard YC application form. Numbers marked [PENDING] get
> filled from `results/vs_unsloth_qwen17b.json` when the head-to-head lands.
> Everything else is real, measured, and reproducible from this repo.

---

**Company name:** TurboPress

**Company URL:** turbopress.ai (to register)

**Describe what your company does in 50 characters or less.**
Certified LLM compression — with the receipt.

**What is your company going to make? Please describe your product and what it does or will do.**

TurboPress is CI/CD for model compression. Teams push a fine-tuned LLM; we
return the smallest version that provably behaves like the original — 2–4
bits per weight — together with a signed **fidelity certificate**: measured
KL divergence to the full-precision model, top-1 agreement, and benchmark
deltas on held-out data.

Today, quantization is vibes. Engineers grab a 4-bit GGUF from a stranger's
Hugging Face page and hope nothing broke. Nobody measures, nobody signs,
nobody is accountable when a quant silently degrades a production model. We
make compression a validated, repeatable pipeline step — like tests before a
deploy — across formats (our own TCQ codec, GGUF, AWQ, FP8), with per-layer
bit allocation tuned to each target device.

The pipeline is built on our own research: trellis-coded quantization with
analytic (data-free) codebooks over randomized rotations, plus a corrected
activation-scaling rule (a quarter-power law; the industry-standard
square-root fold is provably suboptimal under rotation — we measured it
being *worse than nothing* on Qwen3-1.7B at 3–4 bits).

**Why did you pick this idea to work on? Do you have domain expertise in this area? How do you know people need what you're making?**

I built the entire pipeline first and wrote the paper second. The repo
contains a 93-test research codebase, a preprint with 8 figures where every
number is generated from checked-in measurements, and a one-cell quantizer
that takes any Llama/Qwen/Mistral model to 2–4 bits on a single GPU and
ships a self-contained runnable artifact. Along the way we published a
genuine negative result (TurboQuant's celebrated unbiased-inner-product
trick does not transfer from KV-caches to weights) — the kind of honesty
that built Unsloth's brand on the training side.

Demand signal: Unsloth's quantized model mirrors alone have 150M+ downloads.
Every one of those downloads is someone accepting an *unvalidated* quant
because no validated alternative exists. Enterprises we target already pay
for eval platforms (Braintrust, LangSmith) — compression is the eval problem
they haven't noticed is unowned.

**What's new about what you're making? What substitutes do people resort to because it doesn't exist yet (or they don't know about it)?**

Substitutes: (1) grabbing community GGUFs and hoping; (2) running
llama.cpp's quantize or AutoAWQ themselves with default settings and no
measurement; (3) shipping fp16 and eating 4–8x the serving cost.

New: (a) the certificate — a signed, reproducible fidelity claim per
artifact; (b) per-hardware bit allocation driven by measured per-layer KL
sensitivity; (c) a data-free/calibration-light pipeline — our codebooks are
designed analytically because the rotation fixes the weight distribution, so
we can compress regulated-industry models **without ever seeing customer
data**; (d) compression as *recurring* CI (every fine-tune, every base-model
bump, every new target device), not a one-shot script.

**Who are your competitors? What do you understand about your business that other companies in it just don't get?**

Competitors: Unsloth (training-time; uploads quants as marketing, doesn't
certify or sell them), Neural Magic/Red Hat (llm-compressor; kernel-first,
enterprise-sales-first), llama.cpp ecosystem (formats, no accountability),
cloud providers' built-in quantization (opaque, single-format).

What they don't get: **the durable business is the trust layer, not the
codec.** Codecs get replicated — ours will be too. But the certification
flywheel (telemetry on which method/bits/allocation preserves which
architecture) compounds into a proprietary quality model that makes our
recommendations better every week, and a certificate is billable *per
deployment, forever*, in a way a kernel never is.

**How do or will you make money? How much could you make?**

Usage-based compression jobs ($50–500 per validated model by size), team
subscriptions for the compression CI ($500–2,000/mo: private models,
regression testing on base-model updates, fleet allocation), and enterprise
VPC/on-prem appliances ($100k+/yr, led by the no-customer-data compliance
story). Free tier = certified quants of public models (our distribution, not
our cost center — they double as marketing, Unsloth-style).

Market: inference cost is the #1 AI infra line item; every fine-tuned model
(millions/yr and growing) needs deployment compression 1–10x/yr. 300 CI
teams + 15 enterprises ≈ $5–7M ARR; the ceiling is "Docker Hub for deployed
models."

**How far along are you?**

- Working pipeline: rotation → quarter-power equilibration → trellis-coded
  quantization with data-free codebooks; GPTQ/LDLQ error feedback; per-layer
  bit allocation. 93 passing tests.
- Measured on real models: Qwen3-0.6B at 4 bits is within ~5% perplexity of
  fp16 (KL 0.079); 3-bit beats naive 4-bit; validated on Qwen3-1.7B on a
  laptop GPU.
- One-cell quantizer producing signed-manifest-ready artifacts with a
  standalone loader; verified bitwise round-trip from packed bits.
- Preprint + anonymized conference version written, figures fully
  reproducible from checked-in results.
- Head-to-head vs Unsloth Dynamic 2.0 GGUFs on identical tokens: [PENDING —
  results/vs_unsloth_qwen17b.json].

**What is your tech stack?**

PyTorch, Hugging Face transformers, custom quantization pipeline (pure
PyTorch today; Triton/CUDA packed-bit kernels on the roadmap), planned
control plane: FastAPI + Postgres + Modal/RunPod GPU workers + S3/R2
artifact store + Ed25519-signed manifests.

**Who writes code, or does other technical work on your product?**

Founder (Godson Johnson) — full pipeline, experiments, and paper.

**Why did you choose YC / what do you want from the batch?**

Unsloth (YC S24) proved this exact GTM motion works one step upstream of us
and is a natural partner (their users' fine-tunes are our inputs). We want
YC for the enterprise-design-partner intros and for pressure-testing
open-core pricing with partners who have lived the OSS monetization problem.

**Ask:** standard deal; funds go to first hire (inference-kernel engineer)
and GPU costs for the day-zero certified-quant pipeline.
