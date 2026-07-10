# TurboPress — Launch Plan (step-by-step, with technical deploys)

Goal: from this repo to a launched, waitlist-collecting, revenue-ready
product in ~12 weeks. Each phase lists concrete technical steps in order.
Solo-founder-sized; nothing here needs a team.

---

## Phase 0 — Repo goes public (Week 0–1)

The asset is credibility; the repo *is* the pitch.

1. **Version control & license**
   ```bash
   cd NOEQ && git init && git add -A
   git commit -m "TurboPress: certified LLM compression pipeline"
   # Add LICENSE (Apache-2.0 — permissive wins adoption; the moat is the
   # service, not the codec) and a NOTICE file.
   gh repo create turbopress/turbopress --public --source . --push
   ```
2. **Package for pip**
   - `pyproject.toml` already exists; add classifiers, keywords, `README.md`
     as long_description, console entry points:
     ```toml
     [project.scripts]
     turbopress = "turbopress.cli:main"
     ```
   - Write `turbopress/cli.py` wrapping what already exists:
     `turbopress compress <model> --bits 3` (the one-cell pipeline) and
     `turbopress validate <model_a> <model_b>` (the KL/top-1/ppl harness —
     this subcommand is the Trojan horse: useful even to GGUF/AWQ users).
   - Publish: `python -m build && twine upload dist/*` → `pip install turbopress`.
3. **CI**: GitHub Actions workflow — `pytest` (CPU, the 93-test suite runs in
   minutes), lint (`ruff`), on push + PR. Badge in README.
4. **Docs**: `mkdocs-material`, deployed via GitHub Pages
   (`mkdocs gh-deploy`). Pages: quickstart, one-cell notebook, certificate
   format spec, the research (link preprint), honest limitations.
5. **Paper**: put the preprint on arXiv (the submission zip is ready and
   verified to compile). The arXiv link is launch ammunition.

**Exit criteria:** `pip install turbopress` works; README shows the
benchmark table; arXiv link live.

## Phase 1 — Distribution flywheel + landing (Week 1–2)

1. **Certified public quants** (the Unsloth move, with receipts)
   - Create HF org `turbopress`. For 3–5 popular models (Qwen3-4B,
     Llama-3.2-3B, Mistral-7B…): run the one-cell pipeline on a rented GPU
     (RunPod A100 spot, ~$1.5/hr; each model is <1 hr), upload artifact +
     `certificate.json` + README with the measured table.
   - `certificate.json` = the signed manifest: model hash, method, bits,
     KL/top-1/ppl, eval-set hash, seed, pipeline version, Ed25519 signature
     (`pynacl`; publish the public key in the GitHub org).
2. **Landing page deploy** (the page in `business/landing/index.html`)
   - Buy `turbopress.ai` (Cloudflare Registrar).
   - Deploy: Cloudflare Pages — `wrangler pages deploy business/landing`
     (or connect the repo; zero build step, it's one HTML file).
   - Waitlist: swap the `mailto:` CTAs for a form posting to a Cloudflare
     Worker that appends to KV (30 lines), or embed a Tally form. Add
     Plausible analytics (`<script data-domain="turbopress.ai" ...>`), which
     requires moving the page off the strict-CSP artifact host to Pages.
3. **The badge**: an SVG shield "TurboPress verified · KL 0.03" served from
   `badge.turbopress.ai/<model>` (same Worker, reads the certificate KV).
   Every badge on a HF model card is an inbound link.

**Exit criteria:** landing live on turbopress.ai; ≥3 certified models on HF;
waitlist collecting.

## Phase 2 — Hosted compression CI, MVP (Week 3–6)

Architecture (boring on purpose):

```
GitHub/HF OAuth ──> FastAPI control plane (Fly.io) ──> Postgres (Neon)
                          │ enqueue                        │ jobs, certs, users
                          ▼
                    Modal GPU workers  ── artifacts ──> Cloudflare R2
                    (A10G/A100 serverless)               (S3-compatible)
```

1. **Control plane** — FastAPI app:
   - `POST /jobs` {model_ref, targets[], fidelity_gates} → row in Postgres,
     enqueue; `GET /jobs/:id` → status + logs; `GET /certificates/:id`.
   - Auth: GitHub OAuth (authlib), API keys per team (hashed in Postgres).
   - Deploy: `fly launch && fly deploy` (2 shared-cpu VMs, ~$10/mo).
2. **GPU workers** — Modal functions (serverless GPUs, pay-per-second):
   ```python
   @app.function(gpu="A100", timeout=3*3600, volumes={"/cache": hf_cache})
   def compress_job(job): ...   # runs scripts/turbopress_onecell.py logic
   ```
   - Steps inside the worker: pull weights → quantize (existing pipeline) →
     validate (existing harness) → sign manifest → upload artifact to R2 →
     callback to control plane. The one-cell script already is this worker
     minus the plumbing.
   - Cost model: a 8B model ≈ 1 A100-hour ≈ $2–4 → charge $99+; margin holds.
3. **Billing** — Stripe: metered usage (per job, by param count) + `Team`
   subscription. Stripe Checkout + webhooks → Postgres. No invoicing code.
4. **GitHub Action** — repo `turbopress/compress-action`: thin wrapper that
   calls the API and fails the workflow if fidelity gates fail:
   ```yaml
   - uses: turbopress/compress-action@v1
     with: { model: org/model, targets: "a10g:4bit", fail-below: "kl=0.10" }
   ```
5. **Artifact delivery**: R2 presigned URLs; artifacts are the existing
   zip format (packed weights + run_quantized.py + tokenizer/config +
   certificate.json).

**Exit criteria:** a stranger can OAuth in, pay, compress a private model,
and download a certified artifact without talking to you.

## Phase 3 — Launch (Week 7–9)

1. **Day-zero automation**: a scheduled Modal cron watching HF `models`
   RSS/API for major releases → auto-compress + certify + upload + tweet.
   This is the growth engine; budget ~$300/mo GPU.
2. **Launch sequence** (order matters):
   - Show HN: "TurboPress – certified LLM compression (with a signed KL
     receipt)" — lead with the open repo + the vs-Unsloth benchmark table.
   - r/LocalLLaMA: the head-to-head post (this community made Unsloth).
   - arXiv paper thread on X; tag quantization researchers.
   - YC application (draft in `business/yc_application.md`) — apply with the
     waitlist + download numbers from the first two weeks.
3. **Design-partner motion**: every waitlist signup with a company domain
   gets a hand-run compression of one of their models + certificate, free,
   in exchange for a 30-min call. Ten of these = pricing validation + logos.

## Phase 4 — Deepen the moat (Week 10+, funded or revenue-carried)

1. **Packed-bit inference kernels** (the year-one technical bet): Triton
   dequant-fused GEMM for the TCQ format → runtime VRAM savings, not just
   storage. Until then, keep exporting to GGUF/FP8 for serving.
2. **Telemetry flywheel**: every job logs (architecture, layer, method,
   bits) → measured fidelity. This dataset trains the allocation policy —
   recommendations get better with every job; competitors start from zero.
3. **Compliance productization**: certificate → PDF audit report; SOC2;
   the "zero-customer-data compression" white paper for regulated buyers.

## Budget (pre-revenue)

| item | monthly |
|---|---|
| Fly.io control plane + Neon + R2 | ~$30 |
| Modal GPU (day-zero pipeline + demos) | $300–600 |
| Domain, email (Fastmail), Plausible | ~$30 |
| **Total burn** | **< $700/mo** |

## The two honest risks, planned for

1. **"Why not just use free GGUFs?"** → the free tier *is* free GGUFs, but
   certified. Monetization only triggers on private models and CI — things
   community uploads structurally cannot serve.
2. **A lab ships built-in certified quantization** → speed: own the metric
   (the OSS validator), own the registry of certificates, and be
   format-agnostic so their codec becomes another export target, not a
   replacement.
