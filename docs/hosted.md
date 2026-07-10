# Hosted service

Beyond the OSS library, TurboPress ships a hosted control plane so a stranger can
pay and compress a private model with **no human in the loop** — and gate their
CI on measured fidelity.

```
GitHub/HF OAuth ─▶ FastAPI control plane (Fly.io) ─▶ Postgres (Neon)
                        │ enqueue                        │ users, jobs, certs
                        ▼
                  Modal GPU workers ── artifacts ──▶ Cloudflare R2
                  (A100 serverless)                  (S3-compatible)
```

## Pieces

- **`service/`** — FastAPI control plane: API-key auth, `/jobs`, signed
  `/certificates`, Stripe metered billing, R2-presigned artifact URLs. Runs
  locally with an inline (GPU-free) runner. See
  [service/README.md](https://github.com/godsonj64/turbopress/blob/main/service/README.md).
- **`worker/`** — Modal serverless GPU function that runs the real pipeline,
  uploads artifacts to R2, signs a certificate, and calls the control plane back.
- **`action/`** — the `compress-action` GitHub Action: compresses a model in CI
  and **fails the build when fidelity gates are not met**.

## Signed certificates

Each target produces an Ed25519-signed manifest (model, method, bits, measured
KL/top-1/perplexity, seed, pipeline version, artifact hash). Verify anywhere:

```python
from turbopress.certificate import verify_certificate
assert verify_certificate(cert)   # cert = GET /certificates/{id}
```

Install the signing extra with `pip install "turbopress[sign]"`.

## Fidelity gates in CI

```yaml
- uses: godsonj64/turbopress/action@v1
  with:
    api-key: ${{ secrets.TURBOPRESS_API_KEY }}
    model: your-org/your-model
    targets: "4bit,3bit"
    gates: "mean_kl_max=0.10,top1_agreement_min=0.70"
```

The job fails if any target's `mean_kl` exceeds `0.10` or `top1_agreement` falls
below `0.70`, so a quantization regression can't merge silently.
