# TurboPress control plane

The hosted MVP: a FastAPI control plane that authenticates users, accepts
compression jobs, dispatches them to GPU workers, persists **signed
certificates**, meters usage to Stripe, and serves R2-presigned artifact URLs.

```
GitHub/HF OAuth ─▶ FastAPI control plane (Fly.io) ─▶ Postgres (Neon)
                        │ enqueue                        │ users, jobs, certs
                        ▼
                  Modal GPU workers ── artifacts ──▶ Cloudflare R2
                  (A100 serverless)                  (S3-compatible)
```

## Run locally (no GPU, no accounts)

```bash
pip install -e .                       # turbopress (for turbopress.certificate)
pip install -r service/requirements.txt
cp service/.env.example service/.env   # DEBUG=true, RUNNER=inline
cd service && uvicorn service.main:app --reload
```

With `RUNNER=inline` the control plane runs a fast, GPU-free stand-in that
exercises the whole lifecycle (sign → persist → gate → meter). Try it:

```bash
KEY=$(curl -s localhost:8000/signup -d '{"email":"me@example.com"}' \
  -H 'content-type: application/json' | python -c 'import sys,json;print(json.load(sys.stdin)["api_key"])')
curl -s localhost:8000/billing/dev-activate -X POST -H "authorization: Bearer $KEY"
curl -s localhost:8000/jobs -H "authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model_ref":"Qwen/Qwen3-0.6B","targets":["4bit","2bit"],"fidelity_gates":{"mean_kl_max":0.10}}'
```

Tests: `pytest service/tests`.

## API

| method | path | auth | purpose |
|--------|------|------|---------|
| POST | `/signup` | — | create a user, return an API key (once) |
| POST | `/billing/checkout` | API key | Stripe Checkout URL for the metered plan |
| POST | `/webhooks/stripe` | Stripe sig | activate subscription on payment |
| POST | `/jobs` | API key | submit a compression job |
| GET | `/jobs/{id}` | API key | job status + certificates + artifact URLs |
| GET | `/certificates/{id}` | — | public signed certificate manifest |
| POST | `/internal/jobs/{id}/complete` | worker secret | GPU worker result callback |

Fidelity gates are `<metric>_max` / `<metric>_min` (e.g. `mean_kl_max`,
`top1_agreement_min`); each certificate reports `gates_passed` and the
`compress-action` fails CI when a gate is unmet.

## Deploy (production)

1. **Postgres (Neon):** create a database, grab the connection string, and set
   `DATABASE_URL=postgresql+asyncpg://USER:PASS@HOST/db?sslmode=require`.
2. **Signing key:** `python -c "from turbopress.certificate import generate_signing_key as g;print(g())"`
   — keep the private half in secrets, publish the public half.
3. **Stripe:** create a metered recurring price; set `STRIPE_SECRET_KEY`,
   `STRIPE_PRICE_ID`, and `STRIPE_WEBHOOK_SECRET` (point the webhook at
   `/webhooks/stripe`).
4. **R2:** create a bucket + API token; set `R2_ACCOUNT_ID`,
   `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`.
5. **Worker:** deploy the Modal function (see `worker/README.md`) and set
   `RUNNER=modal`.
6. **Fly:**
   ```bash
   flyctl launch --no-deploy --copy-config --config service/fly.toml
   flyctl secrets set DATABASE_URL=... SIGNING_PRIVATE_KEY_B64=... \
     WORKER_CALLBACK_SECRET=... STRIPE_SECRET_KEY=... STRIPE_WEBHOOK_SECRET=... \
     STRIPE_PRICE_ID=... R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
   flyctl deploy . --config service/fly.toml --dockerfile service/Dockerfile
   ```

The `WORKER_CALLBACK_SECRET` and `SIGNING_PRIVATE_KEY_B64` must match the values
in the Modal `turbopress-worker` secret.

## Notes

- Tables auto-create on startup (MVP); switch to Alembic migrations before you
  have data you care about.
- GitHub/HF OAuth for a browser dashboard is scaffolded conceptually (API keys
  are the primary credential); add `authlib` routes when the dashboard lands.
