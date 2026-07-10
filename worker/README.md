# TurboPress GPU worker (Modal)

Serverless GPU function that runs the real TurboPress pipeline, uploads each
artifact to R2, signs a certificate, and calls the control plane back.

## Deploy

```bash
pip install modal
modal token new                     # one-time auth

# Secrets the worker reads at runtime:
modal secret create turbopress-worker \
  WORKER_CALLBACK_SECRET=<same as control plane> \
  SIGNING_PRIVATE_KEY_B64=<Ed25519 private seed, base64> \
  R2_ACCOUNT_ID=<...> R2_ACCESS_KEY_ID=<...> R2_SECRET_ACCESS_KEY=<...> \
  R2_BUCKET=turbopress-artifacts \
  PIPELINE_VERSION=0.3.0

# From the repo root (so the image can `pip install /repo`):
modal deploy worker/modal_app.py
```

Generate the signing key once and keep the private half only in the Modal secret;
publish the public half so anyone can verify certificates:

```bash
python -c "from turbopress.certificate import generate_signing_key as g; priv,pub=g(); print('PRIVATE',priv); print('PUBLIC',pub)"
```

## How it's invoked

The control plane's `ModalJobRunner` calls `compress_job.spawn({job_id,
model_ref, targets, callback_base_url})`. The worker does the rest and POSTs
results to `{callback_base_url}/internal/jobs/{job_id}/complete` with the
`X-Worker-Secret` header. Set `RUNNER=modal` on the control plane to use it.

## Cost

A ~8B model is roughly one A100-hour (~$2–4). Charge accordingly; the metered
Stripe usage is recorded per quantized parameter count on success.
