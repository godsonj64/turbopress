"""Modal serverless GPU worker.

Runs the real TurboPress pipeline for each requested bit-width, uploads each
artifact zip to Cloudflare R2, signs a certificate, and posts the results back
to the control plane's ``/internal/jobs/{id}/complete`` endpoint.

Secrets (callback secret, Ed25519 signing key, R2 credentials) come from a Modal
Secret named ``turbopress-worker``; only the job spec travels in the spawn
payload. Deploy with ``modal deploy worker/modal_app.py`` (see worker/README.md).
"""

from __future__ import annotations

import modal

# Build an image that has torch + transformers and the local turbopress package.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.40",
        "datasets>=2.0",
        "safetensors>=0.4",
        "numpy>=1.24",
        "pynacl>=1.5",
        "boto3>=1.34",
    )
    # Copy the repo and install turbopress from source so certificate.py + the
    # pipeline are available even though PyPI 0.3.0 predates certificate.py.
    .add_local_dir(".", "/repo", copy=True, ignore=["turbopress_out", ".git", "site"])
    .run_commands("pip install /repo")
)

app = modal.App("turbopress-worker")

# `modal secret create turbopress-worker WORKER_CALLBACK_SECRET=... \
#   SIGNING_PRIVATE_KEY_B64=... R2_ACCOUNT_ID=... R2_ACCESS_KEY_ID=... \
#   R2_SECRET_ACCESS_KEY=... R2_BUCKET=turbopress-artifacts`
worker_secret = modal.Secret.from_name("turbopress-worker")

hf_cache = modal.Volume.from_name("turbopress-hf-cache", create_if_missing=True)


def _bits_of(target: str) -> int:
    digits = "".join(ch for ch in target if ch.isdigit())
    return int(digits) if digits else 4


def _post_completion(base_url: str, job_id: str, body: dict, secret: str) -> None:
    import json
    import urllib.request

    url = f"{base_url.rstrip('/')}/internal/jobs/{job_id}/complete"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Worker-Secret": secret},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()


@app.function(
    image=image,
    gpu="A100",
    timeout=3 * 3600,
    secrets=[worker_secret],
    volumes={"/cache": hf_cache},
)
def compress_job(spec: dict) -> dict:
    import hashlib
    import os
    from pathlib import Path

    import boto3

    from turbopress.certificate import build_manifest, sign_manifest
    from turbopress.pipeline import compress

    os.environ.setdefault("HF_HOME", "/cache/hf")

    job_id = spec["job_id"]
    model_ref = spec["model_ref"]
    targets = spec["targets"]
    base_url = spec["callback_base_url"]
    callback_secret = os.environ["WORKER_CALLBACK_SECRET"]
    signing_key = os.environ["SIGNING_PRIVATE_KEY_B64"]

    r2_bucket = os.environ.get("R2_BUCKET", "turbopress-artifacts")
    r2 = boto3.client(
        "s3",
        endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )

    try:
        results = []
        for target in targets:
            bits = _bits_of(target)
            out = compress(
                {
                    "MODEL_ID": model_ref,
                    "BITS": bits,
                    "OUT_DIR": f"/tmp/{job_id}",
                    "DEVICE": "cuda",
                }
            )
            zip_path = Path(out["zip"])
            data = zip_path.read_bytes()
            sha = hashlib.sha256(data).hexdigest()
            key = f"{job_id}/{target}/{zip_path.name}"
            r2.put_object(Bucket=r2_bucket, Key=key, Body=data)

            manifest = build_manifest(
                model_ref=model_ref,
                method="rotate+quarter-power-eq+TCQ",
                bits=bits,
                bits_per_weight=out["bits_per_weight"],
                metrics=out["metrics"],
                seed=0,
                pipeline_version=os.environ.get("PIPELINE_VERSION", "0.3.0"),
                artifact_sha256=sha,
            )
            cert = sign_manifest(manifest, signing_key)
            results.append(
                {
                    "target": target,
                    "bits_per_weight": out["bits_per_weight"],
                    "param_count": int(out.get("n_params", 0)),
                    "metrics": out["metrics"],
                    "artifact_key": key,
                    "artifact_bytes": len(data),
                    "artifact_sha256": sha,
                    "certificate": cert,
                }
            )
        _post_completion(base_url, job_id, {"results": results}, callback_secret)
        return {"ok": True, "targets": len(results)}
    except Exception as exc:  # noqa: BLE001 - report failure back to the control plane
        _post_completion(base_url, job_id, {"error": str(exc)[:1000]}, callback_secret)
        raise
