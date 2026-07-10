"""Signed compression certificates.

A *certificate* is a small JSON manifest describing a quantization run (model,
method, bits, measured KL/top-1/perplexity, seed, pipeline version, and content
hashes) together with an Ed25519 signature over its canonical serialization.
The signer holds a private key; anyone can verify with the embedded public key.

The signing/verification helpers import ``pynacl`` lazily, so the rest of the
package (and ``import turbopress``) never depends on it. Install the extra with
``pip install "turbopress[sign]"``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone
from typing import Any

ALG = "ed25519"
SCHEMA = "turbopress/certificate@1"

__all__ = [
    "ALG",
    "SCHEMA",
    "build_manifest",
    "canonical_json",
    "generate_signing_key",
    "public_key_for",
    "sha256_hex",
    "sign_manifest",
    "verify_certificate",
]


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s)


def canonical_json(obj: Any) -> bytes:
    """Deterministic UTF-8 JSON (sorted keys, no whitespace) — the signed bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate_signing_key() -> tuple[str, str]:
    """Return ``(private_key_b64, public_key_b64)`` for a fresh Ed25519 key."""
    from nacl.signing import SigningKey

    sk = SigningKey.generate()
    return _b64e(bytes(sk)), _b64e(bytes(sk.verify_key))


def public_key_for(private_key_b64: str) -> str:
    """Derive the base64 public key from a base64 Ed25519 private seed."""
    from nacl.signing import SigningKey

    sk = SigningKey(_b64d(private_key_b64))
    return _b64e(bytes(sk.verify_key))


def build_manifest(
    *,
    model_ref: str,
    method: str,
    bits: int,
    bits_per_weight: float,
    metrics: dict[str, Any],
    seed: int,
    pipeline_version: str,
    eval_hash: str | None = None,
    artifact_sha256: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the canonical manifest dict that gets signed."""
    return {
        "schema": SCHEMA,
        "model_ref": model_ref,
        "method": method,
        "bits": int(bits),
        "bits_per_weight": round(float(bits_per_weight), 4),
        "metrics": {
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()
        },
        "seed": int(seed),
        "pipeline_version": pipeline_version,
        "eval_hash": eval_hash,
        "artifact_sha256": artifact_sha256,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def sign_manifest(manifest: dict[str, Any], private_key_b64: str) -> dict[str, Any]:
    """Sign ``manifest`` and return a certificate ``{manifest, alg, public_key, signature}``."""
    from nacl.signing import SigningKey

    sk = SigningKey(_b64d(private_key_b64))
    signature = sk.sign(canonical_json(manifest)).signature
    return {
        "manifest": manifest,
        "alg": ALG,
        "public_key": _b64e(bytes(sk.verify_key)),
        "signature": _b64e(signature),
    }


def verify_certificate(cert: dict[str, Any]) -> bool:
    """True iff ``cert``'s signature matches its manifest under its public key."""
    from nacl.exceptions import BadSignatureError
    from nacl.signing import VerifyKey

    try:
        vk = VerifyKey(_b64d(cert["public_key"]))
        vk.verify(canonical_json(cert["manifest"]), _b64d(cert["signature"]))
        return True
    except (BadSignatureError, KeyError, ValueError):
        return False
