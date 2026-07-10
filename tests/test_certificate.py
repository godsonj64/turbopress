"""Tests for signed certificates (requires the [sign] extra: pynacl)."""

import pytest

pytest.importorskip("nacl")

from turbopress.certificate import (  # noqa: E402
    build_manifest,
    generate_signing_key,
    public_key_for,
    sha256_hex,
    sign_manifest,
    verify_certificate,
)


def _manifest():
    return build_manifest(
        model_ref="Qwen/Qwen3-0.6B",
        method="tcq+eq",
        bits=4,
        bits_per_weight=4.023,
        metrics={"mean_kl": 0.079, "top1_agreement": 0.843, "ppl_q": 30.6},
        seed=0,
        pipeline_version="0.3.0",
        artifact_sha256=sha256_hex(b"artifact"),
    )


def test_sign_and_verify_roundtrip():
    priv, pub = generate_signing_key()
    cert = sign_manifest(_manifest(), priv)
    assert cert["alg"] == "ed25519"
    assert cert["public_key"] == pub == public_key_for(priv)
    assert verify_certificate(cert) is True


def test_tamper_is_detected():
    priv, _ = generate_signing_key()
    cert = sign_manifest(_manifest(), priv)
    cert["manifest"]["bits"] = 2  # forge a better-looking number
    assert verify_certificate(cert) is False


def test_wrong_key_is_rejected():
    priv, _ = generate_signing_key()
    other_priv, other_pub = generate_signing_key()
    cert = sign_manifest(_manifest(), priv)
    cert["public_key"] = other_pub  # claim a different signer
    assert verify_certificate(cert) is False


def test_sha256_hex_known_value():
    assert sha256_hex(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
