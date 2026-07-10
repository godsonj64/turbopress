"""Cloudflare R2 (S3-compatible) artifact storage: presigned download URLs."""

from __future__ import annotations

from service.config import Settings


def _client(settings: Settings):
    import boto3
    from botocore.config import Config

    endpoint = f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def presigned_get_url(settings: Settings, key: str) -> str | None:
    """A time-limited download URL for an artifact key, or None if R2 is off."""
    if not settings.r2_enabled or not key:
        return None
    return _client(settings).generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.r2_bucket, "Key": key},
        ExpiresIn=settings.r2_presign_ttl_seconds,
    )


def presigned_put_url(settings: Settings, key: str) -> str | None:
    """A time-limited upload URL (used by the GPU worker to push artifacts)."""
    if not settings.r2_enabled or not key:
        return None
    return _client(settings).generate_presigned_url(
        "put_object",
        Params={"Bucket": settings.r2_bucket, "Key": key},
        ExpiresIn=settings.r2_presign_ttl_seconds,
    )
