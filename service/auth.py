"""API-key issuance and verification.

Keys look like ``tp_<prefix>_<secret>``. We store only ``sha256(secret)`` and a
lookup ``prefix``; the full key is shown to the user exactly once.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from service.db import get_session
from service.models import ApiKey, User


def _hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str, str]:
    """Return ``(full_key, prefix, key_hash)``."""
    prefix = "tp_" + secrets.token_hex(4)
    secret = secrets.token_urlsafe(32)
    full = f"{prefix}_{secret}"
    return full, prefix, _hash_secret(secret)


def parse_api_key(full_key: str) -> tuple[str, str] | None:
    """Split ``tp_<prefix>_<secret>`` into ``(prefix, secret)``."""
    parts = full_key.split("_")
    if len(parts) < 3 or parts[0] != "tp":
        return None
    prefix = "_".join(parts[:2])  # "tp_<prefix>"
    secret = "_".join(parts[2:])
    return prefix, secret


async def authenticate(
    authorization: str | None = Header(default=None),
    session: AsyncSession = Depends(get_session),
) -> User:
    """FastAPI dependency: resolve a Bearer API key to its owning user."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer API key",
        )
    parsed = parse_api_key(authorization[7:].strip())
    if parsed is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "malformed API key")
    prefix, secret = parsed
    row = (
        await session.execute(select(ApiKey).where(ApiKey.prefix == prefix))
    ).scalar_one_or_none()
    if row is None or row.revoked or row.key_hash != _hash_secret(secret):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    user = (
        await session.execute(select(User).where(User.id == row.user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid API key")
    return user
