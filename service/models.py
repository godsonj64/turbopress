"""Database models: users, API keys, jobs, and certificates."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    # Billing (Stripe).
    stripe_customer_id: Mapped[str | None] = mapped_column(String(64), default=None)
    subscription_item_id: Mapped[str | None] = mapped_column(String(64), default=None)
    billing_active: Mapped[bool] = mapped_column(Boolean, default=False)

    api_keys: Mapped[list[ApiKey]] = relationship(back_populates="user")
    jobs: Mapped[list[Job]] = relationship(back_populates="user")


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    prefix: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    key_hash: Mapped[str] = mapped_column(String(64))  # sha256 hex of the secret
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)

    user: Mapped[User] = relationship(back_populates="api_keys")


# Job status values: queued -> running -> succeeded | failed
class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    model_ref: Mapped[str] = mapped_column(String(256))
    # e.g. ["4bit", "3bit"] — TurboPress bit-width targets.
    targets: Mapped[list] = mapped_column(JSON, default=list)
    # e.g. {"mean_kl_max": 0.10, "top1_agreement_min": 0.70}
    fidelity_gates: Mapped[dict] = mapped_column(JSON, default=dict)
    private: Mapped[bool] = mapped_column(Boolean, default=True)

    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(String(1024), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    user: Mapped[User] = relationship(back_populates="jobs")
    certificates: Mapped[list[Certificate]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )


class Certificate(Base):
    __tablename__ = "certificates"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    target: Mapped[str] = mapped_column(String(16))
    bits_per_weight: Mapped[float] = mapped_column(Float)
    param_count: Mapped[int] = mapped_column(Integer, default=0)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)

    artifact_key: Mapped[str | None] = mapped_column(String(512), default=None)
    artifact_bytes: Mapped[int] = mapped_column(Integer, default=0)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), default=None)

    # Signed manifest {manifest, alg, public_key, signature}.
    certificate: Mapped[dict] = mapped_column(JSON, default=dict)
    gates_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    gate_failures: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    job: Mapped[Job] = relationship(back_populates="certificates")
