"""FastAPI control plane app and routes."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from service import billing
from service.auth import authenticate, generate_api_key
from service.config import Settings, get_settings
from service.db import get_session, get_sessionmaker, init_db
from service.jobs import create_job, persist_results, to_job_out
from service.models import ApiKey, Certificate, Job, User
from service.runner import build_runner
from service.schemas import (
    CheckoutResponse,
    JobCompletion,
    JobCreateRequest,
    JobOut,
    SignupRequest,
    SignupResponse,
)
from turbopress.certificate import generate_signing_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not settings.signing_private_key_b64:
        # Ephemeral dev key; set TP signing key in production for stable certs.
        settings.signing_private_key_b64, _ = generate_signing_key()
    await init_db()
    app.state.settings = settings
    app.state.runner = build_runner(settings, get_sessionmaker())
    yield


app = FastAPI(title="TurboPress control plane", version="0.1.0", lifespan=lifespan)


def settings_dep() -> Settings:
    return get_settings()


async def _load_job(session: AsyncSession, job_id: str) -> Job | None:
    return (
        await session.execute(
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.certificates))
        )
    ).scalar_one_or_none()


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.post("/signup", response_model=SignupResponse)
async def signup(
    req: SignupRequest,
    session: AsyncSession = Depends(get_session),
) -> SignupResponse:
    existing = (
        await session.execute(select(User).where(User.email == req.email))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = User(email=req.email)
    session.add(user)
    await session.flush()
    full_key, prefix, key_hash = generate_api_key()
    session.add(ApiKey(user_id=user.id, prefix=prefix, key_hash=key_hash))
    await session.commit()
    return SignupResponse(user_id=user.id, email=user.email, api_key=full_key)


@app.post("/billing/checkout", response_model=CheckoutResponse)
async def billing_checkout(
    user: User = Depends(authenticate),
    settings: Settings = Depends(settings_dep),
) -> CheckoutResponse:
    if not settings.stripe_enabled:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "billing is not configured")
    return CheckoutResponse(checkout_url=billing.create_checkout_session(settings, user))


@app.post("/billing/dev-activate")
async def billing_dev_activate(
    user: User = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(settings_dep),
) -> dict:
    """Dev-only shortcut to mark billing active without a real Stripe payment."""
    if not settings.debug:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
    db_user = await session.get(User, user.id)
    db_user.billing_active = True
    await session.commit()
    return {"billing_active": True}


@app.post("/webhooks/stripe")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(default=""),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(settings_dep),
) -> dict:
    if not settings.stripe_enabled:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "billing is not configured")
    payload = await request.body()
    try:
        event = billing.parse_webhook(settings, payload, stripe_signature)
    except Exception as exc:  # noqa: BLE001 - surface as 400
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"invalid webhook: {exc}"
        ) from exc

    etype = event["type"]
    obj = event["data"]["object"]
    if etype in ("checkout.session.completed", "customer.subscription.created",
                 "customer.subscription.updated"):
        user_id = (obj.get("metadata") or {}).get("user_id")
        user = await session.get(User, user_id) if user_id else None
        if user is None:
            customer_id = obj.get("customer")
            if customer_id:
                user = (
                    await session.execute(
                        select(User).where(User.stripe_customer_id == customer_id)
                    )
                ).scalar_one_or_none()
        if user is not None:
            user.billing_active = True
            if obj.get("customer"):
                user.stripe_customer_id = obj["customer"]
            item_id = _subscription_item_id(obj)
            if item_id:
                user.subscription_item_id = item_id
            await session.commit()
    return {"received": True}


def _subscription_item_id(obj: dict) -> str | None:
    items = (obj.get("items") or {}).get("data") or []
    if items:
        return items[0].get("id")
    return None


@app.post("/jobs", response_model=JobOut, status_code=status.HTTP_201_CREATED)
async def submit_job(
    req: JobCreateRequest,
    request: Request,
    user: User = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(settings_dep),
) -> JobOut:
    job = await create_job(session, settings, user, req)
    await request.app.state.runner.enqueue(job.id)
    async with get_sessionmaker()() as read:
        fresh = await _load_job(read, job.id)
        return to_job_out(settings, fresh)


@app.get("/jobs/{job_id}", response_model=JobOut)
async def get_job(
    job_id: str,
    user: User = Depends(authenticate),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(settings_dep),
) -> JobOut:
    job = await _load_job(session, job_id)
    if job is None or job.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return to_job_out(settings, job)


@app.get("/certificates/{cert_id}")
async def get_certificate(
    cert_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Public: return the signed certificate manifest for a target."""
    cert = await session.get(Certificate, cert_id)
    if cert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "certificate not found")
    return cert.certificate


@app.post("/internal/jobs/{job_id}/complete", response_model=JobOut)
async def complete_job(
    job_id: str,
    completion: JobCompletion,
    x_worker_secret: str = Header(default=""),
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(settings_dep),
) -> JobOut:
    """Worker callback: persist results, evaluate gates, meter usage."""
    if x_worker_secret != settings.worker_callback_secret:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "bad worker secret")
    job = await _load_job(session, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    user = await session.get(User, job.user_id)
    await persist_results(session, settings, job, completion, user)
    fresh = await _load_job(session, job_id)
    return to_job_out(settings, fresh)
