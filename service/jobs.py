"""Job lifecycle: creation, fidelity-gate evaluation, result persistence."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from service import billing
from service.config import Settings
from service.models import Certificate, Job, User
from service.schemas import (
    CertificateOut,
    JobCompletion,
    JobCreateRequest,
    JobOut,
)
from service.storage import presigned_get_url


def evaluate_gates(
    metrics: dict[str, float], gates: dict[str, float]
) -> tuple[bool, list[str]]:
    """Check metrics against ``*_max`` / ``*_min`` thresholds.

    ``{"mean_kl_max": 0.1}`` requires ``metrics["mean_kl"] <= 0.1``;
    ``{"top1_agreement_min": 0.7}`` requires ``metrics["top1_agreement"] >= 0.7``.
    A gate whose metric is missing is treated as a failure.
    """
    failures: list[str] = []
    for gate, threshold in (gates or {}).items():
        if gate.endswith("_max"):
            metric = gate[:-4]
            value = metrics.get(metric)
            if value is None or value > threshold:
                failures.append(f"{metric}={value} exceeds max {threshold}")
        elif gate.endswith("_min"):
            metric = gate[:-4]
            value = metrics.get(metric)
            if value is None or value < threshold:
                failures.append(f"{metric}={value} below min {threshold}")
        else:
            failures.append(f"unknown gate '{gate}' (use <metric>_max or <metric>_min)")
    return (len(failures) == 0, failures)


def is_public_model(model_ref: str, private: bool) -> bool:
    """A model is treated as public when the caller says so (not private)."""
    return not private


async def create_job(
    session: AsyncSession,
    settings: Settings,
    user: User,
    req: JobCreateRequest,
) -> Job:
    """Create a queued job, enforcing billing for private models."""
    needs_billing = req.private or settings.require_billing_for_public_models
    if needs_billing and settings.stripe_enabled and not user.billing_active:
        from fastapi import HTTPException, status

        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            "active subscription required; POST /billing/checkout first",
        )
    job = Job(
        user_id=user.id,
        model_ref=req.model_ref,
        targets=list(req.targets),
        fidelity_gates=dict(req.fidelity_gates),
        private=req.private,
        status="queued",
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def persist_results(
    session: AsyncSession,
    settings: Settings,
    job: Job,
    completion: JobCompletion,
    user: User | None = None,
) -> Job:
    """Store worker results as certificates, evaluate gates, meter usage."""
    if completion.error:
        job.status = "failed"
        job.error = completion.error[:1024]
        await session.commit()
        await session.refresh(job)
        return job

    all_passed = True
    total_params = 0
    for r in completion.results:
        passed, failures = evaluate_gates(r.metrics, job.fidelity_gates)
        all_passed = all_passed and passed
        total_params += int(r.param_count)
        session.add(
            Certificate(
                job_id=job.id,
                target=r.target,
                bits_per_weight=r.bits_per_weight,
                param_count=r.param_count,
                metrics=r.metrics,
                artifact_key=r.artifact_key,
                artifact_bytes=r.artifact_bytes,
                artifact_sha256=r.artifact_sha256,
                certificate=r.certificate,
                gates_passed=passed,
                gate_failures=failures,
            )
        )

    # The job "succeeds" if it ran; gate failures are reported per-certificate
    # and surfaced to CI via the GitHub Action, not treated as a job error.
    job.status = "succeeded"
    job.error = None
    await session.commit()

    if user is not None and total_params > 0:
        billing.record_usage(settings, user, total_params)

    await session.refresh(job)
    return job


def to_job_out(settings: Settings, job: Job) -> JobOut:
    certs = list(job.certificates)
    gates_passed: bool | None = None
    if certs:
        gates_passed = all(c.gates_passed for c in certs)
    return JobOut(
        id=job.id,
        model_ref=job.model_ref,
        targets=job.targets,
        fidelity_gates=job.fidelity_gates,
        status=job.status,
        error=job.error,
        gates_passed=gates_passed,
        certificates=[
            CertificateOut(
                id=c.id,
                target=c.target,
                bits_per_weight=c.bits_per_weight,
                param_count=c.param_count,
                metrics=c.metrics,
                gates_passed=c.gates_passed,
                gate_failures=c.gate_failures,
                artifact_url=presigned_get_url(settings, c.artifact_key)
                if c.artifact_key
                else None,
                certificate=c.certificate,
            )
            for c in certs
        ],
    )
