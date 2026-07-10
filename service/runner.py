"""Job runners: enqueue compression work to a GPU worker.

- ``ModalJobRunner`` spawns the serverless GPU function (production). The worker
  compresses, signs, uploads to R2, and calls back to ``/internal/...``.
- ``InlineJobRunner`` runs a fast, GPU-free stand-in in-process and completes the
  job synchronously. It exercises the entire lifecycle (sign -> persist -> gate ->
  meter) for local development and CI without downloading models or renting GPUs.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from service.config import Settings
from service.jobs import persist_results
from service.models import Job, User
from service.schemas import JobCompletion, TargetResult
from turbopress.certificate import build_manifest, sha256_hex, sign_manifest


class JobRunner(Protocol):
    async def enqueue(self, job_id: str) -> None: ...


def _bits_of(target: str) -> int:
    digits = "".join(ch for ch in target if ch.isdigit())
    return int(digits) if digits else 4


# Representative fidelity by bit-width (from measured Qwen3-0.6B runs), used only
# by the inline dev runner so its certificates carry plausible numbers.
_MODEL_FIDELITY = {
    2: {"mean_kl": 1.215, "top1_agreement": 0.450, "ppl_q": 91.4},
    3: {"mean_kl": 0.289, "top1_agreement": 0.710, "ppl_q": 38.6},
    4: {"mean_kl": 0.079, "top1_agreement": 0.843, "ppl_q": 30.6},
}


class InlineJobRunner:
    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], settings: Settings):
        self._sessionmaker = sessionmaker
        self._settings = settings

    async def enqueue(self, job_id: str) -> None:
        async with self._sessionmaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = "running"
            await session.commit()
            user = (
                await session.execute(select(User).where(User.id == job.user_id))
            ).scalar_one_or_none()

            results = []
            for target in job.targets:
                bits = _bits_of(target)
                metrics = dict(_MODEL_FIDELITY.get(bits, _MODEL_FIDELITY[4]))
                bpw = bits + 0.023
                artifact_key = f"{job.id}/{target}/turbopress.zip"
                artifact_sha256 = sha256_hex(f"{job.model_ref}:{target}".encode())
                manifest = build_manifest(
                    model_ref=job.model_ref,
                    method="tcq+eq (inline-dev)",
                    bits=bits,
                    bits_per_weight=bpw,
                    metrics=metrics,
                    seed=0,
                    pipeline_version=self._settings.pipeline_version,
                    artifact_sha256=artifact_sha256,
                )
                cert = sign_manifest(manifest, self._settings.signing_private_key_b64)
                results.append(
                    TargetResult(
                        target=target,
                        bits_per_weight=bpw,
                        param_count=600_000_000,
                        metrics=metrics,
                        artifact_key=artifact_key,
                        artifact_bytes=0,
                        artifact_sha256=artifact_sha256,
                        certificate=cert,
                    )
                )
            await persist_results(
                session, self._settings, job, JobCompletion(results=results), user
            )


class ModalJobRunner:
    def __init__(self, settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]):
        self._settings = settings
        self._sessionmaker = sessionmaker

    async def enqueue(self, job_id: str) -> None:
        import modal

        async with self._sessionmaker() as session:
            job = await session.get(Job, job_id)
            if job is None:
                return
            job.status = "running"
            spec = {
                "job_id": job.id,
                "model_ref": job.model_ref,
                "targets": list(job.targets),
                "callback_base_url": self._settings.public_base_url,
            }
            await session.commit()

        fn = modal.Function.from_name(
            self._settings.modal_app_name, self._settings.modal_function_name
        )
        # Fire-and-forget: the worker reads its secrets (callback secret, signing
        # key, R2 creds) from Modal secrets and calls /internal/jobs/{id}/complete.
        fn.spawn(spec)


def build_runner(
    settings: Settings, sessionmaker: async_sessionmaker[AsyncSession]
) -> JobRunner:
    if settings.runner == "modal":
        return ModalJobRunner(settings, sessionmaker)
    return InlineJobRunner(sessionmaker, settings)
