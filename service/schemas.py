"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SignupRequest(BaseModel):
    email: str


class SignupResponse(BaseModel):
    user_id: str
    email: str
    api_key: str = Field(description="Shown once; store it securely.")


class CheckoutResponse(BaseModel):
    checkout_url: str


class JobCreateRequest(BaseModel):
    model_ref: str = Field(description="Hugging Face model id or local path")
    targets: list[str] = Field(default_factory=lambda: ["4bit"])
    fidelity_gates: dict[str, float] = Field(default_factory=dict)
    private: bool = True


class CertificateOut(BaseModel):
    id: str
    target: str
    bits_per_weight: float
    param_count: int
    metrics: dict[str, Any]
    gates_passed: bool
    gate_failures: list[str]
    artifact_url: str | None = None
    certificate: dict[str, Any] | None = None


class JobOut(BaseModel):
    id: str
    model_ref: str
    targets: list[str]
    fidelity_gates: dict[str, float]
    status: str
    error: str | None = None
    gates_passed: bool | None = None
    certificates: list[CertificateOut] = Field(default_factory=list)


class TargetResult(BaseModel):
    target: str
    bits_per_weight: float
    param_count: int
    metrics: dict[str, Any]
    artifact_key: str | None = None
    artifact_bytes: int = 0
    artifact_sha256: str | None = None
    certificate: dict[str, Any]


class JobCompletion(BaseModel):
    results: list[TargetResult] = Field(default_factory=list)
    error: str | None = None
