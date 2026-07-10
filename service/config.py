"""Runtime configuration, read from the environment (12-factor)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Core
    debug: bool = False
    database_url: str = "sqlite+aiosqlite:///./turbopress.db"
    # "inline" runs compression in-process (dev/CI, no GPU); "modal" spawns the
    # serverless GPU worker.
    runner: str = "inline"
    pipeline_version: str = "0.3.0"

    # Auth / internal
    # Shared secret the GPU worker presents when calling back with results.
    worker_callback_secret: str = "dev-callback-secret"
    # Ed25519 private seed (base64) used to sign certificates. Empty => generated
    # ephemerally at startup (fine for dev; set a stable key in production).
    signing_private_key_b64: str = ""

    # Public base URL of this control plane (used to tell the worker where to
    # call back). Defaults to localhost for dev.
    public_base_url: str = "http://localhost:8000"

    # Stripe (metered billing). Empty => billing disabled (dev/free).
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""
    stripe_price_id: str = ""  # a metered ("usage") recurring price
    # Free tier: public HF models never require an active subscription.
    require_billing_for_public_models: bool = False

    # Cloudflare R2 (S3-compatible). Empty => artifact upload/presign disabled.
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = "turbopress-artifacts"
    r2_presign_ttl_seconds: int = 3600

    # Modal
    modal_app_name: str = "turbopress-worker"
    modal_function_name: str = "compress_job"

    @property
    def stripe_enabled(self) -> bool:
        return bool(self.stripe_secret_key and self.stripe_price_id)

    @property
    def r2_enabled(self) -> bool:
        return bool(
            self.r2_account_id and self.r2_access_key_id and self.r2_secret_access_key
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
