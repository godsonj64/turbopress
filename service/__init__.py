"""TurboPress hosted control plane (FastAPI).

A small, boring-on-purpose control plane: it authenticates users by API key,
accepts compression jobs, enqueues them to GPU workers (Modal in production, an
in-process runner for local/dev), persists signed certificates, meters usage to
Stripe, and hands back R2-presigned artifact URLs. See ``service/README.md``.
"""

__version__ = "0.1.0"
