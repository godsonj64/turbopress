"""Test fixtures: configure a fresh in-process app (inline runner, temp SQLite)."""

import os
import tempfile
import uuid

import pytest


@pytest.fixture()
def client():
    # Configure via env BEFORE importing the app / settings.
    db_path = os.path.join(tempfile.gettempdir(), f"tp_test_{uuid.uuid4().hex}.db")
    os.environ.update(
        {
            "DEBUG": "true",
            "RUNNER": "inline",
            "DATABASE_URL": f"sqlite+aiosqlite:///{db_path}",
            "SIGNING_PRIVATE_KEY_B64": "",
            "STRIPE_SECRET_KEY": "",
            "STRIPE_PRICE_ID": "",
        }
    )
    # Reset cached settings + engine so each test gets an isolated database.
    import service.config as config
    import service.db as db

    config.get_settings.cache_clear()
    db._engine = None
    db._sessionmaker = None

    from fastapi.testclient import TestClient

    from service.main import app

    with TestClient(app) as c:
        yield c

    try:
        os.remove(db_path)
    except OSError:
        pass
