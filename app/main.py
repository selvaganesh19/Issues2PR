"""FastAPI application factory for Issue2PR.

Exposes ``create_app()`` which wires the health, webhook, and runs routers,
and a module-level ``app`` for ``uvicorn app.main:app``.

Server dependencies (FastAPI, uvicorn, ...) live in the optional ``server``
extra; install with ``pip install -e '.[server]'``.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api import health, runs, webhooks
from app.config import get_settings

logger = logging.getLogger("issue2pr")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown hooks.

    Startup validates configuration and logs the effective (non-secret) mode.
    Optional resources (db/queue) are initialised lazily by their own modules,
    so the app starts cleanly even without Postgres/Redis in the environment.
    """
    settings = get_settings()
    logger.info(
        "Issue2PR %s starting (provider=%s, model=%s, sandbox=%s)",
        __version__,
        settings.llm_provider_primary,
        settings.llm_model,
        settings.sandbox_backend,
    )
    yield
    logger.info("Issue2PR shutting down")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""
    app = FastAPI(
        title="Issue2PR",
        version=__version__,
        description="Autonomous GitHub issue -> tested pull request agent.",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(webhooks.router)
    app.include_router(runs.router)

    return app


app = create_app()
