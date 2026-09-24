"""Async database engine, session factory, and session dependency.

The engine is created lazily from ``settings.database_url`` (an async driver
URL such as ``postgresql+asyncpg://...``) so importing this module never opens a
connection. Use :func:`get_session` as a FastAPI dependency or as an async
context manager for ad-hoc work.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.models import Base

if TYPE_CHECKING:
    from app.config import Settings

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _build_connect_args(settings: Settings) -> dict:
    """Build asyncpg ``connect_args`` for the configured database.

    * ``database_ssl`` -> require TLS (most managed Postgres).
    * ``database_pgbouncer`` -> disable prepared-statement caching, which a
      transaction-mode pooler (typically port 6543) does not support. We also
      give each prepared statement a unique name to avoid clashes across pooled
      backends.
    """
    connect_args: dict = {}
    if getattr(settings, "database_ssl", False):
        import ssl

        ca = getattr(settings, "database_ssl_root_cert", "") or ""
        if ca:
            # Full verification against a pinned CA cert.
            ctx = ssl.create_default_context(cafile=ca)
            # Some managed-Postgres legacy certs omit the RFC-5280 keyUsage
            # extension, which Python 3.13's default VERIFY_X509_STRICT rejects.
            # Relax only that pedantic format check; chain verification and
            # hostname checking stay ON (peer still verified against the CA).
            ctx.verify_flags &= ~ssl.VerifyFlags.VERIFY_X509_STRICT
        else:
            # Encrypt the wire without verifying the chain. Some managed pooler
            # certs chain to a CA absent from the OS trust store, so default
            # verification fails. Guards passive eavesdropping, NOT active MITM.
            # Set DATABASE_SSL_ROOT_CERT to the CA path for full verify.
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ctx
    if getattr(settings, "database_pgbouncer", False):
        import uuid

        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid.uuid4()}__"
    return connect_args


def get_engine(settings: Settings | None = None) -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        if settings is None:
            from app.config import get_settings

            settings = get_settings()
        _engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            future=True,
            connect_args=_build_connect_args(settings),
        )
    return _engine


def get_sessionmaker(settings: Settings | None = None) -> async_sessionmaker[AsyncSession]:
    """Return the process-wide async session factory, creating it on first use."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(settings),
            expire_on_commit=False,
            class_=AsyncSession,
        )
    return _sessionmaker


async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield an :class:`AsyncSession`, committing on success and rolling back on error.

    Usable as a FastAPI dependency (``Depends(get_session)``) or as an
    ``async for`` iterator. The session is always closed.
    """
    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all() -> None:
    """Create all tables from the model metadata (dev/test convenience).

    Production schema management should go through Alembic migrations; this is a
    shortcut for local development and tests.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    """Dispose the engine and reset module state (used in tests / shutdown)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
