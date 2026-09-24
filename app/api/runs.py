"""Read-only endpoints exposing agent run history.

``GET /runs`` and ``GET /runs/{run_id}`` surface records persisted by the
:mod:`app.db` layer. The database layer may be provided by another component;
these handlers look up its accessors at call time and degrade gracefully (empty
list / 404 / 503) when persistence is not configured, so the API remains
importable in the local/dev path without Postgres.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status

logger = logging.getLogger("issue2pr.runs")

router = APIRouter(prefix="/runs", tags=["runs"])


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise return it unchanged."""
    if inspect.isawaitable(value):
        return await value
    return value


def _load_db() -> Any | None:
    """Import and return the :mod:`app.db` module, or ``None`` if unavailable."""
    try:
        from app import db as db_mod  # type: ignore

        return db_mod
    except Exception:  # pragma: no cover - db module optional in dev
        return None


def _resolve(db_mod: Any, names: tuple[str, ...]) -> Any | None:
    """Return the first callable attribute of ``db_mod`` named in ``names``."""
    for name in names:
        fn = getattr(db_mod, name, None)
        if callable(fn):
            return fn
    return None


@router.get("")
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """List recent agent runs (most recent first)."""
    db_mod = _load_db()
    if db_mod is None:
        return {"runs": [], "limit": limit, "offset": offset}

    fn = _resolve(db_mod, ("list_runs", "get_runs", "fetch_runs"))
    if fn is None:
        return {"runs": [], "limit": limit, "offset": offset}

    try:
        runs = await _maybe_await(fn(limit=limit, offset=offset))
    except TypeError:
        # Accessor may not accept pagination kwargs.
        runs = await _maybe_await(fn())
    except Exception as exc:  # pragma: no cover - depends on db backend
        logger.warning("list_runs failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="run store unavailable",
        ) from exc

    return {"runs": runs or [], "limit": limit, "offset": offset}


@router.get("/{run_id}")
async def get_run(run_id: str) -> Any:
    """Return a single run by id, or 404 if it does not exist."""
    db_mod = _load_db()
    if db_mod is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="run store unavailable",
        )

    fn = _resolve(db_mod, ("get_run", "fetch_run", "get_run_by_id"))
    if fn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="run store unavailable",
        )

    try:
        run = await _maybe_await(fn(run_id))
    except Exception as exc:  # pragma: no cover - depends on db backend
        logger.warning("get_run failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="run store unavailable",
        ) from exc

    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return run
