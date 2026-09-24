"""Structured logging and DB-backed tracing for agent runs.

This module provides two things:

* :func:`configure_logging` / :func:`get_logger` — a thin structured-logging
  setup over the stdlib :mod:`logging` module.
* :class:`Tracer` — records an agent run's lifecycle (run start, per-step tool
  calls, prompts, results) both to the structured log and, when a database is
  configured and reachable, to the ``runs`` / ``steps`` / ``tool_calls`` tables
  defined in :mod:`app.db.models`.

The Tracer degrades gracefully to log-only mode: if the ``server`` extras
(SQLAlchemy/asyncpg) are not installed, or the database cannot be reached, it
keeps emitting structured logs and never raises into the caller. Secrets are
never logged — only prompt/result text and tool metadata are persisted.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

_LOG_CONFIGURED = False

# Cap persisted/loggable text so a single huge tool result cannot blow up the
# database row or the log stream.
_EXCERPT_LIMIT = 4000


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a concise structured-ish format."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    # Avoid duplicate handlers if some other component already added one.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    _LOG_CONFIGURED = True


def get_logger(name: str = "issue2pr") -> logging.Logger:
    """Return a configured logger under the given ``name``."""
    configure_logging()
    return logging.getLogger(name)


def _excerpt(text: Any) -> str:
    """Coerce ``text`` to a string and clamp it to the excerpt limit."""
    s = text if isinstance(text, str) else str(text)
    if len(s) <= _EXCERPT_LIMIT:
        return s
    return s[:_EXCERPT_LIMIT] + f"\n[... truncated {len(s) - _EXCERPT_LIMIT} chars]"


def _kv(fields: dict[str, Any]) -> str:
    """Render a dict as compact ``key=value`` pairs for structured log lines."""
    parts = []
    for key, value in fields.items():
        rendered = value if isinstance(value, (int, float)) else json.dumps(str(value))
        parts.append(f"{key}={rendered}")
    return " ".join(parts)


class Tracer:
    """Persist an agent run's trace to the DB, degrading to log-only on failure.

    Typical usage from the agent loop::

        tracer = Tracer(repo="owner/name", issue_number=42)
        await tracer.start()                      # inserts a Run row
        await tracer.step("llm", prompt=messages) # inserts a Step row
        await tracer.tool_call("read_file", args, result)
        await tracer.finish(status="completed", cost_usd=0.12)

    A synchronous facade (:meth:`log_event`) is also provided for callers that
    only want structured logs without touching the event loop.

    Attributes:
        repo: Repository slug (``owner/name``) for the run.
        issue_number: GitHub issue number driving the run.
        run_id: Primary key of the persisted Run row, or None in log-only mode.
        db_enabled: Whether DB persistence is active for this tracer.
    """

    def __init__(
        self,
        repo: str,
        issue_number: int,
        logger: logging.Logger | None = None,
        use_db: bool = True,
    ) -> None:
        self.repo = repo
        self.issue_number = issue_number
        self.log = logger or get_logger("issue2pr.trace")
        self.run_id: int | None = None
        self._step_index = 0
        self._current_step_id: int | None = None
        self.db_enabled = use_db and self._db_available()

    @staticmethod
    def _db_available() -> bool:
        """Return True if the SQLAlchemy async stack can be imported."""
        try:
            import sqlalchemy.ext.asyncio  # noqa: F401
        except Exception:
            return False
        return True

    async def _session(self):
        """Open a new async session, or None if the DB layer is unavailable."""
        try:
            from app.db.session import get_sessionmaker

            return get_sessionmaker()()
        except Exception as exc:  # noqa: BLE001 - degrade to log-only
            self.log.warning("trace_db_unavailable %s", _kv({"error": exc}))
            self.db_enabled = False
            return None

    def log_event(self, event: str, **fields: Any) -> None:
        """Emit a structured log line (never touches the database)."""
        base = {"event": event, "repo": self.repo, "issue": self.issue_number}
        if self.run_id is not None:
            base["run_id"] = self.run_id
        base.update(fields)
        self.log.info(_kv(base))

    async def start(self, status: str = "running") -> int | None:
        """Record the start of a run; insert a Run row when the DB is enabled.

        Returns the run id (or None in log-only mode).
        """
        self.log_event("run_start", status=status)
        if not self.db_enabled:
            return None
        session = await self._session()
        if session is None:
            return None
        try:
            from app.db.models import Run

            async with session:
                run = Run(
                    repo=self.repo,
                    issue_number=self.issue_number,
                    status=status,
                    cost_usd=0.0,
                )
                session.add(run)
                await session.commit()
                await session.refresh(run)
                self.run_id = run.id
        except Exception as exc:  # noqa: BLE001 - degrade to log-only
            self.log.warning("trace_run_start_failed %s", _kv({"error": exc}))
            self.db_enabled = False
        return self.run_id

    async def step(self, kind: str, prompt: Any = None) -> int | None:
        """Record one agent step (e.g. ``"llm"``, ``"tool"``); insert a Step row.

        ``prompt`` is logged as an excerpt for observability. Returns the step
        id (or None in log-only mode). Subsequent :meth:`tool_call` invocations
        attach to the most recent step.
        """
        self._step_index += 1
        index = self._step_index
        fields: dict[str, Any] = {"kind": kind, "index": index}
        if prompt is not None:
            fields["prompt"] = _excerpt(prompt)
        self.log_event("step", **fields)

        if not self.db_enabled or self.run_id is None:
            return None
        session = await self._session()
        if session is None:
            return None
        try:
            from app.db.models import Step

            async with session:
                step = Step(run_id=self.run_id, index=index, kind=kind)
                session.add(step)
                await session.commit()
                await session.refresh(step)
                self._current_step_id = step.id
        except Exception as exc:  # noqa: BLE001 - degrade to log-only
            self.log.warning("trace_step_failed %s", _kv({"error": exc}))
            self.db_enabled = False
        return self._current_step_id

    async def tool_call(self, name: str, args: Any, result: Any) -> int | None:
        """Record a single tool invocation under the current step.

        ``args`` is serialised to JSON (best-effort) and ``result`` is stored as
        a bounded excerpt. Returns the tool-call id (or None in log-only mode).
        """
        try:
            args_json = json.dumps(args, default=str)
        except (TypeError, ValueError):
            args_json = json.dumps(str(args))
        result_excerpt = _excerpt(result)
        self.log_event("tool_call", name=name, result=result_excerpt)

        if not self.db_enabled or self._current_step_id is None:
            return None
        session = await self._session()
        if session is None:
            return None
        try:
            from app.db.models import ToolCall

            async with session:
                call = ToolCall(
                    step_id=self._current_step_id,
                    name=name,
                    args_json=args_json,
                    result_excerpt=result_excerpt,
                )
                session.add(call)
                await session.commit()
                await session.refresh(call)
                return call.id
        except Exception as exc:  # noqa: BLE001 - degrade to log-only
            self.log.warning("trace_tool_call_failed %s", _kv({"error": exc}))
            self.db_enabled = False
        return None

    async def finish(self, status: str, cost_usd: float = 0.0) -> None:
        """Record run completion; update the Run row's status and cost."""
        self.log_event("run_finish", status=status, cost_usd=cost_usd)
        if not self.db_enabled or self.run_id is None:
            return
        session = await self._session()
        if session is None:
            return
        try:
            from app.db.models import Run

            async with session:
                run = await session.get(Run, self.run_id)
                if run is not None:
                    run.status = status
                    run.cost_usd = cost_usd
                    await session.commit()
        except Exception as exc:  # noqa: BLE001 - degrade to log-only
            self.log.warning("trace_finish_failed %s", _kv({"error": exc}))
            self.db_enabled = False

    # -- Synchronous convenience wrappers -------------------------------------

    def start_sync(self, status: str = "running") -> int | None:
        """Blocking wrapper around :meth:`start` for non-async callers."""
        return self._run_sync(self.start(status))

    def step_sync(self, kind: str, prompt: Any = None) -> int | None:
        """Blocking wrapper around :meth:`step`."""
        return self._run_sync(self.step(kind, prompt))

    def tool_call_sync(self, name: str, args: Any, result: Any) -> int | None:
        """Blocking wrapper around :meth:`tool_call`."""
        return self._run_sync(self.tool_call(name, args, result))

    def finish_sync(self, status: str, cost_usd: float = 0.0) -> None:
        """Blocking wrapper around :meth:`finish`."""
        self._run_sync(self.finish(status, cost_usd))

    def _run_sync(self, coro):
        """Run an async coroutine from sync code, degrading if a loop is active.

        If called while an event loop is already running (e.g. inside async
        code) we cannot block on it, so we fall back to log-only and cancel the
        coroutine to avoid a 'never awaited' warning.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # An event loop is already running in this thread; caller should await
        # the async methods directly. Close the coroutine and degrade.
        coro.close()
        self.log.debug("tracer sync wrapper called within running loop; use async API")
        return None
