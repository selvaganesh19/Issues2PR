"""Gradio dashboard for browsing Issue2PR run history.

This is a lightweight, read-only viewer over the run/step history persisted by
the agent. In this environment (no Postgres) history is read from a local
SQLite database file (default ``app.db`` in the current directory, overridable
via the ``ISSUE2PR_DB_PATH`` environment variable).

The viewer is intentionally defensive: it introspects the database, shows the
``runs`` and ``steps`` tables when they exist, and degrades gracefully to a
helpful message when the database or tables are missing. This keeps the
dashboard decoupled from the exact schema chosen by the persistence layer.

Launch::

    python -m dashboard.app          # serves on http://127.0.0.1:7860

The ``gradio`` import is guarded so importing this module (e.g. for tests or
tooling) does not hard-fail when the optional ``dashboard`` extra is not
installed.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# Port the dashboard listens on (matches docker-compose and the README).
DASHBOARD_PORT = 7860

# Tables we know how to render, most-useful first.
_KNOWN_TABLES = ("runs", "steps")


def _db_path() -> Path:
    """Return the SQLite database path (``ISSUE2PR_DB_PATH`` or ``app.db``)."""
    return Path(os.environ.get("ISSUE2PR_DB_PATH", "app.db"))


def _pg_url() -> str:
    """Return the configured async Postgres URL, or '' if not Postgres.

    When ``DATABASE_URL`` points at Postgres, the dashboard
    reads run history straight from that database instead of the local SQLite
    file. Any import/config failure degrades to SQLite silently.
    """
    try:
        from app.config import get_settings

        url = get_settings().database_url or ""
    except Exception:
        return ""
    return url if url.startswith("postgresql") else ""


def _load_pg(sql: str, params: dict | None = None) -> tuple[list[str], list[list]]:
    """Run a read query against the async Postgres engine and return (cols, rows)."""
    import asyncio

    from sqlalchemy import text

    from app.db.session import get_sessionmaker

    async def _run() -> tuple[list[str], list[list]]:
        sm = get_sessionmaker()
        async with sm() as session:
            res = await session.execute(text(sql), params or {})
            cols = list(res.keys())
            rows = [list(r) for r in res.fetchall()]
            return cols, rows

    return asyncio.run(_run())



def _connect(path: Path) -> sqlite3.Connection | None:
    """Open a read-only-ish connection to ``path`` or return None if absent."""
    if not path.exists():
        return None
    try:
        conn = sqlite3.connect(str(path))
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _existing_tables(conn: sqlite3.Connection) -> list[str]:
    """Return the names of tables present in the connected database."""
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    except sqlite3.Error:
        return []
    return [r[0] for r in rows]


def _fetch_table(
    conn: sqlite3.Connection, table: str, limit: int = 500
) -> tuple[list[str], list[list]]:
    """Fetch up to ``limit`` rows of ``table`` as (columns, rows).

    Returns empty structures on any error so the UI never crashes on a schema
    it does not recognise.
    """
    try:
        cur = conn.execute(f"SELECT * FROM {table} LIMIT ?", (limit,))
    except sqlite3.Error:
        return [], []
    columns = [d[0] for d in cur.description] if cur.description else []
    rows = [list(r) for r in cur.fetchall()]
    return columns, rows


def load_runs() -> tuple[list[str], list[list], str]:
    """Load the ``runs`` table (columns, rows, status message)."""
    pg = _pg_url()
    if pg:
        try:
            cols, rows = _load_pg(
                "SELECT id, repo, issue_number, status, cost_usd, created_at "
                "FROM runs ORDER BY id DESC LIMIT 500"
            )
            return cols, rows, f"Loaded {len(rows)} run(s) from Postgres."
        except Exception as e:  # noqa: BLE001 - degrade to a message, never crash the UI
            return [], [], f"Postgres read failed: {type(e).__name__}: {e}"
    path = _db_path()
    conn = _connect(path)
    if conn is None:
        return [], [], f"No database found at '{path}'. Run the agent first to create history."
    try:
        tables = _existing_tables(conn)
        if "runs" not in tables:
            found = ", ".join(tables) or "(none)"
            return [], [], f"No 'runs' table in '{path}'. Tables present: {found}."
        columns, rows = _fetch_table(conn, "runs")
        return columns, rows, f"Loaded {len(rows)} run(s) from '{path}'."
    finally:
        conn.close()


def load_steps(run_id: str = "") -> tuple[list[str], list[list], str]:
    """Load the ``steps`` table, optionally filtered by ``run_id``.

    The filter is applied only when a ``run_id`` column exists; otherwise all
    steps are returned. ``run_id`` is passed as a bound parameter (no string
    interpolation) to avoid injection.
    """
    pg = _pg_url()
    if pg:
        try:
            run_id = (run_id or "").strip()
            if run_id:
                cols, rows = _load_pg(
                    'SELECT id, run_id, "index", kind FROM steps '
                    'WHERE run_id = :rid ORDER BY "index" LIMIT 2000',
                    {"rid": int(run_id)} if run_id.isdigit() else {"rid": -1},
                )
                return cols, rows, f"Loaded {len(rows)} step(s) for run '{run_id}' (Postgres)."
            cols, rows = _load_pg(
                'SELECT id, run_id, "index", kind FROM steps ORDER BY id DESC LIMIT 2000'
            )
            return cols, rows, f"Loaded {len(rows)} step(s) from Postgres."
        except Exception as e:  # noqa: BLE001
            return [], [], f"Postgres read failed: {type(e).__name__}: {e}"
    path = _db_path()
    conn = _connect(path)
    if conn is None:
        return [], [], f"No database found at '{path}'."
    try:
        tables = _existing_tables(conn)
        if "steps" not in tables:
            found = ", ".join(tables) or "(none)"
            return [], [], f"No 'steps' table in '{path}'. Tables present: {found}."

        columns, _ = _fetch_table(conn, "steps", limit=1)
        run_id = (run_id or "").strip()
        if run_id and "run_id" in columns:
            try:
                cur = conn.execute(
                    "SELECT * FROM steps WHERE run_id = ? LIMIT 2000", (run_id,)
                )
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = [list(r) for r in cur.fetchall()]
                return cols, rows, f"Loaded {len(rows)} step(s) for run '{run_id}'."
            except sqlite3.Error:
                pass  # fall through to unfiltered load

        cols, rows = _fetch_table(conn, "steps", limit=2000)
        note = f"Loaded {len(rows)} step(s)."
        if run_id and "run_id" not in columns:
            note += " (No 'run_id' column to filter on; showing all.)"
        return cols, rows, note
    finally:
        conn.close()


def build_ui():  # type: ignore[no-untyped-def]
    """Construct and return the Gradio Blocks app.

    Imported lazily inside so the module remains importable without gradio.
    """
    import gradio as gr

    def _refresh_runs():
        columns, rows, status = load_runs()
        headers = columns or ["(no data)"]
        return gr.update(value=rows, headers=headers), status

    def _refresh_steps(run_id: str):
        columns, rows, status = load_steps(run_id)
        headers = columns or ["(no data)"]
        return gr.update(value=rows, headers=headers), status

    with gr.Blocks(title="Issue2PR — Run History") as demo:
        _src = "Postgres" if _pg_url() else f"`{_db_path()}`"
        gr.Markdown(
            "# Issue2PR — Run History\n"
            "Read-only viewer over agent run/step history "
            f"(source: {_src})."
        )

        with gr.Tab("Runs"):
            runs_status = gr.Markdown()
            runs_table = gr.Dataframe(interactive=False, wrap=True)
            runs_refresh = gr.Button("Refresh runs", variant="primary")
            runs_refresh.click(_refresh_runs, outputs=[runs_table, runs_status])

        with gr.Tab("Steps"):
            with gr.Row():
                run_id_box = gr.Textbox(
                    label="Filter by run_id (optional)", placeholder="e.g. 42"
                )
                steps_refresh = gr.Button("Load steps", variant="primary")
            steps_status = gr.Markdown()
            steps_table = gr.Dataframe(interactive=False, wrap=True)
            steps_refresh.click(
                _refresh_steps, inputs=[run_id_box], outputs=[steps_table, steps_status]
            )

        # Populate the runs tab on initial load.
        demo.load(_refresh_runs, outputs=[runs_table, runs_status])

    return demo


def main() -> None:
    """Launch the dashboard on ``DASHBOARD_PORT`` (guards missing gradio)."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        raise SystemExit(
            "gradio is not installed. Install the dashboard extra:\n"
            '    pip install -e ".[dashboard]"'
        ) from None

    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=DASHBOARD_PORT)


if __name__ == "__main__":
    main()
