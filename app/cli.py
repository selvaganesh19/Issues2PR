"""Local command-line trigger for the Issue2PR agent.

This is the offline entry point: it runs the autonomous "issue -> tested
fix" loop against a *local* repository, on the *host* machine, without the
GitHub webhook / PR machinery. It is meant for development, demos, and manual
runs.

Usage::

    python -m app.cli fix ./path/to/repo "The parser crashes on empty input"
    python -m app.cli fix ./path/to/repo "..." --dry-run

SECURITY: with the default ``local`` sandbox backend, tests and linters run
directly on the host via ``LocalRunner`` -- NOT inside an isolated Docker
container. The command prints a prominent warning to that effect.
"""

from __future__ import annotations

import pathlib
import secrets

import typer
from rich.console import Console

from app.agent.loop import run_agent
from app.config import Settings, get_settings
from app.llm.providers import PROVIDERS
from app.tools import git_ops

app = typer.Typer(
    add_completion=False,
    help="Issue2PR local runner: turn a GitHub-style issue into a tested fix.",
)

console = Console()


def _has_any_llm_key(settings: Settings) -> bool:
    """Return True if at least one configured provider has an API key.

    We check every provider's ``api_key_attr`` rather than only the primary,
    because the LLM client falls back across providers that have keys.
    """
    for spec in PROVIDERS.values():
        if getattr(settings, spec["api_key_attr"], None):
            return True
    return False


@app.command()
def fix(
    repo: str = typer.Argument(..., help="Path to the local repository to fix."),
    issue: str = typer.Argument(..., help="Issue text describing the problem to solve."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Do everything except require an LLM key / mutate git history.",
    ),
) -> None:
    """Run the agent against a local repo and commit the fix on an agent branch.

    Steps:
      1. Load settings and validate that an LLM key is present (unless dry-run).
      2. Ensure ``repo`` is a git repository (``git init`` if needed).
      3. Create an ``agent/fix-<shortid>`` branch.
      4. Invoke :func:`app.agent.loop.run_agent` with the issue text.
      5. Print the run summary and the resulting ``git diff``.
      6. Commit the changes on the agent branch (skipped on ``--dry-run``).
    """
    settings = get_settings()

    workspace = pathlib.Path(repo).expanduser().resolve()
    if not workspace.exists() or not workspace.is_dir():
        console.print(
            f"[red]error:[/red] repo path does not exist or is not a directory: {workspace}"
        )
        raise typer.Exit(code=1)

    # 1. LLM key gate. Without a key the real agent loop cannot call the model,
    #    so fail fast with a clear message -- unless the user only wants a dry run.
    if not _has_any_llm_key(settings) and not dry_run:
        console.print(
            "[red]error:[/red] no LLM API key configured. Set one of "
            "GROQ_API_KEY, OPENROUTER_API_KEY, or AZURE_OPENAI_API_KEY "
            "(e.g. in a .env file), or re-run with --dry-run."
        )
        raise typer.Exit(code=1)

    # 2. Ensure a git repo exists so we can branch, diff, and commit.
    if not git_ops.ensure_git_repo(workspace):
        console.print(f"[red]error:[/red] failed to initialise a git repository at {workspace}")
        raise typer.Exit(code=1)

    # 3. Create a dedicated agent branch (agent/* namespace only).
    branch = f"agent/fix-{secrets.token_hex(4)}"
    branch_msg = git_ops.create_branch(workspace, branch)
    console.print(f"[cyan]{branch_msg}[/cyan]")
    if "FAILED" in branch_msg:
        console.print("[yellow]warning:[/yellow] continuing on the current branch.")

    # Prominent host-execution warning: the default backend runs code on the host.
    if settings.sandbox_backend != "docker":
        console.print(
            "[bold yellow]WARNING:[/bold yellow] tests and linters will run "
            "ON THE HOST via LocalRunner -- NOT inside an isolated Docker "
            "sandbox. Only run this against issues/repos you trust."
        )

    if dry_run and not _has_any_llm_key(settings):
        console.print(
            "[yellow]--dry-run:[/yellow] no LLM key configured; the agent loop "
            "will not be able to contact a model. Showing setup only."
        )

    # 4. Run the autonomous agent.
    console.print(f"[green]running agent[/green] on {workspace} (branch {branch})...")
    result = run_agent(issue, workspace, settings=settings)

    # 5. Report the outcome.
    status = "FINISHED" if result.finished else "STOPPED"
    color = "green" if result.finished else "red"
    console.print(f"\n[bold {color}]{status}[/bold {color}] after {result.steps} step(s).")
    console.print("[bold]summary:[/bold]")
    console.print(result.summary or "(no summary)")
    if result.error:
        console.print(f"[red]error:[/red] {result.error}")

    console.print("\n[bold]git diff:[/bold]")
    diff = git_ops.git_diff({}, _make_ctx(workspace, settings))
    console.print(diff)

    # 6. Commit the work on the agent branch (never on dry-run).
    if dry_run:
        console.print("\n[yellow]--dry-run:[/yellow] skipping commit.")
        raise typer.Exit(code=0 if result.finished else 1)

    commit_msg = git_ops.commit_all(workspace, f"agent: fix for issue\n\n{issue[:200]}")
    console.print(f"\n[cyan]{commit_msg}[/cyan]")

    raise typer.Exit(code=0 if result.finished else 1)


@app.command()
def initdb() -> None:
    """Create all database tables (Postgres) from the ORM models.

    Convenience for local/dev setup and first-time DB provisioning. Reads
    ``DATABASE_URL`` (and ``DATABASE_SSL`` / ``DATABASE_PGBOUNCER``) from the
    environment / ``.env``. For production, prefer Alembic migrations.
    """
    import asyncio

    settings = get_settings()
    # Redact credentials when echoing the target.
    safe = settings.database_url
    if "@" in safe:
        safe = safe.split("@", 1)[0].rsplit(":", 1)[0] + ":***@" + safe.split("@", 1)[1]
    console.print(f"[cyan]creating tables on[/cyan] {safe}")

    async def _run() -> None:
        from app.db.session import create_all, dispose_engine

        try:
            await create_all()
        finally:
            await dispose_engine()

    try:
        asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]initdb failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]done[/green] — tables created (runs, steps, tool_calls, deliveries).")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host for the API server."),
    port: int = typer.Option(8000, help="Bind port for the API server."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev)."),
) -> None:
    """Run the FastAPI webhook API with uvicorn (no Docker required).

    Requires the ``server`` extra: ``pip install -e '.[server]'``.
    """
    try:
        import uvicorn
    except ModuleNotFoundError as exc:
        console.print(
            "[red]error:[/red] the API server needs the 'server' extra. "
            "Run: pip install -e \".[server]\""
        )
        raise typer.Exit(code=1) from exc
    console.print(f"[green]starting API[/green] on http://{host}:{port}  (docs at /docs)")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


@app.command()
def worker() -> None:
    """Run the background job worker (no Docker required).

    Consumes the queue selected by ``REDIS_URL`` (use ``fake://local`` for an
    in-process queue) and processes issue->PR jobs. Requires the ``server``
    extra for real Redis / GitHub calls.
    """
    from app.workers.__main__ import main as worker_main

    worker_main()


def _make_ctx(workspace: pathlib.Path, settings: Settings):
    """Build a minimal ToolContext for the top-level ``git_diff`` call.

    Imported lazily / locally to avoid a hard import at module load for the
    common ``--help`` path, and to keep the CLI's surface small.
    """
    from app.sandbox.runner import get_runner
    from app.tools.registry import ToolContext

    return ToolContext(
        workspace=workspace,
        runner=get_runner(settings, workspace),
        settings=settings,
    )


@app.callback()
def _main() -> None:
    """Issue2PR local runner (see ``fix --help``)."""


if __name__ == "__main__":
    app()
