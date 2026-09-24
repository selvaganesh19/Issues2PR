"""Worker task: turn a GitHub issue webhook into a tested pull request.

:func:`process_job` is the handler consumed by :func:`app.workers.queue.run_worker`.
It reuses the *same* agent core as the CLI (:func:`app.agent.loop.run_agent`),
so behaviour is identical whether a fix is triggered locally or via a webhook:

1. Mint a GitHub App installation token (``app.github.auth``).
2. Clone the target repo into a throwaway temp workspace using that token.
3. Create an ``agent/*`` branch and run the agent against the issue text.
4. On success: commit, push the branch, open a PR, and comment progress.
5. On failure: comment what was attempted (branch is not pushed).
6. Always clean up the temp workspace.

GitHub network calls go through the documented :class:`app.github.client.GitHubClient`
interface (``create_pull_request`` / ``create_comment`` / ``create_branch``) and
``app.github.auth``. Those modules are imported lazily and defensively so this
module (and its tests) import cleanly even before they exist.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from app.agent.loop import run_agent

_GIT_TIMEOUT_S = 300


def _extract(payload: dict) -> dict:
    """Pull the fields we need out of a GitHub ``issues`` webhook payload.

    Returns a flat dict; missing values become ``None`` / empty so callers can
    validate. Never raises on shape mismatch.
    """
    repo = payload.get("repository", {}) or {}
    issue = payload.get("issue", {}) or {}
    installation = payload.get("installation", {}) or {}
    return {
        "installation_id": installation.get("id"),
        "repo_full_name": repo.get("full_name"),
        "clone_url": repo.get("clone_url"),
        "default_branch": repo.get("default_branch") or "main",
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title") or "",
        "issue_body": issue.get("body") or "",
    }


def _mint_installation_token(settings: Any, installation_id: int) -> str:
    """Mint an installation access token via ``app.github.auth``.

    Tries the documented/likely function names in order so this survives the
    exact naming chosen by the GitHub-auth module. Raises ``RuntimeError`` if
    none is available or all fail.
    """
    from app import github  # noqa: F401  (ensure package import path exists)

    try:
        from app.github import auth  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"app.github.auth is unavailable: {exc}") from exc

    for name in (
        "get_installation_token",
        "installation_token",
        "mint_installation_token",
        "create_installation_token",
    ):
        fn = getattr(auth, name, None)
        if callable(fn):
            try:
                return fn(settings, installation_id)
            except TypeError:
                # Some implementations may take only the installation id.
                return fn(installation_id)
    raise RuntimeError(
        "no installation-token function found in app.github.auth "
        "(expected e.g. get_installation_token(settings, installation_id))"
    )


def _authenticated_clone_url(clone_url: str, token: str) -> str:
    """Embed an installation token into an HTTPS clone URL for push/pull.

    The token is only ever placed in the remote URL passed to ``git`` and is
    never logged. Non-HTTPS URLs are returned unchanged.
    """
    if clone_url.startswith("https://"):
        return "https://x-access-token:" + token + "@" + clone_url[len("https://"):]
    return clone_url


def _git(workspace: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``workspace`` with a timeout, capturing output."""
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,
    )


def _issue_text(title: str, body: str) -> str:
    """Compose the issue text handed to the agent from title + body."""
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if body:
        parts.append(body)
    return "\n\n".join(parts) if parts else "(no issue text provided)"


def _safe_comment(client: Any, repo_full_name: str, issue_number: Any, body: str) -> None:
    """Best-effort issue comment; swallow errors so the worker never crashes."""
    if client is None or issue_number is None:
        return
    try:
        client.create_comment(repo_full_name, issue_number, body)
    except TypeError:
        try:
            client.create_comment(
                repo_full_name=repo_full_name, issue_number=issue_number, body=body
            )
        except Exception:  # noqa: BLE001 - commenting is best-effort
            pass
    except Exception:  # noqa: BLE001 - commenting is best-effort
        pass


def process_job(job: Any, settings: Any = None) -> dict:
    """Process one issue -> PR job. Returns a structured outcome dict.

    Args:
        job: Either a :class:`app.workers.queue.Job` or a raw webhook payload
            dict. The GitHub ``issues`` payload must carry ``installation``,
            ``repository``, and ``issue`` objects.
        settings: Optional :class:`~app.config.Settings`; loaded via
            ``get_settings()`` when omitted.

    Returns:
        A dict with at least ``status`` (one of ``"success"``, ``"no_change"``,
        ``"agent_failed"``, ``"error"``) plus contextual fields (branch, PR
        url, error). Raising is reserved for transient/infra failures so the
        queue can retry; deterministic outcomes are returned, not raised.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    payload = getattr(job, "payload", job)
    info = _extract(payload)

    if not info["repo_full_name"] or info["installation_id"] is None:
        return {
            "status": "error",
            "error": "payload missing repository.full_name or installation.id",
        }

    repo_full_name = info["repo_full_name"]
    issue_number = info["issue_number"]
    branch = f"agent/issue-{issue_number}" if issue_number is not None else "agent/fix"

    # --- authenticate ----------------------------------------------------
    token = _mint_installation_token(settings, info["installation_id"])

    # --- build the GitHub client (documented interface) ------------------
    client: Any = None
    try:
        from app.github.client import GitHubClient  # type: ignore

        client = GitHubClient(token)
    except Exception:  # noqa: BLE001 - degrade to no-PR mode if unavailable
        client = None

    workspace = Path(tempfile.mkdtemp(prefix="issue2pr-"))
    try:
        # --- clone -------------------------------------------------------
        auth_url = _authenticated_clone_url(info["clone_url"] or "", token)
        clone = _git(
            workspace.parent,
            ["clone", "--depth", "1", auth_url, str(workspace)],
        )
        if clone.returncode != 0:
            # Redact any token that may appear in git's error output.
            err = (clone.stderr or clone.stdout).replace(token, "***")
            return {"status": "error", "error": f"git clone failed: {err.strip()}"}

        # --- branch ------------------------------------------------------
        checkout = _git(workspace, ["checkout", "-b", branch])
        if checkout.returncode != 0:
            err = (checkout.stderr or checkout.stdout).replace(token, "***")
            return {"status": "error", "error": f"branch create failed: {err.strip()}"}

        # --- run the shared agent core -----------------------------------
        issue_text = _issue_text(info["issue_title"], info["issue_body"])
        result = run_agent(issue_text, workspace, settings=settings)

        # --- failure path: comment what was tried ------------------------
        if not result.finished:
            _safe_comment(
                client,
                repo_full_name,
                issue_number,
                (
                    "Issue2PR could not complete this issue.\n\n"
                    f"Reason: {result.error or 'unknown'}\n"
                    f"Steps taken: {result.steps}\n\n"
                    f"Summary: {result.summary}"
                ),
            )
            return {
                "status": "agent_failed",
                "branch": branch,
                "steps": result.steps,
                "error": result.error,
                "summary": result.summary,
            }

        # --- commit ------------------------------------------------------
        _git(workspace, ["add", "-A"])
        commit = _git(
            workspace,
            ["commit", "-m", f"fix: {info['issue_title'] or 'address issue'} (#{issue_number})"],
        )
        if commit.returncode != 0 and "nothing to commit" in (
            (commit.stdout or "") + (commit.stderr or "")
        ):
            _safe_comment(
                client,
                repo_full_name,
                issue_number,
                (
                    "Issue2PR ran but produced no code changes.\n\n"
                    f"Summary: {result.summary}"
                ),
            )
            return {
                "status": "no_change",
                "branch": branch,
                "steps": result.steps,
                "summary": result.summary,
            }

        # --- push --------------------------------------------------------
        push = _git(workspace, ["push", "origin", branch])
        if push.returncode != 0:
            err = (push.stderr or push.stdout).replace(token, "***")
            _safe_comment(
                client,
                repo_full_name,
                issue_number,
                f"Issue2PR made changes but failed to push branch `{branch}`.",
            )
            return {"status": "error", "branch": branch, "error": f"git push failed: {err.strip()}"}

        # --- open PR -----------------------------------------------------
        pr_url = None
        if client is not None:
            title = f"Fix #{issue_number}: {info['issue_title']}".strip()
            body = (
                f"Automated fix for #{issue_number} by Issue2PR.\n\n"
                f"{result.summary}\n\n"
                "---\n"
                "Generated from the issue text (treated as untrusted input). "
                "Please review before merging."
            )
            try:
                pr = client.create_pull_request(
                    repo_full_name,
                    head=branch,
                    base=info["default_branch"],
                    title=title,
                    body=body,
                )
                pr_url = pr.get("html_url") if isinstance(pr, dict) else str(pr)
            except Exception as exc:  # noqa: BLE001 - report but keep the branch
                _safe_comment(
                    client,
                    repo_full_name,
                    issue_number,
                    f"Issue2PR pushed `{branch}` but could not open a PR: {exc}",
                )
                return {
                    "status": "error",
                    "branch": branch,
                    "error": f"create_pull_request failed: {exc}",
                }

        # --- progress comment -------------------------------------------
        _safe_comment(
            client,
            repo_full_name,
            issue_number,
            (
                f"Issue2PR opened a pull request from `{branch}`"
                + (f": {pr_url}" if pr_url else ".")
                + f"\n\n{result.summary}"
            ),
        )
        return {
            "status": "success",
            "branch": branch,
            "pr_url": pr_url,
            "steps": result.steps,
            "summary": result.summary,
        }
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
