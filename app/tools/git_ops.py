"""Git tools: read the working diff plus branch/commit helpers.

Everything is driven through ``subprocess`` (not GitPython) so the only
dependency is a ``git`` binary on PATH. Branch creation is restricted to the
``agent/*`` namespace so the agent can never touch protected branches directly.
"""

from __future__ import annotations

import pathlib
import subprocess

from app.tools.registry import ToolContext

_MAX_DIFF_CHARS = 12_000
_GIT_TIMEOUT_S = 60


def _run_git(workspace: pathlib.Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in ``workspace`` and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_S,
        check=False,
    )


def ensure_git_repo(workspace: pathlib.Path) -> bool:
    """Ensure ``workspace`` is a git repository, initialising one if needed.

    Returns:
        True if the workspace is (now) a git repo, False if init failed.
    """
    workspace = pathlib.Path(workspace)
    inside = _run_git(workspace, ["rev-parse", "--is-inside-work-tree"])
    if inside.returncode == 0 and inside.stdout.strip() == "true":
        return True
    init = _run_git(workspace, ["init"])
    return init.returncode == 0


def create_branch(workspace: pathlib.Path, name: str) -> str:
    """Create and switch to a branch. Only ``agent/*`` names are permitted.

    Raises:
        ValueError: If ``name`` is not within the ``agent/`` namespace.
    """
    if not name or not name.startswith("agent/") or name == "agent/":
        raise ValueError(
            f"branch must be within the 'agent/' namespace, got {name!r}"
        )
    workspace = pathlib.Path(workspace)
    result = _run_git(workspace, ["checkout", "-b", name])
    if result.returncode != 0:
        return (
            f"create_branch: FAILED to create {name}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return f"create_branch: switched to new branch {name}."


def commit_all(workspace: pathlib.Path, message: str) -> str:
    """Stage every change and create a commit with ``message``."""
    workspace = pathlib.Path(workspace)
    add = _run_git(workspace, ["add", "-A"])
    if add.returncode != 0:
        return f"commit_all: FAILED to stage: {(add.stderr or add.stdout).strip()}"
    commit = _run_git(workspace, ["commit", "-m", message])
    if commit.returncode != 0:
        out = (commit.stdout or "") + (commit.stderr or "")
        if "nothing to commit" in out:
            return "commit_all: nothing to commit (working tree clean)."
        return f"commit_all: FAILED: {out.strip()}"
    return f"commit_all: committed with message {message!r}."


def git_diff(args: dict, ctx: ToolContext) -> str:
    """Show the current uncommitted git diff for the workspace.

    Includes staged changes (``--staged`` union) via ``git diff HEAD`` when a
    HEAD exists, falling back to a plain ``git diff`` otherwise. Truncated.
    """
    workspace = pathlib.Path(ctx.workspace)
    if not ensure_git_repo(workspace):
        return "git_diff error: workspace is not a git repository."

    has_head = _run_git(workspace, ["rev-parse", "--verify", "HEAD"]).returncode == 0
    diff_args = ["diff", "HEAD"] if has_head else ["diff"]
    try:
        result = _run_git(workspace, diff_args)
    except subprocess.TimeoutExpired:
        return "git_diff error: git diff timed out."
    except FileNotFoundError:
        return "git_diff error: git is not installed or not on PATH."

    if result.returncode != 0:
        return f"git_diff error: {(result.stderr or result.stdout).strip()}"

    diff = result.stdout
    # Include untracked files so newly created files are visible in the diff.
    untracked = _run_git(
        workspace, ["ls-files", "--others", "--exclude-standard"]
    ).stdout.strip()
    if untracked:
        diff += "\n[untracked files]\n" + untracked

    diff = diff.strip()
    if not diff:
        return "git_diff: no changes."
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n[... diff truncated ...]"
    return diff
