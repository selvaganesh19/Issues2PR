"""Search tools: text search (ripgrep or pure-Python) and semantic search.

:func:`search_code` prefers ``rg`` (ripgrep) when it is on PATH for speed, and
otherwise falls back to a pure-Python walk over text files rooted at the
workspace. :func:`semantic_search` uses ``ctx.index`` when a semantic index is
enabled, and otherwise returns a graceful notice pointing at ``search_code``.
"""

from __future__ import annotations

import fnmatch
import pathlib
import shutil

from app.tools.registry import ToolContext

# Caps to keep results and scan cost bounded.
_MAX_MATCHES = 100
_MAX_LINE_LEN = 300
_MAX_FILE_BYTES = 2_000_000

# Directories that are never worth searching.
_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
}


def _looks_binary(chunk: bytes) -> bool:
    """Heuristic: treat data containing a NUL byte as binary."""
    return b"\x00" in chunk


def _rg_search(query: str, glob: str | None, ctx: ToolContext) -> str:
    """Run ripgrep and format its output, returning None-like on failure."""
    cmd = ["rg", "--no-heading", "--line-number", "--color", "never", "--max-count", "50"]
    if glob:
        cmd += ["--glob", glob]
    cmd += ["--", query, "."]
    result = ctx.runner.run(cmd)
    # ripgrep exits 1 when there are no matches; that is not an error.
    if result.returncode not in (0, 1):
        return f"search_code error (rg): {result.stderr.strip() or result.returncode}"
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return f"search_code: no matches for {query!r}."
    capped = lines[:_MAX_MATCHES]
    out = "\n".join(ln[:_MAX_LINE_LEN] for ln in capped)
    if len(lines) > _MAX_MATCHES:
        out += f"\n... ({len(lines) - _MAX_MATCHES} more matches truncated)"
    return out


def _python_search(query: str, glob: str | None, ctx: ToolContext) -> str:
    """Pure-Python fallback: walk text files under the workspace root."""
    root = pathlib.Path(ctx.workspace).resolve()
    matches: list[str] = []
    total = 0
    for path in root.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        if glob and not fnmatch.fnmatch(path.name, glob):
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            with path.open("rb") as fh:
                head = fh.read(1024)
            if _looks_binary(head):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if query in line:
                total += 1
                if len(matches) < _MAX_MATCHES:
                    snippet = line.strip()[:_MAX_LINE_LEN]
                    matches.append(f"{rel}:{lineno}: {snippet}")
    if total == 0:
        return f"search_code: no matches for {query!r}."
    out = "\n".join(matches)
    if total > len(matches):
        out += f"\n... ({total - len(matches)} more matches truncated)"
    return out


def search_code(args: dict, ctx: ToolContext) -> str:
    """Search file contents for ``args['query']`` across the workspace.

    Uses ripgrep (``rg``) when available, otherwise a pure-Python walk. An
    optional ``glob`` filter (e.g. ``"*.py"``) restricts which files are scanned.
    Results are returned as ``path:line: text`` and capped.
    """
    query = args.get("query")
    if not query:
        return "search_code error: 'query' is required"
    glob = args.get("glob")
    if shutil.which("rg"):
        return _rg_search(query, glob, ctx)
    return _python_search(query, glob, ctx)


def semantic_search(args: dict, ctx: ToolContext) -> str:
    """Semantic (embedding) search over the codebase when an index is enabled.

    When ``ctx.index`` is present it is queried; otherwise a graceful notice is
    returned so the model falls back to :func:`search_code`.
    """
    query = args.get("query")
    if not query:
        return "semantic_search error: 'query' is required"
    if ctx.index is not None:
        try:
            return str(ctx.index.search(query))
        except Exception as exc:  # noqa: BLE001 - surface any index failure to model
            return f"semantic_search error: {exc}"
    return (
        "semantic_search: FAISS index not enabled in this environment; "
        "use search_code instead."
    )
