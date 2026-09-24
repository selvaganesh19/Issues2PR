"""Filesystem tools: safe directory listing, reading, and search/replace edits.

All paths supplied by the model are resolved relative to the workspace root and
validated by :func:`_resolve`, which rejects any path that would escape the
workspace (path traversal). Every handler conforms to the shared signature
``handler(args: dict, ctx: ToolContext) -> str`` and returns human-readable text
that is fed back to the model as a tool message.
"""

from __future__ import annotations

import pathlib

from app.tools.registry import ToolContext

# Cap how much file content we return in one read to keep the context bounded.
_MAX_READ_BYTES = 60_000
_MAX_LIST_ENTRIES = 500


def _resolve(workspace: pathlib.Path, rel: str) -> pathlib.Path:
    """Resolve ``rel`` against ``workspace`` and reject workspace escapes.

    Args:
        workspace: Absolute workspace root the agent may operate within.
        rel: A path relative to the workspace (``.`` and empty mean the root).

    Returns:
        The resolved absolute :class:`pathlib.Path` inside the workspace.

    Raises:
        ValueError: If the resolved path lies outside the workspace, or if an
            absolute path is supplied.
    """
    root = pathlib.Path(workspace).resolve()
    rel = (rel or ".").strip()
    # Reject absolute paths under BOTH POSIX and Windows semantics, regardless
    # of the host OS. On Linux a string like "C:/Windows" is not a PosixPath
    # absolute, so without the PureWindowsPath check it would be treated as a
    # relative path and silently allowed -- a cross-platform escape.
    if (
        pathlib.PurePosixPath(rel).is_absolute()
        or pathlib.PureWindowsPath(rel).is_absolute()
        or pathlib.PureWindowsPath(rel).drive
    ):
        raise ValueError(f"absolute paths are not allowed: {rel!r}")
    candidate = pathlib.Path(rel)
    resolved = (root / candidate).resolve()
    # Python 3.9+: is_relative_to guards against '..' traversal and symlinks.
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(f"path escapes workspace: {rel!r}")
    return resolved


def list_dir(args: dict, ctx: ToolContext) -> str:
    """List entries under ``args['path']`` (relative to the workspace root).

    Directories are suffixed with ``/``. Output is capped for safety.
    """
    rel = args.get("path", ".")
    try:
        target = _resolve(ctx.workspace, rel)
    except ValueError as exc:
        return f"list_dir error: {exc}"
    if not target.exists():
        return f"list_dir error: path does not exist: {rel}"
    if not target.is_dir():
        return f"list_dir error: not a directory: {rel}"

    entries = sorted(
        target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
    )
    lines: list[str] = []
    for entry in entries[:_MAX_LIST_ENTRIES]:
        name = entry.name + ("/" if entry.is_dir() else "")
        lines.append(name)
    header = f"{rel} ({len(entries)} entries):"
    if len(entries) > _MAX_LIST_ENTRIES:
        lines.append(f"... ({len(entries) - _MAX_LIST_ENTRIES} more truncated)")
    if not entries:
        return f"{header}\n(empty)"
    return header + "\n" + "\n".join(lines)


def read_file(args: dict, ctx: ToolContext) -> str:
    """Read a text file, optionally limited to a 1-indexed inclusive line range.

    Args in ``args``: ``path`` (required), ``start`` and ``end`` (optional).
    """
    rel = args.get("path")
    if not rel:
        return "read_file error: 'path' is required"
    try:
        target = _resolve(ctx.workspace, rel)
    except ValueError as exc:
        return f"read_file error: {exc}"
    if not target.exists():
        return f"read_file error: file does not exist: {rel}"
    if not target.is_file():
        return f"read_file error: not a file: {rel}"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"read_file error: {exc}"

    start = args.get("start")
    end = args.get("end")
    if start is not None or end is not None:
        lines = text.splitlines()
        s = max(1, int(start)) if start is not None else 1
        e = int(end) if end is not None else len(lines)
        e = min(e, len(lines))
        if s > e:
            return f"read_file error: start ({s}) is after end ({e})"
        selected = lines[s - 1 : e]
        numbered = "\n".join(f"{s + i}: {ln}" for i, ln in enumerate(selected))
        return f"{rel} lines {s}-{e}:\n{numbered}"

    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_READ_BYTES:
        truncated = encoded[:_MAX_READ_BYTES].decode("utf-8", errors="ignore")
        return (
            f"{rel} (truncated to {_MAX_READ_BYTES} bytes; use start/end to page):\n"
            + truncated
        )
    return f"{rel}:\n{text}"


def apply_patch(args: dict, ctx: ToolContext) -> str:
    """Replace an exact ``old`` snippet with ``new`` in a file.

    The ``old`` text must be present exactly once; otherwise the edit is
    rejected so the model can disambiguate. Returns a confirmation string.

    Args in ``args``: ``path``, ``old``, ``new`` (all required).
    """
    rel = args.get("path")
    old = args.get("old")
    new = args.get("new")
    if not rel:
        return "apply_patch error: 'path' is required"
    if old is None:
        return "apply_patch error: 'old' is required"
    if new is None:
        return "apply_patch error: 'new' is required"

    try:
        target = _resolve(ctx.workspace, rel)
    except ValueError as exc:
        return f"apply_patch error: {exc}"
    if not target.exists() or not target.is_file():
        return f"apply_patch error: file does not exist: {rel}"

    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"apply_patch error: {exc}"

    count = content.count(old)
    if count == 0:
        return (
            f"apply_patch error: 'old' text not found in {rel}. "
            "Read the file to copy the exact snippet."
        )
    if count > 1:
        return (
            f"apply_patch error: 'old' text appears {count} times in {rel}; "
            "it must be unique. Include more surrounding context."
        )

    updated = content.replace(old, new, 1)
    try:
        target.write_text(updated, encoding="utf-8")
    except OSError as exc:
        return f"apply_patch error: {exc}"

    delta = updated.count("\n") - content.count("\n")
    return f"apply_patch: {rel} updated (line delta {delta:+d})."
