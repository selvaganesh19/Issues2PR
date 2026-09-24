"""Split source files into overlapping, line-based chunks with metadata.

The semantic index works on small, overlapping windows of source lines rather
than whole files: this keeps each embedded unit focused while the overlap
avoids losing context that straddles a window boundary. Each :class:`Chunk`
carries enough metadata (path + 1-indexed line range) to point the agent at the
exact location in the workspace.

This module is dependency-free (stdlib only) so it can be imported and unit
tested without the optional ``index`` extra (faiss/numpy).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

# Default windowing parameters (in source lines).
DEFAULT_CHUNK_LINES = 60
DEFAULT_OVERLAP_LINES = 15

# Guard rails for workspace scanning.
_MAX_FILE_BYTES = 2_000_000

# File extensions considered indexable source/text.
_TEXT_EXTENSIONS = {
    ".py", ".pyi", ".txt", ".md", ".rst", ".cfg", ".ini", ".toml",
    ".yaml", ".yml", ".json", ".js", ".jsx", ".ts", ".tsx", ".html",
    ".css", ".scss", ".sh", ".bash", ".sql", ".java", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp", ".rb", ".php", ".xml", ".env",
}

# Directories that are never worth indexing.
_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build", ".tox",
}


@dataclass
class Chunk:
    """A contiguous slice of a source file.

    Attributes:
        path: Workspace-relative POSIX path of the source file.
        start_line: First line of the slice (1-indexed, inclusive).
        end_line: Last line of the slice (1-indexed, inclusive).
        text: The raw text of the slice (lines joined with ``\\n``).
        score: Optional similarity score, populated on query results.
    """

    path: str
    start_line: int
    end_line: int
    text: str
    score: float | None = field(default=None)

    def location(self) -> str:
        """Return a ``path:start-end`` human-readable location string."""
        return f"{self.path}:{self.start_line}-{self.end_line}"


def _looks_binary(data: bytes) -> bool:
    """Heuristic: treat data containing a NUL byte as binary."""
    return b"\x00" in data


def chunk_text(
    path: str,
    text: str,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[Chunk]:
    """Split ``text`` into overlapping line-based chunks.

    Args:
        path: Workspace-relative path recorded on each produced chunk.
        text: Full text of the file.
        chunk_lines: Maximum number of lines per chunk (must be > 0).
        overlap_lines: Number of lines shared between consecutive chunks
            (clamped to ``chunk_lines - 1``).

    Returns:
        A list of :class:`Chunk` objects covering the file. Empty/whitespace
        chunks are skipped. An empty ``text`` yields an empty list.
    """
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    overlap = max(0, min(overlap_lines, chunk_lines - 1))
    step = chunk_lines - overlap

    lines = text.splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    start = 0
    total = len(lines)
    while start < total:
        end = min(start + chunk_lines, total)
        window = lines[start:end]
        body = "\n".join(window)
        if body.strip():
            chunks.append(
                Chunk(
                    path=path,
                    start_line=start + 1,
                    end_line=end,
                    text=body,
                )
            )
        if end >= total:
            break
        start += step
    return chunks


def _is_indexable(path: pathlib.Path) -> bool:
    """Return True if ``path`` is a text/source file worth indexing."""
    if path.suffix.lower() not in _TEXT_EXTENSIONS:
        return False
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return False
        with path.open("rb") as fh:
            head = fh.read(1024)
    except OSError:
        return False
    return not _looks_binary(head)


def chunk_workspace(
    root: str | pathlib.Path,
    chunk_lines: int = DEFAULT_CHUNK_LINES,
    overlap_lines: int = DEFAULT_OVERLAP_LINES,
) -> list[Chunk]:
    """Walk ``root`` and produce chunks for every indexable text file.

    Skips version-control, cache, and dependency directories, binary files,
    and files larger than an internal size cap. Paths on the returned chunks
    are workspace-relative POSIX paths.

    Args:
        root: Workspace root directory to scan.
        chunk_lines: Maximum lines per chunk.
        overlap_lines: Overlap between consecutive chunks.

    Returns:
        A flat list of :class:`Chunk` objects across all files.
    """
    root_path = pathlib.Path(root).resolve()
    chunks: list[Chunk] = []
    for path in sorted(root_path.rglob("*")):
        rel_parts = path.relative_to(root_path).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if not path.is_file():
            continue
        if not _is_indexable(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root_path).as_posix()
        chunks.extend(chunk_text(rel, text, chunk_lines, overlap_lines))
    return chunks
