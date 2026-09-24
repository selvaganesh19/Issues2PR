"""Tests for the filesystem tools (:mod:`app.tools.fs`)."""

from __future__ import annotations

import pathlib

import pytest

from app.tools.fs import _resolve, apply_patch, read_file


def _write(ws: pathlib.Path, rel: str, text: str) -> None:
    target = ws / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def test_apply_patch_happy_path(ctx, workspace):
    _write(workspace, "mathlib/__init__.py", "def add(a, b):\n    return a - b\n")
    out = apply_patch(
        {"path": "mathlib/__init__.py", "old": "return a - b", "new": "return a + b"},
        ctx,
    )
    assert "updated" in out
    assert (workspace / "mathlib/__init__.py").read_text(encoding="utf-8") == (
        "def add(a, b):\n    return a + b\n"
    )


def test_apply_patch_old_not_found(ctx, workspace):
    _write(workspace, "a.py", "x = 1\n")
    out = apply_patch({"path": "a.py", "old": "y = 2", "new": "z = 3"}, ctx)
    assert "not found" in out
    assert (workspace / "a.py").read_text(encoding="utf-8") == "x = 1\n"


def test_apply_patch_rejects_non_unique(ctx, workspace):
    _write(workspace, "a.py", "v = 1\nv = 1\n")
    out = apply_patch({"path": "a.py", "old": "v = 1", "new": "v = 2"}, ctx)
    assert "unique" in out


def test_read_file_full(ctx, workspace):
    _write(workspace, "notes.txt", "hello\nworld\n")
    out = read_file({"path": "notes.txt"}, ctx)
    assert "hello" in out and "world" in out


def test_read_file_line_range(ctx, workspace):
    _write(workspace, "lines.txt", "one\ntwo\nthree\nfour\n")
    out = read_file({"path": "lines.txt", "start": 2, "end": 3}, ctx)
    assert "two" in out and "three" in out
    assert "one" not in out and "four" not in out


def test_read_file_missing(ctx):
    out = read_file({"path": "nope.txt"}, ctx)
    assert "does not exist" in out


def test_resolve_rejects_traversal(workspace):
    with pytest.raises(ValueError):
        _resolve(workspace, "../../etc/passwd")


def test_resolve_rejects_absolute(workspace):
    with pytest.raises(ValueError):
        _resolve(workspace, "C:/Windows/system32")
    with pytest.raises(ValueError):
        _resolve(workspace, "/etc/passwd")


def test_resolve_allows_root_and_children(workspace):
    assert _resolve(workspace, ".") == workspace.resolve()
    assert _resolve(workspace, "sub/file.py") == (workspace / "sub/file.py").resolve()
