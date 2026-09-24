"""Tests for the search tools (:mod:`app.tools.search`)."""

from __future__ import annotations

import app.tools.search as search_mod
from app.tools.search import search_code, semantic_search


def _force_rg_absent(monkeypatch):
    """Make ``shutil.which('rg')`` return None so the Python fallback is used."""
    monkeypatch.setattr(search_mod.shutil, "which", lambda name: None)


def test_python_fallback_finds_known_string(ctx, workspace, monkeypatch):
    _force_rg_absent(monkeypatch)
    (workspace / "target.py").write_text(
        "def add(a, b):\n    return a - b  # NEEDLE_MARKER\n", encoding="utf-8"
    )
    (workspace / "other.py").write_text("x = 1\n", encoding="utf-8")

    out = search_code({"query": "NEEDLE_MARKER"}, ctx)
    assert "target.py" in out
    assert "NEEDLE_MARKER" in out
    assert "other.py" not in out


def test_python_fallback_respects_glob(ctx, workspace, monkeypatch):
    _force_rg_absent(monkeypatch)
    (workspace / "a.py").write_text("token_here\n", encoding="utf-8")
    (workspace / "a.txt").write_text("token_here\n", encoding="utf-8")

    out = search_code({"query": "token_here", "glob": "*.py"}, ctx)
    assert "a.py" in out
    assert "a.txt" not in out


def test_python_fallback_no_matches(ctx, workspace, monkeypatch):
    _force_rg_absent(monkeypatch)
    (workspace / "a.py").write_text("nothing interesting\n", encoding="utf-8")
    out = search_code({"query": "absent_string"}, ctx)
    assert "no matches" in out


def test_search_code_requires_query(ctx):
    out = search_code({}, ctx)
    assert "required" in out


def test_semantic_search_without_index(ctx):
    out = semantic_search({"query": "how does add work"}, ctx)
    assert "FAISS index not enabled" in out
    assert "search_code" in out
