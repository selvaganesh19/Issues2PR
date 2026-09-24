"""Tool registry: specs, execution context, and OpenAI-schema conversion.

The concrete tool handlers live in :mod:`app.tools.fs`, :mod:`app.tools.search`,
:mod:`app.tools.shell`, and :mod:`app.tools.git_ops`. They are imported lazily
inside :func:`build_registry` to avoid import cycles (those modules may import
from :mod:`app.sandbox` / :mod:`app.config`).
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolContext:
    """Shared state passed to every tool handler.

    Attributes:
        workspace: Root directory the agent is allowed to operate within.
        runner: Sandbox runner exposing ``run(cmd: list[str]) -> RunOutput``.
        settings: Application :class:`~app.config.Settings`.
        index: Optional semantic search index (None when disabled).
        log: Optional logger/callable for structured events.
    """

    workspace: pathlib.Path
    runner: Any
    settings: Any
    index: Any = None
    log: Any = None


# A handler takes parsed JSON args plus the context and returns a
# human-readable string that is fed back to the model as a tool message.
ToolHandler = Callable[[dict, ToolContext], str]


@dataclass
class ToolSpec:
    """Definition of a single callable tool exposed to the model."""

    name: str
    description: str
    parameters: dict
    handler: ToolHandler


def build_registry(ctx: ToolContext) -> dict[str, ToolSpec]:
    """Construct the full tool registry wired to ``ctx``.

    Handlers are imported here (call time) rather than at module import to keep
    the registry importable even when optional tool dependencies are absent and
    to avoid circular imports.
    """
    from app.tools import fs, git_ops, search, shell

    def _finish(args: dict, _ctx: ToolContext) -> str:
        """Signal completion. Handled specially by the agent loop."""
        return args.get("summary", "done")

    specs: list[ToolSpec] = [
        ToolSpec(
            name="list_dir",
            description="List files and directories under a path (relative to the workspace root).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory relative to workspace root. Defaults to '.'.",
                    }
                },
            },
            handler=fs.list_dir,
        ),
        ToolSpec(
            name="read_file",
            description="Read a text file, optionally a line range (1-indexed, inclusive).",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace."},
                    "start": {"type": "integer", "description": "Optional start line (1-indexed)."},
                    "end": {"type": "integer", "description": "Optional end line (inclusive)."},
                },
                "required": ["path"],
            },
            handler=fs.read_file,
        ),
        ToolSpec(
            name="search_code",
            description="Search file contents for a query string across the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text or regex to search for."},
                    "glob": {"type": "string", "description": "Optional glob filter, e.g. '*.py'."},
                },
                "required": ["query"],
            },
            handler=search.search_code,
        ),
        ToolSpec(
            name="semantic_search",
            description="Semantic (embedding) search over the codebase, when an index is enabled.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language query."},
                },
                "required": ["query"],
            },
            handler=search.semantic_search,
        ),
        ToolSpec(
            name="apply_patch",
            description=(
                "Edit a file by replacing an exact 'old' snippet with 'new'. "
                "The 'old' text must appear exactly once."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to workspace."},
                    "old": {"type": "string", "description": "Exact existing text to replace."},
                    "new": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old", "new"],
            },
            handler=fs.apply_patch,
        ),
        ToolSpec(
            name="run_tests",
            description="Run the project's test suite (pytest) and report PASSED/FAILED.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional path/target to test."},
                },
            },
            handler=shell.run_tests,
        ),
        ToolSpec(
            name="run_linter",
            description="Run the linter (ruff check) and report PASSED/FAILED.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Optional path to lint."},
                },
            },
            handler=shell.run_linter,
        ),
        ToolSpec(
            name="git_diff",
            description="Show the current uncommitted git diff for the workspace.",
            parameters={"type": "object", "properties": {}},
            handler=git_ops.git_diff,
        ),
        ToolSpec(
            name="finish",
            description="Call when the issue is resolved. Provide a summary of the changes made.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was changed and why.",
                    }
                },
                "required": ["summary"],
            },
            handler=_finish,
        ),
    ]
    return {spec.name: spec for spec in specs}


def to_openai_tools(registry: dict[str, ToolSpec]) -> list[dict]:
    """Convert a registry to the OpenAI function-tool schema list."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in registry.values()
    ]
