"""Shared pytest fixtures for the Issue2PR suite.

Provides a temporary workspace, a :class:`~app.tools.registry.ToolContext`
wired to a real :class:`~app.sandbox.runner.LocalRunner`, and a :class:`FakeLLM`
whose ``.chat`` method replays scripted :class:`~app.llm.client.ChatResult`
objects so the agent loop can be driven without any network access.
"""

from __future__ import annotations

import pathlib

import pytest

from app.config import Settings
from app.llm.client import ChatResult, ToolCall, Usage
from app.sandbox.runner import LocalRunner
from app.tools.registry import ToolContext


@pytest.fixture
def settings() -> Settings:
    """A default Settings instance (no secrets; safe for offline tests)."""
    return Settings(agent_max_steps=10, agent_max_test_retries=3, agent_wall_clock_s=60)


@pytest.fixture
def workspace(tmp_path: pathlib.Path) -> pathlib.Path:
    """An empty temporary workspace directory."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def runner(workspace: pathlib.Path) -> LocalRunner:
    """A real LocalRunner rooted at the temp workspace with a short timeout."""
    return LocalRunner(workspace, timeout_s=60)


@pytest.fixture
def ctx(workspace: pathlib.Path, runner: LocalRunner, settings: Settings) -> ToolContext:
    """A ToolContext built from the temp workspace + LocalRunner + settings."""
    return ToolContext(workspace=workspace, runner=runner, settings=settings)


def tool_call(name: str, arguments: dict, call_id: str | None = None) -> ToolCall:
    """Helper to build a :class:`ToolCall` for scripting a FakeLLM."""
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments=arguments)


class FakeLLM:
    """A scripted, offline stand-in for :class:`~app.llm.client.LLMClient`.

    Construct with a list of :class:`ChatResult` objects (or use
    :meth:`default_script`). Each call to :meth:`chat` returns the next scripted
    result. Records every ``messages``/``tools`` invocation for assertions.
    """

    def __init__(self, script: list[ChatResult]) -> None:
        self._script = list(script)
        self._i = 0
        self.calls: list[dict] = []

    def chat(self, messages, tools=None, tool_choice="auto") -> ChatResult:
        """Return the next scripted ChatResult, recording the invocation."""
        self.calls.append(
            {"messages": messages, "tools": tools, "tool_choice": tool_choice}
        )
        if self._i >= len(self._script):
            # Safety net: never let a mis-scripted test loop forever.
            return ChatResult(
                content=None,
                tool_calls=[tool_call("finish", {"summary": "script exhausted"})],
                usage=Usage(1, 1),
            )
        result = self._script[self._i]
        self._i += 1
        return result

    @staticmethod
    def default_script() -> list[ChatResult]:
        """A three-step script: apply_patch -> run_tests -> finish.

        Fixes the buggy ``add`` in the seeded workspace, runs the test suite,
        then finishes with a summary.
        """
        return [
            ChatResult(
                content="I'll fix the subtraction bug.",
                tool_calls=[
                    tool_call(
                        "apply_patch",
                        {
                            "path": "mathlib/__init__.py",
                            "old": "return a - b",
                            "new": "return a + b",
                        },
                    )
                ],
                usage=Usage(50, 10),
            ),
            ChatResult(
                content="Now I'll run the tests.",
                tool_calls=[tool_call("run_tests", {})],
                usage=Usage(30, 5),
            ),
            ChatResult(
                content=None,
                tool_calls=[
                    tool_call(
                        "finish",
                        {"summary": "Fixed add() to return a + b; tests pass."},
                    )
                ],
                usage=Usage(20, 5),
            ),
        ]


@pytest.fixture
def fake_llm_factory():
    """Return the :class:`FakeLLM` class so tests can build custom scripts."""
    return FakeLLM
