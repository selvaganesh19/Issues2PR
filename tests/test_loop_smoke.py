"""Smoke test for the agent loop (:func:`app.agent.loop.run_agent`).

Runs the full loop with an injected :class:`FakeLLM` (no network) and a real
:class:`~app.sandbox.runner.LocalRunner`. The scripted model fixes a buggy
``add``, runs the real (offline) pytest suite in the workspace, then finishes.
"""

from __future__ import annotations

import pathlib

from app.agent.loop import run_agent

from tests.conftest import FakeLLM, tool_call
from app.llm.client import ChatResult, Usage


def _seed_buggy_repo(workspace: pathlib.Path) -> None:
    """Create a minimal package with a failing test into ``workspace``."""
    (workspace / "mathlib").mkdir(parents=True, exist_ok=True)
    (workspace / "mathlib" / "__init__.py").write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8"
    )
    (workspace / "test_add.py").write_text(
        "from mathlib import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        encoding="utf-8",
    )


def test_run_agent_finishes_offline(workspace, runner, settings):
    _seed_buggy_repo(workspace)
    fake = FakeLLM(FakeLLM.default_script())

    result = run_agent(
        issue_text="add(2, 3) returns -1 instead of 5. Please fix.",
        workspace=workspace,
        settings=settings,
        llm=fake,
        runner=runner,
    )

    assert result.finished is True
    assert "add" in result.summary.lower()
    # The patch was actually applied to the workspace file.
    assert "a + b" in (workspace / "mathlib" / "__init__.py").read_text(encoding="utf-8")
    # The loop consumed exactly the three scripted steps.
    assert result.steps == 3
    assert result.error is None


def test_run_agent_stops_after_max_test_retries(workspace, runner, settings):
    _seed_buggy_repo(workspace)
    # Script keeps running tests without ever fixing the bug -> tests keep
    # FAILED; the loop must stop after agent_max_test_retries.
    run_tests_step = ChatResult(
        content=None,
        tool_calls=[tool_call("run_tests", {})],
        usage=Usage(10, 2),
    )
    script = [run_tests_step for _ in range(settings.agent_max_test_retries + 2)]
    fake = FakeLLM(script)

    result = run_agent(
        issue_text="fix it",
        workspace=workspace,
        settings=settings,
        llm=fake,
        runner=runner,
    )

    assert result.finished is False
    assert result.error == "max_test_retries_exceeded"
