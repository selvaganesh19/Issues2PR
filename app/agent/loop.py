"""The core agent loop: drive an LLM + tools to turn an issue into a fix.

:func:`run_agent` builds a tool registry, a run budget, and the initial
messages, then repeatedly calls the LLM. Each requested tool call is dispatched
to its handler and the result fed back as a ``role="tool"`` message. When the
model calls ``finish`` the run ends successfully; budgets (steps / cost /
wall-clock) and a test-retry ceiling bound the run otherwise.

The LLM and runner are injectable so tests can pass fakes (any object with a
``.chat(messages, tools, tool_choice)`` method / a ``.run(cmd)`` method).
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass
from typing import Any

from app.agent.budgets import BudgetExhausted, RunBudget, estimate_cost
from app.agent.prompts import build_messages
from app.sandbox.runner import get_runner
from app.tools.registry import ToolContext, build_registry, to_openai_tools

# Cap on how much of a single tool result is fed back to the model, to keep the
# context bounded. Handlers are expected to truncate large output themselves;
# this is a final safety net.
_MAX_TOOL_RESULT_CHARS = 4000

# Approx char budget for the message history sent each turn. Providers count the
# whole prompt (system + history + tool schemas) against a tokens-per-minute
# limit; low free tiers (e.g. Groq's 8k TPM) reject a single oversized request
# with 413. We keep the running history under this budget by eliding the content
# of older tool/assistant messages (their structure and tool_call_id pairing are
# preserved so the API stays happy — only the text is collapsed). ~4 chars/token.
_MAX_HISTORY_CHARS = 12000
_ELIDED = "[older output elided to fit the context window]"


def _history_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) for m in messages)


def _trim_history(messages: list[dict]) -> None:
    """Shrink the history in place to stay under ``_MAX_HISTORY_CHARS``.

    Keeps the system prompt, the first user message (the issue), and the most
    recent messages intact; collapses the *content* of older ``tool`` and
    ``assistant`` messages to a short placeholder. Messages are never removed,
    so every ``tool`` message keeps its matching ``tool_calls`` parent.
    """
    if _history_chars(messages) <= _MAX_HISTORY_CHARS:
        return
    # Protect: system (0), first user (issue), and the last 4 messages.
    protected = set()
    if messages:
        protected.add(0)
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            protected.add(i)
            break
    protected.update(range(max(0, len(messages) - 4), len(messages)))

    for i, m in enumerate(messages):
        if i in protected:
            continue
        if m.get("role") in ("tool", "assistant") and m.get("content"):
            if len(str(m["content"])) > len(_ELIDED):
                m["content"] = _ELIDED
        if _history_chars(messages) <= _MAX_HISTORY_CHARS:
            break



@dataclass
class RunResult:
    """Outcome of an agent run.

    Attributes:
        finished: True if the model called ``finish`` (a clean completion).
        summary: The model's summary, or a reason the run stopped.
        steps: Number of agent steps consumed.
        diff: The final ``git diff`` of the workspace (may be empty).
        error: Reason string when the run did not finish cleanly, else None.
    """

    finished: bool
    summary: str
    steps: int
    diff: str
    error: str | None = None


def _repo_overview(workspace: pathlib.Path, max_entries: int = 60) -> str:
    """Return a short listing of the workspace top level to orient the model."""
    try:
        entries = sorted(
            p.name + ("/" if p.is_dir() else "") for p in workspace.iterdir()
        )
    except OSError as exc:
        return f"(could not list workspace: {exc})"
    if not entries:
        return "(empty workspace)"
    shown = entries[:max_entries]
    text = "\n".join(shown)
    if len(entries) > max_entries:
        text += f"\n... ({len(entries) - max_entries} more)"
    return text


def _assistant_tool_message(content: str | None, tool_calls: list) -> dict:
    """Reconstruct the assistant turn (with tool calls) in OpenAI format.

    Arguments are re-serialised to JSON strings so downstream ``role="tool"``
    messages correlate by ``tool_call_id``.
    """
    return {
        "role": "assistant",
        "content": content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ],
    }


def _truncate(text: str) -> str:
    """Clamp a tool result string to the loop's safety cap."""
    if len(text) <= _MAX_TOOL_RESULT_CHARS:
        return text
    head = text[:_MAX_TOOL_RESULT_CHARS]
    return head + f"\n[... truncated {len(text) - _MAX_TOOL_RESULT_CHARS} chars]"


def _final_diff(registry: dict, ctx: ToolContext) -> str:
    """Best-effort capture of the workspace diff for the result."""
    spec = registry.get("git_diff")
    if spec is None:
        return ""
    try:
        return spec.handler({}, ctx)
    except Exception:  # noqa: BLE001 - diff is best-effort context only
        return ""


def run_agent(
    issue_text: str,
    workspace: str | pathlib.Path,
    settings: Any = None,
    llm: Any = None,
    runner: Any = None,
) -> RunResult:
    """Run the autonomous agent against ``issue_text`` in ``workspace``.

    Args:
        issue_text: Raw issue text (treated as untrusted data in the prompt).
        workspace: Path to the checked-out repository the agent may edit.
        settings: Optional :class:`~app.config.Settings`; loaded via
            ``get_settings()`` when omitted.
        llm: Optional chat client with ``.chat(messages, tools, tool_choice)``.
            Defaults to a real :class:`~app.llm.client.LLMClient`. Injectable for
            tests.
        runner: Optional sandbox runner with ``.run(cmd)``. Defaults to the
            runner selected by ``settings.sandbox_backend``.

    Returns:
        A :class:`RunResult`. ``finished`` is True only when the model called
        ``finish``; budget exhaustion or the test-retry ceiling yield
        ``finished=False`` with ``error`` set.
    """
    if settings is None:
        from app.config import get_settings

        settings = get_settings()

    workspace = pathlib.Path(workspace)

    if runner is None:
        runner = get_runner(settings, workspace)

    if llm is None:
        from app.llm.client import LLMClient

        llm = LLMClient(settings)

    ctx = ToolContext(workspace=workspace, runner=runner, settings=settings)
    registry = build_registry(ctx)
    tools = to_openai_tools(registry)

    budget = RunBudget(
        max_steps=settings.agent_max_steps,
        max_cost_usd=settings.agent_max_cost_usd,
        wall_clock_s=settings.agent_wall_clock_s,
        started_at=time.time(),
    )

    messages = build_messages(issue_text, _repo_overview(workspace))
    test_failures = 0

    try:
        while True:
            budget.tick_step()
            budget.check()

            _trim_history(messages)
            result = llm.chat(messages, tools=tools, tool_choice="auto")
            budget.add_cost(estimate_cost(result.usage, settings.llm_model))

            # No tool calls: the model produced prose. Record it and nudge it
            # back toward acting/finishing via tools.
            if not result.tool_calls:
                messages.append(
                    {"role": "assistant", "content": result.content or ""}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue by using the tools. When the issue is "
                            "fully resolved and tests pass, call the finish "
                            "tool with a summary."
                        ),
                    }
                )
                continue

            messages.append(
                _assistant_tool_message(result.content, result.tool_calls)
            )

            for tc in result.tool_calls:
                if tc.name == "finish":
                    summary = str(tc.arguments.get("summary", "done"))
                    return RunResult(
                        finished=True,
                        summary=summary,
                        steps=budget.steps_used,
                        diff=_final_diff(registry, ctx),
                    )

                spec = registry.get(tc.name)
                if spec is None:
                    output = f"error: unknown tool '{tc.name}'"
                else:
                    try:
                        output = spec.handler(tc.arguments, ctx)
                    except Exception as exc:  # noqa: BLE001 - report to model
                        output = f"error executing {tc.name}: {exc}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": _truncate(str(output)),
                    }
                )

                # Self-repair accounting: count failed test runs and stop once
                # the retry ceiling is hit so we don't loop forever.
                if tc.name == "run_tests" and "FAILED" in str(output):
                    test_failures += 1
                    if test_failures >= settings.agent_max_test_retries:
                        return RunResult(
                            finished=False,
                            summary=(
                                "Tests still failing after "
                                f"{test_failures} attempts."
                            ),
                            steps=budget.steps_used,
                            diff=_final_diff(registry, ctx),
                            error="max_test_retries_exceeded",
                        )
    except BudgetExhausted as exc:
        return RunResult(
            finished=False,
            summary=f"Run stopped: {exc.reason}.",
            steps=budget.steps_used,
            diff=_final_diff(registry, ctx),
            error=exc.reason,
        )
    except Exception as exc:  # noqa: BLE001 - surface as a failed run, not a crash
        return RunResult(
            finished=False,
            summary=f"Run aborted due to an unexpected error: {exc}",
            steps=budget.steps_used,
            diff=_final_diff(registry, ctx),
            error=str(exc),
        )
