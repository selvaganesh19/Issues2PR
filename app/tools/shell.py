"""Shell tools: run the test suite and linter through the sandbox runner.

Both handlers delegate to ``ctx.runner.run(cmd)`` which enforces a timeout, so
a hanging test process can never block the agent indefinitely. Output is
truncated and tagged with a ``PASSED``/``FAILED`` marker the model can key on.
"""

from __future__ import annotations

from app.sandbox.runner import RunOutput
from app.tools.registry import ToolContext

# Keep returned output bounded; tests/linters can be very chatty.
_MAX_OUTPUT_CHARS = 8_000


def _truncate(text: str) -> str:
    """Truncate ``text`` to the output cap, keeping the tail (most relevant)."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    kept = text[-_MAX_OUTPUT_CHARS:]
    return f"[... {len(text) - _MAX_OUTPUT_CHARS} chars truncated ...]\n{kept}"


def _format(label: str, result: RunOutput) -> str:
    """Format a runner result into a human-readable PASSED/FAILED block."""
    if result.timed_out:
        marker = "FAILED (timed out)"
    elif result.returncode == 127:
        marker = "FAILED (command not found)"
    elif result.returncode == 0:
        marker = "PASSED"
    else:
        marker = f"FAILED (exit {result.returncode})"
    combined = (result.stdout or "") + (result.stderr or "")
    body = _truncate(combined.strip()) or "(no output)"
    return f"{label}: {marker}\n{body}"


def run_tests(args: dict, ctx: ToolContext) -> str:
    """Run the test suite via pytest and report PASSED/FAILED.

    An optional ``args['path']`` narrows the run to a target file/dir/node.
    """
    cmd = ["python", "-m", "pytest", "-q"]
    target = args.get("path")
    if target:
        cmd.append(str(target))
    result = ctx.runner.run(cmd)
    return _format("run_tests", result)


def run_linter(args: dict, ctx: ToolContext) -> str:
    """Run ``ruff check`` and report PASSED/FAILED.

    An optional ``args['path']`` narrows the lint scope; defaults to ``.``.
    """
    target = args.get("path") or "."
    cmd = ["ruff", "check", str(target)]
    result = ctx.runner.run(cmd)
    return _format("run_linter", result)
