"""Run an Issue2PR evaluation suite and print a results table.

For each task in a suite this harness:

1. Copies the task's repo into a fresh temporary directory (isolation).
2. Runs :func:`app.agent.loop.run_agent` against the task's issue text using a
   :class:`~app.sandbox.runner.LocalRunner`.
3. Independently verifies resolution by running ``pytest`` in the temp copy
   (``resolved`` == tests pass), so the metric does not depend on whether the
   model remembered to call ``finish``.
4. Records resolved / steps / cost / wall-clock time and prints a summary.

Usage::

    python -m evals.run --suite smoke
    python -m evals.run --suite smoke --model openai/gpt-oss-120b

Cost is an APPROXIMATE budgeting estimate (see :func:`app.agent.budgets.estimate_cost`),
accumulated via a thin cost-tracking wrapper around the real LLM client. Runs
without an API key still complete (reported as unresolved) so the harness is
safe to invoke in CI / offline environments.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.budgets import estimate_cost
from app.agent.loop import run_agent
from app.config import Settings, get_settings
from app.sandbox.runner import LocalRunner

# Directory holding the bundled suites, resolved relative to this file.
SUITES_DIR = Path(__file__).resolve().parent / "suites"


@dataclass
class TaskResult:
    """Outcome of evaluating a single task."""

    name: str
    resolved: bool
    steps: int
    cost_usd: float
    seconds: float
    detail: str


class _CostTrackingLLM:
    """Wrap a real ``LLMClient`` to accumulate approximate call cost.

    Delegates ``chat`` to the wrapped client and sums ``estimate_cost`` over
    every response's usage. Exposes ``total_cost`` for the harness to read.
    """

    def __init__(self, inner: Any, model: str) -> None:
        self._inner = inner
        self._model = model
        self.total_cost: float = 0.0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ):  # type: ignore[no-untyped-def]
        """Delegate to the inner client and record the call's estimated cost."""
        result = self._inner.chat(messages, tools=tools, tool_choice=tool_choice)
        try:
            self.total_cost += estimate_cost(result.usage, self._model)
        except Exception:  # noqa: BLE001 - cost tracking must never break a run
            pass
        return result


def _discover_tasks(suite: str) -> list[Path]:
    """Return task directories (those containing ``task.json``) for a suite."""
    suite_dir = SUITES_DIR / suite
    if not suite_dir.is_dir():
        available = ", ".join(p.name for p in SUITES_DIR.iterdir() if p.is_dir()) or "(none)"
        raise SystemExit(f"suite not found: {suite_dir}\navailable: {available}")
    tasks = sorted(p.parent for p in suite_dir.glob("*/task.json"))
    if not tasks:
        raise SystemExit(f"no tasks (dirs with task.json) found under {suite_dir}")
    return tasks


def _load_task(task_dir: Path) -> tuple[str, Path]:
    """Load a task's issue text and absolute repo path from ``task.json``."""
    spec = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    issue = str(spec.get("issue", "")).strip()
    repo_rel = str(spec.get("repo", ".")).strip() or "."
    repo_path = (task_dir / repo_rel).resolve()
    if not repo_path.is_dir():
        raise SystemExit(f"task {task_dir.name}: repo path does not exist: {repo_path}")
    return issue, repo_path


def _verify_resolved(workspace: Path) -> bool:
    """Return True if ``pytest`` passes in ``workspace`` (independent check)."""
    runner = LocalRunner(workspace, timeout_s=120)
    out = runner.run([sys.executable, "-m", "pytest", "-q"])
    return out.returncode == 0 and not out.timed_out


def _build_llm(settings: Settings) -> _CostTrackingLLM | None:
    """Construct a cost-tracking LLM client, or None if construction fails."""
    try:
        from app.llm.client import LLMClient

        return _CostTrackingLLM(LLMClient(settings), settings.llm_model)
    except Exception as exc:  # noqa: BLE001 - fall back to loop's default handling
        print(f"  (warning: could not build LLM client: {exc})")
        return None


def evaluate_task(task_dir: Path, settings: Settings) -> TaskResult:
    """Evaluate a single task in an isolated temp copy of its repo."""
    issue, repo_path = _load_task(task_dir)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"eval-{task_dir.name}-"))
    workspace = tmp_root / "repo"
    try:
        shutil.copytree(repo_path, workspace)
        runner = LocalRunner(workspace, timeout_s=min(120, settings.agent_wall_clock_s))
        llm = _build_llm(settings)

        started = time.time()
        result = run_agent(issue, workspace, settings=settings, llm=llm, runner=runner)
        elapsed = time.time() - started

        resolved = _verify_resolved(workspace)
        cost = llm.total_cost if llm is not None else 0.0
        detail = result.summary if result.finished else (result.error or result.summary)
        return TaskResult(
            name=task_dir.name,
            resolved=resolved,
            steps=result.steps,
            cost_usd=cost,
            seconds=elapsed,
            detail=(detail or "")[:80],
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _print_table(results: list[TaskResult]) -> None:
    """Print a fixed-width summary table and an aggregate line."""
    header = f"{'task':<24} {'resolved':<9} {'steps':>5} {'cost($)':>9} {'time(s)':>8}  detail"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<24} {('yes' if r.resolved else 'no'):<9} "
            f"{r.steps:>5} {r.cost_usd:>9.4f} {r.seconds:>8.1f}  {r.detail}"
        )
    resolved = sum(1 for r in results if r.resolved)
    total_cost = sum(r.cost_usd for r in results)
    total_time = sum(r.seconds for r in results)
    print("-" * len(header))
    print(
        f"{'TOTAL':<24} {f'{resolved}/{len(results)}':<9} "
        f"{sum(r.steps for r in results):>5} {total_cost:>9.4f} {total_time:>8.1f}"
    )


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the suite, print the table. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Run an Issue2PR eval suite.")
    parser.add_argument("--suite", default="smoke", help="Suite name under evals/suites/.")
    parser.add_argument("--model", default=None, help="Override the LLM model id.")
    args = parser.parse_args(argv)

    settings = get_settings()
    if args.model:
        settings = settings.model_copy(update={"llm_model": args.model})

    tasks = _discover_tasks(args.suite)
    print(f"suite '{args.suite}': {len(tasks)} task(s), model '{settings.llm_model}'")

    results: list[TaskResult] = []
    for task_dir in tasks:
        print(f"\n-> {task_dir.name}")
        results.append(evaluate_task(task_dir, settings))

    _print_table(results)

    # Exit non-zero if any task was unresolved (useful as a CI gate).
    return 0 if all(r.resolved for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
