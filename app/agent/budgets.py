"""Run budgets: step, cost, and wall-clock limits for an agent run."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.llm.client import Usage


class BudgetExhausted(Exception):
    """Raised when an agent run exceeds one of its configured limits."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# Approximate blended USD cost per 1M tokens, keyed by a substring of the model
# id. These are ROUGH estimates for budgeting only — they are not billing
# figures and providers change pricing frequently. A single blended rate is
# applied to prompt+completion tokens for simplicity.
_APPROX_USD_PER_1M: dict[str, float] = {
    "gpt-oss-120b": 0.30,
    "gpt-oss-20b": 0.15,
    "llama": 0.20,
    "gpt-4o": 5.00,
    "gpt-4": 10.00,
    "default": 0.50,
}


def estimate_cost(usage: Usage, model: str) -> float:
    """Estimate USD cost of a single LLM call. APPROXIMATE — budgeting only.

    Uses a blended per-1M-token rate matched by substring against ``model``,
    falling back to a conservative default when unknown.
    """
    total_tokens = usage.prompt_tokens + usage.completion_tokens
    rate = _APPROX_USD_PER_1M["default"]
    lowered = model.lower()
    for key, value in _APPROX_USD_PER_1M.items():
        if key == "default":
            continue
        if key in lowered:
            rate = value
            break
    return (total_tokens / 1_000_000) * rate


@dataclass
class RunBudget:
    """Tracks and enforces per-run resource limits."""

    max_steps: int
    max_cost_usd: float
    wall_clock_s: int
    started_at: float = field(default_factory=time.time)
    steps_used: int = 0
    cost_usd: float = 0.0

    def tick_step(self) -> None:
        """Record that one agent step has been consumed."""
        self.steps_used += 1

    def add_cost(self, usd: float) -> None:
        """Add ``usd`` to the accumulated run cost."""
        self.cost_usd += usd

    @property
    def remaining_steps(self) -> int:
        """Number of steps left before the step limit is hit."""
        return max(0, self.max_steps - self.steps_used)

    def check(self) -> None:
        """Raise :class:`BudgetExhausted` if any limit has been exceeded."""
        if self.steps_used >= self.max_steps:
            raise BudgetExhausted(f"step limit reached ({self.max_steps} steps)")
        if self.cost_usd >= self.max_cost_usd:
            raise BudgetExhausted(
                f"cost limit reached (${self.cost_usd:.4f} >= ${self.max_cost_usd:.2f})"
            )
        elapsed = time.time() - self.started_at
        if elapsed >= self.wall_clock_s:
            raise BudgetExhausted(f"wall-clock limit reached ({self.wall_clock_s}s)")
