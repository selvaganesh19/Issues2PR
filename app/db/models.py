"""Async SQLAlchemy ORM models for Issue2PR persistence.

The schema captures one row per agent :class:`Run`, its ordered :class:`Step`
records, the individual :class:`ToolCall` invocations within each step, and a
:class:`Delivery` table used purely for GitHub webhook idempotency (the
delivery id is the primary key, so a repeated delivery collides on insert).

These models use SQLAlchemy 2.0 typed ``Mapped`` declarations and are engine
agnostic; the async engine is configured in :mod:`app.db.session`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    """Return a timezone-aware UTC timestamp (Python-side default)."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all Issue2PR ORM models."""


class Run(Base):
    """A single autonomous agent run against one GitHub issue."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo: Mapped[str] = mapped_column(String(255), nullable=False)
    issue_number: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    steps: Mapped[list[Step]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="Step.index",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"<Run id={self.id} repo={self.repo!r} "
            f"issue={self.issue_number} status={self.status!r}>"
        )


class Step(Base):
    """One iteration of the agent loop within a :class:`Run`."""

    __tablename__ = "steps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    index: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)

    run: Mapped[Run] = relationship(back_populates="steps")
    tool_calls: Mapped[list[ToolCall]] = relationship(
        back_populates="step",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Step id={self.id} run_id={self.run_id} index={self.index} kind={self.kind!r}>"


class ToolCall(Base):
    """A single tool invocation recorded within a :class:`Step`."""

    __tablename__ = "tool_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    step_id: Mapped[int] = mapped_column(
        ForeignKey("steps.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    args_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    result_excerpt: Mapped[str] = mapped_column(Text, nullable=False, default="")

    step: Mapped[Step] = relationship(back_populates="tool_calls")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<ToolCall id={self.id} step_id={self.step_id} name={self.name!r}>"


class Delivery(Base):
    """GitHub webhook delivery marker used for idempotency.

    The primary key is the GitHub ``X-GitHub-Delivery`` id, so re-processing the
    same delivery raises a unique-constraint / integrity error the caller can
    treat as "already handled".
    """

    __tablename__ = "deliveries"

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow, server_default=func.now()
    )

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<Delivery id={self.id!r} received_at={self.received_at}>"
