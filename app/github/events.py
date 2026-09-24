"""Pydantic models for the GitHub webhook payloads Issue2PR consumes.

Only the fields the agent actually needs are modelled; ``extra="ignore"`` keeps
us forward-compatible with GitHub's large, evolving payloads.

Security note: the *content* of these payloads (issue titles/bodies, comment
bodies) is **untrusted data**. It is wrapped in a delimited UNTRUSTED block by
``app.agent.prompts`` before ever reaching the model and is never interpreted as
instructions here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    """Base model that tolerates GitHub's many extra payload fields."""

    model_config = ConfigDict(extra="ignore")


class Repository(_Base):
    """A GitHub repository as it appears in webhook payloads."""

    id: int
    name: str
    full_name: str
    default_branch: str = "main"
    private: bool = False


class Issue(_Base):
    """A GitHub issue (also used for the issue attached to a comment event)."""

    number: int
    title: str
    body: str | None = None
    state: str = "open"


class Installation(_Base):
    """The GitHub App installation the event belongs to."""

    id: int


class IssuesEvent(_Base):
    """Payload delivered on the ``issues`` webhook event.

    ``action`` is typically ``opened``, ``edited``, ``labeled`` etc.
    """

    action: str
    issue: Issue
    repository: Repository
    installation: Installation | None = None


class Comment(_Base):
    """A single issue comment body."""

    id: int
    body: str | None = None


class IssueCommentEvent(_Base):
    """Payload delivered on the ``issue_comment`` webhook event."""

    action: str
    issue: Issue
    comment: Comment
    repository: Repository
    installation: Installation | None = None
