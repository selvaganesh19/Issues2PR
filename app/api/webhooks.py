"""GitHub webhook receiver.

Handles ``POST /webhooks/github``:

* Verifies the ``X-Hub-Signature-256`` HMAC-SHA256 signature against
  ``settings.github_webhook_secret`` using a constant-time comparison.
* De-duplicates redelivered events by ``X-GitHub-Delivery`` id.
* Enforces the ``ALLOWED_REPOS`` allowlist.
* Enqueues an agent job when an issue is labeled ``agent-fix`` or an issue
  comment contains ``/agent fix``.

The webhook body is treated purely as untrusted data; nothing from the payload
is executed or interpreted as an instruction here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from collections import OrderedDict

from app.config import get_settings

try:  # FastAPI is an optional ("server") extra; the pure helpers in this module
    # (verify_signature, dedup, allowlist, trigger parsing) must stay importable
    # and testable without it installed.
    from fastapi import APIRouter, Header, Request, Response, status

    _FASTAPI_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - core-only installs
    _FASTAPI_AVAILABLE = False

logger = logging.getLogger("issue2pr.webhooks")

if _FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/webhooks", tags=["webhooks"])

#: Label that triggers an agent run when applied to an issue.
TRIGGER_LABEL = "agent-fix"
#: Comment command that triggers an agent run.
TRIGGER_COMMAND = "/agent fix"

#: Bounded in-memory set of recently seen delivery ids (dedup across retries).
#: Ordered so we can evict the oldest entries once the cap is reached.
_SEEN_DELIVERIES: OrderedDict[str, None] = OrderedDict()
_SEEN_MAX = 2048


def verify_signature(
    secret: str | None,
    payload: bytes,
    signature_header: str | None,
) -> bool:
    """Return ``True`` when ``signature_header`` matches ``payload``.

    Args:
        secret: The shared webhook secret (``settings.github_webhook_secret``).
        payload: The raw request body bytes (must be the exact bytes received).
        signature_header: Value of the ``X-Hub-Signature-256`` header, e.g.
            ``"sha256=abcdef..."``.

    Returns:
        ``True`` only if a secret is configured, a signature was supplied, and
        the HMAC-SHA256 digest matches using a constant-time comparison.
    """
    if not secret or not signature_header:
        return False
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    expected = f"sha256={digest}"
    return hmac.compare_digest(expected, signature_header)


def _mark_seen(delivery_id: str) -> bool:
    """Record ``delivery_id``; return ``True`` if it was already seen."""
    if delivery_id in _SEEN_DELIVERIES:
        return True
    _SEEN_DELIVERIES[delivery_id] = None
    while len(_SEEN_DELIVERIES) > _SEEN_MAX:
        _SEEN_DELIVERIES.popitem(last=False)
    return False


def _repo_allowed(full_name: str | None, allowed_repos: str) -> bool:
    """Return whether ``full_name`` (``owner/repo``) passes the allowlist.

    ``allowed_repos`` is ``"*"`` (allow all) or a comma-separated list of
    ``owner/repo`` entries.
    """
    allow = (allowed_repos or "").strip()
    if allow == "*" or not allow:
        return True
    if not full_name:
        return False
    entries = {e.strip() for e in allow.split(",") if e.strip()}
    return full_name in entries


def _should_trigger(event: str, payload: dict) -> tuple[bool, str]:
    """Decide whether ``event``/``payload`` should start an agent run.

    Returns a ``(triggered, reason)`` tuple; ``reason`` is a short human string.
    """
    if event == "issues" and payload.get("action") == "labeled":
        label = (payload.get("label") or {}).get("name", "")
        if label == TRIGGER_LABEL:
            return True, f"issue labeled '{TRIGGER_LABEL}'"
    if event == "issue_comment" and payload.get("action") in {"created", "edited"}:
        body = (payload.get("comment") or {}).get("body", "") or ""
        if TRIGGER_COMMAND in body.lower():
            return True, f"comment command '{TRIGGER_COMMAND}'"
    return False, ""


def _build_job(payload: dict, reason: str) -> dict:
    """Extract a minimal, serialisable job description from the payload."""
    repo = payload.get("repository") or {}
    issue = payload.get("issue") or {}
    installation = payload.get("installation") or {}
    return {
        "repo_full_name": repo.get("full_name"),
        "repo_clone_url": repo.get("clone_url"),
        "default_branch": repo.get("default_branch"),
        "issue_number": issue.get("number"),
        "issue_title": issue.get("title"),
        "issue_body": issue.get("body"),
        "installation_id": installation.get("id"),
        "reason": reason,
    }


def _enqueue(job: dict) -> None:
    """Enqueue an agent job, degrading gracefully if the queue is unavailable.

    The worker queue lives in :mod:`app.workers.queue` and may be filled in by
    another component. We look up common entrypoints at call time so importing
    this module never hard-depends on Redis being present.
    """
    try:
        from app.workers import queue as queue_mod  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - queue module optional in dev
        logger.warning("workers.queue unavailable; dropping job for %s", job.get("repo_full_name"))
        return

    for fn_name in ("enqueue_agent_job", "enqueue_job", "enqueue"):
        fn = getattr(queue_mod, fn_name, None)
        if callable(fn):
            fn(job)
            return
    logger.warning("workers.queue exposes no enqueue function; dropping job")


if _FASTAPI_AVAILABLE:

    @router.post("/github")
    async def github_webhook(
        request: Request,
        response: Response,
        x_hub_signature_256: str | None = Header(default=None),
        x_github_event: str | None = Header(default=None),
        x_github_delivery: str | None = Header(default=None),
    ) -> dict[str, object]:
        """Receive and dispatch a GitHub webhook delivery."""
        settings = get_settings()

        raw = await request.body()

        if not verify_signature(settings.github_webhook_secret, raw, x_hub_signature_256):
            response.status_code = status.HTTP_401_UNAUTHORIZED
            return {"status": "invalid signature"}

        if x_github_delivery and _mark_seen(x_github_delivery):
            return {"status": "duplicate", "delivery": x_github_delivery}

        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {"status": "invalid payload"}

        event = x_github_event or ""
        repo_full_name = (payload.get("repository") or {}).get("full_name")

        if not _repo_allowed(repo_full_name, settings.allowed_repos):
            response.status_code = status.HTTP_403_FORBIDDEN
            return {"status": "repo not allowed", "repo": repo_full_name}

        triggered, reason = _should_trigger(event, payload)
        if not triggered:
            return {"status": "ignored", "event": event}

        job = _build_job(payload, reason)
        _enqueue(job)
        logger.info("enqueued agent job for %s (%s)", repo_full_name, reason)
        return {"status": "enqueued", "repo": repo_full_name, "reason": reason}
