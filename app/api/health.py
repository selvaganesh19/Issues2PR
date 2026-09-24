"""Health and readiness probes for the Issue2PR API."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns a static payload so orchestrators can confirm the process is up.
    """
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> dict[str, object]:
    """Readiness probe.

    Reports which optional subsystems are configured. This is best-effort and
    never raises: a missing optional dependency simply shows ``False`` so the
    process still reports ready for the local/dev path.
    """
    settings = get_settings()
    checks: dict[str, bool] = {
        "github_webhook_configured": bool(settings.github_webhook_secret),
        "llm_key_configured": bool(
            settings.groq_api_key
            or settings.openrouter_api_key
            or settings.azure_openai_api_key
        ),
    }
    return {"status": "ready", "checks": checks}
