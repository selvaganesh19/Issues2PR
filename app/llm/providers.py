"""LLM provider registry and ordering.

Maps a provider name to the OpenAI-compatible connection details used by
:mod:`app.llm.client`. All providers speak the OpenAI Chat Completions API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings

# provider name -> connection descriptor.
#   base_url:      OpenAI-compatible endpoint.
#   api_key_attr:  name of the Settings attribute holding the API key.
#   default_model: fallback model id when settings.llm_model is unset.
PROVIDERS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_attr": "groq_api_key",
        "default_model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_attr": "openrouter_api_key",
        "default_model": "openai/gpt-oss-120b",
    },
    "azure": {
        # base_url is resolved from settings.azure_openai_endpoint at call time.
        "base_url": None,
        "api_key_attr": "azure_openai_api_key",
        "default_model": "openai/gpt-oss-120b",
    },
}


def provider_base_url(name: str, settings: Settings) -> str | None:
    """Resolve the effective base URL for ``name`` given ``settings``."""
    if name == "azure":
        return settings.azure_openai_endpoint
    spec = PROVIDERS.get(name)
    return spec["base_url"] if spec else None


def _has_key(name: str, settings: Settings) -> bool:
    """Return True if the API key for provider ``name`` is present."""
    spec = PROVIDERS.get(name)
    if not spec:
        return False
    return bool(getattr(settings, spec["api_key_attr"], None))


def provider_order(settings: Settings) -> list[str]:
    """Return providers to try, primary first, then any others with keys.

    The primary provider (``settings.llm_provider_primary``) always comes first
    when known. Remaining providers are included only if they have a
    configured API key.
    """
    order: list[str] = []
    primary = settings.llm_provider_primary
    if primary in PROVIDERS:
        order.append(primary)
    for name in PROVIDERS:
        if name == primary:
            continue
        if _has_key(name, settings):
            order.append(name)
    return order
