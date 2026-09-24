"""OpenAI-compatible LLM client with multi-provider fallback.

Speaks the OpenAI Chat Completions API against whichever provider is
configured (Groq by default, then OpenRouter / Azure). Providers are tried in
:func:`app.llm.providers.provider_order` sequence; on any error the next
provider is attempted, and :class:`LLMError` is raised only when all fail.

API keys are read from :class:`app.config.Settings` and are never logged or
included in exception messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from app.llm.providers import PROVIDERS, provider_base_url, provider_order

if TYPE_CHECKING:
    from app.config import Settings


class LLMError(RuntimeError):
    """Raised when every configured provider fails to produce a completion."""


@dataclass
class ToolCall:
    """A single tool/function call requested by the model."""

    id: str
    name: str
    arguments: dict


@dataclass
class Usage:
    """Token usage reported by the provider for one completion."""

    prompt_tokens: int
    completion_tokens: int


@dataclass
class ChatResult:
    """Normalized result of a chat completion.

    Attributes:
        content: Assistant text content, or ``None`` when only tool calls were
            returned.
        tool_calls: Parsed tool calls (arguments decoded from JSON to dict).
        usage: Token usage for the completion.
        raw: The raw provider response object, for debugging.
    """

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(0, 0))
    raw: object = None


def _parse_arguments(raw_args: Any) -> dict:
    """Decode a tool call's ``arguments`` (a JSON string) into a dict.

    Returns an empty dict for missing/blank arguments and tolerates
    non-JSON payloads by wrapping them under a ``"_raw"`` key.
    """
    if raw_args is None or raw_args == "":
        return {}
    if isinstance(raw_args, dict):
        return raw_args
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        return {"_raw": raw_args}
    if isinstance(parsed, dict):
        return parsed
    return {"_raw": parsed}


def _parse_response(response: Any) -> ChatResult:
    """Convert a raw OpenAI-style response into a :class:`ChatResult`."""
    choice = response.choices[0]
    message = choice.message

    content = getattr(message, "content", None)

    tool_calls: list[ToolCall] = []
    for tc in getattr(message, "tool_calls", None) or []:
        func = tc.function
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=func.name,
                arguments=_parse_arguments(func.arguments),
            )
        )

    raw_usage = getattr(response, "usage", None)
    usage = Usage(
        prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
    )

    return ChatResult(
        content=content,
        tool_calls=tool_calls,
        usage=usage,
        raw=response,
    )


class LLMClient:
    """Chat client that fails over across configured providers."""

    def __init__(self, settings: "Settings") -> None:
        """Store settings; connections are built lazily per ``chat`` call."""
        self.settings = settings

    def _api_key(self, provider: str) -> str | None:
        """Return the API key configured for ``provider`` (never logged)."""
        spec = PROVIDERS.get(provider)
        if not spec:
            return None
        return getattr(self.settings, spec["api_key_attr"], None)

    def _model_for(self, provider: str) -> str:
        """Resolve the model id for ``provider``.

        Precedence: an explicit per-provider override
        (``settings.<provider>_model``, e.g. ``openrouter_model``) wins so a
        fallback provider can run a different model than the primary; then the
        global ``settings.llm_model`` (applied to the primary provider only);
        then the provider's ``default_model``.
        """
        override = getattr(self.settings, f"{provider}_model", None)
        if override:
            return override
        primary = getattr(self.settings, "llm_provider_primary", None)
        model = getattr(self.settings, "llm_model", None)
        if model and provider == primary:
            return model
        spec = PROVIDERS.get(provider, {})
        return spec.get("default_model", "openai/gpt-oss-120b")

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResult:
        """Send a chat completion request, failing over across providers.

        Args:
            messages: OpenAI-format chat messages.
            tools: Optional OpenAI function-tool schemas.
            tool_choice: Tool selection strategy (ignored when ``tools`` is
                falsy, in which case no tool params are sent).

        Returns:
            A normalized :class:`ChatResult`.

        Raises:
            LLMError: If every provider in the order fails (or none are
                configured). Provider error details are chained but API keys
                are never included.
        """
        order = provider_order(self.settings)
        if not order:
            raise LLMError("no LLM providers configured")

        last_error: Exception | None = None
        for provider in order:
            api_key = self._api_key(provider)
            if not api_key:
                continue

            base_url = provider_base_url(provider, self.settings)
            if not base_url:
                continue

            try:
                client = OpenAI(base_url=base_url, api_key=api_key)
                kwargs: dict[str, Any] = {
                    "model": self._model_for(provider),
                    "messages": messages,
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                # Some models (notably Groq's gpt-oss family) occasionally emit a
                # malformed tool call that the provider rejects with a 400
                # "tool_use_failed". This is stochastic — the same request usually
                # succeeds on resample. Rate limits (429) are transient too and
                # often carry a Retry-After hint. Retry both a few times on the
                # same provider before failing over to the next one.
                import re
                import time

                attempts = 0
                while True:
                    try:
                        response = client.chat.completions.create(**kwargs)
                        return _parse_response(response)
                    except Exception as exc:  # noqa: BLE001
                        attempts += 1
                        msg = str(exc)
                        is_tool = "tool_use_failed" in msg
                        is_429 = "429" in msg or "rate limit" in msg.lower()
                        if attempts <= 4 and (is_tool or is_429):
                            if is_429:
                                m = re.search(r"try again in ([\d.]+)s", msg)
                                delay = float(m.group(1)) if m else min(2**attempts, 30)
                                time.sleep(min(delay + 0.5, 45))
                            continue
                        raise
            except Exception as exc:  # noqa: BLE001 - try next provider
                last_error = exc
                continue

        raise LLMError(
            f"all providers failed ({', '.join(order)}): "
            f"{type(last_error).__name__ if last_error else 'no usable provider'}"
            f"{(' - ' + str(last_error)[:200]) if last_error else ''}"
        ) from last_error
