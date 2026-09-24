"""Embed texts through an OpenAI-compatible embeddings endpoint.

The embedder reuses the same provider configuration as :mod:`app.llm.client`
(base URLs / API keys from :class:`app.config.Settings`) but targets the
``/embeddings`` route instead of chat completions. Embeddings are computed in
batches to bound request size, and the embedding model is configurable
independently of the chat model.

Note: not every chat provider exposes an embeddings endpoint. The embedding
provider, model, and credentials are therefore resolved independently and can
be overridden explicitly via the constructor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from openai import OpenAI

from app.llm.providers import PROVIDERS, provider_base_url, provider_order

if TYPE_CHECKING:
    from app.config import Settings

# Default embedding model. Overridable via the ``model`` constructor argument
# so deployments can point at whatever their embeddings provider supports.
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# Number of texts sent per embeddings request.
DEFAULT_BATCH_SIZE = 64


class EmbeddingError(RuntimeError):
    """Raised when no embedding provider is usable or a request fails."""


class Embedder:
    """Compute embeddings for text via an OpenAI-compatible endpoint.

    The first provider from :func:`app.llm.providers.provider_order` that has a
    configured API key and resolvable base URL is used. The chosen model
    defaults to :data:`DEFAULT_EMBED_MODEL` but can be overridden.
    """

    def __init__(
        self,
        settings: Settings,
        model: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize the embedder.

        Args:
            settings: Application settings holding provider credentials.
            model: Embedding model id. Defaults to :data:`DEFAULT_EMBED_MODEL`.
            batch_size: Maximum number of texts per embeddings request.
        """
        self.settings = settings
        self.model = model or DEFAULT_EMBED_MODEL
        self.batch_size = max(1, batch_size)
        self._client: OpenAI | None = None
        self._provider: str | None = None

    def _resolve_client(self) -> OpenAI:
        """Build (and cache) an OpenAI client for the first usable provider.

        Raises:
            EmbeddingError: If no provider has both an API key and a base URL.
        """
        if self._client is not None:
            return self._client

        for provider in provider_order(self.settings):
            spec = PROVIDERS.get(provider)
            if not spec:
                continue
            api_key = getattr(self.settings, spec["api_key_attr"], None)
            if not api_key:
                continue
            base_url = provider_base_url(provider, self.settings)
            if not base_url:
                continue
            self._client = OpenAI(base_url=base_url, api_key=api_key)
            self._provider = provider
            return self._client

        raise EmbeddingError("no embedding provider configured (missing API key/base URL)")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching requests.

        Args:
            texts: Texts to embed. An empty list yields an empty result.

        Returns:
            A list of embedding vectors (one per input text, in order).

        Raises:
            EmbeddingError: If no provider is usable or a request fails.
        """
        if not texts:
            return []

        client = self._resolve_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            try:
                response = client.embeddings.create(model=self.model, input=batch)
            except Exception as exc:  # noqa: BLE001 - normalize to EmbeddingError
                raise EmbeddingError(
                    f"embeddings request failed ({type(exc).__name__})"
                ) from exc
            # The API returns items with an ``index``; sort defensively.
            items = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in items)
        return vectors

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text and return its vector."""
        result = self.embed([text])
        if not result:
            raise EmbeddingError("embedding produced no vector")
        return result[0]
