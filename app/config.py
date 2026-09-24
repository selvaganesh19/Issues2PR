"""Application settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for Issue2PR.

    All fields are populated from environment variables (case-insensitive) or a
    local ``.env`` file. Secrets (API keys, webhook secret) live only here and
    are never hardcoded or logged.
    """

    # GitHub App (optional; only needed for the live webhook/PR flow).
    github_app_id: int | None = None
    github_private_key_path: str | None = None
    github_webhook_secret: str | None = None

    # LLM providers.
    llm_provider_primary: str = "groq"
    groq_api_key: str | None = None
    openrouter_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    llm_model: str = "openai/gpt-oss-120b"
    # Per-provider model override. When set, this model id is used instead of
    # ``llm_model`` for that provider only. Use it when the fallback provider
    # should run a different (e.g. free) model than the primary. Must be a
    # model that supports tool/function calling -- the agent cannot work
    # without it. OpenRouter free models use the ``vendor/model:free`` slug.
    openrouter_model: str = ""

    # Infrastructure (only needed for server/worker mode).
    #
    # redis_url: a normal ``redis://`` / ``rediss://`` URL for a real server
    # (local, Upstash, Redis Cloud, ...). Use the special scheme ``fake://`` to
    # run a pure-Python in-process Redis via ``fakeredis`` -- this exercises the
    # real RedisQueue code path locally with NO Docker and NO server.
    redis_url: str = "redis://localhost:6379/0"

    # database_url: an async SQLAlchemy URL. For managed Postgres use the
    # asyncpg driver, e.g.
    #   postgresql+asyncpg://<user>:<pw>@<host>:6543/postgres
    # (a connection-pooler URI with the driver swapped to +asyncpg).
    database_url: str = "postgresql+asyncpg://issue2pr:issue2pr@localhost:5432/issue2pr"

    # Managed-Postgres connection tuning.
    #   database_ssl: require TLS. Enable for any hosted DB.
    #   database_pgbouncer: set True when connecting through a transaction-mode
    #     pooler (typically port 6543). It disables asyncpg prepared-statement
    #     caching, which pgbouncer transaction pooling does not support.
    database_ssl: bool = False
    database_pgbouncer: bool = False
    #   database_ssl_root_cert: path to the CA cert. When set,
    #     the TLS chain is fully verified; when empty, the connection is encrypted
    #     but the cert chain is not verified.
    database_ssl_root_cert: str = ""

    # Agent budgets.
    agent_max_steps: int = 30
    agent_max_test_retries: int = 3
    agent_max_cost_usd: float = 0.50
    agent_wall_clock_s: int = 600

    # Policy / sandbox.
    allowed_repos: str = "*"
    sandbox_backend: str = "local"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
