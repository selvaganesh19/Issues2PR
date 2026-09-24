"""GitHub App authentication.

Two-step flow (see https://docs.github.com/apps/creating-github-apps):

1. Sign a short-lived **App JWT** (RS256) with the App's private key. The JWT
   is issued by the App itself (``iss = app_id``) and lives at most 10 minutes.
2. Exchange that JWT for an **installation access token** scoped to a single
   installation. Installation tokens are short-lived (~1 hour) and are the
   credential the rest of the code uses to act on a repository.

Security model:
- The private key is read from ``settings.github_private_key_path`` (a file on
  disk, kept out of version control) and never logged or printed.
- We request narrowly-scoped installation tokens; the token value is returned to
  the caller but never emitted to logs.
- ``iat`` is backdated 60s to tolerate minor clock skew, per GitHub guidance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import jwt

from app.config import Settings, get_settings

GITHUB_API = "https://api.github.com"
# GitHub rejects App JWTs whose exp is more than 10 minutes out; use 9 to be safe.
_JWT_TTL_S = 9 * 60
_ACCEPT = "application/vnd.github+json"
_API_VERSION = "2022-11-28"


class GitHubAuthError(RuntimeError):
    """Raised when App configuration is missing or a token exchange fails."""


@dataclass
class InstallationToken:
    """A short-lived installation access token.

    Attributes:
        token: The bearer token used for installation-scoped API calls.
        expires_at: ISO-8601 expiry timestamp returned by GitHub.
    """

    token: str
    expires_at: str


def _read_private_key(settings: Settings) -> str:
    """Read the App private key (PEM) from the configured path.

    Raises:
        GitHubAuthError: If the path is unset or the file is missing.
    """
    if not settings.github_private_key_path:
        raise GitHubAuthError("github_private_key_path is not configured")
    path = Path(settings.github_private_key_path)
    if not path.is_file():
        raise GitHubAuthError(f"private key file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_app_jwt(settings: Settings | None = None) -> str:
    """Build a signed App JWT (RS256) for the configured GitHub App.

    Args:
        settings: Optional settings override; falls back to ``get_settings()``.

    Returns:
        The encoded JWT string.

    Raises:
        GitHubAuthError: If ``github_app_id`` or the private key is missing.
    """
    settings = settings or get_settings()
    if settings.github_app_id is None:
        raise GitHubAuthError("github_app_id is not configured")

    private_key = _read_private_key(settings)
    now = int(time.time())
    payload = {
        "iat": now - 60,  # backdate to tolerate clock skew
        "exp": now + _JWT_TTL_S,
        "iss": settings.github_app_id,
    }
    return jwt.encode(payload, private_key, algorithm="RS256")


def _auth_headers(bearer: str) -> dict[str, str]:
    """Standard GitHub API headers for a given bearer credential."""
    return {
        "Authorization": f"Bearer {bearer}",
        "Accept": _ACCEPT,
        "X-GitHub-Api-Version": _API_VERSION,
    }


def get_installation_token(
    installation_id: int,
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
) -> InstallationToken:
    """Exchange the App JWT for an installation access token.

    Args:
        installation_id: The target installation id (from the webhook payload).
        settings: Optional settings override.
        client: Optional ``httpx.Client`` for injection in tests.

    Returns:
        An :class:`InstallationToken`.

    Raises:
        GitHubAuthError: If the token exchange fails.
    """
    settings = settings or get_settings()
    app_jwt = build_app_jwt(settings)
    url = f"{GITHUB_API}/app/installations/{installation_id}/access_tokens"

    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        resp = client.post(url, headers=_auth_headers(app_jwt))
    finally:
        if owns_client:
            client.close()

    if resp.status_code != 201:
        # Do not include the JWT or key material in the error.
        raise GitHubAuthError(
            f"installation token exchange failed: HTTP {resp.status_code}"
        )
    data = resp.json()
    return InstallationToken(token=data["token"], expires_at=data.get("expires_at", ""))
