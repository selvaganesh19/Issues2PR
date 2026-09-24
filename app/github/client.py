"""Installation-scoped GitHub REST client for Issue2PR.

``GitHubClient`` wraps the small subset of the GitHub REST API the agent needs
to turn a fix into a pull request:

- read a repository (default branch, sha of a ref),
- create a new branch ref (restricted to the ``agent/*`` namespace),
- open a pull request,
- comment on the originating issue.

Security model (mirrors README "Security notes"):
- The client only ever holds a short-lived **installation token** (see
  ``app.github.auth``); the App private key never reaches this layer.
- Branch creation is **hard-restricted to ``agent/*``** so the agent can never
  push to protected/default branches.
- There is deliberately **no merge and no force-update**. Refs are only ever
  *created*; existing refs are never overwritten. A human reviews and merges.
- The token value is never logged or printed.
"""

from __future__ import annotations

from typing import Any

import httpx

from app.github.auth import _ACCEPT, _API_VERSION, GITHUB_API

_AGENT_BRANCH_PREFIX = "agent/"


class GitHubClientError(RuntimeError):
    """Raised when a GitHub API call fails or a policy rule is violated."""


class GitHubClient:
    """Thin REST client authenticated with an installation access token."""

    def __init__(self, token: str, *, client: httpx.Client | None = None) -> None:
        """Initialise the client.

        Args:
            token: A GitHub installation access token.
            client: Optional ``httpx.Client`` for injection in tests. If omitted,
                a client with sane defaults is created and owned by this instance.
        """
        self._token = token
        self._owns_client = client is None
        self._http = client or httpx.Client(base_url=GITHUB_API, timeout=30.0)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP client if this instance owns it."""
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        """Auth headers; the token is never logged."""
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": _ACCEPT,
            "X-GitHub-Api-Version": _API_VERSION,
        }

    def _request(self, method: str, path: str, *, ok: int, **kwargs: Any) -> dict:
        """Perform a request and return parsed JSON, raising on unexpected status.

        Args:
            method: HTTP verb.
            path: API path (relative to the API root).
            ok: The status code that indicates success.
            **kwargs: Passed through to httpx (e.g. ``json=``).

        Raises:
            GitHubClientError: On any non-``ok`` response.
        """
        url = path if path.startswith("http") else f"{GITHUB_API}{path}"
        resp = self._http.request(method, url, headers=self._headers(), **kwargs)
        if resp.status_code != ok:
            raise GitHubClientError(
                f"{method} {path} failed: HTTP {resp.status_code}: {resp.text[:300]}"
            )
        if resp.content:
            return resp.json()
        return {}

    # -- API surface -------------------------------------------------------

    def get_repo(self, owner: str, repo: str) -> dict:
        """Fetch repository metadata (includes ``default_branch``)."""
        return self._request("GET", f"/repos/{owner}/{repo}", ok=200)

    def get_ref_sha(self, owner: str, repo: str, ref: str) -> str:
        """Return the commit SHA a branch ref points at.

        Args:
            ref: A branch name (e.g. ``main``), not prefixed with ``refs/``.
        """
        data = self._request(
            "GET", f"/repos/{owner}/{repo}/git/ref/heads/{ref}", ok=200
        )
        return data["object"]["sha"]

    def create_branch(self, owner: str, repo: str, branch: str, sha: str) -> dict:
        """Create a new branch ref at ``sha``.

        Only branches under the ``agent/`` namespace are permitted, and the ref
        is *created* (never force-updated), so protected branches are safe.

        Args:
            branch: Branch name; MUST start with ``agent/``.
            sha: Base commit SHA the new branch points at.

        Raises:
            GitHubClientError: If ``branch`` is outside the ``agent/*`` namespace.
        """
        if not branch.startswith(_AGENT_BRANCH_PREFIX):
            raise GitHubClientError(
                f"refusing to create branch outside '{_AGENT_BRANCH_PREFIX}*': {branch!r}"
            )
        payload = {"ref": f"refs/heads/{branch}", "sha": sha}
        return self._request(
            "POST", f"/repos/{owner}/{repo}/git/refs", ok=201, json=payload
        )

    def create_pull_request(
        self,
        owner: str,
        repo: str,
        *,
        title: str,
        head: str,
        base: str,
        body: str | None = None,
        draft: bool = False,
    ) -> dict:
        """Open a pull request from ``head`` into ``base``.

        Args:
            head: Source branch (an ``agent/*`` branch created by this client).
            base: Target branch (e.g. the repo default branch).
            draft: Open as a draft PR (safer default for human review).
        """
        payload = {
            "title": title,
            "head": head,
            "base": base,
            "body": body or "",
            "draft": draft,
        }
        return self._request(
            "POST", f"/repos/{owner}/{repo}/pulls", ok=201, json=payload
        )

    def create_issue_comment(
        self, owner: str, repo: str, issue_number: int, body: str
    ) -> dict:
        """Post a comment on an issue (or PR conversation)."""
        payload = {"body": body}
        return self._request(
            "POST",
            f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            ok=201,
            json=payload,
        )
