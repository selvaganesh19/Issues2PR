"""Sandbox command runners.

:class:`LocalRunner` executes commands directly on the host (default, used in
this environment). :class:`DockerRunner` runs them inside a locked-down
container (no network, read-only root, non-root user, cpu/mem caps) and is used
when ``settings.sandbox_backend == "docker"``.
"""

from __future__ import annotations

import pathlib
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import Settings


@dataclass
class RunOutput:
    """Result of running a command in the sandbox."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool


class LocalRunner:
    """Run commands as subprocesses on the host, rooted at ``workspace``."""

    def __init__(self, workspace: pathlib.Path, timeout_s: int = 120) -> None:
        self.workspace = pathlib.Path(workspace)
        self.timeout_s = timeout_s

    def run(self, cmd: list[str]) -> RunOutput:
        """Execute ``cmd`` in the workspace, capturing output with a timeout."""
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            return RunOutput(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunOutput(
                returncode=124,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(
                    exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                )
                + f"\n[timed out after {self.timeout_s}s]",
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return RunOutput(
                returncode=127,
                stdout="",
                stderr=f"command not found: {cmd[0] if cmd else ''} ({exc})",
                timed_out=False,
            )


class DockerRunner:
    """Run commands inside a hardened Docker container.

    Security hardening applied to every invocation:
      * ``--network none``      — no outbound network access.
      * ``--read-only``         — read-only root filesystem.
      * workspace mounted rw    — only the workspace is writable.
      * ``--user``              — runs as a non-root uid.
      * ``--cpus`` / ``--memory`` — resource caps.
      * ``--pids-limit``        — fork-bomb protection.
    """

    def __init__(
        self,
        workspace: pathlib.Path,
        timeout_s: int = 120,
        image: str = "python:3.11-slim",
        cpus: str = "2",
        memory: str = "2g",
        user: str = "1000:1000",
    ) -> None:
        self.workspace = pathlib.Path(workspace)
        self.timeout_s = timeout_s
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.user = user

    def _docker_cmd(self, cmd: list[str]) -> list[str]:
        """Build the full ``docker run`` argument vector for ``cmd``."""
        container_workspace = "/workspace"
        return [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=256m",
            "--user",
            self.user,
            "--cpus",
            self.cpus,
            "--memory",
            self.memory,
            "--pids-limit",
            "256",
            "-v",
            f"{self.workspace}:{container_workspace}:rw",
            "-w",
            container_workspace,
            self.image,
            *cmd,
        ]

    def run(self, cmd: list[str]) -> RunOutput:
        """Execute ``cmd`` inside a fresh hardened container."""
        docker_cmd = self._docker_cmd(cmd)
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
            return RunOutput(
                returncode=proc.returncode,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                timed_out=False,
            )
        except subprocess.TimeoutExpired as exc:
            return RunOutput(
                returncode=124,
                stdout=exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=(
                    exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                )
                + f"\n[timed out after {self.timeout_s}s]",
                timed_out=True,
            )
        except FileNotFoundError as exc:
            return RunOutput(
                returncode=127,
                stdout="",
                stderr=f"docker not available: {exc}",
                timed_out=False,
            )


def get_runner(settings: "Settings", workspace: pathlib.Path):
    """Return the runner selected by ``settings.sandbox_backend``.

    Defaults to :class:`LocalRunner`. Returns :class:`DockerRunner` when the
    backend is ``"docker"``.
    """
    timeout_s = min(120, settings.agent_wall_clock_s)
    if settings.sandbox_backend == "docker":
        return DockerRunner(pathlib.Path(workspace), timeout_s=timeout_s)
    return LocalRunner(pathlib.Path(workspace), timeout_s=timeout_s)
