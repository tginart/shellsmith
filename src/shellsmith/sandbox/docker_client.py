"""Resolve a local Docker endpoint once, before staging or forwarding secrets.

Context show/inspect read the trusted Docker CLI's local metadata; resolution
never contacts a daemon. The launcher must resolve one client before staging
credentials, then reuse it for image checks, create/copy/start, and cleanup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import os
import re
import subprocess
from typing import Any


_LOCAL_ONLY = (
    "Shellsmith requires a local Docker endpoint: an absolute unix socket or "
    "a local Windows named pipe. Remote Docker endpoints are unsupported."
)
_CONTEXT_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.+-]*")
_LOCAL_PIPE = re.compile(r"npipe:////\./pipe/[a-zA-Z0-9_][a-zA-Z0-9_.-]*")


def _local_host(host: str) -> str:
    """Accept only unambiguous filesystem sockets or the local pipe namespace.

    No existence/connectivity probe is needed: an absolute AF_UNIX address is
    local even for Docker Desktop/rootless proxies. Never expand a relative
    address, infer a default, or accept TCP/SSH (including loopback TCP).
    """
    if not isinstance(host, str) or not host or host != host.strip():
        raise RuntimeError(_LOCAL_ONLY)
    if any(ord(character) < 32 or ord(character) == 127 for character in host):
        raise RuntimeError(_LOCAL_ONLY)
    if host.startswith("unix://"):
        path = host[len("unix://"):]
        if (
            path.startswith("/") and not path.startswith("//") and path != "/"
            and not any(character in path for character in "\\?#")
            and "://" not in path
        ):
            return host
    if host.startswith("npipe://"):
        # Docker accepts slash and backslash pipe spellings. Normalize first so
        # mixed separators cannot hide a remote server or another device path.
        normalized = host.replace("\\", "/")
        if _LOCAL_PIPE.fullmatch(normalized) and not normalized.endswith("."):
            return normalized
    # Do not echo an endpoint: a rejected URI can contain embedded credentials.
    raise RuntimeError(_LOCAL_ONLY)


def _metadata(arguments: list[str], environment: dict[str, str]) -> str:
    """Run only a local metadata command, capturing potentially sensitive errors."""
    try:
        result = subprocess.run(
            ["docker", "context", *arguments], env=environment,
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        raise RuntimeError("Could not read local Docker context metadata.") from None
    if result.returncode:
        # CLI diagnostics may quote credentials from configuration/environment.
        raise RuntimeError("Could not read local Docker context metadata.")
    return result.stdout


@dataclass(frozen=True, slots=True)
class DockerClient:
    """A validated, immutable local endpoint, not a credential/environment cache.

    ``resolve()`` follows nonempty DOCKER_CONTEXT, then DOCKER_HOST, then the
    current context. ``run(["docker", ...], **kwargs)`` adds the pinned --host
    and otherwise preserves subprocess.run semantics and the caller's env.
    Environment credentials stay with the trusted CLI unless the authorized
    launcher explicitly forwards them (for example, create's ``-e NAME``).
    """

    host: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", _local_host(self.host))

    @classmethod
    def resolve(cls) -> DockerClient:
        """Resolve without daemon access; call before staging any secrets."""
        environment = dict(os.environ)
        context = environment.pop("DOCKER_CONTEXT", "")
        host = environment.pop("DOCKER_HOST", "")
        if not context and host:
            return cls(host)
        if not context:
            context = _metadata(["show"], environment).rstrip("\r\n")
        if not _CONTEXT_NAME.fullmatch(context):
            raise RuntimeError("Could not resolve a valid local Docker context name.")
        # Inspect an explicit name, not whichever context is current a moment
        # later. Clear both selectors even for the synthetic 'default' context.
        output = _metadata(["inspect", "--", context], environment)
        try:
            contexts = json.loads(output)
            if not isinstance(contexts, list) or len(contexts) != 1:
                raise ValueError
            details = contexts[0]
            if details["Name"] != context:
                raise ValueError
            host = details["Endpoints"]["docker"]["Host"]
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("Could not resolve a local Docker endpoint from context metadata.") from None
        return cls(host)

    def run(self, command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess:
        """Run an existing full Docker argv list using only the pinned endpoint.

        Transport overrides and shell/executable overrides are not supported.
        Copy the supplied env (or the current environment) on every invocation;
        preserve HOME/configuration and secrets without retaining or logging
        them. Authorization to forward secrets remains the launcher's job.
        """
        if (
            isinstance(command, (str, bytes)) or len(command) < 2
            or any(not isinstance(argument, str) for argument in command)
            or command[0] != "docker" or command[1].startswith("-")
        ):
            raise ValueError("Expected a full Docker command beginning with ['docker', subcommand].")
        # This deliberately accepts launcher argv lists, not arbitrary Docker
        # global options. Inherited flags after the subcommand must not repin it.
        if any(
            argument in {"--host", "--context"}
            or argument.startswith(("--host=", "--context=", "-H", "-c"))
            for argument in command[1:]
        ):
            raise ValueError("Docker transport overrides are unsupported for a pinned client.")
        if kwargs.get("shell") or kwargs.get("executable") is not None:
            raise ValueError("Pinned Docker commands must use the Docker CLI without a shell override.")
        supplied_environment = kwargs.pop("env", None)
        environment = dict(os.environ if supplied_environment is None else supplied_environment)
        for key in tuple(environment):
            # subprocess also accepts bytes keys on POSIX; Windows names are
            # case-insensitive even in an explicitly supplied environment.
            name = os.fsdecode(key)
            if os.name == "nt":
                name = name.upper()
            if name in {"DOCKER_CONTEXT", "DOCKER_HOST"}:
                del environment[key]
        return subprocess.run(["docker", "--host", self.host, *command[1:]], env=environment, **kwargs)
