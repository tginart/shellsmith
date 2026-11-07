"""Run one Shellsmith episode inside the local Docker sandbox.

Architecture map
----------------
    host                                          container (one per episode)
    require_fresh_image(image)                     src/ + docker/ build identity
    stage_episode(run_dir, world_file, ...)
        run_dir/world_file.md    --docker cp-->    /opt/shellsmith/world_file.md
        private temporary home --docker cp-->     /tmp/shellsmith-codex
        run_dir/artifacts/     <--docker cp--      /logs/artifacts
    docker create -> copy inputs -> start --attach -> export artifacts -> rm --force
    timeout/failure -> stop -> export artifacts and in-flight state -> rm --force
    sandbox_rollout -> namespace preflight -> restricted Codex -> run_live_smoke
    run_episode_in_docker reads the exported smoke_summary.json

Invariants
----------
- World files never live in the image; exactly one read-only world file is
  copied per episode, outside the Player's vat workspace.
- Secrets never appear on the Docker command line: the launcher passes
  ``-e NAME`` and Docker copies the value from the launcher's environment.
- A stale or unlabelled image is refused. Explicitly selected images may
  carry build-override metadata, but must still match the current source tree.
- Credentials and writable outputs use container-owned storage, not bind
  mounts: Docker user-namespace mappings are preserved and exported artifacts
  belong to the invoking host account. CLI caches and OAuth are not archived.
- Codex always retains its own restricted filesystem profile. Engines without
  nested confinement fail closed; no privileges, broad capabilities or disabled
  LSMs are used. The reviewed seccomp extension permits only required operations.

Why this exists
---------------
The original monorepo launched this container through Verifiers/Prime. Shellsmith drops
both; local Docker is its only sandbox backend.

Main callers:
    ``shellsmith.cli`` (``shellsmith episode``) and, from M4, the trainer's
    rollout runner.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass, field
import math
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import threading
import subprocess
import tempfile
import uuid

from shellsmith.config import (
    DEFAULT_GM_API_KEY_ENV,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
    DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS,
)
from shellsmith.paths import PROJECT_ROOT
from shellsmith.sandbox.docker_client import DockerClient
from shellsmith.sandbox.docker_auth import CodexAuthSession, restore_codex_auth, stage_codex_auth

CONTAINER_WORLD_FILE = "/opt/shellsmith/world_file.md"
CONTAINER_CODEX_HOME = "/tmp/shellsmith-codex"
CONTAINER_ARTIFACTS = "/logs/artifacts"
IMAGE_LABEL = "shellsmith.image_identity"
IMAGE_INSPECT_TIMEOUT_SECONDS = 30.0
CODEX_AUTH_PATH_ENV = "SHELLSMITH_CODEX_AUTH_PATH"
# Reviewed pinned policy; see docker/seccomp-codex.provenance.json. No caller
# override or unconfined fallback is accepted if the policy changes or is absent.
SECCOMP_POLICY_SHA256 = "051cf2a409ab628ba7b3804ea988268f578016116f53a0b5fdbf00a92e373ee2"


@dataclass(frozen=True)
class DockerEpisodeConfig:
    """Everything one containerized episode needs besides the world file.

    ``player_base_url`` must be reachable from inside the container (for a
    host-side endpoint use ``http://host.docker.internal:<port>/v1``).
    ``smoke_env`` holds extra ``SHELLSMITH_*`` settings read by
    ``sandbox_rollout.args_from_env`` (GM model, effort, timeouts, ...).
    """

    image: str
    gm_runtime: str
    player_base_url: str
    player_model: str
    player_provider: str
    player_api_key_env: str
    player_api_key: str | None = field(default=None, repr=False)
    gm_api_key_env: str | None = DEFAULT_GM_API_KEY_ENV
    smoke_env: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float | None = None
    allow_image_overrides: bool = False


def episode_timeout_seconds(config: DockerEpisodeConfig) -> float:
    """Budget the Player plus bootstrap/judgement (three attempts each) and I/O.

    GM tool turns during play are already inside the Player trajectory deadline.
    Keep the outer deadline longer than all supported inner work, not a fixed 4h.
    """
    gm = float(config.smoke_env.get("SHELLSMITH_TIMEOUT_SECONDS", DEFAULT_GM_TURN_TIMEOUT_SECONDS))
    player = float(config.smoke_env.get("SHELLSMITH_PLAYER_PROMPT_TIMEOUT_SECONDS", DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS))
    if not math.isfinite(gm) or not math.isfinite(player) or not 0 < gm < player:
        raise ValueError("Finite deadlines must satisfy 0 < GM timeout < Player trajectory timeout.")
    minimum = player + 6 * gm + 300.0
    outer = minimum if config.timeout_seconds is None else config.timeout_seconds
    if not math.isfinite(outer) or outer < minimum:
        raise ValueError(f"Docker timeout must be at least {minimum:g}s to cover the inner episode deadlines.")
    return outer


def sanitize_codex_auth_bytes(raw: bytes) -> bytes:
    """Return the minimal ChatGPT OAuth document accepted by Codex.

    The local ``auth.json`` may hold an ``OPENAI_API_KEY`` or extra keys. The
    GM must have exactly one auth route, so only ChatGPT OAuth state is copied
    and any API-key material is dropped. Missing tokens raise.

    Main callers:
        ``stage_episode`` before copying ``codex_home`` into the container.
    """
    try:
        auth = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Codex auth file is not valid JSON") from error
    if not isinstance(auth, dict) or auth.get("auth_mode") != "chatgpt":
        raise ValueError("The Codex GM requires ChatGPT auth; run `codex login`.")
    tokens = auth.get("tokens")
    if not isinstance(tokens, dict) or not tokens.get("access_token"):
        raise ValueError("Codex ChatGPT access token is missing")
    if not tokens.get("refresh_token"):
        raise ValueError("Codex ChatGPT refresh token is missing")
    minimal: dict[str, object] = {"auth_mode": "chatgpt", "tokens": tokens}
    if auth.get("last_refresh") is not None:
        minimal["last_refresh"] = auth["last_refresh"]
    return json.dumps(minimal, separators=(",", ":")).encode()


def current_src_tree() -> str:
    """Git tree hash of ``src/`` at HEAD; the identity a fresh image must carry."""
    return subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD:src"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def current_image_identity() -> str:
    """Identity of the committed application and complete Docker build recipe."""
    trees = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD:src", "HEAD:docker"],
        check=True, capture_output=True, text=True,
    ).stdout
    return hashlib.sha256(trees.encode()).hexdigest()


def default_image() -> str:
    """The tag ``docker/build.sh`` assigns to the current source and recipe."""
    return f"shellsmith:{current_image_identity()[:12]}"


def require_fresh_image(image: str, *, allow_overrides: bool = False, client: DockerClient | None = None) -> None:
    """Fail unless the image matches the committed source and Docker build recipe.

    Steps:
        Read the image's ``shellsmith.image_identity`` label with ``docker image inspect``.
        Compare it with both committed trees; also refuse uncommitted
        ``src/`` or ``docker/`` changes, since the container would run different code than
        the checkout shows.

    Main callers:
        ``run_episode_in_docker`` before staging anything.
    """
    client = client or DockerClient.resolve()
    try:
        inspected = client.run(
            ["docker", "image", "inspect", "--format", f'{{{{index .Config.Labels "{IMAGE_LABEL}"}}}}', image],
            capture_output=True, text=True, timeout=IMAGE_INSPECT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Docker image inspection timed out after {IMAGE_INSPECT_TIMEOUT_SECONDS:g}s. "
            "Check or restart the selected local Docker daemon, then retry; "
            "build the current image with docker/build.sh if it is missing."
        ) from None
    if inspected.returncode != 0:
        raise RuntimeError(
            f"Docker image {image} is not available ({inspected.stderr.strip()}). "
            "Build it with docker/build.sh."
        )
    label = inspected.stdout.strip()
    expected = current_image_identity()
    override_label = re.fullmatch(
        re.escape(expected) + r"\+pi=[^\s=]+\+codex=[^\s=]+\+platform=[^\s=]+", label
    )
    if label != expected and not (allow_overrides and override_label):
        raise RuntimeError(
            f"Docker image {image} was built from image identity {label or '<unlabelled>'}, "
            f"but the checkout is at {expected}. Rebuild with docker/build.sh."
        )
    dirty = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--", "src", "docker"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(
            "src/ or docker/ has uncommitted changes, so the image would not match the "
            "checkout. Commit them and rebuild with docker/build.sh."
        )


def _stage_seccomp_policy(directory: Path) -> Path:
    """Read a reviewed regular policy without following links; freeze for Docker."""
    source = PROJECT_ROOT / "docker/seccomp-codex.json"
    if source.is_symlink():
        raise RuntimeError("The Docker seccomp policy must not be a symlink")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise RuntimeError("The Docker seccomp policy must be a regular file")
        policy = stream.read()
    if hashlib.sha256(policy).hexdigest() != SECCOMP_POLICY_SHA256:
        raise RuntimeError("The Docker seccomp policy differs from the reviewed policy; refusing to launch")
    destination = directory / "seccomp.json"
    destination.write_bytes(policy)
    destination.chmod(0o444)
    return destination


@contextmanager
def stage_episode(run_dir: Path, world_file: Path, *, gm_runtime: str, auth_session: CodexAuthSession | None = None) -> Iterator[dict[str, Path]]:
    """Stage inputs, keeping OAuth credentials outside all retained run output.

    The temporary Codex home is removed on success, timeout, cancellation,
    command validation failure, and even a partially failed staging operation.
    """
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Episode run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    run_dir.chmod(0o700)
    staged = {"world_file": run_dir / "world_file.md", "artifacts": run_dir / "artifacts"}
    shutil.copyfile(world_file, staged["world_file"])
    # Copy read-only input into container-owned storage; no host bind label or
    # daemon UID mapping assumptions are needed, including on SELinux hosts.
    staged["world_file"].chmod(0o444)
    staged["artifacts"].mkdir()
    with ExitStack() as cleanup:
        policy_dir = Path(cleanup.enter_context(tempfile.TemporaryDirectory(prefix="shellsmith-seccomp-")))
        staged["seccomp_profile"] = _stage_seccomp_policy(policy_dir)
        if gm_runtime == "codex":
            auth = auth_session or cleanup.enter_context(stage_codex_auth())
            staged["codex_home"] = auth.codex_home
        yield staged


def container_environment(config: DockerEpisodeConfig) -> dict[str, str]:
    """Non-secret ``SHELLSMITH_*`` settings for ``sandbox_rollout.args_from_env``."""
    env = {
        # Keep CLI caches in the disposable container, never among the
        # durable artifacts exported to the host.
        "HOME": "/tmp/shellsmith-home",
        "SHELLSMITH_GM_RUNTIME": config.gm_runtime,
        "SHELLSMITH_PLAYER_BASE_URL": config.player_base_url,
        "SHELLSMITH_PLAYER_MODEL": config.player_model,
        "SHELLSMITH_PLAYER_PROVIDER": config.player_provider,
        "SHELLSMITH_PLAYER_API_KEY_ENV": config.player_api_key_env,
        **config.smoke_env,
    }
    if config.gm_runtime == "codex":
        env["CODEX_HOME"] = CONTAINER_CODEX_HOME
    if config.gm_api_key_env:
        env["SHELLSMITH_GM_API_KEY_ENV"] = config.gm_api_key_env
    return env


def secret_names(config: DockerEpisodeConfig) -> list[str]:
    """Credential name forwarded by Docker, from a private value or host env."""
    if config.gm_runtime != "codex":
        raise ValueError("Docker supports only the confined Codex GM; Pi GM execution is disabled because its tools expose credentials.")
    credential = config.player_api_key if config.player_api_key is not None else os.environ.get(config.player_api_key_env, "")
    if not credential.strip():
        raise RuntimeError(f"Required credentials are not set on the host: {config.player_api_key_env}")
    return [config.player_api_key_env]


def _container_name(run_dir: Path) -> str:
    readable = re.sub(r"[^a-zA-Z0-9_.-]", "-", run_dir.name)[:32]
    return f"shellsmith-{readable}-{uuid.uuid4().hex[:12]}"


def docker_command(config: DockerEpisodeConfig, staged: dict[str, Path], *, name: str) -> list[str]:
    """Create a container without exposing writable host mounts or secret values.

    Docker owns the credential/output files inside its own UID namespace.
    Artifacts are exported with docker cp (without -a), which assigns ownership
    to the invoking host user in native, rootless, and remapped engines alike.
    """
    command = [
        "docker", "create", "--name", name, "--user", "0:0",
        "--security-opt", f"seccomp={staged['seccomp_profile']}",
        "--security-opt", "no-new-privileges=true",
        "--add-host", "host.docker.internal:host-gateway",
    ]
    for key, value in sorted(container_environment(config).items()):
        command += ["-e", f"{key}={value}"]
    for secret in secret_names(config):
        command += ["-e", secret]
    command += [config.image, "python", "-m", "shellsmith.sandbox.sandbox_rollout"]
    return command


def _export_artifacts(client: DockerClient, name: str, staged: dict[str, Path], log, *, partial: bool) -> None:
    """Export only stopped-container output, never its credentials or caches."""
    exported = client.run(
        ["docker", "cp", f"{name}:{CONTAINER_ARTIFACTS}/.", str(staged["artifacts"])],
        stdout=log, stderr=subprocess.STDOUT, timeout=60,
    )
    errors = []
    if exported.returncode:
        errors.append(f"Could not export episode artifacts ({exported.returncode})")
    if partial:
        from shellsmith.episode.smoke import _copy_rollout_without_symlinks

        # A force-stop may bypass Python's error archive. Recover its durable
        # staging tree separately, skipping symlinks, special files and scratch.
        with tempfile.TemporaryDirectory(prefix="shellsmith-recovery-") as temporary:
            copied = client.run(
                ["docker", "cp", f"{name}:/logs/in-progress/.", temporary],
                stdout=log, stderr=subprocess.STDOUT, timeout=60,
            )
            if copied.returncode:
                errors.append(f"Could not recover in-flight artifacts ({copied.returncode})")
            else:
                for episode in Path(temporary).iterdir():
                    if episode.is_dir() and not episode.is_symlink():
                        destination = staged["artifacts"] / "partial" / episode.name
                        destination.parent.mkdir(exist_ok=True)
                        _copy_rollout_without_symlinks(episode, destination)
    if errors:
        raise RuntimeError("; ".join(errors))


def _finalize_container(
    client: DockerClient, name: str, auth: CodexAuthSession, log_path: Path,
    started: bool, stopped: bool, failure: BaseException | None,
) -> None:
    """Defer termination during bounded auth reconciliation/removal, then deliver it."""
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    try:
        _reconcile_and_remove(client, name, auth, log_path, started, stopped, failure)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def _reconcile_and_remove(
    client: DockerClient, name: str, auth: CodexAuthSession, log_path: Path,
    started: bool, stopped: bool, failure: BaseException | None,
) -> None:
    """Reconcile rotated OAuth while its source lock is held, then remove."""
    problems: list[BaseException] = []
    try:
        if started:
            if not stopped:
                raise RuntimeError("Container did not stop; cannot safely recover refreshed OAuth")
            with log_path.open("ab") as log:
                returned = client.run(
                    ["docker", "cp", f"{name}:{CONTAINER_CODEX_HOME}/auth.json", str(auth.returned_auth_path)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=60,
                )
            if returned.returncode:
                raise RuntimeError("Could not retrieve refreshed Codex OAuth; the host login may require recovery")
            restore_codex_auth(auth)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
        problems.append(error)
    finally:
        try:
            cleanup = client.run(
                ["docker", "rm", "--force", name], capture_output=True, text=True, timeout=30,
            )
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                raise RuntimeError(f"Could not force-remove container {name}: {cleanup.stderr.strip()}")
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            problems.append(error)
    if problems:
        primary = failure or problems.pop(0)
        for problem in problems:
            primary.add_note(str(problem))
        if failure is None:
            raise primary
        # Integer SystemExit suppresses traceback/notes, unlike other failures.
        if isinstance(failure, SystemExit):
            for note in getattr(primary, "__notes__", ()):
                print(f"shellsmith: cleanup failed: {note}", file=sys.stderr)


@contextmanager
def _sigterm_cleanup() -> Iterator[None]:
    """Turn supervisor termination into cleanup; ignore repeats while unwinding.

    Python cannot install signal handlers in worker threads. Refuse that route
    before staging credentials; use a main-thread worker process for Docker runs.
    SIGKILL cannot be intercepted and remains outside this cleanup guarantee.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Docker episodes require a main-thread worker process for SIGTERM cleanup")
    previous = signal.getsignal(signal.SIGTERM)

    def terminate(signum, _frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_episode_in_docker(world_file: Path, run_dir: Path, config: DockerEpisodeConfig) -> dict:
    """Run one episode with supervisor termination routed through cleanup."""
    with _sigterm_cleanup():
        return _run_episode_in_docker(world_file, run_dir, config)


def _run_episode_in_docker(world_file: Path, run_dir: Path, config: DockerEpisodeConfig) -> dict:
    """Run one episode; stop, export partial output, then remove on failure."""
    if config.gm_runtime != "codex":
        raise ValueError("The bundled Docker image supports only the confined Codex GM. Pi GM execution is disabled; Claude GM execution is also disabled.")
    timeout = episode_timeout_seconds(config)
    # Resolve only local context metadata, then pin every daemon operation to
    # that endpoint. Never stage OAuth before rejecting a remote context/host.
    client = DockerClient.resolve()
    require_fresh_image(config.image, allow_overrides=config.allow_image_overrides, client=client)
    run_dir = run_dir.expanduser().resolve()
    with stage_codex_auth() as auth, stage_episode(
        run_dir, world_file.expanduser().resolve(), gm_runtime=config.gm_runtime, auth_session=auth,
    ) as staged:
        name = _container_name(run_dir)
        command = docker_command(config, staged, name=name)
        log_path = run_dir / "container.log"
        failure: BaseException | None = None
        start_attempted = False
        container_stopped = False
        try:
            with log_path.open("wb") as log:
                preparations = [
                    command,
                    ["docker", "cp", str(staged["world_file"]), f"{name}:{CONTAINER_WORLD_FILE}"],
                    ["docker", "cp", str(staged["codex_home"]), f"{name}:{CONTAINER_CODEX_HOME}"],
                ]
                for prepare in preparations:
                    # A generated Player key belongs only to create's trusted
                    # Docker client, not this process or unrelated subprocesses.
                    extra = {}
                    if prepare is command and config.player_api_key is not None:
                        extra["env"] = {**os.environ, config.player_api_key_env: config.player_api_key}
                    result = client.run(prepare, stdout=log, stderr=subprocess.STDOUT, timeout=60, **extra)
                    if result.returncode:
                        raise RuntimeError(f"Docker episode preparation failed ({result.returncode}); see {log_path}")
                start_attempted = True
                result = client.run(
                    ["docker", "start", "--attach", name], stdout=log,
                    stderr=subprocess.STDOUT, timeout=timeout,
                )
                if result.returncode:
                    raise RuntimeError(f"Episode container exited {result.returncode}; see {log_path}")
                container_stopped = True
                _export_artifacts(client, name, staged, log, partial=False)
            summary_path = staged["artifacts"] / "smoke_summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"Episode container exited 0 but wrote no {summary_path}; see {log_path}")
            return json.loads(summary_path.read_text(encoding="utf-8"))
        except BaseException as error:
            failure = error
            if start_attempted:
                try:
                    with log_path.open("ab") as log:
                        # A timed-out client is not a stopped container. Stop
                        # execution before copying to avoid racing GM writes.
                        stopped = client.run(
                            ["docker", "stop", "--time", "30", name],
                            stdout=log, stderr=subprocess.STDOUT, timeout=45,
                        )
                        if stopped.returncode:
                            raise RuntimeError(f"Could not stop container {name} for artifact recovery")
                        container_stopped = True
                        _export_artifacts(client, name, staged, log, partial=True)
                except (OSError, subprocess.TimeoutExpired, RuntimeError) as export_error:
                    error.add_note(f"Artifact recovery failed: {export_error}; see {log_path}")
            raise
        finally:
            _finalize_container(client, name, auth, log_path, start_attempted, container_stopped, failure)
