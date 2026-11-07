"""In-container entry point: run one episode and export its summary.

Architecture map:
    1. Read settings from ``SHELLSMITH_*`` env vars set by the host launcher
       (``sandbox.docker_launcher``) and the private world file copied to
       ``/opt/shellsmith/world_file.md``.
    2. Call ``run_live_smoke`` (vat + GM, unchanged).
    3. Write ``smoke_summary.json`` under ``/logs/artifacts`` (exported by host).
       Durable in-flight state lives in ``/logs/in-progress`` for timeout recovery.

Main callers:
    ``docker_launcher.docker_command`` (``python -m shellsmith.sandbox.sandbox_rollout``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import tempfile


from shellsmith.episode.smoke import (
    _parser,
    run_live_smoke,
)
from shellsmith.episode.gm_process.shared import validate_gm_runtime
from shellsmith.episode.gm_process.codex import (
    CodexWorldConfig,
    CodexWorldProcess,
    SANDBOX_PERMISSION_PROFILE,
)


# Fixed in-container locations shared with the host-side sandbox launcher.
ARTIFACTS_ROOT = "/logs/artifacts"
SANDBOX_WORLD_FILE = "/opt/shellsmith/world_file.md"
SUMMARY_ARTIFACT = Path(ARTIFACTS_ROOT) / "smoke_summary.json"


def _optional_arg(argv: list[str], flag: str, env_name: str) -> None:
    value = os.environ.get(env_name, "").strip()
    if value:
        argv.extend([flag, value])


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"shellsmith sandbox requires {name} from the host launcher."
        )
    return value


def args_from_env() -> argparse.Namespace:
    """Build the smoke argparse namespace from sandbox environment variables.

    The Player endpoint is always explicit here; ``run_live_smoke`` defaults
    are never used for it.
    """

    # Screen the runtime before inspecting either role's credentials. Checking
    # the raw value rejects Claude/Pi even when required settings are absent.
    validate_gm_runtime(os.environ.get("SHELLSMITH_GM_RUNTIME", ""))
    parser = _parser()
    player_key_env = _required_env("SHELLSMITH_PLAYER_API_KEY_ENV")
    if not os.environ.get(player_key_env, "").strip():
        raise RuntimeError(
            f"shellsmith sandbox requires {player_key_env} "
            "(the Player endpoint credential)."
        )
    argv = [
        "--world-file",
        os.environ.get("SHELLSMITH_WORLD_FILE", SANDBOX_WORLD_FILE),
        "--output-dir",
        os.environ.get("SHELLSMITH_OUTPUT_DIR", f"{ARTIFACTS_ROOT}/episode"),
        "--player-base-url",
        _required_env("SHELLSMITH_PLAYER_BASE_URL"),
        "--player-model",
        _required_env("SHELLSMITH_PLAYER_MODEL"),
        "--player-api-key-env",
        player_key_env,
    ]
    if os.environ.get("SHELLSMITH_SKIP_PLAYER_MODELS_PREFLIGHT", "") == "1":
        argv.append("--skip-player-models-preflight")
    _optional_arg(argv, "--player-provider", "SHELLSMITH_PLAYER_PROVIDER")
    gm_runtime = _required_env("SHELLSMITH_GM_RUNTIME")
    argv.extend(["--gm-runtime", gm_runtime])
    if gm_runtime == "codex":
        # OAuth belongs to the trusted Codex client, not its model-executed
        # tools. Keep the same restricted filesystem profile as host runs.
        argv.extend(["--codex-sandbox", SANDBOX_PERMISSION_PROFILE])
    _optional_arg(argv, "--gm-model", "SHELLSMITH_GM_MODEL")
    _optional_arg(argv, "--gm-provider", "SHELLSMITH_GM_PROVIDER")
    _optional_arg(argv, "--gm-base-url", "SHELLSMITH_GM_BASE_URL")
    _optional_arg(argv, "--gm-api", "SHELLSMITH_GM_API")
    _optional_arg(argv, "--gm-context-window", "SHELLSMITH_GM_CONTEXT_WINDOW")
    _optional_arg(argv, "--gm-max-tokens", "SHELLSMITH_GM_MAX_TOKENS")
    _optional_arg(
        argv,
        "--gm-reasoning-effort",
        "SHELLSMITH_GM_REASONING_EFFORT",
    )
    _optional_arg(argv, "--episode-id", "SHELLSMITH_EPISODE_ID")
    _optional_arg(argv, "--player-renderer", "SHELLSMITH_PLAYER_RENDERER")
    _optional_arg(argv, "--player-context-window", "SHELLSMITH_PLAYER_CONTEXT_WINDOW")
    _optional_arg(argv, "--node-bin", "SHELLSMITH_NODE_BIN")
    _optional_arg(argv, "--pi-bin", "SHELLSMITH_PI_BIN")
    _optional_arg(argv, "--pi-sdk-module-path", "SHELLSMITH_PI_SDK_MODULE_PATH")
    _optional_arg(argv, "--codex-bin", "SHELLSMITH_CODEX_BIN")
    _optional_arg(argv, "--claude-bin", "SHELLSMITH_CLAUDE_BIN")
    _optional_arg(argv, "--timeout-seconds", "SHELLSMITH_TIMEOUT_SECONDS")
    _optional_arg(argv, "--gm-api-idle-timeout-seconds", "SHELLSMITH_GM_API_IDLE_TIMEOUT_SECONDS")
    _optional_arg(argv, "--max-world-steps", "SHELLSMITH_MAX_WORLD_STEPS")
    _optional_arg(
        argv,
        "--player-prompt-timeout-seconds",
        "SHELLSMITH_PLAYER_PROMPT_TIMEOUT_SECONDS",
    )
    args = parser.parse_args(argv)
    # Keep a durable, container-owned recovery copy even if SIGKILL prevents
    # run_live_smoke's normal error archive from executing.
    args.staging_parent = Path("/logs/in-progress")
    return args


def _require_codex_namespaces() -> None:
    """Fail before model execution if the inner sandbox cannot start.

    This is a compatibility screen, not a substitute for Codex's actual
    filesystem policy. Never relax Docker security or retry unconfined.
    """
    command = ["/usr/bin/unshare", "--user", "--map-root-user", "--mount", "--", "/bin/true"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Codex Docker execution requires working nested user namespaces; use the protected --host mode on this system.") from error
    if result.returncode:
        raise RuntimeError(
            "Codex Docker execution requires working nested user namespaces; "
            "the engine refused the sandbox preflight. Use protected --host mode "
            f"instead. Confinement will not be disabled. {result.stderr.strip()}"
        )


def _require_codex_filesystem(executable: str) -> None:
    """Exercise the actual pinned sandbox without a model or real secret reads.

    Namespace creation alone misses AppArmor mount denials and broken filesystem
    profiles. A failed launch is not evidence of successful credential denial:
    require allowed writes and denied outside reads/writes in the same process.
    """
    with tempfile.TemporaryDirectory(prefix="shellsmith-sandbox-probe-") as temporary:
        root = Path(temporary).resolve()
        world = root / "world"
        world.mkdir()
        private = root / "synthetic-auth.json"
        private.write_text("synthetic-probe-secret")
        (world / "outside-link").symlink_to(private)
        process = CodexWorldProcess(CodexWorldConfig(
            executable=Path(executable), model="unused", workspace_dir=world,
            log_dir=world / "logs", bootstrap_response_schema_path=world / "schema.json",
        ))
        command = [executable, *process._sandbox_args(), "sandbox", "-P", "game_master", "-C", str(world), "--",
                   "/bin/sh", "-c",
                   'printf allowed > permitted.txt || exit 1; '
                   'for target in "$1" outside-link "/proc/1/root$1"; do '
                   'if cat "$target" >/dev/null 2>&1; then exit 2; fi; done; '
                   'if (printf denied > "$2") 2>/dev/null; then exit 3; fi; '
                   'cat permitted.txt',
                   "probe", str(private), str(root / "forbidden-write")]
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError("Codex Docker filesystem preflight could not run; confinement will not be disabled. Use protected --host mode.") from error
        if (result.returncode or result.stdout != "allowed"
                or not (world / "permitted.txt").is_file()
                or (world / "permitted.txt").read_text() != "allowed"
                or (root / "forbidden-write").exists()):
            raise RuntimeError(
                "Codex Docker filesystem preflight failed. Nested namespaces and "
                "a compatible confined LSM policy are required (docker-default "
                "AppArmor denies mounts). Confinement will not be disabled; "
                f"use protected --host mode. {result.stderr.strip()}"
            )


def main() -> int:
    """Run one live shellsmith smoke and export the summary artifact."""

    args = args_from_env()
    if not args.world_file.is_file():
        raise RuntimeError(f"sandbox world file is missing: {args.world_file}")
    if args.gm_runtime == "codex":
        _require_codex_namespaces()
        _require_codex_filesystem(args.codex_bin)
    _output_dir, summary = asyncio.run(run_live_smoke(args))
    SUMMARY_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_ARTIFACT.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
