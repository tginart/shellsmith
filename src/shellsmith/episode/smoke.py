"""Run one no-training shellsmith episode against live model services.

Architecture map:
    CLI preflight
      -> Player endpoint `/models` preflight
      -> temporary private episode workspace + temporary AGENTS-only vat
      -> persistent Codex GM thread
      -> persistent Pi Player process
      -> final world judgment
      -> copy the complete audit bundle to `--output-dir`

The live episode runs outside the repository so Codex sees only the universal
world-simulator `AGENTS.md`. The finished audit is copied into the requested
durable directory after both agents have stopped. Secrets stay in environment
variables and never enter protocol records or episode files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import stat
import hashlib
import tempfile
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from shellsmith.cli_args import GMArgumentParser
from shellsmith.config import (
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_MAX_WORLD_STEPS,
    DEFAULT_GM_API_KEY_ENV,
    DEFAULT_GM_MODEL,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_RUNTIME,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
    DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS,
    GM_PROVIDER_BY_RUNTIME,
    DEFAULT_PLAYER_API_KEY_ENV,
    DEFAULT_PLAYER_BASE_URL,
    DEFAULT_PLAYER_MODEL,
    DEFAULT_PLAYER_PROVIDER,
    DEFAULT_PLAYER_THINKING,
    PLAYER_ENDPOINT_KEY_ENV,
)
from shellsmith.paths import DEFAULT_LOG_ROOT, DEFAULT_WORLD
from shellsmith.world_file import (
    load_world_file,
)
from shellsmith.episode.failure import (
    AttributedExecutionError,
)
from shellsmith.episode.orchestrator import (
    EpisodePaths,
    EpisodeResult,
    EpisodeOrchestrator,
)
from shellsmith.episode.progress import (
    install_stdout_progress,
)
from shellsmith.episode.gm_process import (
    WorldHarnessConfig,
    WorldHarnessLauncher,
)
from shellsmith.episode.gm_process.shared import validate_gm_runtime
from shellsmith.episode.gm_process.codex import (
    CODEX_SANDBOX_MODES,
    SANDBOX_PERMISSION_PROFILE,
)
from shellsmith.episode.vat_process import (
    PiSdkVatConfig,
    PiSdkVatProcess,
)
from shellsmith.episode.vat_workspace import (
    EphemeralVatWorkspace,
)
from shellsmith.episode.world_prompt_profile import (
    resolve_world_prompt_profile,
)


# Episode package root (this directory): owns vat_runner/ next to smoke.py.
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PARENT = DEFAULT_LOG_ROOT / "episodes"


class UnsafeRolloutArchiveError(RuntimeError):
    """The model-writable episode tree cannot be copied by the trusted host."""


EPHEMERAL_ARCHIVE_EXCLUSIONS = frozenset({Path("world/runtime_scratch")})

# Progressive verification gates for the shellsmith harness fix (ops checklist).
# Run these in order before a fresh baby training recipe. Do not skip gates.
PROGRESSIVE_VERIFICATION_GATES: tuple[str, ...] = (
    "1. Captured-payload native renderer test: backend/tests/test_tinker_chat_adapter.py",
    "2. One-action no-train smoke: shellsmith.episode.smoke --no-train with manual wire inspect",
    "3. Thinking smoke: run_config + proxy logs + Pi event + world request show thinking",
    "4. Sibling valid + harness-invalid smokes: vat_runner npm test + Python vat tests",
    "5. Compaction smoke: compaction_generation trains once; replay masked",
    "6. Disposable CISPO update: inspect datums; discard checkpoint",
    "7. Fresh baby run from clean recipe state",
)


def progressive_verification_plan() -> str:
    """Return the ordered ops gate list for harness remediation sign-off."""

    return "\n".join(PROGRESSIVE_VERIFICATION_GATES)


def _parser() -> argparse.ArgumentParser:
    parser = GMArgumentParser(
        description="Run a live Player-in-a-vat shellsmith smoke episode."
    )
    parser.add_argument("--world-file", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--node-bin", default=shutil.which("node") or "node")
    parser.add_argument("--pi-bin", default=shutil.which("pi") or "pi")
    parser.add_argument("--pi-sdk-module-path", type=Path)
    parser.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    # Codex GM confinement. Both host and Docker runs retain the restricted
    # profile. Full access is an explicit unsafe experiment-only override.
    parser.add_argument(
        "--codex-sandbox",
        choices=CODEX_SANDBOX_MODES,
        default=SANDBOX_PERMISSION_PROFILE,
    )
    parser.add_argument("--claude-bin", default=shutil.which("claude") or "claude")
    parser.add_argument("--gm-pi-bin", default=shutil.which("pi") or "pi")
    parser.add_argument(
        "--gm-runtime", choices=tuple(GM_PROVIDER_BY_RUNTIME), default=DEFAULT_GM_RUNTIME
    )
    parser.add_argument("--gm-model", help="GM model (Codex defaults to Sol; required for other runtimes).")
    parser.add_argument("--gm-provider", help="Provider override (default follows --gm-runtime).")
    parser.add_argument("--gm-api-key-env", default=DEFAULT_GM_API_KEY_ENV)
    parser.add_argument("--gm-base-url")
    parser.add_argument("--gm-api", default="openai-completions")
    parser.add_argument("--gm-context-window", type=int, default=65_536)
    parser.add_argument("--gm-max-tokens", type=int, default=16_384)
    parser.add_argument("--gm-reasoning-effort", default=DEFAULT_GM_REASONING_EFFORT)
    parser.add_argument("--gm-api-idle-timeout-seconds", type=float, default=DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--player-base-url", default=DEFAULT_PLAYER_BASE_URL)
    parser.add_argument("--player-provider", default=DEFAULT_PLAYER_PROVIDER)
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL)
    parser.add_argument("--player-api-key-env", default=DEFAULT_PLAYER_API_KEY_ENV)
    parser.add_argument("--player-thinking", default=DEFAULT_PLAYER_THINKING)
    parser.add_argument(
        "--player-renderer",
        help="Tinker renderer the Player endpoint uses; the acceptance audit checks every request against it.",
    )
    parser.add_argument("--player-context-window", type=int, default=65_536)
    parser.add_argument("--player-max-tokens", type=int, default=32_768)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_GM_TURN_TIMEOUT_SECONDS,
        help="Deadline for each individual GM CLI turn.",
    )
    parser.add_argument(
        "--player-prompt-timeout-seconds",
        type=float,
        default=DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS,
        help="Deadline for the complete Player trajectory (default: 12h).",
    )
    parser.add_argument("--max-world-steps", type=int, default=DEFAULT_MAX_WORLD_STEPS)
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument(
        "--skip-player-models-preflight",
        action="store_true",
        help=(
            "Require the Player API-key env var, but skip GET /models. Use when "
            "the endpoint is a chat-completions intercept without a models catalog."
        ),
    )
    return parser


# The vat runner is written against the Pi SDK version pinned in
# vat_runner/package-lock.json; `npm ci` there installs exactly that version.
VAT_RUNNER = PACKAGE_ROOT / "vat_runner"
PINNED_PI_SDK = VAT_RUNNER.joinpath(
    "node_modules", "@earendil-works", "pi-coding-agent", "dist", "index.js"
)


def _resolve_pi_sdk(args: argparse.Namespace) -> Path:
    """Return the Pi SDK entry the vat runner imports.

    Uses ``--pi-sdk-module-path`` when given (the Docker image passes its
    pinned global install), otherwise the lockfile-pinned copy under
    ``vat_runner/node_modules``. It never guesses from the ``pi`` on PATH: a
    different Pi version has a different file layout and fails inside the
    runner after the GM has already booted.
    """
    if args.pi_sdk_module_path is not None:
        path = args.pi_sdk_module_path.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"--pi-sdk-module-path does not exist: {path}")
        return path
    if not PINNED_PI_SDK.is_file():
        raise RuntimeError(
            "The pinned Pi SDK is not installed. Run `npm ci --ignore-scripts` in "
            f"{VAT_RUNNER} or pass --pi-sdk-module-path."
        )
    return PINNED_PI_SDK


def _preflight_player(
    *,
    base_url: str,
    model: str,
    api_key_env: str,
    api_key: str | None = None,
    require_models_catalog: bool = True,
) -> None:
    """Prove the Player credential is present, and optionally that GET /models serves it.

    Main callers:
        ``run_live_smoke`` after argparse.

    Why this exists:
        A served-model check catches a wrong Player model before the GM boots.
        Chat-only endpoints (no /models catalog) skip it with
        --skip-player-models-preflight.
    """
    if api_key is None:
        api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Required Player credential is absent: {api_key_env}."
        )
    if not require_models_catalog:
        return
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        raise RuntimeError(
            f"Cannot reach the authenticated Player endpoint at {base_url}: {error}."
        ) from error
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise RuntimeError("Player /models returned an unexpected payload.")
    model_ids = {
        row.get("id") for row in payload["data"] if isinstance(row, dict)
    }
    if model not in model_ids:
        raise RuntimeError(
            f"Requested Player model {model!r} is not served; found {sorted(model_ids)}."
        )


def _copy_rollout_without_symlinks(source: Path, destination: Path) -> None:
    """Archive a stopped episode without following model-authored symlinks.

    Main callers:
        ``run_live_smoke`` on both success and failure.

    Why this exists:
        The world model may write inside ``simulation_root``. A normal
        ``shutil.copytree`` follows symlinks by default, which could copy a
        symlink target from the host into the durable rollout. Both agents have
        stopped before this function runs, so validation and copying cannot race
        with another episode writer.
    """
    if source.is_symlink():
        raise UnsafeRolloutArchiveError(
            f"Refusing to archive symlinked rollout root: {source}."
        )
    if not stat.S_ISDIR(source.lstat().st_mode):
        raise UnsafeRolloutArchiveError(
            f"Rollout root is not a regular directory: {source}."
        )
    # Unsafe entries (symlinks, device nodes, hard-linked files) are SKIPPED
    # and recorded, never copied and never fatal: refusing the whole rollout
    # threw away otherwise-valid episodes whenever a world checkout contained
    # an ordinary in-repo symlink. The host-safety property is preserved —
    # symlink targets are still never followed or copied.
    skipped_unsafe: list[dict[str, str]] = []
    for directory, directory_names, file_names in os.walk(source, followlinks=False):
        directory_path = Path(directory)
        relative_directory = directory_path.relative_to(source)
        directory_names[:] = [
            name
            for name in directory_names
            if relative_directory.joinpath(name) not in EPHEMERAL_ARCHIVE_EXCLUSIONS
        ]
        for name in [*directory_names, *file_names]:
            candidate: Path = directory_path.joinpath(name)
            mode = candidate.lstat().st_mode
            reason: str | None = None
            if stat.S_ISLNK(mode):
                reason = "symlink"
            elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                reason = "non-regular entry"
            elif stat.S_ISREG(mode) and candidate.lstat().st_nlink > 1:
                reason = "multiply linked file"
            if reason is not None:
                skipped_unsafe.append(
                    {
                        "path": str(relative_directory.joinpath(name)),
                        "reason": reason,
                    }
                )

    skipped_relative = {Path(entry["path"]) for entry in skipped_unsafe}

    def ignore_excluded(directory: str, names: list[str]) -> set[str]:
        relative_directory = Path(directory).relative_to(source)
        return {
            name
            for name in names
            if relative_directory.joinpath(name) in EPHEMERAL_ARCHIVE_EXCLUSIONS
            or relative_directory.joinpath(name) in skipped_relative
        }

    shutil.copytree(source, destination, ignore=ignore_excluded)
    if skipped_unsafe:
        archive_path: Path = destination.joinpath("archive_sanitization.json")
        archive_path.write_text(
            json.dumps(
                {
                    "schema": "shellsmith.archive_sanitization.v1",
                    "skipped_entries": skipped_unsafe,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def _require_trainable_bit(summary: dict[str, Any]) -> bool:
    """Return the summary's ``trainable`` flag, rejecting a broken contract.

    Shared by ``reward_from_smoke_summary`` and ``smoke_summary_is_trainable``
    so the trainability gate is validated identically on both the reward and the
    metric path.

    ``run_live_smoke`` writes ``trainable`` from ``judgement.trainable`` (a real
    bool), so anything other than the literal ``True``/``False`` — a missing
    field, or a truncated/version-skewed value such as ``0`` or ``"false"`` —
    means a malformed artifact, not an untrainable episode. Rejecting it here
    keeps a broken summary from silently scoring reward 0 on the training path.
    """

    if not isinstance(summary, dict):
        raise RuntimeError(f"smoke summary must be an object, got {type(summary).__name__}")
    if "trainable" not in summary:
        raise RuntimeError(
            f"smoke summary is missing trainable: {sorted(summary)}. "
            "Simulator integrity is not optional on this scoring path."
        )
    value = summary["trainable"]
    # A plain bool only: isinstance(1, bool) is False, so 0/1, "true", None are
    # all rejected rather than coerced into the gate.
    if not isinstance(value, bool):
        raise RuntimeError(f"smoke summary trainable must be a boolean, got {value!r}")
    return value


def reward_from_smoke_summary(summary: dict[str, Any]) -> float:
    """Return the policy score for a completed GM episode.

    ``trainable=false`` is a training-gate, not an orchestration failure: the
    pipes ran, the world just failed integrity criteria. That residual
    ``final_reward`` is not a Player result, so this returns 0.0 and
    leaves ``trainable`` as a separate metric.

    A malformed ``trainable`` OR missing/non-numeric ``final_reward`` is a hard
    error: the summary contract broke. ``run_live_smoke`` always emits both
    fields together (from the terminal judgment), so an absent or malformed
    value means a truncated or version-skewed artifact. We therefore validate
    the whole summary BEFORE the trainability gate — otherwise a broken payload
    such as ``{"trainable": false}`` would silently pass as a clean reward-0
    episode and hide the broken simulator contract.

    Main callers: ``ShellsmithTask.final_reward`` (the taskset's
    ``@vf.reward``).
    """

    trainable = _require_trainable_bit(summary)
    if "final_reward" not in summary:
        raise RuntimeError(f"smoke summary is missing final_reward: {sorted(summary)}")
    try:
        reward = float(summary["final_reward"])
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"smoke summary final_reward is not numeric: {summary['final_reward']!r}"
        ) from error
    if not trainable:
        return 0.0
    return reward


def smoke_summary_is_trainable(summary: dict[str, Any]) -> bool:
    """Return whether GM accepted this episode for training."""

    return _require_trainable_bit(summary)


def _summary(result: EpisodeResult, *, args: argparse.Namespace) -> dict[str, Any]:
    judgement = result.judgement
    if judgement is None:
        raise RuntimeError("Completed smoke episode has no terminal judgment.")
    return {
        "episode": result.state.to_jsonable(),
        "gm_model": args.gm_model,
        "gm_reasoning_effort": args.gm_reasoning_effort,
        "gm_api_idle_timeout_seconds": getattr(args, "gm_api_idle_timeout_seconds", DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS),
        "gm_base_url": args.gm_base_url,
        "gm_api": args.gm_api,
        "max_world_steps": args.max_world_steps,
        "player_endpoint": args.player_base_url,
        "player_model": args.player_model,
        "player_thinking": args.player_thinking,
        "player_context_window": args.player_context_window,
        "player_max_tokens": args.player_max_tokens,
        **{key: getattr(args, key) for key in ("player_reasoning_tokens", "player_response_tokens")
           if hasattr(args, key)},
        **({"player_compaction_reserve_tokens": args.player_compaction_reserve_tokens,
            "player_compaction_keep_recent_tokens": args.player_compaction_keep_recent_tokens}
           if hasattr(args, "player_compaction_reserve_tokens") else {}),
        "tool_turn_count": len(result.world_observations),
        "final_reward": judgement.final_reward,
        "completion_percentage": judgement.completion_percentage,
        "trainable": judgement.trainable,
        "integrity_issues": list(judgement.integrity_issues),
        "factual_consistency_passed": judgement.factual_consistency_passed,
        "factual_contradictions": list(judgement.factual_contradictions),
        "task_outcomes": judgement.task_outcomes,
        "final_commentary": judgement.commentary,
    }


def _run_config(args: argparse.Namespace, *, episode_id: str) -> dict[str, Any]:
    """Return the non-secret launch configuration preserved before boot."""
    prompt_inputs = getattr(args, "frozen_prompt_inputs", None)
    binding = prompt_inputs.for_model(runtime=args.gm_runtime, model=args.gm_model) if prompt_inputs else None
    prompt_profile = binding.profile if binding else resolve_world_prompt_profile(
        runtime=args.gm_runtime, model=args.gm_model
    )
    python_runtime = getattr(args, "gm_python_runtime", None)
    return {
        "episode_id": episode_id,
        "world_file": str(args.world_file.expanduser().resolve()),
        "gm_runtime": args.gm_runtime,
        "gm_provider": args.gm_provider,
        "gm_model": args.gm_model,
        "gm_reasoning_effort": args.gm_reasoning_effort,
        "gm_api_idle_timeout_seconds": getattr(args, "gm_api_idle_timeout_seconds", DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS),
        "world_prompt_profile": prompt_profile.name,
        **({"gm_prompt_sha256": binding.sha256} if binding else {}),
        "world_permission_profile": "game_master",
        "world_network_access": True,
        **({"gm_python_runtime": {
            "executable": str(python_runtime.executable), "prefix": str(python_runtime.prefix),
            "executable_sha256": hashlib.sha256(python_runtime.executable.read_bytes()).hexdigest(),
        }} if python_runtime is not None else {}),
        "player_endpoint": args.player_base_url,
        "player_provider": args.player_provider,
        "player_model": args.player_model,
        "player_thinking": args.player_thinking,
        "player_context_window": args.player_context_window,
        "player_max_tokens": args.player_max_tokens,
        **{key: getattr(args, key) for key in ("player_reasoning_tokens", "player_response_tokens")
           if hasattr(args, key)},
        **({"player_compaction_reserve_tokens": args.player_compaction_reserve_tokens,
            "player_compaction_keep_recent_tokens": args.player_compaction_keep_recent_tokens}
           if hasattr(args, "player_compaction_reserve_tokens") else {}),
        "max_world_steps": args.max_world_steps,
        "credential_environment_variable": args.player_api_key_env,
        "world_credential_environment_variable": args.gm_api_key_env,
        **({"player_renderer": args.player_renderer} if args.player_renderer else {}),
    }


def _create_staging_root(episode_id: str, *, parent: Path | None = None) -> Path:
    """Keep in-flight state recoverable when the container is forcibly stopped."""
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{episode_id}-world-", dir=parent)).resolve()


async def run_live_smoke(
    args: argparse.Namespace, *, player_api_key: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute one live episode and return its durable audit directory.

    Main callers:
        `main` and focused operator scripts.

    Failure behavior:
        Even a failed rollout is copied to the output directory before the
        exception is raised, so Codex/Pi logs and the durable FAILED cursor remain
        available for diagnosis.
    """
    # Programmatic campaign callers can mutate an already-parsed namespace.
    # Reject disabled GMs before Player preflight sends credentials anywhere.
    validate_gm_runtime(args.gm_runtime)
    if (player_api_key is not None or args.player_provider == "shellsmith-tinker"
            or args.player_api_key_env == PLAYER_ENDPOINT_KEY_ENV):
        if args.gm_runtime != "codex" or args.codex_sandbox != SANDBOX_PERMISSION_PROFILE:
            raise RuntimeError("Tinker Player credentials require the confined Codex GM; unconfined GMs are unsupported.")
    install_stdout_progress()
    player_prompt_timeout_seconds = float(
        getattr(args, "player_prompt_timeout_seconds", DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS)
    )
    if player_prompt_timeout_seconds <= args.timeout_seconds:
        raise RuntimeError(
            "player_prompt_timeout_seconds must exceed timeout_seconds so the "
            "outer vat deadline cannot mask an individual GM timeout."
        )
    episode_id = args.episode_id or datetime.now(UTC).strftime(
        "shellsmith-%Y%m%dT%H%M%SZ"
    )
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULT_OUTPUT_PARENT.joinpath(episode_id)
    ).expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"Smoke output directory already exists: {output_dir}.")
    try:
        _preflight_player(
            base_url=args.player_base_url,
            model=args.player_model,
            api_key_env=args.player_api_key_env,
            api_key=player_api_key,
            require_models_catalog=not getattr(
                args, "skip_player_models_preflight", False
            ),
        )
        pi_sdk_path = _resolve_pi_sdk(args)
    except Exception as error:
        raise AttributedExecutionError(
            f"Player preflight failed: {type(error).__name__}: {error}",
            component="player",
            stage="player_preflight",
            provider_attributable=False,
        ) from error
    staging_root = _create_staging_root(
        episode_id, parent=getattr(args, "staging_parent", None)
    )
    vat = EphemeralVatWorkspace.create()
    paths = EpisodePaths.from_root(staging_root)
    spec = load_world_file(args.world_file.expanduser().resolve())
    prompt_inputs = getattr(args, "frozen_prompt_inputs", None)
    world_prompt_profile = (prompt_inputs.for_model(runtime=args.gm_runtime, model=args.gm_model).profile
                            if prompt_inputs else resolve_world_prompt_profile(
                                runtime=args.gm_runtime, model=args.gm_model))
    world_executables = {
        "codex": args.codex_bin,
        "claude": args.claude_bin,
        "pi": args.gm_pi_bin,
    }
    world_executable = world_executables[args.gm_runtime]
    gm_agent_process = WorldHarnessLauncher.create(
        WorldHarnessConfig(
            runtime=args.gm_runtime,
            executable=Path(world_executable).expanduser().resolve(),
            model=args.gm_model,
            workspace_dir=staging_root,
            log_dir=paths.log_dir.joinpath(args.gm_runtime),
            bootstrap_response_schema_path=paths.schema_dir.joinpath(
                "world_boot_response.schema.json"
            ),
            additional_writable_dirs=(vat.root,),
            reasoning_effort=args.gm_reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            api_idle_timeout_seconds=getattr(args, "gm_api_idle_timeout_seconds", DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS),
            provider=args.gm_provider,
            api_key_env=args.gm_api_key_env,
            base_url=args.gm_base_url,
            api=args.gm_api,
            context_window=args.gm_context_window,
            max_tokens=args.gm_max_tokens,
            codex_sandbox=args.codex_sandbox,
            python_runtime=getattr(args, "gm_python_runtime", None),
        )
    )
    vat_process = PiSdkVatProcess(
        PiSdkVatConfig(
            node_executable=Path(args.node_bin).expanduser().resolve(),
            runner_script=PACKAGE_ROOT.joinpath("vat_runner", "runner.js"),
            episode_id=episode_id,
            workspace=vat,
            model_provider=args.player_provider,
            model_id=args.player_model,
            model_base_url=args.player_base_url,
            thinking_level=args.player_thinking,
            raw_log_path=paths.log_dir.joinpath("vat.jsonl"),
            model_context_window=args.player_context_window,
            model_max_tokens=args.player_max_tokens,
            compaction_reserve_tokens=getattr(args, "player_compaction_reserve_tokens", 65536),
            compaction_keep_recent_tokens=getattr(args, "player_compaction_keep_recent_tokens", 20000),
            api_key_env=args.player_api_key_env,
            api_key=player_api_key,
            pi_sdk_module_path=pi_sdk_path,
            prompt_timeout_seconds=player_prompt_timeout_seconds,
            retry=getattr(args, "protocol_retry", None),
        )
    )
    orchestrator = EpisodeOrchestrator(
        episode_id=episode_id,
        spec=spec,
        paths=paths,
        vat_process=vat_process,
        gm_agent_process=gm_agent_process,
        world_prompt_profile=world_prompt_profile,
        prompt_inputs=prompt_inputs,
        max_world_steps=args.max_world_steps,
        max_artifact_attempts=getattr(args, "protocol_retry", {}).get("gm_artifact_attempts", 3),
        audit_metadata=_run_config(args, episode_id=episode_id),
    )
    preserve_staging = args.keep_staging
    try:
        result = await orchestrator.run()
        summary = _summary(result, args=args)
        summary_path: Path = staging_root.joinpath("smoke_summary.json")
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _copy_rollout_without_symlinks(staging_root, output_dir)
        return output_dir, summary
    except BaseException as error:
        if isinstance(error, UnsafeRolloutArchiveError):
            preserve_staging = True
            raise
        if staging_root.exists() and not output_dir.exists():
            try:
                _copy_rollout_without_symlinks(staging_root, output_dir)
            except UnsafeRolloutArchiveError:
                preserve_staging = True
                raise
        raise
    finally:
        vat.close()
        if not preserve_staging and staging_root.exists():
            shutil.rmtree(staging_root)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_dir, summary = asyncio.run(run_live_smoke(args))
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
