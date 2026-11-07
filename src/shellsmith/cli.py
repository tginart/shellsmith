"""``shellsmith`` command line.

Architecture map
----------------
    shellsmith episode   one Player <-> GM episode on one world file
        default:  protected host Codex execution (uses host CLIs)
        --docker: local Docker sandbox (requires compatible nested confinement)
        --player openai (default): an OpenAI-compatible endpoint (OpenRouter Qwen3.8)
        --player tinker: start policy.server.PlayerEndpoint on Tinker base weights,
                         then harvest the episode's token-exact capture
                         (capture_summary.json in the log dir)
        --player mlx: serve the native dense MLX Player on the Studio

    shellsmith view      Observatory run browser (viewer.app); default root <temp>/shellsmith
    shellsmith train     the Tinker or MLX CISPO loop from a TOML config
        --config run.toml   settings; [run].log_dir or --log-dir picks the run folder
        re-running with the same log dir resumes from its last full checkpoint
    shellsmith campaign  the adaptive per-world policy over many worlds, no training
        --config run.toml   same file as train ([train] is ignored)
        --sampler-path      evaluate a trained Tinker sampler checkpoint (default: base weights)
        --adapter-path      evaluate an exported MLX adapter
    shellsmith job       detached Studio submit/status/logs/stop/resume/sync/check

Every run prints its resolved log directory first, so a run under the system
temp directory is never lost silently.

Main callers:
    The ``shellsmith`` console script (pyproject ``[project.scripts]``).
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import sys
import uuid

from shellsmith.cli_args import GMArgumentParser
from shellsmith.config import (
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_GM_API_KEY_ENV,
    DEFAULT_GM_MODEL,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_RUNTIME,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
    DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS,
    DEFAULT_PLAYER_API_KEY_ENV,
    DEFAULT_PLAYER_BASE_URL,
    DEFAULT_PLAYER_MODEL,
    DEFAULT_PLAYER_PLAIN_RENDERER,
    DEFAULT_PLAYER_PROVIDER,
    DEFAULT_PLAYER_THINKING_RENDERER,
    DEFAULT_TINKER_BASE_MODEL,
    SUPPORTED_TINKER_BASE_MODELS,
    GM_PROVIDER_BY_RUNTIME,
    PLAYER_ENDPOINT_KEY_ENV,
)
from shellsmith.paths import DEFAULT_LOG_ROOT, DEFAULT_WORLD

# Pi's working window when the Player is Shellsmith's Tinker endpoint (mirrors
# policy.sampler.CONTEXT_WINDOW_TOKENS without importing the Tinker-side module).
TINKER_CONTEXT_WINDOW_TOKENS = 224 * 1024

# Summary fields shown after an episode; the full summary is in the log dir.
SUMMARY_FIELDS = ("final_reward", "trainable", "completion_percentage", "tool_turn_count", "integrity_issues")


def _episode_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("episode", help="Run one episode on one world file.")
    parser.add_argument("--world-file", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--log-dir", type=Path, help="Episode output folder (default: a new folder under the system temp dir).")
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument("--host", dest="host", action="store_true", default=True, help="Run with protected host Codex execution (default).")
    backend.add_argument("--docker", dest="host", action="store_false", help="Run in Docker; requires a compatible confined engine.")
    parser.add_argument("--image", help="Docker image (default: shellsmith:<source and recipe hash>, built by docker/build.sh).")
    parser.add_argument("--gm-runtime", choices=tuple(GM_PROVIDER_BY_RUNTIME), default=DEFAULT_GM_RUNTIME)
    parser.add_argument("--gm-model", help="Codex GM model (defaults to Sol).")
    parser.add_argument("--gm-reasoning-effort", default=DEFAULT_GM_REASONING_EFFORT)
    parser.add_argument("--gm-api-idle-timeout-seconds", type=float, default=DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("--gm-api-key-env", default=DEFAULT_GM_API_KEY_ENV, help="Credential env var for a Pi GM.")
    parser.add_argument("--skip-player-models-preflight", action="store_true", help="Allow chat-only Player endpoints without GET /models; a credential is still required.")
    parser.add_argument("--player-base-url", default=DEFAULT_PLAYER_BASE_URL)
    parser.add_argument("--player-model", default=DEFAULT_PLAYER_MODEL)
    parser.add_argument("--player-provider", default=DEFAULT_PLAYER_PROVIDER)
    parser.add_argument("--player-api-key-env", default=DEFAULT_PLAYER_API_KEY_ENV)
    parser.add_argument("--player", choices=("openai", "tinker", "mlx"), default="openai",
                        help="openai: an external endpoint; tinker or mlx: a private token-exact Player.")
    parser.add_argument("--config", type=Path, help="MLX Player TOML configuration.")
    parser.add_argument("--mlx-model-path", type=Path, help="Execution-host MLX model directory.")
    parser.add_argument("--adapter-path", type=Path, help="MLX adapter export to evaluate.")
    parser.add_argument("--tinker-model", choices=SUPPORTED_TINKER_BASE_MODELS,
                        default=DEFAULT_TINKER_BASE_MODEL,
                        help="Qwen3.8-27B model compatible with the fixed Player renderers; context is checked with Tinker.")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_GM_TURN_TIMEOUT_SECONDS, help="Deadline for each individual GM turn.")
    parser.add_argument("--player-prompt-timeout-seconds", type=float, default=DEFAULT_PLAYER_PROMPT_TIMEOUT_SECONDS, help="Deadline for the complete Player trajectory (default: 12h).")
    parser.add_argument("--docker-timeout-seconds", type=float, help="Outer Docker deadline; defaults to the Player deadline plus six GM turns and 5 minutes for I/O. Cannot be shorter than that budget.")
    parser.add_argument("--max-world-steps", type=int)
    parser.add_argument("--pi-sdk-module-path", type=Path, help="Host mode: Pi SDK dist/index.js (default: the lockfile-pinned copy).")
    parser.set_defaults(handler=run_episode_command)


def _resolve_log_dir(args: argparse.Namespace) -> Path:
    if args.log_dir is not None:
        return args.log_dir.expanduser().resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_LOG_ROOT / "episodes" / f"{args.world_file.stem}-{stamp}-{uuid.uuid4().hex[:12]}"


def _episode_id(args: argparse.Namespace) -> str:
    """An opaque URL-safe ID, independent of user-selected output path text."""
    if not getattr(args, "episode_id", None):
        args.episode_id = f"episode-{uuid.uuid4().hex}"
    return args.episode_id


def _run_docker(args: argparse.Namespace, log_dir: Path, *, player_api_key: str | None = None) -> dict:
    from shellsmith.sandbox.docker_launcher import DockerEpisodeConfig, default_image, run_episode_in_docker

    smoke_env = {
        "SHELLSMITH_GM_MODEL": args.gm_model,
        "SHELLSMITH_GM_PROVIDER": GM_PROVIDER_BY_RUNTIME[args.gm_runtime],
        "SHELLSMITH_GM_REASONING_EFFORT": args.gm_reasoning_effort,
        "SHELLSMITH_TIMEOUT_SECONDS": str(args.timeout_seconds),
        "SHELLSMITH_GM_API_IDLE_TIMEOUT_SECONDS": str(args.gm_api_idle_timeout_seconds),
        "SHELLSMITH_PLAYER_PROMPT_TIMEOUT_SECONDS": str(args.player_prompt_timeout_seconds),
    }
    if args.max_world_steps is not None:
        smoke_env["SHELLSMITH_MAX_WORLD_STEPS"] = str(args.max_world_steps)
    smoke_env["SHELLSMITH_EPISODE_ID"] = _episode_id(args)
    if args.player == "tinker":
        smoke_env["SHELLSMITH_PLAYER_RENDERER"] = DEFAULT_PLAYER_THINKING_RENDERER
        smoke_env["SHELLSMITH_PLAYER_CONTEXT_WINDOW"] = str(TINKER_CONTEXT_WINDOW_TOKENS)
    if args.skip_player_models_preflight:
        smoke_env["SHELLSMITH_SKIP_PLAYER_MODELS_PREFLIGHT"] = "1"
    config = DockerEpisodeConfig(
        image=args.image or default_image(),
        allow_image_overrides=args.image is not None,
        gm_runtime=args.gm_runtime,
        player_base_url=args.player_base_url,
        player_model=args.player_model,
        player_provider=args.player_provider,
        player_api_key_env=args.player_api_key_env,
        player_api_key=player_api_key,
        gm_api_key_env=args.gm_api_key_env,
        smoke_env=smoke_env,
        timeout_seconds=args.docker_timeout_seconds,
    )
    print(f"shellsmith: image {config.image}", flush=True)
    return run_episode_in_docker(args.world_file, log_dir, config)


def _run_host(args: argparse.Namespace, log_dir: Path, *, player_api_key: str | None = None) -> dict:
    from shellsmith.episode.smoke import _parser, run_live_smoke

    argv = [
        "--world-file", str(args.world_file),
        "--output-dir", str(log_dir),
        "--episode-id", _episode_id(args),
        "--gm-runtime", args.gm_runtime,
        "--gm-model", args.gm_model,
        "--gm-provider", GM_PROVIDER_BY_RUNTIME[args.gm_runtime],
        "--gm-reasoning-effort", args.gm_reasoning_effort,
        "--gm-api-idle-timeout-seconds", str(args.gm_api_idle_timeout_seconds),
        "--timeout-seconds", str(args.timeout_seconds),
        "--player-prompt-timeout-seconds", str(args.player_prompt_timeout_seconds),
        "--player-base-url", args.player_base_url,
        "--player-model", args.player_model,
        "--player-provider", args.player_provider,
        "--player-api-key-env", args.player_api_key_env,
    ]
    if args.player in {"tinker", "mlx"}:
        profile = getattr(args, "mlx_profile", None)
        argv += ["--player-renderer", profile["thinking_renderer"] if profile else DEFAULT_PLAYER_THINKING_RENDERER,
                 "--player-context-window", str(profile["context_window"] if profile else TINKER_CONTEXT_WINDOW_TOKENS)]
        if profile:
            argv += ["--player-max-tokens", str(profile["max_tokens"]),
                     "--player-thinking", profile["reasoning_effort"]]
    if args.gm_api_key_env:
        argv += ["--gm-api-key-env", args.gm_api_key_env]
    if args.max_world_steps is not None:
        argv += ["--max-world-steps", str(args.max_world_steps)]
    if args.pi_sdk_module_path is not None:
        argv += ["--pi-sdk-module-path", str(args.pi_sdk_module_path)]
    if args.skip_player_models_preflight:
        argv.append("--skip-player-models-preflight")
    smoke_args = _parser().parse_args(argv)
    if args.player == "mlx":
        from shellsmith.episode.gm_process.python_runtime import current_python_runtime
        smoke_args.gm_python_runtime = current_python_runtime()
        smoke_args.player_reasoning_tokens = args.mlx_profile.get("resolved_reasoning_tokens", args.mlx_profile["thinking_tokens"])
        smoke_args.player_response_tokens = args.mlx_profile["response_tokens"]
        smoke_args.player_compaction_reserve_tokens = args.mlx_profile["compaction_reserve_tokens"]
        smoke_args.player_compaction_keep_recent_tokens = args.mlx_profile["compaction_keep_recent_tokens"]
        smoke_args.protocol_retry = args.protocol_retry
    _output_dir, summary = asyncio.run(run_live_smoke(smoke_args, player_api_key=player_api_key))
    return summary


def _write_player_records(path: Path, records: list) -> None:
    """Persist every endpoint record token-exactly (prompt token ids included).

    Why this exists:
        In-memory records hold the live prompt object. Saving the ids lets a
        record be rebuilt later (``ModelInput.from_ints``) to audit or replay
        the capture, for example the M3 parity check against the legacy stack.
    """
    rows = []
    for record in records:
        row = {key: getattr(record, key) for key in (
            "request_id", "created_at", "prompt_tokens", "sampled_tokens", "sampled_logprobs", "content",
            "finish_reason", "payload", "reasoning_content", "session_id", "protocol_row")}
        row["forced_close_token_span"] = list(record.forced_close_token_span) if record.forced_close_token_span else None
        row["prompt_token_ids"] = list(record.prompt.to_ints())
        row.update(capture_version=2, backend=record.backend, policy_ref=record.policy_ref,
                   policy_identity=record.policy_identity,
                   diagnostic_generation_attempts=list(record.diagnostic_generation_attempts))
        rows.append(row)
    from shellsmith.process_io import write_json

    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    write_json(path, rows)


@contextmanager
def _recording_player_endpoint(endpoint, episode_id: str, records_path: Path):
    """Persist full records after endpoint shutdown, including failed episodes."""
    if records_path.is_file() or records_path.is_symlink():
        raise RuntimeError(f"Player records already exist: {records_path}")
    records: list = []
    failure: BaseException | None = None
    try:
        with endpoint:
            yield records
    except BaseException as error:
        failure = error
        raise
    finally:
        try:
            records.extend(endpoint.take_records(episode_id))
            _write_player_records(records_path, records)
        except Exception as error:
            if failure is None:
                raise
            failure.add_note(f"Could not persist Player records: {type(error).__name__}: {error}")


def _run_with_tinker_player(args: argparse.Namespace, log_dir: Path) -> dict:
    """Serve the Player from Tinker for one episode, then harvest its capture.

    Steps:
        1. Load Tinker, check the model's context, open a base-weights sampling client.
        2. Start a PlayerEndpoint (loopback on the host, all interfaces for --docker)
           and point the episode's Pi at /e/<episode id>/v1 with a fresh token.
        3. Run the episode, take its records, and harvest them: protocol rows ->
           acceptance audit -> records aligned to world turns.
        4. Write capture_summary.json (record and token counts, turn kinds).

    Main callers:
        ``run_episode_command`` for ``--player tinker``.
    """
    from shellsmith.policy.deps import load_tinker_deps
    from shellsmith.policy.harvest import harvest_episode
    from shellsmith.policy.sampler import PlayerSamplingConfig, TinkerPlayerSampler
    from shellsmith.policy.server import SERVED_MODEL_ID, PlayerEndpoint
    from shellsmith.policy.tinker_setup import base_sampling_client

    if args.gm_runtime != "codex":
        raise RuntimeError("Tinker Player credentials require the confined Codex GM; unconfined GMs are unsupported.")
    if args.tinker_model not in SUPPORTED_TINKER_BASE_MODELS:
        raise RuntimeError(f"Unsupported Tinker Player model: {args.tinker_model!r}; use a supported Qwen3.8-27B model.")
    if log_dir.exists() or log_dir.is_symlink():
        raise RuntimeError(f"Episode output directory already exists: {log_dir}")
    deps = load_tinker_deps()
    sampler = TinkerPlayerSampler(
        config=PlayerSamplingConfig(base_model=args.tinker_model, thinking_renderer=DEFAULT_PLAYER_THINKING_RENDERER,
                                    plain_renderer=DEFAULT_PLAYER_PLAIN_RENDERER),
        deps=deps,
        sampling_client=base_sampling_client(deps, model_name=args.tinker_model),
    )
    episode_id = _episode_id(args)
    endpoint = PlayerEndpoint(sampler=sampler, records_dir=log_dir.parent / f"{episode_id}.player_records",
                              host="127.0.0.1" if args.host else "0.0.0.0")
    with _recording_player_endpoint(endpoint, episode_id, log_dir / "player_records.json") as records:
        args.player_base_url = endpoint.base_url(episode_id, host=None if args.host else "host.docker.internal")
        args.player_model, args.player_provider, args.player_api_key_env = SERVED_MODEL_ID, "shellsmith-tinker", PLAYER_ENDPOINT_KEY_ENV
        print(f"shellsmith: Tinker Player {args.tinker_model} at {args.player_base_url}", flush=True)
        runner = _run_host if args.host else _run_docker
        summary = runner(args, log_dir, player_api_key=endpoint.api_key)
    episode_root = log_dir if args.host else log_dir / "artifacts" / "episode"
    group_key = f"shellsmith:{args.world_file.stem}:{args.gm_runtime}:{args.gm_model}:{args.tinker_model}"
    capture = harvest_episode(records, episode_root=episode_root, group_identity_key=group_key)
    capture_summary = {
        "endpoint_records": len(records),
        "accepted_for_training": capture is not None,
        "captured_turns": 0 if capture is None else len(capture.captures),
        "turn_kinds": None if capture is None else sorted({turn.kind for turn in capture.captures}),
        "sampled_tokens": sum(len(record.sampled_tokens) for record in records),
        "group_identity_key": group_key,
    }
    (log_dir / "capture_summary.json").write_text(json.dumps(capture_summary, indent=2) + "\n", encoding="utf-8")
    print(f"shellsmith: capture {json.dumps(capture_summary)}", flush=True)
    return summary


def _run_with_mlx_player(args: argparse.Namespace, log_dir: Path) -> dict:
    from dataclasses import asdict, replace

    from shellsmith.jobs.lease import GpuLease
    from shellsmith.mlx import create_session
    from shellsmith.policy.harvest import harvest_episode
    from shellsmith.policy.server import PlayerEndpoint, SERVED_MODEL_ID
    from shellsmith.process_io import write_json
    from shellsmith.train.backend import player_profile, profile_hash
    from shellsmith.train.config import load_train_config, validate_train_config

    if not args.host:
        raise SystemExit("MLX v1 episodes require protected host execution on the Studio.")
    if args.gm_runtime != "codex":
        raise SystemExit("MLX Player credentials require the confined Codex GM.")
    if log_dir.exists() or log_dir.is_symlink():
        raise RuntimeError(f"Episode output directory already exists: {log_dir}")
    config = load_train_config(getattr(args, "config", None), campaign_mode="inference")
    config = replace(config, compute=replace(config.compute, backend="mlx"))
    if getattr(args, "mlx_model_path", None) is not None:
        config = replace(config, mlx=replace(config.mlx, model_path=str(args.mlx_model_path.expanduser().resolve())))
    validate_train_config(config, campaign_mode="inference")
    _require_mlx_execution_ready(config)
    if args.pi_sdk_module_path is None and config.player.pi_sdk_module_path:
        args.pi_sdk_module_path = Path(config.player.pi_sdk_module_path)
    if getattr(args, "adapter_path", None):
        from shellsmith.mlx.checkpoint import verify_checkpoint
        from shellsmith.mlx.identity import resolve_identity

        verify_checkpoint(args.adapter_path, kind="sampler", expected_identity=resolve_identity(config.mlx))
    args.mlx_profile = player_profile(config)
    args.protocol_retry = asdict(config.retry)
    episode_id = _episode_id(args)
    with GpuLease(config.compute.gpu_lease_root or None):
        session = asyncio.run(create_session(config, checkpoint_root=log_dir.parent / f"{episode_id}.checkpoints"))
        try:
            if getattr(args, "adapter_path", None):
                session.load_adapter(args.adapter_path)
            endpoint = PlayerEndpoint(sampler=session.sampler,
                                      records_dir=log_dir.parent / f"{episode_id}.player_records")
            with _recording_player_endpoint(endpoint, episode_id, log_dir / "player_records.json") as records:
                args.player_base_url = endpoint.base_url(episode_id)
                args.player_model = SERVED_MODEL_ID
                args.player_provider = "shellsmith-mlx"
                args.player_api_key_env = PLAYER_ENDPOINT_KEY_ENV
                summary = _run_host(args, log_dir, player_api_key=endpoint.api_key)
            group_key = f"shellsmith:mlx:{args.world_file.stem}:{profile_hash(args.mlx_profile)}"
            capture = harvest_episode(records, episode_root=log_dir, group_identity_key=group_key)
            write_json(log_dir / "capture_summary.json", {
                "backend": "mlx", "endpoint_records": len(records),
                "accepted_for_training": capture is not None,
                "captured_turns": len(capture.captures) if capture else 0,
                "sampled_tokens": sum(len(record.sampled_tokens) for record in records),
                "group_identity_key": group_key, "policy_identity": session.policy_identity,
                "policy_ref": session.policy_ref,
                "policy_kind": "sampler_checkpoint" if getattr(args, "adapter_path", None) else "base",
            })
            return summary
        finally:
            session.close()


def _readiness(config) -> dict:
    """Run bounded probes without constructing a resident model or a run directory."""
    from shellsmith.train.backend import player_profile
    from shellsmith.train.world_selection import select_world_stream

    worlds = select_world_stream(config)
    if config.compute.backend == "mlx":
        _configure_mlx_precision()
        if config.compute.host:
            from shellsmith.jobs import Endpoint, JobClient
            from shellsmith.jobs.client import DEFAULT_EXECUTION_HOME, DEFAULT_EXECUTION_PYTHON

            endpoint = Endpoint(config.compute.host, config.compute.home or DEFAULT_EXECUTION_HOME,
                                config.compute.python or DEFAULT_EXECUTION_PYTHON)
            report = JobClient(endpoint).check(config=config.to_jsonable())
        else:
            from shellsmith.train.preflight import probe_document

            report = probe_document(config.to_jsonable())
    else:
        from shellsmith.train.harness_identity import codex_identity, pi_sdk_identity

        codex, sdk = codex_identity(), pi_sdk_identity(config.player.pi_sdk_module_path)
        blockers = []
        if codex["executable_sha256"] is None:
            blockers.append("Codex executable not found on PATH.")
        if sdk["module_sha256"] is None:
            blockers.append("The lockfile-pinned Pi SDK is not installed.")
        report = {"ready": not blockers, "blockers": blockers, "codex": codex, "pi_sdk": sdk}
        if config.compute.backend == "studio":
            from studio_compute.client import Client
            from shellsmith.train.studio import load_renderer
            try:
                client = Client(config.studio.endpoint)
                capability = client.capabilities()
                report["service"] = capability
                if config.studio.model_id != capability["model"]:
                    blockers.append("Configured model differs from Studio capability.")
                if not 0 < config.studio.context_window <= capability["context_window"]:
                    blockers.append("Configured context exceeds Studio capability.")
                if config.studio.cache_precision not in capability["cache_precisions"]:
                    blockers.append("Configured cache precision is unsupported by Studio.")
                load_renderer(client, capability, reasoning_effort=config.studio.reasoning_effort)
                report["chat_template_checked"] = True
            except Exception as error:
                blockers.append(f"Studio service unavailable: {type(error).__name__}: {error}")
            report["ready"] = not blockers
    return report | {"backend": config.compute.backend, "profile": player_profile(config),
                     "world_ids": [world.world_id for world in worlds.worlds]}


def _require_mlx_execution_ready(config) -> None:
    from shellsmith.train.preflight import probe_document

    _configure_mlx_precision()
    report = probe_document(config.to_jsonable())
    if not report["ready"]:
        raise SystemExit("MLX execution readiness failed:\n- " + "\n- ".join(report["blockers"]))


def _configure_mlx_precision() -> None:
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    if os.environ["MLX_ENABLE_TF32"] != "0":
        raise SystemExit("The qualified MLX runtime requires MLX_ENABLE_TF32=0 before MLX is imported.")


def run_episode_command(args: argparse.Namespace) -> int:
    """Run one episode, print the log dir first and a short summary last."""
    from shellsmith.episode.gm_process.shared import validate_gm_runtime

    validate_gm_runtime(args.gm_runtime)
    if args.player != "mlx":
        for name in ("config", "mlx_model_path", "adapter_path"):
            if getattr(args, name, None) is not None:
                raise SystemExit(f"--{name.replace('_', '-')} requires --player mlx.")
    if args.host and (args.image is not None or args.docker_timeout_seconds is not None):
        raise SystemExit("--image and --docker-timeout-seconds require --docker.")
    args.world_file = args.world_file.expanduser().resolve()
    if not args.world_file.is_file():
        raise SystemExit(f"World file not found: {args.world_file}")
    log_dir = _resolve_log_dir(args)
    print(f"shellsmith: log dir {log_dir}", flush=True)
    if args.player == "tinker":
        summary = _run_with_tinker_player(args, log_dir)
    elif args.player == "mlx":
        summary = _run_with_mlx_player(args, log_dir)
    else:
        summary = _run_host(args, log_dir) if args.host else _run_docker(args, log_dir)
    print(json.dumps({key: summary.get(key) for key in SUMMARY_FIELDS}, indent=2))
    return 0


def run_train_command(args: argparse.Namespace) -> int:
    """Load the TOML config (plus --log-dir) and run or resume training."""
    from shellsmith.policy.deps import load_tinker_deps
    from shellsmith.train.config import load_train_config
    from shellsmith.train.trainer import run_train

    config = load_train_config(args.config, log_dir=args.log_dir)
    if getattr(args, "dry_run", False):
        report = _readiness(config)
        print(json.dumps(report, indent=2))
        return 0 if report["ready"] else 1
    if config.compute.backend == "mlx" and config.compute.host:
        raise SystemExit("This configuration targets the Studio; start it with 'shellsmith job submit --config … --request-id …'.")
    if config.compute.backend == "mlx":
        _require_mlx_execution_ready(config)
    deps = load_tinker_deps() if config.compute.backend == "tinker" else None
    state = asyncio.run(run_train(config, deps=deps))
    print(json.dumps({key: state.get(key) for key in ("status", "step", "next_world_offset", "halt")}, indent=2))
    return 0 if state.get("status") == "complete" else 1


def _train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train the Player with Tinker or MLX CISPO over streamed worlds.")
    parser.add_argument("--config", type=Path, help="TOML config (see shellsmith.train.config).")
    parser.add_argument("--log-dir", type=Path, help="Run folder; overrides [run].log_dir. Reuse it to resume.")
    parser.add_argument("--dry-run", action="store_true", help="Report readiness without loading weights or starting a run.")
    parser.set_defaults(handler=run_train_command)


def run_campaign_command(args: argparse.Namespace) -> int:
    """Load the TOML config (plus --log-dir) and run one inference campaign."""
    from shellsmith.campaign.inference import run_inference_campaign
    from shellsmith.policy.deps import load_tinker_deps
    from shellsmith.train.config import load_train_config

    config = load_train_config(args.config, log_dir=args.log_dir, campaign_mode="inference")
    adapter_path = getattr(args, "adapter_path", None)
    if adapter_path and config.compute.backend != "mlx":
        raise SystemExit("--adapter-path requires [compute].backend = 'mlx'.")
    if args.sampler_path and config.compute.backend == "mlx":
        raise SystemExit("--sampler-path is a Tinker reference; use --adapter-path for MLX.")
    if config.compute.backend == "mlx" and config.compute.host:
        raise SystemExit("Run MLX campaigns on the execution host with [compute].host empty.")
    if config.compute.backend == "mlx":
        _require_mlx_execution_ready(config)
    deps = load_tinker_deps() if config.compute.backend == "tinker" else None
    summary = asyncio.run(run_inference_campaign(config, deps=deps, sampler_path=args.sampler_path,
                                                **({"adapter_path": adapter_path} if adapter_path else {})))
    return 0 if summary["status"] == "complete" else 1


def _campaign_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("campaign", help="Run the adaptive per-world policy over many worlds (no training).")
    parser.add_argument("--config", type=Path, help="TOML config (see shellsmith.train.config); [train] is ignored.")
    parser.add_argument("--log-dir", type=Path, help="New campaign folder; overrides [run].log_dir.")
    parser.add_argument("--sampler-path", help="Published Tinker (tinker://...) or Studio (studio-policy:...) policy to evaluate.")
    parser.add_argument("--adapter-path", type=Path, help="MLX adapter export to evaluate instead of base weights.")
    parser.set_defaults(handler=run_campaign_command)


def _run_viewer(argv: list[str]) -> int:
    from shellsmith.viewer.app import main as viewer_main

    return viewer_main(argv)


def _job_parser(subparsers: argparse._SubParsersAction) -> None:
    jobs = subparsers.add_parser("job", help="Submit and control Studio-owned training jobs over SSH.")
    operations = jobs.add_subparsers(dest="job_operation", required=True)
    for name in ("submit", "status", "logs", "stop", "resume", "sync", "check"):
        parser = operations.add_parser(name)
        parser.add_argument("--config", type=Path, required=name == "submit", help="Run TOML and compute connection.")
        parser.add_argument("--host", help="SSH user@host; empty string selects this machine. "
                            "Defaults to compute.host, then $SHELLSMITH_STUDIO_HOST.")
        parser.add_argument("--home", help="Execution-host Shellsmith home.")
        parser.add_argument("--python", help="Execution-host Python interpreter.")
        parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".local/state/shellsmith/job-client")
        if name not in {"submit", "check"}:
            parser.add_argument("job_id")
        if name in {"submit", "stop", "resume"}:
            parser.add_argument("--request-id", required=True, help="Stable request identity; retain it when retrying.")
        if name in {"submit", "check"}:
            parser.add_argument("--command", dest="job_command", choices=("train", "campaign"), default="train")
        if name == "submit":
            from shellsmith.paths import PROJECT_ROOT

            parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT)
            parser.add_argument("--adapter-path", type=Path, help="Execution-host MLX sampler export for a campaign.")
        if name == "logs":
            parser.add_argument("--offset", type=int, default=0)
            parser.add_argument("--limit", type=int, default=65536)
        if name == "sync":
            parser.add_argument("--local-dir", type=Path, required=True, help="Private local Observatory artifact mirror.")
        parser.set_defaults(handler=run_job_command)


def run_job_command(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from shellsmith.jobs import Endpoint, JobClient
    from shellsmith.jobs.client import DEFAULT_EXECUTION_HOME, DEFAULT_EXECUTION_PYTHON
    from shellsmith.jobs.storage import identifier
    from shellsmith.process_io import write_json
    from shellsmith.train.config import load_train_config

    config = load_train_config(args.config, campaign_mode="inference" if getattr(args, "job_command", "train") == "campaign" else "tinker_train") if args.config else None
    compute = config.compute if config else None
    host = args.host if args.host is not None else (
        compute.host if compute and compute.host else os.environ.get("SHELLSMITH_STUDIO_HOST"))
    endpoint = None if host is None else Endpoint(
        host,
        args.home or (compute.home if compute and compute.home else DEFAULT_EXECUTION_HOME),
        args.python or (compute.python if compute and compute.python else DEFAULT_EXECUTION_PYTHON),
    )
    receipt_path = None
    if hasattr(args, "job_id"):
        receipt_path = args.cache_dir / "endpoints" / f"{identifier(args.job_id)}.json"
        client_receipt = args.cache_dir / f"{identifier(args.job_id)}.json"
        recorded = None
        if receipt_path.is_file():
            recorded = Endpoint(**json.loads(receipt_path.read_text()))
        elif client_receipt.is_file():
            recorded = Endpoint(**json.loads(client_receipt.read_text())["endpoint"])
        if recorded is not None:
            if any(getattr(args, field) is not None and getattr(args, field) != getattr(recorded, field)
                   for field in ("host", "home", "python")):
                raise SystemExit("This job's destination is pinned; supplied connection differs from its recorded endpoint.")
            endpoint = recorded
    if endpoint is None:
        raise SystemExit("No execution host: pass --host, set compute.host, or export SHELLSMITH_STUDIO_HOST.")
    client = JobClient(endpoint, cache_dir=args.cache_dir)
    operation = args.job_operation
    if operation == "submit":
        if args.adapter_path and args.job_command != "campaign":
            raise SystemExit("Job --adapter-path requires --command campaign; training recovery uses full checkpoints.")
        options = {"command": args.job_command, "adapter_path": str(args.adapter_path) if args.adapter_path else None}
        result = client.submit(config_path=args.config, source_root=args.source_root, request_id=args.request_id,
                               **(options if args.job_command != "train" else {}))
        job_id = result.get("job_id") or result.get("id")
        if job_id:
            receipt_path = args.cache_dir / "endpoints" / f"{identifier(job_id)}.json"
            receipt_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
            write_json(receipt_path, asdict(endpoint))
    elif operation == "check":
        result = client.check(**({"config": config.to_jsonable()} if config else {}))
    elif operation == "logs":
        result = client.logs(args.job_id, offset=args.offset, limit=args.limit)
    elif operation == "sync":
        result = client.sync(args.job_id, local_dir=args.local_dir)
    elif operation in {"stop", "resume"}:
        result = getattr(client, operation)(args.job_id, request_id=args.request_id)
    else:
        result = client.status(args.job_id)
    print(json.dumps(result, indent=2))
    return 1 if result.get("ready") is False or result.get("status") in {"failed", "interrupted"} else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shellsmith")
    subparsers = parser.add_subparsers(dest="command", required=True, parser_class=GMArgumentParser)
    _episode_parser(subparsers)
    _train_parser(subparsers)
    _campaign_parser(subparsers)
    _job_parser(subparsers)
    view = subparsers.add_parser("view", help="Observatory: read-only browser for runs (needs the [viewer] extra).",
                                 add_help=False)
    view.set_defaults(handler=lambda args: _run_viewer(args.passthrough))
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra = build_parser().parse_known_args(argv)
    if args.command == "view":
        args.passthrough = extra  # viewer flags: --runs-root, --host, --port
    elif extra:
        raise SystemExit(f"Unrecognized arguments: {' '.join(extra)}")
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
