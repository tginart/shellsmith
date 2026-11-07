"""``shellsmith train`` configuration: one TOML file, every setting explicit.

Architecture map
----------------
    load_train_config(path, overrides)
        TOML sections  [run] [worlds] [gm] [player] [train] [adaptive]
        -> TrainConfig (frozen dataclasses with the defaults below)
        -> unknown section or key: error (a typo never silently falls back)
    TrainConfig.resolved_log_dir()   default: <system temp>/shellsmith/runs/<run_id>

Example (every key optional)::

    [run]
    log_dir = "/data/shellsmith/runs/canary"
    [worlds]
    select = "manifest"        # manifest | random | ids | hero
    count = 8                  # catalog selection only (not hero)
    seed = 0                   # catalog random selection only (not hero)
    [train]
    steps = 3
    groups_per_update = 2
    [adaptive]
    initial_attempts = 4

Why this exists:
    The plan keeps run settings, including ``log_dir`` and the per-world
    adaptive policy, in a config file instead of a long CLI.

Main callers:
    ``shellsmith train`` (cli.py) and ``train.trainer.run_train``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
import math
import re
from pathlib import Path
import tomllib
from typing import Any
import uuid

from shellsmith.config import (
    DEFAULT_GM_MODEL,
    DEFAULT_GM_PROVIDER,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_RUNTIME,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_MAX_WORLD_STEPS,
    DEFAULT_LORA_RANK,
    DEFAULT_PLAYER_PLAIN_RENDERER,
    DEFAULT_PLAYER_THINKING_RENDERER,
    DEFAULT_TINKER_BASE_MODEL,
    SUPPORTED_TINKER_BASE_MODELS,
)
from shellsmith.catalog import REASONING_EFFORTS
from shellsmith.paths import DEFAULT_LOG_ROOT
from shellsmith.train.name_budget import validate_component_name
from shellsmith.train.stream_policy import StreamPolicyConfig
from shellsmith.mlx.generation import completion_budget, reasoning_budget, reasoning_effort


@dataclass(frozen=True)
class RunSection:
    run_id: str = ""
    log_dir: str = ""


@dataclass(frozen=True)
class TelemetrySection:
    """Select Backprop (default) or W&B; offline/disabled are logging modes.

    Foundry resource IDs and credentials stay outside the serialized training
    config. Model/checkpoint uploads are absent.
    """

    provider: str = "backprop"  # backprop | wandb
    mode: str = "auto"  # auto | online | offline | disabled
    connection_file: str = ""  # BACKPROP_CONFIG wins; otherwise .backprop/config.json
    project: str = "shellsmith"
    entity: str = ""
    log_trajectories: bool = True
    capture_console: bool = True
    system_metrics: bool = True
    system_interval_seconds: float = 15.0
    init_timeout_seconds: float = 300.0


@dataclass(frozen=True)
class WorldsSection:
    """Catalog count/seed never truncate or reshuffle an immutable hero slate.

    With ``select = "hero"``, set ``hero_slate`` to its JSON path and match
    ``train.groups_per_update`` to the slate. Paths are relative to the working
    directory, like ``run.log_dir``.
    """

    select: str = "manifest"  # manifest order | seeded random | explicit ids | hero
    count: int = 8  # catalog only
    seed: int = 0  # random catalog selection only
    ids: tuple[str, ...] = ()
    hero_slate: str = ""


@dataclass(frozen=True)
class GMSection:
    runtime: str = DEFAULT_GM_RUNTIME
    model: str = DEFAULT_GM_MODEL
    provider: str = DEFAULT_GM_PROVIDER
    reasoning_effort: str = DEFAULT_GM_REASONING_EFFORT
    api_key_env: str = "OPENROUTER_API_KEY"
    timeout_seconds: float = DEFAULT_GM_TURN_TIMEOUT_SECONDS
    api_idle_timeout_seconds: float = DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS
    max_world_steps: int = DEFAULT_MAX_WORLD_STEPS


@dataclass(frozen=True)
class PlayerSection:
    base_model: str = DEFAULT_TINKER_BASE_MODEL
    lora_rank: int = DEFAULT_LORA_RANK
    thinking_renderer: str = DEFAULT_PLAYER_THINKING_RENDERER
    plain_renderer: str = DEFAULT_PLAYER_PLAIN_RENDERER
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 32_768
    prompt_timeout_seconds: float = 43_200.0
    pi_sdk_module_path: str = ""


@dataclass(frozen=True)
class TrainSection:
    steps: int = 3
    groups_per_update: int = 2
    learning_rate: float = 5e-5
    cispo_clip_low: float = 0.0
    cispo_clip_high: float = 4.0
    kl_penalty_coef: float = 0.0  # off by default (plan §6), matching rl.train
    kl_discount_factor: float = 0.0
    checkpoint_every: int = 1
    max_update_prompt_tokens: int = 0  # zero means unrestricted
    max_update_completion_tokens: int = 0


@dataclass(frozen=True)
class OptimizerSection:
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8


@dataclass(frozen=True)
class CreditSection:
    advantage_std_floor: float = 0.25
    advantage_cap: float = 3.0
    credit_coefficient: float = 0.5
    reasoning_coefficient: float = 0.25
    reward_std_floor: float = 0.1
    reasoning_std_floor: float = 0.1
    residual_cap: float = 2.0
    residual_cap_floor: float = 0.25
    final_advantage_cap: float = 4.0


@dataclass(frozen=True)
class RetrySection:
    thinking_anomaly_attempts: int = 3
    gm_artifact_attempts: int = 3
    player_enabled: bool = True
    player_max_retries: int = 3
    player_base_delay_ms: int = 2000
    provider_max_retry_delay_ms: int = 60000


@dataclass(frozen=True)
class ComputeSection:
    backend: str = "tinker"
    host: str = ""
    home: str = ""
    python: str = ""
    gpu_lease_root: str = ""


@dataclass(frozen=True)
class MLXSection:
    profile: str = "qwen38_dense_8bit_32k"
    model_id: str = "Qwen/Qwen3.8-27B"
    model_path: str = "~/models/Qwen3.8-27B-8bit"
    revision: str = ""
    quantization_bits: int = 8
    quantization_group_size: int = 64
    lora_rank: int = 16
    lora_scale: float = 2.0
    lora_layers: int = -1
    lora_targets: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    context_window: int = 32_768
    reasoning_effort: str = "medium"
    max_tokens: int = 0  # legacy total ceiling; zero derives it from the phase limits
    thinking_tokens: int = 0  # positive legacy override; zero selects by effort
    response_tokens: int = 4_096
    plain_tokens: int = 0  # legacy plain ceiling; zero uses response_tokens
    compaction_tokens: int = 2_048
    compaction_reserve_tokens: int = 12_288
    compaction_keep_recent_tokens: int = 8_192


@dataclass(frozen=True)
class StudioSection:
    endpoint: str = "http://127.0.0.1:8766"
    training_client: str = ""
    operation_timeout_seconds: float = 43_200.0
    model_id: str = "Qwen/Qwen3.8-27B"
    context_window: int = 262_144
    reasoning_effort: str = "medium"
    max_tokens: int = 0  # legacy total ceiling; zero derives it from the phase limits
    thinking_tokens: int = 0  # positive legacy override; zero selects by effort
    reasoning_tokens_low: int = 2_048
    reasoning_tokens_medium: int = 4_096
    reasoning_tokens_xhigh: int = 8_192
    response_tokens: int = 32_768
    plain_tokens: int = 0  # legacy plain ceiling; zero uses response_tokens
    compaction_tokens: int = 2_048
    compaction_reserve_tokens: int = 65_536
    compaction_keep_recent_tokens: int = 20_000
    lora_rank: int = 16
    lora_scale: float = 2.0
    cache_precision: str = "bf16"


@dataclass(frozen=True)
class AdaptiveSection:
    """The ported per-world adaptive policy (``train.stream_policy``)."""

    initial_attempts: int = 4
    maximum_attempts: int = 8
    minimum_trainable: int = 4
    minimum_reward_range: float = 0.15
    low_spread_top_up: int = 4
    provider_outage: str = "halt"
    outage_failure_threshold: int = 5
    outage_cooldown_seconds: float = 300.0
    parallelism: int = 8
    parallelism_per_provider: int = 8


# Adaptive-policy keys a single world may override.
PER_WORLD_KEYS = (
    "initial_attempts",
    "maximum_attempts",
    "minimum_trainable",
    "minimum_reward_range",
    "low_spread_top_up",
)


# Only these fields are TOML sections / serialized settings. In particular,
# run_id_generated is invocation metadata, never a user-configurable section.
_SECTION_TYPES = {
    "run": RunSection,
    "telemetry": TelemetrySection,
    "worlds": WorldsSection,
    "gm": GMSection,
    "player": PlayerSection,
    "train": TrainSection,
    "optimizer": OptimizerSection,
    "credit": CreditSection,
    "retry": RetrySection,
    "adaptive": AdaptiveSection,
    "compute": ComputeSection,
    "mlx": MLXSection,
    "studio": StudioSection,
}


@dataclass(frozen=True)
class TrainConfig:
    run: RunSection = field(default_factory=RunSection)
    telemetry: TelemetrySection = field(default_factory=TelemetrySection)
    worlds: WorldsSection = field(default_factory=WorldsSection)
    gm: GMSection = field(default_factory=GMSection)
    player: PlayerSection = field(default_factory=PlayerSection)
    train: TrainSection = field(default_factory=TrainSection)
    optimizer: OptimizerSection = field(default_factory=OptimizerSection)
    credit: CreditSection = field(default_factory=CreditSection)
    retry: RetrySection = field(default_factory=RetrySection)
    adaptive: AdaptiveSection = field(default_factory=AdaptiveSection)
    compute: ComputeSection = field(default_factory=ComputeSection)
    mlx: MLXSection = field(default_factory=MLXSection)
    studio: StudioSection = field(default_factory=StudioSection)
    world_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    run_id_generated: bool = field(default=False, repr=False, compare=False)

    def resolved_log_dir(self, *, kind: str = "runs") -> Path:
        """Explicit log_dir, or the default runs/campaigns directory for this ID."""
        if self.run.log_dir:
            return Path(self.run.log_dir).expanduser().resolve()
        return DEFAULT_LOG_ROOT / kind / self.run.run_id

    def to_jsonable(self) -> dict[str, Any]:
        """Serialize settings only, not the invocation's run-ID provenance."""
        return {
            name: {f.name: getattr(getattr(self, name), f.name) for f in fields(cls)}
            for name, cls in _SECTION_TYPES.items()
            if name != "studio" or self.compute.backend == "studio"
        } | {"world_overrides": self.world_overrides}


def _value_type(value: Any, default: Any, label: str) -> None:
    """Check dataclass field types too; bool is not an integer setting."""
    if isinstance(default, float) and type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{label} must be finite.")
    elif type(value) is not type(default):
        raise ValueError(
            f"{label} must be {type(default).__name__}, got {type(value).__name__}."
        )


def _section(cls: type, raw: Any, name: str) -> Any:
    if not isinstance(raw, dict):
        raise ValueError(f"[{name}] must be a table.")
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown key(s) in [{name}]: {', '.join(unknown)}")
    values = {}
    defaults = cls()
    for key, value in raw.items():
        default = getattr(defaults, key)
        if isinstance(default, tuple):
            if not isinstance(value, list):
                raise ValueError(f"[{name}].{key} must be an array of strings.")
            value = tuple(value)
        _value_type(value, default, f"[{name}].{key}")
        if isinstance(default, float):
            value = float(value)
        values[key] = value
    return cls(**values)


def _world_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    """Validate override tables and types; merged semantic checks follow below."""
    if not isinstance(raw, dict):
        raise ValueError("[world_overrides] must be a table of world tables.")
    overrides: dict[str, dict[str, Any]] = {}
    for world_id, table in raw.items():
        if (
            not isinstance(world_id, str)
            or world_id in {".", ".."}
            or re.fullmatch(r"[A-Za-z0-9._-]+", world_id) is None
        ):
            raise ValueError("[world_overrides] keys must be endpoint-safe world IDs.")
        name = f"world_overrides.{world_id}"
        if not isinstance(table, dict) or not table:
            raise ValueError(f"[{name}] must be a non-empty table.")
        extra = sorted(set(table) - set(PER_WORLD_KEYS))
        if extra:
            raise ValueError(
                f"[{name}] may only set {', '.join(PER_WORLD_KEYS)}; got {', '.join(extra)}."
            )
        typed = _section(AdaptiveSection, table, name)
        overrides[world_id] = {key: getattr(typed, key) for key in table}
    return overrides


def merged_stream_policy(
    config: TrainConfig,
    overrides: dict[str, Any] | None = None,
    *,
    campaign_mode: str = "tinker_train",
    groups_per_update: int | None = None,
    label: str = "[adaptive]",
) -> StreamPolicyConfig:
    """Validate the *merged* policy, without passing outage circuit fields to it."""
    adaptive = config.adaptive
    values = {key: getattr(adaptive, key) for key in PER_WORLD_KEYS} | (overrides or {})
    if campaign_mode != "inference" and values["minimum_trainable"] < 2:
        raise ValueError(
            f"{label}.minimum_trainable must be at least two (>= 2) for group-relative training."
        )
    if not 0 <= values["minimum_reward_range"] <= 1:
        raise ValueError(f"{label}.minimum_reward_range must be between 0 and 1.")
    stream = StreamPolicyConfig(
        **values,
        provider_outage=adaptive.provider_outage,
        parallelism=adaptive.parallelism,
        parallelism_per_provider=adaptive.parallelism_per_provider,
        groups_per_update=(
            groups_per_update
            if groups_per_update is not None
            else config.train.groups_per_update
            if campaign_mode != "inference"
            else 1
        ),
        campaign_mode=campaign_mode,
        attempt_ceiling=24
        if config.compute.backend in {"mlx", "studio"} and campaign_mode == "inference"
        else 12,
    )
    try:
        # Match the actual campaign engine ceiling before any caller publishes
        # a snapshot or creates a service client; positivity alone is not enough.
        from shellsmith.campaign.stream_campaign import validate_parallelism_caps

        validate_parallelism_caps(
            parallelism=adaptive.parallelism,
            parallelism_per_provider=adaptive.parallelism_per_provider,
        )
        stream.validate()
    except (ValueError, RuntimeError) as error:
        raise ValueError(f"{label}: {error}") from error
    return stream


def validate_train_config(
    config: TrainConfig, *, campaign_mode: str = "tinker_train"
) -> None:
    """Reject invalid settings before any SDK, client, or paid work is needed.

    Callers constructing dataclasses directly must pass this same validation;
    it is not just a TOML parser check. Hero assignments are loaded separately
    by ``select_world_stream`` and kept intact for the trainer's runtime check.
    """
    if campaign_mode not in {"tinker_train", "mlx_train", "studio_train", "inference"}:
        raise ValueError(f"Unsupported campaign_mode {campaign_mode!r}.")
    if not isinstance(config, TrainConfig):
        raise ValueError("config must be a TrainConfig.")
    if config.compute.backend not in {"tinker", "mlx", "studio"}:
        raise ValueError("[compute].backend must be 'tinker', 'mlx' or 'studio'.")
    if type(config.run_id_generated) is not bool:
        raise ValueError("run_id_generated must be bool.")
    for name, cls in _SECTION_TYPES.items():
        section = getattr(config, name)
        if not isinstance(section, cls):
            raise ValueError(f"[{name}] must be a {cls.__name__}.")
        if name == "train" and campaign_mode == "inference":
            continue  # Training semantics do not constrain inference campaigns.
        defaults = cls()
        for f in fields(cls):
            _value_type(
                getattr(section, f.name),
                getattr(defaults, f.name),
                f"[{name}].{f.name}",
            )

    for key in ("beta1", "beta2"):
        if not 0 <= getattr(config.optimizer, key) < 1:
            raise ValueError(f"[optimizer].{key} must be in [0, 1).")
    if config.optimizer.epsilon <= 0:
        raise ValueError("[optimizer].epsilon must be positive.")
    if config.telemetry.provider not in {"backprop", "wandb"}:
        raise ValueError("[telemetry].provider must be backprop or wandb.")
    if config.telemetry.mode not in {"auto", "online", "offline", "disabled"}:
        raise ValueError("[telemetry].mode must be auto, online, offline or disabled.")
    if not config.telemetry.project.strip():
        raise ValueError("[telemetry].project must be non-empty.")
    if config.telemetry.init_timeout_seconds <= 0:
        raise ValueError("[telemetry].init_timeout_seconds must be positive.")
    if config.telemetry.system_interval_seconds <= 0:
        raise ValueError("[telemetry].system_interval_seconds must be positive.")
    for item in fields(CreditSection):
        value = getattr(config.credit, item.name)
        positive = item.name.endswith(("std_floor", "advantage_cap")) or item.name in {
            "residual_cap", "residual_cap_floor",
        }
        if value < 0 or (positive and value == 0):
            requirement = "positive" if positive else "nonnegative"
            raise ValueError(f"[credit].{item.name} must be {requirement}.")
    if config.credit.residual_cap_floor > config.credit.residual_cap:
        raise ValueError("[credit].residual_cap_floor must fit residual_cap.")
    for key in ("thinking_anomaly_attempts", "gm_artifact_attempts", "player_base_delay_ms", "provider_max_retry_delay_ms"):
        if getattr(config.retry, key) < 1:
            raise ValueError(f"[retry].{key} must be positive.")
    if config.retry.player_max_retries < 0:
        raise ValueError("[retry].player_max_retries must be nonnegative.")

    if not config.run.run_id.strip():
        raise ValueError(
            "[run].run_id must be non-empty, including directly constructed configs."
        )
    if (
        config.run.run_id in {".", ".."}
        or re.fullmatch(r"[A-Za-z0-9._-]+", config.run.run_id) is None
    ):
        raise ValueError(
            "[run].run_id must be an endpoint-safe non-dot path segment: letters, digits, '.', '_' or '-'."
        )
    # Intrinsic/default-directory limits precede path resolution and lock I/O.
    # The trainer separately budgets composed names after selection and resume.
    validate_component_name(config.run.run_id, label="[run].run_id")
    if not config.run.log_dir:
        validate_component_name(
            f".{config.run.run_id}.trainer.lock",
            label="[run].run_id default lock filename",
            directory=DEFAULT_LOG_ROOT / "runs",
        )
    if config.adaptive.provider_outage != "halt":
        raise ValueError(
            "[adaptive].provider_outage must be 'halt' for Tinker training; fallback GMs are unconfined."
        )
    worlds = config.worlds
    if worlds.select not in {"manifest", "random", "ids", "hero"}:
        raise ValueError("[worlds].select must be manifest, random, ids or hero.")
    seen: set[str] = set()
    for world_id in worlds.ids:
        if not isinstance(world_id, str):
            raise ValueError("[worlds].ids must contain only strings.")
        if not world_id.strip() or world_id != world_id.strip():
            raise ValueError(
                "[worlds].ids must contain non-empty IDs without surrounding whitespace."
            )
        if world_id in seen:
            raise ValueError(
                f"Duplicate [worlds].ids entry: {world_id!r}; world IDs must be unique."
            )
        seen.add(world_id)
    if worlds.select == "ids" and not worlds.ids:
        raise ValueError("[worlds].select = 'ids' needs [worlds].ids.")
    if worlds.select == "hero" and not worlds.hero_slate.strip():
        raise ValueError("[worlds].select = 'hero' needs [worlds].hero_slate.")
    if worlds.select != "hero" and worlds.hero_slate:
        raise ValueError("[worlds].hero_slate requires [worlds].select = 'hero'.")

    # Integer bounds are checked before selection or modulo/checkpoint work.
    positive_fields = {
        "worlds": ("count",),
        "gm": ("max_world_steps",),
        "player": ("lora_rank", "max_tokens"),
        "train": ("steps", "groups_per_update", "checkpoint_every"),
        "adaptive": (
            "initial_attempts",
            "maximum_attempts",
            "minimum_trainable",
            "low_spread_top_up",
            "parallelism",
            "parallelism_per_provider",
        ),
    }
    positive_fields["adaptive"] += ("outage_failure_threshold",)
    for name, keys in positive_fields.items():
        if name == "train" and campaign_mode == "inference":
            continue
        for key in keys:
            if getattr(getattr(config, name), key) < 1:
                raise ValueError(f"[{name}].{key} must be at least one (>= 1).")
    for name, key in (("gm", "timeout_seconds"), ("gm", "api_idle_timeout_seconds"), ("player", "prompt_timeout_seconds")):
        if getattr(getattr(config, name), key) <= 0:
            raise ValueError(f"[{name}].{key} must be positive.")

    for key in ("max_update_prompt_tokens", "max_update_completion_tokens"):
        if getattr(config.train, key) < 0:
            raise ValueError(f"[train].{key} must be nonnegative (zero disables the cap).")
    if config.compute.backend not in {"mlx", "studio"} and (
        config.train.max_update_prompt_tokens or config.train.max_update_completion_tokens
    ):
        raise ValueError("Update token caps currently require the MLX or Studio token backend.")

    if config.adaptive.outage_cooldown_seconds < 0:
        raise ValueError("[adaptive].outage_cooldown_seconds must be non-negative.")
    player, train = config.player, config.train
    if player.prompt_timeout_seconds <= config.gm.timeout_seconds:
        raise ValueError(
            "[player].prompt_timeout_seconds must exceed [gm].timeout_seconds."
        )
    if not 0 <= player.temperature <= 2:
        raise ValueError("[player].temperature must be between 0 and 2.")
    if not 0 < player.top_p <= 1:
        raise ValueError("[player].top_p must be greater than 0 and at most 1.")
    if campaign_mode != "inference":
        if not 0 < train.learning_rate <= 1:
            raise ValueError(
                "[train].learning_rate must be greater than 0 and at most 1."
            )
        if train.cispo_clip_low < 0:
            raise ValueError("[train].cispo_clip_low must be non-negative.")
        if train.cispo_clip_high <= 0 or train.cispo_clip_high < train.cispo_clip_low:
            raise ValueError(
                "[train].cispo_clip_high must be positive and at least cispo_clip_low."
            )
        if train.kl_penalty_coef < 0:
            raise ValueError("[train].kl_penalty_coef must be non-negative.")
        if not 0 <= train.kl_discount_factor <= 1:
            raise ValueError("[train].kl_discount_factor must be between 0 and 1.")
    merged_stream_policy(config, campaign_mode=campaign_mode)
    for world_id, overrides in _world_overrides(config.world_overrides).items():
        merged_stream_policy(
            config,
            overrides,
            campaign_mode=campaign_mode,
            label=f"[world_overrides.{world_id}]",
        )

    if config.compute.backend == "mlx":
        validate_mlx_config(config, campaign_mode=campaign_mode)
    elif config.compute.backend == "studio":
        from urllib.parse import urlsplit

        if not 0 <= config.worlds.seed < 2**32:
            raise ValueError("Studio [worlds].seed must be a uint32 (0 through 2**32 - 1).")
        if campaign_mode != "inference":
            from studio_compute.protocol import identifier
            try:
                identifier(config.studio.training_client or config.run.run_id)
            except ValueError as error:
                raise ValueError("Selected Studio training client ID must have 1–200 characters.") from error
        endpoint = urlsplit(config.studio.endpoint)
        if (
            endpoint.scheme not in {"http", "https"}
            or not endpoint.hostname
            or endpoint.username
            or endpoint.password
        ):
            raise ValueError(
                "[studio].endpoint must be an HTTP endpoint without embedded credentials."
            )
        if config.studio.operation_timeout_seconds <= 0:
            raise ValueError("[studio].operation_timeout_seconds must be positive.")
        if config.studio.cache_precision not in {"bf16", "8bit"}:
            raise ValueError("[studio].cache_precision must be bf16 or 8bit.")
        if config.studio.lora_rank != 16 or config.studio.lora_scale != 2.0:
            raise ValueError("Studio 27B v1 requires rank 16, scale 2.")
        if (
            player.temperature != 1.0
            or player.top_p != 1.0
            or (campaign_mode != "inference" and train.kl_penalty_coef != 0.0)
        ):
            raise ValueError(
                "Studio v1 requires temperature=1, top_p=1 and KL disabled."
            )
        for key in (
            "context_window",
            "reasoning_tokens_low",
            "reasoning_tokens_medium",
            "reasoning_tokens_xhigh",
            "response_tokens",
            "compaction_tokens",
            "compaction_reserve_tokens",
            "compaction_keep_recent_tokens",
        ):
            if getattr(config.studio, key) < 1:
                raise ValueError(f"[studio].{key} must be positive.")
        reasoning_effort(config.studio)
        if min(config.studio.max_tokens, config.studio.plain_tokens, config.studio.thinking_tokens) < 0:
            raise ValueError("Studio legacy max_tokens/plain_tokens/thinking_tokens ceilings must be nonnegative.")
        reserve = completion_budget(config.studio)
        if (
            reserve >= config.studio.context_window
            or config.studio.compaction_reserve_tokens >= config.studio.context_window
        ):
            raise ValueError(
                "Studio completion/compaction reserves must fit the context window."
            )
        studio = config.studio
        if (
            (studio.max_tokens and reasoning_budget(studio) + studio.response_tokens > studio.max_tokens)
            or (studio.max_tokens and max(studio.plain_tokens, studio.compaction_tokens) > studio.max_tokens)
            or studio.compaction_reserve_tokens <= reserve
            or studio.compaction_keep_recent_tokens + studio.compaction_tokens
            >= studio.context_window - studio.compaction_reserve_tokens
        ):
            raise ValueError("Studio phase budgets and retained history must fit completion/context limits.")
    elif player.base_model not in SUPPORTED_TINKER_BASE_MODELS:
        raise ValueError(
            f"Unsupported Tinker base model {player.base_model!r}; the fixed Qwen3.8 "
            f"renderer/audit contract supports only {SUPPORTED_TINKER_BASE_MODELS!r}."
        )
    for key, expected in (
        ()
        if config.compute.backend in {"mlx", "studio"}
        else (
            ("thinking_renderer", DEFAULT_PLAYER_THINKING_RENDERER),
            ("plain_renderer", DEFAULT_PLAYER_PLAIN_RENDERER),
        )
    ):
        if getattr(player, key) != expected:
            raise ValueError(
                f"[player].{key} must be {expected!r} for the fixed Qwen3.8 renderer/audit contract."
            )
    validate_training_gm(
        runtime=config.gm.runtime,
        model=config.gm.model,
        provider=config.gm.provider,
        reasoning_effort=config.gm.reasoning_effort,
    )


def validate_mlx_config(
    config: TrainConfig, *, campaign_mode: str = "mlx_train"
) -> None:
    """Reject unsupported local-training contracts before importing MLX or claiming a run."""
    if config.optimizer != OptimizerSection():
        raise ValueError("The historical local MLX profile requires the default optimizer semantics; use Studio for configurable Adam.")
    mlx = config.mlx
    from shellsmith.mlx.profiles import PROFILE_CONTEXTS

    if mlx.profile not in PROFILE_CONTEXTS or mlx.model_id != "Qwen/Qwen3.8-27B":
        raise ValueError(
            "MLX supports the qwen38_dense_8bit_32k Qwen/Qwen3.8-27B profile and explicit context qualification candidates only."
        )
    if not mlx.model_path.strip():
        raise ValueError("[mlx].model_path must be non-empty.")
    if mlx.revision:
        raise ValueError(
            "[mlx].revision must be empty; local model assets supply the immutable identity."
        )
    if mlx.quantization_bits != 8 or mlx.quantization_group_size != 64:
        raise ValueError(
            "[mlx] requires an 8-bit base with quantization_group_size=64."
        )
    if mlx.lora_rank != 16 or mlx.lora_scale != 2.0 or mlx.lora_layers != -1:
        raise ValueError(
            "[mlx] requires lora_rank=16, lora_scale=2.0 and lora_layers=-1."
        )
    if mlx.lora_targets != MLXSection().lora_targets:
        raise ValueError(
            "[mlx].lora_targets must match the approved dense projection targets."
        )
    for key in (
        "context_window",
        "response_tokens",
        "compaction_tokens",
        "compaction_reserve_tokens",
        "compaction_keep_recent_tokens",
    ):
        if getattr(mlx, key) < 1:
            raise ValueError(f"[mlx].{key} must be a positive integer.")
    reasoning_effort(mlx)
    if min(mlx.max_tokens, mlx.plain_tokens, mlx.thinking_tokens) < 0:
        raise ValueError("MLX legacy max_tokens/plain_tokens/thinking_tokens ceilings must be nonnegative.")
    if (
        type(mlx.context_window) is not int
        or mlx.context_window != PROFILE_CONTEXTS[mlx.profile]
    ):
        raise ValueError(
            f"[mlx].context_window must be {PROFILE_CONTEXTS[mlx.profile]} for the selected profile."
        )
    if (
        mlx.max_tokens > 8192
        or (mlx.max_tokens and reasoning_budget(mlx) + mlx.response_tokens > mlx.max_tokens)
    ):
        raise ValueError("[mlx] resolved thinking + response must fit an explicit max_tokens <= 8192.")
    if mlx.max_tokens and max(mlx.plain_tokens, mlx.compaction_tokens) > mlx.max_tokens:
        raise ValueError("[mlx] plain/compaction tokens must fit max_tokens.")
    if not (completion_budget(mlx) < mlx.compaction_reserve_tokens < mlx.context_window):
        raise ValueError(
            "[mlx] compaction reserve must exceed completion budget and fit context."
        )
    if (
        mlx.compaction_keep_recent_tokens + mlx.compaction_tokens
        >= mlx.context_window - mlx.compaction_reserve_tokens
    ):
        raise ValueError(
            "[mlx] retained history and summary must fit the compaction trigger budget."
        )
    if campaign_mode != "inference" and (
        config.train.kl_penalty_coef != 0 or config.train.kl_discount_factor != 0
    ):
        raise ValueError(
            "MLX v1 does not support KL penalties; set both KL fields to zero."
        )
    if config.player.temperature != 1.0 or config.player.top_p != 1.0:
        raise ValueError(
            "The qualified MLX profile requires player.temperature=1.0 and top_p=1.0."
        )


def validate_training_gm(
    *,
    runtime: str,
    model: str,
    provider: str,
    reasoning_effort: str,
    label: str = "[gm]",
) -> None:
    """The same supported simulator contract for global and hero assignments."""
    if runtime != "codex":
        raise ValueError(
            f"{label}.runtime must be 'codex': Tinker training requires the confined Codex GM."
        )
    if provider != "codex-cli":
        raise ValueError(
            f"{label}.provider must be 'codex-cli' for the confined Codex GM."
        )
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{label}.model must be non-empty.")
    if model != model.strip() or any(
        ord(character) < 32 or ord(character) == 127 for character in model
    ):
        raise ValueError(
            f"{label}.model must not contain whitespace padding or control characters."
        )
    if reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"{label}.reasoning_effort must be one of {REASONING_EFFORTS!r}."
        )


def load_train_config(
    path: Path | None,
    *,
    log_dir: Path | None = None,
    campaign_mode: str = "tinker_train",
) -> TrainConfig:
    """Read TOML, apply ``--log-dir``, generate only an omitted ID, and validate.

    ``run_id_generated`` lets resume restore the saved ID instead of treating
    the new invocation's timestamp as a user-requested identity change.
    """
    raw: dict[str, Any] = (
        tomllib.loads(path.read_text(encoding="utf-8")) if path is not None else {}
    )
    overrides = _world_overrides(raw.pop("world_overrides", {}))
    unknown = sorted(set(raw) - set(_SECTION_TYPES))
    if unknown:
        raise ValueError(f"Unknown section(s): {', '.join(unknown)}")
    config = TrainConfig(
        **{
            name: _section(cls, raw.get(name, {}), name)
            for name, cls in _SECTION_TYPES.items()
        }
    )
    if config.compute.backend == "mlx":
        adaptive_raw = raw.get("adaptive", {})
        config = replace(
            config,
            adaptive=replace(
                config.adaptive,
                parallelism=adaptive_raw.get("parallelism", 2),
                parallelism_per_provider=adaptive_raw.get(
                    "parallelism_per_provider", 2
                ),
            ),
        )
    run = config.run
    generated = "run_id" not in raw.get("run", {})
    if generated:
        run = replace(
            run,
            run_id=datetime.now(UTC).strftime("run-%Y%m%dT%H%M%SZ-")
            + uuid.uuid4().hex[:12],
        )
    elif not run.run_id.strip():
        raise ValueError("[run].run_id must be non-empty when explicitly set.")
    if log_dir is not None:
        run = replace(run, log_dir=str(log_dir))
    config = replace(
        config, run=run, run_id_generated=generated, world_overrides=overrides
    )
    validate_train_config(config, campaign_mode=campaign_mode)
    return config
