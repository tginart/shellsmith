"""Shared types and launcher for world-simulator harness adapters.

Architecture map:
    WorldHarnessLauncher selects Codex; Claude and Pi GM execution fail closed
      -> each harness module implements ResumableAgentProcess
      -> orchestrator / bridge only see ProcessTurn + session ids

The Player vat is intentionally separate (`vat_process.py`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TYPE_CHECKING

from shellsmith.config import (
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
)
from shellsmith.types import JsonValue

if TYPE_CHECKING:
    from shellsmith.episode.gm_process.python_runtime import PythonRuntime


PI_GM_DISABLED_MESSAGE = (
    "Pi GM execution is disabled until filesystem/process confinement and a "
    "credential broker are implemented. Environment filtering alone cannot "
    "protect Player/GM credentials or auth files from unconfined shell/read "
    "tools. Use --gm-runtime codex; the Pi Player SDK remains supported."
)


CLAUDE_GM_DISABLED_MESSAGE = (
    "Claude GM execution is disabled; only the Codex GM is supported. "
    "Use --gm-runtime codex."
)


def validate_gm_runtime(runtime: str) -> None:
    """Reject disabled GM execution before defaults, credentials, or setup.

    Historical runtime metadata and the separate Pi Player are unaffected.
    """
    runtime = runtime.strip().lower()
    if runtime == "pi":
        raise RuntimeError(PI_GM_DISABLED_MESSAGE)
    if runtime == "claude":
        raise RuntimeError(CLAUDE_GM_DISABLED_MESSAGE)


@dataclass(frozen=True)
class WorldTurnContext:
    """Runtime-neutral turn metadata for artifact-first world execution.

    Main callers:
        ``WorldBridge`` and ``EpisodeOrchestrator`` when resuming the
        persistent world thread for one action, boot repair, or final judging.

    Why this exists:
        Provider transport failures must never blindly replay the same prompt.
        The durable ``response.json`` artifact is normally authoritative when
        it is structurally valid. Explicit repair turns set
        ``reuse_valid_artifact`` to false so a semantic validation failure
        reaches the model instead of replaying that same artifact.
    """

    stage: str
    expected_artifact_path: Path | None = None
    reuse_valid_artifact: bool = True
    idempotency_key: str | None = None
    output_schema_path: Path | None = None


@dataclass(frozen=True)
class AgentEvent:
    """One normalized event emitted by either underlying agent runtime."""

    type: str
    payload: dict[str, JsonValue]


@dataclass(frozen=True)
class ProcessTurn:
    """One completed start/resume operation plus its durable audit events."""

    session_id: str
    events: tuple[AgentEvent, ...]
    final_response: dict[str, JsonValue]
    raw_log_path: Path


class ResumableAgentProcess(Protocol):
    """Start/resume protocol used by the persistent world-sim agent thread."""

    async def start(
        self,
        prompt: str,
        *,
        turn_context: WorldTurnContext | None = None,
    ) -> ProcessTurn:
        """Start one durable session and return its ID plus emitted events."""
        ...

    async def resume(
        self,
        session_id: str,
        prompt: str,
        *,
        turn_context: WorldTurnContext | None = None,
        output_schema_path: Path | None = None,
    ) -> ProcessTurn:
        """Run one additional turn in an existing durable session."""
        ...

    async def transcript(self, session_id: str) -> tuple[AgentEvent, ...]:
        """Return the complete durable transcript used for judging and replay."""
        ...

    async def close(self) -> None:
        """Release live subprocess resources without deleting durable state."""
        ...


@dataclass(frozen=True)
class WorldHarnessConfig:
    """Runtime-neutral inputs needed to launch a world-simulator harness."""

    runtime: str
    executable: Path
    model: str
    workspace_dir: Path
    log_dir: Path
    bootstrap_response_schema_path: Path
    additional_writable_dirs: tuple[Path, ...] = ()
    reasoning_effort: str = DEFAULT_GM_REASONING_EFFORT
    timeout_seconds: float = DEFAULT_GM_TURN_TIMEOUT_SECONDS
    api_idle_timeout_seconds: float = DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str | None = None
    api: str = "openai-completions"
    context_window: int = 65_536
    max_tokens: int = 16_384
    provider_retry_delays_seconds: tuple[float, ...] = (1.0, 3.0)
    # Codex only. ``permission_profile`` on shared hosts; ``danger-full-access``
    # inside a disposable container (see ``codex.CODEX_SANDBOX_MODES``).
    codex_sandbox: str = "permission_profile"
    python_runtime: PythonRuntime | None = None


class WorldHarnessLauncher:
    """Construct a resumable adapter without exposing runtime details upstream.

    Main callers:
        The live smoke entry point. Campaigns select a runtime through smoke
        arguments, while the orchestrator receives only ``ResumableAgentProcess``.

    Why this exists:
        Episode logic stays runtime-neutral while execution is Codex-only.
        Historical Claude/Pi metadata does not authorize launching either GM.
    """

    @staticmethod
    def create(config: WorldHarnessConfig) -> ResumableAgentProcess:
        runtime = config.runtime.strip().lower()
        validate_gm_runtime(runtime)
        common = {
            "executable": config.executable,
            "model": config.model,
            "workspace_dir": config.workspace_dir,
            "log_dir": config.log_dir,
            "bootstrap_response_schema_path": config.bootstrap_response_schema_path,
            "additional_writable_dirs": config.additional_writable_dirs,
            "reasoning_effort": config.reasoning_effort,
            "timeout_seconds": config.timeout_seconds,
        }
        if runtime == "codex":
            from shellsmith.episode.gm_process.codex import (
                CodexWorldConfig,
                CodexWorldProcess,
            )

            return CodexWorldProcess(
                CodexWorldConfig(**common, sandbox_mode=config.codex_sandbox,
                                 api_idle_timeout_seconds=config.api_idle_timeout_seconds,
                                 python_runtime=config.python_runtime)
            )
        raise ValueError(f"Unsupported world harness runtime: {config.runtime!r}.")


def normalized_event(raw: dict[str, Any]) -> AgentEvent:
    """Convert a runtime-specific event after an adapter validates its shape.

    Main callers:
        Codex / Claude / Pi JSONL parsers.

    Why this exists:
        The orchestrator needs stable event categories for audit and Player
        deltas, but raw runtime events must remain preserved separately.
    """
    event_type = str(raw.get("type", "")).strip()
    if not event_type:
        raise RuntimeError("Agent event is missing a non-empty type.")
    return AgentEvent(
        type=event_type, payload={str(key): value for key, value in raw.items()}
    )
