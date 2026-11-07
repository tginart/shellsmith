"""Live episode state machine for the Player and the Game Master.

Architecture map:
    PREPARED
      -> boot GM with the universal world-simulator instructions, complete
         private world file, and path to an empty vat
      -> Game Master writes the vat's only initial file (`AGENTS.md`) and
         ``world/bootstrap/response.json`` (private summary + commitments)
      -> orchestrator imputes boot success from those paths/structures; Player's
         first prompt comes from the world file Player section
      -> start the Node Pi-SDK runner in the ephemeral vat
      -> RUNNING: Pi tool executors suspend while Python obtains observations
         from the persistent Game Master
      -> JUDGING: resume the same GM thread with the terminal rubric
      -> COMPLETE or UNTRAINABLE

    Any process, contract, timeout, privacy, or ambiguous-delivery failure moves
    the episode to FAILED. An immersion failure discovered by honest Player
    behavior moves it to UNTRAINABLE without assigning that agent blame.

Why this exists:
    The intelligent behavior belongs inside the two agents. This module owns
    only ordering, isolation, durability, budgets, and explicit failure
    semantics. It never decides what an environment action should return.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from shellsmith.config import DEFAULT_MAX_WORLD_STEPS, DEFAULT_PLAYER_THINKING_RENDERER
from shellsmith.types import JsonValue
from shellsmith.episode.bridge import (
    PLACEHOLDER_TOKEN_RE,
    VatWorldToolHandler,
    WorldObservation,
    WorldBridge,
)
from shellsmith.world_file import (
    WorldFileSpec,
)
from shellsmith.episode.consistency_ledger import (
    check_gm_lists,
    create_ledger,
    load_ledger,
)
from shellsmith.episode.failure import (
    serialize_failure,
)
from shellsmith.episode.progress import (
    emit_simple,
)
from shellsmith.episode.gm_process import (
    ProcessTurn,
    ResumableAgentProcess,
    WorldTurnContext,
)
from shellsmith.episode.schemas import (
    BOOT_RESPONSE_SCHEMA,
    FINAL_RESPONSE_SCHEMA,
    TURN_RESPONSE_SCHEMA,
)
from shellsmith.episode.vat_process import (
    PiSdkVatProcess,
    VatPromptResult,
    VatToolHandler,
)
from shellsmith.episode.vat_redaction import (
    redact_private_paths,
)
from shellsmith.episode.world_artifacts import (
    BootResponseArtifact,
    FinalResponseArtifact,
    load_boot_response,
    load_final_response,
)
from shellsmith.episode.world_prompt_profile import (
    FrozenPromptInputs,
    WorldPromptProfile,
    render_world_agents,
)


# Prompts live at the shellsmith package root (sibling of episode/).
PROMPT_ROOT = Path(__file__).resolve().parent.parent.joinpath("prompts")


def _thinking_mode_for_effort(effort: object) -> str:
    """Resolve the same effort/mode contract used by sampling and auditing."""
    from shellsmith.policy.reasoning import thinking_mode_for_effort

    return thinking_mode_for_effort(effort)



def _terminal_trainability(
    artifact: FinalResponseArtifact,
    observations: tuple[WorldObservation, ...],
) -> tuple[bool, tuple[str, ...]]:
    """Combine every simulator-integrity signal into the terminal disposition.

    Main callers:
        ``EpisodeOrchestrator.judge`` after final artifact validation.

    A reward estimate can remain useful when integrity fails, but the rollout
    must not enter training. Nonempty issues are authoritative even if the model
    accidentally leaves a neighboring ``trainable`` boolean set to true.
    """
    turn_issues = [
        issue
        for observation in observations
        for issue in observation.integrity.issues
    ]
    turn_integrity = all(
        observation.integrity.trainable and not observation.integrity.issues
        for observation in observations
    )
    issues = list(artifact.integrity_issues)
    issues.extend(issue for issue in turn_issues if issue not in issues)
    if not turn_integrity and not issues:
        issues.append("One or more world turns were marked untrainable.")
    factual_integrity = (
        artifact.factual_consistency_passed
        and not artifact.factual_contradictions
    )
    trainable = (
        artifact.trainable
        and turn_integrity
        and not issues
        and factual_integrity
    )
    return trainable, tuple(issues)


class EpisodeStatus(str, Enum):
    """Durable lifecycle states for one generative environment episode."""

    CREATED = "created"
    PREPARED = "prepared"
    WORLD_READY = "world_ready"
    RUNNING = "running"
    JUDGING = "judging"
    COMPLETE = "complete"
    UNTRAINABLE = "untrainable"
    FAILED = "failed"


@dataclass(frozen=True)
class EpisodePaths:
    """Every durable filesystem boundary owned by one episode.

    All path fields are ``pathlib.Path`` objects, not strings.
    """

    root: Path
    scenario_dir: Path
    world_dir: Path
    simulation_root: Path
    runtime_scratch: Path
    request_dir: Path
    log_dir: Path
    schema_dir: Path
    turns_dir: Path
    bootstrap_dir: Path
    final_dir: Path
    state_path: Path
    player_agent_prompt_path: Path
    world_boot_result_path: Path
    bootstrap_response_path: Path
    consistency_ledger_path: Path
    vat_workspace_audit_path: Path
    final_request_path: Path
    final_response_path: Path

    @staticmethod
    def from_root(root: Path) -> "EpisodePaths":
        """Derive paths without creating them; preparation owns filesystem writes."""
        return EpisodePaths(
            root=root,
            scenario_dir=root.joinpath("scenario"),
            world_dir=root.joinpath("world"),
            simulation_root=root.joinpath("world", "simulation_root"),
            runtime_scratch=root.joinpath("world", "runtime_scratch"),
            request_dir=root.joinpath("requests"),
            log_dir=root.joinpath("logs"),
            schema_dir=root.joinpath("schemas"),
            turns_dir=root.joinpath("world", "turns"),
            bootstrap_dir=root.joinpath("world", "bootstrap"),
            final_dir=root.joinpath("world", "final"),
            state_path=root.joinpath("episode.json"),
            player_agent_prompt_path=root.joinpath(
                "scenario", "player_agent_prompt.md"
            ),
            world_boot_result_path=root.joinpath(
                "scenario", "world_boot_result.json"
            ),
            bootstrap_response_path=root.joinpath(
                "world", "bootstrap", "response.json"
            ),
            consistency_ledger_path=root.joinpath(
                "world", "consistency_ledger.json"
            ),
            vat_workspace_audit_path=root.joinpath(
                "scenario", "vat_workspace_audit.json"
            ),
            final_request_path=root.joinpath("world", "final", "request.json"),
            final_response_path=root.joinpath("world", "final", "response.json"),
        )


@dataclass(frozen=True)
class EpisodeState:
    """Minimal durable cursor required to recover an interrupted episode."""

    episode_id: str
    world_id: str
    world_hash: str
    status: EpisodeStatus
    vat_session_id: str | None = None
    world_thread_id: str | None = None
    next_action_index: int = 0
    last_player_message_index: int = 0
    pending_action_idempotency_key: str | None = None
    failure: str | None = None
    failure_detail: dict[str, JsonValue] | None = None

    def to_jsonable(self) -> dict[str, JsonValue]:
        return {
            "episode_id": self.episode_id,
            "world_id": self.world_id,
            "world_hash": self.world_hash,
            "status": self.status.value,
            "vat_session_id": self.vat_session_id,
            "world_thread_id": self.world_thread_id,
            "next_action_index": self.next_action_index,
            "last_player_message_index": self.last_player_message_index,
            "pending_action_idempotency_key": self.pending_action_idempotency_key,
            "failure": self.failure,
            "failure_detail": self.failure_detail,
        }


@dataclass(frozen=True)
class TerminalJudgement:
    """Private terminal reward, simulator-integrity result, and trainability gate."""

    final_reward: float
    completion_percentage: float
    trainable: bool
    criteria: tuple[dict[str, JsonValue], ...]
    process_deductions: tuple[dict[str, JsonValue], ...]
    commentary: str
    integrity_issues: tuple[str, ...]
    factual_consistency_passed: bool
    factual_contradictions: tuple[str, ...]
    task_outcomes: dict[str, JsonValue]
    response_path: Path
    raw_log_path: Path


@dataclass(frozen=True)
class EpisodeResult:
    """Final output of one fully audited episode."""

    state: EpisodeState
    judgement: TerminalJudgement | None
    player_transcript: tuple[dict[str, JsonValue], ...]
    world_observations: tuple[WorldObservation, ...]


class EpisodeOrchestrator:
    """Coordinator for one executable vat/world episode.

    Main callers:
        The no-training smoke CLI, followed later by rollout fanout and training
        adapters.

    Stage boundary:
        This class is intentionally not an agent. It may format prompts, validate
        structured events, persist state, and enforce budgets. It must not invent
        tool observations, repair simulator outputs, or assign reward itself.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        spec: WorldFileSpec,
        paths: EpisodePaths,
        vat_process: PiSdkVatProcess,
        gm_agent_process: ResumableAgentProcess,
        prompt_root: Path = PROMPT_ROOT,
        world_prompt_profile: WorldPromptProfile | None = None,
        prompt_inputs: FrozenPromptInputs | None = None,
        max_world_steps: int = DEFAULT_MAX_WORLD_STEPS,
        max_artifact_attempts: int = 3,
        audit_metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.spec = spec
        self.paths = paths
        self.vat_process = vat_process
        self.gm_agent_process = gm_agent_process
        self.prompt_root = prompt_root
        self.prompt_inputs = prompt_inputs
        self.world_prompt_profile = world_prompt_profile or WorldPromptProfile(
            "canonical", None
        )
        self.max_world_steps = max_world_steps
        if type(max_artifact_attempts) is not int or max_artifact_attempts < 1:
            raise ValueError("max_artifact_attempts must be positive.")
        self.max_artifact_attempts = max_artifact_attempts
        self.audit_metadata = audit_metadata
        self._vat_prompt_result: VatPromptResult | None = None
        self._tool_handler: VatWorldToolHandler | None = None
        self._latest_state: EpisodeState | None = None


    async def run(self) -> EpisodeResult:
        """Run one episode from empty directories through judgement and cleanup.

        Main entry point for one full episode.

        Steps (in order):
            1. prepare_episode — create on-disk layout and save the task spec.
            2. bootstrap_world — ask the Game Master to build the scenario.
            3. start_player — start the Node vat, run Pi, and route each
               Pi tool call through WorldBridge back to the Game Master.
            4. judge_episode — score the rollout and set COMPLETE or UNTRAINABLE.
            5. _cleanup — stop world/Pi processes and delete the vat workspace.

        On success, returns an EpisodeResult with state, judgement, and transcripts.

        On any error: write FAILED to episode state, run cleanup, then re-raise.
        Errors must not be hidden as fake tool results or a default reward.
        """
        # EpisodeState: durable cursor for this episode (status, session ids, etc.).
        state: EpisodeState = EpisodeState(
            episode_id=self.episode_id,
            world_id=self.spec.id,
            world_hash=self.spec.content_hash(),
            status=EpisodeStatus.CREATED,
        )
        emit_simple(
            episode_id=self.episode_id,
            phase="episode_start",
            message="episode starting",
            world_id=self.spec.id,
        )
        failure_stage = "prepare_episode"
        try:
            state = await self.prepare_episode()
            failure_stage = "world_bootstrap"
            emit_simple(
                episode_id=self.episode_id,
                phase="boot_start",
                message="GM bootstrap starting",
                world_id=self.spec.id,
            )
            # bootstrap_world also returns ProcessTurn (gm_process/ (package)); run() only
            # needs the updated EpisodeState (world_thread_id, WORLD_READY status).
            state, _ = await self.bootstrap_world(state)
            if state.world_thread_id is None:
                raise RuntimeError("World bootstrap completed without a thread ID.")
            emit_simple(
                episode_id=self.episode_id,
                phase="boot_done",
                message="GM bootstrap ready",
                world_id=self.spec.id,
            )
            # WorldBridge (bridge.py) turns one Player tool call into a world
            # turn: write request.json, resume the world session, validate
            # response.json, append the consistency ledger, return an observation.
            bridge: WorldBridge = WorldBridge(
                gm_agent_process=self.gm_agent_process,
                world_session_id=state.world_thread_id,
                turns_dir=self.paths.turns_dir,
                turn_prompt_template_path=self.prompt_root.joinpath(
                    "WORLD_TURN_PROMPT.md"
                ),
                turn_response_schema_path=self.paths.schema_dir.joinpath(
                    "world_turn_response.schema.json"
                ),
                simulation_root=self.paths.simulation_root,
                consistency_ledger_path=self.paths.consistency_ledger_path,
                private_path_replacements=self._private_path_replacements(),
                prompt_inputs=self.prompt_inputs,
                max_artifact_attempts=self.max_artifact_attempts,
            )
            # VatWorldToolHandler (bridge.py): Pi-facing adapter that forwards
            # each suspended vat tool call through WorldBridge.
            handler: VatWorldToolHandler = VatWorldToolHandler(
                episode_id=self.episode_id,
                bridge=bridge,
                max_world_steps=self.max_world_steps,
            )
            self._tool_handler = handler
            emit_simple(
                episode_id=self.episode_id,
                phase="agent_start",
                message="Player/Pi Player starting",
                world_id=self.spec.id,
            )
            # VatPromptResult (vat_process.py): settled Pi run (events, final text).
            vat_result: VatPromptResult
            failure_stage = "player"
            state, vat_result = await self.start_player(
                state, tool_handler=handler
            )
            emit_simple(
                episode_id=self.episode_id,
                phase="judge_start",
                message="terminal GM judging starting",
                world_id=self.spec.id,
            )
            # TerminalJudgement: private final reward + trainability from judge_episode.
            judgement: TerminalJudgement
            failure_stage = "world_judgement"
            state, judgement = await self.judge_episode(state)
            emit_simple(
                episode_id=self.episode_id,
                phase="judge_done",
                message=(
                    f"status={state.status.value} "
                    f"reward={judgement.final_reward} trainable={judgement.trainable}"
                ),
                world_id=self.spec.id,
                reward=judgement.final_reward,
                trainable=judgement.trainable,
                status=state.status.value,
            )
            transcript: tuple[dict[str, JsonValue], ...] = tuple(
                {
                    "event_type": event.event_type,
                    "payload": event.payload,
                }
                for event in vat_result.events
            )
            result: EpisodeResult = EpisodeResult(
                state=state,
                judgement=judgement,
                player_transcript=transcript,
                world_observations=tuple(handler.observations),
            )
            failure_stage = "finalize_episode"
            await self._finalize_completed_episode()
            emit_simple(
                episode_id=self.episode_id,
                phase="episode_done",
                message=f"episode finished status={state.status.value}",
                world_id=self.spec.id,
                status=state.status.value,
            )
            return result
        except BaseException as error:
            provider_stage = failure_stage in {"world_bootstrap", "world_judgement"}
            failure_detail = serialize_failure(
                error,
                stage=failure_stage,
                provider_attributable=provider_stage,
            )
            emit_simple(
                episode_id=self.episode_id,
                phase="episode_error",
                message=str(failure_detail["error"]),
                world_id=self.spec.id,
                failure_stage=failure_detail["failure_stage"],
                failure_component=failure_detail["failure_component"],
                provider_attributable=failure_detail["provider_attributable"],
                timeout_seconds=failure_detail.get("timeout_seconds"),
            )
            state = self._latest_state or state
            terminal_result_is_durable = state.status in (
                EpisodeStatus.COMPLETE,
                EpisodeStatus.UNTRAINABLE,
            )
            if self.paths.root.exists() and not terminal_result_is_durable:
                # dataclasses.replace: mark the durable cursor FAILED.
                failed: EpisodeState = replace(
                    state,
                    status=EpisodeStatus.FAILED,
                    failure=str(failure_detail["error"]),
                    failure_detail=failure_detail,
                )
                self._persist_state(failed)
            await self._cleanup(strict=False)
            raise

    async def prepare_episode(self) -> EpisodeState:
        """Create durable private paths and persist the pinned spec before boot."""
        if self.paths.root.exists() and any(self.paths.root.iterdir()):
            raise RuntimeError(
                f"Episode root must begin empty: {self.paths.root.resolve()}."
            )
        self.paths.root.mkdir(parents=True, exist_ok=True)
        for path in (
            self.paths.scenario_dir,
            self.paths.world_dir,
            self.paths.simulation_root,
            self.paths.runtime_scratch,
            self.paths.request_dir,
            self.paths.log_dir,
            self.paths.schema_dir,
            self.paths.turns_dir,
            self.paths.bootstrap_dir,
            self.paths.final_dir,
        ):
            path.mkdir(parents=True, exist_ok=False)
        self.vat_process.config.workspace.assert_empty()
        if self.audit_metadata is not None:
            self._write_json(
                self.paths.root.joinpath("run_config.json"), self.audit_metadata
            )
            # Pin the expected Player/Tinker wire protocol for acceptance.
            self._write_json(
                self.paths.scenario_dir.joinpath("model_protocol.json"),
                {
                    "schema": "shellsmith.model_protocol.v1",
                    "expected_renderer": self.audit_metadata.get(
                        "player_renderer",
                        self.audit_metadata.get("renderer_name", DEFAULT_PLAYER_THINKING_RENDERER),
                    ),
                    "expected_thinking_mode": _thinking_mode_for_effort(
                        self.audit_metadata.get("player_thinking")
                    ),
                    "api_shape": "native_tools",
                    "allow_tools_in_tinker_text": False,
                    **({"thinking_token_budget": self.audit_metadata["player_reasoning_tokens"]}
                       if "player_reasoning_tokens" in self.audit_metadata else {}),
                    **({"response_token_budget": self.audit_metadata["player_response_tokens"]}
                       if "player_response_tokens" in self.audit_metadata else {}),
                    # Populated after the episode from proxy rollout_records.
                    "requests": [],
                },
            )
        agents_md_path: Path = self.paths.root.joinpath("AGENTS.md")
        # render_world_agents comes from world_prompt_profile.py. It loads the
        # canonical world-simulator AGENTS.md and optionally appends a model-
        # specific tail (e.g. Sonnet / Qwen difficulty notes).
        agents_md_path.write_text(
            render_world_agents(
                canonical_path=self.prompt_root.joinpath(
                    "game_master", "AGENTS.md"
                ),
                profile=self.world_prompt_profile,
                prompt_inputs=self.prompt_inputs,
            ),
            encoding="utf-8",
        )
        world_path: Path = self.paths.scenario_dir.joinpath("world_file.md")
        world_path.write_text(
            self.spec.markdown,
            encoding="utf-8",
        )
        # BOOT/TURN/FINAL_RESPONSE_SCHEMA are JSON-schema dicts imported from
        # schemas.py. World CLI adapters use them for structured outputs.
        self._write_json(
            self.paths.schema_dir.joinpath("world_boot_response.schema.json"),
            BOOT_RESPONSE_SCHEMA,
        )
        self._write_json(
            self.paths.schema_dir.joinpath("world_turn_response.schema.json"),
            TURN_RESPONSE_SCHEMA,
        )
        self._write_json(
            self.paths.schema_dir.joinpath("world_final_response.schema.json"),
            FINAL_RESPONSE_SCHEMA,
        )
        state = EpisodeState(
            episode_id=self.episode_id,
            world_id=self.spec.id,
            world_hash=self.spec.content_hash(),
            status=EpisodeStatus.PREPARED,
        )
        self._persist_state(state)
        return state

    async def bootstrap_world(
        self, state: EpisodeState
    ) -> tuple[EpisodeState, ProcessTurn]:
        """Boot the Game Master and prove the vat is AGENTS.md-only."""
        self._require_status(state, EpisodeStatus.PREPARED)

        # -----------------------------------------------------------------
        # Build the first user message for the Game Master.
        #
        # prepare_episode already wrote standing instructions into the
        # episode-root AGENTS.md. This prompt is the first *turn*: it gives
        # the world agent the private world file, the empty vat path, the
        # private simulation_root, and the absolute path for
        # world/bootstrap/response.json.
        # -----------------------------------------------------------------
        vat_workspace_path: str = str(
            self.vat_process.config.workspace.workspace_dir.resolve()
        )
        simulation_root_path: str = str(self.paths.simulation_root.resolve())
        bootstrap_response_path: str = str(
            self.paths.bootstrap_response_path.resolve()
        )
        prompt: str = self._render_prompt(
            self.prompt_root.joinpath("WORLD_BOOTSTRAP_PROMPT.md"),
            {
                "{{EPISODE_ID}}": self.episode_id,
                "{{WORLD_MARKDOWN}}": self.spec.markdown,
                "{{VAT_WORKSPACE_PATH}}": vat_workspace_path,
                "{{SIMULATION_ROOT_PATH}}": simulation_root_path,
                "{{BOOT_RESPONSE_PATH}}": bootstrap_response_path,
                "{{BOOT_RESPONSE_SCHEMA_PATH}}": str(
                    self.paths.schema_dir.joinpath(
                        "world_boot_response.schema.json"
                    ).resolve()
                ),
            },
        )
        boot_prompt_path: Path = self.paths.request_dir.joinpath(
            "world_boot_prompt.md"
        )
        boot_prompt_path.write_text(prompt, encoding="utf-8")

        # -----------------------------------------------------------------
        # Launch a brand-new world session and wait for this boot turn.
        #
        # Artifact-first: adapters must not pass Claude --json-schema or
        # Codex --output-schema. The orchestrator accepts boot only when
        # world/bootstrap/response.json validates and the vat is AGENTS-only.
        #
        # ProcessTurn (from gm_process/) is the adapter's return value for
        # one completed start/resume. It is a small frozen dataclass with:
        #   - session_id: durable world-thread id for later resume/judge
        #   - events: normalized audit events from this CLI invocation
        #   - final_response: dict (may be empty when artifact-first)
        #   - raw_log_path: path to the durable stdout/stderr audit log
        # -----------------------------------------------------------------
        boot_turn_context: WorldTurnContext = WorldTurnContext(
            stage="bootstrap",
            expected_artifact_path=self.paths.bootstrap_response_path,
            idempotency_key=f"{self.episode_id}:bootstrap",
            output_schema_path=None,
        )
        turn: ProcessTurn = await self.gm_agent_process.start(
            prompt,
            turn_context=boot_turn_context,
        )

        # -----------------------------------------------------------------
        # Check that boot actually worked.
        #
        # Success is computed from disk only (no model status field):
        #   1. world/bootstrap/response.json at the expected path with correct
        #      structure (private_world_summary + private_initial_commitments)
        #   2. vat contains only AGENTS.md at the known path
        #
        # If either fails, we do not start over — we resume the same world
        # thread with a repair prompt (up to three attempts total).
        # -----------------------------------------------------------------
        boot_errors: list[str] = []
        artifact: BootResponseArtifact | None = None
        for attempt in range(1, self.max_artifact_attempts + 1):
            try:
                candidate = self._validate_boot_turn()
                # create_ledger (below) enforces these rules after the loop.
                check_gm_lists(private_initial_commitments=list(candidate.private_initial_commitments))
                artifact = candidate
                break
            except RuntimeError as error:
                boot_errors.append(str(error))
                if attempt == self.max_artifact_attempts:
                    raise RuntimeError(
                        f"World boot remained invalid after {self.max_artifact_attempts} same-thread "
                        f"attempts: {' | '.join(boot_errors)}"
                    ) from error
                repair_prompt: str = (
                    "Your previous boot turn completed but failed orchestration "
                    "validation. This is a repair of the same boot, not a new "
                    "episode. Inspect the vat again, correct its AGENTS.md-only "
                    "public projection if needed, and rewrite the complete slim "
                    f"boot JSON to `{self.paths.bootstrap_response_path.resolve()}`. "
                    "Required fields are only private_world_summary and "
                    "private_initial_commitments. Do not begin the task. Do not "
                    "report status/ready/failed — the orchestrator decides pass "
                    "or fail from the files on disk.\n\n"
                    f"Validation error: {error}"
                )
                repair_path: Path = self.paths.request_dir.joinpath(
                    f"world_boot_repair_{attempt:02d}.md"
                )
                repair_path.write_text(repair_prompt + "\n", encoding="utf-8")
                # Same ProcessTurn shape as start(); session_id must match.
                turn = await self.gm_agent_process.resume(
                    turn.session_id,
                    repair_prompt,
                    output_schema_path=None,
                    turn_context=WorldTurnContext(
                        stage=f"bootstrap_repair_{attempt:02d}",
                        expected_artifact_path=self.paths.bootstrap_response_path,
                        reuse_valid_artifact=False,
                        idempotency_key=(
                            f"{self.episode_id}:bootstrap:repair:{attempt:02d}"
                        ),
                        output_schema_path=None,
                    ),
                )

        if artifact is None:
            raise RuntimeError(
                "Internal error: bootstrap_world exited the repair loop without "
                "a validated BootResponseArtifact."
            )

        # -----------------------------------------------------------------
        # Freeze the public boot artifacts for the rest of the episode.
        #
        # Critical: Player's first prompt is taken from the world file parser
        # (self.spec.player), not from the world model. We also
        # snapshot vat AGENTS.md, save the slim boot JSON audit copy, and
        # open the consistency ledger with the model's private boot
        # commitments (later turns must review that ledger; they cannot
        # edit it directly).
        # -----------------------------------------------------------------
        # pathlib.Path to the live vat AGENTS.md (not a string).
        expected_agents_path: Path = (
            self.vat_process.config.workspace.agents_path.resolve()
        )
        self.paths.player_agent_prompt_path.write_text(
            self.spec.player.strip() + "\n", encoding="utf-8"
        )
        # Copy vat AGENTS.md into the episode audit folder (also pathlib.Path).
        vat_agents_audit_path: Path = self.paths.scenario_dir.joinpath(
            "vat_AGENTS.md"
        )
        vat_agents_text: str = expected_agents_path.read_text(encoding="utf-8")
        vat_agents_audit_path.write_text(vat_agents_text, encoding="utf-8")
        self._write_json(
            self.paths.world_boot_result_path,
            {
                "private_world_summary": artifact.private_world_summary,
                "private_initial_commitments": list(
                    artifact.private_initial_commitments
                ),
            },
        )
        # create_ledger is imported from consistency_ledger.py. It writes a new
        # append-only world/consistency_ledger.json seeded with the boot
        # commitments the world agent declared (facts later turns must respect).
        create_ledger(
            self.paths.consistency_ledger_path,
            boot_commitments=artifact.private_initial_commitments,
        )

        # -----------------------------------------------------------------
        # Mark the episode WORLD_READY. Pi has not started yet; that happens
        # in start_player, which sends the prompt we just wrote.
        #
        # EpisodeState is the durable episode cursor (status, world_thread_id,
        # vat_session_id, …).
        # replace(...) here is dataclasses.replace (stdlib), not str.replace:
        # it copies the frozen EpisodeState and overlays the named fields.
        # -----------------------------------------------------------------
        ready: EpisodeState = replace(
            state,
            status=EpisodeStatus.WORLD_READY,
            world_thread_id=turn.session_id,
        )
        self._persist_state(ready)
        return ready, turn

    async def start_player(
        self,
        state: EpisodeState,
        *,
        tool_handler: VatToolHandler,
    ) -> tuple[EpisodeState, VatPromptResult]:
        """Launch Pi and run its complete agent loop through the world bridge."""
        self._require_status(state, EpisodeStatus.WORLD_READY)
        session_id = await self.vat_process.launch()
        # dataclasses.replace: copy EpisodeState with new status / vat session.
        running = replace(
            state,
            status=EpisodeStatus.RUNNING,
            vat_session_id=session_id,
        )
        self._persist_state(running)
        prompt = self.paths.player_agent_prompt_path.read_text(encoding="utf-8")
        result = await self.vat_process.prompt(prompt, tool_handler=tool_handler)
        self._vat_prompt_result = result
        # dataclasses.replace: record how far the Pi transcript got.
        settled = replace(
            running,
            last_player_message_index=result.message_count,
        )
        self._persist_state(settled)
        return settled, result

    async def judge_episode(
        self, state: EpisodeState
    ) -> tuple[EpisodeState, TerminalJudgement]:
        """Resume the same world thread and validate its independent final reward.

        Returns:
            Updated EpisodeState (COMPLETE or UNTRAINABLE) and a TerminalJudgement
            scorecard (reward, trainability, integrity, paths to artifacts).
        """
        self._require_status(state, EpisodeStatus.RUNNING)
        if self._vat_prompt_result is None or self._tool_handler is None:
            raise RuntimeError("Player result is absent before judging.")
        if state.world_thread_id is None:
            raise RuntimeError("World thread ID is absent before judging.")

        # EpisodeState cursor moved into JUDGING (dataclasses.replace copy).
        judging: EpisodeState = replace(
            state,
            status=EpisodeStatus.JUDGING,
            next_action_index=self._tool_handler.next_step_index,
        )
        self._persist_state(judging)

        # redact_private_paths comes from vat_redaction.py. It rewrites real
        # host vat paths to stable fake paths (/workspace, /vat) so the final
        # judge prompt never leaks ephemeral filesystem locations.
        budget_exhausted: bool = bool(self._tool_handler.budget_exhausted)
        final_request: dict[str, JsonValue] = {
            "episode_id": self.episode_id,
            "world_path": str(
                self.paths.scenario_dir.joinpath("world_file.md").resolve()
            ),
            "initial_prompt": self.paths.player_agent_prompt_path.read_text(
                encoding="utf-8"
            ),
            "player_final_response": redact_private_paths(
                self._vat_prompt_result.final_text,
                replacements=self._private_path_replacements(),
            ),
            "player_agent_prompt_terminated": self._vat_prompt_result.terminated,
            "world_termination_reason": self._tool_handler.termination_reason,
            "forced_budget_termination": budget_exhausted,
            "max_world_steps": self.max_world_steps,
            "world_turn_count": self._tool_handler.next_step_index,
            "player_events": [
                {
                    "event_type": event.event_type,
                    "payload": redact_private_paths(
                        event.payload,
                        replacements=self._private_path_replacements(),
                    ),
                }
                for event in self._vat_prompt_result.events
            ],
            "turn_directories": [
                str(path.resolve())
                for path in sorted(self.paths.turns_dir.iterdir())
                if path.is_dir()
            ],
            "consistency_ledger": self._ledger_reference(),
        }
        self._write_json(self.paths.final_request_path, final_request)

        final_response_schema_path: Path = self.paths.schema_dir.joinpath(
            "world_final_response.schema.json"
        )
        # WORLD_FINAL_PROMPT.md is the shared final-judge brief. On budget cut,
        # wrap it in a meta notice so GM knows Player was force-stopped
        # after N steps (no further Player turn) and must judge now.
        judge_brief: str = self._render_prompt(
            self.prompt_root.joinpath("WORLD_FINAL_PROMPT.md"),
            {
                "{{EPISODE_ID}}": self.episode_id,
                "{{FINAL_REQUEST_PATH}}": str(self.paths.final_request_path.resolve()),
                "{{FINAL_RESPONSE_PATH}}": str(
                    self.paths.final_response_path.resolve()
                ),
                "{{FINAL_RESPONSE_SCHEMA_PATH}}": str(
                    final_response_schema_path.resolve()
                ),
            },
        )
        prompt: str
        if budget_exhausted:
            prompt = (
                "# Forced episode termination — world-tool budget exhausted\n\n"
                f"Player (the Player) completed "
                f"{self._tool_handler.next_step_index} world-resolved tool "
                f"step(s), which is the configured maximum of "
                f"{self.max_world_steps}. The orchestrator has forcibly "
                "terminated the Player/Pi loop. Player will not receive another "
                "prompt or tool round.\n\n"
                "Your job now is final judgment of this forcibly terminated "
                "episode. Treat the trajectory as complete for scoring purposes "
                "even if Player did not emit a natural closing answer.\n\n"
                "The private final request "
                f"(`{self.paths.final_request_path.resolve()}`) sets "
                "`forced_budget_termination` to true and records "
                "`max_world_steps` / `world_turn_count`.\n\n"
                "---\n\n"
                "## Final judge brief\n\n"
                f"{judge_brief}"
            )
        else:
            prompt = judge_brief
        final_prompt_path: Path = self.paths.request_dir.joinpath(
            "world_final_prompt.md"
        )
        final_prompt_path.write_text(prompt, encoding="utf-8")

        # ProcessTurn (gm_process/ (package)): one completed world resume (session_id,
        # events, final_response dict, raw_log_path).
        turn: ProcessTurn = await self.gm_agent_process.resume(
            state.world_thread_id,
            prompt,
            output_schema_path=None,
        )

        # FinalResponseArtifact (world_artifacts.py): typed parse of
        # world/final/response.json (reward, criteria, integrity flags, etc.).
        artifact: FinalResponseArtifact | None = None
        final_errors: list[str] = []
        for attempt in range(1, self.max_artifact_attempts + 1):
            try:
                # load_final_response comes from world_artifacts.py.
                artifact = load_final_response(self.paths.final_response_path)
                break
            except RuntimeError as error:
                final_errors.append(str(error))
                if attempt == self.max_artifact_attempts:
                    raise RuntimeError(
                        f"Final world artifact remained invalid after {self.max_artifact_attempts} "
                        f"same-thread attempts: {' | '.join(final_errors)}"
                    ) from error
                repair_prompt: str = self._render_artifact_repair_prompt(
                    stage="final judgment",
                    request_path=self.paths.final_request_path,
                    response_path=self.paths.final_response_path,
                    validation_error=str(error),
                )
                final_repair_path: Path = self.paths.final_dir.joinpath(
                    f"repair_{attempt:02d}_prompt.md"
                )
                final_repair_path.write_text(repair_prompt, encoding="utf-8")
                turn = await self.gm_agent_process.resume(
                    state.world_thread_id,
                    repair_prompt,
                    output_schema_path=None,
                )
        if artifact is None:
            raise RuntimeError("Final world artifact was never loaded.")

        # _terminal_trainability is defined above in this file. It merges final
        # artifact flags with per-turn integrity issues into one trainable bool.
        trainable: bool
        issues: tuple[str, ...]
        trainable, issues = _terminal_trainability(
            artifact,
            tuple(self._tool_handler.observations),
        )

        # TerminalJudgement: private scorecard packed for EpisodeResult / training.
        judgement: TerminalJudgement = TerminalJudgement(
            final_reward=artifact.final_reward,
            completion_percentage=artifact.completion_percentage,
            trainable=trainable,
            criteria=artifact.criteria,
            process_deductions=artifact.process_deductions,
            commentary=artifact.commentary,
            integrity_issues=issues,
            factual_consistency_passed=artifact.factual_consistency_passed,
            factual_contradictions=artifact.factual_contradictions,
            task_outcomes=artifact.task_outcomes,
            response_path=artifact.path,
            raw_log_path=turn.raw_log_path,
        )

        # EpisodeState cursor moved into COMPLETE or UNTRAINABLE.
        terminal: EpisodeState = replace(
            judging,
            status=(
                EpisodeStatus.COMPLETE if trainable else EpisodeStatus.UNTRAINABLE
            ),
        )
        self._persist_state(terminal)
        return terminal, judgement

    async def recover(self) -> EpisodeState:
        """Recover only from an unambiguous persisted cursor and idempotent tool result."""
        raise NotImplementedError("Crash recovery is defined but not implemented.")

    @staticmethod
    def _require_status(state: EpisodeState, expected: EpisodeStatus) -> None:
        if state.status is not expected:
            raise RuntimeError(
                f"Episode must be {expected.value}, got {state.status.value}."
            )

    def _render_prompt(self, path: Path, replacements: dict[str, str]) -> str:
        prompt = (self.prompt_inputs.read_text(path) if self.prompt_inputs is not None
                  else path.read_text(encoding="utf-8"))
        for placeholder, value in replacements.items():
            # str.replace (builtin), not dataclasses.replace.
            prompt = prompt.replace(placeholder, value)
        # PLACEHOLDER_TOKEN_RE comes from bridge.py. It matches leftover
        # {{LIKE_THIS}} tokens so we fail loudly if a template was incomplete.
        unresolved = PLACEHOLDER_TOKEN_RE.search(prompt)
        if unresolved:
            raise RuntimeError(
                f"Prompt contains an unresolved placeholder {unresolved.group(0)}: {path}."
            )
        return prompt

    def _persist_state(self, state: EpisodeState) -> None:
        self._write_json(self.paths.state_path, state.to_jsonable(), atomic=True)
        self._latest_state = state

    def _ledger_reference(self) -> dict[str, JsonValue]:
        """Pin the exact append-only history consumed by terminal judging.

        Main callers:
            ``judge_episode`` while constructing its private final request.

        Why this exists:
            A path alone does not prove which ledger revision the final judge saw.
            The revision and byte digest make that boundary explicit and auditable.
        """
        # load_ledger comes from consistency_ledger.py. It reads and validates
        # the on-disk append-only ledger, returning a typed ledger object.
        ledger = load_ledger(self.paths.consistency_ledger_path)
        return {
            "path": str(self.paths.consistency_ledger_path.resolve()),
            "revision": ledger.revision,
            "sha256": hashlib.sha256(
                self.paths.consistency_ledger_path.read_bytes()
            ).hexdigest(),
        }

    def _private_path_replacements(self) -> tuple[tuple[str, str], ...]:
        """Map real ephemeral vat locations to stable world-facing paths.

        Main callers:
            Per-turn bridge construction and terminal-request serialization.

        Why this exists:
            Pi needs the real cwd to execute, but Sol must never receive that host
            path after boot. Both the workspace and its parent vat root are
            covered so reasoning text and tool arguments share one boundary.
        """
        workspace = self.vat_process.config.workspace
        return (
            (str(workspace.workspace_dir.resolve()), "/workspace"),
            (str(workspace.root.resolve()), "/vat"),
        )

    def _write_vat_workspace_audit(self) -> None:
        """Persist proof that the vat retained its original public instructions.

        Main callers:
            ``_finalize_completed_episode`` after terminal judgment and before
            deleting the ephemeral vat.

        Why this exists:
            The live workspace is intentionally deleted during cleanup. This
            durable record proves the only real context file remained unchanged.
            Extra files are recorded but inert because tool access is virtual.
            A failed audit is persisted as ``passed: false`` so acceptance can
            reject it without rewriting the already-settled episode cursor.
        """
        workspace = self.vat_process.config.workspace
        try:
            workspace.assert_agents_only()
            content = workspace.agents_path.read_bytes()
            agents_sha256 = hashlib.sha256(content).hexdigest()
            payload: dict[str, JsonValue] = {
                "passed": True,
                "entries": [path.name for path in workspace.workspace_entries()],
                "extra_entries": [
                    path.name
                    for path in workspace.workspace_entries()
                    if path.name != "AGENTS.md"
                ],
                "agents_sha256": agents_sha256,
                "agents_unchanged": agents_sha256 == workspace._agents_sha256,
            }
        except Exception as error:
            payload = {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
            self._write_json(self.paths.vat_workspace_audit_path, payload)
            raise
        self._write_json(self.paths.vat_workspace_audit_path, payload)

    async def _finalize_completed_episode(self) -> None:
        """Record post-result problems without invalidating a settled episode.

        Main callers:
            ``run`` after ``judge_episode`` has persisted COMPLETE or
            UNTRAINABLE and constructed the returned ``EpisodeResult``.

        Why this exists:
            Workspace auditing and process teardown are necessary evidence and
            resource management, but they happen after the judged result has
            become durable. Their failures must remain visible to acceptance and
            operators without overwriting that result as FAILED.
        """
        post_result_errors: list[str] = []
        try:
            self._write_vat_workspace_audit()
        except Exception as error:
            post_result_errors.append(
                f"vat_workspace_audit: {type(error).__name__}: {error}"
            )
        try:
            await self._cleanup(strict=True)
        except Exception as error:
            post_result_errors.append(f"cleanup: {type(error).__name__}: {error}")
        if not post_result_errors:
            return
        error_path = self.paths.log_dir.joinpath("post_result_errors.log")
        try:
            error_path.write_text("\n".join(post_result_errors) + "\n", encoding="utf-8")
        except Exception as log_error:
            post_result_errors.append(
                f"post_result_error_log: {type(log_error).__name__}: {log_error}"
            )
        emit_simple(
            episode_id=self.episode_id,
            phase="episode_post_result_warning",
            message="; ".join(post_result_errors),
            world_id=self.spec.id,
            status=(self._latest_state.status.value if self._latest_state else None),
        )

    def _validate_boot_turn(self) -> BootResponseArtifact:
        """Return a typed boot result when disk artifacts prove a successful boot.

        Boot succeeds if and only if both are true:
          1. ``world/bootstrap/response.json`` exists at the expected path and
             parses with the slim two-field structure.
          2. The known vat path contains only a regular ``AGENTS.md`` file.

        The model never reports status/ready/failed. Missing or malformed files
        raise ``RuntimeError`` with an orchestrator-owned message.
        """
        artifact = load_boot_response(self.paths.bootstrap_response_path)
        self.vat_process.config.workspace.assert_agents_only()
        return artifact

    def _render_artifact_repair_prompt(
        self,
        *,
        stage: str,
        request_path: Path,
        response_path: Path,
        validation_error: str,
    ) -> str:
        return self._render_prompt(
            self.prompt_root.joinpath("WORLD_ARTIFACT_REPAIR_PROMPT.md"),
            {
                "{{STAGE}}": stage,
                "{{REQUEST_PATH}}": str(request_path.resolve()),
                "{{RESPONSE_PATH}}": str(response_path.resolve()),
                "{{VALIDATION_ERROR}}": validation_error,
            },
        )

    @staticmethod
    def _write_json(
        path: Path, payload: dict[str, JsonValue], *, atomic: bool = False
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if not atomic:
            path.write_text(text, encoding="utf-8")
            return
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        # pathlib.Path.replace: atomically rename temp file onto the final path.
        temporary.replace(path)

    async def _cleanup(self, *, strict: bool) -> None:
        errors: list[str] = []
        for name, close in (
            ("vat", self.vat_process.close),
            ("gm_agent", self.gm_agent_process.close),
        ):
            try:
                await close()
            except BaseException as error:
                errors.append(f"{name}: {type(error).__name__}: {error}")
        self.vat_process.config.workspace.close()
        if errors:
            if self.paths.log_dir.exists():
                cleanup_log_path: Path = self.paths.log_dir.joinpath(
                    "cleanup_errors.log"
                )
                cleanup_log_path.write_text(
                    "\n".join(errors) + "\n", encoding="utf-8"
                )
            if strict:
                raise RuntimeError("Episode cleanup failed: " + "; ".join(errors))
