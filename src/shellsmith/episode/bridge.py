"""Bridge: turn one Player (Pi) tool call into one world-sim-agent turn.

Plain-English architecture map
------------------------------
Think of two agents talking through this file:

    Player (inside the vat / Node Pi runner)
        "I want to call tool X with these args."
              |
              v
    VatWorldToolHandler  (this file)
        Packages the tool call(s) into a WorldStepEnvelope.
              |
              v
    WorldBridge.step  (this file)
        1. Write turns/000012/request.json   (what Player just did)
        2. Resume the world sim agent (Codex/Claude/Pi CLI)
        3. Wait for turns/000012/response.json
        4. Validate that JSON; repair on same thread if broken
        5. Append a receipt to the consistency ledger
        6. Return a WorldObservation
              |
              v
    Only the Player-facing tool_response goes back into the vat.
    Reward / integrity / commentary stay private (training audit only).

Why this exists:
    The world sim agent is allowed to be smart and messy (scratch files, tools).
    This bridge is dumb and strict: predetermined paths, validate artifacts,
    append ledger, hand Player only what it is allowed to see.


Player (Pi)                Bridge / World                  World sim agent
   |                          |                                  |
   |-- tool call ------------>|                                  |  (1)
   |   (suspended)            |-- request.json + resume -------->|  (2)
   |                          |                                  |
   |                          |<-- response.json ----------------|  (3)
   |                          |-- validate, ledger append -------|  (4)
   |<-- tool_response --------|                                  |  (5)
   |   (Promise resolves)     |                                  |

Arrow → routine map (this file only)
------------------------------------
(1) tool call
      VatWorldToolHandler.handle_sibling_batch
      (or __call__ → handle_sibling_batch)
      Builds WorldStepEnvelope from VatToolRequest(s), then awaits bridge.step.

(2) request.json + resume
      WorldBridge.step (early stages)
      Writes turns/<n>/request.json, fills prompt via _render_turn_prompt,
      then gm_agent_process.resume(...).  (CLI spawn lives in gm_process/ (package).)

(3) response.json
      WorldBridge.step (validate loop)
      load_turn_response(turns/<n>/response.json).  On failure: move bad file
      aside, _render_artifact_repair_prompt, resume again on the same thread.

(4) validate, ledger append
      WorldBridge.step (still; Python-only, no Player/world I/O)
      Revision check + append_entry(consistency_ledger) + pack WorldObservation.
      Return that observation to the handler.

(5) tool_response
      Rest of VatWorldToolHandler.handle_sibling_batch
      Maps WorldObservation → list[VatToolResponse] (Player-facing content only).
      Return value goes back to vat_process → Node → Pi Promise resolves.

Layer split: handler owns (1)+(5); WorldBridge.step owns (2)+(3)+(4).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from shellsmith.types import JsonValue
from shellsmith.config import DEFAULT_MAX_WORLD_STEPS
from shellsmith.episode.world_prompt_profile import FrozenPromptInputs
from shellsmith.episode.consistency_ledger import (
    append_entry,
    check_gm_lists,
    load_ledger,
)
from shellsmith.episode.failure import (
    AttributedExecutionError,
    ExecutionTimeoutError,
)
from shellsmith.episode.progress import (
    TurnTimer,
    emit_simple,
)
from shellsmith.episode.gm_process import (
    ProcessTurn,
    ResumableAgentProcess,
    WorldTurnContext,
)
from shellsmith.episode.vat_protocol import (
    VatAgentEvent,
    VatToolRequest,
    VatToolResponse,
)
from shellsmith.episode.vat_redaction import (
    redact_private_paths,
)
from shellsmith.episode.world_artifacts import (
    TurnResponseArtifact,
    load_turn_response,
)


# Interpolated values (Player tool arguments, task text) legitimately contain
# raw braces — Go templates, jq, awk. Only an UPPER_SNAKE token in double
# braces counts as an unresolved template placeholder.
PLACEHOLDER_TOKEN_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")


def _render_prompt_template(
    *, template: str, replacements: dict[str, str], prompt_name: str
) -> str:
    """Validate a prompt template, then interpolate its known placeholders.

    The validation deliberately happens before interpolation. Replacement values
    contain model-authored tool arguments and validation errors, both of which
    may legitimately contain syntax such as ``{{API_HOST}}``.
    """
    unknown_placeholders = {
        match.group(0)
        for match in PLACEHOLDER_TOKEN_RE.finditer(template)
        if match.group(0) not in replacements
    }
    if unknown_placeholders:
        raise RuntimeError(
            f"{prompt_name} contains an unresolved placeholder: "
            f"{sorted(unknown_placeholders)[0]}."
        )
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


# Inline preview cap for WORLD_TURN_PROMPT. Full thinking remains in request.json.
PLAYER_TURN_SUMMARY_THINKING_CHAR_CAP = 4_000


def format_player_turn_summary(
    request_payload: dict[str, JsonValue],
    *,
    thinking_char_cap: int = PLAYER_TURN_SUMMARY_THINKING_CHAR_CAP,
) -> str:
    """Build a short Player thinking + actions preview for the world turn prompt.

    Main callers:
        `WorldBridge._render_turn_prompt` when filling `{{PLAYER_TURN_SUMMARY}}`.

    Why this exists:
        The authoritative Player payload lives in `request.json`. World CLIs still
        must read that file, but pasting the key thinking/action fields into the
        turn prompt makes pre-action reasoning hard to miss when scoring.
    """

    if thinking_char_cap < 1:
        raise ValueError("thinking_char_cap must be at least one.")

    thinking_blocks = _extract_pre_action_thinking(request_payload)
    if thinking_blocks:
        thinking_text = "\n\n".join(thinking_blocks)
        if len(thinking_text) > thinking_char_cap:
            omitted = len(thinking_text) - thinking_char_cap
            thinking_text = (
                thinking_text[:thinking_char_cap]
                + f"\n… [truncated {omitted} chars; full thinking in request.json]"
            )
    else:
        thinking_text = "(no pre-action thinking block in trajectory_delta)"

    actions_raw = request_payload.get("actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        primary = request_payload.get("action")
        actions_raw = [primary] if isinstance(primary, dict) else []

    action_lines: list[str] = []
    for index, action in enumerate(actions_raw):
        if not isinstance(action, dict):
            action_lines.append(f"{index}. <invalid action entry>")
            continue
        action_id = str(action.get("id") or "")
        name = str(action.get("name") or action.get("kind") or "unknown")
        rejected = bool(action.get("harness_rejected"))
        reject_reason = action.get("harness_reject_reason")
        arguments = action.get("arguments")
        args_text = json.dumps(arguments, ensure_ascii=False, indent=2, sort_keys=True)
        header = f"{index}. {name} id={action_id}"
        if rejected:
            header += " harness_rejected=true"
            if reject_reason:
                header += f" reason={reject_reason}"
        action_lines.append(f"{header}\n{args_text}")

    if not action_lines:
        action_lines.append("(no actions in request)")

    return (
        "## Pre-action thinking\n"
        f"{thinking_text}\n\n"
        "## Actions\n"
        + "\n\n".join(action_lines)
    )


def _extract_pre_action_thinking(request_payload: dict[str, JsonValue]) -> list[str]:
    """Collect assistant thinking strings from trajectory_delta before tools run."""

    blocks: list[str] = []
    trajectory = request_payload.get("trajectory_delta")
    if not isinstance(trajectory, list):
        return blocks
    for event in trajectory:
        if not isinstance(event, dict):
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        message = payload.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "thinking":
                continue
            text = part.get("thinking")
            if isinstance(text, str) and text.strip():
                blocks.append(text.strip())
    return blocks


@dataclass(frozen=True)
class PlayerAction:
    """One Player tool call (or message) we are asking the world to react to.

    Built from VatToolRequest fields in VatWorldToolHandler.
    ``id`` is usually the Pi tool_call_id (used to match responses later).
    """

    id: str
    kind: Literal["tool_call", "message"]
    name: str
    arguments: dict[str, JsonValue]
    # True when OUR harness rejected the call before the world saw a real ask
    # (e.g. bad args). The world still "sees" it for scoring, but Player may
    # get a rejection notice instead of a simulated observation.
    harness_rejected: bool = False
    harness_reject_reason: str | None = None


@dataclass(frozen=True)
class WorldStepEnvelope:
    """Everything needed for ONE world-sim turn (one step_index).

    Pseudocode of how it is used:
        envelope = WorldStepEnvelope(...)
        observation = await bridge.step(envelope)

    Sibling tools: when Player emits several tool calls in one assistant
    message, they share ONE envelope / ONE world turn:
        - ``actions`` = every sibling tool slot
        - ``action``  = the first/primary slot (ledger key + single-tool compat)
    """

    episode_id: str
    step_index: int  # 0, 1, 2, ... becomes turns/000000, turns/000001, ...
    # Fresh Pi events since the previous world turn (thinking, messages, etc.).
    player_delta: tuple[VatAgentEvent, ...]
    action: PlayerAction
    # Hash so a replay/crash cannot silently double-append the ledger.
    idempotency_key: str
    actions: tuple[PlayerAction, ...] = ()


@dataclass(frozen=True)
class SimulatorIntegrity:
    """Private: did the world stay immersed / consistent enough to train on?"""

    trainable: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolResponse:
    """The ONLY part of a world turn that is allowed back into the vat / Player.

    Everything else on WorldObservation (reward, commentary, integrity) is
    private rollout data for judging / training — Player must not see it.
    """

    content: JsonValue
    is_error: bool
    details: dict[str, JsonValue] | None = None


@dataclass(frozen=True)
class PerToolSlot:
    """Per-sibling Player-facing answer inside one multi-tool world turn."""

    action_id: str
    tool_response: ToolResponse | None
    omit_player_facing: bool = False
    harness_rejected: bool = False


@dataclass(frozen=True)
class TurnAssessment:
    """Private dense score for this one step (not shown to Player)."""

    turn_reward: float
    difficulty_rating: float
    completion_percentage: float
    commentary: str
    reasoning_quality: float | None = None
    reasoning_efficiency: float | None = None
    reasoning_commentary: str | None = None


@dataclass(frozen=True)
class WorldObservation:
    """Validated, typed result of one WorldBridge.step(...) call.

    Comes from parsing turns/<index>/response.json plus ledger append.
    Stored on VatWorldToolHandler.observations for final judging.
    """

    action_id: str
    tool_response: ToolResponse
    assessment: TurnAssessment
    integrity: SimulatorIntegrity
    terminate_episode: bool = False
    termination_reason: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    response_path: Path | None = None
    raw_log_path: Path | None = None
    tool_slots: tuple[PerToolSlot, ...] = ()


class WorldBridge:
    """One Player-action → world-observation transaction (serial, durable).

    Main callers:
        VatWorldToolHandler while a Pi tool executor is suspended waiting for us.

    Why this exists:
        The world sim agent (Codex/Claude/Pi) may use arbitrary tools and scratch
        files. We do NOT trust its chat text. We only accept a file it writes at
        a predetermined path: turns/<index>/response.json.
    """

    def __init__(
        self,
        *,
        # ResumableAgentProcess (gm_process/ (package)): Codex/Claude/Pi adapter.
        gm_agent_process: ResumableAgentProcess,
        # Thread/session id from bootstrap_world (same conversation for all turns).
        world_session_id: str,
        # pathlib.Path: episode .../world/turns/ directory.
        turns_dir: Path,
        # pathlib.Path: private tree where the world may materialize real files.
        simulation_root: Path,
        # pathlib.Path: append-only world-truth receipts JSON.
        consistency_ledger_path: Path,
        # (real_host_path, fake_stable_path) pairs for redacting vat paths.
        private_path_replacements: tuple[tuple[str, str], ...],
        # pathlib.Path: WORLD_TURN_PROMPT.md template.
        turn_prompt_template_path: Path,
        turn_response_schema_path: Path | None = None,
        artifact_repair_prompt_path: Path | None = None,
        prompt_inputs: FrozenPromptInputs | None = None,
        max_artifact_attempts: int = 3,
    ) -> None:
        if max_artifact_attempts < 1:
            raise ValueError("max_artifact_attempts must be at least one.")
        self.gm_agent_process = gm_agent_process
        self.world_session_id = world_session_id
        self.turns_dir = turns_dir
        self.simulation_root = simulation_root
        self.consistency_ledger_path = consistency_ledger_path
        self.private_path_replacements = private_path_replacements
        self.turn_prompt_template_path = turn_prompt_template_path
        self.prompt_inputs = prompt_inputs
        self.turn_response_schema_path = turn_response_schema_path
        self.artifact_repair_prompt_path = artifact_repair_prompt_path
        self.max_artifact_attempts = max_artifact_attempts

    async def _resume_world(
        self,
        prompt: str,
        *,
        turn_context: WorldTurnContext,
    ) -> ProcessTurn:
        """Preserve GM ownership for failures nested inside vat.prompt()."""
        try:
            return await self.gm_agent_process.resume(
                self.world_session_id,
                prompt,
                output_schema_path=None,
                turn_context=turn_context,
            )
        except (ExecutionTimeoutError, AttributedExecutionError):
            raise
        except Exception as error:
            raise AttributedExecutionError(
                f"GM {turn_context.stage} failed: "
                f"{type(error).__name__}: {error}",
                component="gm",
                stage=turn_context.stage,
                provider_attributable=True,
            ) from error

    async def step(self, envelope: WorldStepEnvelope) -> WorldObservation:
        """Run one durable Player-action → world-observation transaction.

        Pseudocode (this is what the body does, in order):

            mkdir turns/<step_index>          # must not already exist
            write request.json                # what Player did + ledger pointer
            prompt = fill WORLD_TURN_PROMPT.md
            process_turn = resume(world_session, prompt)
            for attempt in 1..max:
                try:
                    artifact = parse(response.json)
                    assert artifact reviewed the expected ledger revision
                    break
                except invalid:
                    move bad response.json aside
                    resume again with a repair prompt
            append_entry(ledger, ...)         # world cannot edit the ledger itself
            return WorldObservation(...)

        Caller (VatWorldToolHandler) forwards ONLY tool_response into Pi.
        """
        # -----------------------------------------------------------------
        # Stage 1: create an empty turn directory.
        #
        # exist_ok=False on purpose. If the folder already exists, crash
        # recovery would be ambiguous (did we already ask the world?).
        # -----------------------------------------------------------------
        turn_dir: Path = self.turns_dir.joinpath(f"{envelope.step_index:06d}")
        try:
            turn_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RuntimeError(
                f"World turn directory already exists; recovery is ambiguous: {turn_dir}."
            ) from error
        request_path: Path = turn_dir.joinpath("request.json")
        response_path: Path = turn_dir.joinpath("response.json")

        # -----------------------------------------------------------------
        # Stage 2: snapshot the consistency ledger BEFORE the world runs.
        #
        # load_ledger / append_entry come from consistency_ledger.py.
        # The world must claim it reviewed THIS revision; we check later.
        # -----------------------------------------------------------------
        ledger = load_ledger(self.consistency_ledger_path)
        ledger_digest: str = hashlib.sha256(
            self.consistency_ledger_path.read_bytes()
        ).hexdigest()

        # Sibling multi-tool message → one world turn covering every slot.
        sibling_actions: tuple[PlayerAction, ...] = (
            envelope.actions or (envelope.action,)
        )

        # -----------------------------------------------------------------
        # Stage 3: write request.json (authoritative Player payload on disk).
        #
        # redact_private_paths (vat_redaction.py): rewrite real host vat paths
        # to stable fakes like /workspace so the world never sees ephemeral
        # filesystem locations.
        # -----------------------------------------------------------------
        request_payload: dict[str, JsonValue] = {
            "episode_id": envelope.episode_id,
            "step_index": envelope.step_index,
            "idempotency_key": envelope.idempotency_key,
            "action": {
                "id": envelope.action.id,
                "kind": envelope.action.kind,
                "name": envelope.action.name,
                "arguments": redact_private_paths(
                    envelope.action.arguments,
                    replacements=self.private_path_replacements,
                ),
                "harness_rejected": envelope.action.harness_rejected,
                "harness_reject_reason": envelope.action.harness_reject_reason,
            },
            "actions": [
                {
                    "id": action.id,
                    "kind": action.kind,
                    "name": action.name,
                    "arguments": redact_private_paths(
                        action.arguments,
                        replacements=self.private_path_replacements,
                    ),
                    "harness_rejected": action.harness_rejected,
                    "harness_reject_reason": action.harness_reject_reason,
                }
                for action in sibling_actions
            ],
            "trajectory_delta": [
                {
                    "event_type": event.event_type,
                    "payload": redact_private_paths(
                        event.payload,
                        replacements=self.private_path_replacements,
                    ),
                }
                for event in envelope.player_delta
            ],
            "consistency_ledger": {
                "path": str(self.consistency_ledger_path.resolve()),
                "expected_revision": ledger.revision,
                "sha256": ledger_digest,
            },
        }
        request_path.write_text(
            json.dumps(request_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        action_label = envelope.action.name or envelope.action.kind
        turn_timer = TurnTimer()
        emit_simple(
            episode_id=envelope.episode_id,
            phase="turn_start",
            message="GM turn starting (Player tool → world resume)",
            step_index=envelope.step_index,
            action_name=action_label,
        )

        # -----------------------------------------------------------------
        # Stage 4: fill WORLD_TURN_PROMPT.md and resume the world thread.
        #
        # WorldTurnContext (gm_process/ (package)): optional metadata some adapters use
        # (stage name, expected artifact path, idempotency key).
        # ProcessTurn: session_id + events + raw_log_path from that CLI run.
        # -----------------------------------------------------------------
        prompt: str = self._render_turn_prompt(
            episode_id=envelope.episode_id,
            step_index=envelope.step_index,
            turn_dir=turn_dir,
            request_path=request_path,
            response_path=response_path,
            ledger_revision=ledger.revision,
            ledger_digest=ledger_digest,
            request_payload=request_payload,
        )
        turn_context: WorldTurnContext = WorldTurnContext(
            stage="world_turn",
            expected_artifact_path=response_path,
            idempotency_key=envelope.idempotency_key,
        )
        # ProcessTurn (gm_process/ (package)): one completed CLI resume result.
        process_turn: ProcessTurn = await self._resume_world(
            prompt,
            turn_context=turn_context,
        )

        # -----------------------------------------------------------------
        # Stage 5: validate response.json; repair on the SAME thread if needed.
        #
        # load_turn_response (world_artifacts.py) parses + type-checks the file;
        # check_gm_lists applies the ledger's unique-claims rule here, so a
        # duplicate gets a repair prompt instead of failing the append below.
        # We do NOT invent a fake observation if the file is missing/invalid.
        # -----------------------------------------------------------------
        artifact: TurnResponseArtifact | None = None
        validation_errors: list[str] = []
        expected_action_ids: tuple[str, ...] = tuple(
            action.id for action in sibling_actions
        )
        for attempt in range(1, self.max_artifact_attempts + 1):
            try:
                artifact = load_turn_response(
                    response_path,
                    expected_action_id=envelope.action.id,
                    expected_action_ids=expected_action_ids,
                )
                if artifact.consistency_review.reviewed_revision != ledger.revision:
                    raise RuntimeError(
                        "World turn reviewed the wrong consistency-ledger revision: "
                        f"expected {ledger.revision}, got "
                        f"{artifact.consistency_review.reviewed_revision}."
                    )
                review = artifact.consistency_review
                check_gm_lists(  # the rules append_entry applies in stage 6
                    observation_claims=list(review.observation_claims),
                    new_commitments=list(review.new_commitments),
                    executed_cross_checks=list(review.executed_cross_checks),
                )
                break
            except RuntimeError as error:
                validation_errors.append(str(error))
                if attempt == self.max_artifact_attempts:
                    raise AttributedExecutionError(
                        "World turn artifact remained invalid after "
                        f"{attempt} attempts: {' | '.join(validation_errors)}",
                        component="gm",
                        stage="world_turn_artifact_validation",
                        provider_attributable=True,
                    ) from error
                if response_path.is_file():
                    # MOVE (not copy) the bad file aside. Some adapters treat
                    # any parseable file at response_path as "already done" and
                    # would skip running the repair prompt if we left it there.
                    rejected_path: Path = turn_dir.joinpath(
                        f"rejected_{attempt:02d}_response.json"
                    )
                    shutil.move(str(response_path), str(rejected_path))
                repair_prompt: str = self._render_artifact_repair_prompt(
                    request_path=request_path,
                    response_path=response_path,
                    validation_error=str(error),
                )
                repair_prompt_path: Path = turn_dir.joinpath(
                    f"repair_{attempt:02d}_prompt.md"
                )
                repair_prompt_path.write_text(repair_prompt, encoding="utf-8")
                process_turn = await self._resume_world(
                    repair_prompt,
                    turn_context=WorldTurnContext(
                        stage="world_turn_repair",
                        expected_artifact_path=response_path,
                        idempotency_key=envelope.idempotency_key,
                    ),
                )
        if artifact is None:
            raise RuntimeError("World turn artifact was never loaded.")

        # -----------------------------------------------------------------
        # Stage 6: append a ledger receipt (orchestrator-owned, not world-owned).
        #
        # The world REPORTS claims/commitments/conflicts in response.json.
        # WE are the only ones allowed to append the ledger file.
        # -----------------------------------------------------------------
        review = artifact.consistency_review
        append_result = append_entry(
            self.consistency_ledger_path,
            expected_revision=ledger.revision,
            turn_index=envelope.step_index,
            action_id=envelope.action.id,
            observation=artifact.tool_response.content,
            claims=review.observation_claims,
            new_commitments=review.new_commitments,
            cross_checks=review.executed_cross_checks,
        )
        if not append_result.appended:
            raise RuntimeError(
                "A newly created world turn unexpectedly replayed an existing "
                "consistency-ledger entry."
            )

        # -----------------------------------------------------------------
        # Stage 7: pack a WorldObservation for the handler / judge.
        #
        # Ledger conflicts force untrainable + terminate_episode.
        # dict.fromkeys(...): dedupe integrity issue strings, keep order.
        # -----------------------------------------------------------------
        conflict_issues: tuple[str, ...] = tuple(
            f"Simulator consistency conflict: {conflict}"
            for conflict in review.conflicts
        )
        integrity_issues: tuple[str, ...] = tuple(
            dict.fromkeys((*artifact.integrity_issues, *conflict_issues))
        )
        has_conflict: bool = bool(review.conflicts)
        observation = WorldObservation(
            action_id=artifact.action_id,
            tool_response=ToolResponse(
                content=artifact.tool_response.content,
                is_error=artifact.tool_response.is_error,
                details=artifact.tool_response.details,
            ),
            assessment=TurnAssessment(
                turn_reward=artifact.turn_reward,
                difficulty_rating=artifact.difficulty_rating,
                completion_percentage=artifact.completion_percentage,
                commentary=artifact.commentary,
                reasoning_quality=(
                    artifact.reasoning_assessment.quality
                    if artifact.reasoning_assessment is not None
                    else None
                ),
                reasoning_efficiency=(
                    artifact.reasoning_assessment.efficiency
                    if artifact.reasoning_assessment is not None
                    else None
                ),
                reasoning_commentary=(
                    artifact.reasoning_assessment.commentary
                    if artifact.reasoning_assessment is not None
                    else None
                ),
            ),
            integrity=SimulatorIntegrity(
                trainable=artifact.trainable and not has_conflict,
                issues=integrity_issues,
            ),
            terminate_episode=artifact.terminate_episode or has_conflict,
            termination_reason=(
                "The simulator detected a conflict with previously committed world "
                "truth; this rollout is untrainable."
                if has_conflict
                else artifact.termination_reason
            ),
            metadata=artifact.metadata,
            response_path=response_path,
            raw_log_path=process_turn.raw_log_path,
            tool_slots=tuple(
                PerToolSlot(
                    action_id=slot.action_id,
                    tool_response=(
                        None
                        if slot.tool_response is None
                        else ToolResponse(
                            content=slot.tool_response.content,
                            is_error=slot.tool_response.is_error,
                            details=slot.tool_response.details,
                        )
                    ),
                    omit_player_facing=slot.omit_player_facing,
                    harness_rejected=slot.harness_rejected,
                )
                for slot in artifact.tool_slots
            ),
        )
        emit_simple(
            episode_id=envelope.episode_id,
            phase="turn_done",
            message=(
                f"GM replied reward={artifact.turn_reward} "
                f"trainable={observation.integrity.trainable} "
                f"terminate={observation.terminate_episode}"
            ),
            step_index=envelope.step_index,
            action_name=action_label,
            elapsed_seconds=turn_timer.elapsed(),
            turn_reward=artifact.turn_reward,
            trainable=observation.integrity.trainable,
            terminate_episode=observation.terminate_episode,
        )
        return observation

    def _render_turn_prompt(
        self,
        *,
        episode_id: str,
        step_index: int,
        turn_dir: Path,
        request_path: Path,
        response_path: Path,
        ledger_revision: int,
        ledger_digest: str,
        request_payload: dict[str, JsonValue],
    ) -> str:
        """Fill WORLD_TURN_PROMPT.md placeholders with concrete paths/ids.

        Pseudocode:
            template = read(WORLD_TURN_PROMPT.md)
            for {{PLACEHOLDER}}, value in replacements:
                template = template.replace({{PLACEHOLDER}}, value)
            if any {{UPPER_SNAKE}} left unresolved:
                raise  # do not send a half-baked prompt to the world
            return template
        """
        template: str = (self.prompt_inputs.read_text(self.turn_prompt_template_path)
                         if self.prompt_inputs is not None else self.turn_prompt_template_path.read_text(encoding="utf-8"))
        schema_path: Path = (
            self.turn_response_schema_path
            if self.turn_response_schema_path is not None
            else self.turns_dir.parents[1].joinpath(
                "schemas", "world_turn_response.schema.json"
            )
        )
        replacements: dict[str, str] = {
            "{{EPISODE_ID}}": episode_id,
            "{{TURN_INDEX}}": str(step_index),
            "{{TURN_DIRECTORY}}": str(turn_dir.resolve()),
            "{{TURN_REQUEST_PATH}}": str(request_path.resolve()),
            "{{TURN_RESPONSE_PATH}}": str(response_path.resolve()),
            "{{TURN_RESPONSE_SCHEMA_PATH}}": str(schema_path.resolve()),
            "{{SIMULATION_ROOT_PATH}}": str(self.simulation_root.resolve()),
            "{{CONSISTENCY_LEDGER_PATH}}": str(
                self.consistency_ledger_path.resolve()
            ),
            "{{CONSISTENCY_LEDGER_REVISION}}": str(ledger_revision),
            "{{CONSISTENCY_LEDGER_SHA256}}": ledger_digest,
            # Short preview; full Player payload still lives in request.json.
            "{{PLAYER_TURN_SUMMARY}}": format_player_turn_summary(request_payload),
        }
        return _render_prompt_template(
            template=template,
            replacements=replacements,
            prompt_name="World turn prompt",
        )

    def _render_artifact_repair_prompt(
        self,
        *,
        request_path: Path,
        response_path: Path,
        validation_error: str,
    ) -> str:
        """Tell the world: your response.json was invalid; rewrite it correctly."""
        path: Path | None = self.artifact_repair_prompt_path
        if path is None:
            path = self.turn_prompt_template_path.parent.joinpath(
                "WORLD_ARTIFACT_REPAIR_PROMPT.md"
            )
        template: str = (self.prompt_inputs.read_text(path) if self.prompt_inputs is not None
                         else path.read_text(encoding="utf-8"))
        replacements: dict[str, str] = {
            "{{STAGE}}": "per-action world",
            "{{REQUEST_PATH}}": str(request_path.resolve()),
            "{{RESPONSE_PATH}}": str(response_path.resolve()),
            "{{VALIDATION_ERROR}}": validation_error,
        }
        return _render_prompt_template(
            template=template,
            replacements=replacements,
            prompt_name="World repair prompt",
        )


class VatWorldToolHandler:
    """Pi-facing adapter: suspended Node tool calls → WorldBridge.step(...).

    Main callers:
        PiSdkVatProcess (vat_process.py) while a tool Promise is waiting.

    Mental model:
        Node/Pi says: "Player called bash(...); please give me the result."
        This class:
            1. builds a WorldStepEnvelope
            2. awaits bridge.step(...)   # talks to the world sim agent
            3. maps WorldObservation → one VatToolResponse per sibling tool
            4. after step N, sets terminate=True (no more Player turns)
        Node then resolves the Pi Promise with only that tool content.
    """

    def __init__(
        self,
        *,
        episode_id: str,
        bridge: WorldBridge,
        max_world_steps: int = DEFAULT_MAX_WORLD_STEPS,
    ) -> None:
        if max_world_steps < 1:
            raise ValueError("max_world_steps must be at least one.")
        self.episode_id = episode_id
        self.bridge = bridge
        self.max_world_steps = max_world_steps
        # Monotonic turn counter → turns/000000, turns/000001, ...
        self.next_step_index: int = 0
        # Every successful bridge.step observation (used later by judge_episode).
        self.observations: list[WorldObservation] = []
        self.termination_reason: str | None = None
        # Set True when the Nth world step completed and we force-stopped Player.
        self.budget_exhausted: bool = False

    async def __call__(
        self,
        request: VatToolRequest,
        trajectory_delta: tuple[VatAgentEvent, ...],
    ) -> VatToolResponse:
        """Single-tool convenience wrapper around handle_sibling_batch."""
        responses: list[VatToolResponse] = await self.handle_sibling_batch(
            [request], trajectory_delta
        )
        return responses[0]

    async def handle_sibling_batch(
        self,
        requests: list[VatToolRequest],
        trajectory_delta: tuple[VatAgentEvent, ...],
    ) -> list[VatToolResponse]:
        """Handle one Player assistant message that may contain N tool calls.

        Main callers:
            PiSdkVatProcess._run_prompt_loop after all sibling tool_requests arrive.

        Pseudocode:

            if already past N steps:
                raise  # Pi should have stopped on the Nth step
            actions = [PlayerAction from each VatToolRequest]
            observation = await bridge.step(WorldStepEnvelope(...))
            next_step_index += 1
            for each request:
                pick the matching PerToolSlot / tool_response
                build VatToolResponse(content=..., is_error=...)
            if world said terminate OR we just finished step N:
                mark the LAST response terminate=True
            return responses

        Why siblings share one world turn:
            One assistant message with three tool calls is one logical "beat"
            of the environment. Scoring / ledger / consistency happen once.
        """
        if not requests:
            raise RuntimeError("Sibling batch requires at least one tool request.")

        # -----------------------------------------------------------------
        # Budget policy (implemented):
        #   Steps 1..N (N = max_world_steps) run through WorldBridge.step as
        #   normal. Immediately after the Nth step completes, set terminate=True
        #   so Player/Pi does NOT get another model/tool turn. Orchestrator
        #   judge_episode then resumes GMAgent with a budget-exhausted
        #   meta prompt that includes the final-judge brief.
        #
        #   A tool request after the budget is already exhausted is a protocol
        #   bug (Pi should have stopped). Fail loudly — no soft "please finish"
        #   tool result that invites Player to continue.
        # -----------------------------------------------------------------
        if self.next_step_index >= self.max_world_steps:
            raise RuntimeError(
                f"Player requested a tool after the {self.max_world_steps}-step "
                "world budget was already exhausted; the vat should have been "
                "terminated on the Nth step."
            )

        # -----------------------------------------------------------------
        # Build the envelope and run one WorldBridge.step.
        # VatToolRequest / VatAgentEvent come from vat_protocol.py.
        # -----------------------------------------------------------------
        actions: tuple[PlayerAction, ...] = tuple(
            PlayerAction(
                id=request.tool_call_id,
                kind="tool_call",
                name=request.tool_name,
                arguments=request.arguments,
                harness_rejected=request.harness_rejected,
                harness_reject_reason=request.harness_reject_reason,
            )
            for request in requests
        )
        primary: PlayerAction = actions[0]
        step_index: int = self.next_step_index
        idempotency_key: str = hashlib.sha256(
            f"{self.episode_id}:{step_index}:{','.join(action.id for action in actions)}".encode(
                "utf-8"
            )
        ).hexdigest()
        observation: WorldObservation = await self.bridge.step(
            WorldStepEnvelope(
                episode_id=self.episode_id,
                step_index=step_index,
                player_delta=trajectory_delta,
                action=primary,
                actions=actions,
                idempotency_key=idempotency_key,
            )
        )
        self.next_step_index += 1
        self.observations.append(observation)

        # After the Nth successful world step: force-stop Player. Deliver the
        # real tool_response(s) for this step, but terminate=True so Pi does
        # not cede another generation turn. Judging follows in the orchestrator.
        budget_hit: bool = self.next_step_index >= self.max_world_steps
        if observation.terminate_episode:
            self.termination_reason = observation.termination_reason
        elif budget_hit:
            self.budget_exhausted = True
            self.termination_reason = (
                f"Player exhausted the {self.max_world_steps}-step world-tool "
                "budget; the episode is being forcibly terminated and handed to "
                "the Game Master for final judgment."
            )
        if budget_hit:
            # Even if the world also asked to terminate, mark budget so the
            # orchestrator wraps the final judge with the forced-end meta prompt.
            self.budget_exhausted = True

        # -----------------------------------------------------------------
        # Map private WorldObservation → public VatToolResponse list.
        #
        # Rules:
        #   - Only OUR harness may mark a tool rejected/omitted.
        #   - Worlds may not quietly suppress a valid Player observation.
        #   - Single-tool legacy artifacts use top-level tool_response.
        # -----------------------------------------------------------------
        slots_by_id: dict[str, PerToolSlot] = {
            slot.action_id: slot for slot in observation.tool_slots
        }
        responses: list[VatToolResponse] = []
        for request in requests:
            slot: PerToolSlot | None = slots_by_id.get(request.tool_call_id)
            if (
                slot is not None
                and (slot.omit_player_facing or slot.harness_rejected)
                and not request.harness_rejected
            ):
                raise RuntimeError(
                    "World marked a harness-valid sibling as harness-rejected/"
                    f"omitted: {request.tool_call_id!r}. Worlds may not "
                    "suppress valid observations."
                )
            if request.harness_rejected or (
                slot is not None and slot.omit_player_facing
            ):
                reason: str = (
                    request.harness_reject_reason
                    or "Tool call rejected by harness validation."
                )
                responses.append(
                    VatToolResponse(
                        tool_call_id=request.tool_call_id,
                        content=reason,
                        is_error=True,
                        details={"harness_rejected": True},
                        terminate=False,
                        omit_player_facing=True,
                    )
                )
                continue
            tool_response: ToolResponse
            if slot is None or slot.tool_response is None:
                if len(requests) == 1:
                    tool_response = observation.tool_response
                else:
                    raise RuntimeError(
                        "Sibling world turn is missing Player-facing tool_response for "
                        f"{request.tool_call_id!r}."
                    )
            else:
                tool_response = slot.tool_response
            responses.append(
                VatToolResponse(
                    tool_call_id=request.tool_call_id,
                    content=tool_response.content,
                    is_error=tool_response.is_error,
                    details=tool_response.details,
                    terminate=False,
                    omit_player_facing=False,
                )
            )

        # Put terminate=True on the LAST response in the batch (even if that
        # sibling was harness-rejected). If we only tagged the last "valid"
        # slot, Pi kept running after the world had already terminated.
        # Also terminate when the N-step budget was just exhausted.
        should_terminate: bool = bool(
            observation.terminate_episode or budget_hit
        )
        if should_terminate and responses:
            # dataclasses.replace: copy the last response with terminate=True.
            responses[-1] = replace(responses[-1], terminate=True)
        return responses
