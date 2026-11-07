"""Durable Tinker capture + policy provenance for in-place campaign resume.

Architecture map
----------------
    accepted CapturedRollout (live prompts in RAM)
        -> serialize prompt_token_ids + tokens/logprobs
        -> {rollout}/logs/tinker_trainable_capture.json
        -> campaign capture_index.json entry
    every finished episode
        -> {rollout}/policy_provenance.json
        -> {rollout}/logs/durable_outcome.json

On resume:
    CaptureStore.load_from_run_dir rehydrates ModelInput via from_ints
    run_campaign_wave skips episode_ids that already have durable_outcome.json

Why this exists
---------------
CaptureStore is RAM-only. Protocol JSONL omits the live prompt object, so a
crash mid-wave cannot rebuild CISPO datums. This module is the durable handoff
the trainer needs to resume the same run directory without re-rolling finished
episodes. Ported from the deprecated legacy trainer.

Main callers:
    The trainer's Tinker rollout runner (persist; M4)
    train.capture_store.CaptureStore.load_from_run_dir (rehydrate)
    campaign.stream_campaign.run_campaign_wave (skip completed)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from shellsmith.credit.to_tinker_bridge import (
    CapturedRollout,
    CapturedModelTurn,
)
from shellsmith.policy.records import TinkerRolloutRecord
from shellsmith.process_io import fsync_directory, write_json


CAPTURE_SCHEMA = "shellsmith.tinker_trainable_capture.v1"
POLICY_CAPTURE_SCHEMA = "shellsmith.policy_trainable_capture.v2"
POLICY_CAPTURE_FILENAME = "policy_trainable_capture.json"
INDEX_SCHEMA = "shellsmith.capture_index.v1"
PROVENANCE_SCHEMA = "shellsmith.policy_provenance.v1"
OUTCOME_SCHEMA = "shellsmith.durable_outcome.v1"
PENDING_UPDATE_SCHEMA = "shellsmith.pending_update.v1"

CAPTURE_FILENAME = "tinker_trainable_capture.json"
PROVENANCE_FILENAME = "policy_provenance.json"
OUTCOME_FILENAME = "durable_outcome.json"
CAPTURE_INDEX_FILENAME = "capture_index.json"
PENDING_UPDATE_FILENAME = "pending_update.json"


def _write_json_atomic(path: Path, value: Any) -> None:
    write_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"Expected JSON object at {path}.")
    return raw


def capture_path_for_rollout(rollout_root: Path) -> Path:
    return Path(rollout_root).joinpath("logs", CAPTURE_FILENAME)


def provenance_path_for_rollout(rollout_root: Path) -> Path:
    return Path(rollout_root).joinpath(PROVENANCE_FILENAME)


def durable_outcome_path_for_rollout(rollout_root: Path) -> Path:
    return Path(rollout_root).joinpath("logs", OUTCOME_FILENAME)


def capture_index_path(run_dir: Path) -> Path:
    return Path(run_dir).joinpath(CAPTURE_INDEX_FILENAME)


def pending_update_path(run_dir: Path) -> Path:
    return Path(run_dir).joinpath(PENDING_UPDATE_FILENAME)


def prompt_token_ids(prompt: object) -> list[int]:
    """Extract flat token ids from a live Tinker ModelInput.

    Main callers:
        serialize_capture.

    Why this exists:
        Shellsmith Player is text+tools. Multimodal prompts must fail
        loudly — we never invent a text-only stand-in.
    """
    if prompt is None:
        raise RuntimeError("Cannot serialize a capture with a missing prompt.")
    to_ints = getattr(prompt, "to_ints", None)
    if not callable(to_ints):
        raise RuntimeError(
            "Tinker prompt object has no to_ints(); cannot durable-serialize."
        )
    try:
        tokens = list(to_ints())
    except Exception as exc:
        raise RuntimeError(
            "Failed to serialize Tinker prompt via to_ints() (multimodal or "
            "unsupported ModelInput). Durable capture requires text-only prompts."
        ) from exc
    if not tokens:
        raise RuntimeError("Serialized Tinker prompt_token_ids is empty.")
    return [int(token) for token in tokens]


def serialize_capture(
    capture: CapturedRollout, *, episode_id: str
) -> dict[str, Any]:
    """Turn one accepted capture into a JSON-serializable durable payload.

    Main callers:
        persist_accepted_capture.
    """
    records: list[dict[str, Any]] = []
    backend = {getattr(item.record, "backend", "tinker") for item in capture.captures}
    if backend not in ({"tinker"}, {"mlx"}, {"studio"}):
        raise RuntimeError("A capture must use one supported policy backend.")
    neutral = backend != {"tinker"}
    for item in capture.captures:
        record = item.record
        ids = prompt_token_ids(getattr(record, "prompt", None))
        forced = getattr(record, "forced_close_token_span", None)
        records.append(
            {
                "request_id": str(getattr(record, "request_id", "")),
                "created_at": float(getattr(record, "created_at", 0.0) or 0.0),
                "prompt_token_ids": ids,
                "prompt_tokens": int(
                    getattr(record, "prompt_tokens", 0) or len(ids)
                ),
                "sampled_tokens": [
                    int(token) for token in (getattr(record, "sampled_tokens", []) or [])
                ],
                "sampled_logprobs": [
                    float(value)
                    for value in (getattr(record, "sampled_logprobs", []) or [])
                ],
                "content": str(getattr(record, "content", "")),
                "finish_reason": str(getattr(record, "finish_reason", "")),
                "payload": dict(getattr(record, "payload", {}) or {}),
                "reasoning_content": str(
                    getattr(record, "reasoning_content", "") or ""
                ),
                "forced_close_token_span": (
                    [int(forced[0]), int(forced[1])]
                    if isinstance(forced, (list, tuple)) and len(forced) == 2
                    else None
                ),
                "kind": item.kind,
                "world_turn_indices": list(item.world_turn_indices),
                **({"backend": record.backend, "policy_ref": getattr(record, "policy_ref", None),
                    "policy_identity": getattr(record, "policy_identity", {}),
                    "diagnostic_generation_attempts": list(getattr(record, "diagnostic_generation_attempts", ())),
                    "protocol_row": getattr(record, "protocol_row", {})} if neutral else {}),
            }
        )
    return {
        "schema": POLICY_CAPTURE_SCHEMA if neutral else CAPTURE_SCHEMA,
        "episode_id": episode_id,
        "group_identity_key": capture.group_identity_key,
        "rollout_root": str(capture.rollout_root),
        "records": records,
    }


def load_capture(path: Path, *, deps: Any = None) -> CapturedRollout:
    """Rebuild a CapturedRollout from durable JSON.

    Main callers:
        CaptureStore.load_from_run_dir / unit tests.
    """
    path = Path(path)
    payload = _read_json(path)
    if payload.get("schema") not in {CAPTURE_SCHEMA, POLICY_CAPTURE_SCHEMA}:
        raise RuntimeError(
            f"Unsupported trainable capture schema at {path}: "
            f"{payload.get('schema')!r}."
        )
    group_key = payload.get("group_identity_key")
    if not isinstance(group_key, str) or not group_key:
        raise RuntimeError(f"Capture at {path} is missing group_identity_key.")
    rollout_root_raw = payload.get("rollout_root")
    if not isinstance(rollout_root_raw, str) or not rollout_root_raw:
        raise RuntimeError(f"Capture at {path} is missing rollout_root.")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Capture at {path} has no records.")
    from shellsmith.train.backend import TokenPrompt
    neutral = payload["schema"] == POLICY_CAPTURE_SCHEMA
    model_input_type = deps.tinker.types.ModelInput if deps is not None and not neutral else None
    turns: list[CapturedModelTurn] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Capture record {index} at {path} is not an object.")
        token_ids = row.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(
                f"Capture record {index} at {path} lacks prompt_token_ids."
            )
        if any(type(token) is not int or token < 0 for token in token_ids):
            raise RuntimeError("Capture prompt IDs must be non-negative integers.")
        prompt = model_input_type.from_ints(token_ids) if model_input_type else TokenPrompt(tuple(token_ids))
        forced_raw = row.get("forced_close_token_span")
        forced: tuple[int, int] | None = None
        if forced_raw is not None:
            if (
                not isinstance(forced_raw, list)
                or len(forced_raw) != 2
            ):
                raise RuntimeError(
                    f"Capture record {index} has invalid forced_close_token_span."
                )
            forced = (int(forced_raw[0]), int(forced_raw[1]))
        attempts = row.get("diagnostic_generation_attempts", [])
        if not isinstance(attempts, list) or any(not isinstance(attempt, dict) for attempt in attempts):
            raise RuntimeError("Capture diagnostic generation attempts must be an object list.")
        record = TinkerRolloutRecord(
            request_id=str(row.get("request_id") or ""),
            created_at=float(row.get("created_at") or 0.0),
            prompt=prompt,
            prompt_tokens=int(row.get("prompt_tokens") or len(token_ids)),
            sampled_tokens=[int(token) for token in (row.get("sampled_tokens") or [])],
            sampled_logprobs=[
                float(value) for value in (row.get("sampled_logprobs") or [])
            ],
            content=str(row.get("content") or ""),
            finish_reason=str(row.get("finish_reason") or ""),
            payload=dict(row.get("payload") or {}),
            reasoning_content=str(row.get("reasoning_content") or ""),
            forced_close_token_span=forced,
            backend=row.get("backend", "tinker"), policy_ref=row.get("policy_ref"),
            policy_identity=row.get("policy_identity", {}),
            protocol_row=row.get("protocol_row", {}),
            diagnostic_generation_attempts=tuple(dict(attempt) for attempt in attempts),
        )
        if neutral and (record.backend not in {"mlx","studio"} or not isinstance(record.policy_ref, str) or not record.policy_ref
                        or not isinstance(record.policy_identity, dict) or not record.policy_identity):
            raise RuntimeError("Neutral capture is missing its immutable MLX policy provenance.")
        if not isinstance(record.protocol_row, dict):
            raise RuntimeError("Capture protocol_row must be an object.")
        from shellsmith.credit.rollout import TrainableRolloutRecordSpec
        TrainableRolloutRecordSpec.from_record(record)
        kind = row.get("kind")
        if not isinstance(kind, str) or not kind:
            raise RuntimeError(f"Capture record {index} at {path} lacks kind.")
        indices_raw = row.get("world_turn_indices") or []
        if not isinstance(indices_raw, list):
            raise RuntimeError(
                f"Capture record {index} has invalid world_turn_indices."
            )
        # Older captures listed the same move once per tool call. Collapse to
        # unique move indices so resume can bank those files.
        turn_indices = tuple(
            dict.fromkeys(int(value) for value in indices_raw)
        )
        turns.append(
            CapturedModelTurn(
                record=record,
                kind=kind,  # type: ignore[arg-type]
                world_turn_indices=turn_indices,
            )
        )
    return CapturedRollout(
        rollout_root=Path(rollout_root_raw),
        group_identity_key=group_key,
        captures=tuple(turns),
    )


@dataclass(frozen=True)
class PolicyProvenance:
    """Which LoRA/base weights produced an episode's Player samples."""

    model_name: str
    lora_rank: int
    policy_kind: str
    policy_ref: str | None
    rft_updates_completed: int
    group_identity_key: str
    episode_id: str
    backend: str = "tinker"
    policy_identity: dict[str, Any] | None = None

    def to_jsonable(self) -> dict[str, Any]:
        if self.policy_kind not in {"base", "sampler_checkpoint"}:
            raise RuntimeError(
                f"Unsupported policy_kind {self.policy_kind!r}; expected "
                "'base' or 'sampler_checkpoint'."
            )
        if self.policy_kind == "sampler_checkpoint":
            if not isinstance(self.policy_ref, str) or not self.policy_ref:
                raise RuntimeError(
                    "sampler_checkpoint provenance requires a non-empty policy_ref."
                )
        elif self.rft_updates_completed > 0:
            raise RuntimeError(
                "policy_kind='base' is invalid when rft_updates_completed > 0; "
                "stamp a sampler_checkpoint policy_ref before sampling."
            )
        return {
            "schema": PROVENANCE_SCHEMA,
            "model_name": self.model_name,
            "lora_rank": int(self.lora_rank),
            "policy_kind": self.policy_kind,
            "policy_ref": self.policy_ref,
            "rft_updates_completed": int(self.rft_updates_completed),
            "group_identity_key": self.group_identity_key,
            "episode_id": self.episode_id,
            **({"backend": self.backend, "policy_identity": self.policy_identity or {}} if self.backend != "tinker" else {}),
        }


def write_policy_provenance(rollout_root: Path, provenance: PolicyProvenance) -> Path:
    path = provenance_path_for_rollout(rollout_root)
    _write_json_atomic(path, provenance.to_jsonable())
    return path


def write_durable_outcome(rollout_root: Path, outcome: dict[str, Any]) -> Path:
    """Persist one *finished-job* outcome so resume can skip re-running it.

    An ``episode_not_started`` row (the provider-down placeholder) is not a
    finished job and must never be persisted: doing so would make the next
    resume inherit a synthetic failure for an attempt slot that never ran, and
    that slot could then never be filled. Writing one is a caller bug, so this
    raises instead of quietly dropping the write.
    """
    if outcome.get("episode_not_started") is True:
        raise RuntimeError(
            "Refusing to write a durable outcome for an episode that never "
            f"started ({outcome.get('episode_id')!r}); resume must be free to "
            "retry that attempt."
        )
    path = durable_outcome_path_for_rollout(rollout_root)
    row = {"schema": OUTCOME_SCHEMA, "outcome": outcome}
    _write_json_atomic(path, row)
    return path


def load_durable_outcome(rollout_root: Path) -> dict[str, Any] | None:
    """Read one settled episode outcome, or None when there is nothing to replay.

    ``None`` also covers a placeholder file left by an older run: an
    ``episode_not_started`` payload describes a blocked attempt slot, not a
    settled episode, so the caller re-runs it rather than replaying it.
    """
    path = durable_outcome_path_for_rollout(rollout_root)
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema") != OUTCOME_SCHEMA:
        raise RuntimeError(f"Unsupported durable outcome schema at {path}.")
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        raise RuntimeError(f"Durable outcome at {path} is missing outcome object.")
    if outcome.get("episode_not_started") is True:
        return None
    return outcome


def mark_durable_outcome_discarded(
    rollout_root: Path, *, reason: str
) -> bool:
    """Strip trainability from one episode's durable outcome, keeping the file.

    Main callers:
        ``stream_campaign.discard_campaign_outcomes`` on a provider fallback.

    Why this exists:
        A provider fallback throws the whole group away and redoes the world file
        on the backup GM. Discarding drops the in-memory captures and the
        ``capture_index`` rows, but ``logs/durable_outcome.json`` still claimed
        ``accepted_count: 1``. On the next resume the wave replayed that row, the
        adaptive policy read the group as trainable, and ``captures_for_outcomes``
        then died with "Accepted outcome ... has no capture in the store" —
        collection wedged for good.

        The file is rewritten, not deleted: the episode really ran and its audit
        is evidence. Only the training claim is withdrawn, and the reason is
        recorded next to it.

    Returns True when a durable outcome existed and was rewritten.
    """
    path = durable_outcome_path_for_rollout(rollout_root)
    if not path.is_file():
        return False
    payload = _read_json(path)
    if payload.get("schema") != OUTCOME_SCHEMA:
        raise RuntimeError(f"Unsupported durable outcome schema at {path}.")
    outcome = payload.get("outcome")
    if not isinstance(outcome, dict):
        raise RuntimeError(f"Durable outcome at {path} is missing outcome object.")
    outcome.update(discarded_outcome_fields(reason))
    _write_json_atomic(path, {"schema": OUTCOME_SCHEMA, "outcome": outcome})
    return True


def discarded_outcome_fields(reason: str) -> dict[str, Any]:
    """The exact keys that withdraw an outcome's training claim.

    Shared by the durable file and the in-memory campaign scoreboard row so the
    two can never disagree about whether an episode is still trainable.
    """
    if not reason:
        raise RuntimeError("A discarded outcome must record why it was discarded.")
    return {
        "accepted_count": 0,
        "discarded_by_provider_fallback": True,
        "discard_reason": reason,
    }


def upsert_capture_index(
    run_dir: Path,
    *,
    episode_id: str,
    rollout_root: Path,
    accepted: bool,
    policy_kind: str,
    policy_ref: str | None,
    group_identity_key: str,
    capture_filename: str = CAPTURE_FILENAME,
) -> None:
    """Update campaign-level capture_index.json for one episode."""
    run_dir = Path(run_dir)
    index_path = capture_index_path(run_dir)
    if index_path.is_file():
        index = _read_json(index_path)
        if index.get("schema") != INDEX_SCHEMA:
            raise RuntimeError(f"Unsupported capture index schema at {index_path}.")
        episodes = index.get("episodes")
        if not isinstance(episodes, dict):
            raise RuntimeError(f"capture_index episodes must be an object: {index_path}.")
    else:
        episodes = {}
        index = {"schema": INDEX_SCHEMA, "episodes": episodes}
    relative_rollout = Path(rollout_root).resolve().relative_to(run_dir.resolve())
    entry: dict[str, Any] = {
        "relative_rollout_dir": str(relative_rollout),
        "accepted": bool(accepted),
        "policy_kind": policy_kind,
        "policy_ref": policy_ref,
        "group_identity_key": group_identity_key,
    }
    if accepted:
        entry["capture_path"] = str(
            relative_rollout.joinpath("logs", capture_filename)
        )
    episodes[episode_id] = entry
    index["episodes"] = episodes
    _write_json_atomic(index_path, index)


def remove_capture_index_entries(run_dir: Path, episode_ids: list[str]) -> None:
    index_path = capture_index_path(run_dir)
    if not index_path.is_file():
        return
    index = _read_json(index_path)
    episodes = index.get("episodes")
    if not isinstance(episodes, dict):
        return
    for episode_id in episode_ids:
        episodes.pop(episode_id, None)
    index["episodes"] = episodes
    _write_json_atomic(index_path, index)


def persist_accepted_capture(
    *,
    run_dir: Path,
    episode_id: str,
    capture: CapturedRollout,
    provenance: PolicyProvenance,
) -> Path:
    """Write trainable capture JSON, provenance, and capture_index entry."""
    payload = serialize_capture(capture, episode_id=episode_id)
    filename = POLICY_CAPTURE_FILENAME if payload["schema"] == POLICY_CAPTURE_SCHEMA else CAPTURE_FILENAME
    path = capture.rollout_root / "logs" / filename
    _write_json_atomic(path, payload)
    write_policy_provenance(capture.rollout_root, provenance)
    upsert_capture_index(
        run_dir,
        episode_id=episode_id,
        rollout_root=capture.rollout_root,
        accepted=True,
        policy_kind=provenance.policy_kind,
        policy_ref=provenance.policy_ref,
        group_identity_key=capture.group_identity_key,
        capture_filename=filename,
    )
    return path


def load_captures_from_index(
    run_dir: Path, *, deps: Any = None
) -> dict[str, CapturedRollout]:
    """Load all accepted captures listed in capture_index.json.

    Main callers:
        CaptureStore.load_from_run_dir.
    """
    run_dir = Path(run_dir)
    index_path = capture_index_path(run_dir)
    if not index_path.is_file():
        return {}
    index = _read_json(index_path)
    if index.get("schema") != INDEX_SCHEMA:
        raise RuntimeError(f"Unsupported capture index schema at {index_path}.")
    episodes = index.get("episodes")
    if not isinstance(episodes, dict):
        raise RuntimeError(f"capture_index episodes must be an object: {index_path}.")
    loaded: dict[str, CapturedRollout] = {}
    for episode_id, entry in episodes.items():
        if not isinstance(entry, dict):
            raise RuntimeError(
                f"capture_index entry for {episode_id!r} must be an object."
            )
        if entry.get("accepted") is not True:
            continue
        relative = entry.get("capture_path")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError(
                f"Accepted capture_index entry {episode_id!r} lacks capture_path."
            )
        path = run_dir.joinpath(relative)
        if not path.is_file():
            raise RuntimeError(
                f"capture_index lists {episode_id!r} but file is missing: {path}."
            )
        loaded[str(episode_id)] = load_capture(path, deps=deps)
    return loaded


def write_pending_update(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = pending_update_path(run_dir)
    row = {"schema": PENDING_UPDATE_SCHEMA, **payload}
    _write_json_atomic(path, row)
    return path


def load_pending_update(run_dir: Path) -> dict[str, Any] | None:
    path = pending_update_path(run_dir)
    if not path.is_file():
        return None
    payload = _read_json(path)
    if payload.get("schema") != PENDING_UPDATE_SCHEMA:
        raise RuntimeError(f"Unsupported pending_update schema at {path}.")
    return payload


def clear_pending_update(run_dir: Path) -> None:
    path = pending_update_path(run_dir)
    if path.is_file():
        path.unlink()
        fsync_directory(path.parent)


def latest_sampler_checkpoint_path(
    state: dict[str, Any], *, rft_update: int | None = None
) -> str | None:
    """Return the newest sampler checkpoint URI, optionally for one RFT step.

    A sampler checkpoint is only valid provenance for the update that produced
    it. Callers that stamp a current sampling policy must pass ``rft_update``
    so a checkpoint from an earlier update cannot be reused by mistake.
    """
    rows = state.get("checkpoints")
    if not isinstance(rows, list):
        return None
    latest: str | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("kind") != "sampler":
            continue
        if rft_update is not None and row.get("rft_update") != rft_update:
            continue
        path = row.get("path")
        if isinstance(path, str) and path:
            latest = path
    return latest


def resolve_policy_stamp(
    *,
    state: dict[str, Any],
    model_name: str,
    lora_rank: int,
    episode_id: str,
    group_identity_key: str,
    pinned_policy_kind: str | None = None,
    pinned_policy_ref: str | None = None,
    pinned_rft_updates_completed: int | None = None,
) -> PolicyProvenance:
    """Build provenance for the policy currently sampling Player.

    ``pinned_*`` (``--retry``):
        Force the group's existing Player policy so top-up attempts cannot mix
        with a newer sampler checkpoint after the failed bank.
    """
    if pinned_policy_kind is not None:
        kind = pinned_policy_kind
        if kind == "base":
            if pinned_policy_ref is not None:
                raise RuntimeError(
                    "Pinned base policy must not include a policy_ref."
                )
            rft = (
                0
                if pinned_rft_updates_completed is None
                else int(pinned_rft_updates_completed)
            )
            return PolicyProvenance(
                model_name=model_name,
                lora_rank=lora_rank,
                policy_kind="base",
                policy_ref=None,
                rft_updates_completed=rft,
                group_identity_key=group_identity_key,
                episode_id=episode_id,
            )
        if kind != "sampler_checkpoint":
            raise RuntimeError(
                f"Unsupported pinned_policy_kind {kind!r}."
            )
        if not isinstance(pinned_policy_ref, str) or not pinned_policy_ref:
            raise RuntimeError(
                "Pinned sampler_checkpoint requires a non-empty policy_ref."
            )
        rft = (
            int(state.get("rft_updates_completed") or 0)
            if pinned_rft_updates_completed is None
            else int(pinned_rft_updates_completed)
        )
        return PolicyProvenance(
            model_name=model_name,
            lora_rank=lora_rank,
            policy_kind="sampler_checkpoint",
            policy_ref=pinned_policy_ref,
            rft_updates_completed=rft,
            group_identity_key=group_identity_key,
            episode_id=episode_id,
        )

    rft = int(state.get("rft_updates_completed") or 0)
    if rft <= 0:
        return PolicyProvenance(
            model_name=model_name,
            lora_rank=lora_rank,
            policy_kind="base",
            policy_ref=None,
            rft_updates_completed=rft,
            group_identity_key=group_identity_key,
            episode_id=episode_id,
        )
    ref = latest_sampler_checkpoint_path(state, rft_update=rft)
    if not ref:
        raise RuntimeError(
            "Cannot stamp sampler_checkpoint provenance: trainer_state has no "
            f"sampler checkpoint for rft_update={rft}. Save a sampler checkpoint "
            "before collecting the next wave."
        )
    return PolicyProvenance(
        model_name=model_name,
        lora_rank=lora_rank,
        policy_kind="sampler_checkpoint",
        policy_ref=ref,
        rft_updates_completed=rft,
        group_identity_key=group_identity_key,
        episode_id=episode_id,
    )
