"""Campaign rollout runner that samples the Player from Tinker and banks captures.

Architecture map
----------------
    make_tinker_rollout_runner(endpoint, capture_store, run_dir, identity_for, policy_stamp)
      -> run_tinker_job(job, args)          (a campaign ``RolloutRunner``)
           smoke args for this job, Player base URL = endpoint /e/<episode_id>/v1
           run_live_smoke                   episode artifacts in job.output_dir
           endpoint.take_records(episode)   token-exact samples for this episode
           harvest_episode                  protocol rows -> audit -> aligned capture
           accepted: capture_store.put + persist_accepted_capture (durable, resumable)
           rejected: provenance + capture-index row (accepted=False)
           outcome dict (accepted_count 0/1, audit, GM fields, failure attribution)

Invariants
----------
- ``accepted_count`` is 1 only when a capture was banked, so the adaptive
  policy's GROUP_READY never points at a missing capture.
- Every finished job writes ``durable_outcome.json``; a resumed wave skips it.
- Failures are recorded on the outcome (with provider attribution for the
  outage circuit), never swallowed.

Why this exists:
    Port of the legacy conductor's ``make_tinker_rollout_runner``. The
    per-episode in-process LLM proxy is replaced by one shared
    ``policy.server.PlayerEndpoint``; records are routed by episode id.

Main callers:
    ``train.trainer`` passes the runner to ``stream_engine.collect_adaptive_world``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any

from shellsmith.campaign.rollout_campaign import RolloutJob, _smoke_args
from shellsmith.config import PLAYER_ENDPOINT_KEY_ENV
from shellsmith.credit.to_tinker_bridge import CapturedRollout
from shellsmith.episode.failure import outcome_failure_fields, read_episode_failure_detail, serialize_failure
from shellsmith.episode.progress import emit_simple
from shellsmith.episode.smoke import run_live_smoke
from shellsmith.episode.world_prompt_profile import FrozenPromptInputs
from shellsmith.policy.harvest import harvest_episode
from shellsmith.policy.server import SERVED_MODEL_ID, PlayerEndpoint
from shellsmith.train.capture_store import CaptureStore
from shellsmith.train.durable_capture import (
    PolicyProvenance,
    persist_accepted_capture,
    upsert_capture_index,
    write_durable_outcome,
    write_policy_provenance,
)
from shellsmith.train.identity import RolloutGroupIdentity

IdentityFor = Callable[[RolloutJob], RolloutGroupIdentity]
PolicyStamp = Callable[[str, str], PolicyProvenance]


def token_usage_for_records(records: list[Any]) -> dict[str, int]:
    """Count real generation work, including discarded thinking retry draws."""
    usage = {key: 0 for key in ("prompt_tokens", "completion_tokens", "thinking_tokens", "response_tokens", "forced_close_tokens")}
    for record in records:
        prompt = getattr(record, "prompt_tokens", 0)
        tokens = getattr(record, "sampled_tokens", ())
        row = getattr(record, "protocol_row", None) or {}
        # The final attempt is already in the main sampled sequence, whether
        # it succeeded or exhausted all retries. Earlier rejected draws never
        # enter the optimizer batch, but consumed real decoding and prefills.
        discarded = [attempt for attempt in getattr(record, "diagnostic_generation_attempts", ())[:-1]
                     if attempt.get("disposition") == "rejected_eos_inside_think"]
        extra = sum(len(attempt.get("sampled_tokens", ())) for attempt in discarded)
        usage["prompt_tokens"] += prompt * (1 + len(discarded))
        usage["completion_tokens"] += len(tokens) + extra
        usage["thinking_tokens"] += row.get("thinking_tokens", len(tokens) if row.get("thinking") else 0) + extra
        usage["response_tokens"] += row.get("response_tokens", 0)
        if (span := getattr(record, "forced_close_token_span", None)) is not None:
            usage["forced_close_tokens"] += span[1] - span[0]
    return usage


def make_tinker_rollout_runner(
    *,
    endpoint: PlayerEndpoint,
    capture_store: CaptureStore,
    run_dir: Path,
    identity_for: IdentityFor,
    policy_stamp: PolicyStamp,
    player_renderer: str,
    player_context_window: int,
    pi_sdk_module_path: Path | None,
    codex_executable: Path | None = None,
    gm_python_runtime: Any = None,
    prompt_inputs_for: Callable[[RolloutJob], FrozenPromptInputs] | None = None,
):
    """Return an async ``RolloutRunner`` bound to one Player endpoint and capture store."""
    capture_store._run_dir = Path(run_dir)

    async def collect_job(job: RolloutJob, args: argparse.Namespace) -> dict[str, Any]:
        started = time.monotonic()
        identity = identity_for(job)
        provenance = policy_stamp(job.episode_id, identity.key)
        emit_simple(episode_id=job.episode_id, phase="campaign_job_start",
                    message=f"{provenance.backend} job starting gm={job.gm_model} attempt={job.attempt}",
                    world_id=job.world_id, gm_model=job.gm_model, attempt=job.attempt)
        smoke_args = _smoke_args(args, job)
        if prompt_inputs_for is not None:
            smoke_args.frozen_prompt_inputs = prompt_inputs_for(job)
        if codex_executable is not None:
            smoke_args.codex_bin = str(codex_executable)
        if gm_python_runtime is not None:
            smoke_args.gm_python_runtime = gm_python_runtime
        smoke_args.player_base_url = endpoint.base_url(job.episode_id)
        smoke_args.player_model = SERVED_MODEL_ID
        smoke_args.player_provider = f"shellsmith-{provenance.backend}"
        smoke_args.player_api_key_env = PLAYER_ENDPOINT_KEY_ENV
        smoke_args.player_renderer = player_renderer
        smoke_args.player_context_window = player_context_window
        if hasattr(args, "protocol_retry"):
            smoke_args.protocol_retry = args.protocol_retry
        for key in ("player_compaction_reserve_tokens", "player_compaction_keep_recent_tokens",
                    "player_reasoning_tokens", "player_response_tokens", "gm_api_idle_timeout_seconds"):
            if hasattr(args, key):
                setattr(smoke_args, key, getattr(args, key))
        smoke_args.pi_sdk_module_path = pi_sdk_module_path

        error: str | None = None
        failure_detail: dict[str, Any] | None = None
        try:
            await run_live_smoke(smoke_args, player_api_key=endpoint.api_key)
        except Exception as caught:  # recorded on the outcome, never swallowed
            failure_detail = serialize_failure(caught, stage="tinker_campaign_job")
            error = str(failure_detail["error"])
        records = endpoint.take_records(job.episode_id)
        if provenance.backend in {"mlx","studio"} and records and any(
            record.backend != provenance.backend or record.policy_ref != provenance.policy_ref
            or record.policy_identity != provenance.policy_identity for record in records
        ):
            failure_detail = serialize_failure(RuntimeError("Player policy revision changed during rollout collection"),
                                               stage="policy_revision", component="player_capture", provider_attributable=False)
            error = str(failure_detail["error"])
        persisted = read_episode_failure_detail(job.output_dir / "episode.json")
        if persisted is not None:
            failure_detail, error = persisted, str(persisted.get("error") or error)

        capture: CapturedRollout | None = None
        audit: dict[str, Any] | None = None
        if (job.output_dir / "episode.json").is_file() and records:
            try:
                capture = harvest_episode(records, episode_root=job.output_dir, group_identity_key=identity.key)
                audit_path = job.output_dir / "acceptance_audit.json"
                audit = _read_json(audit_path) if audit_path.is_file() else None
                if error is None and capture is not None:
                    capture_store.put(capture, episode_id=job.episode_id)
                    persist_accepted_capture(run_dir=run_dir, episode_id=job.episode_id,
                                             capture=capture, provenance=provenance)
                else:
                    capture = None
                    write_policy_provenance(job.output_dir, provenance)
                    upsert_capture_index(run_dir, episode_id=job.episode_id, rollout_root=job.output_dir,
                                         accepted=False, policy_kind=provenance.policy_kind,
                                         policy_ref=provenance.policy_ref, group_identity_key=identity.key)
            except Exception as caught:
                detail = serialize_failure(caught, stage="tinker_capture_audit", component="capture_audit")
                error = f"{error}; harvest: {detail['error']}" if error else f"harvest: {detail['error']}"
                failure_detail = failure_detail or detail
                capture = None
        elif error is None and not records:
            # A finished GM episode with no Player capture is a local failure,
            # not evidence that Codex is down. Persist explicit attribution so
            # live and durable-replayed campaign waves remain health-neutral.
            failure_detail = serialize_failure(
                RuntimeError(f"episode produced no Player samples from the {provenance.backend} endpoint"),
                stage="tinker_record_collection", component="player_capture",
                provider_attributable=False,
            )
            error = str(failure_detail["error"])

        outcome: dict[str, Any] = {
            "episode_id": job.episode_id, "world_id": job.world_id, "flavor": job.flavor,
            "attempt": job.attempt, "rollout_dir": str(job.output_dir), "error": error,
            "provider_infrastructure_failure": bool(error) and bool(failure_detail)
            and failure_detail.get("provider_attributable") is True,
            "audit": audit, "live_tinker_record_count": len(records),
            "live_policy_record_count": len(records), "policy_backend": provenance.backend,
            "wall_seconds": round(time.monotonic() - started, 3),
            "token_usage": token_usage_for_records(records),
            "accepted_count": int(capture is not None), "group_identity_key": identity.key,
            "policy_kind": provenance.policy_kind, "policy_ref": provenance.policy_ref,
            "policy_rft_updates_completed": provenance.rft_updates_completed,
            "gm_runtime": job.gm_runtime, "gm_model": job.gm_model, "gm_provider": job.gm_provider,
            "gm_fallback_applied": job.gm_fallback_applied,
        }
        outcome.update(outcome_failure_fields(failure_detail))
        write_durable_outcome(job.output_dir, outcome)
        emit_simple(episode_id=job.episode_id, phase="campaign_job_done",
                    message=f"{provenance.backend} job finished accepted={outcome['accepted_count']} error={error is not None}",
                    world_id=job.world_id, accepted_count=outcome["accepted_count"], error=error)
        return outcome

    async def run_tinker_job(job: RolloutJob, args: argparse.Namespace) -> dict[str, Any]:
        endpoint.open_episode(job.episode_id)
        try:
            return await collect_job(job, args)
        finally:
            # Fence before draining. In-flight HTTP handlers retain ownership
            # until exit, then release pins without resurrecting episode memory.
            endpoint.close_episode(job.episode_id)

    return run_tinker_job


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))
