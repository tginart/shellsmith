"""Turn one finished episode plus its endpoint records into a trainable capture.

Architecture map
----------------
    harvest_episode(records, episode_root, group_identity_key)
        records_consumed_by_pi       keep only responses Pi's vat log shows it consumed
        persist_tinker_records_for_protocol
                                     logs/tinker_rollout_records.jsonl + fill
                                     model_protocol.requests (the audit needs them)
        _ensure_acceptance_audit     recompute acceptance_audit.json from current evidence
        accepted_for_training?       no -> None (logged); yes ->
        captures_from_vat_log        align records with world turns
        -> CapturedRollout           input to credit.to_tinker_bridge.build_tinker_group

Why this exists
---------------
The GW-specific half of the original monorepo's Verifiers capture code,
ported without Verifiers: records now come from ``policy.server.PlayerEndpoint``.

Main callers:
    ``shellsmith episode --player tinker`` and (M4) the trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from shellsmith.credit.to_tinker_bridge import CapturedRollout, captures_from_vat_log
from shellsmith.credit.trajectory_pi import align_pi_samples, pi_sample_events
from shellsmith.episode.acceptance_audit import write_audit
from shellsmith.episode.protocol_integrity import (
    model_protocol_requests_from_rollout_rows,
    populate_model_protocol_requests,
)


def _accepted_for_training(episode_root: Path) -> bool:
    path = episode_root / "acceptance_audit.json"
    if not path.is_file():
        raise RuntimeError(f"acceptance audit is absent after write_audit: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"acceptance audit must be a JSON object: {path}")
    return payload.get("accepted_for_training") is True


def _protocol_rows_from_records(records: Sequence[Any], *, episode_id: str) -> list[dict[str, Any]]:
    """JSON objects ``populate_model_protocol_requests`` can turn into ``requests``.

    Live Tinker records carry ``protocol_row`` (the same object the proxy writes
    to JSONL). Tests may pass a dict or a FakeRecord with only ``request_id``.
    """

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        row = getattr(record, "protocol_row", None)
        if isinstance(row, dict) and str(row.get("request_id") or "").strip():
            rows.append(dict(row))
            continue
        if isinstance(record, dict) and str(record.get("request_id") or "").strip():
            rows.append(dict(record))
            continue
        request_id = getattr(record, "request_id", None)
        if not isinstance(request_id, str) or not request_id.strip():
            raise RuntimeError(
                f"shellsmith harvest record {index} for episode {episode_id} "
                "has no request_id; cannot fill model_protocol"
            )
        rows.append(
            {
                "request_id": request_id,
                "sampled_tokens": list(getattr(record, "sampled_tokens", []) or []),
                "sampled_logprobs": list(getattr(record, "sampled_logprobs", []) or []),
                "content": str(getattr(record, "content", "") or ""),
                "reasoning_content": str(getattr(record, "reasoning_content", "") or ""),
            }
        )
    return rows


def _record_protocol_row(record: Any) -> dict[str, Any] | None:
    row = getattr(record, "protocol_row", None)
    if isinstance(row, dict):
        return row
    if isinstance(record, dict):
        return record
    return None


def records_consumed_by_pi(
    records: Sequence[Any], *, episode_root: Path, episode_id: str
) -> list[Any]:
    """Select the ordered proxy responses confirmed by Pi's vat journal.

    Unconsumed responses remain in the attempt replay bundle. Missing consumed
    responses are fatal because their training inputs cannot be reconstructed.
    Compare only matches that leave room for every later journal event. Repeated
    genuine turns can share a signature without being retries. If multiple full
    ordered alignments differ in sampled tokens, request id, or log probabilities,
    matching is ambiguous and CISPO must not guess which action Pi consumed.
    """
    values = list(records)
    # Lightweight unit-test records and legacy callers have no protocol row.
    # Their existing strict count alignment remains authoritative.
    if not values or any(_record_protocol_row(record) is None for record in values):
        return values
    events = pi_sample_events(episode_root)
    alignment = align_pi_samples(values, events, episode_id=episode_id, allow_unconsumed=True)
    selected = [values[record_index] for _, record_index in alignment]
    skipped = len(values) - len(selected)
    if skipped:
        print(
            f"shellsmith_harvest retained_unconsumed_proxy_responses episode={episode_id} "
            f"count={skipped}",
            flush=True,
        )
    return selected


def persist_tinker_records_for_protocol(
    episode_root: Path,
    records: Sequence[Any],
    *,
    episode_id: str,
) -> None:
    """Copy this rollout's Tinker protocol rows into the episode, then fill requests.

    Why this exists:
        The endpoint's records live in memory; without this copy the audit sees the orchestrator's empty
        ``model_protocol.requests`` stub and refuses the episode.

    Main callers:
        ``harvest_episode``, before ``write_audit``.
    """

    if not records:
        raise RuntimeError(
            f"shellsmith harvestable rollout {episode_id} has zero Tinker records; "
            "the Player must sample through the Shellsmith Tinker endpoint"
        )
    rows = _protocol_rows_from_records(records, episode_id=episode_id)
    logs_dir = episode_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    backend = getattr(records[0], "backend", "tinker")
    destination = logs_dir / ("policy_rollout_records.jsonl" if backend in {"mlx","studio"} else "tinker_rollout_records.jsonl")
    try:
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    except TypeError as error:
        raise RuntimeError(
            f"shellsmith harvest cannot JSON-encode protocol rows for "
            f"episode {episode_id}: {error}"
        ) from error
    destination.write_text(payload, encoding="utf-8")
    populate_model_protocol_requests(
        episode_root,
        model_protocol_requests_from_rollout_rows(rows),
    )
    print(
        f"shellsmith_harvest persisted protocol rows={len(rows)} episode={episode_id} path={destination}",
        flush=True,
    )


def _ensure_acceptance_audit(episode_root: Path) -> None:
    # Protocol requests were just replaced; an existing decision used old inputs.
    audit_path = episode_root / "acceptance_audit.json"
    print(f"shellsmith_harvest writing acceptance_audit={audit_path}", flush=True)
    write_audit(episode_root)


def harvest_episode(
    records: Sequence[Any], *, episode_root: Path, group_identity_key: str
) -> CapturedRollout | None:
    """Return the episode's trainable capture, or None when the audit rejects it.

    Steps:
        1. Keep the records Pi consumed (fatal if one it consumed is missing).
        2. Copy their protocol rows into the episode and write the acceptance audit.
        3. If the audit did not accept the episode for training, log and return None.
        4. Align the records with world turns into a ``CapturedRollout``.

    Main callers:
        ``shellsmith episode --player tinker``; the trainer's rollout runner (M4).
    """
    episode_id = episode_root.name
    # Retain lightweight diagnostics even for responses the client rejected.
    # Consumed-only training records deliberately exclude those responses.
    diagnostics = []
    for record in records:
        row = _record_protocol_row(record) or {}
        if row.get("generation_notices") or row.get("protocol_error"):
            diagnostics.append({key: row.get(key) for key in (
                "request_id", "reasoning_effort", "reasoning_limit", "response_limit",
                "thinking_tokens", "response_tokens", "generation_notices", "protocol_error",
                "finish_reason")})
    logs = episode_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "generation_diagnostics.json").write_text(
        json.dumps({"schema": "shellsmith.generation_diagnostics.v1", "requests": diagnostics}, indent=2)
        + "\n", encoding="utf-8")
    consumed = records_consumed_by_pi(list(records), episode_root=episode_root, episode_id=episode_id)
    persist_tinker_records_for_protocol(episode_root, consumed, episode_id=episode_id)
    _ensure_acceptance_audit(episode_root)
    if not _accepted_for_training(episode_root):
        print(f"shellsmith_harvest skip rejected-by-audit episode={episode_id} root={episode_root}", flush=True)
        return None
    return CapturedRollout(
        rollout_root=episode_root,
        group_identity_key=group_identity_key,
        captures=captures_from_vat_log(records=consumed, rollout_root=episode_root),
    )
