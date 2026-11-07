#!/usr/bin/env python3
"""Render durable Studio/Shellsmith evidence as a self-contained offline report.

No Flask, model runtime, network access, or third-party packages are required.
Missing artifacts remain missing; process exit success is never treated as a
training or behavior qualification. SVGs show measured observations only.

Inputs (all optional beneath --run-root):
  micro-report.json, trainer_state.json, metrics.jsonl, campaign-report.json,
  compute-final.json, compute/micro-operations.json, instance.json, phase-events.jsonl,
  telemetry.jsonl, state/inference-telemetry.jsonl, and recursively discovered
  episode.json enclosures.
Trainer artifacts may instead live under train/, with campaigns and episodes
under train/segments/. Preflight archive siblings are excluded from discovery.

The operation file accepts {"operations": [...]} or an array. Each receipt may
include id/kind/status/result, submitted_prompt_tokens, reserved_completion_tokens,
and started_at/finished_at (epoch seconds or ISO timestamps). Metrics live under
result.execution_metrics or result.metrics. Phase events accept phase/status,
started_at/finished_at, and/or seconds. Telemetry is a JSONL sequence of actual
metrics, optionally wrapped in {"timestamp": ..., "metrics": {...}}.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shellsmith.jobs.studio_monitor import effective_run_state

GIB = 1024 ** 3
MISSING = "Not recorded"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        return float(value) if math.isfinite(value) else None
    except OverflowError:
        return None


def first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def rows(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def fmt(value: Any, *, unit: str = "") -> str:
    n = number(value)
    if n is None:
        return MISSING
    result = f"{int(n):,}" if n == int(n) else f"{n:,.3f}".rstrip("0").rstrip(".")
    return result + unit


def gib(value: Any) -> str:
    n = number(value)
    return f"{n / GIB:,.2f} GiB" if n is not None else MISSING


def verdict(value: Any) -> str:
    return "Yes" if value is True else "No" if value is False else MISSING


def stamp(value: Any) -> float | None:
    if number(value) is not None:
        return number(value)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            # A timezone-free wall clock is ambiguous. Do not invent its zone.
            return parsed.timestamp() if parsed.tzinfo is not None else None
        except ValueError:
            pass
    return None


def pretty(value: Any) -> str:
    """Bound display size while making every omission explicit."""
    def bounded(item: Any) -> Any:
        if isinstance(item, dict):
            return {str(k): bounded(v) for k, v in item.items()}
        if isinstance(item, list):
            if len(item) > 64:
                return {"display_note": "Array display abridged; source artifact retains every item",
                        "item_count": len(item), "first_eight_items": [bounded(x) for x in item[:8]]}
            return [bounded(x) for x in item]
        if isinstance(item, str) and len(item) > 40000:
            return item[:40000] + f"\n[Display truncated: {len(item):,} characters in source artifact]"
        return item
    return esc(json.dumps(bounded(value), ensure_ascii=False, indent=2, allow_nan=False))


class Evidence:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.errors: list[str] = []
        self.sources: list[str] = []

    def read(self, relative: str | Path, *, jsonl: bool = False) -> Any:
        path = self.root / relative
        try:
            path.relative_to(self.root)
            if any(p.is_symlink() for p in (path, *path.parents) if p != self.root.parent):
                self.errors.append(f"Refused symlinked artifact: {relative}")
                return None
            if not path.exists():
                return None
            if not stat.S_ISREG(path.stat().st_mode):
                self.errors.append(f"Refused non-regular artifact: {relative}")
                return None
            text = path.read_text(encoding="utf-8")
            self.sources.append(str(relative))
            if jsonl:
                result = []
                for i, line in enumerate(text.splitlines(), 1):
                    if not line.strip():
                        continue
                    try:
                        result.append(json.loads(line, parse_constant=_reject_constant))
                    except (ValueError, json.JSONDecodeError) as error:
                        self.errors.append(f"{relative}:{i}: invalid JSON: {error}")
                return result
            return json.loads(text, parse_constant=_reject_constant)
        except (OSError, UnicodeError, ValueError) as error:
            self.errors.append(f"Could not read {relative}: {error}")
            return None

    def read_first(self, *candidates: str, jsonl: bool = False) -> Any:
        for relative in candidates:
            path = self.root / relative
            # An existing but malformed primary artifact is an error, not a
            # reason to silently substitute a possibly stale alternate copy.
            if path.exists() or path.is_symlink():
                return self.read(relative, jsonl=jsonl)
        return None


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value {value}")


def _walk(root: Path, filename: str) -> list[Path]:
    found: list[Path] = []
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = sorted(name for name in names if not (parent / name).is_symlink())
        if filename in files and not (parent / filename).is_symlink():
            found.append((parent / filename).relative_to(root))
    return sorted(found)


def join_phase_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join start/finish event rows without assigning time to unfinished work."""
    phases: list[dict[str, Any]] = []
    active: dict[str, int] = {}
    for event in events:
        row = dict(event)
        name = str(first(row.get("phase"), row.get("kind"), "Phase"))
        status = str(row.get("status", ""))
        if row.get("started_at") is not None and row.get("finished_at") is not None:
            phases.append(row)
            active.pop(name, None)
        elif status in ("finished", "stopped", "succeeded", "failed", "complete", "completed") or row.get("finished_at") is not None:
            row["finished_at"] = first(row.get("finished_at"), row.get("timestamp"))
            if name in active:
                index = active.pop(name)
                phases[index] = {**phases[index], **row,
                                 "started_at": phases[index].get("started_at")}
            else:
                phases.append(row)
        elif status in ("started", "starting", "running") or row.get("started_at") is not None:
            row["started_at"] = first(row.get("started_at"), row.get("timestamp"))
            phases.append(row)
            active[name] = len(phases) - 1
        else:
            phases.append(row)
    return phases


def merge_campaigns(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    if not campaigns:
        return {}
    if len(campaigns) == 1:
        return campaigns[0]
    rollouts, seen = [], set()
    for campaign in campaigns:
        for row in rows(campaign.get("rollouts")):
            identity = str(first(row.get("episode_id"), row.get("rollout_dir"), json.dumps(row, sort_keys=True)))
            if identity not in seen:
                rollouts.append(row)
                seen.add(identity)
    merged: dict[str, Any] = {"rollouts": rollouts, "campaigns": campaigns}
    if rollouts:
        merged["rollout_count"] = len(rollouts)
        accepted = [first(row.get("accepted_count"), obj(row.get("audit")).get("accepted_for_training")) for row in rollouts]
        accepted_numbers = [int(value) if isinstance(value, bool) else number(value) for value in accepted]
        if all(value is not None for value in accepted_numbers):
            merged["accepted_count"] = sum(accepted_numbers)
        failures = [row.get("provider_infrastructure_failure") for row in rollouts]
        if all(isinstance(value, bool) for value in failures):
            merged["infrastructure_failure_count"] = sum(failures)
    else:
        for key in ("rollout_count", "accepted_count", "infrastructure_failure_count"):
            values = [number(campaign.get(key)) for campaign in campaigns]
            if all(value is not None for value in values):
                merged[key] = sum(values)
    return merged


def workflow_conflicts(data: dict[str, Any]) -> list[str]:
    claimed = str(first(data["summary"].get("workflow_status"), data["instance"].get("workflow_status"), ""))
    if not (claimed.startswith("passed") or claimed in ("success", "succeeded", "qualified")):
        return []
    conflicts = []
    state = data["instance"].get("status")
    if state in ("failed", "interrupted", "cancelled", "timed_out"):
        conflicts.append(f"Workflow success conflicts with owner status {state}.")
    returncode = first(data["instance"].get("trainer_returncode"), data["owner"].get("trainer_returncode"))
    if number(returncode) is not None and returncode != 0:
        conflicts.append(f"Workflow success conflicts with trainer return code {returncode}.")
    return conflicts


def load_evidence(root: Path) -> dict[str, Any]:
    reader = Evidence(root)
    summary = obj(reader.read("micro-report.json"))
    trainer = obj(reader.read_first("train/trainer_state.json", "trainer_state.json"))
    metrics = rows(reader.read_first("train/metrics.jsonl", "metrics.jsonl", jsonl=True))
    campaign = obj(reader.read_first("campaign-report.json", "campaign_report.json"))
    segment_root = next((reader.root / name for name in ("train/segments", "segments")
                         if (reader.root / name).is_dir() and not (reader.root / name).is_symlink()), None)
    if not campaign and segment_root is not None:
        paths = sorted(set(_walk(segment_root, "campaign_report.json") + _walk(segment_root, "campaign-report.json")))
        campaign = merge_campaigns([obj(reader.read(segment_root.relative_to(reader.root) / path)) for path in paths])
    final = obj(reader.read("compute-final.json"))
    operation_document = reader.read_first("compute/micro-operations.json", "operations.json")
    operations = rows(operation_document if isinstance(operation_document, list)
                      else obj(operation_document).get("operations"))
    # Durable service receipts have unique IDs. Reject conflicts rather than
    # counting retried copies as additional optimizer updates or token work.
    unique, seen = [], {}
    for operation in operations:
        identity = operation.get("id")
        if identity is not None:
            key = str(identity)
            if key in seen:
                if seen[key] != operation:
                    reader.errors.append(f"Conflicting operation receipts for ID {key}")
                continue
            seen[key] = operation
        unique.append(operation)
    operations = unique
    phases = join_phase_events(rows(reader.read("phase-events.jsonl", jsonl=True)))
    telemetry = rows(reader.read("telemetry.jsonl", jsonl=True))
    inference = rows(reader.read("state/inference-telemetry.jsonl", jsonl=True))
    monitor_health = obj(reader.read("monitor-health.json"))
    owner = obj(reader.read("owner.json"))
    instance = obj(reader.read("instance.json"))
    supervisor = obj(reader.read("owner-limit.json"))
    raw_instance = instance
    instance = effective_run_state(instance, supervisor)
    if instance != raw_instance:
        # A collector/owner can die before its final status write. Do not let
        # an earlier summary's "running" (or success) hide a terminal failure.
        summary = {**summary, "workflow_status": instance["workflow_status"],
                   "process_status": instance["status"],
                   "failure": (summary.get("failure") if str(summary.get("workflow_status", "")).startswith("failed")
                               else None) or instance["error"],
                   "terminal_status_source": "owner-limit.json"}
    if supervisor.get("finished") and supervisor.get("owned_surviving_pids") == [] and supervisor.get("cleanup_verified", True):
        # Preserve the durable journal's original status, but do not display
        # stopped, uncommitted work as a live operation. No metrics or compute
        # completion timestamp are fabricated by this display reconciliation.
        operations = [{**operation, "journal_status": operation["status"], "status": "interrupted",
                       "status_source": "terminal_supervisor_receipt"}
                      if operation.get("status") in {"queued", "running"} else operation
                      for operation in operations]
    config = obj(reader.read_first("train/config_snapshot.json", "config_snapshot.json", "config-snapshot.json"))
    episodes = []
    campaign_by_id = {str(x.get("episode_id")): x for x in rows(campaign.get("rollouts"))}
    summary_by_name = {str(x.get("episode")): x for x in rows(summary.get("episode_results"))}
    # Search the current trainer enclosure or the historical exported episodes
    # enclosure. A failed preflight archive next to either cannot add attempts.
    episode_root = segment_root or (reader.root / "episodes" if (reader.root / "episodes").is_dir()
                                    and not (reader.root / "episodes").is_symlink() else reader.root)
    for relative_episode_path in _walk(episode_root, "episode.json"):
        episode_path = episode_root.relative_to(reader.root) / relative_episode_path
        base = episode_path.parent
        episode = obj(reader.read(episode_path))
        audit = obj(reader.read(base / "acceptance_audit.json"))
        judgment = obj(reader.read(base / "world/final/response.json"))
        final_request = obj(reader.read(base / "world/final/request.json"))
        provenance = obj(reader.read(base / "policy_provenance.json"))
        outcome = obj(reader.read(base / "logs/durable_outcome.json"))
        turns = []
        turn_root = reader.root / base / "world/turns"
        if turn_root.is_dir() and not turn_root.is_symlink():
            for turn in sorted(turn_root.iterdir()):
                if turn.is_dir() and not turn.is_symlink():
                    request = reader.read(turn.relative_to(reader.root) / "request.json")
                    response = reader.read(turn.relative_to(reader.root) / "response.json")
                    turns.append({"index": turn.name, "request": request, "response": response})
        campaign_row = campaign_by_id.get(str(episode.get("episode_id")), {})
        fallback = summary_by_name.get(base.name, {})
        accepted = first(audit.get("accepted_for_training"), fallback.get("accepted"))
        reward = first(judgment.get("final_reward"), obj(audit.get("final_task")).get("reward"),
                       fallback.get("reward"))
        episodes.append({"path": str(base), "id": episode.get("episode_id") or base.name,
                         "world": episode.get("world_id"), "status": episode.get("status"),
                         "accepted": accepted, "reward": reward, "episode": episode,
                         "audit": audit, "judgment": judgment, "final_request": final_request,
                         "provenance": provenance, "outcome": outcome, "turns": turns,
                         "campaign": campaign_row})
    return {"root": str(reader.root), "summary": summary, "trainer": trainer,
            "metrics": metrics, "campaign": campaign, "final": final,
            "operations": operations, "phases": phases, "telemetry": telemetry, "inference": inference,
            "owner": owner, "instance": instance, "raw_instance": raw_instance,
            "supervisor": supervisor, "monitor_health": monitor_health,
            "config": config, "episodes": episodes,
            "sources": reader.sources, "errors": reader.errors}


def metric_sets(data: dict[str, Any]) -> list[dict[str, Any]]:
    observations = []
    for operation in data["operations"]:
        result = obj(operation.get("result"))
        measured = obj(first(result.get("execution_metrics"), result.get("metrics")))
        if measured:
            observations.append({"label": str(operation.get("id", "Operation")),
                                 "timestamp": first(operation.get("finished_at"), operation.get("first_observed_at")),
                                 **measured})
    for i, row in enumerate(data["telemetry"], 1):
        measured = obj(first(row.get("metrics"), row.get("execution_metrics"))) or row
        observations.append({"label": str(first(row.get("timestamp"), f"Telemetry {i}")),
                             "timestamp": row.get("timestamp"), **measured})
    final = obj(data["final"].get("metrics"))
    if final:
        observations.append({"label": "Final inspection", **final})
    return observations


def peak(data: dict[str, Any], key: str) -> float | None:
    values = [number(data["summary"].get(key))]
    values.extend(number(row.get(key)) for row in metric_sets(data))
    known = [value for value in values if value is not None]
    return max(known) if known else None


def run_client(data: dict[str, Any]) -> str | None:
    configuration = obj(data["config"].get("config")) or data["config"]
    value = (obj(configuration.get("studio")).get("training_client")
             or obj(configuration.get("run")).get("run_id") or data["instance"].get("run_id"))
    return value if isinstance(value, str) and value else None


def is_run_operation(data: dict[str, Any], operation: dict[str, Any]) -> bool:
    client = run_client(data)
    # Legacy snapshots without a client identity retain their unscoped view.
    # Once a client is known, unknown/other clients cannot count as its work.
    return client is None or operation.get("client") == client


def sample_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for operation in data["operations"]:
        if operation.get("kind") != "sample" or not is_run_operation(data, operation):
            continue
        receipt = obj(operation.get("result"))
        prompt = first(receipt.get("prompt_tokens"), operation.get("submitted_prompt_tokens"))
        reused = receipt.get("prefix_cache_tokens")
        prefill = receipt.get("actual_prefill_tokens")
        if prefill is None and number(prompt) is not None and number(reused) is not None:
            prefill = number(prompt) - number(reused)
        result.append({"id": operation.get("id"), "status": operation.get("status"),
                       "prompt": prompt, "reserved": operation.get("reserved_completion_tokens"),
                       "completion": receipt.get("completion_tokens"), "reused": reused,
                       "prefill": prefill, "prefill_seconds": receipt.get("prefill_seconds"),
                       "decode_seconds": receipt.get("decode_seconds"),
                       "revision": first(receipt.get("policy_revision"), receipt.get("revision")),
                       "precision": receipt.get("cache_precision")})
    return result


def table(headers: list[str], values: list[list[Any]], caption: str = "") -> str:
    if not values:
        return '<p class="missing">No observations recorded.</p>'
    head = "".join(f'<th scope="col">{esc(x)}</th>' for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(MISSING if x is None else x)}</td>" for x in row) + "</tr>"
                   for row in values)
    return (f'<div class="table-wrap"><table><caption>{esc(caption)}</caption>'
            f'<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')


def detail(label: str, value: Any, *, open_: bool = False) -> str:
    if value is None or value == {} or value == []:
        return f'<p class="missing">{esc(label)}: not recorded.</p>'
    return f'<details{" open" if open_ else ""}><summary>{esc(label)}</summary><pre>{pretty(value)}</pre></details>'


def card(value: str, label: str, note: str = "") -> str:
    return f'<div class="metric"><strong>{esc(value)}</strong><span>{esc(label)}</span><small>{esc(note)}</small></div>'


def dot_chart(observations: list[dict[str, Any]], series: list[tuple[str, str, str]], *, unit: float = 1,
              title: str, y_label: str) -> str:
    """Discrete observations only: no connecting or interpolated lines."""
    points = [(i, name, key, color, number(row.get(key)))
              for i, row in enumerate(observations)
              for name, key, color in series if number(row.get(key)) is not None]
    if not points:
        return '<p class="missing">Chart unavailable: no measured observations.</p>'
    width, height, left, right, top, bottom = 900, 260, 60, 24, 22, 45
    maximum = max(value / unit for *_, value in points)
    maximum = maximum if maximum > 0 else 1
    plot_w, plot_h = width - left - right, height - top - bottom
    elements = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
                f'<title>{esc(title)}. Discrete recorded observations; no interpolation.</title>']
    for fraction in (0, .5, 1):
        y = top + plot_h * (1 - fraction)
        elements.append(f'<line x1="{left}" x2="{width-right}" y1="{y}" y2="{y}" class="gridline"/>')
        elements.append(f'<text x="{left-9}" y="{y+4}" text-anchor="end">{maximum*fraction:.1f}</text>')
    count = max(len(observations) - 1, 1)
    for i, name, key, color, value in points:
        x = left + plot_w * i / count
        y = top + plot_h * (1 - (value / unit) / maximum)
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"><title>'
                        f'{esc(name)} · observation {i+1}: {fmt(value / unit)}</title></circle>')
    elements.extend([f'<text x="{left}" y="{height-12}">Observation 1</text>',
                     f'<text x="{width-right}" y="{height-12}" text-anchor="end">Observation {len(observations)}</text>',
                     f'<text x="{left}" y="14">{esc(y_label)}</text>', '</svg>'])
    legend = "".join(f'<span><i style="background:{color}"></i>{esc(name)}</span>' for name, _, color in series)
    return '<figure>' + "".join(elements) + f'<figcaption class="legend">{legend}</figcaption></figure>'


def timeline(phases: list[dict[str, Any]]) -> str:
    timed = [(row, stamp(row.get("started_at")), stamp(row.get("finished_at"))) for row in phases]
    timed = [(row, start, stop) for row, start, stop in timed if start is not None and stop is not None and stop >= start]
    if not timed:
        return '<p class="missing">Wall-clock phase timeline unavailable: start/end timestamps were not recorded.</p>'
    origin = min(start for _, start, _ in timed)
    span = max(stop for _, _, stop in timed) - origin
    span = max(span, .001)
    elements = [f'<svg class="chart" viewBox="0 0 900 {len(timed)*40+42}" role="img" aria-label="Recorded phase timeline">'
                '<title>Recorded phase intervals, aligned by their explicit timestamps. Polling observations are labeled and are not exact execution timestamps.</title>']
    for i, (row, start, stop) in enumerate(timed):
        x = 220 + 650 * (start - origin) / span
        width = max(1, 650 * (stop - start) / span)
        y = 10 + 40*i
        label = str(first(row.get("phase"), row.get("kind"), row.get("id"), "Phase"))
        timing = " · polling observation" if row.get("timing_source") == "poll_observation" else ""
        elements.append(f'<text x="8" y="{y+17}">{esc(label[:34])}</text>'
                        f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="24" rx="5" fill="#9bd5bb">'
                        f'<title>{esc(label)}: {fmt(stop-start)} seconds, {fmt(start-origin)} seconds after first interval{timing}</title></rect>')
    elements.append(f'<text x="220" y="{len(timed)*40+26}">0 seconds</text>'
                    f'<text x="870" y="{len(timed)*40+26}" text-anchor="end">{fmt(span)} seconds</text></svg>')
    return '<figure>' + "".join(elements) + '</figure>'


def render_episodes(data: dict[str, Any]) -> str:
    episodes = data["episodes"]
    values = [[i+1, ep["id"], ep["world"], ep["status"], verdict(ep["accepted"]), fmt(ep["reward"]),
               sum(turn["request"] is not None and turn["response"] is not None for turn in ep["turns"])]
              for i, ep in enumerate(episodes)]
    content = table(["#", "Episode", "World", "Status", "Accepted", "Reward", "Complete turn pairs"], values,
                    "Recorded episode outcomes")
    for i, episode in enumerate(episodes, 1):
        badge = "accepted" if episode["accepted"] is True else "not accepted" if episode["accepted"] is False else "acceptance pending"
        content += f'<details class="episode"><summary><span class="episode-index">{i:02}</span>{esc(episode["id"])} <span class="badge">{esc(badge)}</span></summary><div class="episode-body">'
        content += f'<p class="muted">Evidence enclosure: <code>{esc(episode["path"])}</code></p>'
        content += detail("Episode state and failures", episode["episode"])
        content += detail("Policy provenance", episode["provenance"])
        content += detail("Campaign token usage and duration", {key: episode["campaign"].get(key)
                           for key in ("wall_seconds", "token_usage", "policy_ref", "policy_rft_updates_completed")
                           if key in episode["campaign"]})
        if not episode["turns"]:
            content += '<p class="missing">Actions and observations: no world-turn artifacts were collected.</p>'
        for turn in episode["turns"]:
            request, response = obj(turn["request"]), obj(turn["response"])
            content += f'<details><summary>World turn {esc(turn["index"])} · {"complete pair" if request and response else "incomplete pair"}</summary>'
            content += detail("Actions", first(request.get("actions"), request.get("action")))
            content += detail("Reasoning / trajectory delta", request.get("trajectory_delta"))
            content += detail("Observations", first(response.get("tool_results"), response.get("tool_response")))
            content += detail("World judgment and turn scores", response)
            content += detail("Complete recorded request", turn["request"])
            content += '</details>'
        content += detail("Final judgment", episode["judgment"])
        content += detail("Terminal request / player final response", episode["final_request"])
        content += detail("Acceptance and integrity audit", episode["audit"])
        content += detail("Durable training outcome", episode["outcome"])
        content += '</div></details>'
    return content


def render_tokens(data: dict[str, Any]) -> str:
    samples = sample_rows(data)
    completed_samples = [row for row in samples if row["status"] == "succeeded"]
    tokens = obj(data["summary"].get("tokens"))
    def total(key: str, sample_key: str) -> Any:
        reported = tokens.get(key)
        if reported is not None:
            return reported
        values = [number(row.get(sample_key)) for row in completed_samples]
        return sum(values) if values and all(x is not None for x in values) else None
    capacity = data["instance"].get("context_window")
    content = (f'<p class="muted">Configured total context capacity: <strong>{esc(fmt(capacity))} tokens</strong>, including completion. '
               'This is an admission limit; only the measured requests below establish the context actually exercised.</p>') if capacity is not None else ''
    content += '<div class="metrics">'
    for value, label in [(total("submitted_prompt_tokens", "prompt"), "Submitted prompt tokens"),
                         (total("sampled_completion_tokens", "completion"), "Sampled completion tokens"),
                         (total("prefix_reused_tokens", "reused"), "Reused prefix tokens"),
                         (total("actual_prefill_tokens", "prefill"), "Actual prefill tokens")]:
        content += card(fmt(value), label)
    content += '</div><p class="muted">Totals cover recorded completed sampling receipts. Pending and interrupted requests remain in the table and are excluded from completed-work totals. Prompt totals count each sampling pass; they are not unique history length. Actual prefill is reported directly where available, otherwise calculated as submitted prompt minus reused prefix. Reserved completion is admission capacity, not generated tokens.</p>'
    chart_rows = [{"prompt": row["prompt"], "completion": row["completion"], "reused": row["reused"]} for row in samples]
    content += dot_chart(chart_rows, [("Submitted prompt", "prompt", "#9bd5bb"), ("Reused prefix", "reused", "#81aee6"),
                                    ("Completion", "completion", "#edc584")], title="Recorded sampling token counts", y_label="Tokens")
    content += table(["Request", "Status", "Prompt", "Completion", "Prompt + sampled", "Reserved completion", "Reused prefix", "Actual prefill", "Policy revision", "KV precision"],
                     [[row["id"], row["status"], fmt(row["prompt"]), fmt(row["completion"]),
                       fmt(number(row["prompt"]) + number(row["completion"])) if number(row["prompt"]) is not None and number(row["completion"]) is not None else MISSING,
                       fmt(row["reserved"]), fmt(row["reused"]), fmt(row["prefill"]), fmt(row["revision"]), row["precision"]]
                      for row in samples], "Exact sampling requests")
    content += table(["Training operation", "Status", "Prompt tokens", "Completion targets", "Loss-bearing targets"],
                     [[operation.get("id"), operation.get("status"),
                       fmt(first(obj(obj(operation.get("result")).get("forward_metrics")).get("prompt_tokens"),
                                 operation.get("submitted_prompt_tokens"))),
                       fmt(first(obj(obj(operation.get("result")).get("forward_metrics")).get("completion_tokens"),
                                 operation.get("reserved_completion_tokens"))),
                       fmt(obj(obj(operation.get("result")).get("forward_metrics")).get("loss_tokens"))]
                      for operation in data["operations"] if operation.get("kind") == "forward_backward" and is_run_operation(data, operation)],
                     "Exact training workload")
    content += '<p class="muted">For training, counts without a completed receipt describe the exact submitted prompt IDs and supplied completion target IDs, not GPU-processed work. The collector’s reserved_completion_tokens field holds supplied target count for forward/backward; it represents generation capacity only for inference. Loss-bearing target counts remain unknown until recorded in a completed receipt.</p>'
    content += detail("Reported training token counts", {key: tokens.get(key) for key in ("training_prompt_tokens", "training_completion_tokens") if key in tokens})
    if run_client(data) is not None:
        content += f'<p class="muted">Run token totals and training numerics cover client <code>{esc(run_client(data))}</code> only. Preflight, control, other-client and unassigned operations are excluded.</p>'
        auxiliary = [operation for operation in data["operations"] if not is_run_operation(data, operation)]
        content += table(["Client", "Operation", "Kind", "Status", "Prompt tokens", "Completion tokens"],
                         [[operation.get("client"), operation.get("id"), operation.get("kind"), operation.get("status"),
                           fmt(first(obj(operation.get("result")).get("prompt_tokens"), obj(obj(operation.get("result")).get("forward_metrics")).get("prompt_tokens"))),
                           fmt(first(obj(operation.get("result")).get("completion_tokens"), obj(obj(operation.get("result")).get("forward_metrics")).get("completion_tokens")))]
                          for operation in auxiliary], "Preflight and other-client operations (excluded from run totals)")
    return content


def throughput_windows(observations: list[dict[str, Any]], seconds: float) -> list[dict[str, Any]]:
    """Coalesce whole measured intervals; never interpolate across gaps/restarts."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Throughput window seconds must be finite and positive.")
    result, group, duration = [], [], 0.0

    def finish():
        nonlocal duration
        if not group:
            return
        duration = sum(row["seconds"] for row in group)
        tokens = sum(row["generated_tokens"] for row in group)
        result.append({"session_id": group[0].get("session_id"),
                       "started_at": group[0]["started_at"], "finished_at": group[-1]["finished_at"],
                       "seconds": duration, "generated_tokens": tokens, "tokens_per_second": tokens / duration,
                       "average_active_requests": sum(row["average_active_requests"] * row["seconds"] for row in group) / duration,
                       "peak_active_requests": max(row["peak_active_requests"] for row in group),
                       "average_queued_requests": sum(row["average_queued_requests"] * row["seconds"] for row in group) / duration,
                       "active_requests": group[-1]["active_requests"], "queued_requests": group[-1]["queued_requests"]})
        group.clear()
        duration = 0.0

    for row in observations:
        required = ("seconds", "generated_tokens", "started_at", "finished_at", "average_active_requests",
                    "peak_active_requests", "average_queued_requests", "active_requests", "queued_requests")
        if any(number(row.get(key)) is None for key in required) or row["seconds"] <= 0:
            finish()
            continue
        if group and (row.get("session_id") != group[-1].get("session_id")
                      or abs(row["started_at"] - group[-1]["finished_at"]) > .001):
            finish()
        group.append(row)
        duration += row["seconds"]
        if duration >= seconds:
            finish()
    finish()
    return result


def utc_label(value: float) -> str:
    try:
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return "Invalid timestamp"


def render_throughput(data: dict[str, Any], window_seconds: float) -> str:
    observations = throughput_windows(data.get("inference", []), window_seconds)
    content = '<h3>Local model throughput and concurrency</h3>'
    content += ('<p class="muted">Output tokens per wall-clock second, including prefill, training interleaving, and idle time. '
                'Active requests have entered local sampling and have not finished generating; queued requests are separate. '
                'Forced-token scoring and prompt tokens are excluded. Failed requests retain tokens already generated. '
                f'Windows combine whole measured intervals to a target of {esc(fmt(window_seconds))} seconds; '
                'the table reports actual duration. Concurrency averages are weighted by time. Missing intervals and service restarts are never bridged.</p>')
    if not observations:
        return content + '<p class="missing">Local inference intervals: not recorded.</p>'
    content += dot_chart(observations, [("Generated tokens/s", "tokens_per_second", "#9bd5bb")],
                         title="Local output throughput by measured window", y_label="Tokens / second")
    content += dot_chart(observations, [("Average active", "average_active_requests", "#81aee6"),
                                       ("Peak active", "peak_active_requests", "#9bd5bb"),
                                       ("Average queued", "average_queued_requests", "#edc584")],
                         title="Local sampling concurrency over the same windows", y_label="Requests")
    content += table(["Window start (UTC)", "Measured seconds", "Output tokens", "Tokens/s", "Active average", "Active peak", "Active at end", "Queued average", "Queued at end"],
                     [[utc_label(row["started_at"]),
                       *[fmt(row[key]) for key in ("seconds", "generated_tokens", "tokens_per_second", "average_active_requests",
                                                  "peak_active_requests", "active_requests", "average_queued_requests", "queued_requests")]]
                      for row in observations], "Throughput and request counts over identical measured time windows")
    if any(row.get("logging_errors", 0) for row in data.get("inference", [])):
        content += '<p class="missing">Inference telemetry encountered logging errors; some intervals may be missing.</p>'
    return content


def render_memory(data: dict[str, Any]) -> str:
    observations = metric_sets(data)
    content = '<div class="metrics">' + card(gib(peak(data, "allocator_peak_bytes")), "Allocator peak", "Maximum recorded peak")
    content += card(gib(peak(data, "rss_peak_bytes")), "Process RSS peak", "Maximum recorded peak")
    content += card(gib(peak(data, "swap_growth_bytes")), "Swap growth", "Maximum reported growth")
    cache_values = [number(obj(row.get("prefix_cache")).get("bytes")) for row in observations]
    cache_values = [x for x in cache_values if x is not None]
    content += card(gib(max(cache_values) if cache_values else obj(data["summary"].get("prefix_cache")).get("bytes")),
                    "Prefix cache occupied", "Maximum observed occupancy") + '</div>'
    chart_rows = [{**row, "prefix_cache_bytes": obj(row.get("prefix_cache")).get("bytes")} for row in observations]
    content += dot_chart(chart_rows, [("Allocator active", "allocator_active_bytes", "#9bd5bb"),
                                     ("RSS", "rss_bytes", "#81aee6"), ("Prefix cache", "prefix_cache_bytes", "#edc584")],
                         unit=GIB, title="Observed memory measurements", y_label="GiB")
    content += '<p class="muted">Dots are actual observations, shown in evidence order. No memory values between samples are inferred. Swap growth uses the service’s recorded baseline; missing growth is never treated as zero.</p>'
    content += table(["Observation", "Process", "Allocator active", "Allocator peak", "RSS", "RSS peak", "Swap growth", "Prefix cache"],
                     [[i+1, row.get("pid"), gib(row.get("allocator_active_bytes")), gib(row.get("allocator_peak_bytes")),
                       gib(row.get("rss_bytes")), gib(row.get("rss_peak_bytes")), gib(row.get("swap_growth_bytes")),
                       gib(obj(row.get("prefix_cache")).get("bytes"))] for i, row in enumerate(observations)],
                     "Actual memory observations")
    precision = first(data["summary"].get("precision"), obj(data["final"].get("metrics")).get("precision"),
                      next((row.get("precision") for row in observations if row.get("precision")), None))
    content += detail("Recorded precision", precision)
    content += detail("Final prefix-cache counters", first(obj(data["final"].get("metrics")).get("prefix_cache"), data["summary"].get("prefix_cache")))
    return content


def render_execution(data: dict[str, Any]) -> str:
    phases = data["phases"]
    instance = data["instance"]
    content = table(["Owner state", "Recorded value"], [
        ["Status", instance.get("status")], ["Phase", instance.get("phase")],
        ["Workflow outcome", instance.get("workflow_status")],
        ["Original launch (deadline origin)", instance.get("original_launch_at")],
        ["Started", instance.get("started")], ["Ready", instance.get("ready")],
        ["Finished", instance.get("finished")], ["Deadline", instance.get("deadline_at")],
        ["Configured attempt cap", fmt(instance.get("attempts"))],
        ["Configured concurrent GMs", fmt(instance.get("concurrent_gms"))],
        ["Configured maximum updates", fmt(instance.get("max_updates"))],
        ["Owner / service / trainer PID", " / ".join(fmt(instance.get(key)) for key in ("owner_pid", "service_pid", "trainer_pid"))],
    ], "Run owner snapshot") if instance else ''
    content += timeline(phases or data["operations"])
    if any(operation.get("timing_source") == "poll_observation" for operation in data["operations"]):
        content += '<p class="muted">Operation start and finish observations come from polling. They bound observed status changes at the recorded polling resolution; they are not exact GPU execution timestamps.</p>'
    values = []
    for phase in phases:
        start, stop = stamp(phase.get("started_at")), stamp(phase.get("finished_at"))
        duration = first(phase.get("seconds"), stop-start if start is not None and stop is not None else None)
        values.append([first(phase.get("phase"), phase.get("kind")), phase.get("status"), fmt(duration, unit=" s"),
                       phase.get("started_at"), phase.get("finished_at")])
    content += table(["Phase", "Status", "Duration", "Start", "End"], values, "Recorded phase intervals")
    content += table(["Step", "Rollout / update seconds", "Training data rows", "Skipped / decision"],
                     [[row.get("step"), fmt(row.get("seconds")), row.get("datums"),
                       first(row.get("skipped"), row.get("decision"))] for row in data["metrics"]], "Durable trainer metrics")
    content += table(["Operation", "Client", "Kind", "Status", "Measured execution", "Prefill", "Decode", "Revision"],
                     [[operation.get("id"), operation.get("client"), operation.get("kind"), operation.get("status"),
                       fmt(first(obj(operation.get("result")).get("seconds"),
                                 obj(first(obj(operation.get("result")).get("execution_metrics"), obj(operation.get("result")).get("metrics"))).get("seconds")), unit=" s"),
                       fmt(obj(operation.get("result")).get("prefill_seconds"), unit=" s"),
                       fmt(obj(operation.get("result")).get("decode_seconds"), unit=" s"),
                       first(obj(operation.get("result")).get("revision"), obj(operation.get("result")).get("policy_revision"))]
                      for operation in data["operations"]], "Operation receipts in recorded order")
    content += '<p class="muted">Execution durations may overlap and batched requests may share work. Their sum is not wall time. Missing timestamps are shown as missing; receipt order is not a fabricated timeline.</p>'
    observed = [operation for operation in data["operations"] if operation.get("started_at") is not None or operation.get("finished_at") is not None]
    content += table(["Operation", "Observed start", "Observed finish", "Timing source", "Polling interval"],
                     [[operation.get("id"), operation.get("started_at"), operation.get("finished_at"),
                       operation.get("timing_source"), fmt(operation.get("poll_interval_seconds"), unit=" s")]
                      for operation in observed], "Operation timing observations") if observed else ''
    return content


def render_publication(data: dict[str, Any]) -> str:
    trainer, final, summary = data["trainer"], data["final"], data["summary"]
    content = table(["Evidence", "Recorded value"], [
        ["Committed updates (summary / trainer / compute)", " / ".join(fmt(x) for x in (summary.get("committed_updates"), trainer.get("rft_updates_completed"), final.get("optimizer_steps")))],
        ["Trainer process state", trainer.get("status")], ["Optimizer update in flight", verdict(trainer.get("optimizer_update_in_flight"))],
        ["Active rollout policy", trainer.get("active_policy_ref")],
        ["Frozen base unchanged", verdict(summary.get("frozen_base_unchanged"))],
        ["Service stopped", verdict(summary.get("service_stopped"))],
        ["Local processes stopped", verdict(summary.get("local_processes_stopped"))],
    ], "Update and policy state")
    relevant = [operation for operation in data["operations"] if is_run_operation(data, operation) and operation.get("kind") in ("optim_step", "optimizer", "save", "restore", "publish", "create")]
    content += table(["Operation", "Kind", "Status", "Revision", "Durable checkpoint", "Published policy"],
                     [[operation.get("id"), operation.get("kind"), operation.get("status"), obj(operation.get("result")).get("revision"),
                       obj(operation.get("result")).get("checkpoint"), obj(operation.get("result")).get("policy")]
                      for operation in relevant], "Durable operation receipts")
    content += '<p class="muted">A receipt reports the service’s durable commit. This page does not independently open remote checkpoints or repeat recovery checks. A revision-0 publication is an initial policy, not a training update.</p>'
    numerical = []
    for operation in data["operations"]:
        if not is_run_operation(data, operation):
            continue
        result = obj(operation.get("result"))
        forward = obj(result.get("forward_metrics"))
        optimizer = obj(result.get("optim_metrics"))
        if forward or optimizer:
            numerical.append([operation.get("id"), operation.get("status"),
                              fmt(forward.get("loss")), forward.get("reduction"),
                              fmt(first(optimizer.get("gradient_norm"), optimizer.get("grad_norm"))),
                              fmt(optimizer.get("optimizer_steps"))])
    content += table(["Operation", "Status", "Recorded loss", "Reduction", "Gradient norm", "Optimizer step"],
                     numerical, "Recorded training numerics")
    content += detail("Trainer checkpoint cursor", trainer.get("checkpoints"))
    content += detail("Committed group IDs", trainer.get("committed_group_ids"))
    content += detail("Final adapter / optimizer / RNG hashes", final.get("state_hashes"))
    content += detail("Recovery results (when recorded)", summary.get("recovery"))
    return content


CSS = r"""
:root{color-scheme:dark;--paper:#0e1418;--card:#161f24;--ink:#e5ece9;--muted:#a6b5b1;--line:#2c3b3f;--green:#9bd5bb;--amber:#edc584;--blue:#81aee6;--red:#f0a6a6}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:85px}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--green);text-underline-offset:4px}a:focus-visible,summary:focus-visible{outline:3px solid var(--green);outline-offset:5px}.wrap{width:min(1260px,calc(100% - 64px));margin:0 auto}.topbar{position:sticky;top:0;z-index:4;background:#0e1418f5;border-bottom:1px solid var(--line)}.topbar .wrap{display:flex;align-items:center;justify-content:space-between;gap:20px;min-height:68px}.brand{font-weight:750;letter-spacing:-.03em;text-decoration:none;color:var(--ink)}nav{display:flex;gap:22px}nav a{font-size:12px;color:var(--muted);text-decoration:none}.hero{padding:58px 0 26px}.eyebrow{font-size:11px;font-weight:750;letter-spacing:.14em;text-transform:uppercase;color:var(--green);margin:0 0 13px}h1{font-size:clamp(38px,5vw,65px);line-height:1.1;letter-spacing:-.055em;margin:0 0 22px;max-width:950px}h2{font-size:27px;letter-spacing:-.035em;line-height:1.2;margin:0 0 18px}h3{font-size:17px;margin:0 0 10px}.lead{font-size:18px;color:var(--muted);max-width:860px;margin:0 0 22px}.status,.badge{display:inline-block;font-size:11px;font-weight:700;border:1px solid #35564b;background:#1c332c;color:var(--green);padding:4px 10px;border-radius:100px}.status.warn{border-color:#665236;background:#30291e;color:var(--amber)}.hero-row{display:flex;justify-content:space-between;gap:25px;align-items:start}.muted,.missing{color:var(--muted)}.missing{font-size:13px;font-style:italic}p{margin:0 0 17px}.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);border-radius:18px;overflow:hidden;background:var(--card);margin:24px 0}.metric{padding:22px;border-right:1px solid var(--line)}.metric:last-child{border-right:0}.metric strong{font-size:clamp(22px,2.9vw,38px);letter-spacing:-.04em;line-height:1.25;display:block;overflow-wrap:anywhere}.metric span{font-size:12px;font-weight:650;display:block;margin-top:8px}.metric small{font-size:11px;color:var(--muted);display:block;margin-top:3px}.callout{border:1px solid #665236;background:#30291e;border-radius:15px;padding:22px 25px;color:#dcc7a5;margin:16px 0 35px}.callout strong{color:var(--amber)}section{margin:0 0 44px;scroll-margin-top:86px}.section-title{display:flex;align-items:center;gap:14px;margin-bottom:10px}.section-title span{font:650 12px/1 ui-monospace,SFMono-Regular,Consolas,monospace;border:1px solid #35564b;background:#1c332c;color:var(--green);width:34px;height:34px;border-radius:10px;display:grid;place-items:center}.section-title h2{margin:0}.panel{border:1px solid var(--line);border-radius:18px;background:var(--card);padding:25px}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:13px;margin:20px 0}table{width:100%;border-collapse:collapse;text-align:left;font-size:12px;line-height:1.55}caption{text-align:left;padding:13px 16px;background:#1c332c;color:var(--green);font-weight:700;font-size:11px;letter-spacing:.05em;text-transform:uppercase}th,td{padding:11px 15px;border-top:1px solid var(--line);vertical-align:top;overflow-wrap:anywhere;max-width:460px}th{color:var(--muted);font-weight:650}td{font-variant-numeric:tabular-nums}details{border-top:1px solid var(--line);padding:13px 0;margin:9px 0}summary{cursor:pointer;font-size:13px;font-weight:650;color:var(--green);overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;padding:18px;background:#10181d;border:1px solid var(--line);border-radius:10px;max-height:580px;overflow:auto}code{font:12px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.episode{border:1px solid var(--line);background:var(--card);border-radius:14px;padding:17px 20px;margin:12px 0}.episode>summary{font-size:14px;color:var(--ink)}.episode-index{color:var(--green);font:650 12px ui-monospace,SFMono-Regular,Consolas,monospace;margin-right:14px}.episode .badge{margin-left:12px;font-size:10px}.episode-body{padding-top:20px}.episode-body details{padding-left:12px}.chart{width:100%;display:block;overflow:visible}.chart text{fill:var(--muted);font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.gridline{stroke:var(--line);stroke-width:1}figure{margin:24px 0;padding:20px 14px 14px;background:#10181d;border:1px solid var(--line);border-radius:14px}.legend{display:flex;gap:20px;flex-wrap:wrap;font-size:11px;color:var(--muted);padding:0 10px}.legend span{display:flex;gap:7px;align-items:center}.legend i{width:8px;height:8px;border-radius:50%;display:inline-block}footer{border-top:1px solid var(--line);padding:25px 0 40px;font-size:11px;color:var(--muted)}.source-list{columns:2;list-style:none;padding:0;font-size:12px}.source-list li{break-inside:avoid;padding:4px 0;overflow-wrap:anywhere}.skip{position:absolute;top:-80px;left:15px;background:var(--card);padding:10px;z-index:99}.skip:focus{top:8px}
@media(max-width:800px){.wrap{width:calc(100% - 32px)}nav{gap:12px}nav a{font-size:11px}.metrics{grid-template-columns:1fr 1fr}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}.hero{padding-top:32px}.hero-row{display:block}.hero-row .status{margin-bottom:15px}.panel{padding:18px}.source-list{columns:1}.chart text{font-size:13px}}
@media(max-width:480px){nav a:nth-child(n+3){display:none}.metric{padding:17px}.metric strong{font-size:26px}.episode .badge{display:block;width:fit-content;margin:8px 0 0}.chart{min-width:500px}figure{overflow:auto}.hero h1{font-size:40px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{:root{color-scheme:light;--paper:#fff;--card:#fff;--ink:#162b2c;--muted:#586a68;--line:#dce3dc;--green:#226653}.topbar{position:static;background:#fff}nav{display:none}.wrap{width:100%}.hero{padding:24px 0}.metric strong{font-size:24px}.panel,.episode{break-inside:avoid}.callout{background:#faf5ea;color:#574526}.callout strong{color:#775621}pre,figure{background:#fff}.chart text{fill:#586a68}.badge,.status{background:#f0f7f2;color:#226653}.source-list{columns:1}a{color:#226653}}
"""


def render_report(data: dict[str, Any], *, title: str | None = None, preview: bool = False,
                  throughput_window_seconds: float = 10) -> str:
    summary, trainer, campaign = data["summary"], data["trainer"], data["campaign"]
    title = title or f'Studio · {Path(data["root"]).name}'
    status = first(summary.get("workflow_status"), data["instance"].get("workflow_status"))
    conflicts = workflow_conflicts(data)
    if conflicts:
        status = "failed · conflicting workflow receipt"
    if status is None:
        # A completed coordinator with failed episodes is not a successful run.
        failures = number(campaign.get("infrastructure_failure_count"))
        if failures is not None and failures > 0:
            status = "infrastructure failures recorded"
        elif data["instance"].get("status") is not None:
            status = f'{data["instance"]["status"]} · workflow outcome not yet recorded'
        else:
            status = "workflow outcome not yet recorded"
    active = bool(data["instance"] and not data["instance"].get("finished"))
    attempts = first(summary.get("attempted_episodes"),
                     max(len(data["episodes"]), number(campaign.get("rollout_count")) or 0)
                     if data["episodes"] or (not active and number(campaign.get("rollout_count")) is not None) else None)
    accepted = first(summary.get("accepted_episodes"), campaign.get("accepted_count"))
    observed_accepted = sum(ep["accepted"] is True for ep in data["episodes"])
    if active and data["episodes"]:
        # Campaign scoreboards are updated at wave boundaries; per-episode
        # audits can already establish additional accepted outcomes mid-wave.
        accepted = max(number(accepted) or 0, observed_accepted)
    elif accepted is None and data["episodes"] and all(ep["accepted"] is not None for ep in data["episodes"]):
        accepted = observed_accepted
    updates = first(summary.get("committed_updates"), trainer.get("rft_updates_completed"))
    instance = data["instance"]
    observed_start = stamp(first(instance.get("original_launch_at"), instance.get("started")))
    observed_finish = stamp(instance.get("finished"))
    elapsed = first(summary.get("elapsed_seconds"),
                    observed_finish-observed_start if observed_start is not None and observed_finish is not None else None,
                    data["owner"].get("elapsed_seconds"))
    player = obj(obj(data["config"].get("config")).get("player"))
    model = first(summary.get("model"), obj(data["config"].get("model")).get("name"), data["config"].get("player_model"),
                  player.get("model"), player.get("base_model"), "Model identity not recorded")
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    attempt_label = "Episode artifacts collected" if active and summary.get("attempted_episodes") is None else "Episode attempts"
    if attempt_label == "Episode artifacts collected":
        attempts = len(data["episodes"])
    accepted_label = "Accepted so far" if active else "Accepted episodes"
    update_label = "Committed updates so far" if active else "Committed updates"
    preview_note = '<div class="callout"><strong>Historical preview.</strong> This page renders an earlier run’s actual evidence to preview the report format. It is not the current run and contains no invented current results.</div>' if preview else ''
    content = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="dark"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'"><title>{esc(title)}</title><style>{CSS}</style></head><body><a class="skip" href="#overview">Skip to report</a><header class="topbar"><div class="wrap"><a class="brand" href="#overview">Studio / Run evidence</a><nav aria-label="Report sections"><a href="#episodes">Episodes</a><a href="#execution">Execution</a><a href="#context">Context</a><a href="#memory">Memory</a><a href="#publication">Publication</a></nav></div></header><main class="wrap" id="overview"><div class="hero"><div class="hero-row"><p class="eyebrow">{esc(model)} · offline evidence report</p><span class="status warn">{esc(str(status).replace('_', ' '))}</span></div><h1>{esc(title)}</h1><p class="lead">A measured view of the rollout, training, and publication workflow. Expand an episode to inspect its actions, observations, and acceptance audit.</p><div class="metrics">{card(fmt(attempts), attempt_label)}{card(fmt(accepted), accepted_label)}{card(fmt(updates), update_label)}{card(fmt(elapsed, unit=' s'), 'Observed wall time')}</div>{preview_note}</div>'''
    if summary.get("failure"):
        content += f'<div class="callout"><strong>Recorded failure</strong><p>{esc(summary["failure"])}</p></div>'
    if conflicts:
        content += '<div class="callout"><strong>Conflicting success receipt.</strong><p>' + esc(" ".join(conflicts)) + '</p></div>'
    if data.get("monitor_health", {}).get("status") == "degraded":
        content += '<div class="callout"><strong>Telemetry degraded.</strong><p>Optional monitoring reported an error. Training lifecycle and its terminal outcome are recorded separately.</p></div>'
    content += '<div class="callout"><strong>Scope of this evidence.</strong> Episode acceptance and durable training receipts are reported separately from model capacity and recovery qualification. A micro run with one update does not establish learning quality. Missing measurements or skipped checks remain incomplete.</div>'
    if data["instance"] and not data["instance"].get("finished"):
        content += '<p class="muted">This is a snapshot of an active or unfinished run. The owner has not recorded a finish time. Refreshing this static page does not collect new evidence; regenerate the report to update it.</p>'
    sections = [("episodes", "01", "Episode outcomes", render_episodes(data)),
                ("execution", "02", "Where the time went", render_execution(data)),
                ("context", "03", "Context, tokens, and prefix reuse", render_throughput(data, throughput_window_seconds) + render_tokens(data)),
                ("memory", "04", "Memory and precision", render_memory(data)),
                ("publication", "05", "Updates, checkpoints, and policy publication", render_publication(data))]
    for identifier, index, label, body in sections:
        content += f'<section id="{identifier}"><div class="section-title"><span>{index}</span><h2>{label}</h2></div>{body}</section>'
    content += '<section id="sources"><div class="section-title"><span>06</span><h2>Evidence and limitations</h2></div><div class="panel">'
    content += table(["Qualification", "Recorded result"], [
        ["Agent behavior qualified", verdict(summary.get("agent_behavior_qualified"))],
        ["Learning quality qualified", verdict(summary.get("learning_quality_qualified"))],
        ["Held-out evaluations", fmt(summary.get("heldout_evaluations_run"))],
    ], "Explicit qualification statements")
    content += detail("Run summary", summary) + detail("Configuration snapshot", data["config"])
    content += detail("Run owner snapshot", data["instance"])
    if data.get("raw_instance") != data["instance"]:
        content += detail("Original stale owner snapshot (preserved)", data["raw_instance"])
    if data.get("supervisor"):
        content += detail("Terminal supervisor receipt", data["supervisor"])
    if data.get("monitor_health"):
        content += detail("Optional monitor: last observation", data["monitor_health"])
    content += detail("Evidence read errors / conflicts", data["errors"] + conflicts, open_=bool(data["errors"] or conflicts))
    content += '<details><summary>Source artifact inventory</summary><ul class="source-list">' + ''.join(f'<li><code>{esc(path)}</code></li>' for path in data["sources"]) + '</ul></details></div></section>'
    content += f'</main><footer><div class="wrap">Generated {esc(generated)} · evidence root <code>{esc(data["root"])}</code><br>Self-contained: no external assets, JavaScript, model imports, or network requests. Expandable evidence uses native browser controls.</div></footer></body></html>'
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--preview", action="store_true", help="Mark this as a historical format preview")
    parser.add_argument("--throughput-window-seconds", type=float, default=10,
                        help="Combine whole inference intervals into measured windows (default: 10 seconds)")
    args = parser.parse_args()
    if not args.run_root.is_dir():
        parser.error(f"Run evidence root is not a directory: {args.run_root}")
    if not math.isfinite(args.throughput_window_seconds) or args.throughput_window_seconds <= 0:
        parser.error("throughput-window-seconds must be finite and positive")
    data = load_evidence(args.run_root)
    document = render_report(data, title=args.title, preview=args.preview,
                             throughput_window_seconds=args.throughput_window_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(data["episodes"]),
                      "operations": len(data["operations"]), "artifact_errors": data["errors"],
                      "workflow_conflicts": workflow_conflicts(data)}))


if __name__ == "__main__":
    main()
