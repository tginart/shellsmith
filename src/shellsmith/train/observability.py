"""Purpose: Preserve run observations independently of optimizer transactions.
Input: Trainer phases, CPU service counters, and durable GM usage receipts.
Output: Replayable operational records with explicit time and coverage.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

from shellsmith.episode.gm_process.usage import TOKEN_FIELDS
from shellsmith.process_io import write_json

RATE_FIELDS = (
    "interval_output_tokens_per_second", "average_active_requests",
    "average_queued_requests", "active_output_tokens_per_second",
)


def reduce_usage(receipts: dict[str, dict]) -> tuple[dict, list[list]]:
    """Purpose: Difference cumulative thread snapshots without double counts.
    Input: The latest receipt for each invocation.
    Output: Totals and invocation rows. Missing or regressed usage stays unknown.
    """
    baselines, totals, rows = {}, {}, []
    stages = {}
    known = errors = pending = 0
    for identity, receipt in sorted(receipts.items(), key=lambda item: (
        item[1]["started_at"], item[0]
    )):
        stage = receipt["stage"]
        stage_row = stages.setdefault(stage, {"invocations": 0, "reported": 0})
        stage_row["invocations"] += 1
        thread = receipt.get("thread_id")
        snapshot = receipt.get("cumulative_usage", {})
        previous = baselines.get(thread, {})
        valid = bool(receipt.get("usage_known") and thread)
        regression = valid and any(value < previous.get(key, 0)
                                   for key, value in snapshot.items())
        delta = {}
        if valid and not regression:
            for key, value in snapshot.items():
                delta[key] = value - previous.get(key, 0)
                totals[key] = totals.get(key, 0) + delta[key]
            baselines[thread] = {**previous, **snapshot}
            known += 1
            stage_row["reported"] += 1
        if regression:
            errors += 1
        running = receipt.get("finished_at") is None
        pending += int(running and not valid)
        rows.append([identity, thread, stage, receipt["outcome"],
                     "counter regression" if regression else "reported" if valid else "pending" if running else "unknown",
                     *[delta.get(key) for key in TOKEN_FIELDS]])
    return {**totals, "invocations_started": len(receipts),
            "invocations_with_usage": known, "invocations_pending": pending,
            "invocations_unknown": len(receipts) - known - pending,
            "accounting_errors": errors, "stages": stages}, rows


class SdkHealth:
    """Purpose: Isolate the pinned SDK journal and acknowledgement formats.
    Input: A local SDK sync directory and the selected run.
    Output: Incremental backlog counters, or explicit unavailable health.
    """

    def __init__(self, directory: Path):
        self.directory = directory
        self.offset = self.count = 0

    def read(self, run: Any) -> dict:
        try:
            journal = self.directory / "events.jsonl"
            size = journal.stat().st_size
            if size < self.offset:
                self.offset = self.count = 0
            with journal.open("rb") as stream:
                stream.seek(self.offset)
                for line in stream:
                    if not line.endswith(b"\n"):
                        break
                    self.count += 1
                    self.offset += len(line)
            path = self.directory / "ack.json"
            ack = json.loads(path.read_text()) if path.exists() else {"count": 0}
            if type(ack.get("count")) is not int or not 0 <= ack["count"] <= self.count:
                raise ValueError("Unsupported SDK acknowledgement")
            return {"telemetry/sdk_health_available": 1,
                    "telemetry/unacknowledged_events": self.count - ack["count"],
                    "telemetry/sdk_upload_error": int(getattr(run, "_error", None) is not None)}
        except (OSError, ValueError, TypeError):
            return {"telemetry/sdk_health_available": 0,
                    "telemetry/unacknowledged_events": None,
                    "telemetry/sdk_upload_error": None}


class RunObservations:
    """Purpose: Own a separate, replayable observation journal and CPU reducers.
    Input: The training config, shared trainer state, and resume segment.
    Output: Time-aligned observations without a change to optimizer authority.
    """

    def __init__(self, config: Any, state: dict, segment: int):
        self.config, self.state, self.segment = config, state, segment
        self.root = config.resolved_log_dir()
        self.directory = self.root / "observability"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.path = self.directory / "events.jsonl"
        metadata = self.directory / "metadata.json"
        if metadata.exists():
            self.metadata = json.loads(metadata.read_text())
        else:
            self.metadata = {"version": 1, "run_id": config.run.run_id,
                             "started_at": time.time(), "interval_seconds": config.telemetry.system_interval_seconds}
            write_json(metadata, self.metadata)
        self.records = []
        if self.path.exists():
            with self.path.open("rb+") as stream:
                boundary = 0
                for line in stream:
                    if not line.endswith(b"\n"):
                        stream.seek(boundary)
                        tail = stream.read()
                        write_json(self.directory / f"torn-{time.time_ns()}.json",
                                   {"tail_hex": tail.hex()})
                        stream.truncate(boundary)
                        break
                    self.records.append(json.loads(line))
                    boundary += len(line)
        self.anchor = time.monotonic()
        self.elapsed_base = max(time.time() - self.metadata["started_at"],
                                self.records[-1]["run/elapsed_seconds"] if self.records else 0)
        self.phase = "initialization"
        self.phase_started = self.elapsed_base
        self.previous = None
        self.session_totals = {}
        self.episodes, self.updates, self.publications = {}, {}, {}
        self.gm_receipts, self.gm_stamps = {}, {}
        self.gm, self.gm_rows = {}, []
        self.last_live = {}
        for record in self.records:
            self._restore(record)
        if self.records:
            self.append("gap", phase="not_observed", gap_started=self.records[-1]["run/elapsed_seconds"])
        self.set_phase("initialization")

    def context(self) -> dict:
        now = time.time()
        return {"observability/version": 1, "run/id": self.config.run.run_id,
                "telemetry/segment": self.segment, "run/observed_at": datetime.fromtimestamp(now, timezone.utc).isoformat(),
                "telemetry/alive_timestamp": now,
                "run/elapsed_seconds": self.elapsed_base + time.monotonic() - self.anchor,
                "train/optimizer_updates_committed": self.state.get("rft_updates_completed", 0)}

    def append(self, kind: str, **fields: Any) -> dict:
        """Purpose: Append one observation before it enters the publication queue.
        Input: The event kind and sanitized observation fields.
        Output: The durable event with a stable sequence identity.
        """
        with self.lock:
            record = {**self.context(), "id": str(len(self.records)), "kind": kind,
                      "phase": self.phase, **fields}
            with self.path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self.records.append(record)
            self._restore(record)
            return record

    def set_phase(self, phase: str, *, outcome: str = "running") -> dict:
        with self.lock:
            previous, started = self.phase, self.phase_started
            self.phase = phase
            self.phase_started = self.context()["run/elapsed_seconds"]
            return self.append("phase", previous_phase=previous,
                               previous_started=started,
                               previous_seconds=max(0, self.phase_started - started),
                               update_target=self.state.get("rft_updates_completed", 0) + 1,
                               outcome=outcome)

    def progress(self, event: Any) -> None:
        """Update episode occupancy before optional progress coalescence."""
        phase = event.phase
        states = {"episode_start": "gm", "boot_start": "gm", "turn_start": "gm",
                  "boot_done": "player", "agent_start": "player", "turn_done": "player",
                  "judge_start": "judging", "judge_done": "settling",
                  "episode_done": "settling", "episode_error": "failed"}
        if phase in states:
            with self.lock:
                previous = self.episodes.get(event.episode_id, {})
                if not previous.get("terminal"):
                    self.episodes[event.episode_id] = {**previous, "stage": states[phase]}

    def scan_usage(self) -> None:
        """Refresh atomic GM receipts, including failures without episode outcomes."""
        for path in (self.root / "segments").rglob("*.usage.json"):
            stamp = path.stat().st_mtime_ns
            identity = str(path.relative_to(self.root))
            if self.gm_stamps.get(identity) != stamp:
                self.gm_receipts[identity] = json.loads(path.read_text())
                self.gm_stamps[identity] = stamp
        self.gm, self.gm_rows = reduce_usage(self.gm_receipts)

    def health(self, live: dict | None, metrics: dict, *, compute_records: list | None = None) -> dict:
        """Purpose: Difference cumulative CPU counters over the actual sample span.
        Input: A service snapshot or an explicit missing sample, and health fields.
        Output: A durable observation. Unknown intervals remain null.
        """
        with self.lock:
            self.scan_usage()
            metrics.update({"inference/" + key: None for key in RATE_FIELDS})
            metrics["inference/interval_seconds"] = None
            counters = live.get("client_metrics") if live else None
            if counters is not None:
                self.last_live = live
                session = counters["session_id"]
                metrics.update({"inference/active_requests": counters["active_requests"],
                                "inference/queued_requests": counters["queued_requests"],
                                "inference/service_active_requests": live["active_requests"],
                                "inference/service_capacity": live.get("max_inference_batch"),
                                "inference/service_session_id": session})
                old = self.previous
                if old and old["session_id"] == session:
                    seconds = counters["uptime_seconds"] - old["uptime_seconds"]
                    tokens = counters["generated_tokens_total"] - old["generated_tokens_total"]
                    keys = ("generated_tokens_total", "active_request_seconds_total",
                            "queued_request_seconds_total", "inference_busy_seconds_total")
                    compatible = all(counters[key] >= old[key] for key in keys)
                    metrics["inference/counter_error"] = int(not compatible)
                    if seconds > 0 and compatible:
                        metrics["inference/interval_seconds"] = seconds
                        metrics["inference/interval_generated_tokens"] = tokens
                        metrics["inference/interval_output_tokens_per_second"] = tokens / seconds
                        for name, key in (("average_active_requests", "active_request_seconds_total"),
                                          ("average_queued_requests", "queued_request_seconds_total")):
                            metrics["inference/" + name] = (counters[key] - old[key]) / seconds
                        busy = counters["inference_busy_seconds_total"] - old["inference_busy_seconds_total"]
                        metrics["inference/active_output_tokens_per_second"] = tokens / busy if busy > 0 else None
                self.previous = dict(counters)
                self.session_totals[session] = max(self.session_totals.get(session, 0), counters["generated_tokens_total"])
            else:
                self.previous = None
                metrics.update({"inference/active_requests": None, "inference/queued_requests": None,
                                "inference/service_active_requests": None,
                                "inference/service_capacity": None})
            metrics["inference/sample_available"] = int(counters is not None)
            total = sum(self.session_totals.values())
            metrics["inference/observed_generated_tokens_total"] = total
            elapsed = self.context()["run/elapsed_seconds"]
            metrics["inference/overall_observed_tokens_per_second"] = total / elapsed if elapsed > 0 else None
            metrics.update({"gm/" + key: value for key, value in self.gm.items() if key != "stages"})
            return self.append("health", metrics=metrics, counters=counters,
                               compute_records=compute_records or [])

    def _restore(self, record: dict) -> None:
        if record["kind"] == "health" and record.get("counters"):
            counters = record["counters"]
            session = counters["session_id"]
            self.session_totals[session] = max(self.session_totals.get(session, 0), counters["generated_tokens_total"])
        elif record["kind"] == "episode":
            self.episodes[record["episode_id"]] = record["episode"]
        elif record["kind"] == "update":
            self.updates[record["update_id"]] = record["update"]
        elif record["kind"] == "publication":
            self.publications[record["publication_id"]] = record["publication"]

    def episode_counts(self) -> dict:
        return dict(Counter(row.get("stage", "unknown") for row in self.episodes.values()))
