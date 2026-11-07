"""Optional Studio telemetry: read-only polling, isolated from training control.

The workflow owns the trainer and service. This module never signals either;
only the separate deadline supervisor has authority to stop owned computation.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time

from .storage import read, write


def spawn_logged(command: list[str], log_path: Path, *, cwd: Path, env: dict):
    """Own a separate process group while retaining the supervisor session.

    The stable session enclosure lets the supervisor discover children after
    their owner dies and the OS reparents them between polling intervals.
    Never retain a log descriptor when a child fails to start.
    """
    with log_path.open("xb") as log:
        return subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, process_group=0)


def read_operations(database: Path) -> list[dict]:
    """Close the connection deterministically, including on query failures.

    sqlite3.Connection.__exit__ only completes a transaction. Relying on GC to
    close it can exhaust descriptors during a long polling run.
    """
    if not database.exists():
        return []
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)) as connection:
        connection.row_factory = sqlite3.Row
        with closing(connection.execute("select rowid,* from operations order by rowid")) as cursor:
            return [dict(row) for row in cursor]


class OperationCollector:
    def __init__(self, root: Path):
        self.root = root
        self.seen: dict[str, dict] = {}
        self.observed_metrics: set[str] = set()

    def restore_observations(self) -> None:
        """Retain child-monitor timings when the owner takes the final snapshot.

        Only observational fields are reused; the journal still supplies every
        operation's current status and result. Telemetry remains optional.
        """
        try:
            snapshot = read(self.root / "compute/micro-operations.json")
        except (OSError, ValueError):
            return
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("operations"), list):
            return
        for row in snapshot["operations"]:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            observation = {"timing_source": "poll_observation"}
            for key in ("first_observed_at", "poll_interval_seconds", "started_at", "finished_at"):
                value = row.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    observation[key] = value
            if "first_observed_at" not in observation:
                continue
            self.seen.setdefault(row["id"], observation)
            receipt = row.get("result")
            if isinstance(receipt, dict) and (receipt.get("execution_metrics") or receipt.get("metrics")):
                self.observed_metrics.add(row["id"])

    def collect(self, *, persist: bool = True) -> list[dict]:
        if not self.seen:
            self.restore_observations()
        result, now = [], time.time()
        for row in read_operations(self.root / "state/journal.sqlite3"):
            value = {key: row[key] for key in ("id", "client", "kind", "status", "error")}
            value.update(result=json.loads(row["result"]) if row["result"] else None,
                         progress=json.loads(row["progress"]) if row["progress"] else None)
            observation = self.seen.setdefault(row["id"], {
                "first_observed_at": now, "timing_source": "poll_observation", "poll_interval_seconds": 1,
            })
            if row["status"] == "running":
                observation.setdefault("started_at", now)
            elif row["status"] != "queued":
                observation.setdefault("finished_at", now)
            body, receipt = json.loads(row["body"]), value["result"] or {}
            if row["kind"] in ("sample", "logprobs"):
                if body.get("prompt_token_ids"):
                    observation.update(submitted_prompt_tokens=len(body["prompt_token_ids"]),
                                       reserved_completion_tokens=body.get("max_tokens", len(body.get("target_token_ids", []))))
                else:
                    observation.setdefault("submitted_prompt_tokens", receipt.get("prompt_tokens"))
            elif row["kind"] == "forward_backward":
                if body.get("sequences"):
                    observation.update(submitted_prompt_tokens=sum(len(x["prompt_token_ids"]) for x in body["sequences"]),
                                       reserved_completion_tokens=sum(len(x["sampled_tokens"]) for x in body["sequences"]))
                else:
                    observation.setdefault("submitted_prompt_tokens", receipt.get("forward_metrics", {}).get("prompt_tokens"))
            value.update(observation)
            metrics = receipt.get("execution_metrics") or receipt.get("metrics")
            if persist and metrics and row["id"] not in self.observed_metrics:
                self.append_telemetry({"timestamp": now, "source": "completed_operation_metrics", "operation_id": row["id"], **metrics})
                self.observed_metrics.add(row["id"])
            result.append(value)
        if persist:
            (self.root / "compute").mkdir(exist_ok=True)
            write(self.root / "compute/micro-operations.json", {"operations": result})
        return result

    def append_telemetry(self, value: dict) -> None:
        with (self.root / "telemetry.jsonl").open("a") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")


def sample_host(pid: int, initial_swap: int | None) -> dict:
    raw = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True)
    match = re.search(r"used = ([0-9.]+)([MG])", raw)
    swap = int(float(match[1]) * 1024 ** (2 if match[2] == "M" else 3)) if match else None
    rss = int(subprocess.check_output(["/bin/ps", "-p", str(pid), "-o", "rss="], text=True).strip()) * 1024
    return {"timestamp": time.time(), "source": "host_observation", "pid": pid, "rss_bytes": rss,
            "swap_used_bytes": swap, "swap_growth_bytes": swap - initial_swap if swap is not None and initial_swap is not None else None}


def best_effort_write(path: Path, value: dict) -> bool:
    try:
        write(path, value)
        return True
    except Exception:
        return False


def monitor_loop(root: Path, *, service_pid: int | None = None, initial_swap: int | None = None,
                 interval: float = 1, iterations: int | None = None) -> None:
    collector, failures, last_host, count = OperationCollector(root), 0, 0.0, 0
    while iterations is None or count < iterations:
        count += 1
        health = {"pid": os.getpid(), "observed_at": time.time(), "failure_count": failures}
        try:
            operations = collector.collect()
            active = next((x["kind"] for x in operations if x["status"] == "running"), None)
            health.update(status="healthy", active_operation=active)
            if service_pid is not None and time.monotonic() - last_host >= 5:
                collector.append_telemetry(sample_host(service_pid, initial_swap))
                last_host = time.monotonic()
        except Exception as error:
            failures += 1
            health.update(status="degraded", failure_count=failures, error=f"{type(error).__name__}: {error}")
            # stderr was opened by the owner before startup; no new descriptor
            # is required even if the collector has reached EMFILE.
            try:
                print(json.dumps(health), file=sys.stderr, flush=True)
            except OSError:
                pass
        best_effort_write(root / "monitor-health.json", health)
        time.sleep(interval if health["status"] == "healthy" else max(interval, .05))


def wait_for_training(trainer, service, monitor, *, on_tick=None, on_monitor_failure=None,
                      interval: float = 1) -> int:
    """Watch computation; optional monitoring and its notification cannot abort it.

    on_tick is reserved for required compute-control checks, never telemetry.
    Deadline/explicit cancellation still propagate to the owning supervisor.
    """
    notified = False
    while trainer.poll() is None:
        if service.poll() is not None:
            raise RuntimeError("Owned compute exited during training")
        if not notified and (monitor is None or monitor.poll() is not None):
            notified = True
            if on_monitor_failure is not None:
                try:
                    on_monitor_failure(None if monitor is None else monitor.returncode)
                except Exception:
                    pass
        if on_tick is not None:
            on_tick()
        time.sleep(interval)
    return trainer.returncode


def effective_run_state(instance: dict, supervisor: dict) -> dict:
    """A terminal watchdog receipt wins over an owner's stale active snapshot.

    Owner success is preserved only when recorded and the supervisor also
    exited cleanly. Missing owner verification is never promoted to success.
    """
    if not supervisor.get("finished"):
        return dict(instance)
    clean = (supervisor.get("returncode") == 0 and not supervisor.get("stop_reason")
             and supervisor.get("owned_surviving_pids") == [] and supervisor.get("cleanup_verified", True))
    if clean and instance.get("finished") and instance.get("status") not in {"running", "starting", "active"}:
        return dict(instance)
    reason = supervisor.get("stop_reason")
    status = "interrupted" if reason else "failed"
    return {**instance, "raw_owner_status": instance.get("status"), "status": status,
            "finished": supervisor["finished"], "terminal_status_source": "owner-limit.json",
            "workflow_status": "incomplete_deadline_or_signal" if reason else "failed_infrastructure",
            "error": instance.get("error") or (f"Supervisor stopped owned work: {reason}" if reason else
                f"Workflow owner exited with return code {supervisor.get('returncode')}; required final verification was not completed.")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--service-pid", type=int)
    parser.add_argument("--initial-swap", type=int)
    args = parser.parse_args()
    monitor_loop(args.run_root, service_pid=args.service_pid, initial_swap=args.initial_swap)


if __name__ == "__main__":
    main()
