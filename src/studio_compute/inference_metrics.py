"""CPU-only inference counters and optional, failure-isolated interval logging.

Active means admitted to sampling, including prefill and time between cooperative
GPU turns. Queued work, logprob scoring, and training are separate populations.
Token counts advance at materialized decode boundaries, including failed work.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import uuid


class InferenceMetrics:
    """Purpose: Count materialized sampling work without an operation on the GPU.
    Input: Request lifecycle events and monotonic clocks.
    Output: Service counters and optional client counters.
    """
    def __init__(self, *, clock=time.monotonic, wall_clock=time.time):
        self.clock, self.wall_clock = clock, wall_clock
        self.lock = threading.RLock()
        self.session_id = uuid.uuid4().hex
        self.started = self.previous = self.last = clock()
        self.started_at = wall_clock()
        self.queued, self.active = set(), {}
        self.generated = self.completed = self.failed = 0
        self.batches = self.batch_slots = self.last_batch_size = 0
        self.active_seconds = self.queued_seconds = 0.0
        self.busy_seconds = 0.0
        self.clients, self.request_clients = {}, {}
        self.peak_active = self.peak_queued = 0
        self.baseline = (0, 0, 0, 0.0, 0.0)
        self.latest = None
        self.logging_errors = 0
        self.stop = threading.Event()
        self.thread = None

    def _accrue(self, now):
        elapsed = now - self.last
        self.active_seconds += elapsed * sum(not row["finished"] for row in self.active.values())
        self.busy_seconds += elapsed * bool(
            any(not row["finished"] for row in self.active.values())
        )
        self.queued_seconds += elapsed * len(self.queued)
        self.last = now

    def _peaks(self):
        self.peak_active = max(self.peak_active, sum(not row["finished"] for row in self.active.values()))
        self.peak_queued = max(self.peak_queued, len(self.queued))

    def submit(self, request_id, *, client=None):
        with self.lock:
            self._accrue(self.clock())
            self.queued.add(request_id)
            if client is not None:
                self.request_clients[request_id] = client
                self.client_readout(client)
                self.clients[client].submit(request_id)
            self._peaks()

    def start(self, request_id):
        with self.lock:
            self._accrue(self.clock())
            self.queued.discard(request_id)
            self.active[request_id] = {"tokens": 0, "finished": False}
            if request_id in self.request_clients:
                self.clients[self.request_clients[request_id]].start(request_id)
            self._peaks()

    def progress(self, request_id, tokens, *, finished=False):
        with self.lock:
            self._accrue(self.clock())
            row = self.active.get(request_id)
            if row is None:
                return
            # Progress and final receipts report cumulative counts. A repeated
            # observation or the terminal receipt must not count tokens twice.
            self.generated += max(0, tokens - row["tokens"])
            row["tokens"] = max(tokens, row["tokens"])
            row["finished"] = row["finished"] or finished
            if request_id in self.request_clients:
                self.clients[self.request_clients[request_id]].progress(
                    request_id, tokens, finished=finished
                )

    def finish(self, request_id, *, tokens=None, failed=False):
        with self.lock:
            self._accrue(self.clock())
            if tokens is not None:
                self.progress(request_id, tokens, finished=True)
            was_active = self.active.pop(request_id, None) is not None
            if was_active or (failed and request_id in self.queued):
                if failed:
                    self.failed += 1
                else:
                    self.completed += 1
            self.queued.discard(request_id)
            client = self.request_clients.pop(request_id, None)
            if client is not None:
                self.clients[client].finish(request_id, tokens=tokens, failed=failed)

    def client_readout(self, client: str) -> dict:
        """Purpose: Read one client's CPU counters without another client's work.
        Input: The client identity. Output: Counters in the service session.
        """
        with self.lock:
            if client not in self.clients:
                child = InferenceMetrics(clock=self.clock, wall_clock=self.wall_clock)
                child.session_id = self.session_id
                self.clients[client] = child
            return {**self.clients[client].readout(), "client_id": client,
                    "scope": "client"}

    def batch(self, size: int) -> None:
        """Record actual admitted batch size, separate from active request count."""
        with self.lock:
            self.batches += 1
            self.batch_slots += size
            self.last_batch_size = size

    def readout(self):
        with self.lock:
            now = self.clock()
            self._accrue(now)
            seconds = now - self.started
            return {
                "session_id": self.session_id, "timestamp": self.wall_clock(),
                "uptime_seconds": seconds,
                "active_requests": sum(not row["finished"] for row in self.active.values()),
                "queued_requests": len(self.queued), "generated_tokens_total": self.generated,
                "completed_requests_total": self.completed, "failed_requests_total": self.failed,
                "active_request_seconds_total": self.active_seconds,
                "queued_request_seconds_total": self.queued_seconds,
                "inference_busy_seconds_total": self.busy_seconds,
                "lifetime_tokens_per_second": self.generated / seconds if seconds > 0 else 0.0,
                "logging_errors": self.logging_errors, "latest_interval": self.latest,
                "last_admitted_batch_size": self.last_batch_size,
                "admitted_batches_total": self.batches,
                "mean_admitted_batch_size": self.batch_slots / self.batches if self.batches else 0.0,
            }

    def interval(self):
        with self.lock:
            now = self.clock()
            self._accrue(now)
            seconds = now - self.previous
            if seconds <= 0:
                return None
            tokens, completed, failed, active_seconds, queued_seconds = self.baseline
            active = sum(not row["finished"] for row in self.active.values())
            row = {
                "source": "local_inference_interval", "session_id": self.session_id,
                "started_at": self.started_at + self.previous - self.started,
                "finished_at": self.started_at + now - self.started,
                "seconds": seconds, "generated_tokens": self.generated - tokens,
                "tokens_per_second": (self.generated - tokens) / seconds,
                "active_requests": active,
                "average_active_requests": (self.active_seconds - active_seconds) / seconds,
                "peak_active_requests": self.peak_active,
                "queued_requests": len(self.queued),
                "average_queued_requests": (self.queued_seconds - queued_seconds) / seconds,
                "peak_queued_requests": self.peak_queued,
                "completed_requests": self.completed - completed, "failed_requests": self.failed - failed,
                "generated_tokens_total": self.generated, "logging_errors": self.logging_errors,
            }
            self.baseline = (self.generated, self.completed, self.failed, self.active_seconds, self.queued_seconds)
            self.previous = now
            self.peak_active, self.peak_queued = active, len(self.queued)
            self.latest = row
            return row

    def start_logging(self, path: Path, *, model=None, max_inference_batch=None, interval=1.0):
        """Disk I/O runs off the GPU owner; failures never abort computation."""
        def append():
            try:
                row = self.interval()
                if row is None:
                    return
                with path.open("a") as stream:
                    stream.write(json.dumps({**row, "model": model,
                                             "max_inference_batch": max_inference_batch,
                                             "clients": {client: self.client_readout(client)
                                                         for client in tuple(self.clients)}},
                                            allow_nan=False) + "\n")
            except Exception:
                with self.lock:
                    self.logging_errors += 1

        def run():
            while not self.stop.wait(interval):
                append()
            append()  # Include the final, possibly partial interval.

        self.thread = threading.Thread(target=run, name="studio-inference-metrics", daemon=True)
        try:
            self.thread.start()
        except RuntimeError:
            self.thread = None
            self.logging_errors += 1

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
