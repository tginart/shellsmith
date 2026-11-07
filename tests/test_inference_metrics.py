"""Deterministic timing checks for measured throughput and request occupancy."""

import json
import threading

import pytest

from studio_compute.inference_metrics import InferenceMetrics


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_matching_time_windows_count_short_requests_early_eos_and_idle_time():
    clock = Clock()
    metrics = InferenceMetrics(clock=clock, wall_clock=lambda: 100 + clock.now)
    for request in ("a", "b"):
        metrics.submit(request)
    clock.now = 2
    for request in ("a", "b"):
        metrics.start(request)
    clock.now = 3
    metrics.progress("a", 1, finished=True)
    metrics.progress("b", 1)
    assert metrics.readout()["active_requests"] == 1
    clock.now = 5
    metrics.progress("b", 3, finished=True)
    metrics.finish("a", tokens=1)
    metrics.finish("b", tokens=3)
    clock.now = 6
    row = metrics.interval()
    assert row["generated_tokens"] == 4  # Not batch_size * longest_completion.
    assert row["tokens_per_second"] == pytest.approx(4 / 6)
    assert row["average_active_requests"] == pytest.approx(4 / 6)
    assert row["average_queued_requests"] == pytest.approx(4 / 6)
    assert row["peak_active_requests"] == row["peak_queued_requests"] == 2
    assert row["active_requests"] == row["queued_requests"] == 0
    assert row["completed_requests"] == 2
    assert (row["started_at"], row["finished_at"]) == (100, 106)
    clock.now = 10
    idle = metrics.interval()
    assert idle["tokens_per_second"] == idle["average_active_requests"] == 0
    assert idle["peak_active_requests"] == 0
    assert idle["generated_tokens_total"] == 4


def test_failure_and_repeated_progress_retain_generated_work_once():
    clock = Clock()
    metrics = InferenceMetrics(clock=clock)
    metrics.submit("a")
    metrics.start("a")
    clock.now = 1
    metrics.progress("a", 3)
    metrics.progress("a", 3)
    metrics.progress("a", 2)  # A stale observation cannot subtract work.
    metrics.finish("a", failed=True)
    metrics.finish("a", failed=True)
    row = metrics.interval()
    assert row["generated_tokens"] == 3 and row["failed_requests"] == 1
    assert row["completed_requests"] == 0 and row["average_active_requests"] == 1
    assert metrics.readout()["generated_tokens_total"] == 3


def test_slow_interval_uses_actual_monotonic_duration():
    clock = Clock()
    metrics = InferenceMetrics(clock=clock, wall_clock=lambda: -500)
    metrics.submit("a")
    metrics.start("a")
    clock.now = 2.5
    metrics.finish("a", tokens=10)
    row = metrics.interval()
    assert row["tokens_per_second"] == 4 and row["seconds"] == 2.5
    assert row["finished_at"] - row["started_at"] == 2.5
    assert metrics.interval() is None


def test_logging_failure_is_isolated_and_final_partial_interval_is_written(tmp_path):
    metrics = InferenceMetrics()
    path = tmp_path / "missing" / "intervals.jsonl"
    metrics.start_logging(path, interval=100)
    metrics.close()
    assert metrics.readout()["logging_errors"] == 1
    working = InferenceMetrics()
    working.start_logging(tmp_path / "intervals.jsonl", model="fixture", interval=100)
    working.close()
    row = json.loads((tmp_path / "intervals.jsonl").read_text())
    assert row["model"] == "fixture" and row["seconds"] > 0
    assert row["generated_tokens"] == 0


def test_failed_logger_thread_start_does_not_stop_computation(tmp_path, monkeypatch):
    def fail_start(thread):
        raise RuntimeError("Unable to start telemetry thread")
    monkeypatch.setattr(threading.Thread, "start", fail_start)
    metrics = InferenceMetrics()
    metrics.start_logging(tmp_path / "intervals.jsonl")
    metrics.start("a")
    metrics.finish("a", tokens=2)
    metrics.close()
    assert metrics.readout()["generated_tokens_total"] == 2
    assert metrics.readout()["logging_errors"] == 1


def test_metrics_cli_reads_cpu_only_endpoint(monkeypatch, capsys):
    from studio_compute.cli import main, parse_args
    from studio_compute.client import Client

    assert parse_args(["metrics"]).endpoint == "http://127.0.0.1:8766"
    def read(client):
        assert client.endpoint == "http://127.0.0.1:1234"
        return {"active_requests": 2, "latest_interval": {"tokens_per_second": 15}}
    monkeypatch.setattr(Client, "metrics", read)
    main(["metrics", "--endpoint", "http://127.0.0.1:1234"])
    value = json.loads(capsys.readouterr().out)
    assert value["active_requests"] == 2 and value["latest_interval"]["tokens_per_second"] == 15


def test_admitted_batch_occupancy_is_distinct_from_request_concurrency():
    metrics = InferenceMetrics()
    metrics.batch(4)
    metrics.batch(1)
    row = metrics.readout()
    assert row["admitted_batches_total"] == 2
    assert row["last_admitted_batch_size"] == 1
    assert row["mean_admitted_batch_size"] == 2.5
    assert row["active_requests"] == 0
