"""Counter, recovery, and dashboard contracts without paid model requests."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.episode.gm_process.usage import save_usage
from shellsmith.train.config import load_train_config
from shellsmith.train.observability import RunObservations, SdkHealth, reduce_usage
from shellsmith.train.observation_panels import (
    PRIMARY_PANELS, _phase_intervals, bucket_health, build_panels,
)
from studio_compute.inference_metrics import InferenceMetrics


@pytest.fixture
def observations(tmp_path):
    config = load_train_config(None, log_dir=tmp_path / "run")
    state = {"status": "running", "rft_updates_completed": 0, "checkpoints": []}
    return RunObservations(config, state, 0)


def _live(*, session="a", seconds=1, tokens=0, busy=0, active_seconds=0, queued_seconds=0):
    return {"active_requests": 0, "max_inference_batch": 8, "client_metrics": {
        "session_id": session, "uptime_seconds": seconds,
        "generated_tokens_total": tokens, "inference_busy_seconds_total": busy,
        "active_request_seconds_total": active_seconds,
        "queued_request_seconds_total": queued_seconds,
        "active_requests": 0, "queued_requests": 0,
    }}


def test_short_bursts_are_counted_over_poll_window_and_idle_is_measured(observations):
    observations.health(_live(), {})
    row = observations.health(_live(seconds=16, tokens=90, busy=3, active_seconds=12, queued_seconds=6), {})
    metrics = row["metrics"]
    assert metrics["inference/interval_output_tokens_per_second"] == 6
    assert metrics["inference/average_active_requests"] == .8
    assert metrics["inference/average_queued_requests"] == .4
    assert metrics["inference/active_output_tokens_per_second"] == 30
    assert metrics["inference/active_requests"] == 0
    idle = observations.health(_live(seconds=31, tokens=90, busy=3, active_seconds=12, queued_seconds=6), {})
    assert idle["metrics"]["inference/interval_output_tokens_per_second"] == 0
    assert idle["metrics"]["inference/active_output_tokens_per_second"] is None


def test_counter_reset_missing_sample_and_resume_do_not_invent_rates(observations):
    observations.health(_live(seconds=1, tokens=10), {})
    observations.health(None, {})
    gap = observations.health(_live(seconds=31, tokens=30), {})
    assert gap["metrics"]["inference/interval_output_tokens_per_second"] is None
    reset = observations.health(_live(session="b", seconds=2, tokens=5), {})
    assert reset["metrics"]["inference/interval_output_tokens_per_second"] is None
    assert reset["metrics"]["inference/observed_generated_tokens_total"] == 35
    resumed = RunObservations(observations.config, observations.state, 1)
    row = resumed.health(_live(session="b", seconds=3, tokens=7), {})
    assert row["metrics"]["inference/observed_generated_tokens_total"] == 37
    assert row["metrics"]["inference/interval_output_tokens_per_second"] is None
    assert any(record["kind"] == "gap" for record in resumed.records)
    assert resumed.metadata["started_at"] == observations.metadata["started_at"]


def _receipt(thread, tokens, *, outcome="succeeded", finished=2, started=1):
    return {"thread_id": thread, "cumulative_usage": tokens, "usage_known": bool(tokens),
            "stage": "bootstrap", "outcome": outcome, "started_at": started, "finished_at": finished}


def test_gm_cumulative_deltas_regression_and_failed_calls_are_honest():
    receipts = {
        "one": _receipt("thread", {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 10}),
        "two": _receipt("thread", {"input_tokens": 300, "cached_input_tokens": 250, "output_tokens": 30}, started=2),
        "failure": _receipt("thread", {}, outcome="failed", started=3),
        "regression": _receipt("thread", {"input_tokens": 20}, started=4),
        "pending": _receipt(None, {}, outcome="running", started=5, finished=None),
        "other": _receipt("another", {"input_tokens": 50, "output_tokens": 5}, started=6),
    }
    totals, rows = reduce_usage(receipts)
    assert totals["input_tokens"] == 350
    assert totals["cached_input_tokens"] == 250
    assert totals["output_tokens"] == 35
    assert totals["invocations_with_usage"] == 3
    assert totals["invocations_pending"] == 1
    assert totals["invocations_unknown"] == 2
    assert totals["accounting_errors"] == 1
    assert rows[1][5:7] == [200, 170]
    assert reduce_usage(dict(receipts))[0] == totals


def test_usage_receipt_survives_partial_jsonl_and_preserves_failure(tmp_path):
    path = tmp_path / "000001_final.jsonl"
    path.write_text(json.dumps({"type": "thread.started", "thread_id": "thread"}) + "\n"
                    + json.dumps({"type": "turn.completed", "usage": {"input_tokens": 42, "secret": "omit"}})
                    + "\n{torn")
    save_usage(path, started_at=1, finished_at=2, stage="final", thread_id=None, outcome="failed")
    receipt = json.loads(path.with_suffix(".usage.json").read_text())
    assert receipt["outcome"] == "failed" and receipt["malformed_log"]
    assert receipt["cumulative_usage"] == {"input_tokens": 42}
    assert receipt["thread_id"] == "thread" and receipt["usage_known"]


def test_per_client_busy_wall_time_is_not_sum_of_concurrent_request_time():
    clock = SimpleNamespace(now=0.)
    metrics = InferenceMetrics(clock=lambda: clock.now)
    for identity, client in (("a", "one"), ("b", "one"), ("c", "two")):
        metrics.submit(identity, client=client)
        metrics.start(identity)
    clock.now = 3
    metrics.progress("a", 3)
    metrics.finish("a", failed=True)
    metrics.finish("b", tokens=9)
    metrics.finish("c", tokens=100)
    one, two = metrics.client_readout("one"), metrics.client_readout("two")
    assert one["generated_tokens_total"] == 12 and two["generated_tokens_total"] == 100
    assert one["inference_busy_seconds_total"] == 3
    assert one["active_request_seconds_total"] == 6
    assert one["failed_requests_total"] == 1
    assert metrics.readout()["generated_tokens_total"] == 112


def test_sdk_backlog_reader_keeps_partial_lines_and_reports_unknown_format(tmp_path):
    journal = tmp_path / "events.jsonl"
    journal.write_bytes(b'{}\n{')
    health = SdkHealth(tmp_path)
    assert health.read(SimpleNamespace())["telemetry/unacknowledged_events"] == 1
    with journal.open("ab") as stream:
        stream.write(b'}\n')
    assert health.read(SimpleNamespace())["telemetry/unacknowledged_events"] == 2
    (tmp_path / "ack.json").write_text('{"count": 1}')
    assert health.read(SimpleNamespace())["telemetry/unacknowledged_events"] == 1
    (tmp_path / "ack.json").write_text('{"unknown": true}')
    assert health.read(SimpleNamespace())["telemetry/sdk_health_available"] == 0


def test_six_panels_exist_before_first_update_and_figure_bucket_keeps_gaps(observations):
    import backprop

    observations.health(None, {})
    panels = build_panels(observations, backprop)
    assert all(key in panels for key in PRIMARY_PANELS)
    learning = panels["train/03_learning_metrics"].spec
    assert "No optimizer update committed yet" in json.dumps(learning)
    assert learning["data"] == []
    assert learning["layout"]["xaxis"]["visible"] is False
    assert learning["layout"]["yaxis"]["visible"] is False
    records = []
    for index in range(2000):
        records.append({"kind": "health", "run/elapsed_seconds": index + 1,
                        "metrics": {"inference/active_requests": 8 if index % 2 else 1,
                                    "inference/queued_requests": 0,
                                    "inference/interval_seconds": 1 if index != 100 else None,
                                    "inference/interval_generated_tokens": 10}})
    buckets = bucket_health(records)
    assert len(buckets) <= 1000
    assert max(row["metrics"]["inference/active_requests"] for row in buckets) == 8
    assert any(row["metrics"]["inference/interval_output_tokens_per_second"] is None for row in buckets)


def test_phase_strip_preserves_short_phases_and_resume_gaps_without_label_collisions(observations):
    import backprop

    rows = [
        {"kind": "phase", "phase": "collection", "run/elapsed_seconds": 0},
        {"kind": "phase", "phase": "collection", "run/elapsed_seconds": 10},
        {"kind": "phase", "phase": "forward_backward", "run/elapsed_seconds": 20},
        {"kind": "gap", "phase": "not_observed", "gap_started": 80, "run/elapsed_seconds": 90},
        {"kind": "phase", "phase": "checkpoint", "run/elapsed_seconds": 90},
        {"kind": "phase", "phase": "publication", "run/elapsed_seconds": 91},
        {"kind": "health", "phase": "publication", "run/elapsed_seconds": 92, "metrics": {}},
    ]
    assert _phase_intervals(rows) == [
        ("collection", 0, 20), ("forward_backward", 20, 80),
        ("not_observed", 80, 90), ("checkpoint", 90, 91), ("publication", 91, 92),
    ]
    observations.records = [{**observations.context(), **row} for row in rows]
    figure = build_panels(observations, backprop)["train/01_operational_timeline"].spec
    bars = [trace for trace in figure["data"] if trace["type"] == "bar"]
    assert [trace["text"][0] for trace in bars] == ["collection", "forward backward", "", "", ""]
    assert bars[-2]["customdata"] == [["checkpoint", 90, 91]]
    assert bars[-1]["customdata"] == [["publication", 91, 92]]


def test_learning_panel_uses_the_reported_optimizer_learning_rate(observations):
    import backprop

    observations.state["rft_updates_completed"] = 1
    observations.updates["update"] = {"datums": 1, "optimizer_updates_committed": 1,
                                      "optim_metrics": {"learning_rate": 0.000012}}
    figure = build_panels(observations, backprop)["train/03_learning_metrics"].spec
    assert figure["data"][-1]["y"] == [0.000012]
    assert figure["layout"]["xaxis"]["anchor"] == "y3"
    lower, upper = figure["layout"]["yaxis3"]["range"]
    assert 0 < lower < 0.000012 < upper < 0.0001


def test_torn_observation_tail_is_archived_without_losing_complete_records(observations):
    before = len(observations.records)
    with observations.path.open("ab") as stream:
        stream.write(b'{"torn":')
    resumed = RunObservations(observations.config, observations.state, 1)
    assert len(resumed.records) == before + 2
    assert list(observations.directory.glob("torn-*.json"))


def test_counter_regression_does_not_publish_negative_occupancy(observations):
    observations.health(_live(seconds=1, tokens=10, busy=1, active_seconds=2), {})
    row = observations.health(_live(seconds=2, tokens=11, busy=1, active_seconds=0), {})
    assert row["metrics"]["inference/average_active_requests"] is None
    assert row["metrics"]["inference/counter_error"] == 1


def test_metrics_client_selects_scope_and_only_changes_telemetry_timeout(monkeypatch):
    from studio_compute.client import Client

    calls = []
    client = Client("http://127.0.0.1:8766", timeout=7200)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: calls.append((args, kwargs)))
    client.metrics(client_id="pilot")
    assert calls == [(("/v1/metrics?client=pilot",), {"timeout_seconds": 2})]
    assert client.timeout == 7200


def test_local_collector_survives_a_blocked_sdk_publisher(tmp_path, monkeypatch):
    import threading
    from shellsmith.train.telemetry import TrainingTelemetry

    config = load_train_config(None, log_dir=tmp_path / "run")
    config = replace(config, telemetry=replace(config.telemetry, mode="offline",
                     capture_console=False, system_metrics=False, system_interval_seconds=1))
    state = {"step": 0, "rft_updates_completed": 0, "next_world_offset": 0,
             "status": "running", "segments": [{"index": 0}]}
    entered, release = threading.Event(), threading.Event()
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        original = sink.run.log

        def blocked(values):
            entered.set()
            assert release.wait(5)
            return original(values)

        monkeypatch.setattr(sink.run, "log", blocked)
        try:
            sink.phase("collection")
            assert entered.wait(2)
            sink._health()
            sink._health()
            health = [row for row in sink.observations.records if row["kind"] == "health"]
            assert len(health) >= 2
            assert all(row["phase"] == "collection" for row in health)
        finally:
            release.set()
            sink.queue.join()
