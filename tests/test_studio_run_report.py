"""Offline report evidence must remain honest, safe, and inspectable."""
from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/render_studio_run_report.py"
spec = importlib.util.spec_from_file_location("studio_run_report", SCRIPT)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Tags(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = []

    def handle_starttag(self, tag, attributes):
        self.tags.append(tag)
        self.attributes.extend(attributes)


class StudioRunReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, path, value):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return target

    def document(self, **kwargs):
        return report.render_report(report.load_evidence(self.root), **kwargs)

    def episode(self, status="complete", accepted=True, reward=.75):
        base = "segments/000/world/rollout_01"
        self.write(base + "/episode.json", {"episode_id": "run-episode-01", "world_id": "world", "status": status})
        self.write(base + "/acceptance_audit.json", {"accepted_for_training": accepted,
                                                      "final_task": {"reward": reward}})
        return base

    def test_empty_report_does_not_fabricate_success_or_zero_measurements(self):
        text = self.document()
        self.assertIn("workflow outcome not yet recorded", text)
        self.assertIn("Not recorded", text)
        self.assertIn("Chart unavailable: no measured observations", text)
        self.assertIn("start/end timestamps were not recorded", text)
        self.assertNotIn("0.00 GiB", text)
        self.assertNotIn("workflow succeeded", text)
        self.assertIn("Local inference intervals: not recorded", text)

    def test_throughput_windows_weight_concurrency_and_do_not_bridge_missing_time(self):
        def interval(start, seconds, tokens, active, session="one"):
            return {"session_id": session, "started_at": start, "finished_at": start + seconds,
                    "seconds": seconds, "generated_tokens": tokens, "average_active_requests": active,
                    "peak_active_requests": 4, "average_queued_requests": 2,
                    "active_requests": 1, "queued_requests": 0}
        observations = [interval(100, 1, 2, 4), interval(101, 3, 10, 2),
                        interval(110, 1, 3, 1), interval(111, 1, 5, 2, "restarted")]
        windows = report.throughput_windows(observations, 10)
        self.assertEqual(len(windows), 3)
        self.assertEqual(windows[0]["generated_tokens"], 12)
        self.assertEqual(windows[0]["tokens_per_second"], 3)
        self.assertEqual(windows[0]["average_active_requests"], 2.5)
        self.assertEqual(windows[0]["peak_active_requests"], 4)
        path = self.root / "state/inference-telemetry.jsonl"
        path.parent.mkdir()
        path.write_text("\n".join(json.dumps(row) for row in observations))
        data = report.load_evidence(self.root)
        self.assertEqual(data["inference"], observations)
        text = report.render_report(data, throughput_window_seconds=10)
        self.assertIn("Output tokens", text)
        self.assertIn("Active average", text)
        self.assertIn("Requests", text)
        self.assertIn("state/inference-telemetry.jsonl", text)

    def test_invalid_throughput_windows_are_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                report.throughput_windows([], value)

    def test_terminal_supervisor_overrides_stale_running_metadata_and_summary(self):
        self.write("instance.json", {"status": "running", "started": 90})
        self.write("micro-report.json", {"workflow_status": "running"})
        self.write("owner-limit.json", {"finished": 100, "returncode": 1, "owned_surviving_pids": []})
        data = report.load_evidence(self.root)
        self.assertEqual(data["instance"]["status"], "failed")
        self.assertEqual(data["raw_instance"]["status"], "running")
        text = report.render_report(data)
        self.assertIn("failed infrastructure", text)
        self.assertNotIn("snapshot of an active or unfinished run", text)
        self.assertIn("Original stale owner snapshot", text)
        self.assertIn("Terminal supervisor receipt", text)

    def test_terminal_reconciliation_preserves_detailed_failure_diagnosis(self):
        self.write("instance.json", {"status": "running"})
        self.write("micro-report.json", {"workflow_status": "failed_infrastructure", "failure": "OSError: Too many open files"})
        self.write("owner-limit.json", {"finished": 100, "returncode": 1, "owned_surviving_pids": []})
        self.assertIn("Too many open files", self.document())

    def test_stopped_supervisor_marks_active_journal_rows_interrupted_for_display(self):
        self.write("instance.json", {"status": "running"})
        self.write("owner-limit.json", {"finished": 100, "returncode": 1, "owned_surviving_pids": []})
        self.write("operations.json", [{"id": "backward", "kind": "forward_backward", "status": "running"}])
        data = report.load_evidence(self.root)
        operation = data["operations"][0]
        self.assertEqual(operation["status"], "interrupted")
        self.assertEqual(operation["journal_status"], "running")
        self.assertNotIn("finished_at", operation)

    def test_optional_monitor_failure_is_visible_without_claiming_training_failed(self):
        self.write("instance.json", {"status": "running", "started": 90})
        self.write("micro-report.json", {"workflow_status": "running"})
        self.write("monitor-health.json", {"status": "degraded", "failure_count": 2})
        data = report.load_evidence(self.root)
        self.assertEqual(data["instance"]["status"], "running")
        text = report.render_report(data)
        self.assertIn("Telemetry degraded", text)
        self.assertNotIn("failed infrastructure", text)

    def test_coordinator_complete_with_infrastructure_failure_is_not_a_pass(self):
        self.episode(status="failed", accepted=False, reward=None)
        self.write("trainer_state.json", {"status": "complete", "rft_updates_completed": 0})
        self.write("campaign-report.json", {"infrastructure_failure_count": 1, "accepted_count": 0})
        text = self.document()
        self.assertIn("infrastructure failures recorded", text)
        self.assertIn("<td>failed</td><td>No</td><td>Not recorded</td>", text)
        self.assertIn("one update does not establish learning quality", text)

    def test_artifact_content_cannot_create_active_html_or_script(self):
        malicious = '</pre><script>alert("owned")</script><img src=x onerror="alert(1)"><iframe src="https://evil.test"></iframe>'
        base = self.episode()
        self.write(base + "/world/turns/000000/request.json", {"actions": [{"name": malicious}], "trajectory_delta": malicious})
        self.write(base + "/world/turns/000000/response.json", {"tool_response": malicious})
        self.write("micro-report.json", {"failure": malicious, "workflow_status": malicious, "model": malicious})
        text = self.document(title=malicious)
        parser = Tags()
        parser.feed(text)
        for tag in ("script", "img", "iframe"):
            self.assertNotIn(tag, parser.tags)
        self.assertFalse(any(name.startswith("on") for name, _ in parser.attributes))
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("default-src 'none'", text)
        self.assertNotIn(malicious, text)

    def test_duplicate_mutation_receipts_are_counted_once_and_conflicts_reported(self):
        op = {"id": "update:optimizer", "kind": "optim_step", "status": "succeeded", "result": {"revision": 1}}
        self.write("compute/micro-operations.json", {"operations": [op, op, {**op, "status": "failed"}]})
        data = report.load_evidence(self.root)
        self.assertEqual(len(data["operations"]), 1)
        self.assertEqual(data["errors"], ["Conflicting operation receipts for ID update:optimizer"])
        self.assertIn("Conflicting operation receipts", report.render_report(data))

    def test_symlinked_artifact_is_never_read(self):
        target = self.write("outside-evidence.json", {"failure": "DO NOT EMBED EXTERNAL SOURCE"})
        (self.root / "micro-report.json").symlink_to(target)
        data = report.load_evidence(self.root)
        self.assertFalse(data["summary"])
        self.assertIn("Refused symlinked artifact", data["errors"][0])
        self.assertNotIn("DO NOT EMBED EXTERNAL SOURCE", report.render_report(data))

    def test_malformed_jsonl_preserves_later_valid_observations(self):
        (self.root / "telemetry.jsonl").write_text('{"rss_bytes": NaN}\n{"rss_bytes": 1073741824}\n')
        data = report.load_evidence(self.root)
        self.assertEqual(data["telemetry"], [{"rss_bytes": 1073741824}])
        self.assertIn("non-finite JSON value", data["errors"][0])
        self.assertIn("1.00 GiB", report.render_report(data))

    def test_exact_tokens_are_distinct_from_reserved_admission_capacity(self):
        self.write("operations.json", [{"id": "draw", "kind": "sample", "status": "succeeded",
                  "submitted_prompt_tokens": 253952, "reserved_completion_tokens": 8192,
                  "result": {"prompt_tokens": 253952, "completion_tokens": 100,
                             "prefix_cache_tokens": 200000, "policy_revision": 7, "cache_precision": "bf16"}}])
        data = report.load_evidence(self.root)
        measured = report.sample_rows(data)[0]
        self.assertEqual(measured["prefill"], 53952)
        text = report.render_report(data)
        self.assertIn("254,052", text)
        self.assertIn("53,952", text)
        self.assertIn("8,192", text)
        self.assertIn("Reserved completion is admission capacity, not generated tokens", text)
        self.assertNotIn("262,144", text)

    def test_measured_memory_is_discrete_and_unknown_swap_is_not_zero(self):
        self.write("telemetry.jsonl", [])
        (self.root / "telemetry.jsonl").write_text(json.dumps({"rss_bytes": 3*report.GIB, "allocator_active_bytes": 4*report.GIB}) + "\n")
        data = report.load_evidence(self.root)
        self.assertIsNone(report.peak(data, "swap_growth_bytes"))
        chart = report.dot_chart(report.metric_sets(data), [("RSS", "rss_bytes", "#81aee6")],
                                unit=report.GIB, title="Observed RSS", y_label="GiB")
        self.assertEqual(chart.count("<circle "), 1)
        self.assertNotIn("polyline", chart)
        self.assertNotIn("<path", chart)
        self.assertIn("no interpolation", chart)

    def test_timeline_uses_explicit_zoned_timestamps_and_preserves_overlap(self):
        phases = [{"phase": "rollouts", "started_at": "2026-10-11T08:00:00Z", "finished_at": "2026-10-11T08:00:10Z"},
                  {"phase": "sample", "started_at": "2026-10-11T08:00:02Z", "finished_at": "2026-10-11T08:00:04Z"}]
        chart = report.timeline(phases)
        self.assertEqual(chart.count("<rect "), 2)
        self.assertIn("sample: 2 seconds, 2 seconds after first interval", chart)
        self.assertIsNone(report.stamp("2026-10-11T08:00:00"))
        self.assertIn("unavailable", report.timeline([{"phase": "bad", "started_at": 10, "finished_at": 9}]))

    def test_turn_drilldowns_include_sibling_actions_observations_and_judgment(self):
        base = self.episode()
        self.write(base + "/world/turns/000000/request.json", {"actions": [{"name": "read_file"}, {"name": "run_test"}]})
        self.write(base + "/world/turns/000000/response.json", {"tool_results": [{"tool_response": "first result"}, {"tool_response": "second result"}], "turn_reward": .4})
        self.write(base + "/world/final/response.json", {"final_reward": .8, "commentary": "judgment evidence"})
        text = self.document()
        for content in ("read_file", "run_test", "first result", "second result", "judgment evidence", "complete pair"):
            self.assertIn(content, text)
        self.assertIn("<td>0.8</td>", text)

    def test_training_work_and_finite_metrics_are_rendered_from_native_receipts(self):
        self.write("operations.json", [
            {"id": "update:backward", "kind": "forward_backward", "status": "succeeded",
             "result": {"forward_metrics": {"loss": 1.25, "reduction": "sum", "prompt_tokens": 253952,
                                              "completion_tokens": 8192, "loss_tokens": 8192}}},
            {"id": "update:optimizer", "kind": "optim_step", "status": "succeeded",
             "result": {"optim_metrics": {"gradient_norm": 2.5, "optimizer_steps": 1}, "revision": 1,
                        "checkpoint": "studio-state:durable"}},
            {"id": "update:publish", "kind": "publish", "status": "succeeded",
             "result": {"revision": 1, "checkpoint": "studio-state:durable", "policy": "studio-policy:new"}}])
        text = self.document()
        for content in ("253,952", "8,192", "1.25", "2.5", "studio-state:durable", "studio-policy:new"):
            self.assertIn(content, text)

    def test_pending_backward_shows_supplied_ids_without_claiming_processed_targets(self):
        self.write("operations.json", [{"id": "pending:backward", "kind": "forward_backward", "status": "running",
                  "submitted_prompt_tokens": 133966, "reserved_completion_tokens": 5805}])
        text = self.document()
        self.assertIn("<td>pending:backward</td><td>running</td><td>133,966</td><td>5,805</td><td>Not recorded</td>", text)
        self.assertIn("supplied completion target IDs, not GPU-processed work", text)
        self.assertIn("represents generation capacity only for inference", text)
        self.assertIn("Loss-bearing target counts remain unknown", text)

    def test_active_instance_and_split_phase_events_do_not_imply_completion(self):
        self.write("instance.json", {"status": "active", "phase": "rollouts", "started": 1000,
                                     "owner_pid": 1, "service_pid": 2, "trainer_pid": 3})
        events = [{"phase": "startup", "status": "started", "timestamp": 1000},
                  {"phase": "startup", "status": "finished", "timestamp": 1010},
                  {"phase": "rollouts", "status": "started", "timestamp": 1010}]
        (self.root / "phase-events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
        data = report.load_evidence(self.root)
        self.assertEqual(data["phases"][0]["started_at"], 1000)
        self.assertEqual(data["phases"][0]["finished_at"], 1010)
        self.assertNotIn("finished_at", data["phases"][1])
        text = report.render_report(data)
        self.assertIn("active · workflow outcome not yet recorded", text)
        self.assertIn("snapshot of an active or unfinished run", text)
        self.assertEqual(report.timeline(data["phases"]).count("<rect "), 1)

    def test_polled_operation_times_are_labeled_and_not_execution_durations(self):
        self.write("operations.json", [{"id": "op", "kind": "sample", "status": "succeeded",
                  "started_at": 1000, "finished_at": 1015, "timing_source": "poll_observation",
                  "poll_interval_seconds": 1,
                  "result": {"execution_metrics": {"seconds": 12}}}])
        text = self.document()
        self.assertIn("not exact GPU execution timestamps", text)
        self.assertIn("polling observation", text)
        self.assertIn("12 s", text)
        self.assertIn("1 s", text)

    def test_telemetry_execution_metrics_wrapper_is_understood(self):
        (self.root / "telemetry.jsonl").write_text(json.dumps({"kind": "completed_operation_metrics", "timestamp": 1000,
                  "execution_metrics": {"rss_bytes": 2*report.GIB, "allocator_peak_bytes": 3*report.GIB}}))
        data = report.load_evidence(self.root)
        self.assertEqual(report.metric_sets(data)[0]["rss_bytes"], 2*report.GIB)
        self.assertEqual(report.peak(data, "allocator_peak_bytes"), 3*report.GIB)

    def test_nested_live_trainer_layout_excludes_archived_preflight_attempts(self):
        base = self.episode(status="running", accepted=None, reward=None)
        self.write("preflight-failed/episode.json", {"episode_id": "must-not-count", "status": "failed"})
        self.write("preflight-failed/trainer_state.json", {"rft_updates_completed": 99})
        self.write("train/trainer_state.json", {"status": "running", "rft_updates_completed": 0})
        self.write("train/config_snapshot.json", {"config": {"player": {"base_model": "Qwen3.8-27B"}}})
        self.write("train/segments/000/campaign_report.json", {"status": "running", "rollout_count": 0,
                                                              "accepted_count": 0, "rollouts": []})
        self.write("train/segments/000/world/rollout_01/episode.json", {"episode_id": "current", "status": "running"})
        self.write("instance.json", {"status": "running", "phase": "rollout_collection"})
        data = report.load_evidence(self.root)
        self.assertEqual(len(data["episodes"]), 1)
        self.assertEqual(data["episodes"][0]["id"], "current")
        self.assertEqual(data["trainer"]["rft_updates_completed"], 0)
        text = report.render_report(data)
        self.assertIn("running · workflow outcome not yet recorded", text)
        self.assertIn("Accepted so far", text)
        self.assertIn("<strong>1</strong><span>Episode artifacts collected", text)
        self.assertIn("Qwen3.8-27B", text)
        self.assertNotIn("must-not-count", text)

    def test_multiple_nested_campaigns_deduplicate_episode_rows(self):
        self.write("train/segments/000/campaign_report.json", {"rollouts": [
            {"episode_id": "shared", "accepted_count": 1, "provider_infrastructure_failure": False}]})
        self.write("train/segments/001/campaign_report.json", {"rollouts": [
            {"episode_id": "shared", "accepted_count": 1, "provider_infrastructure_failure": False},
            {"episode_id": "new", "accepted_count": 0, "provider_infrastructure_failure": True}]})
        data = report.load_evidence(self.root)
        self.assertEqual(data["campaign"]["rollout_count"], 2)
        self.assertEqual(data["campaign"]["accepted_count"], 1)
        self.assertEqual(data["campaign"]["infrastructure_failure_count"], 1)

    def test_partial_sampling_keeps_pending_rows_out_of_completed_totals(self):
        self.write("operations.json", [
            {"id": "done", "kind": "sample", "status": "succeeded", "result": {
                "prompt_tokens": 123, "completion_tokens": 45, "prefix_cache_tokens": 100}},
            {"id": "pending", "kind": "sample", "status": "running", "submitted_prompt_tokens": 1000}])
        text = self.document()
        self.assertIn("<strong>123</strong><span>Submitted prompt tokens", text)
        self.assertIn("<strong>45</strong><span>Sampled completion tokens", text)
        self.assertIn("<td>pending</td><td>running</td>", text)

    def test_active_campaign_finished_count_is_not_a_started_attempt_count(self):
        self.write("instance.json", {"status": "running"})
        self.write("train/segments/000/campaign_report.json", {"status": "running", "rollout_count": 0,
                                                              "accepted_count": 0, "rollouts": []})
        text = self.document()
        self.assertIn("<strong>0</strong><span>Episode artifacts collected", text)
        self.assertIn("<strong>0</strong><span>Accepted so far", text)

    def test_active_acceptance_uses_episode_audit_ahead_of_stale_wave_counter(self):
        self.episode(accepted=True)
        self.write("instance.json", {"status": "running"})
        self.write("campaign-report.json", {"status": "running", "rollout_count": 0, "accepted_count": 0})
        text = self.document()
        self.assertIn("<strong>1</strong><span>Accepted so far", text)

    def test_final_owner_workflow_and_original_deadline_origin_are_used(self):
        self.write("instance.json", {"status": "finished", "workflow_status": "passed",
                                     "original_launch_at": 1000, "started": 1100, "finished": 1300})
        self.write("owner.json", {"elapsed_seconds": 200})
        text = self.document()
        self.assertIn('class="status warn">passed</span>', text)
        self.assertIn("<strong>300 s</strong><span>Observed wall time", text)
        self.assertNotIn("snapshot of an active or unfinished run", text)

    def test_owner_success_is_rejected_when_process_evidence_contradicts_it(self):
        for fields in ({"status": "failed"}, {"status": "interrupted"}, {"status": "finished", "trainer_returncode": 7}):
            with self.subTest(fields=fields):
                self.write("instance.json", {"workflow_status": "passed", "finished": 1300, **fields})
                data = report.load_evidence(self.root)
                text = report.render_report(data)
                self.assertIn('class="status warn">failed · conflicting workflow receipt</span>', text)
                self.assertNotIn('class="status warn">passed</span>', text)
                self.assertTrue(report.workflow_conflicts(data))
                self.assertIn("Conflicting success receipt", text)

    def test_large_arrays_are_explicitly_abridged_not_silently_dropped(self):
        rendered = report.pretty({"tokens": list(range(100))})
        self.assertIn("Array display abridged", rendered)
        self.assertIn("100", rendered)
        self.assertNotIn("99", rendered)

    def test_failed_snapshot_preview_is_explicit(self):
        # Synthetic evidence keeps this regression independent of saved runs.
        self.episode(accepted=False)
        self.write("micro-report.json", {"workflow_status": "failed_infrastructure", "committed_updates": 0})
        self.write("operations.json", [{"id": "fixture-sample", "kind": "sample", "status": "succeeded",
            "result": {"prompt_tokens": 20159, "completion_tokens": 10043}}])
        self.write("telemetry.jsonl", {"swap_growth_bytes": 0})
        data = report.load_evidence(self.root)
        self.assertEqual(data["errors"], [])
        text = report.render_report(data, preview=True)
        self.assertIn("Historical preview", text)
        self.assertIn("not the current run", text)
        self.assertIn("failed infrastructure", text)
        self.assertIn("20,159", text)
        self.assertIn("10,043", text)
        self.assertIn("0.00 GiB", text)

    def test_cli_is_self_contained_and_writes_complete_html(self):
        self.episode()
        output = self.root / "output/report.html"
        result = subprocess.run([sys.executable, str(SCRIPT), "--run-root", str(self.root),
                                 "--output", str(output)], text=True, capture_output=True, check=True)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["episodes"], 1)
        self.assertEqual(receipt["artifact_errors"], [])
        text = output.read_text()
        self.assertTrue(text.startswith("<!doctype html>"))
        self.assertTrue(text.endswith("</html>"))
        self.assertNotIn('<script', text)
        self.assertNotIn('src="http', text)
        self.assertFalse(output.with_name(output.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()



@pytest.mark.parametrize("invalid", ["malformed", "symlink", "nonregular", "unreadable"])
def test_rejected_primary_operations_never_fall_back_to_stale_receipts(tmp_path, invalid):
    import os
    if invalid == "unreadable" and os.geteuid() == 0:
        pytest.skip("Root can read permission-denied fixtures")
    stale = [{"id": "stale-sample", "kind": "sample", "status": "succeeded", "result": {
        "prompt_tokens": 1234567, "completion_tokens": 1000}},
        {"id": "stale-optimizer", "kind": "optim_step", "status": "succeeded", "result": {
            "optim_metrics": {"optimizer_steps": 17}}}]
    (tmp_path / "operations.json").write_text(json.dumps(stale))
    primary = tmp_path / "compute/micro-operations.json"
    primary.parent.mkdir()
    if invalid == "malformed":
        primary.write_text("{")
    elif invalid == "symlink":
        primary.symlink_to(tmp_path / "operations.json")
    elif invalid == "nonregular":
        primary.mkdir()
    else:
        primary.write_text("{}")
        primary.chmod(0)
    try:
        data = report.load_evidence(tmp_path)
        assert data["operations"] == [] and data["errors"]
        document = report.render_report(data)
        assert "1,234,567" not in document and "stale-optimizer" not in document
    finally:
        if invalid == "unreadable":
            primary.chmod(0o600)


def test_absent_primary_operations_still_supports_legacy_snapshot(tmp_path):
    (tmp_path / "operations.json").write_text('[{"id":"legacy","kind":"sample","status":"succeeded","result":{"prompt_tokens":10,"completion_tokens":2}}]')
    data = report.load_evidence(tmp_path)
    assert [row["id"] for row in data["operations"]] == ["legacy"]
    assert not data["errors"]


def test_preflight_control_and_unknown_clients_do_not_count_as_run_work(tmp_path):
    (tmp_path / "instance.json").write_text('{"run_id":"trial"}')
    operations = [
        {"id": "rollout", "client": "trial", "kind": "sample", "status": "succeeded", "result": {
            "prompt_tokens": 10, "completion_tokens": 2, "prefix_cache_tokens": 3, "actual_prefill_tokens": 7}},
        {"id": "cold-preflight", "client": "trial-preflight", "kind": "sample", "status": "succeeded", "result": {
            "prompt_tokens": 100, "completion_tokens": 20}},
        {"id": "recovery-control", "client": "trial-preflight-control", "kind": "sample", "status": "succeeded", "result": {
            "prompt_tokens": 200, "completion_tokens": 30}},
        {"id": "unassigned", "kind": "sample", "status": "succeeded", "result": {
            "prompt_tokens": 300, "completion_tokens": 40}},
        {"id": "preflight-backward", "client": "trial-preflight", "kind": "forward_backward", "status": "succeeded",
         "result": {"forward_metrics": {"loss": 999, "prompt_tokens": 100, "completion_tokens": 4}}},
        {"id": "preflight-optimizer", "client": "trial-preflight", "kind": "optim_step", "status": "succeeded",
         "result": {"optim_metrics": {"optimizer_steps": 1}}},
    ]
    (tmp_path / "operations.json").write_text(json.dumps(operations))
    data = report.load_evidence(tmp_path)
    assert [row["id"] for row in report.sample_rows(data)] == ["rollout"]
    tokens = report.render_tokens(data)
    for value, label in [(10, "Submitted prompt tokens"), (2, "Sampled completion tokens"), (3, "Reused prefix tokens"), (7, "Actual prefill tokens")]:
        assert f"<strong>{value}</strong><span>{label}" in tokens
    assert "excluded from run totals" in tokens and "cold-preflight" in tokens and "recovery-control" in tokens
    assert "preflight-optimizer" not in report.render_publication(data)
    assert "999" not in report.render_publication(data)
    assert "trial-preflight" in report.render_execution(data)


def test_configured_training_client_takes_precedence_over_run_name(tmp_path):
    (tmp_path / "instance.json").write_text('{"run_id":"trial"}')
    (tmp_path / "config_snapshot.json").write_text('{"config":{"studio":{"training_client":"selected"}}}')
    (tmp_path / "operations.json").write_text('[{"id":"selected-sample","client":"selected","kind":"sample"},{"id":"other-sample","client":"trial","kind":"sample"}]')
    data = report.load_evidence(tmp_path)
    assert [row["id"] for row in report.sample_rows(data)] == ["selected-sample"]
