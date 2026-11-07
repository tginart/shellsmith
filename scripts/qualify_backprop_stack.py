"""
Purpose: Publish and read back real Studio qualification receipts and failure fixtures.
Input: A completed disposable RPC probe, optional memorization report, and existing credentials.
Output: A sanitized receipt with verified event and table readback.
Note: This command performs no GPU work and never uploads token replay arrays.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
import json
from pathlib import Path
import time

from shellsmith.process_io import write_json
from shellsmith.train.config import load_train_config
from shellsmith.train.durable_capture import write_durable_outcome
from shellsmith.train.telemetry import TrainingTelemetry
from studio_compute.client import combine_update_receipts
from observe_studio_metrics import has_backward_progress


def qualify(args: argparse.Namespace) -> None:
    """
    Purpose: Verify durable replay, failure visibility, scalar metrics, and table bytes.
    Input: Passed model qualification reports and an explicit test run ID.
    Output: A readback receipt. Any missing event or metric fails this command.
    """
    from backprop.credential_store import default_path, read_token
    from backprop.foundry import FoundryStore

    report = json.loads(args.rpc_report.read_text())
    assert report["status"] == "passed", "RPC qualification must pass first."
    config = load_train_config(None)
    config = replace(
        config,
        run=replace(config.run, run_id=args.run_id, log_dir=str(args.output)),
        compute=replace(config.compute, backend="studio"),
        telemetry=replace(config.telemetry, mode="online", project="shellsmith-qualification",
                          connection_file=str(args.connection), capture_console=False,
                          system_metrics=False, system_interval_seconds=3600.0),
    )
    deployment = json.loads(args.connection.read_text())
    deployment["timeout"] = 15
    store = FoundryStore(deployment, token=read_token(default_path(), deployment["host"]))
    before = store.request(store.stream_path + "/getEndOffsets?" + store._stream_query())
    operations = {row["id"]: row["result"] for row in report["operations"]}
    row = {
        "step": 0, "segment": 0, "optimizer_updates_committed": 1, "next_world_offset": 1,
        "worlds": [{"world_id": "rpc-fixture", "decision": "GROUP_READY", "attempts": 2,
                    "accepted": 2, "discarded": 0, "rewards": [0.1, -0.1]}],
        **combine_update_receipts(operations["tiny-backward"], operations["tiny-optimizer"]),
    }
    assert row["optim_metrics"]["optimizer_steps"] == 1
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    episode_root = args.output / "segments/000/rpc-fixture/rollout_01"
    outcome = {"episode_id": args.run_id + "-s000-rpc-fixture-01", "world_id": "rpc-fixture",
               "accepted_count": 0, "error": "injected qualification failure; no paid GM call",
               "token_usage": {"completion_tokens": 0}, "wall_seconds": 0.01}
    write_durable_outcome(episode_root, outcome)
    state = {"step": 1, "rft_updates_completed": 1, "next_world_offset": 1,
             "status": "failed", "segments": [{"index": 0}]}
    validator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    snapshot = {"publication_validator_sha256": validator_sha256, "scope": "Real 27B RPC update; synthetic failed episode and interruption fixtures",
                "worlds": ["rpc-fixture"], "source": json.loads(args.source_identity.read_text()),
                "policy_identity": report["services"][0]["capabilities"]}
    try:
        with TrainingTelemetry(config, snapshot=snapshot, state=state) as sink:
            checks = [[name, json.dumps(value)] for name, value in report["checks"].items()]
            if args.live_observation:
                live = json.loads(args.live_observation.read_text())
                samples = live["samples"]
                assert live["saw_backward"], "The observer captured no backward phase."
                backward = next(row for row in samples if has_backward_progress(row))
                samples = [row for row in samples if row["session_id"] == backward["session_id"]]
                active = max(row["active_requests"] for row in samples)
                queued = max(row["queued_requests"] for row in samples)
                batch = max(row["last_admitted_batch_size"] for row in samples)
                assert active > 0 and 1 < batch <= 4
                checks.append(["live_service_observation", json.dumps({
                    "passed": True, "samples": len(samples), "max_active_requests": active,
                    "max_queued_requests": queued, "max_admitted_batch_size": batch,
                    "training_operations": backward["training_operations"],
                    "latest_measured_interval": samples[-1]["latest_interval"],
                })])
            if args.memorization:
                memory = json.loads(args.memorization.read_text())
                assert memory["status"] == "passed"
                checks.append(["fixed_memorization_restore", json.dumps({
                    "status": memory["status"], "fixture_sha256": memory["fixture_sha256"],
                    "attempts": [{"seed": a["seed"], "learning_rate": a["learning_rate"],
                                  "status": a["status"], "restore_exact": a["restore_exact"],
                                  "initial_nll": a["initial"]["mean_nll"],
                                  "final_nll": a["final"]["mean_nll"],
                                  "exact_answer": a["final"]["exact_answer"]}
                                 for a in memory["attempts"]],
                })])
            sink.run.log({"qualification/checks": sink.sdk.Table(
                columns=["check", "receipt"], data=checks)})
            raise InterruptedError("injected qualification interruption after committed update")
    except InterruptedError:
        pass
    state.update(status="complete", segments=[{"index": 0}, {"index": 1}])
    with TrainingTelemetry(config, snapshot=snapshot, state=state) as resumed:
        assert resumed.errors == 0
    events = []
    metadata = json.loads((args.output / "backprop.json").read_text())
    for directory in metadata["sync_dirs"]:
        events.extend(json.loads(line) for line in (Path(directory) / "events.jsonl").read_text().splitlines())
    required = {event["event_id"] for event in events}
    observed = _read_back(store, before, required)
    histories = [event["payload"]["metrics"] for event in observed if event["type"] == "history"]
    updates = [row for row in histories if "train/update_id" in row]
    assert len(updates) == 1, "Resume must not count the committed update twice."
    assert updates[0]["optim/optimizer_steps"] == 1
    assert "backward_execution/seconds" in updates[0]
    assert "optimizer_execution/seconds" in updates[0]
    assert any(event["type"] == "run_finish" and event["payload"]["exit_code"] == 1 for event in observed)
    assert any("injected qualification failure" in json.dumps(row) for row in histories)
    references = list(_references(observed))
    downloaded = [json.loads(store.download_file(reference)) for reference in references]
    assert downloaded, "No remote table bytes were verified."
    assert any("injected qualification failure" in json.dumps(value) for value in downloaded)
    write_json(args.output / "readback.json", {
        "status": "passed", "run_id": args.run_id, "url": metadata["url"],
        "events_required": len(required), "events_read_back": len(observed),
        "tables_downloaded": len(downloaded), "committed_update_rows": len(updates),
        "optimizer_steps": 1, "failure_and_interruption_visible": True,
        "source": snapshot["source"], "publication_validator_sha256": validator_sha256,
        "timestamp": time.time(),
    })


def _read_back(store, before: dict, required: set[str]) -> list[dict]:
    """
    Purpose: Read only the bounded stream tail that follows this probe.
    Input: Captured offsets and expected stable event IDs.
    Output: The complete set of matching remote events, or an assertion error.
    """
    observed = {}
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        ends = store.request(store.stream_path + "/getEndOffsets?" + store._stream_query())
        count = 0
        for partition, end in ends.items():
            offset = int(before.get(partition, 0))
            while offset < int(end):
                rows = store.request(store.stream_path + "/getRecords?" + store._stream_query(
                    partitionId=partition, startOffset=str(offset), limit=100))
                if not rows:
                    break
                for row in rows:
                    record = row["value"]
                    if record["event_id"] in required:
                        observed[record["event_id"]] = store.record_to_event(record)
                offset = max(int(row["offset"]) for row in rows) + 1
                count += len(rows)
                assert count <= 5000, "Readback exceeded the bounded stream tail."
        if required <= observed.keys():
            return list(observed.values())
        time.sleep(2)
    raise AssertionError(f"Missing {len(required - observed.keys())} remote events")


def _references(value):
    if isinstance(value, dict):
        if {"dataset_rid", "path", "sha256"} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, list):
        for item in value:
            yield from _references(item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc-report", type=Path, required=True, help="Passed smoke_studio_runtime report.")
    parser.add_argument("--memorization", type=Path, help="Passed fixed memorization report.")
    parser.add_argument("--live-observation", type=Path, help="CPU metrics samples from the real qualification service.")
    parser.add_argument("--source-identity", type=Path, required=True, help="Exact scoped source hashes.")
    parser.add_argument("--output", type=Path, required=True, help="Fresh disposable publication folder.")
    parser.add_argument("--run-id", required=True, help="Clearly labeled qualification run ID.")
    parser.add_argument("--connection", type=Path,
                        default=Path.home() / ".config/shellsmith/backprop.json",
                        help="Existing deployment config. Credentials remain in the credential store.")
    qualify(parser.parse_args())
