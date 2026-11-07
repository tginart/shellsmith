"""Real Backprop and W&B journals: trajectories, recovery and provider selection."""

import asyncio
import importlib.util
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.episode.progress import emit_simple
from shellsmith.process_io import write_json
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config
from shellsmith.train.telemetry import TrainingTelemetry, require_telemetry
from shellsmith.train.durable_capture import write_durable_outcome


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr("shellsmith.train.telemetry.LOCAL_CONNECTION_FILE", None)
    for key in ("BACKPROP_CONFIG", "BACKPROP_HOST", "BACKPROP_STREAM_DATASET_RID",
                "BACKPROP_TOKEN", "FOUNDRY_TOKEN", "WANDB_MODE", "WANDB_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    config = load_train_config(None, log_dir=tmp_path / "run")
    config = replace(config, telemetry=replace(config.telemetry, capture_console=False, system_metrics=False))
    state = {"step": 0, "rft_updates_completed": 0, "next_world_offset": 0,
             "status": "running", "segments": [{"index": 0}]}
    return config, state


def _events(root):
    metadata = json.loads((root / "backprop.json").read_text())
    return [json.loads(line) for directory in metadata["sync_dirs"]
            for line in (Path(directory) / "events.jsonl").read_text().splitlines()]


def _histories(events):
    return [event["payload"]["metrics"] for event in events if event["type"] == "history"]


def _update():
    return {"step": 0, "datums": 2, "next_world_offset": 1, "seconds": 2.0,
            "worlds": [{"world_id": "sample", "decision": "GROUP_READY", "attempts": 2,
                        "accepted": 2, "discarded": 0, "rewards": [.2, .8]}],
            "forward_metrics": {"loss": .4}, "optim_metrics": {"gradient_norm": .7},
            "mlx_metrics": {"peak_memory_bytes": 1024},
            "execution_metrics": {"rss_bytes": 2048}, "logprobs": [[-1.2] * 100]}


def _episode(config, *, accepted=True):
    episode_id = config.run.run_id + "-s000-sample-01"
    root = config.resolved_log_dir() / "segments/000/sample/rollout_01"
    outcome = {"episode_id": episode_id, "world_id": "sample", "accepted_count": int(accepted),
               "policy_ref": "local-checkpoint", "audit": {"final_task": {"reward": .8}},
               "error": None if accepted else "player failed", "token_usage": {"completion_tokens": 30},
               "wall_seconds": 2.0}
    write_json(root / "episode.json", {"episode_id": episode_id, "world_id": "sample", "status": "complete"})
    write_durable_outcome(root, outcome)
    write_json(root / "world/turns/000/request.json", {
        "step_index": 0, "actions": [{"name": "terminal", "arguments": {"command": "ls"}}],
    })
    write_json(root / "world/turns/000/response.json", {
        "tool_results": [{"action_id": "call-1", "tool_response": "README.md"}], "turn_reward": .8,
    })
    records = root.parents[1] / "player_records" / (episode_id + ".jsonl")
    records.parent.mkdir(parents=True)
    records.write_text(json.dumps({"request_id": "request-1", "content": "Inspect files",
        "reasoning_content": "I should list the directory", "tool_calls": [{"name": "terminal"}],
        "prompt_token_ids": [111111] * 100, "sampled_tokens": [222222] * 100,
        "payload": {"messages": [{"content": "duplicated full context"}]},
        "authorization": "secret-marker"}) + "\n")
    (root / "weights.safetensors").write_bytes(b"model-weight-marker")
    return root, outcome


@pytest.mark.parametrize("text", [
    '[telemetry]\nprovider="prime"',
    '[telemetry]\nmode="wandb"', '[telemetry]\nproject=" "',
    '[telemetry]\nsystem_interval_seconds=0', '[telemetry]\ninit_timeout_seconds=0', '[telemetry]\nlog_trajectories=1',
])
def test_telemetry_config_rejects_invalid_settings(tmp_path, text):
    path = tmp_path / "bad.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match="telemetry"):
        load_train_config(path)


@pytest.mark.parametrize("provider", ["backprop", "wandb"])
def test_disabled_needs_no_sdk_or_logging_files(setup, monkeypatch, provider):
    config, state = setup
    config = replace(config, telemetry=replace(config.telemetry, provider=provider, mode="disabled"))
    monkeypatch.setattr("shellsmith.train.telemetry.importlib.import_module",
                        lambda _: pytest.fail("disabled telemetry imported SDK"))
    require_telemetry(config)
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        sink.update(_update())
    assert not config.resolved_log_dir().exists()


@pytest.mark.parametrize("accepted", [True, False])
def test_offline_metrics_trajectory_and_failure_without_weights(setup, accepted):
    config, state = setup
    root, outcome = _episode(config, accepted=accepted)
    with TrainingTelemetry(config, snapshot={"source_sha256": "test-source"}, state=state) as sink:
        sink.update(_update())
        sink.episode(root, outcome)
        emit_simple(episode_id=outcome["episode_id"], phase="turn_done", message="done", elapsed_seconds=.3)
        emit_simple(episode_id="another-run", phase="turn_done", message="unrelated-marker")
        state.update(status="complete", step=1, rft_updates_completed=1)
        assert sink.errors == 0
    metadata = json.loads((config.resolved_log_dir() / "backprop.json").read_text())
    assert metadata["mode"] == "offline" and metadata["run_id"] == config.run.run_id
    events = _events(config.resolved_log_dir())
    histories = _histories(events)
    update = next(row for row in histories if "forward/loss" in row)
    assert (update["train/reward_mean"], update["optim/gradient_norm"],
            update["mlx/peak_memory_bytes"], update["execution/rss_bytes"]) == (.5, .7, 1024, 2048)
    assert "logprobs" not in json.loads(update["train/update"])
    traces = [row["rollouts/trajectories"] for row in histories if "rollouts/trajectories" in row]
    assert len(traces) == 1
    trace = traces[0]["data"][0]
    assert trace[3] == accepted and trace[5] == outcome["error"]
    assert "README.md" in trace[-1] and "I should list the directory" in trace[-1]
    text = json.dumps(events)
    for excluded in ("secret-marker", "111111", "222222", "model-weight-marker", "duplicated full context", "unrelated-marker"):
        assert excluded not in text
    assert events[-1]["type"] == "run_finish"


def test_missed_publications_replay_once_across_offline_resumes(setup):
    config, state = setup
    _episode(config)
    (config.resolved_log_dir() / "metrics.jsonl").write_text(json.dumps(_update()) + "\n")
    for index in range(2):
        state["segments"] = [{"index": i} for i in range(index + 1)]
        with TrainingTelemetry(config, snapshot={}, state=state) as sink:
            assert sink.errors == 0
    histories = _histories(_events(config.resolved_log_dir()))
    assert sum("train/update_id" in row for row in histories) == 1
    assert sum("rollouts/trajectories" in row for row in histories) == 1


def test_discarded_acceptance_is_revised_in_backprop(setup):
    from shellsmith.train.durable_capture import mark_durable_outcome_discarded

    config, state = setup
    root, _ = _episode(config)
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert mark_durable_outcome_discarded(root, reason="provider fallback")
        sink.replay()
        assert sink.errors == 0
    histories = _histories(_events(config.resolved_log_dir()))
    traces = [row["rollouts/trajectories"]["data"][0] for row in histories if "rollouts/trajectories" in row]
    assert [row[3] for row in traces] == [True, False]
    assert "provider fallback" in traces[-1][-1]
    episode_rows = [row for row in histories if "rollout/episode_id" in row]
    assert episode_rows[-1]["rollout/discarded"] == 1
    assert episode_rows[-1]["rollout/rejected"] == 0


def test_disabling_then_enabling_trajectories_replays_text(setup):
    config, state = setup
    _episode(config)
    no_traces = replace(config, telemetry=replace(config.telemetry, log_trajectories=False))
    with TrainingTelemetry(no_traces, snapshot={}, state=state):
        pass
    assert not any("rollouts/trajectories" in row for row in _histories(_events(config.resolved_log_dir())))
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert sink.errors == 0
    assert sum("rollouts/trajectories" in row for row in _histories(_events(config.resolved_log_dir()))) == 1


def test_logging_failure_preserves_outcome_and_replays_episode(setup, monkeypatch):
    config, state = setup
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        root, outcome = _episode(config)
        real_log = sink.run.log
        monkeypatch.setattr(sink.run, "log", lambda *_: (_ for _ in ()).throw(OSError("journal unavailable")))

        async def runner(job, args):
            return outcome

        assert asyncio.run(sink.wrap_runner(runner)(SimpleNamespace(output_dir=root), None)) is outcome
        sink.queue.join()
        prefix = "episode:" + outcome["episode_id"] + ":"
        assert sink.errors >= 1 and not any(item.startswith(prefix) for item in sink.seen)
        monkeypatch.setattr(sink.run, "log", real_log)
        sink.replay()
        assert any(item.startswith(prefix) for item in sink.seen)
    traces = [row["rollouts/trajectories"] for row in _histories(_events(config.resolved_log_dir()))
              if "rollouts/trajectories" in row]
    assert len(traces) == 1 and len(traces[0]["data"]) == 1


def test_failure_summary_and_exit_status_keep_primary_error(setup):
    config, state = setup
    with pytest.raises(RuntimeError, match="optimizer fault"):
        with TrainingTelemetry(config, snapshot={}, state=state):
            state["status"] = "failed"
            raise RuntimeError("optimizer fault")
    events = _events(config.resolved_log_dir())
    assert events[-1]["payload"]["exit_code"] == 1
    assert "optimizer fault" in json.dumps(events)


def test_online_resume_uses_one_run_and_monotonic_sdk_history(setup, monkeypatch):
    import backprop
    config, state = setup
    config = replace(config, telemetry=replace(config.telemetry, mode="online"))
    store = backprop.MemorySink()
    real_init = backprop.init
    monkeypatch.setattr(backprop, "init", lambda **kwargs: real_init(**kwargs, _sink=store))
    for segment in range(2):
        state["segments"] = [{"index": index} for index in range(segment + 1)]
        with TrainingTelemetry(config, snapshot={}, state=state) as sink:
            sink.update(_update() | {"step": segment})
            assert sink.errors == 0
    histories = _histories(store.events)
    assert [row["train/step"] for row in histories if "train/update_id" in row] == [0, 1]
    assert {event["run_id"] for event in store.events} == {config.run.run_id}


def test_logging_settings_do_not_change_training_resume_identity():
    config = load_train_config(None)
    other = replace(config, telemetry=replace(config.telemetry, mode="disabled", project="different"))
    assert trainer._resume_key(config.to_jsonable()) == trainer._resume_key(other.to_jsonable())
    other = replace(config, telemetry=replace(config.telemetry, provider="wandb"))
    assert trainer._resume_key(config.to_jsonable()) == trainer._resume_key(other.to_jsonable())


def test_provider_selection_is_explicit_and_imports_only_selected_sdk(setup, monkeypatch, tmp_path):
    config, _ = setup
    assert config.telemetry.provider == "backprop"
    path = tmp_path / "wandb.toml"
    path.write_text('[telemetry]\nprovider="wandb"\n')
    config = load_train_config(path)
    assert config.telemetry.provider == "wandb"
    calls = []
    def selected(name):
        calls.append(name)
        return SimpleNamespace(__name__=name)

    monkeypatch.setattr("shellsmith.train.telemetry.importlib.import_module", selected)
    require_telemetry(config)
    assert calls == ["wandb"]


def test_optional_wandb_dependency_error_explains_install_and_never_falls_back(setup, monkeypatch):
    config, _ = setup
    config = replace(config, telemetry=replace(config.telemetry, provider="wandb"))
    calls = []

    def missing(name):
        calls.append(name)
        raise ImportError("SDK not installed")

    monkeypatch.setattr("shellsmith.train.telemetry.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match=r"\[wandb\]"):
        require_telemetry(config)
    assert calls == ["wandb"]


@pytest.fixture
def official_wandb():
    if importlib.util.find_spec("wandb") is None:
        pytest.skip("Optional W&B SDK is not installed")
    from shellsmith.train.telemetry import _load_sdk

    return _load_sdk("wandb")


@pytest.mark.parametrize("accepted", [True, False])
def test_real_wandb_offline_logs_metrics_and_trajectories_without_foundry_or_weights(setup, monkeypatch, accepted, official_wandb):
    config, state = setup
    config = replace(config, telemetry=replace(config.telemetry, provider="wandb", mode="offline",
        connection_file="/nonexistent/backprop-connection.json"))
    root, outcome = _episode(config, accepted=accepted)
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert sink.connection_file is None
        logged = []
        original_log = sink.run.log

        def log(values):
            logged.append(values)
            return original_log(values)

        monkeypatch.setattr(sink.run, "log", log)
        sink.update(_update())
        emit_simple(episode_id=outcome["episode_id"], phase="turn_done", message="done", elapsed_seconds=.2)
        assert sink.errors == 0 and sink.pending_receipts and not sink.receipts.exists()
        state.update(status="complete", step=1, rft_updates_completed=1)
    assert sink.errors == 0 and sink.receipts.is_file() and not sink.pending_receipts
    metadata = json.loads((config.resolved_log_dir() / "wandb.json").read_text())
    assert metadata["provider"] == "wandb" and metadata["mode"] == "offline"
    assert not (config.resolved_log_dir() / "backprop.json").exists()
    update = next(row for row in logged if "forward/loss" in row)
    assert update["train/reward_mean"] == .5 and update["optim/gradient_norm"] == .7
    journal = next(Path(metadata["sync_dir"]).glob("run-*.wandb")).read_bytes()
    assert b"forward/loss" in journal and b"train/reward_mean" in journal
    tables = [json.loads(path.read_text()) for path in Path(metadata["sync_dir"]).rglob("*.table.json")]
    traces = [row for table in tables if "trajectory" in table["columns"] for row in table["data"]]
    assert len(traces) == 1 and traces[0][3] == accepted and "README.md" in traces[0][-1]
    assert "I should list the directory" in traces[0][-1]
    assert not any(path.suffix == ".safetensors" for path in Path(metadata["sync_dir"]).rglob("*"))
    for excluded in ("secret-marker", "111111", "222222", "duplicated full context"):
        assert excluded not in json.dumps(tables)


def test_wandb_auto_honors_offline_mode_and_finish_failure_leaves_receipts_uncommitted(setup, monkeypatch, official_wandb):
    config, state = setup
    config = replace(config, telemetry=replace(config.telemetry, provider="wandb"))
    monkeypatch.setenv("WANDB_MODE", "offline")
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert sink.mode == "offline"
        sink.update(_update())
        real_finish = sink.run.finish

        def fail(*args, **kwargs):
            real_finish(*args, **kwargs)
            raise RuntimeError("simulated final flush failure")

        monkeypatch.setattr(sink.run, "finish", fail)
    assert sink.errors == 1 and not sink.receipts.exists()


def test_local_connection_is_used_by_default_and_parent_environment_is_restored(setup, tmp_path, monkeypatch):
    import backprop
    from shellsmith.train import telemetry

    config, state = setup
    connection = tmp_path / "connection.json"
    write_json(connection, {"host": "https://foundry.example", "stream_dataset_rid": "test-stream"})
    monkeypatch.setattr(telemetry, "LOCAL_CONNECTION_FILE", connection)
    real_init = backprop.init
    store = backprop.MemorySink()
    environment = dict(telemetry.os.environ)

    def init(**kwargs):
        assert kwargs["mode"] == "online"
        assert telemetry.os.environ["BACKPROP_CONFIG"] == str(connection)
        return real_init(**kwargs, _sink=store)

    monkeypatch.setattr(backprop, "init", init)
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert sink.errors == 0 and dict(telemetry.os.environ) == environment
    assert dict(telemetry.os.environ) == environment


def test_sdk_init_failure_restores_connection_environment(setup, tmp_path, monkeypatch):
    import backprop
    import os

    config, state = setup
    config = replace(config, telemetry=replace(config.telemetry, connection_file=str(tmp_path / "explicit.json")))
    original = dict(os.environ)

    def fail(**kwargs):
        raise RuntimeError("invalid connection")

    monkeypatch.setattr(backprop, "init", fail)
    with pytest.raises(RuntimeError, match="invalid connection"):
        with TrainingTelemetry(config, snapshot={}, state=state):
            pass
    assert dict(os.environ) == original


# These replace only external providers. The shipped collector, audits, credit
# bridge, endpoint, optimizer loop, checkpoint publication and sink all run.
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def test_shared_trainer_publishes_real_rollouts_and_committed_update(tmp_path, monkeypatch, training_sdk):
    from shellsmith.policy.deps import load_tinker_deps

    config, _ = _runtime(tmp_path, monkeypatch)
    config = replace(config, telemetry=replace(config.telemetry, mode="offline", capture_console=False, system_metrics=False))
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert state["status"] == "complete" and training_sdk.updates == 1
    events = _events(config.resolved_log_dir())
    histories = _histories(events)
    assert sum("rollouts/trajectories" in row for row in histories) == 2
    assert sum("train/update_id" in row for row in histories) == 1
    assert any("events/progress" in row for row in histories)
    summaries = [event["payload"]["values"] for event in events if event["type"] == "summary"]
    assert any(value.get("trainer/status") == "complete" and value.get("telemetry/logging_errors") == 0
               for value in summaries)


def test_update_receipts_preserve_each_operation_measurement():
    from studio_compute.client import combine_update_receipts

    row = combine_update_receipts(
        {"forward_metrics": {"loss": .3}, "execution_metrics": {"seconds": 9, "allocator_peak_bytes": 900}},
        {"optim_metrics": {"optimizer_steps": 1}, "execution_metrics": {"seconds": 2, "allocator_peak_bytes": 950}},
    )
    assert row["backward_execution_metrics"]["seconds"] == 9
    assert row["optimizer_execution_metrics"]["seconds"] == 2
    assert row["backward_execution_metrics"]["allocator_peak_bytes"] == 900
    assert row["optim_metrics"]["optimizer_steps"] == 1


def test_queue_pressure_defers_to_durable_replay(setup):
    import queue

    config, state = setup
    _episode(config)
    (config.resolved_log_dir() / "metrics.jsonl").write_text(json.dumps(_update()) + "\n")
    with TrainingTelemetry(config, snapshot={"worlds": ["sample"]}, state=state) as sink:
        sink.stop.set()
        sink.worker.join(timeout=5)
        sink.queue = queue.Queue(maxsize=1)
        sink.enqueue("update", _update())
        sink.enqueue("update", _update())
        assert sink.dropped == 1
    histories = _histories(_events(config.resolved_log_dir()))
    assert sum("train/update_id" in row for row in histories) == 1


@pytest.mark.parametrize("character", ["a", '"', "\\", "🦦"])
def test_trajectory_upload_size_is_bounded_and_secrets_are_removed(character):
    from shellsmith.train.telemetry import _trajectory_json, MAX_TRAJECTORY_BYTES

    text = _trajectory_json({"text": character * 200000, "custom_api_key": "hidden"})
    value = json.loads(text)
    assert value["truncated"] and value["preview"]
    assert len(text.encode("utf-8")) <= MAX_TRAJECTORY_BYTES
    assert value["original_bytes"] > MAX_TRAJECTORY_BYTES
    assert "hidden" not in text


def test_prepared_epoch_pins_all_worlds_and_final_partial_update():
    from shellsmith.train.world_selection import select_world_stream, freeze_worlds
    from shellsmith.paths import PROJECT_ROOT

    config = load_train_config(PROJECT_ROOT / "examples/studio-catalog-epoch.toml")
    worlds = freeze_worlds(select_world_stream(config).worlds)
    assert len(worlds) == 177
    assert config.adaptive.initial_attempts == config.adaptive.maximum_attempts == 8
    assert config.adaptive.parallelism == config.adaptive.parallelism_per_provider == 8
    assert config.train.groups_per_update == 2
    assert config.train.steps == (len(worlds) + 1) // 2 == 89
    assert all(world.selected.expected_sha256 for world in worlds)
    groups = [worlds[start:start + 2] for start in range(0, len(worlds), 2)]
    assert len(groups) == 89 and len(groups[-1]) == 1
    assert [world for group in groups for world in group] == list(worlds)


def test_policy_publication_duration_replays_without_duplicate_counts(setup):
    config, state = setup
    receipt = {"step": 1, "policy_ref": "test-policy", "seconds": .25, "segment": 0}
    checkpoint = {"step": 1, "policy_ref": "test-policy", "segment": 0,
                  "kind": "state_checkpoint", "state_checkpoint_seconds": .5}
    write_json(config.resolved_log_dir() / "policy-publications/0-1.json", receipt)
    write_json(config.resolved_log_dir() / "policy-publications/checkpoint.json", checkpoint)
    for _ in range(2):
        with TrainingTelemetry(config, snapshot={}, state=state) as sink:
            sink.publication(receipt)
            sink.publication(checkpoint)
    rows = _histories(_events(config.resolved_log_dir()))
    publications = [row for row in rows if "publication/id" in row]
    assert len(publications) == 2
    by_kind = {row["publication/kind"]: row for row in publications}
    assert by_kind["policy"]["publication/seconds"] == .25
    assert by_kind["state_checkpoint"]["publication/state_checkpoint_seconds"] == .5


def test_gm_call_totals_include_events_coalesced_for_upload(setup):
    config, state = setup
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        for _ in range(3):
            emit_simple(episode_id=config.run.run_id + "-s000-world-01",
                        phase="turn_done", message="done", elapsed_seconds=2)
        assert sink.gm_calls == 3 and sink.gm_seconds == 6


def test_sdk_initialization_timeout_uses_configured_bound(setup, monkeypatch):
    import backprop

    config, state = setup
    config = replace(config, telemetry=replace(config.telemetry, init_timeout_seconds=240.0))
    original = backprop.init

    def initialize(**kwargs):
        assert kwargs["settings"].init_timeout == 240.0
        return original(**kwargs)

    monkeypatch.setattr(backprop, "init", initialize)
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert sink.errors == 0



def test_both_operation_metrics_use_the_optimizer_step(setup):
    config, state = setup
    with TrainingTelemetry(config, snapshot={}, state=state):
        pass
    definitions = {event["payload"]["name"]: event["payload"]
                   for event in _events(config.resolved_log_dir())
                   if event["type"] == "metric_definition"}
    for prefix in ("backward_execution", "optimizer_execution"):
        assert definitions[prefix + "/*"]["step_metric"] == "train/optimizer_updates_committed"


def test_live_operations_have_independent_progress_records(setup, monkeypatch):
    config, state = setup
    config = replace(config, compute=replace(config.compute, backend="studio"))
    live = {"active_requests": 1, "queued_requests": 2, "training_operations": {
        "backward-a": {"kind": "forward_backward", "progress": {
            "phase": "backward", "completed_layers": 7}},
        "optimizer-b": {"kind": "optim_step", "progress": {
            "phase": "optimizer", "optimizer_tensors": 11}},
    }}
    monkeypatch.setattr("studio_compute.client.Client.metrics", lambda self, **kwargs: live)
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        sink._health()
        assert sink.errors == 0
    rows = _histories(_events(config.resolved_log_dir()))
    operations = {row["compute/operation_id"]: row
                  for row in rows if "compute/operation_id" in row}
    assert set(operations) == {"backward-a", "optimizer-b"}
    assert operations["backward-a"]["compute/progress/completed_layers"] == 7
    assert "compute/progress/optimizer_tensors" not in operations["backward-a"]
    assert operations["optimizer-b"]["compute/progress/optimizer_tensors"] == 11
    assert "compute/progress/completed_layers" not in operations["optimizer-b"]


def test_no_checkpoint_receipt_for_a_step_zero_tinker_exit(
    tmp_path, monkeypatch, training_sdk,
):
    from shellsmith.policy.deps import load_tinker_deps

    config, _ = _runtime(tmp_path, monkeypatch)
    config = replace(config, telemetry=replace(
        config.telemetry, mode="offline", capture_console=False, system_metrics=False))
    monkeypatch.setattr(trainer, "build_tinker_group", lambda **kwargs: SimpleNamespace(
        datums=(), segments=(), history_rewrites=()))
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert state["status"] == "complete" and state["rft_updates_completed"] == 0
    assert training_sdk.updates == 0
    assert not (config.resolved_log_dir() / "policy-publications").exists()


@pytest.mark.parametrize("log_trajectories", [False, True])
def test_deferred_replay_keeps_the_original_segment(setup, log_trajectories) -> None:
    """
    Purpose: Preserve event provenance when a later process replays deferred files.
    Input: Segment-zero files and a telemetry process for segment two.
    Output: Replayed update and episode events still identify segment zero.
    """
    config, state = setup
    config = replace(config, telemetry=replace(
        config.telemetry, log_trajectories=log_trajectories))
    _episode(config)
    row = _update() | {"segment": 0}
    (config.resolved_log_dir() / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    state["segments"] = [{"index": index} for index in range(3)]
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        assert sink.segment == 2 and sink.errors == 0
    histories = _histories(_events(config.resolved_log_dir()))
    replayed = [row for row in histories
                if "train/update_id" in row or "rollout/episode_id" in row]
    assert len(replayed) == 2
    assert all(row["telemetry/segment"] == 0 for row in replayed)


def test_legacy_update_does_not_guess_the_resume_segment(setup) -> None:
    config, state = setup
    state["segments"] = [{"index": index} for index in range(3)]
    with TrainingTelemetry(config, snapshot={}, state=state) as sink:
        sink.update(_update())
        assert sink.errors == 0
    row = next(row for row in _histories(_events(config.resolved_log_dir()))
               if "train/update_id" in row)
    assert row["telemetry/segment"] is None
