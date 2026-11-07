"""Metrics recovery through the shipped trainer and real offline orchestration.

Only the external SDK/Codex/Pi interfaces use the explicit fixtures from the
resume suite. Faults below interrupt actual filesystem appends/replaces; no
trainer, collector, credit, checkpoint, or recovery implementation is mocked.
Legacy checkpoint/cursor cases are explicitly labelled durable-input fixtures.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from shellsmith.policy.deps import load_tinker_deps
from shellsmith.process_io import write_json
from shellsmith.train import trainer
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def _run(config):
    return asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))


def _rows(config):
    return [json.loads(line) for line in
            (config.resolved_log_dir() / trainer.METRICS_FILE).read_text().splitlines()]


def _worlds(config):
    return [world.world_file.stem for world in trainer.select_world_stream(config).worlds]


def _archives(config):
    return [json.loads(path.read_text()) for path in
            sorted((config.resolved_log_dir() / "metrics_recovery").glob("*.json"))]


def test_off_cadence_optimizer_rollback_reconciles_metrics(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=4, checkpoint_every=2)
    root = config.resolved_log_dir()
    # Update 3 was logged but has no full checkpoint. Update 4 loses its sampler
    # result, so the ordinary exception/finalization path restores update 2.
    training_sdk.fail_sampler_step = 4
    with pytest.raises(RuntimeError, match="external sampler-save interruption"):
        _run(config)
    failed = json.loads((root / trainer.STATE_FILE).read_text())
    checkpoint = trainer.latest_full_checkpoint(failed)
    assert checkpoint["step"] == checkpoint["next_world_offset"] == 2
    before = (root / trainer.METRICS_FILE).read_bytes().splitlines(keepends=True)
    assert len(before) == 3
    restored = _run(config)
    assert restored["step"] == restored["next_world_offset"] == 4
    assert ("restore", checkpoint["path"], 2) in training_sdk.calls
    assert [row["worlds"][0]["world_id"] for row in _rows(config)] == _worlds(config)
    assert (root / trainer.METRICS_FILE).read_bytes().startswith(b"".join(before[:2]))
    assert len([call for call in training_sdk.calls if call[0] == "optim_step"]) == 6
    archive, = _archives(config)
    assert bytes.fromhex(archive["original_metrics_hex"]) == b"".join(before)
    assert archive["checkpoint"] == checkpoint["path"]


def test_interrupted_state_publication_never_double_counts_replayed_work(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=3, checkpoint_every=1)
    root = config.resolved_log_dir()
    replace = Path.replace
    interrupted = []

    def interrupt_state(path, target):
        if Path(target) == root / trainer.STATE_FILE:
            candidate = json.loads(path.read_text())
            if candidate["step"] >= 2 and not candidate["optimizer_update_in_flight"]:
                interrupted.append(candidate)
            if interrupted:
                # Keep finalization from publishing the same lost transaction.
                raise OSError("interrupted trainer-state replacement")
        return replace(path, target)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "replace", interrupt_state)
        with pytest.raises(OSError, match="interrupted trainer-state replacement"):
            _run(config)
    assert interrupted
    durable = json.loads((root / trainer.STATE_FILE).read_text())
    checkpoint = trainer.latest_full_checkpoint(durable)
    assert checkpoint["step"] == checkpoint["next_world_offset"] == 1
    first_row = (root / trainer.METRICS_FILE).read_bytes().splitlines(keepends=True)[0]
    restored = _run(config)
    assert restored["step"] == 3
    assert ("restore", checkpoint["path"], 1) in training_sdk.calls
    assert [row["worlds"][0]["world_id"] for row in _rows(config)] == _worlds(config)
    assert (root / trainer.METRICS_FILE).read_bytes().startswith(first_row)


@pytest.mark.parametrize("tail", ["success", "skip", "zero"])
@pytest.mark.parametrize("partial", [False, True])
def test_committed_last_row_recovers_missing_or_partial_append(
    tmp_path, monkeypatch, training_sdk, tail, partial,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=2, tail=tail, checkpoint_every=1)
    root = config.resolved_log_dir()
    open_file = Path.open
    append_count = []

    class InterruptedAppend:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, data):
            self.handle.write(data[:len(data) // 2])
            self.handle.flush()
            os.fsync(self.handle.fileno())
            raise OSError("interrupted metrics append")

    def interrupt_append(path, mode="r", *args, **kwargs):
        if path == root / trainer.METRICS_FILE and mode in {"a", "ab"}:
            append_count.append(path)
            if len(append_count) == 2:
                # The replay row must already be durable at this boundary, not
                # merely rescued by the trainer's later finally-state write.
                published = json.loads((root / trainer.STATE_FILE).read_text())
                pending = json.loads(published["metrics_head"]["tail"])
                assert pending["worlds"][0]["world_id"] == _worlds(config)[1]
                assert trainer.latest_full_checkpoint(published)["metrics_head"] == published["metrics_head"]
                if not partial:
                    raise OSError("interrupted metrics append")
                return InterruptedAppend(open_file(path, mode, *args, **kwargs))
        return open_file(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", interrupt_append)
        with pytest.raises(OSError, match="interrupted metrics append"):
            _run(config)
    durable = json.loads((root / trainer.STATE_FILE).read_text())
    checkpoint = trainer.latest_full_checkpoint(durable)
    expected_step = 2 if tail == "success" else 1
    assert checkpoint["step"] == expected_step and checkpoint["next_world_offset"] == 2
    first_row = (root / trainer.METRICS_FILE).read_bytes().splitlines(keepends=True)[0]
    before = len(training_sdk.calls)
    restored = _run(config)
    assert restored["step"] == expected_step and restored["next_world_offset"] == 2
    assert not any(call[0] == "optim_step" for call in training_sdk.calls[before:])
    rows = _rows(config)
    assert [row["worlds"][0]["world_id"] for row in rows] == _worlds(config)
    assert rows[-1]["state_checkpoint"] == checkpoint["path"]
    assert rows[-1]["datums"] == (2 if tail == "success" else 0)
    assert (root / trainer.METRICS_FILE).read_bytes().startswith(first_row)
    # Recovery must be idempotent even if there is no further collection.
    recovered = (root / trainer.METRICS_FILE).read_bytes()
    _run(config)
    assert (root / trainer.METRICS_FILE).read_bytes() == recovered


def _legacy_sparse_input(config, training_sdk, *, base=False):
    """Explicit pre-journal input, not an assertion of prior live training."""
    root = config.resolved_log_dir()
    snapshot = trainer.snapshot(config, trainer.select_world_stream(config))
    write_json(root / trainer.SNAPSHOT_FILE, snapshot)
    sampler, checkpoint = "tinker://fixture/prior-sampler", "tinker://fixture/prior-state"
    state = {
        "run_id": config.run.run_id, "status": "halted", "step": 0 if base else 1,
        "rft_updates_completed": 0 if base else 1, "optimizer_update_in_flight": False,
        "next_world_offset": 0, "settled_world_offsets": [2] if base else [1, 2],
        "segments": [], "group_identities": {}, "checkpoints": [],
    }
    if not base:
        training_sdk.sampler_steps[sampler] = 1
        training_sdk.checkpoints[checkpoint] = 1
        state["checkpoints"] = [
            {"kind": "sampler", "path": sampler, "rft_update": 1},
            {"kind": "state", "path": checkpoint, "sampler_path": sampler, "step": 1,
             "next_world_offset": 0, "settled_world_offsets": [1]},
        ]
    write_json(root / trainer.STATE_FILE, state)
    retained = {
        "step": 0, "datums": 0 if base else 2,
        "worlds": [{"world_id": snapshot["worlds"][2 if base else 1]}],
        "next_world_offset": 0, "settled_world_offsets": [2] if base else [1],
    }
    if not base:
        retained["state_checkpoint"] = checkpoint
    rolled_back = {
        "step": 0 if base else 1, "datums": 0,
        "worlds": [{"world_id": "rolled-back-no-datums"}],
        "next_world_offset": 1 if base else 0,
        "settled_world_offsets": [2] if base else [1, 2],
    }
    prefix = (json.dumps(retained) + "\n").encode()
    suffix = (json.dumps(rolled_back) + "\n").encode() + b'{"partial":'
    (root / trainer.METRICS_FILE).write_bytes(prefix + suffix)
    return prefix, suffix


@pytest.mark.parametrize("base", [False, True])
def test_legacy_sparse_cursor_keeps_committed_no_datum_rows_only(tmp_path, monkeypatch, training_sdk, base):
    config, _ = _runtime(tmp_path, monkeypatch, count=3, checkpoint_every=1)
    prefix, suffix = _legacy_sparse_input(config, training_sdk, base=base)
    restored = _run(config)
    assert restored["step"] == (2 if base else 3)
    assert restored["next_world_offset"] == 3 and restored["settled_world_offsets"] == []
    expected = [2, 0, 1] if base else [1, 0, 2]
    worlds = _worlds(config)
    assert [row["worlds"][0]["world_id"] for row in _rows(config)] == [worlds[i] for i in expected]
    assert (config.resolved_log_dir() / trainer.METRICS_FILE).read_bytes().startswith(prefix)
    assert sum(call[0] == "optim_step" for call in training_sdk.calls) == 2
    archive, = _archives(config)
    assert bytes.fromhex(archive["original_metrics_hex"]) == prefix + suffix


@pytest.mark.parametrize("tail", ["success", "skip", "zero"])
def test_first_committed_row_recovers_even_if_metrics_file_was_never_created(
    tmp_path, monkeypatch, training_sdk, tail,
):
    config, _ = _runtime(tmp_path, monkeypatch, tail=tail)
    root = config.resolved_log_dir()
    open_file = Path.open

    def interrupt_append(path, mode="r", *args, **kwargs):
        if path == root / trainer.METRICS_FILE and mode in {"a", "ab"}:
            raise OSError("metrics file creation interrupted")
        return open_file(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", interrupt_append)
        with pytest.raises(OSError, match="metrics file creation interrupted"):
            _run(config)
    assert not (root / trainer.METRICS_FILE).exists()
    durable = json.loads((root / trainer.STATE_FILE).read_text())
    assert durable["step"] == (1 if tail == "success" else 0)
    assert durable["optimizer_update_in_flight"] is False
    before = len(training_sdk.calls)
    actual = _run(config)
    assert actual["status"] == "complete" and actual["next_world_offset"] == 1
    assert not any(call[0] == "optim_step" for call in training_sdk.calls[before:])
    row, = _rows(config)
    assert row["worlds"][0]["world_id"] == _worlds(config)[0]
    assert row["datums"] == (2 if tail == "success" else 0)
    assert actual["metrics_head"]["tail"].encode() == (root / trainer.METRICS_FILE).read_bytes()


def test_rollback_archives_unappended_write_ahead_row_instead_of_replaying_it(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=4, checkpoint_every=2)
    root = config.resolved_log_dir()
    open_file = Path.open
    appends = []

    def interrupt_append(path, mode="r", *args, **kwargs):
        if path == root / trainer.METRICS_FILE and mode in {"a", "ab"}:
            appends.append(path)
            if len(appends) == 3:
                raise OSError("third metrics append interrupted")
        return open_file(path, mode, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "open", interrupt_append)
        with pytest.raises(OSError, match="third metrics append interrupted"):
            _run(config)
    durable = json.loads((root / trainer.STATE_FILE).read_text())
    assert durable["step"] == 3 and trainer.latest_full_checkpoint(durable)["step"] == 2
    prefix = (root / trainer.METRICS_FILE).read_bytes()
    assert len(prefix.splitlines()) == 2
    actual = _run(config)
    assert actual["step"] == 4
    assert [row["worlds"][0]["world_id"] for row in _rows(config)] == _worlds(config)
    assert (root / trainer.METRICS_FILE).read_bytes().startswith(prefix)
    archive, = _archives(config)
    assert bytes.fromhex(archive["original_metrics_hex"]) == prefix
    assert archive["previous_state_head"] == durable["metrics_head"]
    pending = json.loads(archive["previous_state_head"]["tail"])
    assert pending["worlds"][0]["world_id"] == _worlds(config)[2]


def test_reconciliation_failure_preserves_original_and_retries_idempotently(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=3, checkpoint_every=1)
    prefix, suffix = _legacy_sparse_input(config, training_sdk)
    root = config.resolved_log_dir()
    original_state = (root / trainer.STATE_FILE).read_bytes()
    replace = Path.replace

    def interrupt_replacement(path, target):
        if Path(target) == root / trainer.METRICS_FILE:
            archive, = _archives(config)
            assert bytes.fromhex(archive["original_metrics_hex"]) == prefix + suffix
            raise OSError("metrics reconciliation replacement interrupted")
        return replace(path, target)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "replace", interrupt_replacement)
        with pytest.raises(OSError, match="metrics reconciliation replacement interrupted"):
            _run(config)
    assert training_sdk.calls == []  # reconciliation precedes paid SDK authentication
    assert (root / trainer.STATE_FILE).read_bytes() == original_state
    assert (root / trainer.METRICS_FILE).read_bytes() == prefix + suffix
    archive_paths = sorted((root / "metrics_recovery").glob("*.json"))
    actual = _run(config)
    assert actual["step"] == 3
    assert (root / trainer.METRICS_FILE).read_bytes().startswith(prefix)
    assert sorted((root / "metrics_recovery").glob("*.json")) == archive_paths
    assert not list(root.glob(".metrics.jsonl.*.tmp"))


@pytest.mark.parametrize("damage", ["altered-prefix", "missing-prefix"])
def test_corrupt_committed_prefix_fails_closed_before_auth_without_deleting_evidence(
    tmp_path, monkeypatch, training_sdk, damage,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=2, checkpoint_every=1)
    _run(config)
    root = config.resolved_log_dir()
    metrics_path = root / trainer.METRICS_FILE
    original = metrics_path.read_bytes()
    corrupted = (original.replace(b'"datums": 2', b'"datums": 9', 1)
                 if damage == "altered-prefix" else b"")
    assert corrupted != original
    metrics_path.write_bytes(corrupted)
    original_state = (root / trainer.STATE_FILE).read_bytes()
    calls = list(training_sdk.calls)
    with pytest.raises(RuntimeError, match="Committed metrics prefix is missing or corrupt"):
        _run(config)
    assert training_sdk.calls == calls
    assert metrics_path.read_bytes() == corrupted
    assert (root / trainer.STATE_FILE).read_bytes() == original_state
    assert not _archives(config)


@pytest.mark.parametrize("damage", ["missing-row", "replayed-row", "malformed-row"])
def test_ambiguous_legacy_history_is_not_silently_dropped(tmp_path, monkeypatch, training_sdk, damage):
    config, _ = _runtime(tmp_path, monkeypatch, count=3, checkpoint_every=1)
    prefix, _ = _legacy_sparse_input(config, training_sdk)
    if damage == "missing-row":
        data = prefix[:len(prefix) // 2]
    elif damage == "replayed-row":
        data = prefix + prefix
    else:
        data = prefix + b'{"invalid":}\n'
    root = config.resolved_log_dir()
    metrics_path = root / trainer.METRICS_FILE
    metrics_path.write_bytes(data)
    original_state = (root / trainer.STATE_FILE).read_bytes()
    with pytest.raises(RuntimeError, match="[Ll]egacy metrics"):
        _run(config)
    assert not training_sdk.calls and not _archives(config)
    assert metrics_path.read_bytes() == data
    assert (root / trainer.STATE_FILE).read_bytes() == original_state
