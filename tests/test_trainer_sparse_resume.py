"""Resume an explicit halted-checkpoint input with an unfinished stream gap.

The prior checkpoint is a durable-input fixture, not a claimed live training
result. Resume, collection, audits, credits, updates, and new checkpoints run
through shipped code with only the external SDK/CLI interfaces supplied.
"""
import asyncio
import json

from shellsmith.policy.deps import load_tinker_deps
from shellsmith.process_io import write_json
from shellsmith.train import trainer
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def test_sparse_halt_checkpoint_resumes_gap_without_retraining_settled_world(tmp_path, monkeypatch, training_sdk):
    config, pi_audit = _runtime(tmp_path, monkeypatch, count=3, checkpoint_every=1)
    stream = trainer.select_world_stream(config)
    snapshot = trainer.snapshot(config, stream)
    prior_sampler, prior_state = "tinker://fixture/prior-sampler", "tinker://fixture/prior-state"
    # World 1 finished and was updated before worlds 0/2 were interrupted.
    # This is the exact sparse state produced by the real collector regression.
    training_sdk.sampler_steps[prior_sampler] = 1
    training_sdk.checkpoints[prior_state] = 1
    state = {
        "run_id": config.run.run_id, "status": "halted", "step": 1, "rft_updates_completed": 1,
        "next_world_offset": 0, "settled_world_offsets": [1], "segments": [], "group_identities": {},
        "checkpoints": [
            {"kind": "sampler", "path": prior_sampler, "rft_update": 1},
            {"kind": "state", "path": prior_state, "sampler_path": prior_sampler, "step": 1,
             "next_world_offset": 0, "settled_world_offsets": [1]},
        ],
    }
    root = config.resolved_log_dir()
    write_json(root / trainer.SNAPSHOT_FILE, snapshot)
    write_json(root / trainer.STATE_FILE, state)
    actual = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert ("restore", prior_state, 1) in training_sdk.calls
    assert actual["step"] == 3 and actual["next_world_offset"] == 3 and actual["status"] == "complete"
    assert actual["settled_world_offsets"] == []
    assert sum(call[0] == "optim_step" for call in training_sdk.calls) == 2
    metrics = [json.loads(line) for line in (root / trainer.METRICS_FILE).read_text().splitlines()]
    assert [row["worlds"][0]["world_id"] for row in metrics] == [snapshot["worlds"][0], snapshot["worlds"][2]]
    assert [row["next_world_offset"] for row in metrics] == [2, 3]
    for path in (root / "segments").rglob("durable_outcome.json"):
        outcome = json.loads(path.read_text())["outcome"]
        assert outcome["world_id"] != snapshot["worlds"][1]
    assert len(pi_audit.read_text().splitlines()) == 8
    checkpoint = trainer.latest_full_checkpoint(actual)
    assert checkpoint["settled_world_offsets"] == [] and checkpoint["next_world_offset"] == 3
    assert training_sdk.checkpoints[checkpoint["path"]] == 3


def test_full_checkpoint_serializes_sparse_progress_and_detects_its_change():
    from types import SimpleNamespace
    names = []

    class Client:
        async def save_state_async(self, name):
            names.append(name)

            async def result_async():
                return SimpleNamespace(path="fixture://" + name)

            return SimpleNamespace(result_async=result_async)

    state = {"run_id": "sparse", "step": 1, "next_world_offset": 0,
             "settled_world_offsets": [2], "checkpoints": []}

    async def run():
        first = await trainer._ensure_full_checkpoint(training_client=Client(), state=state, sampler_path="sampler")
        assert trainer.latest_full_checkpoint(state)["settled_world_offsets"] == [2]
        state["settled_world_offsets"] = [1, 2]
        second = await trainer._ensure_full_checkpoint(training_client=Client(), state=state, sampler_path="sampler")
        assert first != second
        assert state["checkpoints"][0]["settled_world_offsets"] == [2]
        assert trainer.latest_full_checkpoint(state)["settled_world_offsets"] == [1, 2]

    asyncio.run(run())
    assert len(names) == len(set(names)) == 2
