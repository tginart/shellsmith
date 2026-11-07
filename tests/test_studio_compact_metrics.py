"""Token outputs remain in RPC receipts, outside journal/state/checkpoint metadata."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from shellsmith.train.backend import TokenSequence
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection
from shellsmith.train.durable_metrics import MetricsJournal
from shellsmith.train.trainer import submit_update, _ensure_full_checkpoint, _recover_studio_pending


@pytest.mark.parametrize("recovery", [False, True])
def test_per_token_outputs_never_enter_durable_metrics(tmp_path, recovery):
    raw = {"forward_metrics": {"loss": 2.5, "completion_tokens": 20000},
           "logprobs": [[-0.123456789] * 20000], "execution_metrics": {"seconds": 1.2}}
    optimizer = {"optim_metrics": {"optimizer_steps": 1}, "checkpoint": "studio-state:committed"}
    checkpoint_metadata = []
    class Backend:
        update = AsyncMock(return_value={**raw, **optimizer})
        async def operation_status(self, request_id):
            return {"status": "succeeded", "result": optimizer if request_id.endswith("optimizer") else raw}
        async def restore_state(self, *args, **kwargs): pass
        async def save_checkpoint(self, *, kind, name, metadata=None):
            if metadata is not None:
                checkpoint_metadata.append(metadata)
            return "studio-policy:p1" if kind == "sampler" else "studio-state:c1"
        async def publish_policy(self, *, revision): return "studio-policy:p1"
    backend = Backend()
    state = {"run_id": "metrics", "step": 0, "rft_updates_completed": 0, "next_world_offset": 0,
             "settled_world_offsets": [], "segments": [{}], "committed_group_ids": [], "checkpoints": [],
             "compute_backend": "studio", "policy_identity": {"model": "fixture"}}
    state_path = tmp_path / "state.json"
    journal = MetricsJournal.recover(tmp_path / "metrics.jsonl", state=state, checkpoint=None)
    row = {"step": 0, "segment": 0, "datums": 1, "next_world_offset": 2, "settled_world_offsets": []}
    async def publish():
        if recovery:
            state["pending_studio_update"] = {"id": "u", "step": 0, "started": 0., "metrics": row,
                "group_ids": ["g"], "next_world_offset": 2, "settled_world_offsets": []}
            await _recover_studio_pending(backend=backend, state=state, state_path=state_path, journal=journal)
        else:
            config = TrainConfig(run=RunSection(run_id="metrics"), compute=ComputeSection(backend="studio"))
            sequence = TokenSequence((1,), (2,), (-1.,), (.1,), (1.,), "sample")
            result = await submit_update(deps=None, training_client=backend, reference_client=None,
                datums=[SimpleNamespace(datum=sequence)], config=config)
            row.update(result)
            state.update(step=1, rft_updates_completed=1, next_world_offset=2)
            await _ensure_full_checkpoint(training_client=backend, state=state, sampler_path="studio-policy:p1",
                                          metrics_head=journal.preview_head(row))
            journal.publish(row=row, state_path=state_path, state=state, checkpoint=state["checkpoints"][-1])
    asyncio.run(publish())
    assert raw["logprobs"][0] == [-0.123456789] * 20000
    assert raw["forward_metrics"]["completion_tokens"] == 20000
    for data in [(tmp_path / "metrics.jsonl").read_text(), state_path.read_text(), json.dumps(checkpoint_metadata)]:
        assert "logprobs" not in data
        assert len(data) < 5000
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["forward_metrics"]["loss"] == 2.5
    durable_row = json.loads((tmp_path / "metrics.jsonl").read_text())
    state_tail = json.loads(json.loads(state_path.read_text())["metrics_head"]["tail"])
    checkpoint_tail = json.loads(checkpoint_metadata[-1]["metrics_head"]["tail"])
    for item in (durable_row, state_tail, checkpoint_tail):
        assert item["forward_metrics"]["completion_tokens"] == 20000
        assert item["segment"] == 0
