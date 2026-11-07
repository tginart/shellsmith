"""Unit tests for durable Tinker capture serialize/load and resume helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pytest

from shellsmith.train.durable_capture import (
    PolicyProvenance,
    clear_pending_update,
    load_capture,
    load_captures_from_index,
    load_durable_outcome,
    load_pending_update,
    persist_accepted_capture,
    resolve_policy_stamp,
    serialize_capture,
    write_durable_outcome,
    write_pending_update,
)
from shellsmith.train.capture_store import (
    CaptureStore,
)
from shellsmith.credit.to_tinker_bridge import (
    CapturedRollout,
    CapturedModelTurn,
)


@dataclass
class FakePrompt:
    tokens: list[int]

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    @property
    def length(self) -> int:
        return len(self.tokens)

    def append(self, other: object) -> "FakePrompt":
        return FakePrompt(self.tokens + list(getattr(other, "tokens", other)))


@dataclass
class FakeRecord:
    request_id: str
    created_at: float
    prompt: FakePrompt
    prompt_tokens: int
    sampled_tokens: list[int]
    sampled_logprobs: list[float]
    content: str = "hi"
    finish_reason: str = "stop"
    payload: dict | None = None
    hydrations: list | None = None
    reasoning_content: str = ""
    forced_close_token_span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.payload is None:
            self.payload = {}
        if self.hydrations is None:
            self.hydrations = []


class FakeDeps:
    class tinker:
        class types:
            class ModelInput:
                @staticmethod
                def from_ints(tokens: list[int]) -> FakePrompt:
                    return FakePrompt(list(tokens))


def _sample_capture(tmp_path: Path) -> CapturedRollout:
    record = FakeRecord(
        request_id="req-1",
        created_at=1.0,
        prompt=FakePrompt([10, 11, 12]),
        prompt_tokens=3,
        sampled_tokens=[20, 21],
        sampled_logprobs=[-0.1, -0.2],
    )
    return CapturedRollout(
        rollout_root=tmp_path / "task" / "rollout_01",
        group_identity_key="group-key",
        captures=(
            CapturedModelTurn(
                record=record,
                kind="final_text",
                world_turn_indices=(),
            ),
        ),
    )


def test_serialize_load_capture_round_trip(tmp_path: Path) -> None:
    capture = _sample_capture(tmp_path)
    payload = serialize_capture(capture, episode_id="ep-1")
    assert payload["records"][0]["prompt_token_ids"] == [10, 11, 12]
    path = tmp_path / "capture.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    loaded = load_capture(path, deps=FakeDeps())
    assert loaded.group_identity_key == "group-key"
    assert len(loaded.captures) == 1
    assert loaded.captures[0].kind == "final_text"
    assert list(loaded.captures[0].record.prompt.to_ints()) == [10, 11, 12]
    assert list(loaded.captures[0].record.sampled_tokens) == [20, 21]


def test_persist_accepted_capture_updates_index(tmp_path: Path) -> None:
    rollout = tmp_path / "geo" / "rollout_01"
    rollout.mkdir(parents=True)
    capture = CapturedRollout(
        rollout_root=rollout,
        group_identity_key="gk",
        captures=_sample_capture(tmp_path).captures,
    )
    provenance = PolicyProvenance(
        model_name="Qwen/Qwen3.6-35B-A3B",
        lora_rank=64,
        policy_kind="base",
        policy_ref=None,
        rft_updates_completed=0,
        group_identity_key="gk",
        episode_id="ep-1",
    )
    persist_accepted_capture(
        run_dir=tmp_path,
        episode_id="ep-1",
        capture=capture,
        provenance=provenance,
    )
    assert (rollout / "logs" / "tinker_trainable_capture.json").is_file()
    assert (rollout / "policy_provenance.json").is_file()
    store = CaptureStore.load_from_run_dir(tmp_path, deps=FakeDeps())
    assert store.get("ep-1") is not None
    loaded = load_captures_from_index(tmp_path, deps=FakeDeps())
    assert set(loaded) == {"ep-1"}


def test_policy_provenance_requires_ref_after_rft() -> None:
    with pytest.raises(RuntimeError, match="sampler_checkpoint"):
        PolicyProvenance(
            model_name="m",
            lora_rank=8,
            policy_kind="base",
            policy_ref=None,
            rft_updates_completed=1,
            group_identity_key="g",
            episode_id="e",
        ).to_jsonable()


def test_resolve_policy_stamp_uses_current_rft_sampler() -> None:
    state = {
        "rft_updates_completed": 2,
        "checkpoints": [
            {"kind": "sampler", "path": "tinker://old", "rft_update": 1},
            {"kind": "state", "path": "tinker://state", "rft_update": 2},
            {"kind": "sampler", "path": "tinker://new", "rft_update": 2},
        ],
    }
    stamp = resolve_policy_stamp(
        state=state,
        model_name="m",
        lora_rank=8,
        episode_id="e",
        group_identity_key="g",
    )
    assert stamp.policy_kind == "sampler_checkpoint"
    assert stamp.policy_ref == "tinker://new"


def test_resolve_policy_stamp_rejects_a_stale_sampler_checkpoint() -> None:
    state = {
        "rft_updates_completed": 2,
        "checkpoints": [
            {"kind": "sampler", "path": "tinker://step-1", "rft_update": 1},
        ],
    }

    with pytest.raises(RuntimeError, match=r"rft_update=2"):
        resolve_policy_stamp(
            state=state,
            model_name="m",
            lora_rank=8,
            episode_id="e",
            group_identity_key="g",
        )


def test_durable_outcome_round_trip(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout_01"
    rollout.mkdir()
    write_durable_outcome(rollout, {"episode_id": "e1", "accepted_count": 1})
    loaded = load_durable_outcome(rollout)
    assert loaded is not None
    assert loaded["episode_id"] == "e1"


def test_pending_update_write_load_clear(tmp_path: Path) -> None:
    write_pending_update(
        tmp_path,
        {
            "step": 0,
            "next_world_offset": 2,
            "accepted_group_count": 2,
            "tasks": [],
        },
    )
    pending = load_pending_update(tmp_path)
    assert pending is not None
    assert pending["next_world_offset"] == 2
    clear_pending_update(tmp_path)
    assert load_pending_update(tmp_path) is None


def test_run_campaign_wave_skips_durable_outcomes(tmp_path: Path) -> None:
    """Finished episodes with durable_outcome.json are not re-run on resume."""
    import argparse
    import asyncio

    from shellsmith.campaign.stream_campaign import (
        OpenCampaignSpec,
        build_task_attempt_jobs,
        open_campaign,
        run_campaign_wave,
    )

    handle = open_campaign(
        tmp_path / "camp",
        OpenCampaignSpec(
            campaign_id="camp",
            mode="tinker_train",
            parallelism=2,
            parallelism_per_provider=2,
            provider_outage="halt",
            timeout_seconds=60.0,
            extra={},
        ),
    )
    world_file = tmp_path / "task.md"
    world_file.write_text("x", encoding="utf-8")
    jobs = build_task_attempt_jobs(
        campaign_id=handle.campaign_id,
        campaign_root=handle.root,
        world_file=world_file,
        world_id="demo",
        flavor="agentic",
        attempt_start=1,
        attempt_count=1,
        gm_runtime="codex",
        gm_model="gpt",
        gm_provider="openrouter",
        gm_reasoning_effort="low",
    )
    jobs[0].output_dir.mkdir(parents=True)
    write_durable_outcome(
        jobs[0].output_dir,
        {
            "episode_id": jobs[0].episode_id,
            "world_id": "demo",
            "flavor": "agentic",
            "attempt": 1,
            "rollout_dir": str(jobs[0].output_dir),
            "error": None,
            "audit": None,
            "accepted_count": 0,
        },
    )
    calls = {"n": 0}

    async def runner(job, args):
        calls["n"] += 1
        return {"episode_id": job.episode_id, "accepted_count": 0}

    outcomes = asyncio.run(
        run_campaign_wave(
            handle, jobs, args=argparse.Namespace(), rollout_runner=runner
        )
    )
    assert calls["n"] == 0
    assert len(outcomes) == 1
    assert outcomes[0]["episode_id"] == jobs[0].episode_id
