"""Cancellation releases actual sampler pins and drains partial memory records."""

import argparse
import asyncio
import gc
from types import SimpleNamespace
import weakref

import pytest

from shellsmith.campaign.rollout_campaign import RolloutJob
from shellsmith.paths import DEFAULT_WORLD
from shellsmith.policy.sampler import PlayerSamplingConfig, TinkerPlayerSampler
from shellsmith.policy.server import PlayerEndpoint
from shellsmith.train.capture_store import CaptureStore
from shellsmith.train.durable_capture import PolicyProvenance
from shellsmith.train import runner as runner_module
from test_player_endpoint import FakeSampler


@pytest.mark.parametrize("setup_failure", [False, True])
def test_cancelled_or_unstarted_jobs_release_old_policy_and_partial_state(tmp_path, monkeypatch, setup_failure):
    class SamplingClient:
        pass
    old = SamplingClient()
    old_ref = weakref.ref(old)
    sampler = TinkerPlayerSampler(config=PlayerSamplingConfig(base_model="fixture"), deps=None, sampling_client=old)
    del old
    endpoint = PlayerEndpoint(sampler=sampler, records_dir=tmp_path / "records")
    endpoint.start()
    capture = CaptureStore()
    started = asyncio.Event()
    async def smoke(args, **kwargs):
        record, _ = FakeSampler().sample({}, request_id="partial", session_id=args.episode_id)
        endpoint._store(args.episode_id, record)
        started.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(runner_module, "run_live_smoke", smoke)
    def stamp(episode_id, key):
        sampler.pin(episode_id)
        sampler.set_sampling_client(SamplingClient())  # Publish while the old policy is pinned.
        return PolicyProvenance("fixture", 16, "base", None, 0, key, episode_id)
    run_job = runner_module.make_tinker_rollout_runner(
        endpoint=endpoint, capture_store=capture, run_dir=tmp_path,
        identity_for=lambda job: SimpleNamespace(key="group"), policy_stamp=stamp,
        player_renderer="fixture", player_context_window=262144, pi_sdk_module_path=None,
    )
    job = RolloutJob(world_file=DEFAULT_WORLD, world_id=DEFAULT_WORLD.stem, flavor="agentic", attempt=1,
                     episode_id="cancelled", output_dir=tmp_path / "episode", gm_runtime="codex",
                     gm_model="fixture", gm_provider="codex-cli", gm_reasoning_effort="low")
    args = argparse.Namespace(gm_api_key_env="fixture", max_world_steps=1, timeout_seconds=1)
    async def cancel():
        task = asyncio.create_task(run_job(job, args))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    try:
        if setup_failure:
            with pytest.raises(AttributeError, match="gm_api_key_env"):
                asyncio.run(run_job(job, argparse.Namespace()))
        else:
            asyncio.run(cancel())
            assert (endpoint.records_dir / "cancelled.jsonl").read_text()
        assert sampler._session_clients == {} and endpoint._records == {}
        gc.collect()
        assert old_ref() is None
        assert capture.get(job.episode_id) is None
    finally:
        endpoint.stop()
