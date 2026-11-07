"""Missing samples are local failures through the real runner and campaign.

Only the external Pi/Tinker SDK and Codex executable are fixtures; smoke,
Node, HTTP preflight, durable outcomes and provider health run unchanged.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
import shutil
import sys

import pytest

from shellsmith.campaign.provider_outage import ProviderHealthTracker
from shellsmith.campaign.stream_campaign import (
    OpenCampaignSpec, build_task_attempt_jobs, open_campaign, run_campaign_wave,
)
from shellsmith.paths import DEFAULT_WORLD
from shellsmith.train.capture_store import CaptureStore
from shellsmith.train.config import load_train_config
from shellsmith.train.durable_capture import PolicyProvenance, load_durable_outcome
from shellsmith.train.identity import group_identity
from shellsmith.train.runner import make_tinker_rollout_runner
from shellsmith.train.trainer import _episode_args
from test_player_credentials import (
    CODEX_FIXTURE, PRESERVED_ENV, _endpoint, _pi_sdk, external_tinker_sdk,
)


@pytest.mark.parametrize("prior_failures", [0, 1])
def test_sampleless_player_failure_stays_neutral_in_live_and_replayed_waves(
    tmp_path, monkeypatch, external_tinker_sdk, prior_failures,
):
    if shutil.which("node") is None:
        pytest.skip("Node is required for the actual runner boundary")
    sdk, audit_path = _pi_sdk(tmp_path)
    source = sdk.read_text()
    begin = source.index("      const response = await fetch(modelRuntime.config.baseUrl + '/chat/completions'")
    end = source.index("      this.messages.push(message);", begin)
    # External SDK returns a message without requesting a model sample. The
    # real endpoint therefore has zero records; no Shellsmith result is mocked.
    source = source[:begin] + "      const message = { role: 'assistant', content: [{ type: 'text', text: 'No model requested.' }] };\n" + source[end:]
    sdk.write_text(source)
    tools = tmp_path / "tools"
    tools.mkdir()
    codex = tools / "codex"
    codex.write_text(f"#!{sys.executable}\n" + CODEX_FIXTURE)
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv(PRESERVED_ENV, "unrelated-setting")
    config = load_train_config(None)
    args = _episode_args(config)
    args.timeout_seconds = 10
    args.player_prompt_timeout_seconds = 20
    root = tmp_path / "campaign"
    handle = open_campaign(root, OpenCampaignSpec(
        campaign_id="missing-samples", mode="tinker_train", parallelism=2,
        parallelism_per_provider=2,
    ))
    handle.health = ProviderHealthTracker(consecutive_failure_threshold=2, cooldown_seconds=0)
    for _ in range(prior_failures):
        handle.health.observe("codex", success=False, error="prior genuine GM failure")
    before = handle.health.snapshot()
    jobs = build_task_attempt_jobs(
        campaign_id=handle.campaign_id, campaign_root=root, world_file=DEFAULT_WORLD,
        world_id=DEFAULT_WORLD.stem, flavor="agentic", attempt_start=1, attempt_count=4,
        gm_runtime="codex", gm_model=config.gm.model, gm_provider="codex-cli",
        gm_reasoning_effort="low",
    )
    identity = group_identity(
        DEFAULT_WORLD, gm_runtime="codex", gm_model=config.gm.model,
        gm_provider="codex-cli", gm_reasoning_effort="low",
        max_world_steps=config.gm.max_world_steps, player_model=config.player.base_model,
        renderer_name=config.player.thinking_renderer, max_tokens=config.player.max_tokens,
        temperature=config.player.temperature, top_p=config.player.top_p,
    )
    store = CaptureStore()
    with _endpoint(tmp_path) as endpoint:
        runner = make_tinker_rollout_runner(
            endpoint=endpoint, capture_store=store, run_dir=root,
            identity_for=lambda job: identity,
            policy_stamp=lambda episode_id, group_key: PolicyProvenance(
                config.player.base_model, config.player.lora_rank, "base", None, 0,
                group_key, episode_id,
            ),
            player_renderer=config.player.thinking_renderer,
            player_context_window=224 * 1024, pi_sdk_module_path=sdk,
        )

        async def waves():
            rows = []
            for batch in (jobs[:2], jobs[2:]):
                rows.extend(await run_campaign_wave(handle, batch, args=args, rollout_runner=runner))
            return rows

        outcomes = asyncio.run(waves())
        assert len(outcomes) == 4
        for row, job in zip(outcomes, jobs):
            assert "no Player samples" in row["error"]
            assert row["provider_attributable"] is False
            assert row["provider_infrastructure_failure"] is False
            assert row["failure_component"] == "player_capture"
            assert row["failure_stage"] == "tinker_record_collection"
            assert row["provider_health_event"] == "ignored_non_provider_failure"
            assert row["live_tinker_record_count"] == row["accepted_count"] == 0
            assert endpoint.take_records(job.episode_id) == []
            assert store.get(job.episode_id) is None
            durable = load_durable_outcome(job.output_dir)
            assert durable["provider_attributable"] is False
            assert durable["failure_component"] == "player_capture"
        assert handle.health.state("codex").consecutive_failures == prior_failures
        assert handle.health.state("codex").trip_count == 0
        assert not handle.health.is_down("codex") and not handle.provider_health_events
        if prior_failures:
            assert handle.health.snapshot() == before
        # Replay real durable files into a new campaign health instance; the
        # runner must not execute again and the same failure stays neutral.
        audit_before = audit_path.read_bytes()
        replay = replace(handle, outcomes=[], health=ProviderHealthTracker(
            consecutive_failure_threshold=2, cooldown_seconds=0,
        ))
        replayed = asyncio.run(run_campaign_wave(replay, jobs, args=args, rollout_runner=runner))
        assert all(row["provider_health_event"] == "ignored_non_provider_failure" for row in replayed)
        assert replay.health.state("codex").trip_count == 0
        assert audit_path.read_bytes() == audit_before
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    assert len(rows) == 8 and all(row["status"] == 200 for row in rows)
    assert "sample" not in external_tinker_sdk.calls
