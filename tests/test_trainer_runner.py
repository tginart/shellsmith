"""Trainer runner integrates M3's private key flow without global credentials.

Real runner/smoke/orchestrator/HTTP/vat code; only external SDK and Codex
interfaces are fixtures. These are not live-provider training results.
"""
import asyncio
import json
import os
import shutil
import sys

import pytest

from shellsmith.campaign.rollout_campaign import RolloutJob
from shellsmith.config import PLAYER_ENDPOINT_KEY_ENV
from shellsmith.paths import DEFAULT_WORLD
from shellsmith.train.capture_store import CaptureStore
from shellsmith.train.config import load_train_config
from shellsmith.train.durable_capture import PolicyProvenance, load_durable_outcome
from shellsmith.train.identity import group_identity
from shellsmith.train.runner import make_tinker_rollout_runner
from shellsmith.train.trainer import _episode_args

from test_player_credentials import (
    CODEX_FIXTURE, RUNNER, _assert_environment, _assert_not_saved,
    _assert_private_spawn, _endpoint, _observe_processes, _pi_sdk,
    ambient_key, external_tinker_sdk,
)


@pytest.mark.parametrize("failure", [None, "initialize", "prompt"])
def test_trainer_runner_privately_authenticates_and_persists_real_outcomes(
    tmp_path, monkeypatch, ambient_key, external_tinker_sdk, failure,
):
    if shutil.which("node") is None:
        pytest.skip("Node is required for the real trainer rollout boundary")
    sdk, audit = _pi_sdk(tmp_path, failure=failure)
    tools = tmp_path / "external-tools"
    tools.mkdir()
    codex = tools / "codex"
    codex.write_text(f"#!{sys.executable}\n" + CODEX_FIXTURE)
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    parent = dict(os.environ)
    calls = _observe_processes(monkeypatch, parent)
    config = load_train_config(None)
    args = _episode_args(config)
    args.timeout_seconds = 10
    args.player_prompt_timeout_seconds = 20
    store = CaptureStore()
    root = tmp_path / "segment"
    job = RolloutJob(
        world_file=DEFAULT_WORLD, world_id=DEFAULT_WORLD.stem, flavor="agentic", attempt=0,
        episode_id="private-training-episode", output_dir=root / "episode",
        gm_runtime="codex", gm_model=config.gm.model, gm_provider="codex-cli",
        gm_reasoning_effort="low",
    )
    identity = group_identity(
        DEFAULT_WORLD, gm_runtime=job.gm_runtime, gm_model=job.gm_model,
        gm_provider=job.gm_provider, gm_reasoning_effort=job.gm_reasoning_effort,
        max_world_steps=config.gm.max_world_steps, player_model=config.player.base_model,
        renderer_name=config.player.thinking_renderer, max_tokens=config.player.max_tokens,
        temperature=config.player.temperature, top_p=config.player.top_p,
    )
    with _endpoint(tmp_path) as endpoint:
        runner = make_tinker_rollout_runner(
            endpoint=endpoint, capture_store=store, run_dir=root,
            identity_for=lambda actual_job: identity,
            policy_stamp=lambda episode_id, group_key: PolicyProvenance(
                config.player.base_model, config.player.lora_rank, "base", None, 0,
                group_key, episode_id,
            ),
            player_renderer=config.player.thinking_renderer, player_context_window=224 * 1024,
            pi_sdk_module_path=sdk,
        )
        outcome = asyncio.run(runner(job, args))
        _assert_environment(parent)
        assert endpoint.api_key not in os.environ.values()
        assert endpoint.take_records(job.episode_id) == []
    node = [call for call in calls if str(RUNNER) in call.command]
    gm = [call for call in calls if call.command[0] == str(codex)]
    assert len(node) == 1
    _assert_private_spawn(node[0], parent, endpoint.api_key)
    assert len(gm) == (1 if failure else 2)
    for call in gm:
        _assert_environment(parent, call.environment)
    assert outcome["live_tinker_record_count"] == (0 if failure else 1)
    assert outcome["accepted_count"] == 0  # fixture GM honestly marks untrainable
    assert store.get(job.episode_id) is None
    if failure:
        assert f"injected Pi SDK {failure} failure" in outcome["error"]
    else:
        assert outcome["error"] is None
        assert outcome["audit"]["accepted_for_training"] is False
    assert load_durable_outcome(job.output_dir) == outcome
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert all(row["status"] == 200 for row in rows)
    assert os.environ.get(PLAYER_ENDPOINT_KEY_ENV) == ambient_key
    _assert_not_saved(tmp_path, endpoint.api_key)
