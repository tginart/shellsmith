"""Offline campaign integration: shipped collectors, HTTP, Node and harvesting.

Only external Tinker/Cookbook/torch, Pi SDK and Codex CLI interfaces are fixtures.
No campaign/collector/runner, policy decision, snapshot or report is mocked.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from shellsmith.campaign import inference
from shellsmith.config import PLAYER_ENDPOINT_KEY_ENV
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train.world_selection import select_world_stream
from test_player_credentials import _assert_endpoint_closed, _assert_environment, _assert_not_saved, _observe_processes
from test_trainer_resume import _assert_private_runtime, _runtime, external_tinker_sdk, training_sdk
from test_trainer_world_ids import _hero, _world


def _read(path):
    return json.loads(path.read_text())


def _worlds(config):
    return select_world_stream(config, campaign_mode="inference").worlds


def _run(config, *, sampler_path=None):
    return asyncio.run(inference.run_inference_campaign(config, deps=load_tinker_deps(), sampler_path=sampler_path))


def _codex(tmp_path):
    return tmp_path / "external-tools/codex"


def _edit_codex(tmp_path, before_response):
    """Change only the external GM fixture, before it writes genuine artifacts."""
    path = _codex(tmp_path)
    source = path.read_text()
    assert "response.write_text(json.dumps(payload))" in source
    path.write_text(source.replace("response.write_text(json.dumps(payload))", before_response + "\nresponse.write_text(json.dumps(payload))"))


def _assert_no_training(training_sdk):
    assert not any(row[0] in {"lora", "restore", "forward_backward", "optim_step", "sampler", "state"}
                   for row in training_sdk.calls)
    assert training_sdk.updates == 0 and not training_sdk.checkpoints


def _assert_captures(config, summary, *, policy_ref=None):
    root = config.resolved_log_dir() / "campaign"
    index = _read(root / "capture_index.json")["episodes"]
    identities = {item["world_id"]: item for item in summary["group_identities"].values()}
    snapshot = _read(config.resolved_log_dir() / inference.SNAPSHOT_FILE)
    selected = {item["world_id"]: item for item in snapshot["world_identities"]}
    for world_id, identity in identities.items():
        # The experiment's model is stable; checkpoint weights belong only in
        # policy provenance, never in a contradictory second group identity.
        assert identity["player_model"] == config.player.base_model
        assert identity == {key: value for key, value in selected[world_id].items() if key != "world_file"}
    accepted = 0
    for row in summary["worlds"]:
        for attempt in range(1, row["attempts"] + 1):
            rollout = root / row["world_id"] / f"rollout_{attempt:02d}"
            outcome = _read(rollout / "logs/durable_outcome.json")["outcome"]
            assert outcome["world_id"] == row["world_id"]
            assert outcome["audit"]["final_task"]["reward"] == row["rewards"][attempt - 1]
            if outcome["accepted_count"] != 1:
                continue
            accepted += 1
            capture = _read(rollout / "logs/tinker_trainable_capture.json")
            provenance = _read(rollout / "policy_provenance.json")
            episode_id = f"{config.run.run_id}-{row['world_id']}-{attempt:02d}"
            assert capture["episode_id"] == outcome["episode_id"] == episode_id
            assert capture["group_identity_key"] == outcome["group_identity_key"] == identities[row["world_id"]]["key"]
            assert provenance["group_identity_key"] == capture["group_identity_key"]
            assert provenance["policy_ref"] == policy_ref and provenance["rft_updates_completed"] == 0
            assert index[episode_id]["accepted"] is True
            records = [json.loads(line) for line in (root / "player_records" / f"{episode_id}.jsonl").read_text().splitlines()]
            assert len(records) == len(capture["records"]) == 1
            assert records[0]["request_id"] == capture["records"][0]["request_id"]
    assert accepted == summary["totals"]["accepted_episodes"]


@pytest.mark.parametrize("pinned", [False, True], ids=["base", "sampler-checkpoint"])
def test_campaign_real_private_runtime_and_policy_provenance(tmp_path, monkeypatch, training_sdk, pinned):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    environment = dict(os.environ)
    training_sdk.observe = lambda: _assert_environment(environment)
    calls = _observe_processes(monkeypatch, environment)
    sampler_path = "tinker://fixture/existing-sampler" if pinned else None
    if pinned:
        training_sdk.sampler_steps[sampler_path] = 0
    summary = _run(config, sampler_path=sampler_path)
    assert summary["status"] == "complete" and summary["worlds_selected"] == 1
    assert summary["worlds_started"] == summary["worlds_completed"] == 1
    assert summary["worlds_not_started"] == summary["worlds_incomplete"] == 0
    assert summary["worlds"][0]["rewards"] == [0.2, 0.8]
    assert summary["worlds"][0]["initial_attempts"] == 2
    assert summary == _read(config.resolved_log_dir() / inference.SUMMARY_FILE)
    assert pi_audit.is_file() and training_sdk.budgets and max(training_sdk.budgets) <= config.player.max_tokens
    snapshot = _read(config.resolved_log_dir() / inference.SNAPSHOT_FILE)
    assert snapshot["sampler_path"] == sampler_path
    assert snapshot["worlds"] == [world.world_id for world in _worlds(config)]
    assert snapshot["pi_sdk"]["module_sha256"] == hashlib.sha256(Path(config.player.pi_sdk_module_path).read_bytes()).hexdigest()
    _assert_captures(config, summary, policy_ref=sampler_path)
    _assert_no_training(training_sdk)
    _assert_private_runtime(tmp_path, environment, calls)


@pytest.mark.parametrize("initial,maximum,minimum,attempts,topped_up", [
    (6, 6, 4, 6, 0),
    (1, 3, 2, 2, 1),
], ids=["override-above-global-no-top-up", "override-below-global-real-top-up"])
def test_top_up_totals_use_each_world_initial_override(
    tmp_path, monkeypatch, training_sdk, initial, maximum, minimum, attempts, topped_up,
):
    # Separate campaigns: a false positive above the global count must not
    # cancel out a false negative below it and accidentally pass the total.
    config, _ = _runtime(tmp_path, monkeypatch)
    world = _worlds(config)[0]
    config = replace(config, adaptive=replace(config.adaptive, initial_attempts=4, maximum_attempts=8,
                                              minimum_trainable=4, parallelism=8, parallelism_per_provider=8),
                     world_overrides={world.world_id: {"initial_attempts": initial, "maximum_attempts": maximum,
                                                       "minimum_trainable": minimum}})
    summary = _run(config)
    assert (summary["worlds"][0]["initial_attempts"], summary["worlds"][0]["attempts"]) == (initial, attempts)
    assert summary["totals"]["worlds_topped_up"] == topped_up
    assert summary["totals"]["accepted_episodes"] == attempts
    _assert_captures(config, summary)


def test_adaptive_top_ups_and_skips(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=3, tail="skip")
    middle = _worlds(config)[1].world_id
    _edit_codex(tmp_path, f"if 'resume' in args and episode['world_id'] == {middle!r}:\n"
                         "    payload['final_reward'] = 0.5\n    payload['completion_percentage'] = 50")
    config = replace(config, adaptive=replace(config.adaptive, initial_attempts=4, maximum_attempts=8,
                                              minimum_trainable=4, parallelism=4, parallelism_per_provider=4))
    summary = _run(config)
    ready, flat, never = summary["worlds"]
    assert ready["decision"] == "group_ready" and ready["attempts"] == 4
    assert flat["decision"] == "group_ready" and flat["attempts"] == 8
    assert never["decision"] == "skip_insufficient" and never["attempts"] == 8
    assert summary["status"] == "complete" and summary["totals"]["worlds_topped_up"] == 2
    assert summary["totals"]["episodes"] == 20 and summary["totals"]["accepted_episodes"] == 12
    _assert_captures(config, summary)


def test_top_ups_finish_before_later_window_starts(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=2)
    first, second = _worlds(config)
    audit = tmp_path / "gm-order.jsonl"
    _edit_codex(tmp_path, f"with pathlib.Path({str(audit)!r}).open('a') as log:\n"
                         "    log.write(json.dumps([episode['world_id'], attempt, 'resume' in args]) + '\\n')\n"
                         "if 'resume' in args:\n    payload['final_reward'] = 0.5\n    payload['completion_percentage'] = 50")
    config = replace(config, adaptive=replace(config.adaptive, maximum_attempts=3, low_spread_top_up=1))
    summary = _run(config)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert rows.index([first.world_id, 3, True]) < rows.index([second.world_id, 1, False])
    assert summary["world_window"] == 1 and summary["totals"]["worlds_topped_up"] == 2


def test_inference_allows_one_trainable_and_ignores_training_settings(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch)
    path = _hero(tmp_path, [_world(tmp_path / "inputs/one.md")])
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(path)),
                     train=replace(config.train, steps=0, groups_per_update=0, checkpoint_every=0,
                                   learning_rate=-1, cispo_clip_low=-2, cispo_clip_high=-3,
                                   kl_penalty_coef=-1, kl_discount_factor=2),
                     adaptive=replace(config.adaptive, initial_attempts=1, maximum_attempts=1,
                                      minimum_trainable=1, minimum_reward_range=0))
    summary = _run(config)
    assert summary["status"] == "complete"
    assert summary["worlds"][0]["accepted"] == summary["worlds"][0]["attempts"] == 1
    _assert_no_training(training_sdk)


def test_provider_halt_does_not_require_sparse_cursor_to_advance(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=2)
    # A real external GM timeout gets provider attribution from CodexWorldProcess.
    source = _codex(tmp_path).read_text()
    _codex(tmp_path).write_text(source.replace("sys.stdin.read()", "sys.stdin.read()\nimport time\ntime.sleep(10)"))
    config = replace(config, gm=replace(config.gm, timeout_seconds=0.25),
                     adaptive=replace(config.adaptive, outage_failure_threshold=1, outage_cooldown_seconds=0))
    summary = _run(config)
    assert summary["status"] == "halted"
    assert summary["halt"]["world_id"] == _worlds(config)[0].world_id
    assert summary["worlds"][0]["decision"] == "halt_stream"
    assert summary["worlds_started"] == 1 and summary["worlds_not_started"] == 1
    assert summary["worlds_incomplete"] == 0
    report = _read(config.resolved_log_dir() / "campaign/campaign_report.json")
    assert report["status"] == "halted"
    assert any(row["provider_attributable"] for row in report["rollouts"])
    _assert_no_training(training_sdk)


@pytest.mark.parametrize("completed_index", [0, 1], ids=["completed-before-gap", "completed-after-gap"])
def test_interrupted_window_keeps_real_completed_sibling_and_counts_incomplete(
    tmp_path, monkeypatch, training_sdk, completed_index,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=3)
    selected = _worlds(config)
    completed, blocked, not_started = selected[completed_index], selected[1 - completed_index], selected[2]
    config = replace(config, adaptive=replace(config.adaptive, parallelism=4, parallelism_per_provider=4))
    marker = tmp_path / "blocked-gm-started"
    source = _codex(tmp_path).read_text()
    needle = "attempt = int(episode['episode_id'].rsplit('-', 1)[1])"
    source = source.replace(needle, needle + f"\nif episode['world_id'] == {blocked.world_id!r}:\n"
                            f"    pathlib.Path({str(marker)!r}).touch()\n    import time\n    time.sleep(60)")
    _codex(tmp_path).write_text(source)
    environment = dict(os.environ)
    calls = _observe_processes(monkeypatch, environment)
    root = config.resolved_log_dir()

    async def interrupt():
        task = asyncio.create_task(inference.run_inference_campaign(config, deps=load_tinker_deps(), sampler_path=None))
        try:
            async with asyncio.timeout(20):
                while True:
                    if task.done():
                        await task
                        pytest.fail("Campaign finished before sibling could be interrupted")
                    report = root / "campaign/campaign_report.json"
                    if marker.exists() and report.exists():
                        rows = _read(report)["rollouts"]
                        finished = [row for row in rows if row["world_id"] == completed.world_id]
                        if len(finished) == 2 and all(row["accepted_count"] == 1 for row in finished):
                            break
                    await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Cancellation must drain the real collector and its processes, not
            # rely on asyncio.run's global shutdown to clean abandoned siblings.
            assert not [pending for pending in asyncio.all_tasks() if pending is not asyncio.current_task()]
            assert calls and all(call.process.returncode is not None for call in calls)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(interrupt())
    summary = _read(root / inference.SUMMARY_FILE)
    assert summary["status"] == "interrupted"
    assert [row["world_id"] for row in summary["worlds"]] == [completed.world_id]
    assert summary["worlds"][0]["decision"] == "group_ready"
    assert summary["worlds"][0]["rewards"] == [0.2, 0.8]
    assert summary["worlds_started"] == 2 and summary["worlds_completed"] == 1
    assert summary["worlds_incomplete"] == summary["worlds_not_started"] == 1
    assert summary["incomplete_world_ids"] == [blocked.world_id]
    assert not (root / "campaign" / not_started.world_id).exists()
    _assert_captures(config, summary)
    _assert_environment(environment)
    for log in root.rglob("vat.jsonl"):
        for line in log.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("type") == "initialize":
                _assert_endpoint_closed(message["model"]["base_url"])
    for call in calls:
        if "runner.js" in " ".join(call.command):
            key = call.environment[PLAYER_ENDPOINT_KEY_ENV]
            assert key not in os.environ.values()
            _assert_not_saved(tmp_path, key)
    _assert_no_training(training_sdk)


def test_endpoint_closes_even_when_final_summary_cannot_be_written(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch)
    root = config.resolved_log_dir()
    _edit_codex(tmp_path, f"if 'resume' in args:\n    pathlib.Path({str(root / inference.SUMMARY_FILE)!r}).mkdir(exist_ok=True)")
    environment = dict(os.environ)
    calls = _observe_processes(monkeypatch, environment)
    with pytest.raises(OSError):
        _run(config)
    _assert_private_runtime(tmp_path, environment, calls)


@pytest.mark.parametrize("changed", ["sdk-symlink", "codex-symlink", "relative-cwd"])
def test_campaign_binds_snapshotted_canonical_sdk_and_codex(tmp_path, monkeypatch, training_sdk, changed):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    original_sdk = Path(config.player.pi_sdk_module_path).resolve()
    original_codex = tmp_path / "installed/codex"
    original_codex.parent.mkdir()
    original_codex.write_bytes(_codex(tmp_path).read_bytes())
    original_codex.chmod(0o755)
    _codex(tmp_path).unlink()
    _codex(tmp_path).symlink_to(original_codex)
    sdk_link = tmp_path / "selected-sdk.js"
    sdk_link.symlink_to(original_sdk)
    other = tmp_path / "other"
    other.mkdir()
    wrong_sdk = other / "wrong-sdk.js"
    wrong_sdk.write_text("throw new Error('SDK reselected after snapshot');\n")
    wrong_codex = other / "codex"
    wrong_codex.write_text("#!/bin/sh\nexit 98\n")
    wrong_codex.chmod(0o755)
    config = replace(config, player=replace(config.player, pi_sdk_module_path=str(sdk_link)))
    if changed == "relative-cwd":
        monkeypatch.chdir(tmp_path)
        config = replace(config, player=replace(config.player, pi_sdk_module_path=str(original_sdk.relative_to(tmp_path))))
        monkeypatch.setenv("PATH", "external-tools" + os.pathsep + os.environ["PATH"])
    changed_once = []

    def change_resolution():
        if changed_once:
            return
        changed_once.append(True)
        if changed == "sdk-symlink":
            sdk_link.unlink()
            sdk_link.symlink_to(wrong_sdk)
        elif changed == "codex-symlink":
            _codex(tmp_path).unlink()
            _codex(tmp_path).symlink_to(wrong_codex)
        else:
            monkeypatch.chdir(other)

    training_sdk.observe = change_resolution
    summary = _run(config)
    assert summary["status"] == "complete" and summary["totals"]["accepted_episodes"] == 2
    assert pi_audit.is_file() and changed_once
    snapshot = _read(config.resolved_log_dir() / inference.SNAPSHOT_FILE)
    assert snapshot["pi_sdk"]["module_path"] == str(original_sdk)
    assert snapshot["codex"]["executable_path"] == str(original_codex.resolve())
    initialized = []
    for log in config.resolved_log_dir().rglob("vat.jsonl"):
        for line in log.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("type") == "initialize":
                initialized.append(message["pi_sdk_module_path"])
    assert initialized and set(initialized) == {str(original_sdk)}


def test_campaign_immutable_hero_assignments_and_canonical_ids(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch)
    targets = [_world(tmp_path / side / f"{world_id}.md", side)
               for side, world_id in (("left", "Canonical-A.01"), ("right", "canonical_B-02"))]
    aliases = []
    for target in targets:
        alias = target.with_name("world.md")
        alias.symlink_to(target.name)
        aliases.append(alias)
    slate = _hero(tmp_path, aliases)
    payload = _read(slate)
    payload["simulators"]["other"] = {"runtime": "codex", "model": "other-codex-model", "provider": "codex-cli",
                                        "reasoning_effort": "low"}
    payload["assignments"] = {"other": [targets[1].stem]}
    payload["expected_simulator_counts"] = {"terra": 1, "other": 1}
    slate.write_text(json.dumps(payload))
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(slate), count=1),
                     adaptive=replace(config.adaptive, parallelism=4, parallelism_per_provider=4))
    selected = _worlds(config)
    # Mutating the source slate at the external auth boundary must not reselect
    # worlds or replace their snapshotted simulator assignment.
    def change_slate():
        payload["assignments"] = {}
        slate.write_text(json.dumps(payload))
    training_sdk.observe = change_slate
    summary = _run(config)
    assert summary["status"] == "complete" and summary["worlds_selected"] == 2
    assert [row["world_id"] for row in summary["worlds"]] == [path.stem for path in targets]
    assert [row["gm_model"] for row in summary["worlds"]] == [world.gm_model for world in selected]
    snapshot = _read(config.resolved_log_dir() / inference.SNAPSHOT_FILE)
    assert [item["world_file"] for item in snapshot["world_identities"]] == [str(path) for path in targets]
    assert [item["gm_model"] for item in snapshot["world_identities"]] == [world.gm_model for world in selected]
    _assert_captures(config, summary)
    assert len(list((config.resolved_log_dir() / "campaign/player_records").glob("*.jsonl"))) == 4


@pytest.mark.parametrize("missing", ["pi-sdk", "codex"])
def test_campaign_missing_harness_rejects_before_auth_or_output(tmp_path, monkeypatch, training_sdk, missing):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    if missing == "pi-sdk":
        config = replace(config, player=replace(config.player, pi_sdk_module_path=str(tmp_path / "missing.js")))
        message = "Pi SDK module must exist"
    else:
        tools = tmp_path / "git-only"
        tools.mkdir()
        (tools / "git").symlink_to(shutil.which("git"))
        monkeypatch.setenv("PATH", str(tools))
        message = "Codex executable not found"
    with pytest.raises(RuntimeError, match=message):
        _run(config)
    assert not training_sdk.calls and not training_sdk.external.calls
    assert not config.resolved_log_dir().exists() and not pi_audit.exists()


def test_campaign_token_budget_preflight_precedes_service_auth(tmp_path, monkeypatch, training_sdk):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    config = replace(config, player=replace(config.player, max_tokens=1))
    with pytest.raises(RuntimeError, match="Thinking completion budget|thinking budget"):
        _run(config)
    assert not training_sdk.calls and not training_sdk.external.calls
    assert not pi_audit.exists() and not config.resolved_log_dir().exists()


def test_campaign_long_context_precedes_sampling_client(tmp_path, monkeypatch, training_sdk):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    service = deps.tinker.ServiceClient

    class ShortContextService(service):
        async def get_server_capabilities_async(self):
            return SimpleNamespace(supported_models=[SimpleNamespace(model_name=config.player.base_model,
                                                                     max_context_length=128)])

    monkeypatch.setattr(deps.tinker, "ServiceClient", ShortContextService)
    with pytest.raises(RuntimeError, match="Shellsmith needs"):
        _run(config)
    assert not any(call[0] == "sampling_client" for call in training_sdk.calls)
    assert not training_sdk.budgets and not pi_audit.exists()
    assert not (config.resolved_log_dir() / "campaign").exists()
