"""Immutable inputs through the shipped trainer/campaign, HTTP, Node and credit.

Only external Tinker/Cookbook/torch, Codex CLI and Pi SDK boundaries are fixtures.
World selection, snapshotting, collection, launch, identity and harvesting run
unmodified. Source mutation happens at real external SDK boundaries, not by
mocking any production snapshot/runner/collector function.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import threading

import pytest

from shellsmith.campaign import inference
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import trainer
from shellsmith.world_file import WorldFileSpec
from test_trainer_resume import _no_sdk, _runtime, external_tinker_sdk, training_sdk
from test_trainer_world_ids import _hero, _world


def _setup(tmp_path, monkeypatch, *, worlds=2, groups=2, top_up=False):
    config, audit = _runtime(tmp_path, monkeypatch)
    paths = [_world(tmp_path / "inputs" / f"frozen_{index}.md", f"ORIGINAL_{index}") for index in range(worlds)]
    # Preserve exact bytes (including CRLF) separately from normalized Markdown.
    paths[0].write_bytes(paths[0].read_bytes().replace(b"\n", b"\r\n"))
    originals = {path: path.read_bytes() for path in paths}
    codex = tmp_path / "external-tools/codex"
    gm_audit = tmp_path / "external-gm-world-inputs.jsonl"
    source = codex.read_text()
    needle = "episode = json.loads((root / 'episode.json').read_text())"
    assert needle in source
    codex.write_text(source.replace(needle, needle +
        f"\nwith pathlib.Path({str(gm_audit)!r}).open('a') as audit:\n"
        "    audit.write(json.dumps({'world_id': episode['world_id'], 'markdown': "
        "(root / 'scenario/world_file.md').read_text()}) + '\\n')"))
    slate = _hero(tmp_path, paths)
    payload = json.loads(slate.read_text())
    payload["groups_per_update"] = groups
    payload["simulators"]["sol"] = {
        "runtime": "codex", "model": "gpt-6.1-sol", "provider": "codex-cli", "reasoning_effort": "low",
    }
    payload["assignments"] = {"sol": [paths[0].stem]}
    payload["expected_simulator_counts"] = {"sol": 1, "terra": worlds - 1}
    slate.write_text(json.dumps(payload))
    config = replace(
        config, worlds=replace(config.worlds, select="hero", hero_slate=str(slate)),
        train=replace(config.train, groups_per_update=groups, steps=worlds),
        adaptive=replace(config.adaptive, parallelism=2 * groups, parallelism_per_provider=2 * groups),
        world_overrides={paths[0].stem: {"maximum_attempts": 4, "minimum_trainable": 4}} if top_up else {},
    )
    return config, paths, originals, audit


def _mutate(paths, kind):
    for path in paths:
        if kind == "edit":
            path.write_bytes(path.read_bytes().replace(b"ORIGINAL", b"CHANGED")
                             .replace(b"Score the final result.", b"task_outcomes.paper_audit CHANGED scoring."))
        elif kind == "delete":
            path.unlink()
        else:
            target = _world(path.with_name("retargeted_" + path.name), "CHANGED")
            path.unlink()
            path.symlink_to(target.name)


def _run(config, mode, deps):
    if mode == "train":
        return asyncio.run(trainer.run_train(config, deps=deps))
    return asyncio.run(inference.run_inference_campaign(config, deps=deps, sampler_path=None))


def _assert_original_run(config, mode, originals, result, *, top_up=False):
    root = config.resolved_log_dir()
    snapshot = json.loads((root / trainer.SNAPSHOT_FILE).read_text())
    expected = {path.stem: (path, content) for path, content in originals.items()}
    assert snapshot["worlds"] == list(expected)
    for row in snapshot["world_identities"]:
        path, content = expected[row["world_id"]]
        assert row["world_file"] == str(path)
        assert row["world_sha256"] == hashlib.sha256(content).hexdigest()
    identities = {row["world_id"]: row for row in result["group_identities"].values()}
    assert len(identities) == len(expected)
    assert identities["frozen_0"]["gm_model"] == "gpt-6.1-sol"
    if len(expected) > 1:
        assert identities["frozen_1"]["gm_model"] == "gpt-5.6-terra"
    outcomes = list(root.rglob("durable_outcome.json"))
    assert len(outcomes) == 2 * len(expected) + (2 if top_up else 0)
    for outcome_path in outcomes:
        outcome = json.loads(outcome_path.read_text())["outcome"]
        assert outcome["accepted_count"] == 1 and outcome["error"] is None
        path, content = expected[outcome["world_id"]]
        identity = identities[outcome["world_id"]]
        assert identity["world_sha256"] == hashlib.sha256(content).hexdigest()
        assert outcome["group_identity_key"] == identity["key"]
        rollout = outcome_path.parent.parent
        spec = WorldFileSpec.from_markdown(task_id=path.stem, markdown=content.decode().replace("\r\n", "\n"))
        assert (rollout / "scenario/world_file.md").read_text() == spec.markdown
        episode = json.loads((rollout / "episode.json").read_text())
        assert episode["world_hash"] == spec.content_hash() and episode["world_id"] == path.stem
        run_config = json.loads((rollout / "run_config.json").read_text())
        # run_config honestly names the actual immutable launch input, not a
        # source pathname which now refers to different bytes (or is absent).
        launch_input = Path(run_config["world_file"])
        assert launch_input != path and launch_input.read_bytes() == content
        provenance = json.loads((launch_input.parent / "provenance.json").read_text())
        assert provenance["source_world_file"] == str(path)
        assert provenance["world_sha256"] == identity["world_sha256"]
    # Both bootstrap and judging really ran in external processes and read the
    # original world, not merely an identity row manufactured after collection.
    audit = [json.loads(line) for line in (root.parent / "external-gm-world-inputs.jsonl").read_text().splitlines()]
    assert len(audit) == 2 * len(outcomes)
    for row in audit:
        path, content = expected[row["world_id"]]
        assert row["markdown"] == WorldFileSpec.from_markdown(
            task_id=path.stem, markdown=content.decode().replace("\r\n", "\n")).markdown
    reports = [json.loads(path.read_text()) for path in root.rglob("campaign_report.json")]
    assert reports and all(row["flavor"] == "agentic" for report in reports for row in report["rollouts"])
    assert result["status"] == "complete"
    if mode == "train":
        assert result["step"] >= 1 and trainer.latest_full_checkpoint(result)["step"] == result["step"]
    else:
        assert result["worlds_completed"] == len(expected)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("mutation", ["edit", "delete", "retarget"])
def test_source_changes_after_snapshot_cannot_change_any_launch(
    tmp_path, monkeypatch, training_sdk, mode, mutation,
):
    config, paths, originals, _ = _setup(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient
    mutated = []

    class ExternalService(base):
        def create_sampling_client(self, **kwargs):
            client = super().create_sampling_client(**kwargs)
            if not mutated:
                assert (config.resolved_log_dir() / trainer.SNAPSHOT_FILE).is_file()
                _mutate(paths, mutation)
                mutated.append(True)
            return client

    monkeypatch.setattr(deps.tinker, "ServiceClient", ExternalService)
    result = _run(config, mode, deps)
    assert mutated
    _assert_original_run(config, mode, originals, result)
    if mode == "train":
        before = (config.resolved_log_dir() / trainer.STATE_FILE).read_bytes()
        with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
            _run(config, mode, _no_sdk())
        assert (config.resolved_log_dir() / trainer.STATE_FILE).read_bytes() == before


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("phase", ["concurrent-top-ups", "between-updates"])
def test_live_source_edits_do_not_change_later_top_ups_or_worlds(
    tmp_path, monkeypatch, training_sdk, mode, phase,
):
    top_up = phase == "concurrent-top-ups"
    config, paths, originals, _ = _setup(tmp_path, monkeypatch, worlds=2 if top_up else 4, top_up=top_up)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient
    mutations = []
    samples = []
    mutation_lock = threading.Lock()

    class ExternalService(base):
        async def create_lora_training_client_async(self, **kwargs):
            client = await super().create_lora_training_client_async(**kwargs)
            optimizer = client.optim_step_async

            async def optim_step_async(*args, **kw):
                result = await optimizer(*args, **kw)
                if not top_up and not mutations:
                    _mutate(paths, "edit")
                    mutations.append(True)
                return result

            client.optim_step_async = optim_step_async
            return client

        def create_sampling_client(self, **kwargs):
            client = super().create_sampling_client(**kwargs)
            sample = client.sample

            def external_sample(**sample_kwargs):
                with mutation_lock:
                    samples.append(True)
                    if not mutations and (top_up or (mode == "inference" and len(samples) >= 8)):
                        _mutate(paths, "edit")
                        mutations.append(True)
                return sample(**sample_kwargs)

            client.sample = external_sample
            return client

    monkeypatch.setattr(deps.tinker, "ServiceClient", ExternalService)
    result = _run(config, mode, deps)
    assert mutations
    _assert_original_run(config, mode, originals, result, top_up=top_up)


def test_auth_failure_keeps_snapshot_only_retry_without_staged_inputs(tmp_path, monkeypatch, training_sdk):
    config, paths, originals, _ = _setup(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient

    class FailedStartup(base):
        async def create_lora_training_client_async(self, **kwargs):
            assert (config.resolved_log_dir() / trainer.SNAPSHOT_FILE).is_file()
            raise RuntimeError("external creation failure before staging")

    monkeypatch.setattr(deps.tinker, "ServiceClient", FailedStartup)
    with pytest.raises(RuntimeError, match="external creation failure"):
        _run(config, "train", deps)
    root = config.resolved_log_dir()
    assert {path.name for path in root.iterdir()} == {trainer.SNAPSHOT_FILE}
    snapshot = (root / trainer.SNAPSHOT_FILE).read_bytes()
    monkeypatch.setattr(deps.tinker, "ServiceClient", base)
    result = _run(config, "train", deps)
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == snapshot
    _assert_original_run(config, "train", originals, result)
