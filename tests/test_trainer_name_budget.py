"""POSIX output-name budgets through shipped selection, resume and preflight.

Only external SDK/tokenizer and Codex/Pi entry points are fixtures. Episode IDs
come from the real campaign job builder; accepted boundary names are written by
the real PlayerEndpoint, not a second implementation of its persistence logic.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.campaign.stream_campaign import build_task_attempt_jobs
from shellsmith.episode.smoke import _create_staging_root
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.policy.records import TinkerRolloutRecord
from shellsmith.policy.server import PlayerEndpoint
from shellsmith.process_io import write_json
from shellsmith.train import config as config_module
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config, validate_train_config
from shellsmith.train.stream_policy import ABSOLUTE_MAXIMUM_ATTEMPTS
from shellsmith.train.world_selection import select_world_stream
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk
from test_trainer_world_ids import _hero, _world


class _NoSDK:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f"Name preflight reached external SDK attribute {name}")


class _AtServiceBoundary(Exception):
    pass


def _accepted_sdk():
    calls = []
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [42])

    def get_tokenizer(model):
        calls.append("tokenizer")
        return tokenizer

    def get_renderer(name, actual_tokenizer):
        assert actual_tokenizer is tokenizer
        calls.append("renderer")
        return SimpleNamespace(tokenizer=tokenizer)

    def service():
        calls.append("service")
        raise _AtServiceBoundary

    return SimpleNamespace(
        get_tokenizer=get_tokenizer, renderers=SimpleNamespace(get_renderer=get_renderer),
        tinker=SimpleNamespace(ServiceClient=service),
    ), calls


@pytest.fixture
def local_config(tmp_path, monkeypatch):
    tools = tmp_path / "external-tools"
    tools.mkdir()
    codex = tools / "codex"
    codex.write_text("#!/bin/sh\nexit 97\n")  # fingerprint only; never executed
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    sdk = tools / "pi-sdk.js"
    sdk.write_text("export {};\n")
    config = load_train_config(None, log_dir=tmp_path / "r")
    return replace(config, worlds=replace(config.worlds, count=1),
                   player=replace(config.player, pi_sdk_module_path=str(sdk)))


def _limit(directory):
    # Same documented 255-byte portability allowance, checked against this host.
    return min(255, os.pathconf(directory, "PC_NAME_MAX"))


def _record_budget(directory):
    # Measure the shipped staging name, including tempfile's random suffix.
    staging = _create_staging_root("probe")
    try:
        overhead = len(os.fsencode(staging.name)) - len("probe")
        return min(_limit(directory), _limit(staging.parent) - overhead + len(".jsonl"))
    finally:
        staging.rmdir()


def _job(config, *, segment=0, world=None):
    if world is None:
        world = select_world_stream(config).worlds[0]
    return build_task_attempt_jobs(
        campaign_id=f"{config.run.run_id}-s{segment:03d}",
        campaign_root=config.resolved_log_dir() / "segments" / f"{segment:03d}",
        world_file=world.world_file, world_id=world.world_id, flavor="agentic",
        attempt_start=config.adaptive.maximum_attempts, attempt_count=1,
        gm_runtime=world.gm_runtime, gm_model=world.gm_model,
        gm_provider=world.gm_provider, gm_reasoning_effort=world.gm_reasoning_effort,
    )[0]


def _record_name(job):
    return f"{job.episode_id}.jsonl"


def _fit_run_id(config, limit, *, segment=0, overflow=0):
    config = replace(config, run=replace(config.run, run_id="r"), run_id_generated=False)
    widest = max(len(os.fsencode(_record_name(_job(config, segment=segment, world=world))))
                 for world in select_world_stream(config).worlds)
    return replace(config, run=replace(config.run, run_id="r" * (1 + limit - widest + overflow)))


def _assert_writable(config, *, segment=0, world=None, staging=True):
    job = _job(config, segment=segment, world=world)
    if staging:
        temporary = _create_staging_root(job.episode_id)
        try:
            assert len(os.fsencode(temporary.name)) <= _limit(temporary.parent)
        finally:
            temporary.rmdir()
    job.output_dir.mkdir(parents=True)
    endpoint = PlayerEndpoint(sampler=None, records_dir=job.output_dir.parent.parent / "player_records")
    record = TinkerRolloutRecord(
        request_id="boundary", created_at=0, prompt=None, prompt_tokens=0,
        sampled_tokens=[], sampled_logprobs=[], content="", finish_reason="stop",
        payload={}, protocol_row={"boundary": True}, session_id=job.episode_id,
    )
    endpoint._store(job.episode_id, record)
    paths = list(endpoint.records_dir.iterdir())
    assert len(paths) == 1 and paths[0].name == _record_name(job)
    assert json.loads(paths[0].read_text()) == {"boundary": True}
    return paths[0]


def _contents(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _rejected(config):
    root = config.resolved_log_dir()
    before = _contents(root)
    existed = root.exists()
    deps = _NoSDK()
    with pytest.raises(ValueError, match="(Player record filename|Episode staging directory).*byte"):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert deps.calls == []
    assert _contents(root) == before and root.exists() == existed


def _accepted(config):
    deps, calls = _accepted_sdk()
    with pytest.raises(_AtServiceBoundary):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert calls == ["tokenizer", "renderer", "service"]


@pytest.mark.parametrize("explicit_log_dir", [False, True])
def test_intrinsic_run_id_limit_is_shared_before_default_directory_or_lock_work(
    tmp_path, monkeypatch, explicit_log_dir,
):
    monkeypatch.setattr(config_module, "DEFAULT_LOG_ROOT", tmp_path / "never-created")
    config = load_train_config(None)
    run_id = "r" * 256
    log_dir = str(tmp_path / "r") if explicit_log_dir else ""
    config = replace(config, run=replace(config.run, run_id=run_id, log_dir=log_dir))
    source = tmp_path / "invalid.toml"
    source.write_text(f'[run]\nrun_id = {json.dumps(run_id)}\nlog_dir = {json.dumps(log_dir)}\n')
    deps = _NoSDK()
    for check in (lambda: validate_train_config(config), lambda: load_train_config(source),
                  lambda: asyncio.run(trainer.run_train(config, deps=deps))):
        with pytest.raises(ValueError, match="run_id.*byte"):
            check()
    assert deps.calls == [] and set(tmp_path.iterdir()) == {source}


@pytest.mark.parametrize("overflow", [0, 1])
def test_default_run_lock_component_is_checked_without_creating_parents(tmp_path, monkeypatch, overflow):
    monkeypatch.setattr(config_module, "DEFAULT_LOG_ROOT", tmp_path / "never-created")
    config = load_train_config(None)
    run_id = "r" * (1 + _limit(tmp_path) - len(".r.trainer.lock") + overflow)
    config = replace(config, run=replace(config.run, run_id=run_id))
    if overflow:
        with pytest.raises(ValueError, match="run_id.*byte"):
            validate_train_config(config)
    else:
        validate_train_config(config)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("maximum_attempts", [2, 9, 10, ABSOLUTE_MAXIMUM_ATTEMPTS])
@pytest.mark.parametrize("overflow", [0, 1])
def test_selected_record_name_boundary_and_maximum_attempt_width(
    tmp_path, local_config, maximum_attempts, overflow,
):
    config = replace(local_config, adaptive=replace(
        local_config.adaptive, initial_attempts=2, minimum_trainable=2, maximum_attempts=maximum_attempts))
    config = _fit_run_id(config, _record_budget(tmp_path), overflow=overflow)
    job = _job(config)
    assert job.attempt == maximum_attempts
    assert job.episode_id.endswith(f"-{maximum_attempts:02d}")
    assert len(os.fsencode(_record_name(job))) == _record_budget(tmp_path) + overflow
    # A regex-valid ID below the intrinsic component limit is not sufficient.
    validate_train_config(config)
    if overflow:
        _rejected(config)
    else:
        _accepted(config)
        path = _assert_writable(config)
        assert len(os.fsencode(path.name)) == _record_budget(tmp_path)


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("overflow", [0, 1])
def test_custom_hero_world_name_budget_includes_generated_run_id(
    tmp_path, local_config, generated, overflow,
):
    config = local_config if generated else replace(
        local_config, run=replace(local_config.run, run_id="r"), run_id_generated=False)
    shortest = _world(tmp_path / "inputs" / "w.md")
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(_hero(tmp_path, [shortest]))))
    length = 1 + _record_budget(tmp_path) - len(os.fsencode(_record_name(_job(config)))) + overflow
    long_world = _world(tmp_path / "inputs" / ("w" * length + ".md"))
    _hero(tmp_path, [shortest, long_world])  # hero count=1 must not hide the second world
    worlds = select_world_stream(config).worlds
    assert len(worlds) == 2 and worlds[-1].world_id == "w" * length
    assert config.run_id_generated is generated
    assert len(os.fsencode(_record_name(_job(config, world=worlds[-1])))) == _record_budget(tmp_path) + overflow
    if overflow:
        _rejected(config)
    else:
        _accepted(config)
        path = _assert_writable(config, world=worlds[-1])
        assert len(os.fsencode(path.name)) == _record_budget(tmp_path)
        assert path.parent.parent.joinpath(worlds[-1].world_id).is_dir()


def _saved_identity(config, *, segment, snapshot_only=False):
    root = config.resolved_log_dir()
    write_json(root / trainer.SNAPSHOT_FILE, trainer.snapshot(config, select_world_stream(config)))
    if not snapshot_only:
        write_json(root / trainer.STATE_FILE, {
            "run_id": config.run.run_id, "status": "halted", "step": 0,
            "rft_updates_completed": 0, "optimizer_update_in_flight": False,
            "next_world_offset": 0, "settled_world_offsets": [], "checkpoints": [],
            "segments": [{"index": index, "root": str(root / "segments" / f"{index:03d}"), "start_step": 0}
                         for index in range(segment)],
            "group_identities": {},
        })


@pytest.mark.parametrize("segment,snapshot_only", [(0, True), (1, False), (999, False), (1000, False), (10000, False)])
@pytest.mark.parametrize("overflow", [0, 1])
def test_resume_budgets_original_identity_and_actual_segment_digits(
    tmp_path, local_config, segment, snapshot_only, overflow,
):
    saved = _fit_run_id(local_config, _record_budget(tmp_path), segment=segment, overflow=overflow)
    _saved_identity(saved, segment=segment, snapshot_only=snapshot_only)
    # Reject using a shorter new invocation; accept using a longer one. Neither
    # invocation ID may substitute for the recovered campaign's original ID.
    invocation = replace(saved, run=replace(saved.run, run_id="new-id" if overflow else "x" * 240),
                         run_id_generated=True)
    before = _contents(saved.resolved_log_dir())
    if overflow:
        _rejected(invocation)
    else:
        _accepted(invocation)
        assert _contents(saved.resolved_log_dir()) == before
        path = _assert_writable(saved, segment=segment)
        assert len(os.fsencode(path.name)) == _record_budget(tmp_path)


def test_resuming_segment_1000_rejects_name_that_fit_segment_999(tmp_path, local_config):
    config = _fit_run_id(local_config, _record_budget(tmp_path), segment=999)
    assert len(os.fsencode(_record_name(_job(config, segment=999)))) == _record_budget(tmp_path)
    assert len(os.fsencode(_record_name(_job(config, segment=1000)))) == _record_budget(tmp_path) + 1
    _saved_identity(config, segment=1000)
    _rejected(config)


def test_overlong_composed_record_rejects_even_with_short_explicit_log_dir(tmp_path, local_config):
    config = _fit_run_id(local_config, _limit(tmp_path), overflow=1)
    assert len(os.fsencode(_record_name(_job(config)))) == _limit(tmp_path) + 1
    validate_train_config(config)
    deps = _NoSDK()
    with pytest.raises(ValueError, match="Player record filename.*byte"):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert not deps.calls and not config.resolved_log_dir().exists()


def test_endpoint_can_write_full_record_component_but_trainer_also_budgets_staging(tmp_path, local_config):
    config = _fit_run_id(local_config, _limit(tmp_path))
    _rejected(config)
    path = _assert_writable(config, staging=False)
    assert len(os.fsencode(path.name)) == _limit(tmp_path)


def test_ordinary_generated_run_id_reaches_service_boundary(local_config):
    assert local_config.run_id_generated
    _accepted(local_config)


@pytest.mark.parametrize("overflow", [0, 1])
def test_real_sampling_pipeline_at_record_name_boundary(tmp_path, monkeypatch, training_sdk, overflow):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    config = _fit_run_id(config, _record_budget(tmp_path), overflow=overflow)
    if overflow:
        with pytest.raises(ValueError, match="(Player record filename|Episode staging directory).*byte"):
            asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
        assert not training_sdk.calls and not training_sdk.budgets and not pi_audit.exists()
        assert not config.resolved_log_dir().exists()
    else:
        state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
        assert state["status"] == "complete" and state["step"] == 1
        records = list((config.resolved_log_dir() / "segments/000/player_records").iterdir())
        assert len(records) == 2 and training_sdk.budgets and pi_audit.exists()
        assert all(len(os.fsencode(path.name)) == _record_budget(tmp_path) for path in records)
        assert _record_name(_job(config)) in {path.name for path in records}
