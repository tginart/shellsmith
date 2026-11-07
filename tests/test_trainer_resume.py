"""Trainer resume/checkpoint regressions through real offline orchestration.

Only Tinker/Cookbook/torch and the external Codex/Pi executables are fixtures in
runtime tests. Collection, HTTP, Node, harvesting, acceptance, group credit,
CISPO submission and checkpoint decisions are the shipped implementations.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shellsmith.config import DEFAULT_TINKER_BASE_MODEL, PLAYER_ENDPOINT_KEY_ENV
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.process_io import write_json
from shellsmith.train import identity, trainer
from shellsmith.train.config import load_train_config
from shellsmith.train.world_selection import SelectedWorld, WorldStream
from test_player_credentials import (
    CODEX_FIXTURE,
    PRESERVED_ENV,
    _assert_endpoint_closed,
    _assert_environment,
    _assert_not_saved,
    _observe_processes,
    _pi_sdk,
    external_tinker_sdk,
)


@pytest.fixture
def training_sdk(monkeypatch, external_tinker_sdk):
    """Extend the credential test's external SDK with optimizer/checkpoint I/O."""
    deps = load_tinker_deps()
    state = SimpleNamespace(calls=[], checkpoints={}, sampler_steps={}, updates=0,
                            fail_sampler_step=None, budgets=[], observe=lambda: None)
    original_service = deps.tinker.ServiceClient
    original_renderer = deps.renderers.get_renderer

    class Prompt:
        def __init__(self, tokens):
            self.tokens = list(tokens)
            self.length = len(self.tokens)

        def to_ints(self):
            return list(self.tokens)

        def append(self, chunk):
            return Prompt(self.tokens + list(chunk.tokens))

    def renderer(*args):
        result = original_renderer(*args)
        build = result.build_generation_prompt
        result.build_generation_prompt = lambda messages: Prompt(build(messages).to_ints())
        result.tokenizer = SimpleNamespace(
            encode=lambda text, **_: list(map(ord, text)), decode=lambda tokens: "".join(map(chr, tokens)))
        result.parse_response = lambda tokens: (
            SimpleNamespace(role="assistant", content="".join(map(chr, tokens)).split("</think>")[-1]),
            SimpleNamespace(is_clean=True, value="stop_sequence"))
        return result

    def future(**values):
        return SimpleNamespace(result=lambda: SimpleNamespace(**values))

    class AsyncFuture:
        """External SDK adapter, retaining injected sync result faults in tests."""
        def __init__(self, value):
            self.value = value

        async def result_async(self):
            return await asyncio.to_thread(self.value.result)

    class TrainingClient:
        # The shipped trainer uses only the async SDK surface. Delegate inside
        # this explicit external fixture so inherited sync fault hooks remain
        # effective; production must never detect mocks or fall back to sync.
        async def forward_backward_async(self, *args, **kwargs):
            return AsyncFuture(await asyncio.to_thread(self.forward_backward, *args, **kwargs))

        async def optim_step_async(self, *args, **kwargs):
            return AsyncFuture(await asyncio.to_thread(self.optim_step, *args, **kwargs))

        async def save_weights_for_sampler_async(self, *args, **kwargs):
            return AsyncFuture(await asyncio.to_thread(self.save_weights_for_sampler, *args, **kwargs))

        async def save_state_async(self, *args, **kwargs):
            return AsyncFuture(await asyncio.to_thread(self.save_state, *args, **kwargs))

        def forward_backward(self, datums, *, loss_fn, loss_fn_config):
            state.observe()
            assert loss_fn == "cispo"
            assert loss_fn_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
            assert datums and all("mask" not in item.loss_fn_inputs for item in datums)
            advantages = [value for item in datums for value in item.loss_fn_inputs["advantages"]]
            assert min(advantages) < 0 < max(advantages), "Fixture must produce real nonzero reward spread"
            state.calls.append(("forward_backward", len(datums)))
            return future(metrics={"loss:sum": 0.5})

        def optim_step(self, adam):
            state.observe()
            assert adam.learning_rate > 0
            state.updates += 1
            state.calls.append(("optim_step", state.updates))
            return future(metrics={"grad_norm": 0.1})

        def save_weights_for_sampler(self, name):
            state.observe()
            if state.fail_sampler_step == state.updates:
                state.fail_sampler_step = None
                raise RuntimeError("external sampler-save interruption")
            path = f"tinker://fixture/sampler/{name}"
            state.sampler_steps[path] = state.updates
            state.calls.append(("sampler", name, state.updates))
            return future(path=path)

        def save_state(self, name):
            state.observe()
            path = f"tinker://fixture/state/{name}"
            state.checkpoints[path] = state.updates
            state.calls.append(("state", name, state.updates))
            return future(path=path)

    class ServiceClient(original_service):
        async def get_server_capabilities_async(self):
            return await asyncio.to_thread(self.get_server_capabilities)

        async def create_lora_training_client_async(self, **kwargs):
            return await asyncio.to_thread(self.create_lora_training_client, **kwargs)

        async def create_training_client_from_state_with_optimizer_async(self, path):
            return await asyncio.to_thread(self.create_training_client_from_state_with_optimizer, path)

        async def create_sampling_client_async(self, **kwargs):
            return await asyncio.to_thread(self.create_sampling_client, **kwargs)

        def __init__(self):
            state.observe()
            super().__init__()
            state.calls.append(("service",))

        def create_lora_training_client(self, *, base_model, rank):
            assert base_model == DEFAULT_TINKER_BASE_MODEL and rank > 0
            state.calls.append(("lora",))
            return TrainingClient()

        def create_training_client_from_state_with_optimizer(self, path):
            state.observe()
            state.updates = state.checkpoints[path]
            state.calls.append(("restore", path, state.updates))
            return TrainingClient()

        def create_sampling_client(self, *, base_model=None, model_path=None):
            state.observe()
            assert (base_model is None) != (model_path is None)
            if model_path is not None:
                assert state.sampler_steps[model_path] == state.updates
            state.calls.append(("sampling_client", model_path))
            client = super().create_sampling_client(base_model=base_model or DEFAULT_TINKER_BASE_MODEL)
            original_sample = client.sample

            def sample(**kwargs):
                state.observe()
                state.budgets.append(kwargs["sampling_params"].max_tokens)
                if "</think>" in kwargs["sampling_params"].stop:
                    tokens = list(map(ord, "fixture-thought</think>"))
                    return future(sequences=[SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens),
                                                             stop_reason="stop")])
                return original_sample(**kwargs)

            client.sample = sample
            return client

    monkeypatch.setattr(deps.tinker, "ServiceClient", ServiceClient)
    monkeypatch.setattr(deps.renderers, "get_renderer", renderer)
    for name in ("EncodedTextChunk", "Datum", "AdamParams"):
        monkeypatch.setattr(deps.types, name, SimpleNamespace, raising=False)
    monkeypatch.setattr(deps.types, "ModelInput", SimpleNamespace(from_ints=Prompt), raising=False)
    monkeypatch.setattr(deps.torch, "tensor", list, raising=False)
    monkeypatch.setattr(sys.modules["tinker.types.tensor_data"], "TensorData",
                        SimpleNamespace(from_torch=lambda value: value))
    state.external = external_tinker_sdk
    return state


def _runtime(tmp_path, monkeypatch, *, count=1, tail="success", checkpoint_every=10):
    if shutil.which("node") is None:
        pytest.skip("Node is required for the shipped trainer/Pi regression")
    sdk, pi_audit = _pi_sdk(tmp_path)
    # Trainer episodes advertise the fixed thinking renderer; the external Pi
    # fixture must request it too, rather than the credential test's plain mode.
    sdk.write_text(sdk.read_text().replace("reasoning_effort: 'none'", "reasoning_effort: 'high'"))
    monkeypatch.setenv(PRESERVED_ENV, "unrelated-setting")
    monkeypatch.setenv(PLAYER_ENDPOINT_KEY_ENV, "previous-ambient-player-token")
    config_file = tmp_path / "train.toml"
    config_file.write_text(
        f'[worlds]\ncount = {count}\n'
        f'[train]\nsteps = 5\ngroups_per_update = 1\ncheckpoint_every = {checkpoint_every}\n'
        '[adaptive]\ninitial_attempts = 2\nmaximum_attempts = 2\nminimum_trainable = 2\n'
        'parallelism = 2\nparallelism_per_provider = 2\n'
        '[gm]\ntimeout_seconds = 10.0\n'
        f'[player]\nmax_tokens = 256\nprompt_timeout_seconds = 20.0\npi_sdk_module_path = {json.dumps(str(sdk))}\n'
    )
    config = load_train_config(config_file, log_dir=tmp_path / "run")
    selected = trainer.select_worlds(config)
    last_world = selected[-1].stem
    tools = tmp_path / "external-tools"
    tools.mkdir()
    codex = tools / "codex"
    setup = (
        "root = pathlib.Path.cwd()\n"
        "episode = json.loads((root / 'episode.json').read_text())\n"
        "attempt = int(episode['episode_id'].rsplit('-', 1)[1])\n"
        f"tail_world = episode['world_id'] == {last_world!r}\n"
        f"trainable = not (tail_world and {tail!r} == 'skip')\n"
        f"reward = 0.5 if tail_world and {tail!r} == 'zero' else (0.2 if attempt % 2 else 0.8)\n"
    )
    source = CODEX_FIXTURE.replace("root = pathlib.Path.cwd()", setup)
    source = source.replace("'final_reward': 0.25, 'completion_percentage': 25,",
                            "'final_reward': reward, 'completion_percentage': 100 * reward,")
    source = source.replace("'trainable': False", "'trainable': trainable")
    codex.write_text(f"#!{sys.executable}\n" + source)
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    return config, pi_audit


def _assert_private_runtime(tmp_path, environment, calls):
    _assert_environment(environment)
    nodes = [call for call in calls if "runner.js" in " ".join(call.command)]
    assert nodes
    for call in nodes:
        key = call.environment[PLAYER_ENDPOINT_KEY_ENV]
        assert key != environment[PLAYER_ENDPOINT_KEY_ENV]
        assert key not in os.environ.values()
        assert key not in " ".join(call.command)
        assert call.process.returncode == 0 and not call.cwd.exists()
        _assert_not_saved(tmp_path, key)
    for call in calls:
        if call not in nodes:
            _assert_environment(environment, call.environment)
            assert call.process.returncode == 0
    urls = set()
    for log in (tmp_path / "run").rglob("vat.jsonl"):
        for line in log.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("type") == "initialize":
                urls.add(message["model"]["base_url"])
    assert urls
    for url in urls:
        _assert_endpoint_closed(url)


@pytest.mark.parametrize("count,tail,cadence", [
    (1, "success", 10),
    (2, "skip", 10),
    (2, "zero", 10),
    (2, "skip", 1),
])
def test_real_trainer_checkpoints_exhaustion_including_later_no_datums(
    tmp_path, monkeypatch, training_sdk, count, tail, cadence,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch, count=count, tail=tail, checkpoint_every=cadence)
    environment = dict(os.environ)
    training_sdk.observe = lambda: _assert_environment(environment)
    calls = _observe_processes(monkeypatch, environment)
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert state["status"] == "complete" and state["step"] == 1
    assert state["next_world_offset"] == count
    full = trainer.latest_full_checkpoint(state)
    assert full["step"] == 1 and full["next_world_offset"] == count
    assert training_sdk.checkpoints[full["path"]] == 1
    assert training_sdk.sampler_steps[full["sampler_path"]] == 1
    assert sum(call[0] == "optim_step" for call in training_sdk.calls) == 1
    assert sum(call[0] == "state" for call in training_sdk.calls) == (2 if cadence == 1 else 1)
    assert training_sdk.budgets and all(0 < budget <= 256 for budget in training_sdk.budgets)
    assert json.loads((config.resolved_log_dir() / trainer.STATE_FILE).read_text()) == state
    metrics = [json.loads(line) for line in (config.resolved_log_dir() / trainer.METRICS_FILE).read_text().splitlines()]
    assert metrics[0]["worlds"][0]["rewards"] == [0.2, 0.8]
    assert metrics[0]["datums"] == 2
    assert metrics[-1]["state_checkpoint"] == full["path"]
    if count == 2:
        assert metrics[-1]["datums"] == 0 and "skipped" in metrics[-1]
        expected = "skip_insufficient" if tail == "skip" else "group_ready"
        assert metrics[-1]["worlds"][0]["decision"] == expected
    assert len([json.loads(row) for row in pi_audit.read_text().splitlines()]) == count * 4
    _assert_private_runtime(tmp_path, environment, calls)

    # A fresh auto-generated invocation restores full optimizer + sampler state,
    # not just the most recent sampling artifact, and keeps the original ID.
    resumed_config = replace(config, run=replace(config.run, run_id="run-later-invocation"))
    before = len(training_sdk.calls)
    resumed = asyncio.run(trainer.run_train(resumed_config, deps=load_tinker_deps()))
    assert resumed["run_id"] == state["run_id"] and resumed["resumes"] == 1
    assert ("restore", full["path"], 1) in training_sdk.calls[before:]
    assert not any(call[0] == "optim_step" for call in training_sdk.calls[before:])
    campaign = json.loads((config.resolved_log_dir() / "segments/001/campaign_manifest.json").read_text())
    assert campaign["campaign_id"] == f"{state['run_id']}-s001"
    _assert_environment(environment)


def test_real_trainer_resume_keeps_auto_id_for_new_checkpoints(tmp_path, monkeypatch, training_sdk):
    config, _ = _runtime(tmp_path, monkeypatch, count=2, checkpoint_every=1)
    training_sdk.fail_sampler_step = 2
    with pytest.raises(RuntimeError, match="external sampler-save interruption"):
        asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    failed = json.loads((config.resolved_log_dir() / trainer.STATE_FILE).read_text())
    assert failed["status"] == "failed" and failed["step"] == 2
    full = trainer.latest_full_checkpoint(failed)
    assert full["step"] == 1 and full["next_world_offset"] == 1
    resumed_config = replace(config, run=replace(config.run, run_id="run-later-invocation"))
    before = len(training_sdk.calls)
    resumed = asyncio.run(trainer.run_train(resumed_config, deps=load_tinker_deps()))
    assert resumed["run_id"] == config.run.run_id and resumed["step"] == 2
    assert ("restore", full["path"], 1) in training_sdk.calls[before:]
    new_saves = [call for call in training_sdk.calls[before:] if call[0] in {"state", "sampler"}]
    assert new_saves and all(call[1].startswith(config.run.run_id + "-step00002") for call in new_saves)
    campaign = json.loads((config.resolved_log_dir() / "segments/001/campaign_manifest.json").read_text())
    assert campaign["campaign_id"] == f"{config.run.run_id}-s001"


def _no_sdk():
    def forbidden():
        pytest.fail("Local validation/resume checks must precede ServiceClient authentication")

    return SimpleNamespace(tinker=SimpleNamespace(ServiceClient=forbidden))


def _saved_run(tmp_path):
    config = load_train_config(None, log_dir=tmp_path / "run")
    config = replace(config, worlds=replace(config.worlds, select="ids", ids=("checkout_idempotency_001",), count=1))
    current = trainer.snapshot(config, trainer.select_world_stream(config))
    full = {"kind": "state", "path": "tinker://full", "sampler_path": "tinker://sampler", "step": 1,
            "next_world_offset": 1}
    state = {"run_id": config.run.run_id, "checkpoints": [full]}
    write_json(config.resolved_log_dir() / trainer.STATE_FILE, state)
    write_json(config.resolved_log_dir() / trainer.SNAPSHOT_FILE, current)
    return config, state, current


@pytest.mark.parametrize("field", ["git_sha", "source_sha256", "manifest_sha256", "manifest_path",
                                  "world_file", "world_sha256", "gm_prompt_sha256", "world_stream"])
def test_resume_rejects_changed_provenance_before_service_auth(tmp_path, field):
    config, _, saved = _saved_run(tmp_path)
    if field in {"world_file", "world_sha256", "gm_prompt_sha256"}:
        saved["world_identities"][0][field] = "changed"
    elif field == "world_stream":
        saved[field] = {"select": "hero", "hero_slate_sha256": "changed"}
    else:
        saved[field] = "changed"
    write_json(config.resolved_log_dir() / trainer.SNAPSHOT_FILE, saved)
    before = {path: path.read_bytes() for path in config.resolved_log_dir().iterdir()}
    with pytest.raises(RuntimeError, match="Source, manifest, worlds, prompts or hero metadata changed"):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))
    assert before == {path: path.read_bytes() for path in config.resolved_log_dir().iterdir()}


def test_resume_rejects_explicit_id_mismatch_before_auth(tmp_path):
    config, _, _ = _saved_run(tmp_path)
    changed = replace(config, run=replace(config.run, run_id="explicit-other"), run_id_generated=False)
    with pytest.raises(RuntimeError, match="Explicit run_id.*does not match saved run_id"):
        asyncio.run(trainer.run_train(changed, deps=_no_sdk()))


def test_resume_normalizes_json_and_preserves_original_id(tmp_path):
    config, state, saved = _saved_run(tmp_path)
    current_config = replace(config, run=replace(config.run, run_id="later-auto-id"))
    current = trainer.snapshot(current_config, trainer.select_world_stream(current_config))
    resumed = trainer._resume_config(config=current_config, state=state, saved=saved, current=current,
                                     log_dir=config.resolved_log_dir())
    assert resumed.run.run_id == config.run.run_id
    assert resumed.worlds.ids == ("checkout_idempotency_001",)
    assert current_config.run.run_id == "later-auto-id", "Frozen input config must not be mutated"


@pytest.mark.parametrize("section,field,value", [("train", "checkpoint_every", 0),
                                                  ("train", "steps", 0),
                                                  ("adaptive", "parallelism_per_provider", 100)])
def test_invalid_config_precedes_directory_creation_and_auth(tmp_path, section, field, value):
    config = load_train_config(None, log_dir=tmp_path / "must-not-exist")
    config = replace(config, **{section: replace(getattr(config, section), **{field: value})})
    with pytest.raises((ValueError, RuntimeError)):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))
    assert not config.resolved_log_dir().exists()


def test_unconfined_selected_world_precedes_directory_creation_and_auth(tmp_path, monkeypatch):
    config = load_train_config(None, log_dir=tmp_path / "must-not-exist")
    selected = trainer.select_world_stream(config)
    unsafe = replace(selected.worlds[0], gm_runtime="claude", gm_provider="claude-cli")
    monkeypatch.setattr(trainer, "select_world_stream", lambda _: WorldStream((unsafe,), selected.provenance))
    with pytest.raises(RuntimeError, match="confined Codex GM for every selected world"):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))
    assert not config.resolved_log_dir().exists()


def test_resume_allows_only_log_directory_change_in_run_settings(tmp_path):
    config, state, saved = _saved_run(tmp_path)
    relocated = replace(config, run=replace(config.run, log_dir=str(tmp_path / "relocated")), run_id_generated=False)
    current = trainer.snapshot(relocated, trainer.select_world_stream(relocated))
    resolved = trainer._resume_config(config=relocated, state=state, saved=saved, current=current,
                                      log_dir=relocated.resolved_log_dir())
    assert resolved.run.run_id == state["run_id"]
    assert resolved.resolved_log_dir() == tmp_path / "relocated"
    saved["config"]["run"]["unexpected_saved_setting"] = "must-not-be-ignored"
    with pytest.raises(RuntimeError, match="Config changed"):
        trainer._resume_config(config=relocated, state=state, saved=saved, current=current,
                               log_dir=relocated.resolved_log_dir())


def test_episode_args_are_independent_per_selected_assignment():
    config = load_train_config(None)
    selected = trainer.select_world_stream(config).worlds[0]
    changed = replace(selected, gm_model="other-codex-model", gm_reasoning_effort="low")
    first, second = trainer._episode_args(config, selected), trainer._episode_args(config, changed)
    assert first is not second
    assert first.gm_model == selected.gm_model and second.gm_model == "other-codex-model"
    assert second.gm_reasoning_effort == "low" and second.player_max_tokens == config.player.max_tokens


def test_full_checkpoint_helper_covers_orderly_halt_and_new_cursor():
    saved = []

    class Client:
        async def save_state_async(self, name):
            saved.append(name)

            async def result_async():
                return SimpleNamespace(path=f"tinker://{name}")

            return SimpleNamespace(result_async=result_async)

    state = {"run_id": "original", "step": 1, "next_world_offset": 1, "status": "halted", "checkpoints": []}
    client = Client()

    async def run():
        first = await trainer._ensure_full_checkpoint(training_client=client, state=state, sampler_path="sampler-1")
        assert first and trainer.latest_full_checkpoint(state)["next_world_offset"] == 1
        assert await trainer._ensure_full_checkpoint(training_client=client, state=state, sampler_path="sampler-1") is None
        state["next_world_offset"] = 2  # later skipped/halted world, no additional optimizer update
        second = await trainer._ensure_full_checkpoint(training_client=client, state=state, sampler_path="sampler-1")
        assert second != first and trainer.latest_full_checkpoint(state)["next_world_offset"] == 2

    asyncio.run(run())
    assert len(saved) == len(set(saved)) == 2


@pytest.mark.parametrize("changed", ["world-content", "world-path", "prompt", "manifest"])
def test_snapshot_hashes_real_world_paths_contents_prompts_and_manifest(tmp_path, monkeypatch, changed):
    config = load_train_config(None)
    original = trainer.select_world_stream(config).worlds[0]
    world_path = tmp_path / original.world_file.name
    world_path.write_bytes(original.world_file.read_bytes())
    world = SelectedWorld(world_path, original.gm_runtime, original.gm_model,
                          original.gm_provider, original.gm_reasoning_effort)
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(trainer.CATALOG_MANIFEST.read_bytes())
    monkeypatch.setattr(trainer, "CATALOG_MANIFEST", manifest)
    prompts = []
    for path in identity.PROMPT_PATHS:
        target = tmp_path / "prompts" / path.name
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(path.read_bytes())
        prompts.append(target)
    monkeypatch.setattr(identity, "PROMPT_PATHS", tuple(prompts))
    stream = WorldStream((world,), {"select": "manifest"})
    before = trainer.snapshot(config, stream)
    assert before["world_identities"][0]["world_sha256"] == hashlib.sha256(world_path.read_bytes()).hexdigest()
    if changed == "world-path":
        alternate = tmp_path / "moved" / world_path.name
        alternate.parent.mkdir()
        alternate.write_bytes(world_path.read_bytes())
        stream = replace(stream, worlds=(replace(world, world_file=alternate),))
    else:
        path = {"world-content": world_path, "prompt": prompts[0], "manifest": manifest}[changed]
        path.write_bytes(path.read_bytes() + b"\n")
    after = trainer.snapshot(config, stream)
    assert before["worlds"] == after["worlds"], "Stable IDs must not hide a changed experiment"
    assert before != after
    with pytest.raises(RuntimeError, match="Source, manifest, worlds, prompts or hero metadata changed"):
        trainer._resume_config(config=config, state={"run_id": config.run.run_id}, saved=before, current=after,
                               log_dir=tmp_path)


def test_source_hash_catches_dirty_and_untracked_runtime_changes(tmp_path, monkeypatch):
    # No commit or repository mutation: this isolated index is only file input
    # to the production source fingerprint helper.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src/shellsmith/trainer.py"
    source.parent.mkdir(parents=True)
    source.write_text("original = True\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "src"], check=True)
    monkeypatch.setattr(trainer, "PROJECT_ROOT", tmp_path)
    original = trainer._source_sha256()
    source.write_text("original = False\n")
    changed = trainer._source_sha256()
    assert original != changed
    (source.parent / "helper.py").write_text("new_runtime_helper = True\n")
    assert trainer._source_sha256() != changed


def test_resume_rejects_changed_hero_metadata_even_when_queue_is_identical(tmp_path):
    config = load_train_config(None, log_dir=tmp_path / "run")
    slate = tmp_path / "hero.json"
    payload = json.loads((trainer.PROJECT_ROOT / "hero_runs/first_hero_baby_terra_4.json").read_text())
    slate.write_text(json.dumps(payload))
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(slate)))
    original = trainer.select_world_stream(config)
    saved = trainer.snapshot(config, original)
    write_json(config.resolved_log_dir() / trainer.SNAPSHOT_FILE, saved)
    write_json(config.resolved_log_dir() / trainer.STATE_FILE,
               {"run_id": config.run.run_id, "checkpoints": [{"kind": "state", "path": "tinker://full"}]})
    payload["shuffle_seed"] += 1  # explicit world_order remains exactly the same
    slate.write_text(json.dumps(payload))
    assert trainer.select_world_stream(config).worlds == original.worlds
    with pytest.raises(RuntimeError, match="hero metadata changed"):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))


@pytest.mark.parametrize("interrupted", [False, True])
def test_real_trainer_monotonic_extension_preserves_identity_and_provenance(
    tmp_path, monkeypatch, training_sdk, interrupted,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=4, checkpoint_every=1)
    config = replace(config, train=replace(config.train, steps=2))
    deps = load_tinker_deps()
    if interrupted:
        training_sdk.fail_sampler_step = 2
        with pytest.raises(RuntimeError, match="external sampler-save interruption"):
            asyncio.run(trainer.run_train(config, deps=deps))
        state = json.loads((config.resolved_log_dir() / trainer.STATE_FILE).read_text())
    else:
        state = asyncio.run(trainer.run_train(config, deps=deps))
    root = config.resolved_log_dir()
    snapshot = (root / trainer.SNAPSHOT_FILE).read_bytes()
    checkpoint = trainer.latest_full_checkpoint(state)
    before = (root / trainer.STATE_FILE).read_bytes()
    with pytest.raises(RuntimeError, match="may only extend monotonically"):
        asyncio.run(trainer.run_train(replace(config, train=replace(config.train, steps=1)), deps=_no_sdk()))
    assert (root / trainer.STATE_FILE).read_bytes() == before
    environment = dict(os.environ)
    training_sdk.observe = lambda: _assert_environment(environment)
    processes = _observe_processes(monkeypatch, environment)
    extended = replace(config, run=replace(config.run, run_id="later-generated-id"),
                       train=replace(config.train, steps=3))
    after = asyncio.run(trainer.run_train(extended, deps=deps))
    assert after["run_id"] == config.run.run_id and after["target_steps"] == after["step"] == 3
    assert after["status"] == "complete" and after["next_world_offset"] == 3
    assert ("restore", checkpoint["path"], checkpoint["step"]) in training_sdk.calls
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == snapshot
    saved = json.loads(snapshot)
    assert saved["config"]["train"]["steps"] == 2 and saved["config"]["run"]["run_id"] == config.run.run_id
    assert all(row["name"].startswith(config.run.run_id + "-") for row in after["checkpoints"])
    rows = [json.loads(line) for line in (root / trainer.METRICS_FILE).read_text().splitlines()]
    assert [row["step"] for row in rows] == [0, 1, 2]
    assert [row["worlds"][0]["world_id"] for row in rows] == saved["worlds"][:3]
    assert trainer.latest_full_checkpoint(after)["step"] == 3
    _assert_private_runtime(tmp_path, environment, processes)


@pytest.mark.parametrize("snapshot_only", [False, True])
def test_failed_extension_durably_ratchets_target_before_sdk_and_cannot_decrease(
    tmp_path, monkeypatch, training_sdk, snapshot_only,
):
    config, _ = _runtime(tmp_path, monkeypatch, count=3, checkpoint_every=1)
    config = replace(config, train=replace(config.train, steps=1))
    deps = load_tinker_deps()
    root = config.resolved_log_dir()
    if snapshot_only:
        write_json(root / trainer.SNAPSHOT_FILE, trainer.snapshot(config, trainer.select_world_stream(config)))
    else:
        assert asyncio.run(trainer.run_train(config, deps=deps))["step"] == 1
    snapshot = (root / trainer.SNAPSHOT_FILE).read_bytes()
    original_service = deps.tinker.ServiceClient
    extended = replace(config, run=replace(config.run, run_id="later-generated-id"),
                       train=replace(config.train, steps=3))

    class FailedExtension(original_service):
        async def get_server_capabilities_async(self):
            durable = json.loads((root / trainer.STATE_FILE).read_text())
            assert durable["target_steps"] == 3 and durable["run_id"] == config.run.run_id
            raise RuntimeError("external capability failure during extension")

    monkeypatch.setattr(deps.tinker, "ServiceClient", FailedExtension)
    with pytest.raises(RuntimeError, match="external capability failure"):
        asyncio.run(trainer.run_train(extended, deps=deps))
    before = (root / trainer.STATE_FILE).read_bytes()
    assert json.loads(before)["target_steps"] == 3
    with pytest.raises(RuntimeError, match="requires at least 3"):
        asyncio.run(trainer.run_train(replace(extended, train=replace(extended.train, steps=2)), deps=_no_sdk()))
    assert (root / trainer.STATE_FILE).read_bytes() == before
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == snapshot
    monkeypatch.setattr(deps.tinker, "ServiceClient", original_service)
    after = asyncio.run(trainer.run_train(extended, deps=deps))
    assert after["step"] == after["target_steps"] == 3 and after["status"] == "complete"
    assert after["run_id"] == config.run.run_id and after["next_world_offset"] == 3
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == snapshot


def test_resume_target_must_cover_current_checkpoint_even_if_saved_target_is_lower(tmp_path):
    config, state, saved = _saved_run(tmp_path)
    state["checkpoints"][0]["step"] = config.train.steps + 1
    with pytest.raises(RuntimeError, match="current checkpoint requires at least"):
        trainer._resume_config(config=config, state=state, saved=saved, current=saved,
                               log_dir=config.resolved_log_dir())
