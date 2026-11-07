"""Portable checks for backend separation, exact credit and durable policy publication."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.credit.grpo import GRPOTrainableDatumSpec
from shellsmith.credit.compaction import CompactionAwareTrajectory, CompactionTurn
from shellsmith.credit.compaction_datums import build_normalized_tinker_trajectory
from shellsmith.credit.to_tinker_bridge import CapturedModelTurn, CapturedRollout, build_token_group
from shellsmith.episode.vat_protocol import initialize_command
from shellsmith.policy.records import PolicyRolloutRecord
from shellsmith.train.backend import TokenBatch, TokenPrompt, TokenSequence, player_profile
from shellsmith.train.config import ComputeSection, RunSection, TrainConfig, load_train_config, validate_train_config
from shellsmith.train.durable_capture import PolicyProvenance, load_capture, persist_accepted_capture, serialize_capture
from shellsmith.train import trainer
from shellsmith.train.stream_policy import WorldAction


def test_mlx_config_resolves_profile_without_changing_hosted_defaults(tmp_path):
    path = tmp_path / "run.toml"
    path.write_text('[compute]\nbackend="mlx"\n')
    config = load_train_config(path)
    assert config.adaptive.parallelism == config.adaptive.parallelism_per_provider == 2
    assert player_profile(config)["context_window"] == 32768
    assert config.mlx.lora_rank == 16 and config.mlx.lora_scale == 2
    hosted = load_train_config(None)
    assert hosted.compute.backend == "tinker" and hosted.player.lora_rank == 64
    assert player_profile(hosted)["compaction_reserve_tokens"] == 65536
    path.write_text('[compute]\nbackend="mlx"\n[mlx]\npretend_setting=true\n')
    with pytest.raises(ValueError, match="Unknown key"):
        load_train_config(path)


def test_mlx_held_out_quota_is_bounded_to_inference():
    from shellsmith.train.config import merged_stream_policy
    config = TrainConfig(run=RunSection(run_id="qualification"), compute=ComputeSection(backend="mlx"))
    config = replace(config, adaptive=replace(config.adaptive, initial_attempts=16, maximum_attempts=24,
                                              minimum_trainable=16))
    evaluation = merged_stream_policy(config, campaign_mode="inference")
    assert evaluation.adaptive_policy().minimum_trainable == 16
    assert evaluation.adaptive_policy().maximum_attempts == 24
    with pytest.raises(ValueError, match="maximum_attempts"):
        merged_stream_policy(config, campaign_mode="mlx_train")
    with pytest.raises(ValueError, match="maximum_attempts"):
        merged_stream_policy(replace(config, compute=ComputeSection()), campaign_mode="inference")


def test_mlx_held_out_quota_reaches_world_collection(tmp_path):
    from shellsmith.train.config import merged_stream_policy
    from shellsmith.train.rollout_stack import episode_args
    from shellsmith.train.stream_engine import collect_update_worlds, WorldCollectResult
    from shellsmith.train.stream_policy import WorldDecision
    config = TrainConfig(run=RunSection(run_id="qualification"), compute=ComputeSection(backend="mlx"))
    config = replace(config, adaptive=replace(config.adaptive, initial_attempts=16, maximum_attempts=24,
                                              minimum_trainable=16))
    base = merged_stream_policy(config, campaign_mode="inference")
    async def collect_one(path, args, world_config, flavor):
        world_config.validate()
        assert world_config.maximum_attempts == 24 and world_config.minimum_trainable == 16
        return WorldCollectResult("heldout", WorldDecision(WorldAction.GROUP_READY))
    result = asyncio.run(collect_update_worlds(task_runs=[(tmp_path / "heldout.md", episode_args(config))],
        start_offset=0, groups_per_update=1, base_config=base,
        classify_flavor=lambda _: "agentic", collect_one=collect_one))
    assert result.next_world_offset == 1 and len(result.worlds) == 1


@pytest.mark.parametrize("campaign_mode", ["mlx_train", "inference"])
@pytest.mark.parametrize("field,value", [("context_window", 65536), ("lora_rank", 32),
    ("lora_targets", ("q_proj",)), ("compaction_reserve_tokens", 8192), ("max_tokens", 32768)])
def test_mlx_rejects_unqualified_profiles_before_work(field, value, campaign_mode):
    config = TrainConfig(run=RunSection(run_id="test"), compute=ComputeSection(backend="mlx"))
    with pytest.raises(ValueError):
        validate_train_config(replace(config, mlx=replace(config.mlx, **{field: value})), campaign_mode=campaign_mode)


@pytest.mark.parametrize("coefficient,discount", [(.01, 0.), (0., .95), (.01, .95), (-1., 2.)])
def test_mlx_inference_ignores_kl_settings_but_training_rejects_them(tmp_path, coefficient, discount):
    from shellsmith.paths import DEFAULT_WORLD
    from shellsmith.train.rollout_stack import world_task_runs

    path = tmp_path / "run.toml"
    path.write_text('[compute]\nbackend="mlx"\n[train]\n'
                    f'kl_penalty_coef={coefficient}\nkl_discount_factor={discount}\n')
    # The public campaign loader and its repeated rollout preflight both ignore
    # optimizer-only settings, without rewriting the recorded configuration.
    config = load_train_config(path, campaign_mode="inference")
    assert config.train.kl_penalty_coef == coefficient and config.train.kl_discount_factor == discount
    assert len(world_task_runs(config, [DEFAULT_WORLD], campaign_mode="inference")) == 1
    for campaign_mode in ("tinker_train", "mlx_train"):
        with pytest.raises(ValueError, match="(?i)kl"):
            load_train_config(path, campaign_mode=campaign_mode)
        with pytest.raises(ValueError, match="(?i)kl"):
            validate_train_config(config, campaign_mode=campaign_mode)


def test_mlx_inference_ignores_learning_rate_but_keeps_sampling_admission():
    config = TrainConfig(run=RunSection(run_id="inference"), compute=ComputeSection(backend="mlx"))
    config = replace(config, train=replace(config.train, learning_rate=-1., kl_penalty_coef=.01, kl_discount_factor=.95))
    validate_train_config(config, campaign_mode="inference")
    for campaign_mode in ("tinker_train", "mlx_train"):
        with pytest.raises(ValueError, match="learning_rate"):
            validate_train_config(config, campaign_mode=campaign_mode)
    with pytest.raises(ValueError, match="qualified MLX profile"):
        validate_train_config(replace(config, player=replace(config.player, temperature=.5)), campaign_mode="inference")


def test_portable_datum_masks_injected_and_replayed_tokens():
    spec = GRPOTrainableDatumSpec(schema="shellsmith.grpo_trainable_datum.v0", datum_index=0,
        group_id="g", step=0, rollout_index=0, record_index=0, request_id="r",
        prompt=TokenPrompt((1, 2)), sampled_tokens=(3, 4, 5), sampled_logprobs=(-1., -2., -3.),
        reward=.5, advantage=.25, forced_close_token_span=(1, 2))
    sequence = spec.to_token_sequence()
    assert sequence.prompt_token_ids == (1, 2)
    assert sequence.advantages == (.25, 0., .25) and sequence.loss_mask == (1., 0., 1.)
    replay = replace(spec, train_sampled_tokens=False).to_token_sequence()
    assert replay.loss_mask == replay.advantages == (0., 0., 0.)
    assert TokenBatch((sequence,)).reduction == "sum"
    with pytest.raises(ValueError, match="sum reduction"):
        TokenBatch((sequence,), reduction="mean")


def test_token_usage_counts_discarded_draws_without_counting_final_attempt_twice():
    from shellsmith.train.runner import token_usage_for_records
    accepted = PolicyRolloutRecord("accepted", 1., TokenPrompt((1, 2)), 2, [3, 4, 5, 6], [-1.] * 4,
        "ok", "stop", {}, backend="mlx", forced_close_token_span=(1, 2),
        protocol_row={"thinking_tokens": 1, "response_tokens": 2},
        diagnostic_generation_attempts=(
            {"sampled_tokens": [9, 8], "disposition": "rejected_eos_inside_think"},
            {"sampled_tokens": [3], "disposition": "accepted"}))
    failed = replace(accepted, request_id="failed", sampled_tokens=[9], sampled_logprobs=[-1.],
        forced_close_token_span=None, protocol_row={"thinking": True, "protocol_error": "exhausted"},
        diagnostic_generation_attempts=tuple({"sampled_tokens": [9], "disposition": "rejected_eos_inside_think"}
                                             for _ in range(3)))
    assert token_usage_for_records([accepted, failed]) == {
        "prompt_tokens": 10, "completion_tokens": 9, "thinking_tokens": 6,
        "response_tokens": 2, "forced_close_tokens": 1}


def test_neutral_compaction_keeps_one_rollout_weight():
    def record(name, tokens, forced=None):
        return PolicyRolloutRecord(name, 1., TokenPrompt((1, 2)), 2, list(tokens), [-.5] * len(tokens),
                                   "text", "stop", {}, forced_close_token_span=forced)
    trajectory = CompactionAwareTrajectory("rollout", "pi", (
        CompactionTurn(record("action", (3, 4, 5), (1, 2)), "world_action", (0,), 0),
        CompactionTurn(record("summary", (6, 7)), "compaction_generation", (), 0),
        CompactionTurn(record("replay", (6, 7)), "compaction_replay", (), 1),
        CompactionTurn(record("later", (8,)), "world_action", (1,), 1)))
    built = build_normalized_tinker_trajectory(deps=None, trajectory=trajectory,
        advantages_by_request_id={"action": 1., "summary": 1., "replay": 1., "later": 1.},
        group_id="group", step=0, rollout_index=0, reward=.5)
    assert built.trainable_token_count == 5
    assert sum(sum(item.datum.advantages) for item in built.datums) == pytest.approx(1.)
    replay = next(item for item in built.datums if item.request_id == "replay")
    assert replay.datum.advantages == replay.datum.loss_mask == (0., 0.)


def test_neutral_capture_round_trip_keeps_policy_identity_without_sdk(tmp_path):
    record = PolicyRolloutRecord(request_id="r", created_at=1., prompt=TokenPrompt((1, 2)),
        prompt_tokens=2, sampled_tokens=[3, 4], sampled_logprobs=[-1., -2.], content="text",
        finish_reason="stop", payload={}, backend="mlx", policy_ref="adapter-step1", policy_identity={"sha": "abc"},
        diagnostic_generation_attempts=({"attempt": 1, "sampled_tokens": [8], "sampled_logprobs": [-.4],
                                         "disposition": "rejected_eos_inside_think"},))
    capture = CapturedRollout(tmp_path / "world" / "episode", "group", (CapturedModelTurn(record, "final_text"),))
    provenance = PolicyProvenance("Qwen/Qwen3.8-27B", 16, "sampler_checkpoint", "adapter-step1", 1,
                                  "group", "episode", backend="mlx", policy_identity={"sha": "abc"})
    path = persist_accepted_capture(run_dir=tmp_path, episode_id="episode", capture=capture, provenance=provenance)
    assert path.name == "policy_trainable_capture.json"
    recovered = load_capture(path).captures[0].record
    assert recovered.prompt.to_ints() == [1, 2]
    assert recovered.policy_ref == record.policy_ref and recovered.policy_identity == record.policy_identity
    assert recovered.diagnostic_generation_attempts == record.diagnostic_generation_attempts
    # Legacy token captures remain readable without installing the hosted SDK.
    legacy = serialize_capture(replace(capture, captures=(CapturedModelTurn(replace(record, backend="tinker"), "final_text"),)), episode_id="episode")
    path.write_text(json.dumps(legacy))
    assert load_capture(path).captures[0].record.backend == "tinker"


def test_neutral_group_uses_existing_advantages_and_refuses_mixed_revisions(tmp_path):
    captures = []
    for index, reward in enumerate((.2, .8)):
        root = tmp_path / str(index)
        root.mkdir()
        (root / "acceptance_audit.json").write_text(json.dumps({"accepted_for_training": True,
            "world_id": "world", "final_task": {"reward": reward}}))
        record = PolicyRolloutRecord(f"r{index}", 1., TokenPrompt((1, 2)), 2, [3, 4], [-.2, -.3],
            "text", "stop", {}, backend="mlx", policy_ref="step1", policy_identity={"sha": "abc"})
        captures.append(CapturedRollout(root, "group", (CapturedModelTurn(record, "final_text"),)))
    result = build_token_group(group_id="g", step=1, rollouts=tuple(captures))
    assert result.rollout_advantages == pytest.approx((-1., 1.))
    assert sum(result.datums[0].datum.advantages) == pytest.approx(-1.)
    assert sum(result.datums[1].datum.advantages) == pytest.approx(1.)
    changed = replace(captures[1].captures[0].record, policy_ref="step2")
    captures[1] = replace(captures[1], captures=(CapturedModelTurn(changed, "final_text"),))
    with pytest.raises(RuntimeError, match="policy identities or revisions"):
        build_token_group(group_id="g", step=1, rollouts=tuple(captures))


def test_small_context_compaction_is_on_initialize_wire():
    wire = initialize_command(request_id="r", episode_id="e", workspace_path="/workspace", agent_home_path="/agent",
        model_provider="shellsmith-mlx", model_id="player", model_base_url="http://127.0.0.1", model_api="openai-completions",
        model_context_window=32768, model_max_tokens=8192, model_reasoning=True, thinking_level="high",
        compaction_reserve_tokens=12288, compaction_keep_recent_tokens=8192)
    assert wire["model"]["compaction"] == {"reserve_tokens": 12288, "keep_recent_tokens": 8192}


def _fake_runtime(tmp_path, monkeypatch):
    import shellsmith.mlx
    config = TrainConfig(run=RunSection(run_id="mlx-test", log_dir=str(tmp_path / "run")),
        compute=ComputeSection(backend="mlx", gpu_lease_root=str(tmp_path / "gpu")))
    config = replace(config, worlds=replace(config.worlds, count=3), train=replace(config.train, steps=1, groups_per_update=1))
    events = []
    identities = {"sha": "immutable"}
    def snapshot(current, *args, **kwargs):
        return {"config": json.loads(json.dumps(current.to_jsonable())), "pi_sdk": {"module_sha256": "sdk", "module_path": str(tmp_path)},
                "codex": {"executable_sha256": "codex", "executable_path": str(tmp_path)}, "policy_identity": identities,
                "runtime_dependencies": {"mlx": "pinned", "tokenizers": "pinned"},
                "python_runtime": {"version": "3.12.9", "executable": "/pinned/python", "implementation": "CPython"}}
    monkeypatch.setattr(trainer, "snapshot", snapshot)
    class Session:
        policy_identity = identities
        def __init__(self, checkpoint):
            self.step = checkpoint["step"] if checkpoint else 0
            self.policy_ref = checkpoint["sampler_path"] if checkpoint else "base:immutable"
            self.pending = self.policy_ref
            self.sampler = SimpleNamespace(config=SimpleNamespace(base_model="Qwen/Qwen3.8-27B"))
        async def update(self, batch, **kwargs):
            assert isinstance(batch, TokenBatch) and batch.sequences[0].advantages == (.5,)
            events.append(("update", self.step))
            self.step += 1
            return {"forward_metrics": {"loss": .2}, "optim_metrics": {}}
        async def save_checkpoint(self, kind, name, metadata=None):
            path = Path(config.run.log_dir) / "checkpoints" / name
            path.mkdir(parents=True)
            if kind == "sampler":
                self.pending = str(path)
            (path / "manifest.json").write_text(json.dumps({"kind": kind, "complete": True, "metadata": metadata}))
            events.append((kind, self.step))
            return str(path)
        async def publish_policy(self, revision):
            durable = json.loads((Path(config.run.log_dir) / trainer.STATE_FILE).read_text())
            full = trainer.latest_full_checkpoint(durable)
            assert full["step"] == revision and full["sampler_path"] == self.pending
            assert full["next_world_offset"] == durable["next_world_offset"]
            envelope = json.loads((Path(full["path"]) / "manifest.json").read_text())
            assert envelope["metadata"][trainer.HEAD_KEY] == full[trainer.HEAD_KEY] == durable[trainer.HEAD_KEY]
            if revision:
                metrics = (Path(config.run.log_dir) / trainer.METRICS_FILE).read_bytes()
                assert metrics.endswith(full[trainer.HEAD_KEY]["tail"].encode())
            events.append(("publish", revision))
            self.policy_ref = self.pending
            return self.policy_ref
        def close(self):
            events.append(("close", self.step))
    async def create_session(current, checkpoint, checkpoint_root):
        events.append(("restore" if checkpoint else "new", checkpoint["step"] if checkpoint else 0))
        return Session(checkpoint)
    monkeypatch.setattr(shellsmith.mlx, "create_session", create_session)
    monkeypatch.setattr(trainer, "open_rollout_stack", lambda *args, **kwargs: SimpleNamespace(
        capture_store=SimpleNamespace(captures_for_outcomes=lambda outcomes: ()), stream=None,
        runner=None, classify_world=lambda world: "test", collect_one=None, close=lambda **kwargs: None))
    async def collect(**kwargs):
        offset = kwargs["start_offset"]
        result = SimpleNamespace(world_id="world", decision=SimpleNamespace(action=WorldAction.GROUP_READY, reason="ready"),
            gm_model="codex", gm_fallback_applied=False, original_gm_model=None, outcomes=[{}], discarded_outcomes=[])
        return SimpleNamespace(worlds=[SimpleNamespace(result=result)], next_world_offset=offset + 1, settled_world_offsets=[])
    monkeypatch.setattr(trainer, "collect_update_worlds", collect)
    sequence = TokenSequence((1, 2), (3,), (-.2,), (.5,), (1.,), "request")
    monkeypatch.setattr(trainer, "build_tinker_group", lambda **kwargs: SimpleNamespace(datums=(SimpleNamespace(datum=sequence),), segments=(sequence,), history_rewrites=()))
    return config, events


def test_mlx_engine_saves_before_publish_and_restores_without_tinker(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    state = asyncio.run(trainer.run_train(config))
    assert state["step"] == 1
    assert events[:5] == [("new", 0), ("sampler", 0), ("state", 0), ("publish", 0), ("update", 0)]
    assert events[5:8] == [("sampler", 1), ("state", 1), ("publish", 1)]
    assert trainer.latest_full_checkpoint(state)["policy_identity"] == {"sha": "immutable"}
    assert trainer.latest_full_checkpoint(state)["committed_group_ids"] == state["committed_group_ids"] == ["step0:world"]
    assert trainer.latest_full_checkpoint(state)["runtime_dependencies"] == {"mlx": "pinned", "tokenizers": "pinned"}
    assert trainer.latest_full_checkpoint(state)["python_runtime"]["executable"] == "/pinned/python"
    root = Path(config.run.log_dir)
    row = json.loads((root / "metrics.jsonl").read_text())
    assert row["optimizer_updates_committed"] == 1
    timings = [json.loads(path.read_text()) for path in
               (root / "policy-publications").glob("checkpoint-*.json")]
    assert {row["step"] for row in timings} == {0, 1}
    assert all(row["state_checkpoint_seconds"] >= 0 for row in timings)
    extended = replace(config, train=replace(config.train, steps=2))
    resumed = asyncio.run(trainer.run_train(extended))
    assert ("restore", 1) in events and resumed["step"] == 2 and resumed["resumes"] == 1
    assert resumed["committed_group_ids"] == ["step0:world", "step1:world"]
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
    assert [row["segment"] for row in rows] == [0, 1]


def test_mlx_first_update_failure_can_resume_step_zero(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.submit_update
    async def fail(**kwargs):
        kwargs["before_submit"]()
        raise RuntimeError("interrupted optimizer")
    monkeypatch.setattr(trainer, "submit_update", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(trainer.run_train(config))
    state = json.loads((Path(config.run.log_dir) / trainer.STATE_FILE).read_text())
    assert state["optimizer_update_in_flight"] and trainer.latest_full_checkpoint(state)["step"] == 0
    monkeypatch.setattr(trainer, "submit_update", original)
    recovered = asyncio.run(trainer.run_train(config))
    assert recovered["step"] == 1 and ("restore", 0) in events


@pytest.mark.parametrize("kind", ["sampler", "state"])
@pytest.mark.parametrize("saved_before_failure", [False, True])
def test_mlx_initial_checkpoint_failure_retries_untouched_base(tmp_path, monkeypatch, kind, saved_before_failure):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.save_checkpoint
    async def fail(**kwargs):
        if kwargs["kind"] == kind and "-initial" in kwargs["name"]:
            if saved_before_failure:
                await original(**kwargs)
            raise RuntimeError("interrupted initial checkpoint")
        return await original(**kwargs)
    monkeypatch.setattr(trainer, "save_checkpoint", fail)
    with pytest.raises(RuntimeError, match="interrupted initial"):
        asyncio.run(trainer.run_train(config))
    root = Path(config.run.log_dir)
    state = json.loads((root / trainer.STATE_FILE).read_text())
    assert state["status"] == "failed" and state["baseline_pending"]
    assert state["checkpoints"] == [] and state["optimizer_update_in_flight"] is False
    monkeypatch.setattr(trainer, "save_checkpoint", original)
    recovered = asyncio.run(trainer.run_train(config))
    assert recovered["step"] == 1 and recovered["initial_checkpoint_attempt"] == 2
    baseline = next(row for row in recovered["checkpoints"] if row["kind"] == "state" and row["step"] == 0)
    assert "initial00002" in baseline["path"] and not recovered["baseline_pending"]
    assert events.count(("update", 0)) == 1


def test_mlx_initial_state_publication_failure_leaves_snapshot_only_retry(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.write_json
    def fail(path, *args, **kwargs):
        if Path(path).name == trainer.STATE_FILE:
            raise OSError("interrupted first state write")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(trainer, "write_json", fail)
    with pytest.raises(OSError, match="first state write"):
        asyncio.run(trainer.run_train(config))
    root = Path(config.run.log_dir)
    assert sorted(path.name for path in root.iterdir()) == [trainer.SNAPSHOT_FILE]
    monkeypatch.setattr(trainer, "write_json", original)
    assert asyncio.run(trainer.run_train(config))["step"] == 1
    assert events.count(("update", 0)) == 1


def test_mlx_failed_full_save_rolls_back_without_orphan_name_collision(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.save_checkpoint
    async def fail(**kwargs):
        if kwargs["kind"] == "state" and kwargs.get("metadata", {}).get("step") == 1:
            raise RuntimeError("interrupted full state save")
        return await original(**kwargs)
    monkeypatch.setattr(trainer, "save_checkpoint", fail)
    with pytest.raises(RuntimeError, match="interrupted full"):
        asyncio.run(trainer.run_train(config))
    root = Path(config.run.log_dir)
    old_sampler = root / "checkpoints" / "mlx-test-step00001-s000"
    assert old_sampler.is_dir()
    monkeypatch.setattr(trainer, "save_checkpoint", original)
    recovered = asyncio.run(trainer.run_train(config))
    assert recovered["step"] == 1 and ("restore", 0) in events
    assert old_sampler.is_dir()  # Preserve the uncommitted artifact for diagnosis.
    assert recovered["active_policy_ref"].endswith("mlx-test-step00001-s001")
    assert recovered["committed_group_ids"] == ["step0:world"]


def test_mlx_full_save_before_journal_recovers_exact_write_ahead_row(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.MetricsJournal.publish
    def fail(*args, **kwargs):
        raise RuntimeError("interrupted journal publication")
    monkeypatch.setattr(trainer.MetricsJournal, "publish", fail)
    with pytest.raises(RuntimeError, match="interrupted journal"):
        asyncio.run(trainer.run_train(config))
    root = Path(config.run.log_dir)
    state = json.loads((root / trainer.STATE_FILE).read_text())
    checkpoint = trainer.latest_full_checkpoint(state)
    envelope = json.loads((Path(checkpoint["path"]) / "manifest.json").read_text())
    expected = envelope["metadata"][trainer.HEAD_KEY]["tail"]
    assert checkpoint["step"] == 1 and expected
    assert not (root / trainer.METRICS_FILE).exists()
    monkeypatch.setattr(trainer.MetricsJournal, "publish", original)
    recovered = asyncio.run(trainer.run_train(config))
    assert recovered["step"] == 1 and (root / trainer.METRICS_FILE).read_text() == expected
    assert events.count(("update", 0)) == 1 and ("restore", 1) in events


def test_mlx_skipped_world_cursor_is_checkpointed_at_step_zero(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    collections = []
    async def collect(**kwargs):
        collections.append(kwargs["start_offset"])
        result = SimpleNamespace(world_id="world", decision=SimpleNamespace(action=WorldAction.SKIP_INSUFFICIENT, reason="no data"),
            gm_model="codex", gm_fallback_applied=False, original_gm_model=None, outcomes=[], discarded_outcomes=[])
        return SimpleNamespace(worlds=[SimpleNamespace(result=result)], next_world_offset=3, settled_world_offsets=[])
    monkeypatch.setattr(trainer, "collect_update_worlds", collect)
    state = asyncio.run(trainer.run_train(config))
    full = trainer.latest_full_checkpoint(state)
    assert state["step"] == full["step"] == 0 and full["next_world_offset"] == 3
    asyncio.run(trainer.run_train(config))
    assert collections == [0] and not any(kind == "update" for kind, _ in events)


def test_mlx_identity_change_closes_session_before_collection(tmp_path, monkeypatch):
    import shellsmith.mlx
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = shellsmith.mlx.create_session
    async def changed(*args, **kwargs):
        session = await original(*args, **kwargs)
        session.policy_identity = {"sha": "changed-after-preflight"}
        return session
    monkeypatch.setattr(shellsmith.mlx, "create_session", changed)
    with pytest.raises(RuntimeError, match="identity changed"):
        asyncio.run(trainer.run_train(config))
    assert events == [("new", 0), ("close", 0)]


def test_mlx_final_publication_failure_persists_failed_status_and_resumes(tmp_path, monkeypatch):
    config, events = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.open_rollout_stack
    def broken(*args, **kwargs):
        stack = original(*args, **kwargs)
        def close(**kwargs):
            raise OSError("interrupted campaign report")
        stack.close = close
        return stack
    monkeypatch.setattr(trainer, "open_rollout_stack", broken)
    with pytest.raises(OSError, match="campaign report"):
        asyncio.run(trainer.run_train(config))
    state = json.loads((Path(config.run.log_dir) / trainer.STATE_FILE).read_text())
    assert state["status"] == "failed" and state["finalization_error"]["type"] == "OSError"
    assert trainer.latest_full_checkpoint(state)["step"] == 1
    monkeypatch.setattr(trainer, "open_rollout_stack", original)
    assert asyncio.run(trainer.run_train(config))["status"] == "complete"
    assert events.count(("update", 0)) == 1


def test_mlx_final_publication_failure_preserves_original_update_error(tmp_path, monkeypatch):
    config, _ = _fake_runtime(tmp_path, monkeypatch)
    original = trainer.open_rollout_stack
    def broken(*args, **kwargs):
        stack = original(*args, **kwargs)
        def close(**kwargs):
            raise OSError("interrupted campaign report")
        stack.close = close
        return stack
    async def fail_update(**kwargs):
        kwargs["before_submit"]()
        raise RuntimeError("original update error")
    monkeypatch.setattr(trainer, "open_rollout_stack", broken)
    monkeypatch.setattr(trainer, "submit_update", fail_update)
    with pytest.raises(RuntimeError, match="original update error") as caught:
        asyncio.run(trainer.run_train(config))
    assert any("finalization also failed" in note for note in caught.value.__notes__)
    state = json.loads((Path(config.run.log_dir) / trainer.STATE_FILE).read_text())
    assert state["status"] == "failed" and state["finalization_error"]["type"] == "OSError"


def test_frozen_runtime_hash_needs_no_git_and_rejects_extra_code(tmp_path, monkeypatch):
    from shellsmith.jobs.bundle import pack, unpack
    source = tmp_path / "source"
    worker = source / "src/shellsmith/jobs/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# runtime\n")
    (source / "pyproject.toml").write_text("[project]\nname='test'\n")
    runtime = unpack(tmp_path / "home", pack(source))
    monkeypatch.setattr(trainer, "PROJECT_ROOT", runtime)
    assert trainer._git_sha().startswith("runtime:")
    assert len(trainer._source_sha256()) == 64
    (runtime / "src/shellsmith/unexpected.py").write_text("# mutation\n")
    with pytest.raises(ValueError, match="Unexpected source"):
        trainer._source_sha256()



@pytest.mark.parametrize("mode", ["disabled", "offline", "auto_disabled"])
def test_publication_receipt_io_cannot_fail_a_committed_update(
    tmp_path, monkeypatch, capsys, mode,
):
    from shellsmith.train import telemetry

    config, _ = _fake_runtime(tmp_path, monkeypatch)
    config = replace(config, telemetry=replace(
        config.telemetry, mode="auto" if mode == "auto_disabled" else mode,
        provider="wandb" if mode == "auto_disabled" else "backprop",
        capture_console=False, system_metrics=False))
    if mode == "auto_disabled":
        monkeypatch.setenv("WANDB_MODE", "disabled")
    original = telemetry.write_json
    attempted = []

    def fail_receipt(path, value):
        if path.parent.name == "policy-publications":
            attempted.append(path)
            raise OSError("receipt-specific disk failure")
        return original(path, value)

    monkeypatch.setattr(telemetry, "write_json", fail_receipt)
    state = asyncio.run(trainer.run_train(config))
    assert state["status"] == "complete" and state["rft_updates_completed"] == 1
    assert not (config.resolved_log_dir() / "policy-publications").exists()
    assert bool(attempted) == (mode == "offline")
    if mode == "offline":
        assert "receipt-specific disk failure" in capsys.readouterr().err


def test_fresh_baseline_receipts_precede_updates_and_replay_once(
    tmp_path, monkeypatch,
) -> None:
    """
    Purpose: Verify initial checkpoint and policy receipts reach the sink before updates.
    Input: A fresh MLX run with the offline Backprop sink.
    Output: Two baseline receipts, each uploaded once across telemetry resume.
    """
    from shellsmith.train.telemetry import TrainingTelemetry

    config, _ = _fake_runtime(tmp_path, monkeypatch)
    config = replace(config, telemetry=replace(
        config.telemetry, mode="offline", capture_console=False, system_metrics=False))
    state = asyncio.run(trainer.run_train(config))
    root = config.resolved_log_dir()
    baseline = [json.loads(path.read_text())
                for path in (root / "policy-publications").glob("*.json")]
    baseline = [row for row in baseline if row.get("context") == "baseline"]
    assert len(baseline) == 2 and all(row["step"] == 0 for row in baseline)
    checkpoint = next(row for row in baseline if row.get("kind") == "state_checkpoint")
    assert checkpoint["sampler_checkpoint_seconds"] >= 0
    assert checkpoint["state_checkpoint_seconds"] >= 0
    assert next(row for row in baseline if "seconds" in row)["seconds"] >= 0
    with TrainingTelemetry(config, snapshot={}, state=state):
        pass
    metadata = json.loads((root / "backprop.json").read_text())
    histories = []
    for directory in metadata["sync_dirs"]:
        for line in (Path(directory) / "events.jsonl").read_text().splitlines():
            event = json.loads(line)
            if event["type"] == "history":
                histories.append(event["payload"]["metrics"])
    baseline_indices = [index for index, row in enumerate(histories)
                        if row.get("publication/context") == "baseline"]
    assert len(baseline_indices) == 2
    assert max(baseline_indices) < next(index for index, row in enumerate(histories)
                                      if "train/update_id" in row)
