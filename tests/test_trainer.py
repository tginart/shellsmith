"""Trainer pieces that need no live Tinker: config, world selection, CISPO submit, resume rows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.paths import DEFAULT_LOG_ROOT
from shellsmith.train import rollout_stack, trainer
from shellsmith.train.config import load_train_config


def _toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(text)
    return path


def test_defaults_and_log_dir_under_temp() -> None:
    config = load_train_config(None)
    assert config.run.run_id.startswith("run-")
    assert config.resolved_log_dir() == DEFAULT_LOG_ROOT / "runs" / config.run.run_id
    assert config.player.base_model == "Qwen/Qwen3.8-27B:peft:262144"
    assert config.gm.model == "gpt-6.1-sol"
    assert (config.adaptive.initial_attempts, config.adaptive.maximum_attempts, config.adaptive.minimum_trainable,
            config.adaptive.minimum_reward_range) == (4, 8, 4, 0.15)
    assert config.train.kl_penalty_coef == 0.0


def test_config_file_and_log_dir_override(tmp_path: Path) -> None:
    path = _toml(tmp_path, '[run]\nlog_dir = "/x/run"\n[worlds]\nselect = "ids"\nids = ["a", "b"]\ncount = 2\n'
                           '[train]\nsteps = 3\nlearning_rate = 1\n')
    config = load_train_config(path)
    assert config.resolved_log_dir() == Path("/x/run").resolve()
    assert config.worlds.ids == ("a", "b") and config.train.learning_rate == 1.0
    assert load_train_config(path, log_dir=tmp_path / "o").resolved_log_dir() == (tmp_path / "o").resolve()


@pytest.mark.parametrize("text, message", [
    ("[train]\nstep = 3\n", "Unknown key"),
    ("[trian]\n", "Unknown section"),
    ('[train]\nsteps = "3"\n', "must be int"),
    ('[worlds]\nselect = "ids"\n', "needs"),
    ('[gm]\nruntime = "claude"\n', "must be 'codex'"),
])
def test_config_typos_are_errors(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_train_config(_toml(tmp_path, text))


def test_world_selection_modes(tmp_path: Path) -> None:
    manifest = trainer.select_worlds(load_train_config(_toml(tmp_path, "[worlds]\ncount = 3\n")))
    assert [path.stem for path in manifest][0] == "checkout_idempotency_001" and len(manifest) == 3
    shuffled = trainer.select_worlds(load_train_config(_toml(tmp_path, '[worlds]\nselect = "random"\nseed = 7\ncount = 3\n')))
    assert shuffled == trainer.select_worlds(load_train_config(_toml(tmp_path, '[worlds]\nselect = "random"\nseed = 7\ncount = 3\n')))
    with pytest.raises(ValueError, match="not in the manifest"):
        trainer.select_worlds(load_train_config(_toml(tmp_path, '[worlds]\nselect = "ids"\nids = ["nope"]\n')))


class FakeFuture:
    def __init__(self, value: object) -> None:
        self.value = value

    async def result_async(self) -> object:
        return self.value


class FakeTrainingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def forward_backward_async(self, datums: list, *, loss_fn: str, loss_fn_config: dict) -> FakeFuture:
        self.calls.append(("forward_backward", (datums, loss_fn, loss_fn_config)))
        return FakeFuture(SimpleNamespace(metrics={"loss:sum": 1.5}))

    async def optim_step_async(self, adam: object) -> FakeFuture:
        self.calls.append(("optim_step", adam))
        return FakeFuture(SimpleNamespace(metrics={"grad_norm": 0.7}))


def test_submit_update_strips_mask_and_uses_cispo() -> None:
    deps = SimpleNamespace(types=SimpleNamespace(
        Datum=lambda model_input, loss_fn_inputs: SimpleNamespace(model_input=model_input, loss_fn_inputs=loss_fn_inputs),
        AdamParams=lambda **kwargs: kwargs))
    datum = SimpleNamespace(datum=SimpleNamespace(model_input="mi", loss_fn_inputs={"advantages": 1, "mask": 0}))
    client = FakeTrainingClient()
    metrics = asyncio.run(trainer.submit_update(deps=deps, training_client=client, reference_client=None,
                                                datums=[datum], config=load_train_config(None)))
    sent, loss_fn, loss_config = client.calls[0][1]  # type: ignore[misc]
    assert loss_fn == "cispo" and loss_config == {"clip_low_threshold": 0.0, "clip_high_threshold": 4.0}
    assert "mask" not in sent[0].loss_fn_inputs
    assert client.calls[1][1]["learning_rate"] == 5e-5  # type: ignore[index]
    assert metrics == {"forward_metrics": {"loss:sum": 1.5}, "optim_metrics": {"grad_norm": 0.7}}


def test_resume_uses_latest_full_checkpoint_only() -> None:
    state = {"checkpoints": [
        {"kind": "sampler", "path": "s1"}, {"kind": "state", "path": "f1", "step": 1},
        {"kind": "sampler", "path": "s2"}, {"kind": "state", "path": "f2", "step": 2},
        {"kind": "sampler", "path": "s3"}]}
    assert trainer.latest_full_checkpoint(state)["path"] == "f2"
    assert trainer.latest_full_checkpoint({"checkpoints": []}) is None


def test_update_collector_accepts_trainer_episode_args(tmp_path: Path) -> None:
    """``collect_update_worlds`` reads the adaptive settings off each world's args."""
    from shellsmith.train.stream_engine import collect_update_worlds

    config = load_train_config(_toml(tmp_path, "[adaptive]\ninitial_attempts = 2\nminimum_trainable = 2\n"))
    seen: list[int] = []

    async def collect_one(world_file, task_args, world_config, flavor):  # type: ignore[no-untyped-def]
        seen.append(world_config.initial_attempts)
        raise LookupError("stop after admission")

    args = rollout_stack.episode_args(config)
    with pytest.raises(LookupError, match="stop after admission"):
        asyncio.run(collect_update_worlds(
            task_runs=[(tmp_path / "w.md", args)], start_offset=0, groups_per_update=1,
            base_config=rollout_stack.stream_config(config, campaign_mode="tinker_train", groups_per_update=1), classify_flavor=lambda path: "f", collect_one=collect_one))
    assert seen == [2]


def test_world_overrides_reach_only_that_world(tmp_path: Path) -> None:
    config = load_train_config(_toml(tmp_path, '[worlds]\ncount = 2\n'
                                               '[world_overrides.checkout_idempotency_001]\nminimum_reward_range = 1\n'))
    assert config.world_overrides == {"checkout_idempotency_001": {"minimum_reward_range": 1.0}}
    runs = rollout_stack.world_task_runs(config, trainer.select_worlds(config))
    assert [args.minimum_reward_range for _, args in runs] == [1.0, 0.15]


@pytest.mark.parametrize("text, message", [
    ('[world_overrides.w]\nparallelism = 2\n', "may only set"),
    ('[world_overrides.w]\nminimum_trainable = "4"\n', "must be int"),
])
def test_world_override_typos_are_errors(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_train_config(_toml(tmp_path, text))


def test_world_override_for_an_unselected_world_is_an_error(tmp_path: Path) -> None:
    config = load_train_config(_toml(tmp_path, '[worlds]\ncount = 1\n[world_overrides.nope]\ninitial_attempts = 2\n'))
    with pytest.raises(ValueError, match="not in this run: nope"):
        rollout_stack.world_task_runs(config, trainer.select_worlds(config))


def test_resume_may_only_raise_steps(tmp_path: Path) -> None:
    base = load_train_config(_toml(tmp_path, "[train]\nsteps = 1\n")).to_jsonable()
    more = load_train_config(_toml(tmp_path, "[train]\nsteps = 2\n")).to_jsonable()
    other = load_train_config(_toml(tmp_path, "[train]\nsteps = 2\nlearning_rate = 1e-4\n")).to_jsonable()
    assert trainer._resume_key(base) == trainer._resume_key(more)
    saved = json.loads(json.dumps(base))  # what config_snapshot.json holds after a restart
    assert trainer._resume_key(saved) == trainer._resume_key(more)
    assert trainer._resume_key(base) != trainer._resume_key(other)


@pytest.mark.parametrize("override,message", [
    ({"initial_attempts": 9}, "initial_attempts must be within"),
    ({"maximum_attempts": 3}, "initial_attempts must be within"),
    ({"maximum_attempts": 13}, "maximum_attempts cannot exceed"),
    ({"minimum_trainable": 9}, "minimum_trainable must be within"),
    ({"minimum_trainable": 1}, "at least two"),
    ({"low_spread_top_up": 0}, "between one and four"),
    ({"low_spread_top_up": 5}, "between one and four"),
    ({"minimum_reward_range": -0.1}, "between 0 and 1"),
    ({"minimum_reward_range": 1.1}, "between 0 and 1"),
    ({"minimum_reward_range": float("nan")}, "finite"),
    ({"minimum_reward_range": float("inf")}, "finite"),
    ({"initial_attempts": True}, "must be int"),
])
def test_merged_world_policy_rejects_before_auth_or_output(tmp_path, override, message):
    from dataclasses import replace
    from shellsmith.train.config import validate_train_config

    config = load_train_config(None, log_dir=tmp_path / "not-started")
    config = replace(config, world_overrides={"checkout_idempotency_001": override})

    class NoSDK:
        def __getattr__(self, name):
            pytest.fail(f"Local merged-policy validation reached external SDK {name}")

    with pytest.raises(ValueError, match=message):
        validate_train_config(config)
    with pytest.raises(ValueError, match=message):
        asyncio.run(trainer.run_train(config, deps=NoSDK()))
    with pytest.raises(ValueError, match=message):
        rollout_stack.open_rollout_stack(
            config, deps=NoSDK(), sampling_client=None, root=config.resolved_log_dir(),
            campaign_id=config.run.run_id, campaign_mode="tinker_train", groups_per_update=1, state={})
    assert not config.resolved_log_dir().exists()


def test_toml_world_overrides_validate_merged_not_standalone_defaults(tmp_path):
    valid = '[adaptive]\ninitial_attempts = 2\nmaximum_attempts = 2\nminimum_trainable = 2\n'
    config = load_train_config(_toml(tmp_path, valid +
                                    '[world_overrides.checkout_idempotency_001]\nmaximum_attempts = 3\n'))
    stream = rollout_stack.world_task_runs(config, trainer.select_worlds(config))
    assert stream[0][1].initial_attempts == 2 and stream[0][1].maximum_attempts == 3
    with pytest.raises(ValueError, match=r"world_overrides.*initial_attempts must be within"):
        load_train_config(_toml(tmp_path, '[world_overrides.checkout_idempotency_001]\nmaximum_attempts = 3\n'))


def test_inference_validates_rollout_semantics_but_not_training_only_fields(tmp_path):
    from dataclasses import replace
    from shellsmith.train.config import validate_train_config

    text = ('[worlds]\ncount = 1\n[train]\nsteps = 0\ncheckpoint_every = 0\ngroups_per_update = 0\n'
            'learning_rate = -1\ncispo_clip_high = -1\nkl_penalty_coef = -1\nkl_discount_factor = 2\n'
            '[adaptive]\ninitial_attempts = 1\nmaximum_attempts = 1\nminimum_trainable = 1\n'
            '[world_overrides.checkout_idempotency_001]\nminimum_trainable = 1\n')
    config = load_train_config(_toml(tmp_path, text), campaign_mode="inference")
    validate_train_config(config, campaign_mode="inference")
    worlds = trainer.select_world_stream(config, campaign_mode="inference").worlds
    tasks = rollout_stack.world_task_runs(config, worlds, campaign_mode="inference")
    assert len(tasks) == 1 and tasks[0][1].minimum_trainable == 1
    stream = rollout_stack.stream_config(config, campaign_mode="inference", groups_per_update=1)
    assert stream.minimum_trainable == 1
    with pytest.raises(ValueError):
        validate_train_config(config)
    for runtime in ("claude", "pi"):
        with pytest.raises(ValueError, match="confined Codex"):
            validate_train_config(replace(config, gm=replace(config.gm, runtime=runtime)), campaign_mode="inference")
    assert config.resolved_log_dir(kind="campaigns") == DEFAULT_LOG_ROOT / "campaigns" / config.run.run_id


@pytest.mark.parametrize("field,value", [
    ("outage_failure_threshold", 0), ("outage_failure_threshold", True),
    ("outage_cooldown_seconds", -1.0), ("outage_cooldown_seconds", float("inf")),
])
def test_outage_circuit_fields_are_locally_validated(tmp_path, field, value):
    from dataclasses import replace
    from shellsmith.train.config import validate_train_config

    config = load_train_config(None)
    with pytest.raises(ValueError, match=field):
        validate_train_config(replace(config, adaptive=replace(config.adaptive, **{field: value})))


def test_outage_circuit_fields_do_not_leak_into_stream_policy(tmp_path):
    config = load_train_config(_toml(tmp_path,
        '[adaptive]\noutage_failure_threshold = 2\noutage_cooldown_seconds = 0\n'))
    stream = rollout_stack.stream_config(config, campaign_mode="tinker_train", groups_per_update=1)
    assert not hasattr(stream, "outage_failure_threshold") and not hasattr(stream, "outage_cooldown_seconds")
    assert config.adaptive.outage_failure_threshold == 2 and config.adaptive.outage_cooldown_seconds == 0


@pytest.mark.parametrize("only_unstarted", [False, True])
def test_world_summary_excludes_unstarted_provider_slots(only_unstarted: bool) -> None:
    """
    Purpose: Exclude provider outage placeholders from all rollout counts and rewards.
    Input: Placeholder slots, with or without real eligible and discarded episodes.
    Output: Counts that describe only episodes that actually started.
    """
    from shellsmith.train.stream_engine import WorldCollectResult
    from shellsmith.train.stream_policy import WorldAction, WorldDecision

    placeholder = {"episode_not_started": True, "error": "provider down"}
    outcomes = [placeholder, {"episode_not_started": True}]
    discarded = [placeholder]
    if not only_unstarted:
        outcomes += [
            {"accepted_count": 1, "audit": {"final_task": {"reward": .8}}},
            {"accepted_count": 0},
            {"error": "episode failed"},
        ]
        discarded.append({"accepted_count": 0, "discarded_for_training": True})
    result = WorldCollectResult(
        world_id="world", decision=WorldDecision(
            action=WorldAction.HALT_STREAM, reason="provider outage"),
        outcomes=outcomes, discarded_outcomes=discarded)
    row = rollout_stack.summarize_world(result)
    expected = {"attempts": 0, "attempted_total": 0, "accepted": 0,
                "discarded": 0, "failed": 0, "rejected": 0, "rewards": []}
    if not only_unstarted:
        expected.update(attempts=3, attempted_total=4, accepted=1, discarded=1,
                        failed=1, rejected=1, rewards=[.8, None, None])
    assert {key: row[key] for key in expected} == expected
