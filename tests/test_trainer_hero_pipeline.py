"""Per-world hero GM assignments reach the real collection/capture pipeline.

Only external model/SDK/CLI interfaces are fixtures, as in test_trainer_resume;
no selector, collector, trainer, identity, acceptance, or credit code is mocked.
"""
import asyncio
from dataclasses import replace
import json

from shellsmith.paths import PROJECT_ROOT
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import trainer
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def test_real_hero_training_keeps_per_world_models_and_order(tmp_path, monkeypatch, training_sdk):
    config, _audit = _runtime(tmp_path, monkeypatch)
    payload = json.loads((PROJECT_ROOT / "hero_runs/first_hero_baby_terra_4.json").read_text())
    world_ids = payload["world_order"]
    payload["simulators"]["sol"] = {
        "runtime": "codex", "model": "gpt-6.1-sol", "provider": "codex-cli", "reasoning_effort": "low",
    }
    payload["assignments"] = {"sol": [world_ids[0]]}
    payload["expected_simulator_counts"] = {"sol": 1, "terra": 3}
    slate = tmp_path / "assigned-hero.json"
    slate.write_text(json.dumps(payload))
    config = replace(
        config,
        worlds=replace(config.worlds, select="hero", hero_slate=str(slate), count=1),
        train=replace(config.train, groups_per_update=2, steps=1),
    )
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert state["status"] == "complete" and state["step"] == 1 and state["next_world_offset"] == 2
    root = config.resolved_log_dir()
    snap = json.loads((root / trainer.SNAPSHOT_FILE).read_text())
    assert snap["worlds"] == world_ids  # catalog count=1 must not truncate a hero slate
    metrics = json.loads((root / trainer.METRICS_FILE).read_text().splitlines()[0])
    assert [world["world_id"] for world in metrics["worlds"]] == world_ids[:2]
    actual_models = {world["world_id"]: world["gm_model"] for world in metrics["worlds"]}
    assert actual_models == {world_ids[0]: "gpt-6.1-sol", world_ids[1]: "gpt-5.6-terra"}
    assert metrics["datums"] == 4
    identities = list(state["group_identities"].values())
    assert {item["world_id"]: item["gm_model"] for item in identities} == actual_models
    assert len({item["key"] for item in identities}) == 2
    for outcome_path in (root / "segments/000").rglob("durable_outcome.json"):
        outcome = json.loads(outcome_path.read_text())["outcome"]
        assert outcome["gm_model"] == actual_models[outcome["world_id"]]
        assert outcome["accepted_count"] == 1
    assert trainer.latest_full_checkpoint(state)["step"] == 1


def test_real_hero_overrides_preserve_canonical_ids_gm_and_private_credentials(
    tmp_path, monkeypatch, training_sdk,
):
    import os
    from test_player_credentials import _assert_environment, _observe_processes
    from test_trainer_resume import _assert_private_runtime
    from test_trainer_world_ids import _hero, _world

    config, _audit = _runtime(tmp_path, monkeypatch)
    targets = [_world(tmp_path / "left" / "canonical_A.md"), _world(tmp_path / "right" / "canonical_B.md")]
    aliases = []
    for target in targets:
        alias = target.with_name("alias.md")
        alias.symlink_to(target.name)
        aliases.append(alias)
    slate = _hero(tmp_path, aliases)
    payload = json.loads(slate.read_text())
    payload["simulators"]["sol"] = {
        "runtime": "codex", "model": "gpt-6.1-sol", "provider": "codex-cli", "reasoning_effort": "low",
    }
    payload["assignments"] = {"sol": ["canonical_A"]}
    payload["expected_simulator_counts"] = {"sol": 1, "terra": 1}
    slate.write_text(json.dumps(payload))
    config = replace(
        config, worlds=replace(config.worlds, select="hero", hero_slate=str(slate), count=1),
        train=replace(config.train, groups_per_update=2, steps=1),
        world_overrides={"canonical_A": {"maximum_attempts": 4, "minimum_trainable": 4}},
    )
    environment = dict(os.environ)
    training_sdk.observe = lambda: _assert_environment(environment)
    processes = _observe_processes(monkeypatch, environment)
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    root = config.resolved_log_dir()
    assert state["step"] == 1 and state["next_world_offset"] == 2 and state["status"] == "complete"
    row = json.loads((root / trainer.METRICS_FILE).read_text().splitlines()[0])
    assert [(world["world_id"], world["attempts"], world["accepted"], world["gm_model"]) for world in row["worlds"]] == [
        ("canonical_A", 4, 4, "gpt-6.1-sol"), ("canonical_B", 2, 2, "gpt-5.6-terra"),
    ]
    assert row["datums"] == 6 and row["worlds"][0]["rewards"] == [0.2, 0.8, 0.2, 0.8]
    identities = {item["world_id"]: item for item in state["group_identities"].values()}
    assert identities["canonical_A"]["gm_reasoning_effort"] == "low"
    assert identities["canonical_B"]["gm_reasoning_effort"] == "high"
    snapshot = json.loads((root / trainer.SNAPSHOT_FILE).read_text())
    assert snapshot["worlds"] == ["canonical_A", "canonical_B"]
    assert [item["world_file"] for item in snapshot["world_identities"]] == [str(path) for path in targets]
    assert snapshot["config"]["world_overrides"] == config.world_overrides
    assert len(list((root / "segments/000").rglob("durable_outcome.json"))) == 6
    # Each real Player request uses a thinking sample and a final plain sample.
    assert len(training_sdk.budgets) == 12 and all(0 < budget <= config.player.max_tokens for budget in training_sdk.budgets)
    _assert_private_runtime(tmp_path, environment, processes)
