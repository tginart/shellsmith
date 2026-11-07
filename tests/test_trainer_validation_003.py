"""Validate experiment identities, GRPO group size, and selected hero GMs offline."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from shellsmith.paths import DEFAULT_LOG_ROOT, PROJECT_ROOT
from shellsmith.train.config import load_train_config, validate_train_config
from shellsmith.train import trainer


@pytest.mark.parametrize("run_id", [".", "..", "../outside", "a/b", "/absolute", "a b", "line\nbreak", "日本語", "percent%20", "query?x", "back\\slash"])
def test_run_id_cannot_escape_run_root_or_endpoint_grammar(tmp_path, run_id):
    path = tmp_path / "config.toml"
    path.write_text('[run]\nrun_id = ' + json.dumps(run_id, ensure_ascii=False) + '\n')
    with pytest.raises(ValueError, match="endpoint-safe non-dot path segment"):
        load_train_config(path)
    config = load_train_config(None)
    config = replace(config, run=replace(config.run, run_id=run_id))
    with pytest.raises(ValueError, match="endpoint-safe non-dot path segment"):
        asyncio.run(trainer.run_train(config, deps=SimpleNamespace()))


@pytest.mark.parametrize("run_id", ["trial", "trial.1-2_3", ".hidden-run", "_run", "UPPER-123"])
def test_safe_run_id_stays_within_default_namespace(tmp_path, run_id):
    path = tmp_path / "config.toml"
    path.write_text('[run]\nrun_id = ' + json.dumps(run_id) + '\n')
    config = load_train_config(path)
    assert config.resolved_log_dir().parent == DEFAULT_LOG_ROOT / "runs"
    validate_train_config(config)


def test_training_group_requires_two_samples_before_sdk_or_output(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[adaptive]\ninitial_attempts = 1\nmaximum_attempts = 1\nminimum_trainable = 1\n')
    with pytest.raises(ValueError, match="minimum_trainable must be at least two"):
        load_train_config(path)
    config = load_train_config(None, log_dir=tmp_path / "not-created")
    bad = replace(config, adaptive=replace(config.adaptive, minimum_trainable=1))
    with pytest.raises(ValueError, match="minimum_trainable must be at least two"):
        asyncio.run(trainer.run_train(bad, deps=SimpleNamespace()))
    assert not bad.resolved_log_dir().exists()
    # Starting with one attempt is still valid if a second can be collected.
    validate_train_config(replace(config, adaptive=replace(config.adaptive, initial_attempts=1, maximum_attempts=2, minimum_trainable=2, low_spread_top_up=1)))


@pytest.mark.parametrize("field,value,match", [
    ("model", "", "model must be non-empty"),
    ("model", "   ", "model must be non-empty"),
    ("model", " padded", "model must not contain"),
    ("reasoning_effort", "unsupported", "reasoning_effort"),
    ("reasoning_effort", "", "reasoning_effort"),
    ("provider", "openrouter", "provider"),
])
def test_selected_hero_simulators_are_validated_before_sdk(tmp_path, field, value, match):
    payload = json.loads((PROJECT_ROOT / "hero_runs/first_hero_baby_terra_4.json").read_text())
    payload["simulators"]["terra"][field] = value
    slate = tmp_path / "bad-hero.json"
    slate.write_text(json.dumps(payload))
    config = load_train_config(None, log_dir=tmp_path / "not-created")
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(slate)))
    with pytest.raises((ValueError, RuntimeError), match=match):
        asyncio.run(trainer.run_train(config, deps=SimpleNamespace()))
    assert not config.resolved_log_dir().exists()
