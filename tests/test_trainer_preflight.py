"""Fail unusable training configurations before model or collection work."""
import asyncio
from dataclasses import replace
from datetime import UTC, datetime
import os
from types import SimpleNamespace

import pytest

from shellsmith.train import config as config_module
from shellsmith.train import trainer
from shellsmith.train.config import TrainConfig, load_train_config, validate_train_config
from shellsmith.policy.sampler import PlayerRequestError


def _no_sdk():
    def forbidden():
        pytest.fail("Configuration validation must precede authenticated SDK work")
    return SimpleNamespace(tinker=SimpleNamespace(ServiceClient=forbidden))


@pytest.mark.parametrize("run_id", ["", " ", "\t\n"])
def test_direct_run_config_rejects_blank_identity_before_auth(tmp_path, run_id):
    config = TrainConfig()
    config = replace(config, run=replace(config.run, run_id=run_id, log_dir=str(tmp_path / "run")))
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError, match="run_id must be non-empty"):
        validate_train_config(replace(config, run=replace(config.run, log_dir="")))


def test_generated_run_ids_are_distinct_even_at_identical_time(monkeypatch):
    class Clock:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 6, tzinfo=UTC)
    monkeypatch.setattr(config_module, "datetime", Clock)
    configs = [load_train_config(None) for _ in range(20)]
    assert all(config.run_id_generated for config in configs)
    assert len({config.run.run_id for config in configs}) == len(configs)
    assert len({config.resolved_log_dir() for config in configs}) == len(configs)


def test_training_fallback_is_rejected_by_file_and_programmatic_validation(tmp_path):
    source = tmp_path / "run.toml"
    source.write_text('[adaptive]\nprovider_outage = "fallback"\n')
    with pytest.raises(ValueError, match="provider_outage must be 'halt'"):
        load_train_config(source)
    config = load_train_config(None, log_dir=tmp_path / "not-created")
    config = replace(config, adaptive=replace(config.adaptive, provider_outage="fallback"))
    with pytest.raises(ValueError, match="fallback GMs are unconfined"):
        asyncio.run(trainer.run_train(config, deps=_no_sdk()))
    assert not config.resolved_log_dir().exists()


@pytest.mark.parametrize("close_tokens", [1, 5, 11])
@pytest.mark.parametrize("offset", [-2, -1, 0])
def test_training_budget_matches_actual_tokenizer_before_sdk(tmp_path, monkeypatch, close_tokens, offset):
    calls = []
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: [42] * close_tokens)
    # These are only the external Cookbook/tokenizer interfaces, not a mocked
    # Shellsmith sampler or validation result.
    def get_renderer(name, actual_tokenizer):
        calls.append((name, actual_tokenizer))
        return SimpleNamespace(tokenizer=actual_tokenizer)
    def service():
        raise LookupError("valid-budget-reached-service-boundary")
    deps = SimpleNamespace(
        get_tokenizer=lambda model: tokenizer,
        renderers=SimpleNamespace(get_renderer=get_renderer),
        tinker=SimpleNamespace(ServiceClient=service),
    )
    codex = tmp_path / "codex"
    codex.write_text("#!/bin/sh\nexit 97\n")  # preflight fingerprints, but must never execute, this fixture
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    sdk_entry = tmp_path / "external-pi-sdk.js"
    sdk_entry.write_text("export {};\n")  # existing external file; no SDK import is needed for budget validation
    config = load_train_config(None, log_dir=tmp_path / "run")
    config = replace(config, player=replace(config.player, max_tokens=close_tokens + 2 + offset,
                                           pi_sdk_module_path=str(sdk_entry)))
    if offset < 0:
        with pytest.raises(PlayerRequestError, match=f"requires at least {close_tokens + 2} tokens"):
            asyncio.run(trainer.run_train(config, deps=deps))
        assert not config.resolved_log_dir().exists()
    else:
        with pytest.raises(LookupError, match="valid-budget-reached-service-boundary"):
            asyncio.run(trainer.run_train(config, deps=deps))
    assert calls == [(config.player.thinking_renderer, tokenizer)]
