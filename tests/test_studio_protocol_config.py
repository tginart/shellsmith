"""Editable protocol settings reach the portable boundary without loading MLX."""

from dataclasses import replace
from pathlib import Path
import asyncio

import pytest

from shellsmith.mlx.generation import completion_budget, reasoning_budget
from shellsmith.train.backend import player_profile
from shellsmith.train.config import ComputeSection, RunSection, TrainConfig, load_train_config, validate_train_config
from shellsmith.train.rollout_stack import episode_args
from shellsmith.train.studio import StudioBackend
from studio_compute.cli import parse_args
from studio_compute.protocol import validate_request

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("effort,limit", [("low", 2048), ("medium", 4096), ("xhigh", 8192)])
def test_effort_budget_is_inside_one_shared_context(effort, limit):
    base = TrainConfig(run=RunSection(run_id="config"), compute=ComputeSection(backend="studio"))
    config = replace(base, studio=replace(base.studio, reasoning_effort=effort))
    validate_train_config(config)
    assert reasoning_budget(config.studio) == limit
    assert completion_budget(config.studio) == limit + 3 + 32768
    assert player_profile(config)["context_window"] == 262144
    assert player_profile(config)["resolved_reasoning_tokens"] == limit


def test_explicit_reasoning_override_and_configured_retries():
    config = load_train_config(ROOT / "examples/studio-micro-train.toml")
    assert config.train.max_update_prompt_tokens == config.train.max_update_completion_tokens == 0
    assert config.gm.max_world_steps == 256
    changed = replace(config, studio=replace(config.studio, reasoning_tokens_medium=3000),
                      retry=replace(config.retry, player_max_retries=1, gm_artifact_attempts=2))
    assert reasoning_budget(changed.studio) == 3000
    assert reasoning_budget(replace(changed.studio, thinking_tokens=100)) == 100
    args = episode_args(changed)
    assert args.player_max_tokens == 35771
    assert args.protocol_retry["player_max_retries"] == 1
    assert args.protocol_retry["gm_artifact_attempts"] == 2
    assert changed.to_jsonable()["retry"] == args.protocol_retry


def test_studio_creation_passes_configured_optimizer(tmp_path):
    calls = []
    class RPC:
        async def acall(self, kind, body, *, request_id):
            calls.append((kind, body, request_id))
            return {"revision": 0}
    config = TrainConfig()
    backend = StudioBackend.__new__(StudioBackend)
    backend.config = replace(config, optimizer=replace(config.optimizer, beta2=0.98))
    backend.operation_prefix, backend.client = "config", RPC()
    from shellsmith.train.studio_initialization import InitializationJournal
    backend.capabilities = {"identity": {"model": "fixture"}}
    backend._acquired = False
    backend._initialization_target = backend._initialization_result = None
    backend.initializations = InitializationJournal(tmp_path / "run", "config", {})
    asyncio.run(backend.initialize())
    assert calls[0][1]["optimizer"] == {"beta1": 0.9, "beta2": 0.98, "epsilon": 1e-8}


@pytest.mark.parametrize("section,key,value", [
    ("optimizer", "beta1", 1.0), ("optimizer", "epsilon", 0.0),
    ("credit", "advantage_std_floor", 0.0), ("credit", "credit_coefficient", -0.1),
    ("retry", "thinking_anomaly_attempts", 0), ("retry", "player_max_retries", -1),
    ("studio", "reasoning_tokens_medium", 0),
])
def test_invalid_protocol_values_fail_before_execution(section, key, value):
    config = TrainConfig(run=RunSection(run_id="config"), compute=ComputeSection(backend="studio"))
    config = replace(config, **{section: replace(getattr(config, section), **{key: value})})
    with pytest.raises(ValueError):
        validate_train_config(config)


def test_service_toml_and_cli_precedence(tmp_path):
    settings = parse_args(["serve", "--config", str(ROOT / "examples/studio-service.toml"), "--head-rows", "64"])
    assert settings.head_rows == 64 and settings.segment_tokens == 2048
    assert settings.allocator_gib is settings.prefix_cache_gib is None
    assert settings.headroom_gib == 16 and settings.metal_cache_gib == 2
    file = tmp_path / "service.toml"
    file.write_text('[service]\nmodel="tiny"\nsegmant_tokens=32\n')
    with pytest.raises(ValueError, match="Unknown service settings"):
        parse_args(["serve", "--config", str(file)])
    file.write_text('[service]\nmodel="tiny"\nsegment_tokens=true\n')
    with pytest.raises(ValueError, match="invalid type"):
        parse_args(["serve", "--config", str(file)])
    with pytest.raises(SystemExit):
        parse_args(["serve", "--model", "tiny", "--prefill-tokens", "4096"])


def test_optimizer_create_request_validates_portable_settings():
    caps = {"context_window": 262144, "vocabulary_size": 100}
    validate_request("create", {"optimizer": {"beta2": 0.98}}, caps)
    for settings in ({"beta1": True}, {"epsilon": 0}, {"unknown": 1}, {"beta2": 1}):
        with pytest.raises(ValueError, match="optimizer"):
            validate_request("create", {"optimizer": settings}, caps)
