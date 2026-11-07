"""Real parsed/validated MLX effort settings reach phase budgets and sampling."""

from dataclasses import replace
from pathlib import Path

import pytest

from shellsmith.mlx.generation import reasoning_budget, completion_budget
from shellsmith.mlx.sampler import MLXPlayerSampler
from shellsmith.train.backend import player_profile
from shellsmith.train.config import TrainConfig, RunSection, ComputeSection, load_train_config, validate_train_config
from test_mlx_sampler import FakeSession


@pytest.mark.parametrize("effort,limit", [("low", 2048), ("medium", 4096), ("xhigh", 8192)])
def test_parsed_mlx_effort_selects_budget_without_legacy_override(tmp_path, effort, limit):
    # 8K reasoning + 4K response requires more compaction headroom than medium.
    reserve = 16384 if effort == "xhigh" else 12288
    source = tmp_path / "effort.toml"
    source.write_text(f'[compute]\nbackend="mlx"\n[mlx]\nreasoning_effort="{effort}"\n'
                      f'compaction_reserve_tokens={reserve}\n')
    config = load_train_config(source)
    assert config.mlx.thinking_tokens == 0
    assert reasoning_budget(config.mlx) == limit
    assert completion_budget(config.mlx) == limit + 3 + 4096
    profile = player_profile(config)
    assert profile["resolved_reasoning_tokens"] == limit
    assert profile["thinking_renderer"] == f"qwen3_8_{effort}_reasoning"
    session = FakeSession([([10] * limit, "length"), ([11, 98], "stop")])
    session.settings = config.mlx
    record, _ = MLXPlayerSampler(session).sample({"reasoning_effort": effort}, request_id="effort", session_id="parsed")
    assert session.calls == [limit, 4096]
    assert record.protocol_row["reasoning_limit"] == limit
    assert record.protocol_row["response_limit"] == 4096


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_explicit_legacy_thinking_limit_still_overrides_effort(effort):
    config = TrainConfig(run=RunSection(run_id="legacy"), compute=ComputeSection(backend="mlx"))
    config = replace(config, mlx=replace(config.mlx, reasoning_effort=effort, thinking_tokens=1024))
    validate_train_config(config)
    assert reasoning_budget(config.mlx) == 1024
    session = FakeSession([([10] * 1024, "length"), ([11, 98], "stop")])
    session.settings = config.mlx
    MLXPlayerSampler(session).sample({"reasoning_effort": effort}, request_id="override", session_id="legacy")
    assert session.calls == [1024, 4096]


def test_xhigh_reserve_and_legacy_total_caps_fail_explicitly_before_execution():
    base = TrainConfig(run=RunSection(run_id="xhigh"), compute=ComputeSection(backend="mlx"))
    config = replace(base, mlx=replace(base.mlx, reasoning_effort="xhigh"))
    with pytest.raises(ValueError, match="compaction reserve must exceed completion budget"):
        validate_train_config(config)
    config = replace(config, mlx=replace(config.mlx, compaction_reserve_tokens=16384))
    validate_train_config(config)
    with pytest.raises(ValueError, match="resolved thinking.*explicit max_tokens"):
        validate_train_config(replace(config, mlx=replace(config.mlx, max_tokens=8192)))
    with pytest.raises(ValueError, match="thinking_tokens.*nonnegative"):
        validate_train_config(replace(config, mlx=replace(config.mlx, thinking_tokens=-1)))


@pytest.mark.parametrize("name", ["mlx-studio.toml", "mlx-studio-qualification-train.toml", "mlx-studio-qualification-eval.toml"])
def test_historical_example_limits_remain_explicit(name):
    config = load_train_config(Path(__file__).parents[1] / "examples" / name,
                               campaign_mode="inference" if name.endswith("eval.toml") else "mlx_train")
    assert config.mlx.thinking_tokens == reasoning_budget(config.mlx) == 4096
    assert config.mlx.max_tokens == completion_budget(config.mlx) == 8192


@pytest.mark.parametrize("backend", ["mlx", "studio"])
def test_positive_legacy_total_ceiling_includes_forced_ids_and_reports_effective_limit(backend):
    config = TrainConfig(run=RunSection(run_id="legacy-total"), compute=ComputeSection(backend=backend))
    settings = replace(getattr(config, backend), thinking_tokens=4096, response_tokens=4096, max_tokens=8192)
    config = replace(config, **{backend: settings})
    validate_train_config(config)
    assert completion_budget(settings) == 8192
    session = FakeSession([([10] * 4096, "length"), ([11] * 4093, "length")])
    session.settings = settings
    record, _ = MLXPlayerSampler(session).sample({"reasoning_effort": "medium"}, request_id="legacy", session_id="total")
    assert session.calls == [4096, 4093]
    assert len(record.sampled_tokens) == 8192
    assert record.forced_close_token_span == (4096, 4099)
    assert record.protocol_row["response_limit"] == 4093
    assert {notice["code"] for notice in record.protocol_row["generation_notices"]} == {
        "reasoning_limit_exceeded", "response_limit_exceeded"}
