"""Purpose: Reject unsupported distribution probes before costly model allocation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def benchmark_module():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/benchmark_studio_continuous.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_studio_continuous", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("target_length", [1, 2, 8])
def test_short_distribution_targets_fail_before_engine_access(
    benchmark_module, target_length
):
    conversations = [[{"target_token_ids": [1] * target_length}]]
    with pytest.raises(ValueError, match="at least nine completion tokens"):
        benchmark_module.verify_distributions(object(), conversations, concurrency=1)


def test_distribution_validation_only_checks_selected_first_turns(benchmark_module):
    conversations = [
        [{"target_token_ids": [1] * 9}, {"target_token_ids": [1]}],
        [{"target_token_ids": [1]}],
    ]
    benchmark_module.validate_distribution_targets(conversations, concurrency=1)
    with pytest.raises(ValueError, match="at least nine completion tokens"):
        benchmark_module.validate_distribution_targets(conversations, concurrency=2)


def test_short_loaded_targets_fail_before_model_import(benchmark_module, monkeypatch):
    monkeypatch.setattr(
        benchmark_module,
        "load_conversations",
        lambda *args, **kwargs: [[{"target_token_ids": [1] * 8}]],
    )
    args = SimpleNamespace(
        records_root=Path("unused"),
        sessions=1,
        turns=1,
        max_completion_tokens=8,
        distribution_parity=True,
        concurrency=[1],
    )
    with pytest.raises(ValueError, match="at least nine completion tokens"):
        benchmark_module.benchmark(args)


def test_nine_target_distribution_probe_completes_on_metal(benchmark_module):
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_vlm")
    from test_studio_numerics import TinyResident

    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        "tiny", model_factory=TinyResident, checkpoint_layers=2, prefill_tokens=3
    )
    try:
        engine.policies["benchmark"] = {"factors": engine.idle_factors, "revision": 0}
        conversations = [
            [
                {
                    "policy": "benchmark",
                    "revision": 0,
                    "seed": 17,
                    "prompt_token_ids": list(range(1, length + 1)),
                    "target_token_ids": list(range(3, 12)),
                    "max_tokens": 9,
                }
            ]
            for length in (3, 8)
        ]
        result = benchmark_module.verify_distributions(
            engine, conversations, concurrency=2
        )
        assert result["rows"] == 2 and result["decode_steps"] == 8
        assert not engine.reserved and not engine.policy_users
    finally:
        engine.close()
