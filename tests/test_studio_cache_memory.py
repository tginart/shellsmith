"""Memory-pressure behavior; numerical cache correctness has native fixtures."""

from collections import OrderedDict
from types import SimpleNamespace

import pytest

from studio_compute.cache import PrefixCache
from studio_compute.cli import cache_limit
from studio_compute.runtime import RuntimeEngine


class Array:
    def __init__(self, size):
        self.nbytes = size


def fixture_engine():
    engine = RuntimeEngine.__new__(RuntimeEngine)
    engine.reserved = {}
    engine.policies, engine.policy_users = OrderedDict(), {}
    engine.working = 0
    engine.mx = SimpleNamespace(
        array=Array,
        stop_gradient=lambda x: x,
        clear_cache=lambda: None,
        get_active_memory=lambda: 30 + engine.prefixes.bytes + engine.working,
    )
    engine.prefixes = PrefixCache(engine.mx, 100, capacity=engine._cache_capacity)
    engine.prefixes._copy = lambda x: list(x)
    engine.persistent_bytes = lambda: 30 + engine.prefixes.bytes
    engine.identity = {"model": "m"}
    engine.prefix_identity = "m"
    engine.execution_profile = "legacy"

    def admit(additional, reserved, persistent):
        if max(engine.mx.get_active_memory() + additional, persistent + reserved + additional) > 100:
            raise MemoryError("budget")

    engine.resident = SimpleNamespace(
        budget=100, admit=admit,
        estimate_inference=lambda length, batch=1, **options: length * batch,
    )
    return engine


def put(engine, token, size):
    cache = [SimpleNamespace(state=[Array(size)])]
    engine.prefixes.put("m", "p", "bf16", [token], cache, Array(0))


def test_auto_cache_uses_spare_memory_and_yields_to_training():
    engine = fixture_engine()
    for token in range(1, 8):
        put(engine, token, 10)
    assert engine.prefixes.bytes == 60  # Leave room for a detached copy.
    # Actual allocations leave no room; admission evicts reusable snapshots,
    # then admits replay without touching weights or active work.
    engine.admit(40)
    assert engine.prefixes.bytes == 30
    assert engine.prefixes.metrics()["evictions"] == 4
    assert engine.prefixes.contains("m", "p", "bf16", [7, 8])
    assert not engine.prefixes.contains("m", "p", "bf16", [1])


def test_cache_respects_live_replay_reservation_and_clone_peak():
    engine = fixture_engine()
    engine.reserved["replay"] = 40
    for token in range(1, 5):
        put(engine, token, 10)
    assert engine.prefixes.bytes == 30
    engine.working = 25
    # A copy must fit alongside the working replay, even when its permanent
    # footprint fits the logical reservation accounting.
    put(engine, 5, 20)
    assert engine.prefixes.bytes <= 30
    assert engine.mx.get_active_memory() <= engine.resident.budget
    engine.admit(30)
    assert engine.prefixes.bytes == 0
    assert engine.reserved == {"replay": 40} and engine.working == 25


def test_cache_does_not_hide_oversized_active_work():
    engine = fixture_engine()
    put(engine, 1, 10)
    with pytest.raises(MemoryError, match="budget"):
        engine.admit(80)
    assert engine.prefixes.bytes == 0


def test_warm_requests_preserve_cache_and_cold_batches_fit_memory():
    engine = fixture_engine()
    body = {"policy": "p", "prompt_token_ids": [1], "max_tokens": 9}
    put(engine, 1, 10)
    assert engine.inference_batch_capacity(body, 4) == 4
    cold = {**body, "prompt_token_ids": [2]}
    assert engine.inference_batch_capacity(cold, 4) == 4
    engine.reserved["replay"] = 45
    assert engine.inference_batch_capacity(cold, 4) == 2
    assert engine.prefixes.bytes == 10  # Capacity inspection never evicts.


def test_cache_identity_and_telemetry():
    engine = fixture_engine()
    put(engine, 1, 10)
    assert engine.prefixes.get("m", "other-policy", "bf16", [1]) is None
    assert engine.prefixes.get("other-model", "p", "bf16", [1]) is None
    assert engine.prefixes.get("m", "p", "8bit", [1]) is None
    assert engine.prefixes.get("m", "p", "bf16", [1, 2])[2] == 1
    assert engine.prefixes.metrics()["hits"] == 1
    assert engine.prefixes.metrics()["misses"] == 3
    assert engine.prefixes.metrics()["hit_tokens"] == 1


def test_cache_disabled_and_explicit_cap():
    engine = fixture_engine()
    engine.prefixes.limit = 0
    put(engine, 1, 10)
    assert engine.prefixes.bytes == 0
    engine.prefixes.limit = 20
    put(engine, 1, 10)
    put(engine, 2, 10)
    put(engine, 3, 10)
    assert engine.prefixes.bytes == 20
    assert not engine.prefixes.contains("m", "p", "bf16", [1])


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "bogus"])
def test_invalid_cache_cli_cap(value):
    from argparse import ArgumentTypeError
    with pytest.raises(ArgumentTypeError):
        cache_limit(value)


def test_cache_cli_auto_and_cap():
    assert cache_limit("auto") is None
    assert cache_limit("0") == 0
    assert cache_limit("128") == 128


def test_bf16_cache_eviction_and_batch_capacity_use_recommended_limit():
    """
    Purpose: Apply the recommended cap before cache or inference batch admission.
    Input: An allocator cap above the recommended cap and a yielded replay.
    Output: Cache eviction and batch size preserve the same shared byte cap.
    """
    engine = fixture_engine()
    engine.execution_profile = "bf16-fast"
    engine.capabilities = {"recommended_working_set_bytes": 80}
    engine.reserved["replay"] = 20
    put(engine, 1, 10)
    cold = {"policy": "p", "prompt_token_ids": [2], "max_tokens": 9}
    assert engine.inference_batch_capacity(cold, 4) == 3
    engine.admit(30)
    assert engine.prefixes.bytes == 0
    assert engine.reserved == {"replay": 20}


def test_bf16_admission_checks_active_bytes_against_recommended_limit():
    """Reject live allocations above the cap even if logical reservations fit."""
    engine = fixture_engine()
    engine.execution_profile = "bf16-fast"
    engine.capabilities = {"recommended_working_set_bytes": 80}
    engine.working = 60
    with pytest.raises(MemoryError, match="Admission needs"):
        engine.admit(0)


def test_legacy_admission_keeps_its_allocator_limit():
    engine = fixture_engine()
    engine.capabilities = {"recommended_working_set_bytes": 80}
    engine.admit(70)
