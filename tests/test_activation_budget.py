"""Verify deterministic retention costs, alias ownership, and mandatory admission."""

import pytest
from studio_compute.activation_budget import (
    RetentionGroup,
    activation_budget,
    select_residuals,
)


def test_unique_allocation_benefit_and_stable_selection():
    groups = [
        RetentionGroup("slow", (("a", 10),), 100),
        RetentionGroup("alias", (("a", 10),), 20),
        RetentionGroup("cheap", (("b", 10),), 30),
        RetentionGroup("resident", (("existing", 99),), 1),
    ]
    plan = select_residuals(groups, 10, resident_allocations=[("existing", 99)])
    assert plan.selected == frozenset(("slow", "alias", "resident"))
    assert plan.used_bytes == 10
    assert (
        select_residuals(reversed(groups), 10, resident_allocations=[("existing", 99)])
        == plan
    )


def test_overlapping_groups_must_share_the_complete_allocation_set():
    with pytest.raises(ValueError, match="Overlapping"):
        select_residuals(
            [
                RetentionGroup("a", (("a", 1), ("b", 2)), 1),
                RetentionGroup("b", (("a", 1),), 1),
            ],
            3,
        )


def test_budget_reserves_recommended_limit_and_safety_margin():
    facts = dict(
        allocator_bytes=200,
        recommended_bytes=180,
        resident_bytes=40,
        checkpoint_bytes=30,
        scratch_bytes=20,
        safety_bytes=16,
    )
    assert activation_budget(**facts, requested_gib=None) == 74
    with pytest.raises(MemoryError):
        activation_budget(**facts, requested_gib=1)
    facts["checkpoint_bytes"] = 120
    with pytest.raises(MemoryError):
        activation_budget(**facts, requested_gib=None)


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_cli_rejects_invalid_activation_budget(value):
    from studio_compute.cli import parse_args

    with pytest.raises(SystemExit):
        parse_args(["serve", "--model", "tiny", "--activation-budget-gib", value])


def test_cli_preserves_legacy_and_defaults_bf16_to_automatic_admission():
    from studio_compute.cli import parse_args

    legacy = parse_args(["serve", "--model", "tiny", "--execution-profile", "legacy"])
    bf16 = parse_args(["serve", "--model", "tiny", "--execution-profile", "bf16-fast"])
    explicit = parse_args(
        [
            "serve",
            "--model",
            "tiny",
            "--execution-profile",
            "bf16-fast",
            "--checkpoint-layers",
            "4",
            "--activation-budget-gib",
            "12",
        ]
    )
    default = parse_args(["serve", "--model", "tiny"])
    assert (
        default.execution_profile == "bf16-fast" and default.checkpoint_layers is None
    )
    assert legacy.checkpoint_layers == 8
    assert bf16.checkpoint_layers is None and bf16.activation_budget_gib is None
    assert explicit.checkpoint_layers == 4 and explicit.activation_budget_gib == 12


@pytest.mark.parametrize(
    "override,requested,expected", [(None, None, 2), (None, 50, 3), (1, None, None)]
)
def test_engine_chooses_smallest_admitted_interval(
    monkeypatch, override, requested, expected
):
    """
    Purpose: Verify automatic interval selection and explicit insufficient-memory rejection.
    Input: Conservative interval-dependent mandatory bytes. Output: The smallest valid interval.
    """
    import sys
    from types import SimpleNamespace, ModuleType
    from studio_compute.engine import ResidentEngine

    gib = 1 << 30
    retention = ModuleType("studio_compute.numerics.retention")
    retention.mandatory_saved_bytes = lambda *_: 5 * gib
    monkeypatch.setitem(sys.modules, "studio_compute.numerics.retention", retention)
    engine = ResidentEngine.__new__(ResidentEngine)
    engine.checkpoint_override = override
    engine.activation_budget_gib = requested
    engine.saved_activation_backward = True
    engine.lm = SimpleNamespace(layers=[None] * 4)
    engine.mx = SimpleNamespace(get_active_memory=lambda: 30 * gib)
    engine.persistent_bytes = lambda: 30 * gib
    engine.reserved = {}
    engine.capabilities = {"recommended_working_set_bytes": 180 * gib}
    engine.segment_tokens, engine.head_rows = 2048, 128
    engine.resident = SimpleNamespace(
        budget=200 * gib,
        estimate_replay=lambda group, interval, **_: {
            "additional_bytes": 180 * gib // interval
        },
    )
    groups = {(1,): [{"prompt_token_ids": [1], "sampled_tokens": [2]}]}
    if expected is None:
        with pytest.raises(MemoryError):
            engine._saved_admission(groups, gib)
    else:
        interval, peak, budgets = engine._saved_admission(groups, gib)
        assert interval == expected
        assert peak + 30 * gib <= 180 * gib
        assert peak == (180 * gib // interval + 5 * gib
                        + budgets[(1,)] + 2 * gib + 16 * gib)
        assert budgets[(1,)] == (37 * gib if requested is None else requested * gib)


@pytest.mark.parametrize("allocator_gib,requested_gib", [(100, None), (100, 5), (80, None)])
def test_saved_replay_reservation_protects_interleaved_inference(
    monkeypatch, allocator_gib, requested_gib
):
    """
    Purpose: Keep replay scratch and safety bytes unavailable to another request.
    Input: An admitted BF16 replay and a concurrent inference request.
    Output: Admission rejects a request that would consume protected headroom.
    """
    import sys
    from types import MethodType, ModuleType, SimpleNamespace
    from studio_compute.model import Qwen27Model
    from test_studio_cache_memory import fixture_engine

    gib = 1 << 30
    retention = ModuleType("studio_compute.numerics.retention")
    retention.mandatory_saved_bytes = lambda *_: 5 * gib
    monkeypatch.setitem(sys.modules, "studio_compute.numerics.retention", retention)
    engine = fixture_engine()
    engine.execution_profile = "bf16-fast"
    engine.checkpoint_override = 1
    engine.activation_budget_gib = requested_gib
    engine.saved_activation_backward = True
    engine.lm = SimpleNamespace(layers=[None])
    engine.mx.get_active_memory = lambda: 30 * gib
    engine.persistent_bytes = lambda: 30 * gib
    engine.capabilities = {"recommended_working_set_bytes": 90 * gib}
    engine.segment_tokens, engine.head_rows = 2048, 128
    engine.resident.budget = allocator_gib * gib
    engine.resident.mx = engine.mx
    engine.resident.admit = MethodType(Qwen27Model.admit, engine.resident)
    engine.resident.estimate_replay = lambda *_, **__: {
        "additional_bytes": 20 * gib
    }
    groups = {(1,): [{"prompt_token_ids": [1], "sampled_tokens": [2]}]}
    _, peak, _ = engine._saved_admission(groups, 2 * gib)
    engine.admit(peak)
    engine.reserved["yielded-replay"] = peak
    available_bytes = min(allocator_gib, 90) * gib - 30 * gib - peak
    engine.admit(available_bytes)
    with pytest.raises(MemoryError, match="Admission needs"):
        engine.admit(available_bytes + 1)
