"""CPU lifecycle probes: admission must precede allocation and mutation."""

import json
import sys
from types import SimpleNamespace

import pytest

from test_studio_cache_memory import Array, fixture_engine, put


class Optimizer:
    def __init__(self):
        self.master = {"a": Array(4)}
        self.first = {"a": Array(4)}
        self.second = {"a": Array(4)}
        self.accumulated = {"a": Array(4)}
        self.steps = self.step_calls = self.zero_calls = 0

    def step(self, **kwargs):
        self.step_calls += 1
        self.steps += 1
        self.accumulated = {}
        return 1.0

    def zero(self):
        self.zero_calls += 1
        self.accumulated = {}


def engine_with_client(budget=100):
    engine = fixture_engine()
    engine.resident.budget = budget
    def admit(additional, reserved, persistent):
        if max(engine.mx.get_active_memory() + additional,
               persistent + reserved + additional) > engine.resident.budget:
            raise MemoryError("budget")
    engine.resident.admit = admit
    factors = {"a": Array(2)}  # BF16; corresponding FP32 tensors are 4 bytes.
    optimizer = Optimizer()
    engine.clients = {"a": {"adapter": SimpleNamespace(state=lambda: factors),
                            "optimizer": optimizer, "backward": "back"}}
    engine.idle_factors = factors
    engine.resident.selected = None
    engine.resident.select = lambda factors: setattr(engine.resident, "selected", factors)
    engine.metrics = lambda started: {}
    engine.checkpoint_layers, engine.segment_tokens, engine.head_rows = 1, 1, 1
    engine.replay_layout, engine.replay_ops = "prompt_groups", None
    engine.resident.estimate_replay = lambda *a, **k: {"additional_bytes": 1}
    return engine, optimizer


def drain(work):
    while True:
        try:
            next(work)
        except StopIteration as done:
            return done.value


def test_optimizer_evicts_cache_before_step_and_consumes_held_reservation():
    engine, optimizer = engine_with_client()
    put(engine, 1, 20)
    engine.reserved[("optimizer", "a")] = engine.optimizer_step_bytes("a")
    result = drain(engine.work("a", "optim_step", {"backward_operation": "back", "learning_rate": 0.01}))
    assert engine.prefixes.bytes <= 12
    assert optimizer.step_calls == optimizer.steps == 1
    assert result["optim_metrics"]["optimizer_steps"] == 1
    assert engine.clients["a"]["backward"] is None
    assert engine.reserved == {}


def test_bf16_optimizer_reserves_fused_replacement_tensors() -> None:
    engine, optimizer = engine_with_client()
    engine.execution_profile = "bf16-fast"
    optimizer.master = {"a": Array(200), "b": Array(100)}
    optimizer.first = {"a": Array(200), "b": Array(100)}
    optimizer.second = {"a": Array(200), "b": Array(100)}
    assert engine.optimizer_step_bytes("a") == 900 + 400 + 4096


def test_recomputed_model_reserves_no_unused_saved_activation_cap(monkeypatch) -> None:
    """
    Purpose: Do not reserve optional tapes for a model that cannot retain tapes.
    Input: A configured 16 GiB cap and insufficient space for unused tapes.
    Output: The recompute path admits its real work with zero optional tape bytes.
    """
    engine, _ = engine_with_client()
    engine.resident.budget = 32 << 30
    engine.capabilities = {"recommended_working_set_bytes": 32 << 30}
    engine.activation_budget_gib = 16
    engine.saved_activation_backward = False
    engine.checkpoint_override = 1
    engine.lm = SimpleNamespace(layers=[object()])
    engine.resident.estimate_replay = lambda *args, **kwargs: {"additional_bytes": 1 << 30}
    monkeypatch.setitem(sys.modules, "studio_compute.numerics.retention",
                        SimpleNamespace(mandatory_saved_bytes=lambda *args: pytest.fail("No tape exists")))
    group = [{"prompt_token_ids": [1], "sampled_tokens": [2]}]
    _, _, budgets = engine._saved_admission({(1,): group}, future_gradient_bytes=0)
    assert budgets == {(1,): 0}


def test_optimizer_rejection_preserves_gradients_counter_and_ownership():
    engine, optimizer = engine_with_client(budget=80)
    gradients = optimizer.accumulated
    held = engine.optimizer_step_bytes("a")
    engine.reserved[("optimizer", "a")] = held
    put(engine, 1, 10)
    with pytest.raises(MemoryError):
        drain(engine.work("a", "optim_step", {"backward_operation": "back", "learning_rate": 0.01}))
    assert optimizer.accumulated is gradients
    assert optimizer.step_calls == optimizer.steps == 0
    assert engine.clients["a"]["backward"] == "back"
    assert engine.reserved == {("optimizer", "a"): held}


@pytest.mark.parametrize("release", ["abort", "close"])
def test_pending_optimizer_peak_protects_interleaved_clients_until_released(release):
    engine, optimizer = engine_with_client()
    engine.reserved[("optimizer", "a")] = engine.optimizer_step_bytes("a")
    with pytest.raises(MemoryError):
        engine.admit(20)  # Another client cannot consume the pending step's peak.
    engine.abort("a", "unrelated-operation")
    assert engine.reserved and optimizer.zero_calls == 0
    if release == "abort":
        engine.abort("a", "back")
    else:
        engine.close_client("a")
    assert not engine.reserved
    engine.admit(20)


@pytest.mark.parametrize("context_only", [False, True])
def test_backward_rejects_unaffordable_following_step_before_mutation(context_only):
    engine, optimizer = engine_with_client(budget=90)
    engine.clients["a"]["backward"] = None
    gradients = optimizer.accumulated
    sequence = {"prompt_token_ids": [1], "sampled_tokens": [2]}
    if context_only:
        sequence.update(loss_mask=[0.0], source_spans=[("zero", 1, 2)])
    with pytest.raises(MemoryError):
        drain(engine.work("a", "forward_backward", {"operation_id": "new-back", "sequences": [
            sequence]}))
    assert optimizer.zero_calls == 0 and optimizer.accumulated is gradients
    assert engine.clients["a"]["backward"] is None
    assert not engine.reserved


@pytest.mark.parametrize("live_prompt", [None, [1], [2]])
def test_replay_admission_excludes_context_only_paths(live_prompt: list[int] | None) -> None:
    """
    Purpose: Admit an update when only a skipped path exceeds the replay budget.
    Input: A context-only path, with an optional live path in either prompt group.
    Output: Admission reaches the allocation boundary with only live replay costs.
    """
    engine, optimizer = engine_with_client()
    engine.clients["a"]["backward"] = None
    sequences = [{
        "prompt_token_ids": [1],
        "sampled_tokens": list(range(20)),
        "loss_mask": [0.0] * 20,
        "source_spans": [("zero", 1, 21)],
    }]
    if live_prompt is not None:
        sequences.append({
            "prompt_token_ids": live_prompt,
            "sampled_tokens": [3],
            "loss_mask": [1.0],
            "source_spans": [("live", 1, 2)],
        })
    estimates = []

    def estimate(group: list[dict], checkpoint_layers: int, **options: int) -> dict[str, int]:
        estimates.append(group)
        return {"additional_bytes": 10 * sum(len(s["sampled_tokens"]) for s in group)}

    class AllocationProbe(Exception):
        pass

    def stop_before_allocation() -> None:
        raise AllocationProbe

    engine.resident.estimate_replay = estimate
    optimizer.zero = stop_before_allocation
    with pytest.raises(AllocationProbe):
        drain(engine.work("a", "forward_backward", {
            "operation_id": "new-back", "sequences": sequences,
        }))
    assert estimates == ([] if live_prompt is None else [[sequences[1]]])
    assert engine.clients["a"]["backward"] is None
    assert not engine.reserved


@pytest.mark.parametrize("kind, packed", [
    ("forward", True), ("forward_backward", False),
])
def test_executed_masked_paths_still_require_replay_admission(kind: str, packed: bool) -> None:
    """
    Purpose: Keep memory admission for masked paths that still execute the model.
    Input: A packed forward request or a legacy backward request above the budget.
    Output: Admission rejects the request before optimizer state changes.
    """
    engine, optimizer = engine_with_client()
    engine.clients["a"]["backward"] = None
    sequence = {
        "prompt_token_ids": [1], "sampled_tokens": [2], "loss_mask": [0.0],
    }
    if packed:
        sequence["source_spans"] = [("zero", 1, 2)]
    engine.resident.estimate_replay = lambda *a, **k: {"additional_bytes": 200}
    with pytest.raises(MemoryError):
        drain(engine.work("a", kind, {
            "operation_id": "new-back", "sequences": [sequence],
        }))
    assert optimizer.zero_calls == 0
    assert engine.clients["a"]["backward"] is None
    assert not engine.reserved


@pytest.mark.parametrize("cold", [True, False])
def test_restore_rejects_transient_peak_before_create_or_load(tmp_path, cold):
    engine, optimizer = engine_with_client(budget=70)
    held = engine.optimizer_step_bytes("a")
    if cold:
        engine.clients.clear()
    else:
        engine.reserved[("optimizer", "a")] = held
    engine.create = lambda *a: pytest.fail("Admission must precede blank allocation")
    engine.mx.load = lambda *a: pytest.fail("Admission must precede tensor loading")
    with pytest.raises(MemoryError):
        engine.restore("a", tmp_path, {"seed": 0})
    assert optimizer.step_calls == optimizer.zero_calls == 0
    assert engine.reserved == ({} if cold else {("optimizer", "a"): held})
    assert ("a" in engine.clients) is not cold


def test_cold_restore_evicts_cache_before_blank_client_creation(tmp_path):
    engine, _ = engine_with_client()
    engine.clients.clear()
    put(engine, 1, 20)
    (tmp_path / "optimizer.json").write_text(json.dumps({"learning_rate": 0.01,
                                                       "beta1": 0.9, "beta2": 0.95, "eps": 1e-8}))
    class AllocationProbe(Exception):
        pass
    def create(*args):
        assert engine.prefixes.bytes <= 10
        raise AllocationProbe
    engine.create = create
    engine.mx.load = lambda *a: pytest.fail("Probe stops before native loading")
    with pytest.raises(AllocationProbe):
        engine.restore("a", tmp_path, {"seed": 0})
    assert not engine.clients and not engine.reserved
