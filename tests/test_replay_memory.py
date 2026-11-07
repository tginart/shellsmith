"""Verify lazy balanced sums, the trie gradient-copy model, and the replay guard."""

import random
from types import SimpleNamespace

import pytest

from studio_compute.numerics.sums import BalancedSums

GIB = 1 << 30


def _eager_balanced_sums(values: list[float]) -> list[float]:
    """Return every running total of the previous eager BalancedSums.add."""
    levels, totals = [], []
    for value in values:
        level = 0
        while level < len(levels) and levels[level] is not None:
            value = levels[level] + value
            levels[level] = None
            level += 1
        if level == len(levels):
            levels.append(value)
        else:
            levels[level] = value
        parts = [part for part in reversed(levels) if part is not None]
        total = parts[0]
        for part in parts[1:]:
            total = total + part
        totals.append(total)
    return totals


def test_lazy_total_is_bitwise_the_eager_running_total() -> None:
    """
    Purpose: Make sure that a lazy total uses the same additions in the same order.
    Input: Random floats, where the addition order changes the rounded result.
    Output: Each lazy total is exactly equal to the previous eager total.
    """
    generator = random.Random(7)
    values = [generator.uniform(-1, 1) * 10 ** generator.randint(-8, 8) for _ in range(70)]
    sums = BalancedSums()
    for value, expected in zip(values, _eager_balanced_sums(values), strict=True):
        sums.add("key", value)
        assert sums.total("key") == expected
    assert sums.pop("key") == _eager_balanced_sums(values)[-1]
    assert "key" not in sums and sums.arrays() == []


@pytest.mark.parametrize("max_levels", [1, 2, 3, 4])
def test_level_cap_bounds_partials_and_keeps_the_sum(max_levels: int) -> None:
    sums = BalancedSums(max_levels)
    for count in range(1, 100):
        sums.add("key", 1)
        assert sums.partial_count("key") <= max_levels
        assert sums.total("key") == count


def test_tree_copy_model_counts_node_and_tree_partials() -> None:
    """
    Purpose: Count the gradient partials of one shared node and its two leaves.
    Input: Two branches that share two segments, then diverge for three segments.
    Output: The exact copy count, and a smaller count with a level cap.
    """
    pytest.importorskip("mlx.core")
    from studio_compute.numerics.replay import tree_gradient_copies

    shared = list(range(4))
    branches = [shared + [10, 11, 12, 13, 14, 15], shared + [20, 21, 22, 23, 24, 25]]
    # Leaves: 3 segments each (peaks 2 + 0, then 2 + 1). Root: 2 segments + 1 tree partial.
    assert tree_gradient_copies(branches, 2) == 3
    assert tree_gradient_copies(branches, 2, max_levels=1) == 2


def test_replay_guard_stops_above_the_cap_and_at_low_host_memory(monkeypatch) -> None:
    from studio_compute import engine as engine_module
    from studio_compute.engine import ResidentEngine

    active = {"bytes": 100 * GIB}
    engine = ResidentEngine.__new__(ResidentEngine)
    engine.execution_profile = "bf16-fast"
    engine.capabilities = {"recommended_working_set_bytes": 200 * GIB}
    engine.resident = SimpleNamespace(budget=220 * GIB)
    engine.mx = SimpleNamespace(get_active_memory=lambda: active["bytes"])
    monkeypatch.setattr(engine_module, "_memory_level", lambda: 50)
    checked_s = engine._check_replay_memory(0.0)
    assert checked_s > 0
    assert engine._check_replay_memory(checked_s) == checked_s
    monkeypatch.setattr(engine_module, "_memory_level", lambda: 3)
    with pytest.raises(MemoryError, match="memory level 3"):
        engine._check_replay_memory(0.0)
    active["bytes"] = 200 * GIB + 1
    with pytest.raises(MemoryError, match="exceeds"):
        engine._check_replay_memory(checked_s)


def test_tree_admission_caps_sum_levels_only_when_exact_sums_do_not_fit() -> None:
    """
    Purpose: Prefer exact balanced sums, and cap the levels only under memory pressure.
    Input: A two-leaf trie, a 4 GiB gradient, and two allocator budgets.
    Output: The exact sums with the larger budget. A level cap with the smaller one.
    """
    pytest.importorskip("mlx.core")
    from studio_compute.engine import ResidentEngine

    shared = list(range(4))
    sequences = [
        {"prompt_token_ids": shared, "sampled_tokens": [10, 11, 12, 13, 14, 15, 16],
         "loss_mask": [0.0] * 7},
        {"prompt_token_ids": shared, "sampled_tokens": [20, 21, 22, 23, 24, 25, 26],
         "loss_mask": [0.0] * 7},
    ]
    engine = ResidentEngine.__new__(ResidentEngine)
    engine.checkpoint_override = None
    engine.activation_budget_gib = 0
    engine.saved_activation_backward = False
    engine.lm = SimpleNamespace(layers=[None] * 2,
                                args=SimpleNamespace(hidden_size=1, hc_count=1))
    engine.mx = SimpleNamespace(get_active_memory=lambda: 100 * GIB)
    engine.persistent_bytes = lambda: 100 * GIB
    engine.reserved = {}
    engine.segment_tokens, engine.head_rows = 2, 128
    engine.resident = SimpleNamespace(
        estimate_replay=lambda group, interval, **_: {"additional_bytes": GIB})
    for budget_gib, expected_levels, expected_copies in [(200, None, 3), (127, 1, 2)]:
        engine.capabilities = {"recommended_working_set_bytes": budget_gib * GIB}
        engine.resident.budget = budget_gib * GIB
        interval, peak, _, levels = engine._tree_admission(sequences, 4 * GIB)
        assert (interval, levels) == (1, expected_levels)
        assert peak == GIB + expected_copies * 4 * GIB + 16 * GIB
    engine.capabilities = {"recommended_working_set_bytes": 120 * GIB}
    with pytest.raises(MemoryError, match="No trie gradient sum level fits"):
        engine._tree_admission(sequences, 4 * GIB)


def test_segment_embeddings_match_sliced_branch_embeddings() -> None:
    """
    Purpose: Make sure that per-segment embedding gives the same trie replay.
    Input: A tiny Flash trie, with and without the segment_embeddings callable.
    Output: Bitwise equal logprobs and loss.
    """
    mx = pytest.importorskip("mlx.core")
    from test_flash_lora import branch, consume, tiny_flash
    from studio_compute.numerics.flash_ops import FlashOps
    from studio_compute.numerics.replay import Replay

    lm, _ = tiny_flash(indexer_budget=8)
    prefix = list(range(1, 11))

    def items(*, per_segment: bool) -> list:
        values = [branch(lm, prefix, [11, 12, 13, 14, 15]),
                  branch(lm, prefix + [11, 12], [20, 21, 22, 23, 24, 25])]
        for value in values:
            value["token_ids"] = value["input_ids"][0].tolist()
            if per_segment:
                ids_BL = value["input_ids"]
                value["segment_embeddings"] = lambda start, stop, ids_BL=ids_BL: mx.tile(
                    lm.model.embed_tokens(ids_BL[:, start:stop]), (1, 1, lm.args.hc_count))
        return values

    replay = Replay(SimpleNamespace(language_model=lm), segment_tokens=4,
                    checkpoint_layers=1, execution_profile="bf16-fast",
                    model_ops=FlashOps())
    sliced, sliced_loss, _, _ = consume(replay.run_tree(items(per_segment=False)))
    segmented, segmented_loss, _, _ = consume(replay.run_tree(items(per_segment=True)))
    for left, right in zip(sliced, segmented, strict=True):
        assert bool(mx.array_equal(left, right).item())
    assert float(sliced_loss.item()) == float(segmented_loss.item())
