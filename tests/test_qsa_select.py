"""
Purpose: Check the Metal top-k block selection against MLX argpartition.
Input: Random FP32 score rows, some with masked (-inf) tails.
Output: The same selected set for each row, in ascending order.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from studio_compute.numerics.qsa_select import top_blocks


@pytest.mark.parametrize(("rows", "blocks", "top_k"),
                         [(64, 12288, 512), (33, 700, 512), (17, 512, 512), (5, 4100, 7)])
def test_top_blocks_matches_argpartition(rows: int, blocks: int, top_k: int) -> None:
    mx.random.seed(rows + blocks)
    scores = mx.random.normal((rows, blocks))
    visible = mx.random.randint(top_k // 2, blocks + 1, (rows,))
    scores = mx.where(mx.arange(blocks)[None] < visible[:, None], scores, -mx.inf)
    expected = mx.sort(mx.argpartition(scores, kth=-top_k, axis=-1)[:, -top_k:], axis=-1).astype(mx.int32)
    actual = top_blocks(scores, top_k)
    assert actual.shape == (rows, top_k)
    assert bool(mx.all(actual[:, 1:] > actual[:, :-1]).item())
    # Rows with fewer visible blocks than top_k tie at -inf; compare the visible part.
    keep = mx.arange(top_k)[None] < mx.minimum(visible, top_k)[:, None]
    expected_visible = mx.where(mx.take_along_axis(scores, expected, axis=-1) > -mx.inf,
                                expected, -1)
    actual_visible = mx.where(mx.take_along_axis(scores, actual, axis=-1) > -mx.inf,
                              actual, -1)
    assert bool(mx.array_equal(mx.sort(expected_visible, axis=-1),
                               mx.sort(actual_visible, axis=-1)).item())
    assert int(mx.sum(actual_visible >= 0).item()) == int(mx.sum(keep).item())


@pytest.mark.parametrize(("length", "blocks", "width"), [(64, 3000, 3000), (37, 300, 512), (8, 33, 512)])
def test_block_scores_match_the_fp32_formula(length: int, blocks: int, width: int) -> None:
    import math

    from studio_compute.numerics.qsa_select import block_scores

    mx.random.seed(length + blocks)
    query = mx.random.normal((4, length, 128)).astype(mx.bfloat16)
    pooled = mx.random.normal((blocks, 128)).astype(mx.bfloat16)
    complete = mx.random.randint(0, blocks + 1, (length,)).astype(mx.int32)
    product = query.astype(mx.float32) @ pooled.astype(mx.float32).T
    expected = mx.sum(mx.maximum(product, 0), axis=0) / math.sqrt(128)
    expected = mx.where(mx.arange(blocks)[None] < complete[:, None], expected, -mx.inf)
    expected = mx.pad(expected, [(0, 0), (0, width - blocks)], constant_values=-mx.inf)
    actual = block_scores(query, pooled, complete, width=width)
    assert actual.shape == (length, width)
    finite = expected > -mx.inf
    assert bool(mx.array_equal(actual > -mx.inf, finite).item())
    error = mx.max(mx.abs(mx.where(finite, actual - expected, 0.0)))
    assert float(error.item()) <= 1e-4 * float(mx.max(mx.abs(mx.where(finite, expected, 0.0))).item())
