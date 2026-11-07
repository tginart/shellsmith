"""
Purpose: Verify fused D256 attention against independent FP64 GQA equations.
Input: BF16 tensors, suffix masks, short tails, and arbitrary causal offsets.
Output: Forward and all derivative bounds, including the custom VJP.
"""

from __future__ import annotations

from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
from studio_compute.numerics.attention_metal import (
    attention_backward,
    attention_forward,
    fused_attention,
)


@pytest.mark.parametrize(
    "queries,keys,offset",
    [
        (1, 1, 0),
        (1, 17, 16),
        (8, 41, 33),
        (17, 41, 24),
        (194, 194, 0),
        (33, 47, 7),
    ],
)
def test_attention_all_derivatives(
    monkeypatch, queries: int, keys: int, offset: int
) -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from mlx_attention_qualify import numpy_attention

    mx.random.seed(23)
    q = mx.random.normal((1, 24, queries, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, keys, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    dy = mx.random.normal(q.shape).astype(mx.bfloat16)
    output, tape = attention_forward(q, k, v, 0.0625, offset)
    gradients = attention_backward(tape, dy)
    expected = numpy_attention(
        *[np.asarray(x.astype(mx.float32), dtype=np.float64) for x in (q, k, v, dy)],
        scale=0.0625,
        query_offset=offset,
    )
    for actual, reference in zip([output, *gradients], expected, strict=True):
        assert actual.dtype == mx.bfloat16
        plain = np.asarray(actual.astype(mx.float32), dtype=np.float64)
        reference_norm = np.linalg.norm(reference)
        if reference_norm < 1e-8:
            assert np.max(np.abs(plain)) < 1e-3
        else:
            assert np.linalg.norm(plain - reference) / reference_norm < 0.025
    if offset == keys - queries:
        _, automatic = mx.vjp(
            lambda q, k, v: fused_attention(q, k, v, 0.0625, offset),
            [q, k, v],
            [dy],
        )
        for actual, direct in zip(automatic, gradients, strict=True):
            assert bool(mx.array_equal(actual, direct).item())


def test_attention_cannot_observe_future_values() -> None:
    mx.random.seed(19)
    q = mx.random.normal((1, 24, 8, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, 41, 256)).astype(mx.bfloat16)
    v = mx.random.normal(k.shape).astype(mx.bfloat16)
    changed = mx.concatenate([v[:, :, :24], v[:, :, 24:] * 1000], axis=2)
    original, _ = attention_forward(q, k, v, 0.0625, 16)
    result, _ = attention_forward(q, k, changed, 0.0625, 16)
    assert bool(mx.array_equal(original, result).item())
