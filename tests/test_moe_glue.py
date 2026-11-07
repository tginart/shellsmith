"""
Purpose: Check the fused routed-MoE glue kernels against the MLX operations.
Input: Random BF16 rows, route scores, and a random expert order.
Output: Close values and close input gradients.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from studio_compute.numerics.moe_glue import combine_routed, gather_sum, swiglu


def _cosine(left, right) -> float:
    left, right = left.astype(mx.float32), right.astype(mx.float32)
    return float((mx.sum(left * right) / mx.sqrt(mx.sum(left**2) * mx.sum(right**2))).item())


def test_swiglu_matches_mlx() -> None:
    mx.random.seed(1)
    up, gate = (mx.random.normal((300, 1, 640)).astype(mx.bfloat16) for _ in range(2))
    dh = mx.random.normal(up.shape).astype(mx.bfloat16)
    expected = mx.vjp(lambda u, g: nn.silu(g) * u, [up, gate], [dh])
    actual = mx.vjp(swiglu, [up, gate], [dh])
    assert _cosine(actual[0][0], expected[0][0]) > 0.9999
    for a, b in zip(actual[1], expected[1]):
        assert _cosine(a, b) > 0.9999


@pytest.mark.parametrize(("tokens", "top_k", "width"), [(64, 10, 2560), (7, 3, 128)])
def test_combine_routed_matches_gather_sum(tokens: int, top_k: int, width: int) -> None:
    mx.random.seed(tokens)
    routed = mx.random.normal((tokens * top_k, width)).astype(mx.bfloat16)
    scores = mx.softmax(mx.random.normal((tokens, top_k)), axis=-1).astype(mx.bfloat16)
    order = mx.random.permutation(tokens * top_k).astype(mx.int32)
    inverse = mx.argsort(order).astype(mx.int32).reshape(tokens, top_k)
    dy = mx.random.normal((tokens, width)).astype(mx.bfloat16)

    def reference(r, s):
        rows = r[inverse.reshape(-1)].reshape(tokens, top_k, width)
        return mx.sum(rows.astype(mx.float32) * s[..., None].astype(mx.float32), axis=1)

    expected = mx.vjp(reference, [routed, scores], [dy.astype(mx.float32)])
    actual = mx.vjp(lambda r, s: combine_routed(r, s, inverse, order), [routed, scores], [dy])
    assert _cosine(actual[0][0], expected[0][0]) > 0.9999
    for a, b in zip(actual[1], expected[1]):
        assert _cosine(a, b) > 0.9999


def test_gather_sum_matches_mlx() -> None:
    mx.random.seed(11)
    tokens, top_k, width = 29, 10, 256
    rows = mx.random.normal((tokens * top_k, width)).astype(mx.bfloat16)
    order = mx.random.permutation(tokens * top_k).astype(mx.int32)
    inverse = mx.argsort(order).astype(mx.int32).reshape(tokens, top_k)
    expected = rows.astype(mx.float32)[inverse.reshape(-1)].reshape(tokens, top_k, -1)
    assert _cosine(gather_sum(rows, inverse), expected.sum(axis=1)) > 0.99999


def test_inject_matches_mlx() -> None:
    from studio_compute.numerics.hyper_metal import inject

    mx.random.seed(5)
    hyper = mx.random.normal((1, 33, 4 * 2560)).astype(mx.bfloat16)
    output = mx.random.normal((1, 33, 2560)).astype(mx.bfloat16)
    weight = mx.random.uniform(0, 2, (1, 33, 4)).astype(mx.bfloat16)
    dy = mx.random.normal(hyper.shape).astype(mx.bfloat16)

    def reference(h, o, w):
        injected = o[..., None, :].astype(mx.float32) * w[..., None].astype(mx.float32)
        return h.astype(mx.float32) + injected.reshape(h.shape)

    expected = mx.vjp(reference, [hyper, output, weight], [dy.astype(mx.float32)])
    actual = mx.vjp(inject, [hyper, output, weight], [dy])
    assert _cosine(actual[0][0], expected[0][0]) > 0.99999
    for a, b in zip(actual[1], expected[1]):
        assert _cosine(a, b) > 0.9999
