"""
Purpose: Check the fused GDN kernels against the MLX operations they replace.
Input: Random BF16 rows, carried taps, and weights of several shapes.
Output: Close outputs and close input gradients.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from studio_compute.numerics.gdn_fused import causal_conv_silu, gated_rms_norm


def _cosine(left, right) -> float:
    left, right = left.astype(mx.float32), right.astype(mx.float32)
    return float(
        (mx.sum(left * right) / mx.sqrt(mx.sum(left**2) * mx.sum(right**2))).item()
    )


@pytest.mark.parametrize(("length", "channels", "kernel", "dilation"),
                         [(2048, 512, 4, 1), (37, 96, 4, 1), (2, 64, 4, 1),
                          (130, 32, 3, 1), (300, 64, 4, 3), (5, 32, 4, 3)])
def test_causal_conv_silu_matches_conv1d(length: int, channels: int,
                                         kernel: int, dilation: int) -> None:
    mx.random.seed(length + channels)
    conv = nn.Conv1d(channels, channels, kernel, dilation=dilation, groups=channels,
                     bias=False)
    conv.set_dtype(mx.bfloat16)
    taps_BKC = mx.random.normal((1, (kernel - 1) * dilation, channels)).astype(
        mx.bfloat16)
    x_BLC = mx.random.normal((1, length, channels)).astype(mx.bfloat16)
    cotangent = mx.random.normal((1, length, channels)).astype(mx.bfloat16)

    def reference(taps, x):
        return nn.silu(conv(mx.concatenate([taps, x], axis=1)))

    def fused(taps, x):
        return causal_conv_silu(taps, x, conv.weight, dilation=dilation)

    expected, expected_grads = mx.vjp(reference, [taps_BKC, x_BLC], [cotangent])
    actual, actual_grads = mx.vjp(fused, [taps_BKC, x_BLC], [cotangent])
    assert actual[0].shape == expected[0].shape
    assert _cosine(actual[0], expected[0]) > 0.99995
    for name, a, b in zip(("taps", "x"), actual_grads, expected_grads):
        assert _cosine(a, b) > 0.9999, name


@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize(("rows", "heads", "width"), [(64, 48, 128), (5, 3, 64)])
def test_gated_rms_norm_matches_mlx(rows: int, heads: int, width: int,
                                    sigmoid: bool) -> None:
    mx.random.seed(rows + width)
    y_BLHV = mx.random.normal((1, rows, heads, width)).astype(mx.bfloat16)
    z_BLHV = mx.random.normal((1, rows, heads, width)).astype(mx.bfloat16)
    weight_V = (1 + 0.1 * mx.random.normal((width,))).astype(mx.bfloat16)
    cotangent = mx.random.normal(y_BLHV.shape).astype(mx.bfloat16)

    def reference(y, z):
        gate = mx.sigmoid(z) if sigmoid else nn.silu(z)
        return mx.fast.rms_norm(y, weight_V, 1e-6) * gate

    def fused(y, z):
        return gated_rms_norm(y, z, weight_V, 1e-6, sigmoid=sigmoid)

    expected, expected_grads = mx.vjp(reference, [y_BLHV, z_BLHV], [cotangent])
    actual, actual_grads = mx.vjp(fused, [y_BLHV, z_BLHV], [cotangent])
    assert _cosine(actual[0], expected[0]) > 0.99995
    for name, a, b in zip(("y", "z"), actual_grads, expected_grads):
        assert _cosine(a, b) > 0.9999, name
