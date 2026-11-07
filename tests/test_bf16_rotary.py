"""
Purpose: Verify BF16 MRoPE and inverse derivatives with independent long-position equations.
Input: Multiple position axes, short tails, nonzero Q/K, and cotangents.
Output: Forward, derivative, and unchanged-pass-through error bounds.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding
from studio_compute.numerics.rotary import apply_rotary


def reference(value, positions, frequency, selector, dim, *, inverse=False):
    value = np.asarray(value.astype(mx.float32), dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    frequencies = np.asarray(frequency, dtype=np.float64)
    selectors = np.asarray(selector, dtype=np.int32)
    batch, heads, length, width = value.shape
    angles = np.empty((batch, length, dim // 2), dtype=np.float64)
    for i, axis in enumerate(selectors):
        angles[:, :, i] = positions[axis] * frequencies[i]
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    if inverse:
        sine = -sine
    first, second = value[..., : dim // 2], value[..., dim // 2 : dim]
    out = value.copy()
    out[..., : dim // 2] = first * cosine - second * sine
    out[..., dim // 2 : dim] = second * cosine + first * sine
    return out


@pytest.mark.parametrize("length", [1, 8, 17])
@pytest.mark.parametrize("rotary_dim", [64, 128])
def test_rotary_forward_and_inverse_vjp(length, rotary_dim):
    mx.random.seed(23)
    rotary = Qwen3_5RotaryEmbedding(
        rotary_dim,
        base=10000000,
        mrope_section=[
            rotary_dim // 6,
            rotary_dim // 6,
            rotary_dim // 2 - 2 * (rotary_dim // 6),
        ],
    )
    q = mx.random.normal((1, 24, length, 256)).astype(mx.bfloat16)
    k = mx.random.normal((1, 4, length, 256)).astype(mx.bfloat16)
    dy = [mx.random.normal(value.shape).astype(mx.bfloat16) for value in (q, k)]
    p = np.empty((3, 1, length), dtype=np.int32)
    for axis in range(3):
        p[axis, 0] = np.linspace(
            axis * 3, 262143 - axis * 32768, length, dtype=np.int32
        )
    positions = mx.array(p, mx.int32)
    values, gradients = mx.vjp(
        lambda q, k: apply_rotary(q, k, positions, rotary), [q, k], dy
    )
    for actual, original, cotangent, gradient in zip(
        values, [q, k], dy, gradients, strict=True
    ):
        forward = reference(
            original, positions, rotary.inv_freq, rotary.position_selector, rotary_dim
        )
        backward = reference(
            cotangent,
            positions,
            rotary.inv_freq,
            rotary.position_selector,
            rotary_dim,
            inverse=True,
        )
        for got, wanted in ((actual, forward), (gradient, backward)):
            plain = np.asarray(got.astype(mx.float32), dtype=np.float64)
            assert np.linalg.norm(plain - wanted) / np.linalg.norm(wanted) < 0.01
        assert actual.dtype == gradient.dtype == mx.bfloat16
        assert bool(
            mx.array_equal(actual[..., rotary_dim:], original[..., rotary_dim:]).item()
        )
