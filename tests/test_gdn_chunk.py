"""
Purpose: Check the fused GDN chunk kernels against an FP32 reference.
Input: Random normalized q and k, values, gate and beta logits, and a state.
Output: The fused path is at least as close to FP32 as the BF16 path that it
    replaces, for the outputs and for every input gradient.
"""

from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from studio_compute.numerics.gdn_fused import gated_delta_chunks
from studio_compute.numerics.ops import gated_delta_chunked

SHAPES = [(1, 128, 2, 6), (1, 200, 1, 3), (2, 64, 2, 2)]


def _inputs(batch: int, length: int, key_heads: int, heads: int, seed: int) -> list:
    mx.random.seed(seed)

    def unit(shape):
        value = mx.random.normal(shape)
        return value / mx.sqrt(mx.sum(value * value, axis=-1, keepdims=True))

    q = (unit((batch, length, key_heads, 128)) / math.sqrt(128)).astype(mx.bfloat16)
    k = unit((batch, length, key_heads, 128)).astype(mx.bfloat16)
    v = mx.random.normal((batch, length, heads, 128)).astype(mx.bfloat16)
    a = (2 * mx.random.normal((batch, length, heads)) - 2).astype(mx.bfloat16)
    b = mx.random.normal((batch, length, heads)).astype(mx.bfloat16)
    a_log = mx.random.uniform(-2.0, 1.0, (heads,))
    dt_bias = 0.5 * mx.random.normal((heads,))
    state = (0.1 * mx.random.normal((batch, heads, 128, 128))).astype(mx.bfloat16)
    return [q, k, v, a, b, a_log, dt_bias, state]


def _reference(q, k, v, a, b, a_log, dt_bias, state):
    """FP32 gated delta rule with the gates made in FP32 from the logits."""
    log_gate = mx.maximum(-mx.exp(a_log) * nn.softplus(a + dt_bias), math.log(1e-6))
    group = v.shape[2] // q.shape[2]
    return gated_delta_chunked(
        mx.repeat(q, group, -2), mx.repeat(k, group, -2), v, mx.exp(log_gate),
        mx.sigmoid(b), state, compute_dtype=mx.float32, backend="reference")


def _bf16(q, k, v, a, b, a_log, dt_bias, state):
    """The BF16 path of bf16.gdn_inner with the explicit-inverse solve."""
    g = mx.exp(-mx.exp(a_log.astype(mx.bfloat16))
               * nn.softplus(a + dt_bias.astype(mx.bfloat16)))
    return gated_delta_chunked(q, k, v, g, mx.sigmoid(b), state,
                               compute_dtype=mx.bfloat16, backend="metal_inverse")


def _fused(q, k, v, a, b, a_log, dt_bias, state):
    return gated_delta_chunks(q, k, v, a, b, a_log, dt_bias, state)


def _error(actual, expected) -> float:
    actual, expected = actual.astype(mx.float32), expected.astype(mx.float32)
    return float(mx.sqrt(mx.sum((actual - expected) ** 2) / mx.sum(expected**2)).item())


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_chunks_forward_is_at_least_as_accurate(shape: tuple) -> None:
    inputs = _inputs(*shape, seed=sum(shape))
    fp32 = [value.astype(mx.float32) for value in inputs]
    expected = _reference(*fp32)
    fused = _fused(*inputs)
    old = _bf16(*inputs)
    for name, actual, baseline, target in zip(("y", "state"), fused, old, expected):
        assert actual.shape == target.shape, name
        assert _error(actual, target) < 0.02, name
        assert _error(actual, target) <= 1.05 * _error(baseline, target), name


@pytest.mark.parametrize("shape", SHAPES)
def test_fused_chunks_gradients_are_at_least_as_accurate(shape: tuple) -> None:
    inputs = _inputs(*shape, seed=7 + sum(shape))
    fp32 = [value.astype(mx.float32) for value in inputs]
    batch, length, _, heads = shape
    cotangents = [mx.random.normal((batch, length, heads, 128)),
                  0.1 * mx.random.normal((batch, heads, 128, 128))]
    primals = [0, 1, 2, 3, 4, 7]

    def gradients(function, values, dtype):
        def wrapped(*chosen):
            full = list(values)
            for index, value in zip(primals, chosen):
                full[index] = value
            return list(function(*full))

        seeds = [value.astype(dtype) for value in cotangents]
        return mx.vjp(wrapped, [values[index] for index in primals], seeds)[1]

    expected = gradients(_reference, fp32, mx.float32)
    fused = gradients(_fused, inputs, mx.bfloat16)
    old = gradients(_bf16, inputs, mx.bfloat16)
    for name, actual, baseline, target in zip(("dq", "dk", "dv", "da", "db", "dstate"),
                                              fused, old, expected):
        assert actual.shape == target.shape, name
        assert _error(actual, target) <= 1.05 * _error(baseline, target) + 1e-3, name
