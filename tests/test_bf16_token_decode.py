"""
Purpose: Check the one-token BF16 inference recurrence against independent equations.
Input: Nonzero states, grouped heads, and both single-request and batched layouts.
Output: Value, state, and carry-continuation error bounds.
"""

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
from studio_compute.numerics.gdn_token import gated_delta_token


@pytest.mark.parametrize("batch,keys,values", [(1, 32, 16), (2, 128, 128)])
def test_token_recurrence_matches_independent_sequential_reference(batch, keys, values):
    mx.random.seed(91)
    state = (mx.random.normal((batch, 6, values, keys)) * 0.02).astype(mx.bfloat16)
    reference = np.asarray(state.astype(mx.float32), dtype=np.float64)
    for step in range(8):
        q = (mx.random.normal((batch, 1, 2, keys)) * 0.05).astype(mx.bfloat16)
        k = (mx.random.normal(q.shape) * 0.08).astype(mx.bfloat16)
        v = (mx.random.normal((batch, 1, 6, values)) * 0.1).astype(mx.bfloat16)
        g = mx.full((batch, 1, 6), 0.984375, mx.bfloat16)
        beta = mx.full(g.shape, 0.5, mx.bfloat16)
        out, state = gated_delta_token(q, k, v, g, beta, state)
        mx.eval(out, state)
        q, k, v, g, beta = [
            np.asarray(x.astype(mx.float32), dtype=np.float64)[:, 0]
            for x in (q, k, v, g, beta)
        ]
        q, k = [np.repeat(x, 3, axis=1) for x in (q, k)]
        decayed = g[:, :, None, None] * reference
        delta = beta[:, :, None] * (v - np.einsum("bhvk,bhk->bhv", decayed, k))
        exact = decayed + delta[:, :, :, None] * k[:, :, None, :]
        expected = np.einsum("bhvk,bhk->bhv", exact, q)
        for actual, target in ((out[:, 0], expected), (state, exact)):
            plain = np.asarray(actual.astype(mx.float32), dtype=np.float64)
            assert np.linalg.norm(plain - target) / np.linalg.norm(target) < 0.015
        # Use the actual BF16 carry for the next independent one-step comparison.
        reference = np.asarray(state.astype(mx.float32), dtype=np.float64)
