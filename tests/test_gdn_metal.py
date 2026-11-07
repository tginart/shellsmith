"""
Purpose: Compare the Metal GDN solve and state VJP with independent equations.
Input: Small BF16 systems and chunk transforms. Output: Forward and derivative bounds.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
from studio_compute.numerics.gdn_metal import (
    scan,
    solve_unit_lower,
    solve_unit_lower_inverse,
)
from studio_compute.numerics.ops import _scan_forward


def _relative(actual, expected) -> float:
    actual = actual.astype(mx.float32)
    expected = expected.astype(mx.float32)
    return float(mx.linalg.norm(actual - expected).item()) / max(
        float(mx.linalg.norm(expected).item()), 1e-8
    )


@pytest.mark.parametrize("solve", [solve_unit_lower, solve_unit_lower_inverse])
@pytest.mark.parametrize("rows,columns", [(8, 7), (64, 64), (64, 256)])
def test_unit_solve_and_transposed_derivative(rows: int, columns: int, solve) -> None:
    mx.random.seed(3)
    matrix = mx.tril(mx.random.normal((2, rows, rows)) * 0.01, -1)
    matrix = (matrix + mx.eye(rows)).astype(mx.bfloat16)
    rhs = mx.random.normal((2, rows, columns)).astype(mx.bfloat16)
    cotangent = mx.random.normal(rhs.shape).astype(mx.bfloat16)
    result, gradients = mx.vjp(solve, [matrix, rhs], [cotangent])
    result = result[0]
    expected = np.linalg.solve(np.array(matrix.astype(mx.float32)),
                               np.array(rhs.astype(mx.float32)))
    assert _relative(result, mx.array(expected)) < 0.015
    expected_rhs = np.linalg.solve(
        np.swapaxes(np.array(matrix.astype(mx.float32)), -1, -2),
        np.array(cotangent.astype(mx.float32)),
    )
    assert _relative(gradients[1], mx.array(expected_rhs)) < 0.015
    expected_matrix = np.tril(
        -(expected_rhs @ np.swapaxes(expected, -1, -2)), -1
    )
    assert _relative(gradients[0], mx.array(expected_matrix)) < 0.025


@pytest.mark.parametrize("chunks,rows,keys,values", [(1, 8, 32, 32), (3, 64, 128, 128)])
def test_state_scan_all_cotangents(chunks: int, rows: int, keys: int, values: int) -> None:
    mx.random.seed(19)
    value_shape = (1, 2, chunks, rows, values)
    key_shape = (1, 2, chunks, rows, keys)
    inputs = [
        (mx.random.normal(value_shape) * 0.02).astype(mx.bfloat16),
        (mx.random.normal(key_shape) * 0.02).astype(mx.bfloat16),
        (mx.random.normal(value_shape) * 0.02).astype(mx.bfloat16),
        (mx.random.normal(key_shape) * 0.02).astype(mx.bfloat16),
        (mx.random.normal(key_shape) * 0.02).astype(mx.bfloat16),
        mx.full((1, 2, chunks), 0.9, mx.bfloat16),
        mx.full((1, 2, chunks, rows), 0.8, mx.bfloat16),
        (mx.random.normal((1, 2, values, keys)) * 0.02).astype(mx.bfloat16),
    ]
    cotangents = [
        mx.random.normal(value_shape).astype(mx.bfloat16),
        mx.random.normal(inputs[-1].shape).astype(mx.bfloat16),
    ]

    def reference(*arguments):
        output, _, final = _scan_forward(*arguments)
        return output, final

    expected, expected_gradients = mx.vjp(reference, inputs, cotangents)
    actual, gradients = mx.vjp(scan, inputs, cotangents)
    for left, right in zip(actual, expected, strict=True):
        assert _relative(left, right) < 0.025
    for left, right in zip(gradients, expected_gradients, strict=True):
        assert left.dtype == mx.bfloat16
        assert _relative(left, right) < 0.04
