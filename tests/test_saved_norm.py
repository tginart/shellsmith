"""
Purpose: Check saved RMS values and input derivatives against independent FP64 equations.
Input: BF16 rows and frozen weights at tiny and actual model widths.
Output: Numerical bounds and proof that backward does not recalculate statistics.
"""

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
from studio_compute.numerics import saved_norm


@pytest.mark.parametrize("shape", [(2, 7, 64), (1, 8, 5120)])
def test_saved_rms_values_and_derivatives(shape, monkeypatch):
    mx.random.seed(23)
    x = (mx.random.normal(shape) * 0.1).astype(mx.bfloat16)
    weight = (mx.random.normal((shape[-1],)) * 0.2 + 1).astype(mx.bfloat16)
    dy = mx.random.normal(shape).astype(mx.bfloat16)
    output, statistics = saved_norm.forward_saved(x, weight, 1e-6)
    mx.eval(output, statistics)
    assert statistics.shape == (*shape[:-1], 2) and statistics.dtype == mx.bfloat16

    def forbidden(*_):
        raise AssertionError("Backward recalculated forward RMS statistics.")

    monkeypatch.setattr(saved_norm, "_statistics_kernel", forbidden)
    repeated = saved_norm.apply_saved(x, weight, statistics)
    dx = saved_norm.backward_saved(x, weight, statistics, dy)
    mx.eval(repeated, dx)
    assert bool(mx.array_equal(repeated, output).item())
    plain = lambda x: np.asarray(x.astype(mx.float32), dtype=np.float64)
    x, w, dy = map(plain, (x, weight, dy))
    inverse = 1 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    expected_output = x * inverse * w
    expected_gradient = inverse * (
        dy * w - x * inverse**2 * np.mean(x * dy * w, axis=-1, keepdims=True)
    )
    for actual, expected in ((output, expected_output), (dx, expected_gradient)):
        assert actual.dtype == mx.bfloat16
        assert (
            np.linalg.norm(plain(actual) - expected) / np.linalg.norm(expected) < 0.01
        )
