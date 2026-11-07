"""
Purpose: Check the fused hyper-connection kernels against the upstream module.
Input: A quantized Qwen4ExpGatedResidual module and random BF16 hyper rows.
Output: Close outputs and close input gradients for all three outputs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
language = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from studio_compute.numerics.hyper_metal import hyper_connection


def _cosine(left, right) -> float:
    left, right = left.astype(mx.float32), right.astype(mx.float32)
    return float(
        (mx.sum(left * right) / mx.sqrt(mx.sum(left**2) * mx.sum(right**2))).item()
    )


@pytest.mark.parametrize(("hidden", "rows"), [(256, 37), (64, 5), (2560, 16)])
def test_fused_hyper_connection_matches_module(hidden: int, rows: int) -> None:
    mx.random.seed(5)
    config = SimpleNamespace(
        hc_count=4, hidden_size=hidden, hc_lowrank=64, rms_norm_eps=1e-6
    )
    module = language.Qwen4ExpGatedResidual(config)
    module.hc_norm.weight = mx.random.normal((4 * hidden,)) * 0.1
    module.set_dtype(mx.bfloat16)
    nn.quantize(module, group_size=64, bits=8)
    module.freeze()
    hyper_BLW = mx.random.normal((1, rows, 4 * hidden)).astype(mx.bfloat16)
    cotangents = [
        mx.random.normal((1, rows, hidden)).astype(mx.bfloat16),
        mx.random.normal((1, rows, 4 * hidden)).astype(mx.bfloat16),
        mx.random.normal((1, rows, 4)).astype(mx.bfloat16),
    ]
    expected, (expected_dx,) = mx.vjp(module, [hyper_BLW], cotangents)
    actual, (actual_dx,) = mx.vjp(
        lambda x: hyper_connection(module, x), [hyper_BLW], cotangents
    )
    for left, right in zip(expected, actual):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert _cosine(left, right) > 0.99999
    assert _cosine(expected_dx, actual_dx) > 0.9999


@pytest.mark.parametrize(("hidden", "rows"), [(64, 5), (2560, 33)])
def test_mix_kernel_matches_upstream(hidden: int, rows: int) -> None:
    from studio_compute.numerics import hyper_metal

    mx.random.seed(7)
    projected = mx.random.normal((1, rows, 4 * hidden)).astype(mx.bfloat16)
    normed = mx.random.normal((1, rows, 4 * hidden)).astype(mx.bfloat16)
    upstream = hyper_metal._pointwise(4, hidden)[1](projected, normed)
    actual = hyper_metal._mix_function(4, hidden)(projected, normed)
    assert actual.shape == upstream.shape and actual.dtype == mx.bfloat16
    assert mx.mean(actual == upstream).item() > 0.999
