"""
Purpose: Check the sorted routed-expert input gradient kernel against MLX autograd.
Input: Random 4-bit experts and rows sorted by expert, with empty experts.
Output: Equal forward values and close input gradients.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from studio_compute.numerics.expert_metal import (
    sorted_expert_qmm,
    sorted_lora_projections,
)


@pytest.mark.parametrize(
    ("experts", "input_width", "output_width", "rows", "group_size"),
    [(8, 256, 64, 300, 64), (6, 128, 96, 37, 32), (4, 384, 160, 129, 128)],
)
def test_sorted_expert_input_gradient_matches_mlx(
    experts: int, input_width: int, output_width: int, rows: int, group_size: int
) -> None:
    mx.random.seed(3)
    weight = (mx.random.normal((experts, output_width, input_width)) * 0.05).astype(
        mx.bfloat16
    )
    packed, scales, biases = mx.quantize(weight, group_size=group_size, bits=4)
    base = SimpleNamespace(
        weight=packed, scales=scales, biases=biases, bits=4, mode="affine",
        group_size=group_size,
    )
    # Expert 1 has no rows; the others have uneven segments.
    ids_R = mx.random.randint(0, experts, (rows,))
    ids_R = mx.sort(mx.where(ids_R == 1, 0, ids_R)).astype(mx.uint32)
    x_R1I = mx.random.normal((rows, 1, input_width)).astype(mx.bfloat16)
    dy_R1O = mx.random.normal((rows, 1, output_width)).astype(mx.bfloat16)

    def reference(x):
        return mx.gather_qmm(
            x, packed, scales, biases, rhs_indices=ids_R, transpose=True,
            group_size=group_size, bits=4, sorted_indices=True,
        )

    expected, (expected_dx,) = mx.vjp(reference, [x_R1I], [dy_R1O])
    actual, (actual_dx,) = mx.vjp(
        lambda x: sorted_expert_qmm(x, base, ids_R), [x_R1I], [dy_R1O]
    )
    assert mx.array_equal(expected[0], actual[0])
    expected_dx = expected_dx.astype(mx.float32)
    actual_dx = actual_dx.astype(mx.float32)
    scale = float(mx.abs(expected_dx).max().item())
    assert float(mx.abs(expected_dx - actual_dx).max().item()) <= 0.02 * scale
    cosine = mx.sum(expected_dx * actual_dx) / mx.sqrt(
        mx.sum(expected_dx**2) * mx.sum(actual_dx**2)
    )
    assert float(cosine.item()) > 0.99999


def _assert_close(expected, actual) -> None:
    expected = expected.astype(mx.float32)
    actual = actual.astype(mx.float32)
    scale = float(mx.abs(expected).max().item())
    assert float(mx.abs(expected - actual).max().item()) <= 0.03 * scale
    cosine = mx.sum(expected * actual) / mx.sqrt(
        mx.sum(expected**2) * mx.sum(actual**2)
    )
    assert float(cosine.item()) > 0.9999


@pytest.mark.parametrize("pairs", [1, 2])
def test_sorted_lora_projections_match_mlx(pairs: int) -> None:
    mx.random.seed(5)
    experts, input_width, output_width, rows, group_size = 6, 256, 96, 211, 64
    modules = []
    for _ in range(pairs):
        weight = mx.random.normal((experts, output_width, input_width)) * 0.05
        packed, scales, biases = mx.quantize(
            weight.astype(mx.bfloat16), group_size=group_size, bits=4
        )
        base = SimpleNamespace(
            weight=packed, scales=scales, biases=biases, bits=4, mode="affine",
            group_size=group_size,
        )
        a = (mx.random.normal((experts, 16, input_width)) * 0.1).astype(mx.bfloat16)
        b = (mx.random.normal((experts, output_width, 16)) * 0.1).astype(mx.bfloat16)
        modules.append(SimpleNamespace(base=base, a=a, b=b, scale=2.0))
    ids_R = mx.random.randint(0, experts, (rows,))
    ids_R = mx.sort(mx.where(ids_R == 1, 0, ids_R)).astype(mx.uint32)
    x_R1I = mx.random.normal((rows, 1, input_width)).astype(mx.bfloat16)
    dys = [
        mx.random.normal((rows, 1, output_width)).astype(mx.bfloat16)
        for _ in range(pairs)
    ]
    factors = [factor for module in modules for factor in (module.a, module.b)]

    def reference(x, *factors):
        outputs = []
        for module, a, b in zip(modules, factors[::2], factors[1::2]):
            base = module.base
            value = mx.gather_qmm(
                x, base.weight, base.scales, base.biases, rhs_indices=ids_R,
                transpose=True, group_size=group_size, bits=4, sorted_indices=True,
            )
            low = mx.gather_mm(x, mx.swapaxes(a, -1, -2), rhs_indices=ids_R,
                               sorted_indices=True)
            update = mx.gather_mm(low, mx.swapaxes(b, -1, -2), rhs_indices=ids_R,
                                  sorted_indices=True)
            outputs.append(value + (2.0 * update).astype(value.dtype))
        return outputs

    def fused(x, *factors):
        for module, a, b in zip(modules, factors[::2], factors[1::2]):
            module.a, module.b = a, b
        return sorted_lora_projections(x, modules, ids_R)

    expected, expected_grads = mx.vjp(reference, [x_R1I, *factors], dys)
    actual, actual_grads = mx.vjp(fused, [x_R1I, *factors], dys)
    for want, got in zip(expected, actual):
        assert mx.array_equal(want, got)
    for want, got in zip(expected_grads, actual_grads):
        assert want.shape == got.shape
        _assert_close(want, got)
