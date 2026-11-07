"""
Purpose: Update BF16 Adam tensors with a stochastically rounded parameter sum.
Input: BF16 parameters, moments, gradients, and a dedicated counter seed.
Output: Three BF16 tensors. No FP32 arrays or master weights exist.
Note: Scalar register intermediates use native float instructions inside the kernel.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

# N: elements, C: five optimizer constants, R: three counter words.


def update(
    parameter_N: Any, first_N: Any, second_N: Any, gradient_N: Any,
    constants_C: Any, counter_R: Any,
) -> tuple:
    """
    Purpose: Apply one counter-seeded Adam update without FP32 tensor state.
    Input: BF16 state and gradient tensors. Output: New BF16 state tensors.
    """
    assert parameter_N.shape == first_N.shape == second_N.shape == gradient_N.shape
    assert all(value.dtype == mx.bfloat16
               for value in (parameter_N, first_N, second_N, gradient_N, constants_C))
    if mx.default_device() != mx.gpu:
        raise RuntimeError("BF16 Adam requires its qualified Metal kernel.")
    return tuple(_kernel()(
        inputs=[parameter_N, first_N, second_N, gradient_N, constants_C, counter_R],
        template=[("T", mx.bfloat16)],
        grid=(parameter_N.size, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[parameter_N.shape] * 3,
        output_dtypes=[mx.bfloat16] * 3,
    ))


@lru_cache(maxsize=1)
def _kernel() -> Any:
    """
    Purpose: Cache one Metal program for the BF16 optimizer.
    Input: None. Output: A kernel with explicit state and counter inputs.
    """
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_adam_sr_v1",
        input_names=["parameter", "first", "second", "gradient", "constants", "counter"],
        output_names=["next_parameter", "next_first", "next_second"],
        source="""
            uint i = thread_position_in_grid.x;
            float g = float(gradient[i]);
            float b1 = float(constants[0]);
            float b2 = float(constants[1]);
            T m = T(b1 * float(first[i]) + (1.0f - b1) * g);
            T v = T(b2 * float(second[i]) + (1.0f - b2) * g * g);
            float denominator = metal::sqrt(float(v)) / float(constants[3])
                + float(constants[4]);
            T delta = T(float(constants[2]) * float(m) / denominator);
            float proposed = float(parameter[i]) - float(delta);
            uint random = i ^ counter[0] ^ (counter[1] * 0x9e3779b9u) ^ counter[2];
            random ^= random >> 16;
            random *= 0x7feb352du;
            random ^= random >> 15;
            random *= 0x846ca68bu;
            random ^= random >> 16;
            uint bits = as_type<uint>(proposed);
            bits = (bits + (random & 0xffffu)) & 0xffff0000u;
            next_parameter[i] = T(as_type<float>(bits));
            next_first[i] = m;
            next_second[i] = v;
        """,
    )
