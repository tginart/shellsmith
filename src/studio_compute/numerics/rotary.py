"""
Purpose: Apply MRoPE with BF16 tensors and an explicit inverse-rotation derivative.
Input: BF16 Q/K, integer positions, and frozen frequency and axis constants.
Output: BF16 values or cotangents without FP32 activation tensors.
Note: Kernels calculate phase angles in float registers to preserve long positions.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

# B: batch, H: Q heads, J: KV heads, L: rows, D: head width, R: rotary width.


def apply_rotary(
    q_BHLD: Any,
    k_BJLD: Any,
    positions_ABL: Any,
    rotary: Any,
) -> tuple:
    """
    Purpose: Rotate Q/K with the admitted MRoPE axis selectors and half pairing.
    Input: Current projected tensors and immutable position metadata.
    Output: Rotated tensors with a BF16 custom derivative.
    """
    assert q_BHLD.dtype == k_BJLD.dtype == mx.bfloat16
    assert q_BHLD.shape[0] == k_BJLD.shape[0] and q_BHLD.shape[2:] == k_BJLD.shape[2:]
    assert positions_ABL.shape == (3, q_BHLD.shape[0], q_BHLD.shape[2])
    if rotary.pairing != "half_split" or rotary.position_selector is None:
        raise ValueError(
            "BF16 MRoPE requires the admitted half-paired position selector."
        )
    assert rotary.dim % 2 == 0 and rotary.dim <= q_BHLD.shape[-1]

    @mx.custom_function
    def rotate(q_BHLD, k_BJLD):
        return _rotate(q_BHLD, k_BJLD, positions_ABL, rotary, inverse=False)

    @rotate.vjp
    def derivative(primals, cotangents, outputs):
        return _rotate(*cotangents, positions_ABL, rotary, inverse=True)

    return rotate(q_BHLD, k_BJLD)


def _rotate(
    q_BHLD: Any, k_BJLD: Any, positions_ABL: Any, rotary: Any, *, inverse: bool
) -> tuple:
    return tuple(
        _kernel()(
            inputs=[
                mx.contiguous(q_BHLD),
                mx.contiguous(k_BJLD),
                mx.contiguous(positions_ABL),
                rotary.inv_freq,
                rotary.position_selector,
            ],
            template=[("R", rotary.dim), ("INVERSE", inverse)],
            grid=(q_BHLD.size + k_BJLD.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[q_BHLD.shape, k_BJLD.shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
    )


@lru_cache(maxsize=1)
def _kernel() -> Any:
    """Compile one BF16 forward/inverse rotation with exact integer position input."""
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_mrope_inverse_v1",
        input_names=["q", "k", "positions", "frequencies", "selector"],
        output_names=["output_q", "output_k"],
        source="""
            uint index=thread_position_in_grid.x;
            uint batch=q_shape[0],length=q_shape[2],width=q_shape[3];
            uint q_count=batch*q_shape[1]*length*width;
            uint k_count=batch*k_shape[1]*length*width;
            if(index>=q_count+k_count)return;
            bool is_query=index<q_count;
            uint local=is_query ? index : index-q_count;
            uint heads=is_query ? q_shape[1] : k_shape[1];
            const device bfloat* input=is_query ? q : k;
            device bfloat* output=is_query ? output_q : output_k;
            uint b=local/(heads*length*width),row=(local/width)%length,d=local%width;
            if(d>=R) {output[local]=input[local];return;}
            uint half_width=R/2,frequency=d%half_width;
            uint other=d<half_width ? d+half_width : d-half_width;
            int axis=selector[frequency];
            float angle=float(positions[(axis*batch+b)*length+row])*float(frequencies[frequency]);
            float sign=d<half_width ? -1.0f : 1.0f;
            if(INVERSE)sign=-sign;
            float value=float(input[local])*metal::cos(angle)
                +sign*float(input[local-d+other])*metal::sin(angle);
            output[local]=bfloat(value);
        """,
    )
