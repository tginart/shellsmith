"""
Purpose: Retain compact BF16 RMS statistics and reverse normalization without a VJP replay.
Input: BF16 rows, frozen weights, and optional saved inverse RMS statistics.
Output: BF16 normalized rows or input gradients. Register reductions use float.
Note: Both forward and reconstruction use the same two-component BF16 statistic.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from typing import Any

import mlx.core as mx

# B: batch, L: rows, D: hidden width, S: two BF16 statistic components.


def forward_saved(x_BLD: Any, weight_D: Any, epsilon: float) -> tuple:
    """
    Purpose: Calculate inverse RMS once and retain its compact BF16 representation.
    Input: BF16 rows and frozen weights. Output: Normalized rows and row statistics.
    """
    _validate(x_BLD, weight_D)
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMS epsilon must be finite and positive.")
    rows = x_BLD.size // x_BLD.shape[-1]
    statistics_BLS = _statistics_kernel(epsilon)(
        inputs=[mx.contiguous(x_BLD), weight_D],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*x_BLD.shape[:-1], 2)],
        output_dtypes=[mx.bfloat16],
    )[0]
    return apply_saved(x_BLD, weight_D, statistics_BLS), statistics_BLS


def apply_saved(x_BLD: Any, weight_D: Any, statistics_BLS: Any) -> Any:
    """Recreate normalized rows with the exact same saved statistic as forward."""
    _validate(x_BLD, weight_D, statistics_BLS)
    return _apply_kernel()(
        inputs=[mx.contiguous(x_BLD), weight_D, statistics_BLS],
        grid=(x_BLD.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x_BLD.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def backward_saved(x_BLD: Any, weight_D: Any, statistics_BLS: Any, dy_BLD: Any) -> Any:
    """
    Purpose: Apply the RMS derivative with the retained inverse RMS.
    Input: Original rows, frozen weights, statistics, and output adjoints.
    Output: BF16 input adjoints. The function does not recalculate RMS statistics.
    """
    _validate(x_BLD, weight_D, statistics_BLS)
    assert dy_BLD.shape == x_BLD.shape and dy_BLD.dtype == mx.bfloat16
    rows = x_BLD.size // x_BLD.shape[-1]
    return _backward_kernel()(
        inputs=[mx.contiguous(x_BLD), weight_D, statistics_BLS, mx.contiguous(dy_BLD)],
        grid=(256, rows, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x_BLD.shape],
        output_dtypes=[mx.bfloat16],
    )[0]


def _validate(x_BLD: Any, weight_D: Any, statistics_BLS: Any = None) -> None:
    assert x_BLD.ndim == 3 and x_BLD.dtype == weight_D.dtype == mx.bfloat16
    assert weight_D.shape == (x_BLD.shape[-1],) and x_BLD.size > 0
    if statistics_BLS is not None:
        assert statistics_BLS.shape == (*x_BLD.shape[:-1], 2)
        assert statistics_BLS.dtype == mx.bfloat16


@lru_cache(maxsize=8)
def _statistics_kernel(epsilon: float) -> Any:
    """Compile a row reduction with the fixed model epsilon and unique row ownership."""
    literal = f"{epsilon:.17e}f"
    identity = hashlib.sha256(literal.encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name="shellsmith_saved_rms_statistics_" + identity,
        input_names=["x", "weight"],
        output_names=["statistics"],
        source="""
            uint lane=thread_position_in_threadgroup.x;
            uint row=threadgroup_position_in_grid.y;
            uint width=weight_shape[0];
            float sum=0.0f;
            for(uint i=lane;i<width;i+=256) {
                float value=float(x[row*width+i]);
                sum+=value*value;
            }
            sum=simd_sum(sum);
            threadgroup float partial[8];
            if((lane%32)==0)partial[lane/32]=sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane<32) {
                sum=lane<8 ? partial[lane] : 0.0f;
                sum=simd_sum(sum);
                if(lane==0) {
                    float inverse=metal::rsqrt(sum/float(width)+EPSILON);
                    bfloat high=bfloat(inverse);
                    statistics[row*2]=high;
                    statistics[row*2+1]=bfloat(inverse-float(high));
                }
            }
        """.replace("EPSILON", literal),
    )


@lru_cache(maxsize=1)
def _apply_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="shellsmith_saved_rms_apply_v1",
        input_names=["x", "weight", "statistics"],
        output_names=["output"],
        source="""
            uint i=thread_position_in_grid.x;
            if(i>=x_shape[0]*x_shape[1]*x_shape[2])return;
            uint width=weight_shape[0],row=i/width;
            float inverse=float(statistics[row*2])+float(statistics[row*2+1]);
            output[i]=bfloat((float(x[i])*inverse)*float(weight[i%width]));
        """,
    )


@lru_cache(maxsize=1)
def _backward_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="shellsmith_saved_rms_backward_v1",
        input_names=["x", "weight", "statistics", "dy"],
        output_names=["output"],
        source="""
            uint lane=thread_position_in_threadgroup.x;
            uint row=threadgroup_position_in_grid.y;
            uint width=weight_shape[0];
            float sum=0.0f;
            for(uint i=lane;i<width;i+=256) {
                uint offset=row*width+i;
                sum+=float(x[offset])*float(dy[offset])*float(weight[i]);
            }
            sum=simd_sum(sum);
            threadgroup float partial[8];
            if((lane%32)==0)partial[lane/32]=sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane<32) {
                sum=lane<8 ? partial[lane] : 0.0f;
                sum=simd_sum(sum);
                if(lane==0)partial[0]=sum/float(width);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float inverse=float(statistics[row*2])+float(statistics[row*2+1]);
            for(uint i=lane;i<width;i+=256) {
                uint offset=row*width+i;
                float projected=float(dy[offset])*float(weight[i]);
                float correction=float(x[offset])*(inverse*inverse)*partial[0];
                output[offset]=bfloat(inverse*(projected-correction));
            }
        """,
    )
