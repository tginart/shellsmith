"""
Purpose: Fuse the elementwise and gather steps of the sorted routed MoE replay.
Input: Expert-sorted routed rows, the route scores, and the row permutation.
Output: The SwiGLU hidden rows and the score-weighted sum of each token's rows,
    with custom VJPs.
Method:
    1. swiglu: one kernel makes silu(gate) * up, and one kernel makes both input
       gradients. Each thread owns four values.
    2. combine_routed: one kernel sums the top-k routed rows of a token, each
       times its route score. Its VJP gathers the routed-row gradients in expert
       order, and makes each score gradient with one simdgroup dot product.
    3. gather_sum: one kernel sums the top-k routed rows of each token, for the
       VJP of the row permutation.
Note: The kernels compute in FP32 registers and round once to BF16. MLX rounds
    after each elementwise step.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

# R: routed rows (T * K, in expert order), T: tokens, K: experts per token,
# D: hidden width, F: expert width.
THREADS = 256

SWIGLU_FORWARD = """
    const uint i = 4 * thread_position_in_grid.x;
    const float4 g = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(gate + i));
    const float4 u = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(up + i));
    *reinterpret_cast<device vec<bfloat, 4>*>(h + i) =
        vec<bfloat, 4>(g / (1.0f + metal::exp(-g)) * u);
"""

SWIGLU_BACKWARD = """
    const uint i = 4 * thread_position_in_grid.x;
    const float4 g = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(gate + i));
    const float4 u = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(up + i));
    const float4 d = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(dh + i));
    const float4 s = 1.0f / (1.0f + metal::exp(-g));
    *reinterpret_cast<device vec<bfloat, 4>*>(du + i) = vec<bfloat, 4>(d * g * s);
    *reinterpret_cast<device vec<bfloat, 4>*>(dg + i) =
        vec<bfloat, 4>(d * u * s * (1.0f + g * (1.0f - s)));
"""

# out[t] = sum_k scores[t, k] * routed[inverse[t, k]].
COMBINE_FORWARD = """
    const uint index = thread_position_in_grid.x;
    const int t = index / (D / 4);
    const int c = 4 * (index % (D / 4));
    float4 total = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int row = inverse[t * K + k];
        total += float(scores[t * K + k])
            * float4(*reinterpret_cast<const device vec<bfloat, 4>*>(routed + row * D + c));
    }
    *reinterpret_cast<device vec<bfloat, 4>*>(out + t * D + c) = vec<bfloat, 4>(total);
"""

# out[t] = sum_k rows[inverse[t, k]].
GATHER_SUM = """
    const uint index = thread_position_in_grid.x;
    const int t = index / (D / 4);
    const int c = 4 * (index % (D / 4));
    float4 total = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int row = inverse[t * K + k];
        total += float4(*reinterpret_cast<const device vec<bfloat, 4>*>(rows + row * D + c));
    }
    *reinterpret_cast<device vec<bfloat, 4>*>(out + t * D + c) = vec<bfloat, 4>(total);
"""

# d_routed[r] = scores[order[r]] * dy[order[r] / K], in expert order.
COMBINE_ROUTED_BACKWARD = """
    const uint index = thread_position_in_grid.x;
    const int r = index / (D / 4);
    const int c = 4 * (index % (D / 4));
    const int f = order[r];
    const float4 d = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(dy + (f / K) * D + c));
    *reinterpret_cast<device vec<bfloat, 4>*>(d_routed + r * D + c) =
        vec<bfloat, 4>(float(scores[f]) * d);
"""

# d_scores[t, k] = routed[inverse[t, k]] . dy[t], one simdgroup per (t, k).
COMBINE_SCORES_BACKWARD = """
    const uint pair = thread_position_in_grid.x / 32;
    const ushort lane = thread_index_in_simdgroup;
    const int t = pair / K;
    const int row = inverse[pair];
    float total = 0.0f;
    for (int c = 4 * lane; c < D; c += 128) {
        const float4 a = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(routed + row * D + c));
        const float4 b = float4(*reinterpret_cast<const device vec<bfloat, 4>*>(dy + t * D + c));
        total += metal::dot(a, b);
    }
    total = simd_sum(total);
    if (lane == 0) d_scores[pair] = total;
"""


def is_supported(value: Any) -> bool:
    """Return whether the kernels cover these rows: BF16 with a width of 4 n."""
    return value.dtype == mx.bfloat16 and value.shape[-1] % 4 == 0


def swiglu(up_R1F: Any, gate_R1F: Any) -> Any:
    """Return silu(gate) * up for BF16 rows, as switch_layers.SwiGLU does."""
    assert up_R1F.shape == gate_R1F.shape
    assert up_R1F.dtype == gate_R1F.dtype == mx.bfloat16 and up_R1F.size % 4 == 0
    return _swiglu(up_R1F, gate_R1F)


def combine_routed(routed_RD: Any, scores_TK: Any, inverse_TK: Any, order_R: Any) -> Any:
    """
    Purpose: Sum the routed rows of each token with its route scores.
    Input: Expert-sorted routed rows (R, D) BF16, scores (T, K), inverse_TK (the
        routed row of each token and slot), and order_R (the token slot of each
        routed row, as token * K + slot).
    Output: (T, D) BF16.
    """
    rows, width = routed_RD.shape
    tokens, top_k = scores_TK.shape
    assert rows == tokens * top_k == order_R.shape[0] and inverse_TK.shape == scores_TK.shape
    assert is_supported(routed_RD)
    return _combine(routed_RD, scores_TK, inverse_TK, order_R)


def gather_sum(rows_RD: Any, inverse_TK: Any) -> Any:
    """
    Purpose: Sum the routed rows of each token.
    Input: Expert-sorted rows (R, D) BF16 and inverse_TK, the routed row of each
        token and slot.
    Output: (T, D) BF16, with FP32 sums.
    """
    rows, width = rows_RD.shape
    tokens, top_k = inverse_TK.shape
    assert rows == tokens * top_k and is_supported(rows_RD)
    return _kernel("gather_sum", GATHER_SUM, ("rows", "inverse"), ("out",))(
        inputs=[rows_RD, inverse_TK], template=[("K", top_k), ("D", width)],
        grid=(tokens * width // 4, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(tokens, width)], output_dtypes=[mx.bfloat16])[0]


@mx.custom_function
def _swiglu(up_R1F, gate_R1F):
    return _kernel("swiglu_forward", SWIGLU_FORWARD, ("up", "gate"), ("h",))(
        inputs=[up_R1F, gate_R1F], grid=(up_R1F.size // 4, 1, 1),
        threadgroup=(THREADS, 1, 1), output_shapes=[up_R1F.shape],
        output_dtypes=[mx.bfloat16])[0]


@_swiglu.vjp
def _swiglu_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
    up_R1F, gate_R1F = primals
    du, dg = _kernel("swiglu_backward", SWIGLU_BACKWARD, ("up", "gate", "dh"),
                     ("du", "dg"))(
        inputs=[up_R1F, gate_R1F, cotangent.astype(mx.bfloat16)],
        grid=(up_R1F.size // 4, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[up_R1F.shape] * 2, output_dtypes=[mx.bfloat16] * 2)
    return du, dg


@mx.custom_function
def _combine(routed_RD, scores_TK, inverse_TK, order_R):
    tokens, top_k = scores_TK.shape
    width = routed_RD.shape[1]
    return _kernel("combine_forward", COMBINE_FORWARD,
                   ("routed", "scores", "inverse"), ("out",))(
        inputs=[routed_RD, scores_TK, inverse_TK],
        template=[("K", top_k), ("D", width)],
        grid=(tokens * width // 4, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[(tokens, width)], output_dtypes=[mx.bfloat16])[0]


@_combine.vjp
def _combine_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
    routed_RD, scores_TK, inverse_TK, order_R = primals
    tokens, top_k = scores_TK.shape
    rows, width = routed_RD.shape
    dy_TD = cotangent.astype(mx.bfloat16)
    template = [("K", top_k), ("D", width)]
    (d_routed,) = _kernel("combine_routed_backward", COMBINE_ROUTED_BACKWARD,
                          ("dy", "scores", "order"), ("d_routed",))(
        inputs=[dy_TD, scores_TK, order_R], template=template,
        grid=(rows * width // 4, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[routed_RD.shape], output_dtypes=[mx.bfloat16])
    (d_scores,) = _kernel("combine_scores_backward", COMBINE_SCORES_BACKWARD,
                          ("routed", "dy", "inverse"), ("d_scores",))(
        inputs=[routed_RD, dy_TD, inverse_TK], template=template,
        grid=(rows * 32, 1, 1), threadgroup=(THREADS, 1, 1),
        output_shapes=[scores_TK.shape], output_dtypes=[mx.float32])
    return d_routed, d_scores.astype(scores_TK.dtype), mx.zeros_like(inverse_TK), mx.zeros_like(order_R)


@lru_cache(maxsize=None)
def _kernel(name: str, source: str, inputs: tuple, outputs: tuple) -> Any:
    return mx.fast.metal_kernel(
        name=f"shellsmith_moe_{name}", input_names=list(inputs),
        output_names=list(outputs), source=source, ensure_row_contiguous=True)
