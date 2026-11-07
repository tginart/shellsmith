"""
Purpose: Execute causal D256 GQA with fused NAX Metal forward and backward kernels.
Input: BF16 Q/K/V and an absolute query offset. Output: BF16 values and derivatives.
Note: Score tiles and reduction accumulators remain in GPU registers.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

# B: batch, H: query heads, J: KV heads, Q: query rows, K: key rows, D: width.
# C: two BF16 row-statistic components, P: packed host metadata bytes.
SHADER_ROOT = Path(__file__).with_name("metal")
SCALE = 0.0625
BLOCK_ROWS = 64


@dataclass(frozen=True)
class AttentionTape:
    """
    Purpose: Retain the evaluated attention inputs and compact forward statistics.
    Input: BF16 tensors and the query offset. Output: Explicit backward residuals.
    """

    q_BHQD: Any
    k_BJKD: Any
    v_BJKD: Any
    output_BHQD: Any
    statistics_BHQC: Any
    query_offset: int


def attention_forward(
    q_BHQD: Any,
    k_BJKD: Any,
    v_BJKD: Any,
    scale: float,
    query_offset: int,
) -> tuple[Any, AttentionTape]:
    """
    Purpose: Evaluate full-context suffix-causal attention without a score buffer.
    Input: Production D256 GQA tensors. Output: Values and an explicit tape.
    """
    _validate(q_BHQD, k_BJKD, v_BJKD, scale, query_offset)
    q_BHQD, k_BJKD, v_BJKD = [
        mx.contiguous(value) for value in (q_BHQD, k_BJKD, v_BJKD)
    ]
    output_BHQD, statistics_BHQC = _kernel("forward")(
        inputs=[
            q_BHQD,
            k_BJKD,
            v_BJKD,
            _parameters("forward", q_BHQD.shape, k_BJKD.shape, query_offset),
        ],
        # The tail-safe specialization is faster on the qualified M5, even for aligned rows.
        template=[
            ("ALIGN_Q", False),
            ("ALIGN_K", False),
        ],
        grid=(_blocks(q_BHQD.shape[2]) * 256, 24, q_BHQD.shape[0]),
        threadgroup=(256, 1, 1),
        output_shapes=[q_BHQD.shape, (*q_BHQD.shape[:3], 2)],
        output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return output_BHQD, AttentionTape(
        q_BHQD, k_BJKD, v_BJKD, output_BHQD, statistics_BHQC, query_offset
    )


def attention_backward(tape: AttentionTape, dy_BHQD: Any) -> tuple:
    """
    Purpose: Calculate dQ and jointly owned dK/dV without global atomics.
    Input: The forward tape and BF16 output gradient. Output: Three BF16 gradients.
    """
    q_BHQD, k_BJKD, v_BJKD = tape.q_BHQD, tape.k_BJKD, tape.v_BJKD
    assert dy_BHQD.shape == q_BHQD.shape and dy_BHQD.dtype == mx.bfloat16
    dy_BHQD = mx.contiguous(dy_BHQD)
    delta_BHQC = _row_dot(tape.output_BHQD, dy_BHQD)
    values = [q_BHQD, k_BJKD, v_BJKD]
    residuals = [tape.output_BHQD, tape.statistics_BHQC, dy_BHQD, delta_BHQC]
    gradients = []
    for phase in ("query", "key_value"):
        shape = q_BHQD.shape if phase == "query" else k_BJKD.shape
        heads = 24 if phase == "query" else 4
        rows = shape[2]
        gradients.extend(
            _kernel(phase)(
                inputs=[
                    *values,
                    _parameters(phase, q_BHQD.shape, k_BJKD.shape, tape.query_offset),
                    *residuals,
                ],
                grid=(_blocks(rows, 32) * 256, heads, q_BHQD.shape[0]),
                threadgroup=(256, 1, 1),
                output_shapes=[shape] * (1 if phase == "query" else 2),
                output_dtypes=[mx.bfloat16] * (1 if phase == "query" else 2),
            )
        )
    return tuple(gradients)


def fused_attention(
    q_BHQD: Any,
    k_BJKD: Any,
    v_BJKD: Any,
    scale: float,
    query_offset: int,
) -> Any:
    """Apply the fused operator with its custom derivative for a complete suffix."""
    _validate(q_BHQD, k_BJKD, v_BJKD, scale, query_offset)
    assert query_offset == k_BJKD.shape[2] - q_BHQD.shape[2]
    output_BHQD, _ = _differentiable(q_BHQD, k_BJKD, v_BJKD)
    return output_BHQD


@mx.custom_function
def _differentiable(q_BHQD: Any, k_BJKD: Any, v_BJKD: Any) -> tuple:
    output_BHQD, tape = attention_forward(
        q_BHQD, k_BJKD, v_BJKD, SCALE, k_BJKD.shape[2] - q_BHQD.shape[2]
    )
    return output_BHQD, tape.statistics_BHQC


@_differentiable.vjp
def _attention_vjp(primals: tuple, cotangents: tuple, outputs: tuple) -> tuple:
    q_BHQD, k_BJKD, v_BJKD = primals
    output_BHQD, statistics_BHQC = outputs
    tape = AttentionTape(
        q_BHQD,
        k_BJKD,
        v_BJKD,
        output_BHQD,
        statistics_BHQC,
        k_BJKD.shape[2] - q_BHQD.shape[2],
    )
    return attention_backward(tape, cotangents[0])


@lru_cache(maxsize=4)
def _kernel(phase: str) -> Any:
    """
    Purpose: Compile immutable repository-owned Metal shader sources.
    Input: A forward or derivative phase. Output: A source-keyed MLX kernel.
    """
    inputs = ["input_q", "input_k", "input_v", "parameters"]
    outputs = ["output", "statistics"]
    if phase != "forward":
        inputs += ["input_o", "input_statistics", "input_dy", "input_delta"]
        outputs = ["output"] if phase == "query" else ["output_key", "output_value"]
    if phase == "forward":
        header = (SHADER_ROOT / "gdn_nax_common.metal").read_text()
        header += (SHADER_ROOT / "attention_dsplit_common.metal").read_text()
        source = (SHADER_ROOT / "attention_dsplit_forward.metal").read_text()
    else:
        header = (SHADER_ROOT / "attention_mpp_common.metal").read_text()
        source = (
            SHADER_ROOT
            / (
                "attention_mpp_query.metal"
                if phase == "query"
                else "attention_mpp_joint.metal"
            )
        ).read_text()
    identity = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"shellsmith_attention_{phase}_{identity}",
        input_names=inputs,
        output_names=outputs,
        header=header,
        source=source,
    )


@lru_cache(maxsize=128)
def _parameters(phase: str, q_shape: tuple, k_shape: tuple, query_offset: int) -> Any:
    batch, heads, queries, width = q_shape
    kv_heads, keys = k_shape[1:3]
    query_rows = 64 if phase == "forward" else 32
    key_rows = 32 if phase == "forward" else 256
    common = [
        queries,
        keys,
        heads // kv_heads,
        _blocks(queries, query_rows),
        _blocks(keys, key_rows),
        queries % query_rows,
        keys % key_rows,
        query_offset,
    ]
    q_strides = [heads * queries * width, queries * width, width]
    k_strides = [kv_heads * keys * width, keys * width, width]
    row_strides = [heads * queries * 2, queries * 2, 2]
    strides = [q_strides, k_strides, k_strides, q_strides, row_strides]
    flat = [item for values in strides for item in values]
    packed = struct.pack(f"<8i{len(flat)}q", *common, *flat)
    return mx.array(list(packed), mx.uint8)


def _row_dot(output_BHQD: Any, dy_BHQD: Any) -> Any:
    return _row_dot_kernel()(
        inputs=[output_BHQD, dy_BHQD],
        template=[("T", mx.bfloat16)],
        grid=(32, output_BHQD.size // 256, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(*output_BHQD.shape[:3], 2)],
        output_dtypes=[mx.bfloat16],
    )[0]


@lru_cache(maxsize=1)
def _row_dot_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="shellsmith_attention_row_dot_bf16_pair_v1",
        input_names=["value", "cotangent"],
        output_names=["output"],
        source="""
            uint lane = thread_position_in_grid.x;
            uint row = thread_position_in_grid.y;
            float sum = 0.0f;
            for (uint i = lane; i < 256; i += 32) {
                sum += float(value[row * 256 + i]) * float(cotangent[row * 256 + i]);
            }
            sum = simd_sum(sum);
            if (lane == 0) {
                T high = T(sum);
                output[row * 2] = high;
                output[row * 2 + 1] = T(sum - float(high));
            }
        """,
    )


def _blocks(rows: int, block: int = BLOCK_ROWS) -> int:
    return (rows + block - 1) // block


def _validate(q: Any, k: Any, v: Any, scale: float, query_offset: int) -> None:
    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("Attention requires BHQD/BJKD tensors and aligned K/V.")
    if q.shape[1] != 24 or k.shape[1] != 4 or q.shape[-1] != 256 or k.shape[-1] != 256:
        raise ValueError("Fused attention requires the admitted 24/4-head D256 model.")
    if q.shape[0] != k.shape[0] or not 1 <= q.shape[2] <= k.shape[2] <= 262144:
        raise ValueError("Fused attention requires valid query and key lengths.")
    if (
        type(query_offset) is not int
        or not 0 <= query_offset <= k.shape[2] - q.shape[2]
    ):
        raise ValueError("Invalid absolute causal query offset.")
    if scale != SCALE or any(x.dtype != mx.bfloat16 for x in (q, k, v)):
        raise ValueError("Fused attention requires BF16 operands and scale 1/16.")
