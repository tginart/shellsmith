"""
Purpose: Apply the BF16 chunk solve and recurrent scan with device-resident kernels.
Input: BF16 chunk transforms and explicit state. Output: Values and all input adjoints.
Note: Dot products use float register accumulators. Every tensor uses BF16.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from functools import lru_cache
from typing import Any

import mlx.core as mx

# B: batch, H: heads, N: chunks, C: chunk rows, K: key width, V: value width.


@mx.custom_function
def solve_unit_lower(matrix_BHNCC: Any, rhs_BHNCR: Any) -> Any:
    """
    Purpose: Solve one unit lower-triangular system for all right-hand sides.
    Input: A BF16 unit lower matrix and BF16 right-hand sides.
    Output: A BF16 solution without an explicit inverse.
    """
    return _solve(matrix_BHNCC, rhs_BHNCR, transpose=False)


@solve_unit_lower.vjp
def _solve_vjp(primals: tuple, cotangents: Any, outputs: Any) -> tuple:
    matrix_BHNCC, _ = primals
    gradient_BHNCR = _solve(matrix_BHNCC, cotangents, transpose=True)
    matrix_gradient_BHNCC = mx.tril(
        -(gradient_BHNCR @ mx.swapaxes(outputs, -1, -2)), -1
    )
    return matrix_gradient_BHNCC, gradient_BHNCR


@mx.custom_function
def solve_unit_lower_inverse(matrix_BHNCC: Any, rhs_BHNCR: Any) -> Any:
    """
    Purpose: Solve one unit lower-triangular system with an explicit FP32 inverse.
    Input: A BF16 unit lower matrix and BF16 right-hand sides.
    Output: The BF16 solution inverse(matrix) @ rhs.
    Note: Training replay uses this solve. One kernel inverts each 64-by-64
        matrix in FP32. The products with the right-hand sides and the
        cotangents are batched BF16 matmuls.
    """
    return _invert(matrix_BHNCC) @ rhs_BHNCR


@solve_unit_lower_inverse.vjp
def _solve_inverse_vjp(primals: tuple, cotangents: Any, outputs: Any) -> tuple:
    matrix_BHNCC, _ = primals
    gradient_BHNCR = mx.swapaxes(_invert(matrix_BHNCC), -1, -2) @ cotangents
    matrix_gradient_BHNCC = mx.tril(
        -(gradient_BHNCR @ mx.swapaxes(outputs, -1, -2)), -1
    )
    return matrix_gradient_BHNCC, gradient_BHNCR


def scan(
    mu0_BHNCV: Any, q_BHNCK: Any, u0_BHNCV: Any, kt_BHNCK: Any,
    k_BHNCK: Any, decay_BHN: Any, ratio_BHNC: Any, state_BHVK: Any,
) -> tuple:
    """
    Purpose: Traverse all chunks on the GPU without Python chunk dispatch.
    Input: Chunk transforms and the incoming state. Output: Values and final state.
    """
    values_BHNCV, final_BHVK, _ = _scan(
        mu0_BHNCV, q_BHNCK, u0_BHNCV, kt_BHNCK,
        k_BHNCK, decay_BHN, ratio_BHNC, state_BHVK,
    )
    return values_BHNCV, final_BHVK


@mx.custom_function
def _scan(*inputs: Any) -> tuple:
    mu0_BHNCV, q_BHNCK, _, _, _, _, _, state_BHVK = inputs
    batch, heads, chunks, rows, values = mu0_BHNCV.shape
    keys = q_BHNCK.shape[-1]
    assert keys % 32 == 0 and rows <= 64
    assert state_BHVK.shape == (batch, heads, values, keys)
    assert all(value.dtype == mx.bfloat16 for value in inputs)
    return tuple(_scan_forward_kernel()(
        inputs=list(inputs), template=[("T", mx.bfloat16), ("C", rows),
                                      ("DK", keys), ("DV", values)],
        grid=(32, values, batch * heads), threadgroup=(32, 4, 1),
        output_shapes=[mu0_BHNCV.shape, state_BHVK.shape,
                       (batch, heads, chunks, values, keys)],
        output_dtypes=[mx.bfloat16] * 3,
    ))


@_scan.vjp
def _scan_vjp(primals: tuple, cotangents: tuple, outputs: tuple) -> tuple:
    """
    Purpose: Propagate state adjoints in reverse, then batch the local derivatives.
    Input: BF16 primals, cotangents, and saved chunk states.
    Output: All eight BF16 input gradients.
    """
    mu0_BHNCV, q_BHNCK, u0_BHNCV, kt_BHNCK, k_BHNCK, decay_BHN, ratio_BHNC, state_BHVK = primals
    dy_BHNCV, final_gradient_BHVK, _ = cotangents
    _, _, history_BHNVK = outputs
    batch, heads, chunks, rows, values = mu0_BHNCV.shape
    keys = q_BHNCK.shape[-1]
    du_BHNCV, ds_BHNVK, dw_BHNCV, uc_BHNCV, initial_gradient_BHVK = _scan_backward_kernel()(
        inputs=[q_BHNCK, u0_BHNCV, kt_BHNCK, k_BHNCK, decay_BHN,
                ratio_BHNC, history_BHNVK, dy_BHNCV, final_gradient_BHVK],
        template=[("T", mx.bfloat16), ("C", rows), ("DK", keys), ("DV", values)],
        grid=(32, values, batch * heads), threadgroup=(32, 4, 1),
        output_shapes=[mu0_BHNCV.shape, history_BHNVK.shape, mu0_BHNCV.shape,
                       mu0_BHNCV.shape, state_BHVK.shape],
        output_dtypes=[mx.bfloat16] * 5,
    )
    dq_BHNCK = dy_BHNCV @ history_BHNVK
    dkt_BHNCK = -(du_BHNCV @ history_BHNVK)
    weighted_BHNCV = ratio_BHNC[..., None] * uc_BHNCV
    dk_BHNCK = weighted_BHNCV @ ds_BHNVK
    ddecay_BHN = mx.sum(ds_BHNVK * history_BHNVK, axis=(-2, -1))
    dratio_BHNC = mx.sum(dw_BHNCV * uc_BHNCV, axis=-1)
    return (dy_BHNCV, dq_BHNCK, du_BHNCV, dkt_BHNCK, dk_BHNCK,
            ddecay_BHN, dratio_BHNC, initial_gradient_BHVK)


def _solve(matrix_BHNCC: Any, rhs_BHNCR: Any, *, transpose: bool) -> Any:
    rows, columns = rhs_BHNCR.shape[-2:]
    assert matrix_BHNCC.shape == (*rhs_BHNCR.shape[:-2], rows, rows)
    assert 0 < rows <= 64
    assert matrix_BHNCC.dtype == rhs_BHNCR.dtype == mx.bfloat16
    batches = rhs_BHNCR.size // (rows * columns)
    return _solve_kernel()(
        inputs=[matrix_BHNCC, rhs_BHNCR],
        template=[("T", mx.bfloat16), ("C", rows), ("R", columns),
                  ("TRANSPOSE", transpose)],
        grid=(columns, batches, 1), threadgroup=(min(columns, 128), 1, 1),
        output_shapes=[rhs_BHNCR.shape], output_dtypes=[mx.bfloat16],
    )[0]


@lru_cache(maxsize=1)
def _solve_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_unit_solve_v1",
        input_names=["matrix", "rhs"], output_names=["solution"],
        source="""
            uint column = thread_position_in_grid.x;
            uint batch = threadgroup_position_in_grid.y;
            uint lane = thread_index_in_threadgroup;
            uint threads = threads_per_threadgroup.x;
            threadgroup T lower[C * C];
            for (uint i = lane; i < C * C; i += threads) {
                lower[i] = matrix[batch * C * C + i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            T values[C];
            for (int step = 0; step < C; ++step) {
                int row = TRANSPOSE ? C - 1 - step : step;
                float value = float(rhs[(batch * C + row) * R + column]);
                for (int prior = 0; prior < step; ++prior) {
                    int j = TRANSPOSE ? C - 1 - prior : prior;
                    T coefficient = TRANSPOSE ? lower[j * C + row] : lower[row * C + j];
                    value -= float(coefficient) * float(values[j]);
                }
                values[row] = T(value);
            }
            for (int row = 0; row < C; ++row) {
                solution[(batch * C + row) * R + column] = values[row];
            }
        """,
    )


def _invert(matrix_BHNCC: Any) -> Any:
    """Return the BF16 inverse of each unit lower-triangular matrix, made in FP32."""
    rows = matrix_BHNCC.shape[-1]
    assert matrix_BHNCC.shape[-2] == rows and 0 < rows <= 64
    assert matrix_BHNCC.dtype == mx.bfloat16
    batches = matrix_BHNCC.size // (rows * rows)
    return _inverse_kernel()(
        inputs=[matrix_BHNCC], template=[("C", rows)],
        grid=(rows, batches, 1), threadgroup=(rows, 1, 1),
        output_shapes=[matrix_BHNCC.shape], output_dtypes=[mx.bfloat16],
    )[0]


@lru_cache(maxsize=1)
def _inverse_kernel() -> Any:
    """
    Purpose: Invert unit lower-triangular matrices by row substitution.
    Input: matrix (batches, C, C) BF16. Output: inverse (batches, C, C) BF16.
    Method: Thread j owns column j of the inverse X. Row by row, every thread
        sets X[i][j] = [i == j] - sum(L[i][k] X[k][j]) for k < i, in FP32. All
        threads read the same L[i][k], and X stays in threadgroup memory.
    """
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_unit_inverse_v3",
        input_names=["matrix"], output_names=["inverse"],
        source="""
            const uint column = thread_position_in_threadgroup.x;
            const uint batch = threadgroup_position_in_grid.y;
            const device bfloat* lower = matrix + batch * C * C;
            threadgroup float x[C * C];
            for (uint i = 0; i < C; ++i) {
                // Four partial sums keep four independent FMA chains in flight.
                float4 sums = 0.0f;
                uint k = 0;
                for (; k + 4 <= i; k += 4) {
                    sums -= float4(lower[i * C + k], lower[i * C + k + 1],
                                   lower[i * C + k + 2], lower[i * C + k + 3])
                        * float4(x[k * C + column], x[(k + 1) * C + column],
                                 x[(k + 2) * C + column], x[(k + 3) * C + column]);
                }
                float value = (i == column ? 1.0f : 0.0f) + (sums.x + sums.y)
                    + (sums.z + sums.w);
                for (; k < i; ++k) value -= float(lower[i * C + k]) * x[k * C + column];
                x[i * C + column] = value;
                inverse[(batch * C + i) * C + column] = bfloat(value);
            }
        """,
    )


@lru_cache(maxsize=1)
def _scalar_scan_forward_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_chunk_scan_forward_v1",
        input_names=["mu0", "q", "u0", "kt", "k", "decay", "ratio", "initial"],
        output_names=["output", "final_state", "history"],
        source="""
            int lane = int(thread_position_in_grid.x);
            int dv = int(thread_position_in_grid.y);
            int bh = int(thread_position_in_grid.z);
            int chunks = int(q_shape[2]);
            constexpr int per_lane = DK / 32;
            T state[per_lane];
            T update[C];
            for (int i = 0; i < per_lane; ++i) {
                state[i] = initial[(bh * DV + dv) * DK + lane * per_lane + i];
            }
            for (int c = 0; c < chunks; ++c) {
                int block = bh * chunks + c;
                for (int i = 0; i < per_lane; ++i) {
                    history[(block * DV + dv) * DK + lane * per_lane + i] = state[i];
                }
                for (int t = 0; t < C; ++t) {
                    float output_dot = 0.0f;
                    float update_dot = 0.0f;
                    for (int i = 0; i < per_lane; ++i) {
                        int index = (block * C + t) * DK + lane * per_lane + i;
                        output_dot += float(q[index]) * float(state[i]);
                        update_dot += float(kt[index]) * float(state[i]);
                    }
                    output_dot = simd_sum(output_dot);
                    update_dot = simd_sum(update_dot);
                    int row = (block * C + t) * DV + dv;
                    update[t] = T(float(u0[row]) - float(T(update_dot)));
                    if (lane == 0) {
                        output[row] = T(float(mu0[row]) + float(T(output_dot)));
                    }
                }
                for (int i = 0; i < per_lane; ++i) {
                    float delta = 0.0f;
                    for (int t = 0; t < C; ++t) {
                        T weighted = T(float(ratio[block * C + t]) * float(update[t]));
                        int index = (block * C + t) * DK + lane * per_lane + i;
                        delta += float(weighted) * float(k[index]);
                    }
                    T decayed = T(float(decay[block]) * float(state[i]));
                    state[i] = T(float(decayed) + float(T(delta)));
                }
            }
            for (int i = 0; i < per_lane; ++i) {
                final_state[(bh * DV + dv) * DK + lane * per_lane + i] = state[i];
            }
        """,
    )


@lru_cache(maxsize=1)
def _scalar_scan_backward_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="shellsmith_bf16_chunk_scan_backward_v1",
        input_names=["q", "u0", "kt", "k", "decay", "ratio", "history", "dy", "dfinal"],
        output_names=["du", "ds", "dw", "uc", "dinitial"],
        source="""
            int lane = int(thread_position_in_grid.x);
            int dv = int(thread_position_in_grid.y);
            int bh = int(thread_position_in_grid.z);
            int chunks = int(q_shape[2]);
            constexpr int per_lane = DK / 32;
            T adjoint[per_lane];
            T local_du[C];
            for (int i = 0; i < per_lane; ++i) {
                adjoint[i] = dfinal[(bh * DV + dv) * DK + lane * per_lane + i];
            }
            for (int c = chunks - 1; c >= 0; --c) {
                int block = bh * chunks + c;
                for (int i = 0; i < per_lane; ++i) {
                    ds[(block * DV + dv) * DK + lane * per_lane + i] = adjoint[i];
                }
                for (int t = 0; t < C; ++t) {
                    float weighted_dot = 0.0f;
                    float memory_dot = 0.0f;
                    for (int i = 0; i < per_lane; ++i) {
                        int key = lane * per_lane + i;
                        int index = (block * C + t) * DK + key;
                        weighted_dot += float(k[index]) * float(adjoint[i]);
                        memory_dot += float(kt[index]) * float(history[(block * DV + dv) * DK + key]);
                    }
                    T weighted = T(simd_sum(weighted_dot));
                    T memory = T(simd_sum(memory_dot));
                    int row = (block * C + t) * DV + dv;
                    local_du[t] = T(float(ratio[block * C + t]) * float(weighted));
                    if (lane == 0) {
                        dw[row] = weighted;
                        uc[row] = T(float(u0[row]) - float(memory));
                        du[row] = local_du[t];
                    }
                }
                for (int i = 0; i < per_lane; ++i) {
                    float hidden_dot = 0.0f;
                    float memory_dot = 0.0f;
                    int key = lane * per_lane + i;
                    for (int t = 0; t < C; ++t) {
                        int index = (block * C + t) * DK + key;
                        int row = (block * C + t) * DV + dv;
                        hidden_dot += float(q[index]) * float(dy[row]);
                        memory_dot += float(kt[index]) * float(local_du[t]);
                    }
                    T local = T(float(T(hidden_dot)) - float(T(memory_dot)));
                    T decayed = T(float(decay[block]) * float(adjoint[i]));
                    adjoint[i] = T(float(decayed) + float(local));
                }
            }
            for (int i = 0; i < per_lane; ++i) {
                dinitial[(bh * DV + dv) * DK + lane * per_lane + i] = adjoint[i];
            }
        """,
    )


@lru_cache(maxsize=2)
def _nax_kernel(phase: str) -> Any:
    """
    Purpose: Compile the production matrix-based state scan from immutable sources.
    Input: The forward or backward phase. Output: A source-keyed Metal kernel.
    Note: Wider accumulators exist only in registers and private threadgroup scratch.
    """
    root = Path(__file__).with_name("metal")
    header = (root / "gdn_nax_common.metal").read_text()
    source = (root / f"gdn_scan_{phase}.metal").read_text()
    identity = hashlib.sha256((header+source).encode()).hexdigest()[:12]
    if phase == "forward":
        inputs = ["mu0", "q", "u0", "kt", "k", "decay", "ratio", "initial"]
        outputs = ["output", "final_state", "history"]
    else:
        inputs = ["q", "u0", "kt", "k", "decay", "ratio", "history", "dy", "dfinal"]
        outputs = ["du", "ds", "dw", "uc", "dinitial"]
    return mx.fast.metal_kernel(
        name=f"shellsmith_gdn_nax_{phase}_{identity}", input_names=inputs,
        output_names=outputs, header=header, source=source,
    )


def _dispatch_scan(phase: str, arguments: dict) -> Any:
    """
    Purpose: Select the admitted production tile or the explicit small-shape reference.
    Input: BF16 scan arguments. Output: The three forward or five reverse tensors.
    Note: The admitted 128-by-128 model always uses the matrix kernel.
    """
    dimensions = dict(arguments["template"])
    keys, values, rows = (dimensions[name] for name in ("DK", "DV", "C"))
    if keys % 64 or values % 16 or rows % 16:
        kernel = (_scalar_scan_forward_kernel() if phase == "forward"
                  else _scalar_scan_backward_kernel())
        return kernel(**arguments)
    groups = arguments["grid"][2]
    arguments["grid"] = ((values//16)*64, 1, groups)
    arguments["threadgroup"] = (64, 1, 1)
    return _nax_kernel(phase)(**arguments)


def _scan_forward_kernel() -> Any:
    return lambda **arguments: _dispatch_scan("forward", arguments)


def _scan_backward_kernel() -> Any:
    return lambda **arguments: _dispatch_scan("backward", arguments)
