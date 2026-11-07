"""
Purpose: Fused Metal kernels for the elementwise parts of the GDN training layer.
Input: BF16 GDN rows of one replay segment and the frozen layer weights.
Output: The same values as the MLX operations they replace, with custom VJPs.
Method:
    1. causal_conv_silu applies the depthwise causal convolution and the SiLU
       in one pass. Its VJP recomputes the pre-activations and gives the
       gradients of the carried taps and of the projected rows.
    2. gated_rms_norm applies rms_norm(y) * w * gate(z) in one pass, and its
       VJP gives dy and dz in one pass.
    3. gated_delta_chunks runs the chunked gated delta rule. One kernel makes
       all intra-chunk transforms of a chunk from q, k, v and the gate and beta
       logits. The existing Metal state scan then crosses the chunks.
Note: Each kernel keeps FP32 values and rounds only its outputs. The MLX
    operations round after each step, so the results differ only in rounding.
    The convolution and norm weights are frozen; their gradients are zeros.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

# B: batch, L: rows, C: convolution channels, K: kernel taps, N: head rows,
# V: head width. Chunk layout: H value heads, M chunks, R chunk rows, D width.
SHADER_ROOT = Path(__file__).with_name("metal")
THREADS = 256
CHUNK = 64
HEAD_WIDTH = 128
CHUNK_THREADS = 128
NAMES = {
    "conv_forward": (["s", "x", "w"], ["y"]),
    "conv_backward": (["s", "x", "w", "dy"], ["ds", "dx"]),
    "norm_forward": (["y", "z", "w", "eps"], ["o"]),
    "norm_backward": (["y", "z", "w", "eps", "dout"], ["dy", "dz"]),
    "chunk_forward": (["q", "k", "v", "a", "b", "a_log", "dt_bias"],
                      ["mu0", "q_eff", "u0", "kt", "k_chunks", "decay_last",
                       "ratio_last"]),
    "chunk_backward": (["q", "k", "v", "a", "b", "a_log", "dt_bias", "u0", "kt",
                        "d_mu0", "d_q_eff", "d_u0", "d_kt", "d_k_chunks",
                        "d_decay_last", "d_ratio_last"],
                       ["dq", "dk", "dv", "da", "db", "s_du", "s_dkt", "s_dkb"]),
}
# The NAX MMA kernels share these headers.
CHUNK_HEADERS = ("gdn_nax_common.metal", "gdn_chunk_common.metal")


def causal_conv_silu(
    taps_BKC: Any, x_BLC: Any, weight_CK1: Any, *, dilation: int = 1
) -> Any:
    """
    Purpose: Apply silu(conv1d([taps; x])) for a depthwise causal convolution.
    Input: The carried taps (B, (K - 1) * dilation, C), the rows (B, L, C), and
        the MLX depthwise weight (C, K, 1), all BF16, and the dilation.
    Output: (B, L, C) BF16, the same rows as silu(conv1d(concatenate)).
    """
    assert taps_BKC.dtype == x_BLC.dtype == weight_CK1.dtype == mx.bfloat16
    channels, kernel, _ = weight_CK1.shape
    assert weight_CK1.shape[2] == 1 and taps_BKC.shape[1] == (kernel - 1) * dilation
    assert taps_BKC.shape[2] == x_BLC.shape[2] == channels and channels % 4 == 0
    return _conv_function(kernel, channels, dilation)(taps_BKC, x_BLC, weight_CK1)


def gated_rms_norm(y_BLHV: Any, z_BLHV: Any, weight_V: Any, eps: float, *,
                   sigmoid: bool) -> Any:
    """
    Purpose: Apply mx.fast.rms_norm(y, w, eps) * gate(z) over the last axis.
    Input: y and z (B, L, H, V) BF16, the BF16 weight (V), eps, and the gate:
        the sigmoid when sigmoid is true, else SiLU.
    Output: (B, L, H, V) BF16.
    """
    assert y_BLHV.dtype == z_BLHV.dtype == weight_V.dtype == mx.bfloat16
    assert y_BLHV.shape == z_BLHV.shape and y_BLHV.shape[-1] == weight_V.shape[0]
    width = weight_V.shape[0]
    assert width % 32 == 0
    eps_1 = mx.array([eps], mx.float32)
    flat = [value.reshape(-1, width) for value in (y_BLHV, z_BLHV)]
    output_NV = _norm_function(width, sigmoid)(*flat, weight_V, eps_1)
    return output_NV.reshape(y_BLHV.shape)


def gated_delta_chunks(q_BLGD: Any, k_BLGD: Any, v_BLHD: Any, a_BLH: Any,
                       b_BLH: Any, a_log_H: Any, dt_bias_H: Any,
                       state_BHDD: Any) -> tuple:
    """
    Purpose: Run the chunked gated delta rule of one GDN layer and segment.
    Input: Normalized q and k (B, L, G, 128) for G key heads, v (B, L, H, 128),
        the gate logits a and beta logits b (B, L, H), all BF16, the per-head
        A_log and dt_bias, and the BF16 state (B, H, 128, 128).
    Output: y (B, L, H, 128) BF16 and the final state, as gated_delta_chunked
        returns them for g = exp(-exp(A_log) softplus(a + dt_bias)) and
        beta = sigmoid(b).
    Method:
        1. Pad L to a multiple of 64 with rows that have g = 1 and beta = 0.
        2. One kernel per (batch, head, chunk) makes the chunk transforms.
        3. The Metal state scan crosses the chunks.
    """
    batch, length, key_heads, width = q_BLGD.shape
    heads = v_BLHD.shape[2]
    assert width == HEAD_WIDTH == v_BLHD.shape[3] and heads % key_heads == 0
    assert k_BLGD.shape == q_BLGD.shape and a_BLH.shape == b_BLH.shape
    assert all(value.dtype == mx.bfloat16
               for value in (q_BLGD, k_BLGD, v_BLHD, a_BLH, b_BLH, state_BHDD))
    pad = (-length) % CHUNK
    if pad:
        rows = [(0, 0), (0, pad)]
        q_BLGD, k_BLGD, v_BLHD = [mx.pad(value, [*rows, (0, 0), (0, 0)])
                                  for value in (q_BLGD, k_BLGD, v_BLHD)]
        # A logit of -inf gives softplus 0, so g = 1, and beta = 0.
        a_BLH, b_BLH = [mx.pad(value, [*rows, (0, 0)], constant_values=-float("inf"))
                        for value in (a_BLH, b_BLH)]
    from .gdn_metal import scan

    transforms = _chunk_function(heads // key_heads)(
        q_BLGD, k_BLGD, v_BLHD, a_BLH, b_BLH, a_log_H, dt_bias_H)
    y_BHMRD, final_BHDD = scan(*transforms, state_BHDD)
    y_BLHD = y_BHMRD.transpose(0, 2, 3, 1, 4).reshape(batch, length + pad, heads, width)
    return y_BLHD[:, :length], final_BHDD


@lru_cache(maxsize=None)
def _chunk_function(group: int) -> Any:
    """Return the fused chunk transforms with their custom VJP for one head group."""

    @mx.custom_function
    def transforms(q_BLGD, k_BLGD, v_BLHD, a_BLH, b_BLH, a_log_H, dt_bias_H):
        batch, length, heads, width = v_BLHD.shape
        chunks = length // CHUNK
        layout = (batch, heads, chunks, CHUNK, width)
        return tuple(_kernel("chunk_forward")(
            inputs=[q_BLGD, k_BLGD, v_BLHD, a_BLH, b_BLH, a_log_H, dt_bias_H],
            template=[("GROUP", group)],
            grid=(chunks * CHUNK_THREADS, heads, batch),
            threadgroup=(CHUNK_THREADS, 1, 1),
            output_shapes=[layout] * 5 + [(batch, heads, chunks),
                                          (batch, heads, chunks, CHUNK)],
            output_dtypes=[mx.bfloat16] * 7,
        ))

    @transforms.vjp
    def transforms_vjp(primals: tuple, cotangents: tuple, outputs: tuple) -> tuple:
        q_BLGD, k_BLGD, v_BLHD, a_BLH, b_BLH, a_log_H, dt_bias_H = primals
        u0_BHMRD, kt_BHMRD = outputs[2], outputs[3]
        batch, length, heads, width = v_BLHD.shape
        chunks = length // CHUNK
        layout = (batch, heads, chunks, CHUNK, width)
        dq_BLHD, dk_BLHD, dv_BLHD, da_BLH, db_BLH, *_ = _kernel("chunk_backward")(
            inputs=[q_BLGD, k_BLGD, v_BLHD, a_BLH, b_BLH, a_log_H, dt_bias_H,
                    u0_BHMRD, kt_BHMRD,
                    *[value.astype(mx.bfloat16) for value in cotangents]],
            template=[("GROUP", group)],
            grid=(chunks * CHUNK_THREADS, heads, batch),
            threadgroup=(CHUNK_THREADS, 1, 1),
            output_shapes=[v_BLHD.shape] * 3 + [a_BLH.shape] * 2 + [layout] * 3,
            output_dtypes=[mx.bfloat16] * 8,
        )
        key_heads = q_BLGD.shape[2]

        def fold(value_BLHD):
            # Each key head serves `group` consecutive value heads.
            grouped = value_BLHD.reshape(batch, length, key_heads, group, width)
            return mx.sum(grouped, axis=3)

        return (fold(dq_BLHD), fold(dk_BLHD), dv_BLHD, da_BLH, db_BLH,
                mx.zeros_like(a_log_H), mx.zeros_like(dt_bias_H))

    return transforms


@lru_cache(maxsize=None)
def _conv_function(kernel: int, channels: int, dilation: int) -> Any:
    """Return the fused convolution with its custom VJP for one shape."""
    template = [("K", kernel), ("C", channels), ("DIL", dilation)]

    @mx.custom_function
    def conv(taps_BKC, x_BLC, weight_CK1):
        return _kernel("conv_forward")(
            inputs=[taps_BKC, x_BLC, weight_CK1], template=template,
            grid=(x_BLC.size // 4, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=[x_BLC.shape], output_dtypes=[mx.bfloat16],
        )[0]

    @conv.vjp
    def conv_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
        taps_BKC, x_BLC, weight_CK1 = primals
        batch, length, _ = x_BLC.shape
        rows = length + (kernel - 1) * dilation
        dtaps_BKC, dx_BLC = _kernel("conv_backward")(
            inputs=[taps_BKC, x_BLC, weight_CK1, cotangent.astype(mx.bfloat16)],
            template=template, grid=(batch * rows * channels // 4, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[taps_BKC.shape, x_BLC.shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
        return dtaps_BKC, dx_BLC, mx.zeros_like(weight_CK1)

    return conv


@lru_cache(maxsize=None)
def _norm_function(width: int, sigmoid: bool) -> Any:
    """Return the fused gated norm with its custom VJP for one width and gate."""
    template = [("V", width), ("SIGMOID", sigmoid)]

    def launch(phase: str, inputs: list, outputs: int) -> list:
        rows = inputs[0].shape[0]
        return _kernel(phase)(
            inputs=inputs, template=template, grid=(rows * 32, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[inputs[0].shape] * outputs,
            output_dtypes=[mx.bfloat16] * outputs,
        )

    @mx.custom_function
    def norm(y_NV, z_NV, weight_V, eps_1):
        return launch("norm_forward", [y_NV, z_NV, weight_V, eps_1], 1)[0]

    @norm.vjp
    def norm_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
        y_NV, z_NV, weight_V, eps_1 = primals
        dy_NV, dz_NV = launch(
            "norm_backward",
            [y_NV, z_NV, weight_V, eps_1, cotangent.astype(mx.bfloat16)], 2)
        return dy_NV, dz_NV, mx.zeros_like(weight_V), mx.zeros_like(eps_1)

    return norm


@lru_cache(maxsize=None)
def _kernel(phase: str) -> Any:
    """Compile one repository-owned GDN shader, keyed by its source hash."""
    source = (SHADER_ROOT / f"gdn_{phase}.metal").read_text()
    header = ""
    if phase.startswith("chunk"):
        header = "".join((SHADER_ROOT / name).read_text() for name in CHUNK_HEADERS)
    identity = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    inputs, outputs = NAMES[phase]
    return mx.fast.metal_kernel(
        name=f"shellsmith_gdn_{phase}_{identity}", input_names=inputs,
        output_names=outputs, header=header, source=source,
        ensure_row_contiguous=True,
    )
