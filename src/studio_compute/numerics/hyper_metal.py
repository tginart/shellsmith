"""
Purpose: Run the Flash hyper-connection of training replay with fused Metal norms.
Input: One Qwen4ExpGatedResidual module and its BF16 hyper input.
Output: The mixed input, the hyper input, and the injection weights, as the
    module returns them, with a custom VJP for the grouped RMS norm.
Method:
    1. One kernel calculates the grouped RMS norm and keeps the inverse RMS.
    2. A second kernel calculates the norm input gradient from that inverse RMS.
    3. The low-rank mix, the stream mix, and the injection gate use the module
       weights and the upstream compiled pointwise functions.
    4. With the injection, one custom VJP covers the connection up to its
       normed rows. One norm backward kernel adds the three normed-row
       gradients in registers.
Note: The upstream norm runs several unfused FP32 passes. The kernels use the
    same FP32 formula, so the results differ only in the order of the sum.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm.models.qwen4_exp import language as qwen4

# B: batch, L: rows, W: hyper width (G * D), G: streams, D: hidden width.
SHADER_ROOT = Path(__file__).with_name("metal")
THREADS = 256
NORM_THREADS = (256, 128, 64, 32)
# Threads per row of the injection VJP kernel.
INJECT_THREADS = 128


def hyper_connection(module: Any, hyper_BLW: Any) -> tuple:
    """
    Purpose: Apply one gated residual hyper-connection.
    Input: The module and the hyper input (B, L, W) BF16.
    Output: (mixed (B, L, D), hyper input, injection weights (B, L, G)).
    """
    assert hyper_BLW.dtype == mx.bfloat16
    norm = module.hc_norm
    assert norm.group_size == module.hidden_size
    if "block_inject_weight" in module:
        mixed_BLD, injection_BLG = _fused_connection(module, hyper_BLW)
        return mixed_BLD, hyper_BLW, injection_BLG
    eps_1 = mx.array([norm.eps], mx.float32)
    normed_BLW = _norm_function(module.hidden_size)(hyper_BLW, norm.weight, eps_1)
    activate, _, injection_gate = _pointwise(
        module.hc_count, module.hidden_size
    )
    mix_BLR = activate(module.input_mix_weight_down(normed_BLW))
    mixed_BLD = _mix_function(module.hc_count, module.hidden_size)(
        module.input_mix_weight_up(mix_BLR), normed_BLW
    )
    return mixed_BLD


def _fused_connection(module: Any, hyper_BLW: Any) -> tuple:
    """
    Purpose: Apply the mix and injection paths of a connection under one VJP.
    Input: The module and the hyper input (B, L, W) BF16.
    Output: (mixed (B, L, D), injection weights (B, L, G)).
    Method:
        1. The forward also returns the normed rows, the inverse RMS, the
           projected logits, and the low-rank and injection logits. The VJP
           reads them, so it runs no forward pass again.
        2. The VJP runs the mix kernel, the two plain dy W projections, and one
           norm kernel that sums the normed-row gradients.
    Note: The custom function does not return its input. MLX adds the
        passthrough gradient of the hyper input outside it.
    """
    width, streams = module.hidden_size, module.hc_count
    norm = module.hc_norm
    down, up = module.input_mix_weight_down, module.input_mix_weight_up
    inject_linear = module.block_inject_weight
    activate, _, injection_gate = _pointwise(streams, width)
    mix = _mix_function(streams, width)
    eps_1 = mx.array([norm.eps], mx.float32)
    # The VJP can see its primal without unit leading axes, so the gradient
    # takes the shape of the caller's input.
    hyper_shape = hyper_BLW.shape

    @mx.custom_function
    def connection(hyper_BLW):
        normed_BLW, inverse_NG = _norm_forward(hyper_BLW, norm.weight, eps_1, width)
        low_BLR = down(normed_BLW)
        projected_BLW = up(activate(low_BLR))
        injection_BLG = qwen4._row_invariant_quantized_projection(
            inject_linear, normed_BLW
        )
        mixed_BLD = mix(projected_BLW, normed_BLW)
        return (mixed_BLD, injection_gate(injection_BLG), normed_BLW, inverse_NG,
                projected_BLW, low_BLR, injection_BLG)

    @connection.vjp
    def connection_vjp(primals: tuple, cotangents: Any, outputs: Any) -> tuple:
        (hyper_BLW,) = primals
        d_mixed_BLD, d_gate_BLG = cotangents[:2]
        _, _, normed_BLW, inverse_NG, projected_BLW, low_BLR, injection_BLG = outputs
        d_injection_BLG = mx.vjp(injection_gate, [injection_BLG], [d_gate_BLG])[1][0]
        d_projected_BLW, d_normed_BLW = mx.vjp(
            mix, [projected_BLW, normed_BLW], [d_mixed_BLD]
        )[1]
        d_mix_BLR = _input_gradient(up, d_projected_BLW)
        d_low_BLR = mx.vjp(activate, [low_BLR], [d_mix_BLR])[1][0]
        d_down_BLW = _input_gradient(down, d_low_BLR)
        inject_GW = mx.dequantize(
            inject_linear.weight, inject_linear.scales, inject_linear.biases,
            group_size=inject_linear.group_size, bits=inject_linear.bits,
            mode=inject_linear.mode,
        ).astype(mx.bfloat16)
        bf16 = [d_normed_BLW, d_down_BLW, d_injection_BLG]
        d_hyper_BLW = _kernel("norm_backward_sum")(
            inputs=[hyper_BLW, norm.weight, inverse_NG,
                    *[value.astype(mx.bfloat16) for value in bf16], inject_GW],
            template=_template(norm.weight, width),
            grid=(inverse_NG.size * _threads(width), 1, 1),
            threadgroup=(_threads(width), 1, 1),
            output_shapes=[hyper_shape],
            output_dtypes=[mx.bfloat16],
        )[0]
        return (d_hyper_BLW,)

    return tuple(connection(hyper_BLW)[:2])


def _input_gradient(linear: Any, dy: Any) -> Any:
    """Return dy W for a frozen quantized linear."""
    return mx.quantized_matmul(
        dy.astype(mx.bfloat16), linear.weight, linear.scales, linear.biases,
        transpose=False, group_size=linear.group_size, bits=linear.bits,
        mode=linear.mode,
    )


def inject(hyper_BLW: Any, output_BLD: Any, injection_BLG: Any) -> Any:
    """
    Purpose: Return hyper + injection * output, broadcast over the G streams.
    Input: The hyper residual (B, L, G * D), the branch output (B, L, D), and the
        injection weights (B, L, G), all BF16.
    Output: (B, L, G * D) BF16, with one fused kernel and a one-pass VJP.
    Note: Other dtypes and widths that are not a multiple of 4 use MLX operations.
    """
    width = output_BLD.shape[-1]
    streams = injection_BLG.shape[-1]
    assert hyper_BLW.shape[-1] == streams * width
    values = (hyper_BLW, output_BLD, injection_BLG)
    if width % 4 or any(value.dtype != mx.bfloat16 for value in values):
        injected = output_BLD[..., None, :] * injection_BLG[..., None]
        return hyper_BLW + injected.reshape(hyper_BLW.shape)
    return _inject_function(streams, width)(*values)


@lru_cache(maxsize=None)
def _inject_function(streams: int, width: int) -> Any:
    """Return the residual injection with its custom VJP for one shape."""
    template = [("G", streams), ("D", width), ("TG", INJECT_THREADS)]

    @mx.custom_function
    def injected(hyper_BLW, output_BLD, injection_BLG):
        return _kernel("inject_forward")(
            inputs=[hyper_BLW, output_BLD, injection_BLG], template=template,
            grid=(hyper_BLW.size // 4, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=[hyper_BLW.shape], output_dtypes=[mx.bfloat16],
        )[0]

    @injected.vjp
    def injected_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
        hyper_BLW, output_BLD, injection_BLG = primals
        rows = output_BLD.size // width
        d_output, d_injection = _kernel("inject_backward")(
            inputs=[output_BLD, injection_BLG, cotangent.astype(mx.bfloat16)],
            template=template, grid=(rows * INJECT_THREADS, 1, 1),
            threadgroup=(INJECT_THREADS, 1, 1),
            output_shapes=[output_BLD.shape, injection_BLG.shape],
            output_dtypes=[mx.bfloat16, mx.float32],
        )
        return cotangent, d_output, d_injection.astype(injection_BLG.dtype)

    return injected


def _injection(linear: Any, normed_BLW: Any) -> Any:
    """
    Purpose: Keep the row-invariant injection forward with a plain input gradient.
    Input: The quantized injection linear and the normed rows.
    Output: The (B, L, G) projection.
    Note: The derivative of the row-invariant reduction is the plain dy W.
    """

    @mx.custom_function
    def project(normed_BLW):
        return qwen4._row_invariant_quantized_projection(linear, normed_BLW)

    @project.vjp
    def project_vjp(primals: tuple, cotangent: Any, output: Any) -> Any:
        return mx.quantized_matmul(
            cotangent.astype(primals[0].dtype), linear.weight, linear.scales,
            linear.biases, transpose=False, group_size=linear.group_size,
            bits=linear.bits, mode=linear.mode,
        )

    return project(normed_BLW)


@lru_cache(maxsize=None)
def _mix_function(hc_count: int, hidden_size: int) -> Any:
    """Return the stream mix with one forward kernel and one VJP kernel."""
    mix_streams = _pointwise(hc_count, hidden_size)[1]
    template = [("G", hc_count), ("D", hidden_size)]

    @mx.custom_function
    def mix(projected_BLW, normed_BLW):
        values = (projected_BLW, normed_BLW)
        if hidden_size % 4 or any(value.dtype != mx.bfloat16 for value in values):
            return mix_streams(projected_BLW, normed_BLW)
        rows = projected_BLW.size // (hc_count * hidden_size)
        return _kernel("mix_forward")(
            inputs=[projected_BLW, normed_BLW], template=template,
            grid=(rows * hidden_size // 4, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=[(*projected_BLW.shape[:-1], hidden_size)],
            output_dtypes=[mx.bfloat16],
        )[0]

    @mix.vjp
    def mix_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
        projected_BLW, normed_BLW = primals
        assert projected_BLW.dtype == normed_BLW.dtype == mx.bfloat16
        d_projected, d_normed = _kernel("mix_backward")(
            inputs=[projected_BLW, normed_BLW, cotangent],
            template=template,
            grid=(cotangent.size, 1, 1),
            threadgroup=(THREADS, 1, 1),
            output_shapes=[projected_BLW.shape, normed_BLW.shape],
            output_dtypes=[mx.bfloat16, mx.bfloat16],
        )
        return d_projected, d_normed

    return mix


@lru_cache(maxsize=None)
def _pointwise(hc_count: int, hidden_size: int) -> tuple:
    """Reuse one set of compiled upstream pointwise functions per shape."""
    return qwen4._qwen4_hyper_pointwise(hc_count, hidden_size)


@lru_cache(maxsize=None)
def _norm_function(width: int) -> Any:
    """Return the grouped RMS norm with a custom VJP for group width D."""

    @mx.custom_function
    def norm(hyper_BLW, weight_W, eps_1):
        return _norm_forward(hyper_BLW, weight_W, eps_1, width)[0]

    @norm.vjp
    def norm_vjp(primals: tuple, cotangent: Any, output: Any) -> tuple:
        hyper_BLW, weight_W, eps_1 = primals
        # The inverse RMS is cheap to recompute, so no pass keeps it.
        _, inverse_NG = _norm_forward(hyper_BLW, weight_W, eps_1, width)
        dx_BLW = _kernel("norm_backward")(
            inputs=[hyper_BLW, weight_W, inverse_NG, cotangent],
            template=_template(weight_W, width),
            grid=(inverse_NG.size * _threads(width), 1, 1),
            threadgroup=(_threads(width), 1, 1),
            output_shapes=[hyper_BLW.shape],
            output_dtypes=[mx.bfloat16],
        )[0]
        return (dx_BLW.astype(cotangent.dtype), mx.zeros_like(weight_W),
                mx.zeros_like(eps_1))

    return norm


def _norm_forward(hyper_BLW: Any, weight_W: Any, eps_1: Any, width: int) -> tuple:
    """Run the norm kernel. Return the BF16 norm and the FP32 inverse RMS (N, G)."""
    assert hyper_BLW.shape[-1] == weight_W.shape[0]
    groups = weight_W.shape[0] // width
    pairs = hyper_BLW.size // width
    return tuple(_kernel("norm_forward")(
        inputs=[hyper_BLW, weight_W, eps_1],
        template=_template(weight_W, width),
        grid=(pairs * _threads(width), 1, 1),
        threadgroup=(_threads(width), 1, 1),
        output_shapes=[hyper_BLW.shape, (pairs // groups, groups)],
        output_dtypes=[mx.bfloat16, mx.float32],
    ))


def _template(weight_W: Any, width: int) -> list:
    assert weight_W.shape[0] % width == 0
    return [("G", weight_W.shape[0] // width), ("D", width), ("TG", _threads(width))]


def _threads(width: int) -> int:
    """Return the largest norm threadgroup that divides the group width."""
    threads = [count for count in NORM_THREADS if width % count == 0]
    assert threads, "The group width must be a multiple of 32."
    return threads[0]


@lru_cache(maxsize=None)
def _kernel(phase: str) -> Any:
    """Compile one repository-owned hyper-connection shader, keyed by its hash."""
    names = {
        "norm_forward": (["x", "w", "eps"], ["y", "r"]),
        "norm_backward": (["x", "w", "r", "dy"], ["dx"]),
        "norm_backward_sum": (["x", "w", "r", "dm", "dd", "di", "wi"], ["dx"]),
        "mix_forward": (["u", "n"], ["m"]),
        "mix_backward": (["u", "n", "dm"], ["du", "dn"]),
        "inject_forward": (["h", "o", "w"], ["y"]),
        "inject_backward": (["o", "w", "dy"], ["d_o", "d_w"]),
    }[phase]
    source = (SHADER_ROOT / f"hyper_{phase}.metal").read_text()
    identity = hashlib.sha256(source.encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"shellsmith_hyper_{phase}_{identity}",
        input_names=names[0],
        output_names=names[1],
        source=source,
        ensure_row_contiguous=True,
    )
