"""
Purpose: Run the sorted routed-expert 4-bit projection with a fast input gradient.
Input: Routed rows sorted by expert, the expert IDs, and a frozen quantized base.
Output: The same values as mx.gather_qmm, with a custom VJP for the input rows.
Method:
    1. The forward pass calls mx.gather_qmm with sorted indices.
    2. The VJP runs one Metal kernel. The kernel dequantizes each expert weight
       tile once per tile of 64 rows and uses NAX fragments for dy W.
    3. sorted_lora_projections applies one or two LoRA projections of the same
       rows. Its VJP writes the input gradient of all base and LoRA terms with
       one kernel, and the factor gradients with segmented matmuls.
Note: The base weights are frozen, so the VJP returns no weight gradient. The
    MLX gradient of a sorted gather_qmm with transpose=False is slow.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlx.core as mx

# R: routed rows, E: experts, I: input width, O: output width.
SHADER_ROOT = Path(__file__).with_name("metal")
TILE_ROWS = 64
TILE_COLUMNS = 128
STEP_ROWS = 32
THREADS = 128


def sorted_expert_qmm(x_R1I: Any, base: Any, experts_R: Any) -> Any:
    """
    Purpose: Apply the quantized expert projection to rows sorted by expert.
    Input: x (R, 1, I) BF16, a QuantizedSwitchLinear base, and sorted uint32 IDs.
    Output: y (R, 1, O) with the custom input gradient.
    Note: Shapes that the kernel tiles do not cover use the MLX base call.
    """
    if not _is_supported(x_R1I, base):
        return base(x_R1I, experts_R, sorted_indices=True)
    value_R1O = _qmm(x_R1I, base.weight, base.scales, base.biases, experts_R)
    if hasattr(base, "bias"):
        value_R1O = value_R1O + mx.expand_dims(base.bias[experts_R], -2)
    return value_R1O


@mx.custom_function
def _qmm(x_R1I, weight, scales, biases, experts_R):
    return mx.gather_qmm(
        x_R1I, weight, scales, biases, rhs_indices=experts_R, transpose=True,
        group_size=x_R1I.shape[-1] // scales.shape[-1], bits=4, sorted_indices=True,
    )


@_qmm.vjp
def _qmm_vjp(primals: tuple, cotangents: Any, outputs: Any) -> tuple:
    x_R1I, weight, scales, biases, experts_R = primals
    dy_R1O = cotangents[0] if isinstance(cotangents, (list, tuple)) else cotangents
    dx_RI = sorted_input_gradient(
        dy_R1O[:, 0].astype(mx.bfloat16), weight, scales, biases, experts_R
    )
    return (dx_RI[:, None].astype(x_R1I.dtype), mx.zeros_like(weight),
            mx.zeros_like(scales), mx.zeros_like(biases), mx.zeros_like(experts_R))


def sorted_lora_projections(x_R1I: Any, modules: list, experts_R: Any) -> list:
    """
    Purpose: Apply routed-expert LoRA projections that read the same sorted rows.
    Input: x (R, 1, I) BF16 rows sorted by expert, one or two SwitchLoRALinear
        modules with the same output width and scale, and sorted uint32 IDs.
    Output: One (R, 1, O) result per module, equal to module(x, experts, True).
    Note: Shapes that the kernel tiles do not cover use the module calls.
    """
    assert 1 <= len(modules) <= 2
    if not _is_lora_supported(x_R1I, modules):
        return [module(x_R1I, experts_R, sorted_indices=True) for module in modules]
    factors = [factor for module in modules for factor in (module.a, module.b)]
    return _lora_function([module.base for module in modules], modules[0].scale)(
        x_R1I, experts_R, *factors
    )


def _lora_function(bases: list, scale: float) -> Any:
    """Return the fused LoRA projections over these frozen bases."""
    group_size = bases[0].group_size

    @mx.custom_function
    def project(x_R1I, experts_R, *factors):
        outputs = []
        for base, a_ERI, b_EOR in zip(bases, factors[::2], factors[1::2]):
            value_R1O = mx.gather_qmm(
                x_R1I, base.weight, base.scales, base.biases, rhs_indices=experts_R,
                transpose=True, group_size=group_size, bits=4, sorted_indices=True,
            )
            update_R1O = _lora_update(x_R1I, a_ERI, b_EOR, experts_R, scale)
            outputs.append(value_R1O + update_R1O.astype(value_R1O.dtype))
        return outputs

    @project.vjp
    def project_vjp(primals: tuple, cotangents: Any, outputs: Any) -> tuple:
        x_R1I, experts_R, *factors = primals
        rows = x_R1I.shape[0]
        experts = factors[0].shape[0]
        counts_E = mx.zeros((experts,), mx.int32).at[experts_R.astype(mx.int32)].add(1)
        ends_E = mx.cumsum(counts_E)
        segments_E2 = mx.stack([ends_E - counts_E, ends_E], axis=1)
        x_RI = x_R1I[:, 0]
        dys, dlows, d_factors = [], [], []
        for a_ERI, b_EOR, cotangent in zip(factors[::2], factors[1::2], cotangents):
            dy_RO = cotangent[:, 0].astype(mx.bfloat16)
            # The scale multiplies the rank-16 rows, not the (R, O) rows.
            low_RR = scale * mx.gather_mm(
                x_R1I, mx.swapaxes(a_ERI, -1, -2), rhs_indices=experts_R,
                sorted_indices=True,
            )[:, 0]
            dlow_RR = scale * mx.gather_mm(
                dy_RO[:, None], b_EOR, rhs_indices=experts_R, sorted_indices=True
            )[:, 0]
            d_b = mx.segmented_mm(dy_RO.T, low_RR, segments_E2)
            d_a = mx.segmented_mm(dlow_RR.T, x_RI, segments_E2)
            dys.append(dy_RO)
            dlows.append(dlow_RR)
            d_factors += [d_a.astype(a_ERI.dtype), d_b.astype(b_EOR.dtype)]
        dx_RI = _lora_input_gradient(dys, bases, factors[::2], dlows, counts_E)
        assert dx_RI.shape == (rows, x_R1I.shape[-1])
        return (dx_RI[:, None].astype(x_R1I.dtype), mx.zeros_like(experts_R),
                *d_factors)

    return project


def _lora_update(
    x_R1I: Any, a_ERI: Any, b_EOR: Any, experts_R: Any, scale: float
) -> Any:
    a_EIR = mx.swapaxes(a_ERI.astype(x_R1I.dtype), -1, -2)
    b_ERO = mx.swapaxes(b_EOR.astype(x_R1I.dtype), -1, -2)
    low_R1R = mx.gather_mm(x_R1I, a_EIR, rhs_indices=experts_R, sorted_indices=True)
    return mx.gather_mm(scale * low_R1R, b_ERO, rhs_indices=experts_R,
                        sorted_indices=True)


def _lora_input_gradient(
    dys: list, bases: list, lora_as: list, dlows: list, counts_E: Any
) -> Any:
    """Return dx = sum_p dy_p W_p[e] + dlow_p A_p[e] with one kernel."""
    rows, output_width = dys[0].shape
    experts, _, packed = bases[0].weight.shape
    input_width = packed * 8
    pairs = len(dys)
    zero_1 = mx.zeros((1,), mx.int32)
    row_offsets = mx.concatenate([zero_1, mx.cumsum(counts_E)])
    tiles_E = (counts_E + TILE_ROWS - 1) // TILE_ROWS
    tile_offsets = mx.concatenate([zero_1, mx.cumsum(tiles_E)])
    max_tiles = (rows + experts * (TILE_ROWS - 1)) // TILE_ROWS
    dims_4 = mx.array([rows, output_width, input_width, experts], mx.int32)
    inputs = []
    for p in (0, min(1, pairs - 1)):
        base = bases[p]
        inputs += [dys[p], base.weight, base.scales, base.biases,
                   lora_as[p].astype(mx.bfloat16)]
    if pairs == 1:
        dlows = dlows + [mx.zeros_like(dlows[0])]
    dlow_RP = mx.concatenate(dlows, axis=1).astype(mx.bfloat16)
    return _lora_kernel()(
        inputs=inputs + [dlow_RP, row_offsets, tile_offsets, dims_4],
        template=[("GS", bases[0].group_size), ("PAIRS", pairs)],
        grid=(input_width // TILE_COLUMNS * THREADS, max_tiles, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, input_width)],
        output_dtypes=[mx.bfloat16],
    )[0]


def sorted_input_gradient(
    dy_RO: Any, weight: Any, scales: Any, biases: Any, experts_R: Any
) -> Any:
    """
    Purpose: Calculate dx = dy W[e] for rows sorted by expert e.
    Input: dy (R, O) BF16, the packed 4-bit weights (E, O, I / 8), scales and
        biases (E, O, I / group size), and sorted expert IDs (R).
    Output: dx (R, I) BF16.
    """
    rows, output_width = dy_RO.shape
    experts, weight_rows, packed = weight.shape
    input_width = packed * 8
    group_size = input_width // scales.shape[-1]
    assert weight_rows == output_width and experts_R.shape == (rows,)
    assert output_width % STEP_ROWS == 0 and input_width % TILE_COLUMNS == 0
    assert group_size in (32, 64, 128) and dy_RO.dtype == mx.bfloat16
    counts_E = mx.zeros((experts,), mx.int32).at[experts_R.astype(mx.int32)].add(1)
    zero_1 = mx.zeros((1,), mx.int32)
    row_offsets = mx.concatenate([zero_1, mx.cumsum(counts_E)])
    tiles_E = (counts_E + TILE_ROWS - 1) // TILE_ROWS
    tile_offsets = mx.concatenate([zero_1, mx.cumsum(tiles_E)])
    # Each expert adds at most one partial tile.
    max_tiles = (rows + experts * (TILE_ROWS - 1)) // TILE_ROWS
    dims_4 = mx.array([rows, output_width, input_width, experts], mx.int32)
    return _kernel()(
        inputs=[dy_RO, weight, scales, biases, row_offsets, tile_offsets, dims_4],
        template=[("GS", group_size)],
        grid=(input_width // TILE_COLUMNS * THREADS, max_tiles, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(rows, input_width)],
        output_dtypes=[mx.bfloat16],
    )[0]


def _is_supported(x_R1I: Any, base: Any) -> bool:
    input_width = x_R1I.shape[-1]
    return (
        base.bits == 4 and base.mode == "affine" and base.biases is not None
        and base.group_size in (32, 64, 128) and x_R1I.ndim == 3
        and x_R1I.shape[1] == 1 and x_R1I.dtype == mx.bfloat16
        and input_width % TILE_COLUMNS == 0 and base.weight.shape[1] % STEP_ROWS == 0
    )


def _is_lora_supported(x_R1I: Any, modules: list) -> bool:
    # SwitchLoRALinear imports this module, so the check uses its attributes.
    names = ("base", "a", "b", "scale")
    if not all(hasattr(module, name) for module in modules for name in names):
        return False
    bases = [module.base for module in modules]
    first = modules[0]
    return (
        all(_is_supported(x_R1I, base) and not hasattr(base, "bias") for base in bases)
        and all(module.scale == first.scale for module in modules)
        and all(module.a.shape[1] == 16 for module in modules)
        and all(base.weight.shape[1] == bases[0].weight.shape[1] for base in bases)
        and all(base.group_size == bases[0].group_size for base in bases)
    )


@lru_cache(maxsize=1)
def _lora_kernel() -> Any:
    """Compile the fused LoRA input-gradient shader, keyed by its source hash."""
    header = (SHADER_ROOT / "gdn_nax_common.metal").read_text()
    header += (SHADER_ROOT / "expert_sorted_common.metal").read_text()
    source = (SHADER_ROOT / "expert_sorted_lora_dx.metal").read_text()
    identity = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    names = []
    for p in range(2):
        names += [f"dy{p}", f"w{p}", f"scales{p}", f"biases{p}", f"lora{p}"]
    return mx.fast.metal_kernel(
        name=f"shellsmith_expert_sorted_lora_dx_{identity}",
        input_names=names + ["dlow", "row_offsets", "tile_offsets", "dims"],
        output_names=["dx"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _kernel() -> Any:
    """Compile the repository-owned sorted-expert shader, keyed by its source hash."""
    header = (SHADER_ROOT / "gdn_nax_common.metal").read_text()
    header += (SHADER_ROOT / "expert_sorted_common.metal").read_text()
    source = (SHADER_ROOT / "expert_sorted_dx.metal").read_text()
    identity = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"shellsmith_expert_sorted_dx_{identity}",
        input_names=["dy", "w", "scales", "biases", "row_offsets", "tile_offsets",
                     "dims"],
        output_names=["dx"],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )
