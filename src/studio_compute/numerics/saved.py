"""
Purpose: Differentiate decoder layers from explicit retained forward residuals.
Input: BF16 hidden tensors, carried state, and canonical frozen-base LoRA modules.
Output: Layer values, tapes, and input, carry, and adapter gradients.
Note: Small nonlinear graphs can reconstruct their own intermediates.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_vlm.models.activations import swiglu

from .bf16 import gdn_inner, prepare_attention
from . import saved_norm

# B: batch, L: token rows, D: hidden width, F: MLP width, R: adapter rank.
# H: attention heads, J: KV heads, K: head width.


@dataclass
class LayerTape:
    """
    Purpose: Own one layer's retained residuals until its final backward consumer.
    Input: Hidden inputs, carry references, and optional projection results.
    Output: A complete explicit backward input without a whole-layer retrace.
    """

    layer: Any
    hidden_BLD: Any
    middle_BLD: Any
    carries: tuple
    positions: Any
    factors: tuple
    retained: dict = field(default_factory=dict)
    low_rank: dict = field(default_factory=dict)
    scratch: dict = field(default_factory=dict)
    attention: Any = None
    local_kv: tuple = ()
    counters: dict = field(default_factory=dict)
    input_statistics: Any = None
    middle_statistics: Any = None


def forward_saved(
    layer: Any,
    hidden_BLD: Any,
    carries: list,
    positions: Any,
    *,
    retain: frozenset[str] = frozenset(),
    counters: dict | None = None,
    retain_hidden: bool = True,
    retain_middle: bool = True,
) -> tuple[Any, Any, Any, LayerTape]:
    """
    Purpose: Execute one BF16 layer and retain admitted projection results.
    Input: Layer inputs, immutable carries, positions, and selected projection names.
    Output: Hidden output, local carries, and a tape that owns useful forward work.
    """
    assert hidden_BLD.dtype == mx.bfloat16
    factors = tuple(
        (name, module, module.a, module.b) for name, module in _projections(layer)
    )
    tape = LayerTape(
        layer,
        hidden_BLD,
        None,
        tuple(carries),
        positions,
        factors,
        counters={} if counters is None else counters,
    )
    x_BLD, tape.input_statistics = saved_norm.forward_saved(
        hidden_BLD,
        layer.input_layernorm.weight.astype(mx.bfloat16),
        layer.input_layernorm.eps,
    )
    _count(tape, "input_norm_statistics", "forward")
    if layer.is_linear:
        names = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
        raw = [
            _forward_projection(tape, f"linear_attn.{name}", x_BLD, retain)
            for name in names
        ]
        _count(tape, "gdn_inner", "forward")
        mixer_BLD, first, second = gdn_inner(layer.linear_attn, *raw, *carries)
        mixed_BLD = _forward_projection(tape, "linear_attn.out_proj", mixer_BLD, retain)
    else:
        attention = layer.self_attn
        raw = [
            _forward_projection(tape, f"self_attn.{name}", x_BLD, retain)
            for name in ("q_proj", "k_proj", "v_proj")
        ]
        _count(tape, "attention_prepare", "forward")
        q_BHLK, k_BJLK, v_BJLK, gate_BLD = _prepare(attention, raw, positions)
        keys = mx.concatenate([*carries[0::2], k_BJLK], axis=2) if carries else k_BJLK
        values = mx.concatenate([*carries[1::2], v_BJLK], axis=2) if carries else v_BJLK
        from .attention_metal import attention_forward

        output, tape.attention = attention_forward(
            q_BHLK, keys, values, attention.scale, keys.shape[2] - q_BHLK.shape[2]
        )
        # Concatenated history is temporary. The original producer allocations persist.
        tape.attention = (output, tape.attention.statistics_BHQC)
        _count(tape, "attention_gate", "forward")
        mixer_BLD = _attention_gate(output, gate_BLD)
        mixed_BLD = _forward_projection(tape, "self_attn.o_proj", mixer_BLD, retain)
        first, second = k_BJLK, v_BJLK
        tape.local_kv = (first, second)
    tape.middle_BLD = hidden_BLD + mixed_BLD
    normalized_BLD, tape.middle_statistics = saved_norm.forward_saved(
        tape.middle_BLD,
        layer.post_attention_layernorm.weight.astype(mx.bfloat16),
        layer.post_attention_layernorm.eps,
    )
    _count(tape, "middle_norm_statistics", "forward")
    gate_BLF = _forward_projection(tape, "mlp.gate_proj", normalized_BLD, retain)
    up_BLF = _forward_projection(tape, "mlp.up_proj", normalized_BLD, retain)
    _count(tape, "mlp_swiglu", "forward")
    mlp_BLD = _forward_projection(
        tape, "mlp.down_proj", swiglu(gate_BLF, up_BLF), retain
    )
    output_BLD = tape.middle_BLD + mlp_BLD
    if not retain_hidden:
        tape.hidden_BLD = None
    if not retain_middle:
        tape.middle_BLD = None
    return output_BLD, first, second, tape


def backward_saved(
    tape: LayerTape,
    cotangents: list,
    *,
    hidden_BLD: Any = None,
) -> tuple[tuple, list]:
    """
    Purpose: Consume saved residuals without a decoder-layer VJP.
    Input: A forward tape and optional hidden and local-carry cotangents.
    Output: Named adapter gradients, hidden gradient, and incoming-carry gradients.
    Note: Temporary factor bindings preserve the policy that produced the tape.
    """
    assert len(cotangents) == 3 and any(x is not None for x in cotangents)
    if tape.hidden_BLD is None:
        assert hidden_BLD is not None, (
            "A discarded layer input requires its reconstructed checkpoint group."
        )
        tape = replace(tape, hidden_BLD=hidden_BLD)
    tape = replace(tape, scratch={})
    original = [(module, module.a, module.b) for _, module, _, _ in tape.factors]
    try:
        for _, module, a_RD, b_DR in tape.factors:
            module.a, module.b = a_RD, b_DR
        if tape.middle_BLD is None and cotangents[0] is not None:
            tape = replace(tape, middle_BLD=_reconstruct_middle(tape))
        names = tuple(
            name for name, _ in tree_flatten(tape.layer.trainable_parameters())
        )
        gradients = {
            name: mx.zeros_like(value)
            for name, value in tree_flatten(tape.layer.trainable_parameters())
        }
        dh_BLD = cotangents[0]
        if dh_BLD is not None:
            dmiddle_BLD = _mlp_backward(tape, dh_BLD, gradients)
        else:
            dmiddle_BLD = None
        if tape.layer.is_linear:
            dx_BLD, incoming = _gdn_backward(
                tape, dmiddle_BLD, cotangents[1:], gradients
            )
        else:
            dx_BLD, incoming = _attention_backward(
                tape, dmiddle_BLD, cotangents[1:], gradients
            )
        dnormalized_BLD = saved_norm.backward_saved(
            tape.hidden_BLD,
            tape.layer.input_layernorm.weight.astype(mx.bfloat16),
            tape.input_statistics,
            dx_BLD,
        )
        _count(tape, "input_norm_statistics", "retained")
        dh_BLD = (
            dnormalized_BLD if dmiddle_BLD is None else dmiddle_BLD + dnormalized_BLD
        )
        return names, [*[gradients[name] for name in names], dh_BLD, *incoming]
    finally:
        for module, a_RD, b_DR in original:
            module.a, module.b = a_RD, b_DR


def reconstruct_output(tape: LayerTape, hidden_BLD: Any) -> Any:
    """
    Purpose: Recreate a group boundary with the tape's original policy factors.
    Input: A tape and checkpoint-derived hidden rows. Output: The layer output.
    Note: The function restores every current factor binding before it returns.
    """
    original = [(module, module.a, module.b) for _, module, _, _ in tape.factors]
    try:
        for _, module, a_RD, b_DR in tape.factors:
            module.a, module.b = a_RD, b_DR
        return _reconstruct_output(tape, hidden_BLD)
    finally:
        for module, a_RD, b_DR in original:
            module.a, module.b = a_RD, b_DR


def _reconstruct_output(tape: LayerTape, hidden_BLD: Any) -> Any:
    """
    Purpose: Recreate a discarded group boundary with retained local work.
    Input: A tape and reconstructed layer input. Output: The layer's hidden result.
    Note: This function preserves the original carry producer allocations.
    """
    working = replace(tape, hidden_BLD=hidden_BLD, scratch={})
    middle_BLD = tape.middle_BLD
    if middle_BLD is None:
        middle_BLD = _reconstruct_middle(working)
    if "mlp.down_proj" in tape.retained:
        _count(tape, "mlp.down_proj", "retained")
        return middle_BLD + tape.retained["mlp.down_proj"]
    normalized_BLD = _middle_normalized(tape, middle_BLD)
    gate_BLF = _replay_projection(working, "mlp.gate_proj", normalized_BLD)
    up_BLF = _replay_projection(working, "mlp.up_proj", normalized_BLD)
    _count(tape, "mlp_swiglu", "reconstructed")
    product_BLF = swiglu(gate_BLF, up_BLF)
    return middle_BLD + _replay_projection(working, "mlp.down_proj", product_BLF)


def _reconstruct_middle(tape: LayerTape) -> Any:
    """
    Purpose: Recreate an unretained post-mixer residual with explicit local inputs.
    Input: A tape with its hidden input. Output: The post-mixer hidden result.
    Note: Saved attention output avoids score or decoder-layer reconstruction.
    """
    x_BLD = _input_normalized(tape)
    if tape.layer.is_linear:
        raw = [
            _replay_projection(tape, f"linear_attn.{name}", x_BLD)
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
        ]
        mixer_BLD, _, _ = gdn_inner(tape.layer.linear_attn, *raw, *tape.carries)
        _count(tape, "gdn_inner", "reconstructed")
        mixed_BLD = _replay_projection(tape, "linear_attn.out_proj", mixer_BLD)
    else:
        attention = tape.layer.self_attn
        raw_BLD = _replay_projection(tape, "self_attn.q_proj", x_BLD)
        _, gate_BLHK = mx.split(
            raw_BLD.reshape(*x_BLD.shape[:2], attention.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate_BLD = gate_BLHK.reshape(*x_BLD.shape[:2], -1)
        _count(tape, "attention_gate", "reconstructed")
        mixer_BLD = _attention_gate(tape.attention[0], gate_BLD)
        mixed_BLD = _replay_projection(tape, "self_attn.o_proj", mixer_BLD)
    return tape.hidden_BLD + mixed_BLD


def _mlp_backward(tape: LayerTape, dy_BLD: Any, gradients: dict) -> Any:
    """
    Purpose: Reverse the MLP with explicit projection derivatives.
    Input: Saved post-mixer hidden rows and output gradients. Output: Residual gradients.
    """
    layer = tape.layer
    x_BLD = _middle_normalized(tape, tape.middle_BLD)
    gate_BLF = _replay_projection(tape, "mlp.gate_proj", x_BLD)
    up_BLF = _replay_projection(tape, "mlp.up_proj", x_BLD)
    dproduct_BLF = _projection_input(tape, "mlp.down_proj", dy_BLD)
    _count(tape, "mlp_swiglu", "reconstructed")
    (product_BLF,), (dgate_BLF, dup_BLF) = mx.vjp(
        lambda gate, up: [swiglu(gate, up)],
        [gate_BLF, up_BLF],
        [dproduct_BLF],
    )
    _projection_parameters(tape, "mlp.down_proj", product_BLF, dy_BLD, gradients)
    dnormalized_BLD = _projection_input(
        tape, "mlp.gate_proj", dgate_BLF
    ) + _projection_input(tape, "mlp.up_proj", dup_BLF)
    _projection_parameters(tape, "mlp.gate_proj", x_BLD, dgate_BLF, gradients)
    _projection_parameters(tape, "mlp.up_proj", x_BLD, dup_BLF, gradients)
    dx_BLD = saved_norm.backward_saved(
        tape.middle_BLD,
        layer.post_attention_layernorm.weight.astype(mx.bfloat16),
        tape.middle_statistics,
        dnormalized_BLD,
    )
    _count(tape, "middle_norm_statistics", "retained")
    return dy_BLD + dx_BLD


def _gdn_backward(
    tape: LayerTape, dy_BLD: Any, carried: list, gradients: dict
) -> tuple:
    """
    Purpose: Reverse GDN without executing its retained frozen projections again.
    Input: Projection residuals and live output cotangents. Output: Hidden and carry adjoints.
    """
    linear = tape.layer.linear_attn
    x_BLD = _input_normalized(tape)
    names = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
    raw = []
    for name in names:
        if dy_BLD is None and name == "in_proj_z":
            raw.append(
                mx.zeros(
                    (*x_BLD.shape[:2], linear.num_v_heads * linear.head_v_dim),
                    x_BLD.dtype,
                )
            )
        else:
            raw.append(_replay_projection(tape, f"linear_attn.{name}", x_BLD))
    active = [i for i in (1, 2, 0) if ([dy_BLD, *carried][i] is not None)]
    cots = [
        _projection_input(tape, "linear_attn.out_proj", dy_BLD)
        if i == 0
        else carried[i - 1]
        for i in active
    ]

    def inner(*arguments):
        values = gdn_inner(linear, *arguments)
        return [values[i] for i in active]

    outputs, back = mx.vjp(inner, [*raw, *tape.carries], cots)
    _count(tape, "gdn_inner_vjp_primal", "reconstructed")
    if dy_BLD is not None:
        mixer_BLD = outputs[active.index(0)]
        _projection_parameters(
            tape, "linear_attn.out_proj", mixer_BLD, dy_BLD, gradients
        )
    dx_BLD = mx.zeros_like(x_BLD)
    for name, gradient in zip(names, back[:4], strict=True):
        if dy_BLD is None and name == "in_proj_z":
            continue
        path = f"linear_attn.{name}"
        dx_BLD = dx_BLD + _projection_input(tape, f"linear_attn.{name}", gradient)
        _projection_parameters(tape, path, x_BLD, gradient, gradients)
    return dx_BLD, back[4:]


def _attention_backward(
    tape: LayerTape, dy_BLD: Any, carried: list, gradients: dict
) -> tuple:
    """
    Purpose: Consume attention output and row statistics with explicit KV ownership.
    Input: Saved local KV, immutable history, and output adjoints. Output: Full-context gradients.
    """
    from .attention_metal import AttentionTape, attention_backward

    attention = tape.layer.self_attn
    x_BLD = _input_normalized(tape)
    raw = []
    for name in ("q_proj", "k_proj"):
        if name == "q_proj" and dy_BLD is None:
            raw.append(
                mx.zeros(
                    (*x_BLD.shape[:2], attention.num_attention_heads * 512), x_BLD.dtype
                )
            )
        else:
            raw.append(_replay_projection(tape, f"self_attn.{name}", x_BLD))
    raw.append(tape.local_kv[1].transpose(0, 2, 1, 3).reshape(*x_BLD.shape[:2], -1))
    _count(tape, "attention_prepare", "reconstructed")
    q_BHLK, k_BJLK, v_BJLK, gate_BLD = _prepare(attention, raw, tape.positions)
    local_gradients = [mx.zeros_like(k_BJLK), mx.zeros_like(v_BJLK)]
    incoming = [mx.zeros_like(value) for value in tape.carries]
    if dy_BLD is not None:
        output, statistics = tape.attention
        dmixer_BLD = _projection_input(tape, "self_attn.o_proj", dy_BLD)
        _count(tape, "attention_gate_vjp_primal", "reconstructed")
        (mixer_BLD,), (do_BHLK, dgate_BLD) = mx.vjp(
            lambda o, gate: [_attention_gate(o, gate)], [output, gate_BLD], [dmixer_BLD]
        )
        _projection_parameters(tape, "self_attn.o_proj", mixer_BLD, dy_BLD, gradients)
        keys = (
            mx.concatenate([*tape.carries[0::2], k_BJLK], axis=2)
            if tape.carries
            else k_BJLK
        )
        values = (
            mx.concatenate([*tape.carries[1::2], v_BJLK], axis=2)
            if tape.carries
            else v_BJLK
        )
        dq_BHLK, dk_BJK, dv_BJK = attention_backward(
            AttentionTape(
                q_BHLK,
                keys,
                values,
                output,
                statistics,
                keys.shape[2] - q_BHLK.shape[2],
            ),
            do_BHLK,
        )
        boundaries = []
        total = 0
        for value in tape.carries[0::2]:
            total += value.shape[2]
            boundaries.append(total)
        key_parts = mx.split(dk_BJK, boundaries, axis=2) if boundaries else [dk_BJK]
        value_parts = mx.split(dv_BJK, boundaries, axis=2) if boundaries else [dv_BJK]
        incoming = [
            item
            for pair in zip(key_parts[:-1], value_parts[:-1], strict=True)
            for item in pair
        ]
        local_gradients = [key_parts[-1], value_parts[-1]]
    else:
        dq_BHLK, dgate_BLD = mx.zeros_like(q_BHLK), mx.zeros_like(gate_BLD)
    local_gradients = [
        value if cot is None else value + cot
        for value, cot in zip(local_gradients, carried, strict=True)
    ]
    _count(tape, "attention_prepare_vjp_primal", "reconstructed")
    _, draw = mx.vjp(
        lambda *values: list(_prepare(attention, values, tape.positions)),
        raw,
        [dq_BHLK, *local_gradients, dgate_BLD],
    )
    dx_BLD = mx.zeros_like(x_BLD)
    for name, gradient in zip(("q_proj", "k_proj", "v_proj"), draw, strict=True):
        if name == "q_proj" and dy_BLD is None:
            continue
        dx_BLD = dx_BLD + _projection_input(tape, f"self_attn.{name}", gradient)
        _projection_parameters(tape, f"self_attn.{name}", x_BLD, gradient, gradients)
    return dx_BLD, incoming


def _forward_projection(
    tape: LayerTape, name: str, x_BLD: Any, retain: frozenset
) -> Any:
    module = dict((key, value) for key, value, _, _ in tape.factors)[name]
    low_BLR = x_BLD @ module.a.T
    value_BLD = module.base(x_BLD) + (module.scale * (low_BLR @ module.b.T)).astype(
        x_BLD.dtype
    )
    tape.low_rank[name] = low_BLR
    if name in retain:
        tape.retained[name] = value_BLD
    _count(tape, name, "forward")
    return value_BLD


def _replay_projection(tape: LayerTape, name: str, x_BLD: Any) -> Any:
    if name in tape.retained:
        _count(tape, name, "retained")
        return tape.retained[name]
    if name in tape.scratch:
        return tape.scratch[name]
    module = dict((key, value) for key, value, _, _ in tape.factors)[name]
    _count(tape, name, "reconstructed")
    result_BLD = module.base(x_BLD) + (
        module.scale * (tape.low_rank[name] @ module.b.T)
    ).astype(x_BLD.dtype)
    tape.scratch[name] = result_BLD
    return result_BLD


def _projection_input(tape: LayerTape, name: str, dy_BLD: Any) -> Any:
    module = dict((key, value) for key, value, _, _ in tape.factors)[name]
    _count(tape, name, "backward")
    base = module.base
    if hasattr(base, "bits"):
        dx_BLD = mx.quantized_matmul(
            dy_BLD,
            base.weight,
            base.scales,
            base.biases,
            transpose=False,
            group_size=base.group_size,
            bits=base.bits,
            mode=base.mode,
        )
    else:
        dx_BLD = dy_BLD @ base.weight
    return dx_BLD + ((module.scale * dy_BLD) @ module.b) @ module.a


def _projection_parameters(
    tape: LayerTape, name: str, x_BLD: Any, dy_BLD: Any, gradients: dict
) -> None:
    module = dict((key, value) for key, value, _, _ in tape.factors)[name]
    dy_LD = (module.scale * dy_BLD).reshape(-1, dy_BLD.shape[-1])
    x_LD = x_BLD.reshape(-1, x_BLD.shape[-1])
    low_LR = tape.low_rank[name].reshape(-1, module.a.shape[0])
    gradients[name + ".a"] = (dy_LD @ module.b).T @ x_LD
    gradients[name + ".b"] = dy_LD.T @ low_LR


def _projections(layer: Any) -> list:
    if layer.is_linear:
        parent, names = (
            "linear_attn",
            ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"),
        )
    else:
        parent, names = "self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")
    return [
        (f"{parent}.{name}", getattr(getattr(layer, parent), name)) for name in names
    ] + [
        (f"mlp.{name}", getattr(layer.mlp, name))
        for name in ("gate_proj", "up_proj", "down_proj")
    ]


def _input_normalized(tape: LayerTape) -> Any:
    _count(tape, "input_norm_output", "reconstructed")
    return saved_norm.apply_saved(
        tape.hidden_BLD,
        tape.layer.input_layernorm.weight.astype(mx.bfloat16),
        tape.input_statistics,
    )


def _middle_normalized(tape: LayerTape, middle_BLD: Any) -> Any:
    _count(tape, "middle_norm_output", "reconstructed")
    return saved_norm.apply_saved(
        middle_BLD,
        tape.layer.post_attention_layernorm.weight.astype(mx.bfloat16),
        tape.middle_statistics,
    )


def _prepare(attention: Any, raw: list, positions: Any) -> tuple:
    return prepare_attention(attention, raw, positions)


def _attention_gate(output_BHLK: Any, gate_BLD: Any) -> Any:
    return output_BHLK.transpose(0, 2, 1, 3).reshape(gate_BLD.shape) * mx.sigmoid(
        gate_BLD
    )


def _count(tape: LayerTape, name: str, stage: str) -> None:
    key = f"{stage}.{name}"
    tape.counters[key] = tape.counters.get(key, 0) + 1
