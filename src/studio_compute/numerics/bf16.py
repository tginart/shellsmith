"""
Purpose: Execute the GDN layer with BF16 tensors and explicit carried state.
Input: Frozen quantized projections, BF16 adapters, hidden rows, and two carries.
Output: BF16 hidden rows, convolution taps, and recurrent state.
Note: Fast norm and matrix kernels may use internal wider accumulators.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

# B: batch, L: tokens, H: value heads, K: key width, V: value width, D: hidden.


def gdn_forward(
    linear: Any, x_BLD: Any, states: list, *, decode_one: bool = False,
    gate_activation: str = "silu", backend: str = "metal",
) -> tuple:
    """
    Purpose: Apply the admitted GDN formula without explicit FP32 tensor operations.
    Input: One layer and BF16 segment inputs. Output: Mixer output and both carries.
    Note: backend selects the gated_delta_chunked backend of the chunk scan.
        "metal_fused" is the training backend: the explicit-inverse chunk
        scan and the fused convolution and gated-norm kernels.
    """
    assert x_BLD.dtype == mx.bfloat16 and len(states) == 2
    raw = [
        getattr(linear, name)(x_BLD)
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")
    ]
    mixed_BLD, taps_BCD, next_BHVK = gdn_inner(
        linear, *raw, *states, decode_one=decode_one, gate_activation=gate_activation,
        backend=backend,
    )
    return linear.out_proj(mixed_BLD), taps_BCD, next_BHVK


def gdn_inner(
    linear: Any,
    qkv_BLD: Any,
    z_BLD: Any,
    a_BLH: Any,
    beta_BLH: Any,
    conv_BCD: Any,
    state_BHVK: Any,
    *,
    decode_one: bool = False,
    gate_activation: str = "silu",
    backend: str = "metal",
) -> tuple:
    """
    Purpose: Execute GDN nonlinear operations with explicit projection results.
    Input: Four BF16 projections and both carries. Output: Pre-projection mixer and carries.
    Note: The caller owns projection execution and retained residuals.
    """
    from .ops import gated_delta_chunked

    fused = backend == "metal_fused" and not decode_one
    assert all(
        value.dtype == mx.bfloat16
        for value in (qkv_BLD, z_BLD, a_BLH, beta_BLH, conv_BCD, state_BHVK)
    )
    batch, length, _ = qkv_BLD.shape
    z_BLHV = z_BLD.reshape(batch, length, linear.num_v_heads, linear.head_v_dim)
    if fused:
        from .gdn_fused import causal_conv_silu

        convolved_BLD = causal_conv_silu(conv_BCD, qkv_BLD, linear.conv1d.weight)
        convolution_BLD = mx.concatenate([conv_BCD, qkv_BLD[:, -conv_BCD.shape[1]:]],
                                         axis=1)
    else:
        convolution_BLD = mx.concatenate([conv_BCD, qkv_BLD], axis=1)
        convolved_BLD = nn.silu(linear.conv1d(convolution_BLD))
    q_BLHK, k_BLHK, v_BLHV = [
        value.reshape(batch, length, heads, width)
        for value, heads, width in zip(
            mx.split(convolved_BLD, [linear.key_dim, 2 * linear.key_dim], axis=-1),
            (linear.num_k_heads, linear.num_k_heads, linear.num_v_heads),
            (linear.head_k_dim, linear.head_k_dim, linear.head_v_dim),
            strict=True,
        )
    ]
    q_BLHK, k_BLHK = linear._normalize_qk(q_BLHK, k_BLHK)
    from .gdn_fused import HEAD_WIDTH

    if fused and linear.head_k_dim == linear.head_v_dim == HEAD_WIDTH:
        from .gdn_fused import gated_delta_chunks

        # The fused chunk kernel makes g and beta from the logits in FP32.
        y_BLHV, next_BHVK = gated_delta_chunks(
            q_BLHK, k_BLHK, v_BLHV, a_BLH, beta_BLH, linear.A_log, linear.dt_bias,
            state_BHVK,
        )
    else:
        beta_BLH = mx.sigmoid(beta_BLH)
        g_BLH = mx.exp(
            -mx.exp(linear.A_log.astype(mx.bfloat16))
            * nn.softplus(a_BLH + linear.dt_bias.astype(mx.bfloat16))
        )
        if decode_one:
            from .gdn_token import gated_delta_token

            y_BLHV, next_BHVK = gated_delta_token(
                q_BLHK, k_BLHK, v_BLHV, g_BLH, beta_BLH, state_BHVK
            )
        else:
            y_BLHV, next_BHVK = gated_delta_chunked(
                q_BLHK,
                k_BLHK,
                v_BLHV,
                g_BLH,
                beta_BLH,
                state_BHVK,
                compute_dtype=mx.bfloat16,
                backend="metal_inverse" if fused else backend,
            )
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError("Unsupported GDN output gate activation.")
    if fused:
        from .gdn_fused import gated_rms_norm

        gated_BLHV = gated_rms_norm(
            y_BLHV, z_BLHV, linear.norm.weight.astype(mx.bfloat16), linear.norm.eps,
            sigmoid=gate_activation == "sigmoid",
        )
    else:
        normalized_BLHV = mx.fast.rms_norm(
            y_BLHV, linear.norm.weight.astype(mx.bfloat16), linear.norm.eps
        )
        gate_BLHV = (
            mx.sigmoid(z_BLHV) if gate_activation == "sigmoid" else nn.silu(z_BLHV)
        )
        gated_BLHV = normalized_BLHV * gate_BLHV
    mixed_BLD = gated_BLHV.reshape(batch, length, -1)
    taps_BCD = convolution_BLD[:, -(linear.conv_kernel_size - 1) :]
    return mixed_BLD, taps_BCD + mx.zeros_like(taps_BCD), next_BHVK


def decode_hidden(lm: Any, ids_BL: Any, positions: Any, caches: list) -> Any:
    """
    Purpose: Use the BF16 layer implementation for real prefill and autoregression.
    Input: Token IDs, absolute positions, and owned native caches.
    Output: Normalized hidden rows. The function advances the supplied caches.
    """
    from .ops import segment_layer

    hidden_BLD = lm.model.embed_tokens(ids_BL).astype(mx.bfloat16)
    for layer, cache in zip(lm.layers, caches, strict=True):
        if layer.is_linear:
            linear = layer.linear_attn
            incoming = [
                cache[0]
                if cache[0] is not None
                else mx.zeros(
                    (ids_BL.shape[0], linear.conv_kernel_size - 1, linear.conv_dim),
                    mx.bfloat16,
                ),
                cache[1]
                if cache[1] is not None
                else mx.zeros(
                    (
                        ids_BL.shape[0],
                        linear.num_v_heads,
                        linear.head_v_dim,
                        linear.head_k_dim,
                    ),
                    mx.bfloat16,
                ),
            ]
        else:
            hidden_BLD = _decode_attention(layer, hidden_BLD, positions, cache)
            continue
        if ids_BL.shape[1] == 1:
            hidden_BLD, first, second = _decode_gdn_token(layer, hidden_BLD, incoming)
        else:
            hidden_BLD, first, second = segment_layer(
                layer, hidden_BLD, incoming, positions, execution_profile="bf16-fast"
            )
        if layer.is_linear:
            cache.cache = [first, second]
        else:
            cache.update_and_fetch(first, second)
    return mx.fast.rms_norm(
        hidden_BLD, lm.model.norm.weight.astype(mx.bfloat16), lm.model.norm.eps
    )


def _decode_gdn_token(layer: Any, hidden_BLD: Any, carries: list) -> tuple:
    """
    Purpose: Advance one GDN decoder row without the training chunk pipeline.
    Input: BF16 hidden input and two carries. Output: Decoder output and both new carries.
    """
    x_BLD = mx.fast.rms_norm(
        hidden_BLD,
        layer.input_layernorm.weight.astype(mx.bfloat16),
        layer.input_layernorm.eps,
    )
    mixed_BLD, conv_BCD, state_BHVK = gdn_forward(
        layer.linear_attn, x_BLD, carries, decode_one=True
    )
    middle_BLD = hidden_BLD + mixed_BLD
    normalized_BLD = mx.fast.rms_norm(
        middle_BLD,
        layer.post_attention_layernorm.weight.astype(mx.bfloat16),
        layer.post_attention_layernorm.eps,
    )
    return middle_BLD + layer.mlp(normalized_BLD), conv_BCD, state_BHVK


def _decode_attention(layer: Any, hidden_BLD: Any, positions: Any, cache: Any) -> Any:
    """
    Purpose: Preserve native attention-cache masks, offsets, and quantized storage.
    Input: BF16 hidden rows, positions, and a request-owned cache.
    Output: BF16 decoder output. The native cache advances once.
    Note: The pinned forward kernels remain faster for inference without a tape.
    """
    x_BLD = mx.fast.rms_norm(
        hidden_BLD,
        layer.input_layernorm.weight.astype(mx.bfloat16),
        layer.input_layernorm.eps,
    )
    mask = cache.make_mask(hidden_BLD.shape[1], return_array=False, window_size=None)
    mixed_BLD = layer.self_attn(x_BLD, cache=cache, position_ids=positions, mask=mask)
    middle_BLD = hidden_BLD + mixed_BLD
    normalized_BLD = mx.fast.rms_norm(
        middle_BLD,
        layer.post_attention_layernorm.weight.astype(mx.bfloat16),
        layer.post_attention_layernorm.eps,
    )
    return middle_BLD + layer.mlp(normalized_BLD)


def prepare_attention(attention: Any, raw: list, positions: Any) -> tuple:
    """
    Purpose: Prepare BF16 GQA and MRoPE without the pinned FP32 rotary VJP graph.
    Input: Q/K/V projections and three-axis positions. Output: Q, K, V, and the gate.
    """
    from .rotary import apply_rotary

    q_raw_BLD, k_raw_BLD, v_raw_BLD = raw
    batch, length, _ = q_raw_BLD.shape
    q_BLHD, gate_BLHD = mx.split(
        q_raw_BLD.reshape(batch, length, attention.num_attention_heads, -1), 2, axis=-1
    )
    k_BLJD = k_raw_BLD.reshape(batch, length, attention.num_key_value_heads, -1)
    q_BHLD = mx.fast.rms_norm(
        q_BLHD, attention.q_norm.weight.astype(mx.bfloat16), attention.q_norm.eps
    ).transpose(0, 2, 1, 3)
    k_BJLD = mx.fast.rms_norm(
        k_BLJD, attention.k_norm.weight.astype(mx.bfloat16), attention.k_norm.eps
    ).transpose(0, 2, 1, 3)
    q_BHLD, k_BJLD = apply_rotary(q_BHLD, k_BJLD, positions, attention.rotary_emb)
    v_BJLD = v_raw_BLD.reshape(
        batch, length, attention.num_key_value_heads, -1
    ).transpose(0, 2, 1, 3)
    return q_BHLD, k_BJLD, v_BJLD, gate_BLHD.reshape(batch, length, -1)
