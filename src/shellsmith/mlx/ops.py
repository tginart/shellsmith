"""Dense Qwen layer math using ordinary differentiable MLX operations.

Segment/carry and tiled attention design follow High Command 10e830.
The gated-delta equations are adapted from mlx-vlm 0.7.4 (MIT License,
Copyright 2024-2026 Prince Canuma and contributors). No handwritten VJP,
compiled graph, extension, or custom Metal kernel is installed here.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

ATTENTION_TILE_BYTES = 128 * 1024 * 1024
INVERSE_BLOCK = 16


def _substitute(strict, size):
    eye = mx.eye(size, dtype=mx.float32)
    inverse = mx.broadcast_to(eye, strict.shape)
    for index in range(1, size):
        row = eye[index] - strict[..., index:index + 1, :] @ inverse
        inverse = inverse + eye[:, index:index + 1] * (row - inverse[..., index:index + 1, :])
    return inverse


def _invert_unit_lower(matrix, size):
    strict = matrix * mx.tril(mx.ones((size, size), mx.float32), -1)
    block = min(INVERSE_BLOCK, size)
    if size % block or size & (size - 1):
        raise ValueError("GDN chunk size must be a power of two and compatible with the inverse block.")
    lead = strict.shape[:-2]
    diagonal = mx.stack([strict[..., i * block:(i + 1) * block, i * block:(i + 1) * block]
                         for i in range(size // block)], axis=-3)
    solved = _substitute(diagonal, block)
    inverses = [solved[..., i, :, :] for i in range(size // block)]
    width = block
    while len(inverses) > 1:
        merged = []
        for index in range(0, len(inverses), 2):
            start = index * width
            lower = strict[..., start + width:start + 2 * width, start:start + width]
            first, second = inverses[index], inverses[index + 1]
            corner = -(second @ (lower @ first))
            top = mx.concatenate([first, mx.zeros((*lead, width, width), mx.float32)], axis=-1)
            merged.append(mx.concatenate([top, mx.concatenate([corner, second], axis=-1)], axis=-2))
        inverses, width = merged, width * 2
    return inverses[0]


def gated_delta_chunked(q, k, v, g, beta, state, *, chunk=64):
    """Native-autodiff chunked recurrence with a differentiable FP32 carry."""
    batch, length, key_heads, key_width = q.shape
    value_heads, value_width = v.shape[-2:]
    input_dtype = v.dtype
    if value_heads % key_heads:
        raise ValueError("GDN value heads must be a multiple of key heads.")
    if value_heads != key_heads:
        q, k = (mx.repeat(value, value_heads // key_heads, -2) for value in (q, k))
    if length == 1:
        current = state.astype(mx.float32) * g[:, 0, :, None, None]
        key = k[:, 0].astype(mx.float32)
        delta = (v[:, 0].astype(mx.float32) - mx.sum(current * key[..., None, :], axis=-1)) * beta[:, 0, :, None]
        current = current + delta[..., None] * key[..., None, :]
        output = mx.sum(current * q[:, 0].astype(mx.float32)[..., None, :], axis=-1)
        return output[:, None].astype(input_dtype), current
    pad = (chunk - length % chunk) % chunk
    if pad:
        q = mx.concatenate([q, mx.zeros((batch, pad, value_heads, key_width), q.dtype)], axis=1)
        k = mx.concatenate([k, mx.zeros((batch, pad, value_heads, key_width), k.dtype)], axis=1)
        v = mx.concatenate([v, mx.zeros((batch, pad, value_heads, value_width), v.dtype)], axis=1)
        g = mx.concatenate([g, mx.ones((batch, pad, value_heads), g.dtype)], axis=1)
        beta = mx.concatenate([beta, mx.zeros((batch, pad, value_heads), beta.dtype)], axis=1)
    chunks = (length + pad) // chunk

    def blocks(value, width):
        return value.reshape(batch, chunks, chunk, value_heads, width).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    q, k, v = blocks(q, key_width), blocks(k, key_width), blocks(v, value_width)
    g = g.reshape(batch, chunks, chunk, value_heads).transpose(0, 3, 1, 2).astype(mx.float32)
    beta = beta.reshape(batch, chunks, chunk, value_heads).transpose(0, 3, 1, 2).astype(mx.float32)
    log_decay = mx.cumsum(mx.log(mx.clip(g, 1e-6, 1.0)), axis=-1)
    decay = mx.exp(log_decay)
    lower = mx.tril(mx.ones((chunk, chunk), mx.float32), 0)
    strict = mx.tril(mx.ones((chunk, chunk), mx.float32), -1)
    ratio = mx.exp(mx.where(lower > 0, log_decay[..., :, None] - log_decay[..., None, :], -1e30))
    kk = k @ mx.swapaxes(k, -1, -2)
    solve = _invert_unit_lower(mx.eye(chunk, dtype=mx.float32) + beta[..., :, None] * ratio * kk * strict, chunk)
    u0 = solve @ (beta[..., :, None] * v)
    kt = solve @ ((beta * decay)[..., :, None] * k)
    projection = ratio * (q @ mx.swapaxes(k, -1, -2)) * lower
    q_effective = decay[..., :, None] * q - projection @ kt
    local_output = projection @ u0
    decay_last = decay[..., -1]
    ratio_last = mx.exp(log_decay[..., -1, None] - log_decay)
    current, outputs = state.astype(mx.float32), []
    for index in range(chunks):
        transposed = mx.swapaxes(current, -1, -2)
        outputs.append(local_output[:, :, index] + q_effective[:, :, index] @ transposed)
        update = u0[:, :, index] - kt[:, :, index] @ transposed
        current = decay_last[:, :, index][..., None, None] * current + (
            mx.swapaxes(ratio_last[:, :, index][..., None] * update, -1, -2) @ k[:, :, index])
    output = mx.stack(outputs, axis=2).transpose(0, 2, 3, 1, 4).reshape(
        batch, length + pad, value_heads, value_width)[:, :length]
    return output.astype(input_dtype), current


def gdn_mixer(mixer, x, states):
    """Upstream projection/conv/norm math with pure-MLX decay and recurrence."""
    batch, length, _ = x.shape
    projected = mixer.in_proj_qkv(x)
    z = mixer.in_proj_z(x).reshape(batch, length, mixer.num_v_heads, mixer.head_v_dim)
    b, a = mixer.in_proj_b(x), mixer.in_proj_a(x)
    convolution = mx.concatenate([states[0], projected], axis=1)
    taps = mixer.conv_kernel_size - 1
    # Force a small owned value: a trailing view can retain the full QKV allocation.
    carry_conv = convolution[:, -taps:] + mx.zeros_like(convolution[:, -taps:]) if taps else convolution[:, :0]
    conv = nn.silu(mixer.conv1d(convolution))
    q, k, v = [part.reshape(batch, length, heads, width) for part, heads, width in zip(
        mx.split(conv, [mixer.key_dim, 2 * mixer.key_dim], -1),
        (mixer.num_k_heads, mixer.num_k_heads, mixer.num_v_heads),
        (mixer.head_k_dim, mixer.head_k_dim, mixer.head_v_dim))]
    inv_scale = mixer.head_k_dim ** -0.5
    q = inv_scale ** 2 * mx.fast.rms_norm(q, None, 1e-6)
    k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
    g = mx.exp(-mx.exp(mixer.A_log.astype(mx.float32)) * nn.softplus(a + mixer.dt_bias))
    beta = mx.sigmoid(b)
    output, carry_recurrent = gated_delta_chunked(q, k, v, g, beta, states[1])
    output = mixer.norm(output, z).reshape(batch, length, -1)
    return mixer.out_proj(output), (carry_conv, carry_recurrent)


def tiled_causal_attention(q, keys, values, scale, offset):
    """Query tiles attend their complete causal key prefix without truncation."""
    batch, heads, length, _ = q.shape
    span = offset + length
    tile = max(1, min(length, ATTENTION_TILE_BYTES // max(1, batch * heads * span * 4)))

    def attend(query, key, value):
        return mx.fast.scaled_dot_product_attention(query, key, value, scale=scale, mask="causal")

    attend = mx.checkpoint(attend)
    outputs = []
    for start in range(0, length, tile):
        stop = min(length, start + tile)
        outputs.append(attend(q[:, :, start:stop], keys[:, :, :offset + stop], values[:, :, :offset + stop]))
    return mx.concatenate(outputs, axis=2) if len(outputs) > 1 else outputs[0]


def initial_states(layers, dtype):
    states = []
    for layer in layers:
        if layer.is_linear:
            mixer = layer.linear_attn
            states.append([(None, mx.zeros((1, mixer.conv_kernel_size - 1, mixer.conv_dim), dtype)),
                           (None, mx.zeros((1, mixer.num_v_heads, mixer.head_v_dim, mixer.head_k_dim), mx.float32))])
        else:
            states.append([])
    return states


def segment_layer(layer, hidden, states, positions):
    """Pure layer unit returning hidden, then its two newly produced carries."""
    x = layer.input_layernorm(hidden)
    if layer.is_linear:
        mixed, carried = gdn_mixer(layer.linear_attn, x, states)
    else:
        attention = layer.self_attn
        q, k, v, gate, _ = attention._prepare_projected_qkv(
            attention.q_proj(x), attention.k_proj(x), attention.v_proj(x),
            cache=None, position_ids=positions, position_embeddings=None, mask=None)
        keys = mx.concatenate([*states[0::2], k], axis=2) if states else k
        values = mx.concatenate([*states[1::2], v], axis=2) if states else v
        output = tiled_causal_attention(q, keys, values, attention.scale, keys.shape[2] - q.shape[2])
        output = output.transpose(0, 2, 1, 3).reshape(hidden.shape[0], hidden.shape[1], -1)
        mixed = attention.o_proj(output * mx.sigmoid(gate))
        carried = (k, v)
    hidden = hidden + mixed
    hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
    return hidden, *carried
