"""Replay layer operations for the resident MLX-VLM Qwen decoder.

DGX counterpart: DenseGroupOps (runtime/dense/autograd/native_model.py), the
layer-operation interface behind group_replay.

Architecture: one decoder layer over one token segment, continuing explicit
carried state (GDN conv + recurrent state; attention keys/values of earlier
segments) and returning the next carries. Model-specific math lives here;
mlx_group_replay owns segmenting, checkpointing, sharing and the objective.

Kernels (chosen by docs/MLX_REPLAY_BENCHMARK.md):
- Attention: query-tiled causal SDPA, each tile checkpointed, one FP32 score
  tile under ATTENTION_TILE_BYTES. Under autodiff MLX SDPA uses the unfused
  explicit math, so tiling sets memory and speed, not numerics. Opt-in:
  mlx_flash_attention (cooperative Metal FlashAttention extension).
- GDN: mlx-vlm's projections, convolution, decay and normalization, with the
  chunked gated-delta recurrence replaced by a scan with a hand-written VJP
  (2.4x faster per layer than autodiff, same error). The replacement is bound
  into private copies of mlx-vlm's functions; the installed module, model
  methods and training flags are never modified.
- Positions: the attention projection uses mlx-vlm's three-axis M-RoPE path,
  so image and video prompts keep their positions.

Borrowed code: ``gated_delta_chunked`` is adapted from mlx-vlm
(mlx_vlm/models/qwen3_5/gated_delta.py, MIT License, Copyright 2024-2026 Prince
Canuma and contributors); the async_eval graph break was removed so the scan
is differentiable. Main caller: mlx_group_replay.Replay.
"""
from functools import lru_cache
from types import FunctionType

import mlx.core as mx

# Upper bound on one attention tile's FP32 score matrix during replay.
ATTENTION_TILE_BYTES = 2 << 30


INVERSE_BLOCK = 16


def _substitute(strict, size):
    """Inverse of I + strict (strict lower triangular) by row substitution.

    Row i is e_i - strict[i] @ inverse; each step is one small matmul and one
    masked add on a (block x block) tensor.
    """
    eye = mx.eye(size, dtype=mx.float32)
    inverse = mx.broadcast_to(eye, strict.shape)
    for i in range(1, size):
        row = eye[i] - strict[..., i:i + 1, :] @ inverse
        inverse = inverse + eye[:, i:i + 1] * (row - inverse[..., i:i + 1, :])
    return inverse


def _invert_unit_lower(matrix, size):
    """Inverse of a unit lower-triangular matrix (fp32), blocked.

    The diagonal INVERSE_BLOCK blocks are inverted together by substitution
    (INVERSE_BLOCK sequential steps on small tensors), then neighbouring blocks
    merge level by level with the block formula
        [[A, 0], [B, D]]^-1 = [[A^-1, 0], [-D^-1 B A^-1, D^-1]],
    two batched matmuls per merge: the same stable scheme as FLA's solve_tril.
    Why this exists: substituting over the whole 64-wide chunk rewrote the
    full (heads x chunks x 64 x 64) tensor 64 times, forward and backward.
    """
    strict = matrix * mx.tril(mx.ones((size, size), mx.float32), -1)
    block = min(INVERSE_BLOCK, size)
    if size % block:
        raise ValueError('GDN chunk must be a multiple of the inverse block')
    lead = strict.shape[:-2]
    diagonal = mx.stack([strict[..., b * block:(b + 1) * block, b * block:(b + 1) * block]
                         for b in range(size // block)], axis=-3)
    solved = _substitute(diagonal, block)
    inverses = [solved[..., b, :, :] for b in range(size // block)]
    width = block
    while len(inverses) > 1:
        merged = []
        for b in range(0, len(inverses), 2):
            start = b * width
            lower = strict[..., start + width:start + 2 * width, start:start + width]
            first, second = inverses[b], inverses[b + 1]
            corner = -(second @ (lower @ first))
            top = mx.concatenate([first, mx.zeros((*lead, width, width), mx.float32)], axis=-1)
            merged.append(mx.concatenate([top, mx.concatenate([corner, second], axis=-1)], axis=-2))
        inverses, width = merged, width * 2
    return inverses[0]


def _scan_forward(mu0, q_eff, u0, kt, k, decay_last, ratio_last, state):
    """Inter-chunk recurrence; returns (outputs per chunk, state before each chunk, final state).

    Per chunk c (matrices per batch and head; S is [Dv, Dk]):
        O_c = mu0_c + Q_c S^T
        U_c = u0_c - Kt_c S^T
        S  <- d_c S + (r_c * U_c)^T K_c
    """
    chunks = mu0.shape[2]
    s, outputs, states = state, [], []
    for c in range(chunks):
        states.append(s)
        st = mx.swapaxes(s, -1, -2)
        outputs.append(mu0[:, :, c] + q_eff[:, :, c] @ st)
        uc = u0[:, :, c] - kt[:, :, c] @ st
        s = decay_last[:, :, c][..., None, None] * s + mx.swapaxes(ratio_last[:, :, c][..., None] * uc, -1, -2) @ k[:, :, c]
    return mx.stack(outputs, axis=2), states, s


@mx.custom_function
def _chunk_scan(mu0, q_eff, u0, kt, k, decay_last, ratio_last, state):
    """Differentiable inter-chunk scan of the gated delta rule (see _scan_forward).

    Why this exists: autodiff through the Python loop accumulated every
    per-chunk slice gradient into full-size input tensors, so backward cost
    grew with the square of the chunk count (~0.5 s per 8k-token GDN layer on
    an M5 Ultra). The explicit VJP below walks the chunks once in reverse and
    stacks each input gradient once.
    """
    outputs, _, final = _scan_forward(mu0, q_eff, u0, kt, k, decay_last, ratio_last, state)
    return outputs, final


@_chunk_scan.vjp
def _chunk_scan_vjp(primals, cotangents, _outputs):
    mu0, q_eff, u0, kt, k, decay_last, ratio_last, state = primals
    d_outputs, d_state = cotangents
    _, states, _ = _scan_forward(mu0, q_eff, u0, kt, k, decay_last, ratio_last, state)
    chunks = mu0.shape[2]
    grads = {name: [None] * chunks for name in ('q', 'u0', 'kt', 'k', 'decay', 'ratio')}
    ds = d_state
    for c in reversed(range(chunks)):
        s = states[c]
        st = mx.swapaxes(s, -1, -2)
        uc = u0[:, :, c] - kt[:, :, c] @ st
        ratio = ratio_last[:, :, c][..., None]
        weighted = ratio * uc
        # S_next = d S + weighted^T K
        d_weighted = k[:, :, c] @ mx.swapaxes(ds, -1, -2)
        grads['k'][c] = weighted @ ds
        grads['decay'][c] = (ds * s).sum(axis=(-2, -1))
        d_u = ratio * d_weighted
        grads['ratio'][c] = (d_weighted * uc).sum(axis=-1)
        grads['u0'][c] = d_u
        grads['kt'][c] = -(d_u @ s)
        d_o = d_outputs[:, :, c]
        grads['q'][c] = d_o @ s
        d_st = mx.swapaxes(q_eff[:, :, c], -1, -2) @ d_o - mx.swapaxes(kt[:, :, c], -1, -2) @ d_u
        ds = decay_last[:, :, c][..., None, None] * ds + mx.swapaxes(d_st, -1, -2)
    stack = lambda name: mx.stack(grads[name], axis=2)
    return (d_outputs, stack('q'), stack('u0'), stack('kt'), stack('k'), stack('decay'), stack('ratio'), ds)


def gated_delta_chunked(q, k, v, g, beta, state, chunk=64):
    """Differentiable chunked gated-delta scan (scalar per-head gating).

    Same recurrence as mlx-lm's sequential kernel (state [B, Hv, Dv, Dk] fp32):
    parallel intra-chunk matmuls, a unit-lower-triangular solve, and a T/chunk
    inter-chunk state scan. Returns (y [B, T, Hv, Dv] in v's dtype, final state).
    Main caller: chunk_recurrence (replay GDN layers). Adapted from mlx-vlm.
    """
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = v.dtype
    if Hv // Hk > 1:
        q = mx.repeat(q, Hv // Hk, -2)
        k = mx.repeat(k, Hv // Hk, -2)
    pad = (chunk - T % chunk) % chunk
    if pad:
        q = mx.concatenate([q, mx.zeros((B, pad, Hv, Dk), q.dtype)], axis=1)
        k = mx.concatenate([k, mx.zeros((B, pad, Hv, Dk), k.dtype)], axis=1)
        v = mx.concatenate([v, mx.zeros((B, pad, Hv, Dv), v.dtype)], axis=1)
        g = mx.concatenate([g, mx.ones((B, pad, Hv), g.dtype)], axis=1)
        beta = mx.concatenate([beta, mx.zeros((B, pad, Hv), beta.dtype)], axis=1)
    chunks = (T + pad) // chunk

    def blocks(x, width):
        return x.reshape(B, chunks, chunk, Hv, width).transpose(0, 3, 1, 2, 4).astype(mx.float32)

    q, k, v = blocks(q, Dk), blocks(k, Dk), blocks(v, Dv)
    g = g.reshape(B, chunks, chunk, Hv).transpose(0, 3, 1, 2).astype(mx.float32)
    beta = beta.reshape(B, chunks, chunk, Hv).transpose(0, 3, 1, 2).astype(mx.float32)
    # log(0) would poison the decay ratios; g underflows to 0 in fp32.
    log_decay = mx.cumsum(mx.log(mx.clip(g, 1e-6, 1.0)), axis=-1)
    decay = mx.exp(log_decay)
    lower = mx.tril(mx.ones((chunk, chunk), mx.float32), 0)
    strict = mx.tril(mx.ones((chunk, chunk), mx.float32), -1)
    # Mask the exponent before exp: the upper triangle would overflow.
    ratio = mx.exp(mx.where(lower > 0, log_decay[..., :, None] - log_decay[..., None, :], -1e30))
    kk = k @ mx.swapaxes(k, -1, -2)
    solve = _invert_unit_lower(mx.eye(chunk, dtype=mx.float32) + beta[..., :, None] * ratio * kk * strict, chunk)
    u0 = solve @ (beta[..., :, None] * v)
    kt = solve @ ((beta * decay)[..., :, None] * k)
    m = ratio * (q @ mx.swapaxes(k, -1, -2)) * lower
    q_eff = decay[..., :, None] * q - m @ kt
    mu0 = m @ u0
    decay_last = decay[..., -1]
    ratio_last = mx.exp(log_decay[..., -1, None] - log_decay)
    y, s = _chunk_scan(mu0, q_eff, u0, kt, k, decay_last, ratio_last, state.astype(mx.float32))
    y = y.transpose(0, 2, 3, 1, 4).reshape(B, T + pad, Hv, Dv)[:, :T]
    return y.astype(in_dtype), s


_CHECKPOINTED_ATTENTION = {}


def _tile_attention(scale):
    """One checkpointed causal attention tile (queries end at the last key)."""
    if scale not in _CHECKPOINTED_ATTENTION:
        def tile(q, k, v):
            return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask='causal')
        _CHECKPOINTED_ATTENTION[scale] = mx.checkpoint(tile)
    return _CHECKPOINTED_ATTENTION[scale]


def tiled_causal_attention(q, keys, values, scale, offset):
    """Causal attention for queries at positions offset..offset+L-1.

    ``keys``/``values`` cover positions 0..offset+L-1. Each query tile sees its
    causal key prefix; tiles are sized so one fp32 score matrix stays under
    ATTENTION_TILE_BYTES and are checkpointed individually.
    """
    batch, heads, length, _ = q.shape
    span = offset + length
    tile = max(16, min(length, ATTENTION_TILE_BYTES // max(1, batch * heads * span * 4)))
    attend = _tile_attention(scale)
    outputs = []
    for start in range(0, length, tile):
        end = min(length, start + tile)
        outputs.append(attend(q[:, :, start:end], keys[:, :, :offset + end], values[:, :, :offset + end]))
    return mx.concatenate(outputs, axis=2) if len(outputs) > 1 else outputs[0]


def chunk_recurrence(q, k, v, g, beta, state, mask=None, C=64):
    """mlx-vlm ``gated_delta_chunked`` signature over the hand-VJP scan."""
    if mask is not None:
        raise ValueError('Replay GDN never receives a padding mask')
    return gated_delta_chunked(q, k, v, g, beta, state, chunk=C)


def _clone(native, name, replacements, required_names):
    """Private copy of a dependency function with selected globals rebound."""
    if not isinstance(native, FunctionType) or not set(required_names).issubset(native.__code__.co_names):
        raise RuntimeError('Pinned mlx-vlm GDN implementation no longer exposes the audited call hierarchy')
    namespace = dict(native.__globals__)
    namespace.update(replacements)
    replacement = FunctionType(native.__code__, namespace, name, native.__defaults__, native.__closure__)
    replacement.__kwdefaults__ = native.__kwdefaults__
    return replacement


@lru_cache(maxsize=1)
def _gdn_update():
    from mlx_vlm.models.qwen3_5.gated_delta import gated_delta_update as native
    replacement = _clone(native, 'hc_gated_delta_update', {'gated_delta_chunked': chunk_recurrence},
                         ('gated_delta_chunked', 'update_recurrent'))
    # ArraysCache invokes a closure which recursively calls this global name;
    # rebind it in the private namespace too.
    replacement.__globals__['gated_delta_update'] = replacement
    return replacement


@lru_cache(maxsize=1)
def _gdn_forward():
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
    return _clone(Qwen3_5GatedDeltaNet.__call__, 'hc_gdn_forward', {'gated_delta_update': _gdn_update()},
                  ('gated_delta_update', 'update_window', 'training'))


def gdn_mixer(linear, x, cache):
    """mlx-vlm's GDN layer call with the hand-VJP recurrence."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5GatedDeltaNet
    if type(linear).__call__ is not Qwen3_5GatedDeltaNet.__call__:
        raise RuntimeError('Replay expects the pinned mlx-vlm Qwen3.5 GDN layer')
    return _gdn_forward()(linear, x, cache=cache)


def initial_states(layers, dtype):
    """Empty carries: GDN (conv taps, FP32 recurrent state); attention none."""
    states = []
    for layer in layers:
        if layer.is_linear:
            mixer = layer.linear_attn
            states.append([(None, mx.zeros((1, mixer.conv_kernel_size - 1, mixer.conv_dim), dtype)),
                           (None, mx.zeros((1, mixer.num_v_heads, mixer.head_v_dim, mixer.head_k_dim), mx.float32))])
        else:
            states.append([])
    return states


def segment_layer(layer, h, states, positions, *, flash=None):
    """Continue one decoder layer without hiding state in mutable model caches.

    ``flash`` is None or the pinned extension's (path, sha256).
    """
    x = layer.input_layernorm(h)
    if layer.is_linear:
        from mlx_vlm.models.cache import ArraysCache
        cache = ArraysCache(size=2)
        cache.cache = list(states)
        mixed = gdn_mixer(layer.linear_attn, x, cache)
        # A contiguous trailing view can still own the full segment's QKV
        # allocation. Materialize this small carry without detaching its VJP.
        conv, recurrent = cache.cache
        carried = (conv + mx.zeros_like(conv), recurrent)
    else:
        attention = layer.self_attn
        q, k, v, gate, _ = attention._prepare_projected_qkv(
            attention.q_proj(x), attention.k_proj(x), attention.v_proj(x),
            cache=None, position_ids=positions, position_embeddings=None, mask=None)
        keys = mx.concatenate([*states[0::2], k], axis=2) if states else k
        values = mx.concatenate([*states[1::2], v], axis=2) if states else v
        if flash is not None:
            from .mlx_flash_attention import attention as attend
            out = attend(q, keys, values, attention.scale, extension_path=flash[0], extension_sha256=flash[1])
        else:
            out = tiled_causal_attention(q, keys, values, attention.scale, keys.shape[2] - q.shape[2])
        out = out.transpose(0, 2, 1, 3).reshape(h.shape[0], h.shape[1], -1)
        mixed = attention.o_proj(out * mx.sigmoid(gate))
        carried = (k, v)
    h = h + mixed
    return (h + layer.mlp(layer.post_attention_layernorm(h)), *carried)
