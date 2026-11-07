"""
Purpose: Batch projection work across requests with independent caches and positions.
Input: Ready requests for one pinned policy. Output: One decode step for each request.
Note: Adapted from High Command mlx_decode.py at 1f51929bf55c for Qwen3.5.
"""

from typing import Any

import mlx.core as mx
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import ArraysCache

from .mlx_request_cache import state_arrays

# B: batch, L: tokens, D: hidden width, V: vocabulary, H: attention heads.


def decode_rows(engine: Any, requests: list) -> None:
    """
    Purpose: Advance one pending token per request without padding tokens.
    Input: Requests with materialized tokens. Output: Updated caches and next logits.
    Method: Batch projections and recurrent state. Apply attention per request.
    """
    if getattr(engine, "replay_ops", None) is not None:
        engine.resident.decode_rows(engine, requests)
        return
    lm = engine.lm
    ids_BL = mx.array([[request.tokens[-1]] for request in requests], mx.int32)
    hidden_BLD = lm.model.embed_tokens(ids_BL).astype(mx.bfloat16)
    positions_B = mx.array(
        [
            len(request.body["prompt_token_ids"]) + len(request.tokens) - 1
            for request in requests
        ],
        mx.int32,
    )
    positions = mx.broadcast_to(positions_B[None, :, None], (3, len(requests), 1))
    is_bf16 = engine.execution_profile == "bf16-fast"
    for index, layer in enumerate(lm.layers):
        caches = [request.cache[index] for request in requests]
        if layer.is_linear:
            combined = ArraysCache(size=len(caches[0].cache))
            combined.cache = [
                mx.concatenate(values, axis=0)
                for values in zip(*(cache.cache for cache in caches))
            ]
            if is_bf16:
                from .numerics.bf16 import _decode_gdn_token

                hidden_BLD, first, second = _decode_gdn_token(
                    layer, hidden_BLD, combined.cache
                )
                combined.cache = [first, second]
            else:
                mixed_BLD = layer.linear_attn(
                    layer.input_layernorm(hidden_BLD), mask=None, cache=combined
                )
                hidden_BLD = _residual(layer, hidden_BLD, mixed_BLD, is_bf16=False)
            for row, cache in enumerate(caches):
                cache.cache = [value[row : row + 1] for value in combined.cache]
        else:
            mixed_BLD = _attention_rows(
                layer.self_attn,
                _norm(layer.input_layernorm, hidden_BLD, is_bf16),
                caches,
                positions,
            )
            hidden_BLD = _residual(layer, hidden_BLD, mixed_BLD, is_bf16=is_bf16)
    hidden_BLD = _norm(lm.model.norm, hidden_BLD, is_bf16)
    head = (
        lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head
    )
    logits_BV = head(hidden_BLD)[:, -1].astype(engine.compute_dtype)
    mx.eval(logits_BV, [state_arrays(cache) for r in requests for cache in r.cache])
    for row, request in enumerate(requests):
        request.logits = logits_BV[row : row + 1]
        request.needs_decode = False


def _attention_rows(
    attention: Any, mixed_BLD: Any, caches: list, positions: Any
) -> Any:
    """
    Purpose: Reuse batched QKV projections with native request-local attention math.
    Input: Hidden rows, native caches, and absolute positions. Output: Attention rows.
    Note: Native dispatch preserves packed 8-bit cache behavior.
    """
    raw = [
        projection(mixed_BLD)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    outputs = []
    for row, cache in enumerate(caches):
        q_BHLD, k_BHLD, v_BHLD, gate_BLD, _ = attention._prepare_projected_qkv(
            *(value[row : row + 1] for value in raw),
            cache,
            positions[:, row : row + 1],
            None,
            None,
        )
        output_BHLD = scaled_dot_product_attention(
            q_BHLD, k_BHLD, v_BHLD, cache=cache, scale=attention.scale, mask=None
        )
        output_BLD = output_BHLD.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        outputs.append(output_BLD * mx.sigmoid(gate_BLD))
    return attention.o_proj(mx.concatenate(outputs, axis=0))


def _residual(layer: Any, hidden_BLD: Any, mixed_BLD: Any, *, is_bf16: bool) -> Any:
    middle_BLD = hidden_BLD + mixed_BLD
    return middle_BLD + layer.mlp(
        _norm(layer.post_attention_layernorm, middle_BLD, is_bf16)
    )


def _norm(norm: Any, hidden_BLD: Any, is_bf16: bool) -> Any:
    if is_bf16:
        return mx.fast.rms_norm(hidden_BLD, norm.weight.astype(mx.bfloat16), norm.eps)
    return norm(hidden_BLD)
