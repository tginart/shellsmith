"""
Purpose: Execute Flash layers with explicit differentiable replay state.
Input: One resident decoder, token segments, and shared-prompt state references.
Output: Hyper-connection rows, recurrent state, QSA chunks, and adapter gradients.
Note: QSA routing decisions are discrete. Their source history has no gradient.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import mlx.core as mx
from mlx_lm.models.switch_layers import SwiGLU, _gather_sort, _scatter_unsort
from mlx_vlm.models.base import scaled_dot_product_attention
from mlx_vlm.models.cache import ArraysCache
from mlx_vlm.models.qwen4_exp.language import QSAKVCache, QSAQuantizedKVCache
from mlx_vlm.models.qwen4_exp.qsa_kernel import dispatch_qsa_attention
from mlx_vlm.models.quantized_verifier import (
    optimized_affine_linear, optimized_affine_linears, singleton_quantized_linear,
)

from .adapter import LoRALinear
from .bf16 import gdn_forward
from .gdn_fused import causal_conv_silu
from .expert_metal import sorted_lora_projections
from .hyper_metal import hyper_connection, inject
from .moe_glue import combine_routed, gather_sum, is_supported, swiglu
from .qsa_metal import selection_slots, sparse_attention
from .qsa_select import select_blocks

# B: batch, L: segment tokens, D: hidden, H: heads, K: head width, V: vocabulary.
ATTENTION_SCORE_BYTES = 512 << 20
# Below this many tokens, fixed expert-tile capacity costs more than it saves
# (bench prefill-v2: tiles and sorted rows tie at 256 tokens).
PREFILL_TILE_MIN_TOKENS = 256


class OwnedQSAKVCache(QSAKVCache):
    """Share immutable QSA prefixes and replace complete buffers on each append."""

    def update_and_fetch(self, keys_BHLK: Any, values_BHLK: Any) -> tuple:
        self.keys = (
            keys_BHLK
            if self.keys is None
            else mx.concatenate([self.keys[:, :, : self.offset], keys_BHLK], axis=2)
        )
        self.values = (
            values_BHLK
            if self.values is None
            else mx.concatenate([self.values[:, :, : self.offset], values_BHLK], axis=2)
        )
        self.offset += keys_BHLK.shape[2]
        return self.keys, self.values


def fork_qsa_cache(source: Any) -> Any:
    result = OwnedQSAKVCache()
    result.state = source.state
    return result


class FlashOps:
    """
    Purpose: Bind Flash-specific state and math to the shared replay coordinator.
    Input: Flash layer objects and complete branch token IDs.
    Output: Three carried arrays per layer with explicit producer identities.
    Note: training_moe_chunk selects grouped expert GEMMs for training replay only.
        None keeps the original per-row routed-expert products.
    """

    def __init__(self, *, training_moe_chunk: int | None = None):
        assert training_moe_chunk is None or training_moe_chunk >= 1
        self.training_moe_chunk = training_moe_chunk

    @staticmethod
    def expand(lm: Any, hidden_BLD: Any) -> Any:
        return mx.tile(hidden_BLD, (1, 1, lm.args.hc_count))

    @staticmethod
    def context(
        branch: dict, start: int, stop: int, *, store: dict | None = None,
        mode: str | None = None,
    ) -> tuple:
        """
        Purpose: Return the token context of one replay segment.
        Input: The branch, the segment bounds, and optionally the replay's memo
            store with a mode ("keep" or "take", see SegmentMemo).
        Output: (ids, preceding ids), and the segment memo when a store is given.
        """
        ids_BL = branch["input_ids"]
        context = (ids_BL[:, start:stop], ids_BL[:, max(0, start - 2) : start])
        if store is None:
            return context
        return (*context, SegmentMemo(store, (id(branch), start), mode))

    @staticmethod
    def carry_count(layer: Any) -> int:
        return 3 if not layer.is_linear or "ple" in layer else 2

    @staticmethod
    def replace_state(layer: Any) -> bool:
        return layer.is_linear

    @staticmethod
    def differentiable_carry(layer: Any, kind: int) -> bool:
        return layer.is_linear or kind != 2

    @staticmethod
    def initial_states(layers: list, dtype: Any) -> list:
        """Create empty BF16 recurrent state and PLE convolution state."""
        states = []
        for layer in layers:
            if not layer.is_linear:
                states.append([])
                continue
            mixer = layer.linear_attn
            values = [
                mx.zeros((1, mixer.conv_kernel_size - 1, mixer.conv_dim), dtype),
                mx.zeros(
                    (1, mixer.num_v_heads, mixer.head_v_dim, mixer.head_k_dim), dtype
                ),
            ]
            if "ple" in layer:
                values.append(
                    mx.zeros(
                        (
                            1,
                            layer.ple.short_conv_state_len,
                            layer.ple.hidden_size * layer.ple.hc_count,
                        ),
                        dtype,
                    )
                )
            states.append([(None, value) for value in values])
        return states

    def layer(
        self, layer: Any, hidden_BLD: Any, states: list, positions: Any, context: tuple
    ) -> tuple:
        """Run one replay layer with this instance's training MoE layout."""
        return FlashOps.layer_math(
            layer, hidden_BLD, states, positions, context,
            moe_sorted=bool(self.training_moe_chunk) and layer.training,
            training_kernels=layer.training,
        )

    @staticmethod
    def layer_math(
        layer: Any,
        hidden_BLD: Any,
        states: list,
        positions: Any,
        context: tuple,
        *,
        inference_cache: Any = None,
        moe_chunk: int | None = None,
        moe_sorted: bool = False,
        training_kernels: bool = False,
    ) -> tuple:
        """
        Purpose: Continue the exact Flash residual graph over one replay segment.
        Input: BF16 residual rows, token IDs, and immutable state arrays.
        Output: BF16 rows and new local state arrays. Earlier chunks retain identity.
        Note: Quantized inference updates its request cache. Replay uses BF16 arrays.
            moe_sorted selects exact per-expert row segments for training replay.
            moe_chunk selects fixed expert tiles for prompt prefill.
            training_kernels selects the fused hyper-connection, residual
            injection, and QSA selection kernels and the fused GDN kernels
            (backend "metal_fused").
        """
        assert hidden_BLD.dtype == mx.bfloat16
        ids_BL, preceding_BL, *memo = context
        ple_state = None
        if "ple" in layer:
            cache = ArraysCache(size=4)
            embedding = layer.ple.ple_embedding
            previous = mx.full(
                (ids_BL.shape[0], embedding.context_len),
                embedding.eos_token_id,
                mx.int64,
            )
            if preceding_BL.shape[1]:
                previous[:, -preceding_BL.shape[1] :] = preceding_BL.astype(mx.int64)
            cache.cache = [*states[:2], states[2], previous]
            if memo and training_kernels:
                rows_BLE = memo[0].get(
                    "ple_rows", lambda: layer.ple.ple_embedding(ids_BL, cache)
                )
                hidden_BLD = hidden_BLD + _ple_residual(
                    layer.ple, hidden_BLD, rows_BLE, cache
                )
            else:
                hidden_BLD = hidden_BLD + layer.ple(hidden_BLD, ids_BL, cache, None)
            ple_state = cache[2]
        hyper = hyper_connection if training_kernels else _module_call
        residual = inject if training_kernels else _inject
        mixed_BLD, hyper_BLD, injection_BLH = hyper(
            layer.attn_hyper_connection, hidden_BLD
        )
        if layer.is_linear:
            output_BLD, conv_BCD, recurrent_BHVK = gdn_forward(
                layer.linear_attn,
                mixed_BLD,
                states[:2],
                gate_activation=layer.linear_attn.norm.activation,
                backend="metal_fused" if training_kernels else "metal",
            )
            carried = [conv_BCD, recurrent_BHVK]
            if ple_state is not None:
                carried.append(ple_state + mx.zeros_like(ple_state))
        elif isinstance(inference_cache, QSAQuantizedKVCache):
            assert not layer.training
            output_BLD = layer.self_attn(
                mixed_BLD,
                mask="causal",
                cache=inference_cache,
                position_ids=positions,
            )
            carried = []
        else:
            output_BLD, carried = _attention(
                layer.self_attn, mixed_BLD, states, positions,
                training_kernels=training_kernels,
            )
        hidden_BLD = residual(hyper_BLD, output_BLD, injection_BLH)
        mixed_BLD, hyper_BLD, injection_BLH = hyper(
            layer.mlp_hyper_connection, hidden_BLD
        )
        if moe_sorted:
            output_BLD = _moe_sorted(layer.mlp, mixed_BLD)
        elif moe_chunk is None:
            output_BLD = _moe(layer.mlp, mixed_BLD, sort_rows=not layer.training)
        else:
            output_BLD = _moe_chunked(layer.mlp, mixed_BLD, chunk=moe_chunk)
        return residual(hyper_BLD, output_BLD, injection_BLH), *carried

    @staticmethod
    def head(lm: Any, hidden_LD: Any, tokens_L: Any, dtype: Any) -> Any:
        mixed_LD = lm.model.hyper_connection_mixer(hidden_LD[None])[0]
        head = (
            lm.model.embed_tokens.as_linear
            if lm.args.tie_word_embeddings
            else lm.lm_head
        )
        logits_LV = head(mixed_LD).astype(dtype)
        return mx.take_along_axis(logits_LV, tokens_L[:, None], axis=-1)[
            :, 0
        ] - mx.logsumexp(logits_LV, axis=-1)


class SegmentMemo:
    """
    Purpose: Carry token-only values of one segment from its recorded forward to
        its group VJP.
    Input: The replay's store dict, the segment key, and a mode. "keep" stores
        each value that the recorded forward makes. "take" returns and removes it.
    Note: A take that misses makes the value again, so no result depends on the
        memo. The store holds a value only between the forward and the backward
        of its segment.
    """

    def __init__(self, store: dict, key: tuple, mode: str):
        assert mode in ("keep", "take")
        self.store, self.key, self.mode = store, key, mode

    def get(self, name: str, make: Any) -> Any:
        key = (*self.key, name)
        if self.mode == "take" and key in self.store:
            return self.store.pop(key)
        value = make()
        if self.mode == "keep":
            self.store[key] = value
        return value


def _ple_residual(ple: Any, hidden_BLD: Any, rows_BLE: Any, cache: Any) -> Any:
    """
    Purpose: Apply the upstream PLE residual to given n-gram embedding rows.
    Input: The Qwen4ExpPLELayer, the hidden rows, the n-gram rows of the
        segment, and the layer cache with the convolution state.
    Output: The PLE residual, with the operations of Qwen4ExpPLELayer. The
        dilated short convolution and its SiLU use the fused GDN conv kernel.
    Note: The VJP recompute gets the rows from the segment memo. The n-gram
        table is ~50 GB. Under replay memory pressure the system pages out its
        rows, and a lookup in the VJP took ~0.45 s against ~0.3 ms in the forward.
    """
    keys = ple.norm_key(ple.key_proj(rows_BLE)).reshape(
        *hidden_BLD.shape[:-1], ple.hc_count, ple.hidden_size
    )
    values = ple.value_proj(rows_BLE)
    queries = ple.norm_query(hidden_BLD).reshape(
        *hidden_BLD.shape[:-1], ple.hc_count, ple.hidden_size
    )
    gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(ple.hidden_size)
    gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
    gated = (mx.sigmoid(gate) * values[..., None, :]).reshape(*hidden_BLD.shape)
    normed = ple.norm_conv(gated)
    state = cache[2]
    width = ple.short_conv_state_len
    cache.update_window(2, mx.concatenate([state, normed[:, -width:]], axis=1), width)
    return gated + causal_conv_silu(
        state, normed, ple.conv1d.weight, dilation=ple.conv_dilation
    )


def _module_call(module: Any, hidden_BLD: Any) -> tuple:
    return module(hidden_BLD)


def _inject(hyper_BLW: Any, output_BLD: Any, injection_BLG: Any) -> Any:
    """Return hyper + injection * output with MLX operations, as inference does."""
    injected = output_BLD[..., None, :] * injection_BLG[..., None]
    return hyper_BLW + injected.reshape(hyper_BLW.shape)


def _upstream_select(indexer: Any, hidden_BLD: Any, cache: Any, positions: Any) -> Any:
    return indexer.select(hidden_BLD, cache, positions)


def _attention(
    attention: Any, hidden_BLD: Any, states: list, positions: Any, *,
    training_kernels: bool = False,
) -> tuple:
    """
    Purpose: Preserve QSA selection with a bounded differentiable attention kernel.
    Input: Current rows and prior KV/indexer chunks.
    Output: The projected result and three current-segment chunks.
    Note: With an indexer selection, the sparse Metal kernel reads only the
        selected blocks and the tail of each query. Without one, query tiles
        checkpoint dense causal score matrices. Base weights remain quantized.
        training_kernels selects the blocks with the Metal top-k of qsa_select.
    """
    cache = QSAKVCache()
    if states:
        cache.keys = mx.concatenate(states[0::3], axis=2)
        cache.values = mx.concatenate(states[1::3], axis=2)
        cache.offset = cache.keys.shape[2]
        cache.index_keys = mx.concatenate(states[2::3], axis=1)
        cache.index_position_ids = mx.broadcast_to(
            mx.arange(cache.offset, dtype=mx.int32)[None],
            (hidden_BLD.shape[0], cache.offset),
        )
    select = select_blocks if training_kernels else _upstream_select
    selection = select(
        attention.indexer, mx.stop_gradient(hidden_BLD), cache, positions
    )
    raw = [
        projection(hidden_BLD)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    query_BHLK, key_BJLK, value_BJLK, gate_BLD, _ = attention._prepare_projected_qkv(
        *raw, cache, positions, None, None
    )
    length = hidden_BLD.shape[1]
    local_keys = key_BJLK[:, :, -length:] + mx.zeros_like(key_BJLK[:, :, -length:])
    local_values = value_BJLK[:, :, -length:] + mx.zeros_like(
        value_BJLK[:, :, -length:]
    )
    local_index = mx.stop_gradient(
        cache.index_keys[:, -length:] + mx.zeros_like(cache.index_keys[:, -length:])
    )
    batch, heads, _, _ = query_BHLK.shape
    if selection is not None:
        slots_LS, ends_L = selection_slots(selection)
        output_BLHK = sparse_attention(
            query_BHLK.transpose(0, 2, 1, 3), key_BJLK, value_BJLK, slots_LS, ends_L,
            attention.scale,
        )
        output_BLD = output_BLHK.reshape(batch, length, -1)
        output_BLD = attention.o_proj(output_BLD * mx.sigmoid(gate_BLD))
        return output_BLD, [local_keys, local_values, local_index]
    tile_size = max(
        1, min(128, ATTENTION_SCORE_BYTES // (batch * heads * key_BJLK.shape[2] * 4))
    )
    outputs = []
    for start in range(0, length, tile_size):
        stop = min(length, start + tile_size)
        tile_selection = (
            None
            if selection is None
            else SimpleNamespace(
                **{
                    **vars(selection),
                    "selected_blocks": selection.selected_blocks[:, start:stop],
                    "query_ends": selection.query_ends[:, start:stop],
                    "complete_counts": selection.complete_counts[:, start:stop],
                }
            )
        )

        def tile(
            query: Any, key: Any, value: Any, selected: Any = tile_selection
        ) -> Any:
            if selected is None:
                ends = positions[0, :, start:stop] + 1
                mask = (
                    mx.arange(key.shape[2])[None, None, None, :]
                    < ends[:, None, :, None]
                )
            else:
                mask = attention.indexer.build_mask(selected)
            return mx.fast.scaled_dot_product_attention(
                query, key, value, scale=attention.scale, mask=mask
            )

        outputs.append(
            mx.checkpoint(tile)(query_BHLK[:, :, start:stop], key_BJLK, value_BJLK)
        )
    output_BHLK = mx.concatenate(outputs, axis=2)
    output_BLD = output_BHLK.transpose(0, 2, 1, 3).reshape(batch, length, -1)
    output_BLD = attention.o_proj(output_BLD * mx.sigmoid(gate_BLD))
    return output_BLD, [local_keys, local_values, local_index]


def _moe(mixer: Any, hidden_BLD: Any, *, sort_rows: bool = False) -> Any:
    """
    Purpose: Apply routed experts with fixed expert IDs and differentiable route scores.
    Input: Hidden rows and an optional inference-only row permutation.
    Output: Routed and shared expert contributions in the original token order.
    Note: The permutation improves locality. The projection reduction algorithm stays unchanged.
    """
    gates_BLE = mx.softmax(mixer.gate(hidden_BLD), axis=-1, precise=True)
    indices_BLK = mx.stop_gradient(
        mx.argpartition(gates_BLE, kth=-mixer.top_k, axis=-1)[..., -mixer.top_k :]
    )
    scores_BLK = mx.take_along_axis(gates_BLE, indices_BLK, axis=-1)
    scores_BLK = scores_BLK / mx.sum(scores_BLK, axis=-1, keepdims=True)
    expanded = mx.expand_dims(hidden_BLD, (-2, -3))
    indices = indices_BLK
    inverse = None
    if sort_rows and indices_BLK.size >= 64:
        expanded, indices, inverse = _gather_sort(expanded, indices_BLK)
    expert = mixer.switch_mlp
    up = expert.up_proj(expanded, indices)
    gate = expert.gate_proj(expanded, indices)
    routed = expert.down_proj(expert.activation(up, gate), indices)
    if inverse is not None:
        routed = _scatter_unsort(routed, inverse, indices_BLK.shape)
    routed_BLKD = routed.squeeze(-2)
    routed_BLD = mx.sum(routed_BLKD * scores_BLK[..., None], axis=-2)
    shared_BLD = mixer.shared_expert(hidden_BLD) * mixer._shared_expert_scale(
        hidden_BLD
    )
    return routed_BLD + shared_BLD


def _moe_chunked(
    mixer: Any, hidden_BLD: Any, *, chunk: int, dequantize: bool = False
) -> Any:
    """
    Purpose: Apply routed experts as grouped GEMMs over fixed-size row chunks.
    Input: Hidden rows, a chunk height, and an optional BF16 dequantized path.
    Output: The same routed and shared expert contributions as _moe.
    Method:
        1. Route with the production softmax, top-k IDs, and renormalized scores.
        2. Sort (token, expert) rows by expert. Split each expert run into chunks.
        3. Gather rows into a fixed (chunks, chunk, D) layout. Empty slots stay zero.
        4. Run each projection once per chunk with its expert weight and LoRA.
        5. Gather the rows back and apply the route scores.
    Note: The chunk count is a static upper bound, so no host synchronization occurs.
        Each output row has its own reduction; only the kernel tiling changes.
    """
    gates_BLE = mx.softmax(mixer.gate(hidden_BLD), axis=-1, precise=True)
    indices_BLK = mx.stop_gradient(
        mx.argpartition(gates_BLE, kth=-mixer.top_k, axis=-1)[..., -mixer.top_k :]
    )
    scores_BLK = mx.take_along_axis(gates_BLE, indices_BLK, axis=-1)
    scores_BLK = scores_BLK / mx.sum(scores_BLK, axis=-1, keepdims=True)
    layout = _chunk_layout(indices_BLK, gates_BLE.shape[-1], chunk)
    rows_TD = hidden_BLD.reshape(-1, hidden_BLD.shape[-1])
    padded_TD = mx.concatenate([rows_TD, mx.zeros_like(rows_TD[:1])], axis=0)
    chunks_NCD = padded_TD[layout["source_token"]].reshape(-1, chunk, rows_TD.shape[-1])
    expert = mixer.switch_mlp
    experts_N = layout["chunk_expert"]
    up = _chunk_projection(expert.up_proj, chunks_NCD, experts_N, dequantize)
    gate = _chunk_projection(expert.gate_proj, chunks_NCD, experts_N, dequantize)
    routed_NCD = _chunk_projection(
        expert.down_proj, expert.activation(up, gate), experts_N, dequantize
    )
    routed_RD = routed_NCD.reshape(-1, routed_NCD.shape[-1])[layout["row_slot"]]
    routed_BLKD = routed_RD.reshape(*indices_BLK.shape, -1)
    routed_BLD = mx.sum(routed_BLKD * scores_BLK[..., None], axis=-2)
    shared_BLD = mixer.shared_expert(hidden_BLD) * mixer._shared_expert_scale(
        hidden_BLD
    )
    return routed_BLD + shared_BLD


def _moe_sorted(mixer: Any, hidden_BLD: Any) -> Any:
    """
    Purpose: Apply routed experts as exact per-expert segments of sorted rows.
    Input: Hidden rows of one training replay segment.
    Output: The same routed and shared expert contributions as _moe.
    Method:
        1. Route with the production softmax, top-k IDs, and renormalized scores.
        2. Sort the (token, expert) rows by expert. Each expert owns one segment.
        3. Run each projection once with the sorted-segment quantized kernel. The
           up and gate projections share one input-gradient kernel.
        4. One kernel sums each token's routed rows with the route scores.
    Note: No row is padding. The row permutations have gather-only derivatives,
        so the backward pass has no scatter-add.
    """
    gates_BLE = mx.softmax(mixer.gate(hidden_BLD), axis=-1, precise=True)
    indices_BLK = mx.stop_gradient(
        mx.argpartition(gates_BLE, kth=-mixer.top_k, axis=-1)[..., -mixer.top_k :]
    )
    scores_BLK = mx.take_along_axis(gates_BLE, indices_BLK, axis=-1)
    scores_BLK = scores_BLK / mx.sum(scores_BLK, axis=-1, keepdims=True)
    flat_R = indices_BLK.reshape(-1)
    order_R = mx.stop_gradient(mx.argsort(flat_R).astype(mx.int32))
    inverse_TK = mx.stop_gradient(
        mx.argsort(order_R).astype(mx.int32).reshape(-1, mixer.top_k)
    )
    rows_TD = hidden_BLD.reshape(-1, hidden_BLD.shape[-1])
    x_R1D = _permute_rows(rows_TD, order_R, inverse_TK)[:, None]
    experts_R = mx.stop_gradient(flat_R[order_R].astype(mx.uint32))
    expert = mixer.switch_mlp
    up_R1F, gate_R1F = sorted_lora_projections(
        x_R1D, [expert.up_proj, expert.gate_proj], experts_R
    )
    if isinstance(expert.activation, SwiGLU) and is_supported(up_R1F):
        hidden_R1F = swiglu(up_R1F, gate_R1F)
    else:
        hidden_R1F = expert.activation(up_R1F, gate_R1F)
    routed_RD = expert.down_proj(hidden_R1F, experts_R, sorted_indices=True)[:, 0]
    if is_supported(routed_RD):
        routed_TD = combine_routed(
            routed_RD, scores_BLK.reshape(-1, mixer.top_k), inverse_TK, order_R
        )
        routed_BLD = routed_TD.reshape(hidden_BLD.shape)
    else:
        routed_BLKD = routed_RD[inverse_TK.reshape(-1)].reshape(*indices_BLK.shape, -1)
        routed_BLD = mx.sum(routed_BLKD * scores_BLK[..., None], axis=-2)
    shared_BLD = mixer.shared_expert(hidden_BLD) * mixer._shared_expert_scale(
        hidden_BLD
    )
    return routed_BLD + shared_BLD


@mx.custom_function
def _permute_rows(rows_TD: Any, order_R: Any, inverse_TK: Any) -> Any:
    """Return routed row r = token order_R[r] // top_k, in expert order."""
    return rows_TD[order_R // inverse_TK.shape[1]]


@_permute_rows.vjp
def _permute_rows_vjp(primals: tuple, cotangent_RD: Any, output: Any) -> tuple:
    rows_TD, order_R, inverse_TK = primals
    # Each token gathers its top_k routed-row gradients. No scatter-add occurs.
    if is_supported(cotangent_RD):
        d_TD = gather_sum(cotangent_RD, inverse_TK)
    else:
        d_TKD = cotangent_RD[inverse_TK.reshape(-1)].reshape(*inverse_TK.shape, -1)
        d_TD = d_TKD.sum(axis=1)
    return d_TD.astype(rows_TD.dtype), mx.zeros_like(order_R), mx.zeros_like(inverse_TK)


def _chunk_layout(indices_BLK: Any, experts: int, chunk: int) -> dict:
    """
    Purpose: Map routed rows to fixed-size per-expert chunks without host reads.
    Input: Top-k expert IDs, the expert count, and the chunk height.
    Output: Source token per slot, expert per chunk, and slot per routed row.
    """
    flat_R = indices_BLK.reshape(-1).astype(mx.int32)
    rows, top_k = flat_R.size, indices_BLK.shape[-1]
    capacity = -(-rows // chunk) + experts
    order_R = mx.argsort(flat_R)
    expert_R = flat_R[order_R]
    counts_E = mx.zeros((experts,), mx.int32).at[flat_R].add(1)
    starts_E = mx.cumsum(counts_E) - counts_E
    chunks_E = (counts_E + chunk - 1) // chunk
    chunk_starts_E = mx.cumsum(chunks_E) - chunks_E
    rank_R = mx.arange(rows, dtype=mx.int32) - starts_E[expert_R]
    slot_R = (chunk_starts_E[expert_R] + rank_R // chunk) * chunk + rank_R % chunk
    source = mx.full((capacity * chunk,), rows // top_k, mx.int32)
    source = source.at[slot_R].add(order_R // top_k - rows // top_k)
    chunk_expert = mx.zeros((capacity,), mx.int32).at[slot_R // chunk].maximum(expert_R)
    row_slot = mx.zeros((rows,), mx.int32).at[order_R].add(slot_R)
    return {
        "source_token": mx.stop_gradient(source),
        "chunk_expert": mx.stop_gradient(chunk_expert),
        "row_slot": mx.stop_gradient(row_slot),
    }


def _chunk_projection(module: Any, x_NCD: Any, experts_N: Any, dequantize: bool) -> Any:
    """Project each chunk with its expert's quantized weight and BF16 LoRA factors."""
    base = module.base
    if dequantize:
        weight_EFD = mx.dequantize(
            base.weight, base.scales, base.biases,
            group_size=base.group_size, bits=base.bits,
        ).astype(x_NCD.dtype)
        value_NCF = mx.gather_mm(
            x_NCD, mx.swapaxes(weight_EFD, -1, -2), rhs_indices=experts_N
        )
    else:
        value_NCF = mx.gather_qmm(
            x_NCD, base.weight, base.scales, base.biases, rhs_indices=experts_N,
            transpose=True, group_size=base.group_size, bits=base.bits,
        )
    low_NCR = mx.gather_mm(
        x_NCD, mx.swapaxes(module.a.astype(x_NCD.dtype), -1, -2), rhs_indices=experts_N
    )
    update_NCF = mx.gather_mm(
        low_NCR, mx.swapaxes(module.b.astype(x_NCD.dtype), -1, -2), rhs_indices=experts_N
    )
    return value_NCF + (module.scale * update_NCF).astype(value_NCF.dtype)


def decode_hidden(
    lm: Any, ids_BL: Any, positions: Any, caches: list, *, moe_chunk: int | None = None
) -> Any:
    """
    Purpose: Advance real request caches with the same Flash residual and LoRA math.
    Input: Token rows, positions, request caches, and an optional expert tile height.
    Output: Mixed hidden rows. The caches advance in place.
    Note: moe_chunk applies to multi-token forwards such as prompt prefill.
        Single-token rows keep the sorted per-row expert products.
    """
    if ids_BL.shape[1] < PREFILL_TILE_MIN_TOKENS:
        moe_chunk = None
    hidden_BLD = FlashOps.expand(lm, lm.model.embed_tokens(ids_BL).astype(mx.bfloat16))
    for layer, cache in zip(lm.layers, caches, strict=True):
        states = _cache_states(layer, cache)
        previous = (
            cache[3] if "ple" in layer and cache[3] is not None else ids_BL[:, :0]
        )
        hidden_BLD, *carried = FlashOps.layer_math(
            layer,
            hidden_BLD,
            states,
            positions,
            (ids_BL, previous),
            inference_cache=cache,
            moe_chunk=moe_chunk,
        )
        if not isinstance(cache, QSAQuantizedKVCache):
            _update_cache(layer, cache, carried, ids_BL, positions)
    return lm.model.hyper_connection_mixer(hidden_BLD)


def decode_rows(engine: Any, requests: list) -> None:
    """
    Purpose: Batch Flash mixer and expert projections across request-owned caches.
    Input: Ready requests for the same immutable policy.
    Output: One new logits row per request. Ragged QSA histories remain independent.
    """
    lm = engine.lm
    ids_BL = mx.array([[request.tokens[-1]] for request in requests], mx.int32)
    offsets = mx.array(
        [
            len(request.body["prompt_token_ids"]) + len(request.tokens) - 1
            for request in requests
        ],
        mx.int32,
    )
    positions = mx.broadcast_to(offsets[None, :, None], (3, len(requests), 1))
    hidden_BLD = FlashOps.expand(lm, lm.model.embed_tokens(ids_BL).astype(mx.bfloat16))
    for index, layer in enumerate(lm.layers):
        caches = [request.cache[index] for request in requests]
        combined = None
        if "ple" in layer:
            combined = ArraysCache(size=4)
            combined.cache = [
                mx.concatenate([cache[slot] for cache in caches], axis=0)
                for slot in range(4)
            ]
            hidden_BLD = hidden_BLD + layer.ple(hidden_BLD, ids_BL, combined, None)
        mixed_BLD, hyper_BLD, injection_BLH = layer.attn_hyper_connection(hidden_BLD)
        if layer.is_linear:
            states = [
                mx.concatenate([cache[slot] for cache in caches], axis=0)
                for slot in range(2)
            ]
            output_BLD, conv_BCD, recurrent_BHVK = gdn_forward(
                layer.linear_attn,
                mixed_BLD,
                states,
                decode_one=True,
                gate_activation=layer.linear_attn.norm.activation,
            )
            for row, cache in enumerate(caches):
                carried = [conv_BCD[row : row + 1], recurrent_BHVK[row : row + 1]]
                if combined is not None:
                    carried.append(combined[2][row : row + 1])
                _update_cache(
                    layer,
                    cache,
                    carried,
                    ids_BL[row : row + 1],
                    positions[:, row : row + 1],
                )
        elif len(requests) > 1 and all(
            isinstance(cache, QSAQuantizedKVCache) for cache in caches
        ):
            output_BLD = _quantized_attention_rows(
                layer.self_attn, mixed_BLD, caches, positions
            )
        else:
            values = []
            for row, cache in enumerate(caches):
                if isinstance(cache, QSAQuantizedKVCache):
                    values.append(
                        layer.self_attn(
                            mixed_BLD[row : row + 1],
                            mask="causal",
                            cache=cache,
                            position_ids=positions[:, row : row + 1],
                        )
                    )
                    continue
                value, carried = _attention(
                    layer.self_attn,
                    mixed_BLD[row : row + 1],
                    _cache_states(layer, cache),
                    positions[:, row : row + 1],
                )
                _update_cache(
                    layer,
                    cache,
                    carried,
                    ids_BL[row : row + 1],
                    positions[:, row : row + 1],
                )
                values.append(value)
            output_BLD = mx.concatenate(values, axis=0)
        injected = output_BLD[..., None, :] * injection_BLH[..., None]
        hidden_BLD = hyper_BLD + injected.reshape(hyper_BLD.shape)
        mixed_BLD, hyper_BLD, injection_BLH = layer.mlp_hyper_connection(hidden_BLD)
        output_BLD = _moe(layer.mlp, mixed_BLD, sort_rows=True)
        injected = output_BLD[..., None, :] * injection_BLH[..., None]
        hidden_BLD = hyper_BLD + injected.reshape(hyper_BLD.shape)
    mixed_BLD = lm.model.hyper_connection_mixer(hidden_BLD)
    head = (
        lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head
    )
    logits_BV = head(mixed_BLD)[:, -1].astype(engine.compute_dtype)
    mx.eval(logits_BV, [cache.state for request in requests for cache in request.cache])
    for row, request in enumerate(requests):
        request.logits = logits_BV[row : row + 1]
        request.needs_decode = False


def _quantized_attention_rows(
    attention: Any, hidden_BLD: Any, caches: list, positions: Any
) -> Any:
    """
    Purpose: Share attention projections while each request keeps its quantized cache.
    Input: One decode token per row, private QSA caches, and absolute positions.
    Output: Batched attention output with the native dense or sparse dispatch.
    Note: Index selection, RoPE, KV writes, masks, and attention stay request-local.
    """
    assert hidden_BLD.shape[:2] == (len(caches), 1)
    projected = _attention_projections(attention, hidden_BLD)
    index_BLI = projected.pop()
    outputs = []
    for row, cache in enumerate(caches):
        row_positions = positions[:, row : row + 1]
        selection = attention.indexer.select_from_projected(
            index_BLI[row : row + 1], cache, row_positions
        )
        query_BHLD, key_BHLD, value_BHLD, gate_BLD, _ = (
            attention._prepare_projected_qkv(
                *(value[row : row + 1] for value in projected),
                cache, row_positions, None, None,
            )
        )
        if selection is None:
            output_BHLD = scaled_dot_product_attention(
                query_BHLD, key_BHLD, value_BHLD,
                cache=cache, scale=attention.scale, mask="causal",
            )
        else:
            output_BHLD = dispatch_qsa_attention(
                query_BHLD, key_BHLD, value_BHLD,
                selection.selected_blocks, selection.query_ends,
                cache=cache, scale=attention.scale,
                block_size=attention.indexer.compress_ratio,
                causal=True, mask=None,
                mask_factory=lambda selected=selection: attention.indexer.build_mask(selected),
            )
        output_BLD = output_BHLD.transpose(0, 2, 1, 3).reshape(1, 1, -1)
        outputs.append(output_BLD * mx.sigmoid(gate_BLD))
    return _project_rows(attention.o_proj, mx.concatenate(outputs, axis=0))


def _project_rows(linear: Any, hidden_BLD: Any) -> Any:
    """
    Purpose: Batch projections with the original singleton quantized reduction order.
    Input: An affine quantized projection, optional LoRA factors, and hidden rows.
    Output: Projection rows with request-local matrix-vector arithmetic.
    Note: Ordinary batched GEMM changes BF16 rounding and can change QSA routing.
    """
    base = linear.base if isinstance(linear, LoRALinear) else linear
    flat_1TD = hidden_BLD.reshape(1, -1, hidden_BLD.shape[-1])
    value_1TF = optimized_affine_linear(base, flat_1TD)
    value_BLF = (
        value_1TF.reshape(*hidden_BLD.shape[:2], -1)
        if value_1TF is not None else singleton_quantized_linear(base, hidden_BLD)
    )
    if value_BLF is None:
        raise TypeError("Flash decode requires a supported affine quantized projection.")
    return _add_lora_rows(linear, hidden_BLD, value_BLF)


def _attention_projections(attention: Any, hidden_BLD: Any) -> list:
    """
    Purpose: Fuse Q, K, V, and indexer base projections with invariant row reductions.
    Input: Native attention modules and one hidden row per request.
    Output: Four projected arrays. LoRA factors remain separate BF16 operands.
    """
    linears = (attention.q_proj, attention.k_proj, attention.v_proj,
               attention.indexer.index_qk_proj)
    bases = [linear.base if isinstance(linear, LoRALinear) else linear
             for linear in linears]
    flat_1TD = hidden_BLD.reshape(1, -1, hidden_BLD.shape[-1])
    values = optimized_affine_linears(bases, flat_1TD)
    if values is None:
        return [_project_rows(linear, hidden_BLD) for linear in linears]
    return [
        _add_lora_rows(linear, hidden_BLD,
                       value_1TF.reshape(*hidden_BLD.shape[:2], -1))
        for linear, value_1TF in zip(linears, values, strict=True)
    ]


def _add_lora_rows(linear: Any, hidden_BLD: Any, value_BLF: Any) -> Any:
    """
    Purpose: Add LoRA updates without a batch-dependent matrix reduction.
    Input: The projection module, hidden rows, and quantized base output.
    Output: The original BF16 LoRA expression with independent row arithmetic.
    """
    if not isinstance(linear, LoRALinear):
        return value_BLF
    indices_BLK = mx.zeros((*hidden_BLD.shape[:2], 1), mx.int32)
    expanded_BLKJD = mx.expand_dims(hidden_BLD, (-2, -3))
    low_BLKJR = mx.gather_mm(
        expanded_BLKJD, linear.a.T.astype(hidden_BLD.dtype)[None],
        rhs_indices=indices_BLK,
    )
    update_BLKJF = mx.gather_mm(
        low_BLKJR, linear.b.T.astype(hidden_BLD.dtype)[None],
        rhs_indices=indices_BLK,
    )
    update_BLF = update_BLKJF.squeeze(-2).squeeze(-2)
    return value_BLF + (linear.scale * update_BLF).astype(value_BLF.dtype)


def _cache_states(layer: Any, cache: Any) -> list:
    """Read native request caches without a second frozen model or a host copy."""
    if not layer.is_linear:
        if cache.keys is None or isinstance(cache, QSAQuantizedKVCache):
            return []
        return [
            cache.keys[:, :, : cache.offset],
            cache.values[:, :, : cache.offset],
            cache.index_keys,
        ]
    if cache[0] is None:
        return [value for _, value in FlashOps.initial_states([layer], mx.bfloat16)[0]]
    return list(cache.cache[: 3 if "ple" in layer else 2])


def _update_cache(
    layer: Any, cache: Any, carried: list, ids_BL: Any, positions: Any
) -> None:
    """Publish only request-local cache changes after one Flash layer."""
    if layer.is_linear:
        previous = (
            cache[3] if "ple" in layer and cache[3] is not None else ids_BL[:, :0]
        )
        cache.cache = list(carried)
        if "ple" in layer:
            length = layer.ple.ple_embedding.context_len
            history_BL = mx.concatenate([previous, ids_BL], axis=1)[:, -length:].astype(
                mx.int64
            )
            if history_BL.shape[1] < length:
                missing_BL = mx.full(
                    (ids_BL.shape[0], length - history_BL.shape[1]),
                    layer.ple.ple_embedding.eos_token_id,
                    mx.int64,
                )
                history_BL = mx.concatenate([missing_BL, history_BL], axis=1)
            cache.cache.append(history_BL)
    else:
        cache.update_and_fetch(carried[0], carried[1])
        cache.index_keys = (
            carried[2]
            if cache.index_keys is None
            else mx.concatenate([cache.index_keys, carried[2]], axis=1)
        )
        cache.index_position_ids = mx.broadcast_to(
            mx.arange(cache.offset, dtype=mx.int32)[None],
            (ids_BL.shape[0], cache.offset),
        )
        cache.clear_index_blocks()
