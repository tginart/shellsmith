"""
Purpose: Select the QSA key blocks of one training replay segment with a Metal top-k.
Input: The upstream indexer, BF16 hidden rows, the QSA cache of the prior
    segments, and the position IDs.
Output: The selection namespace of Qwen4ExpQSAIndexer.select.
Method:
    1. One kernel makes the block scores with the upstream formula: FP32
       products of the normalized, rotated query heads and mean-pooled block
       keys, ReLU, a sum over the heads, and the causal block mask.
    2. One kernel per query row selects the top-k blocks with a radix select.
Note: The upstream argpartition takes most of the selection time on real
    scores. The selected set is the same, except for exactly equal scores at
    the cut-off, where this kernel takes the lowest block indices.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlx.core as mx

# L: query rows, N: indexer heads, B: complete key blocks, K: selected blocks,
# D: indexer head width.
SHADER_ROOT = Path(__file__).with_name("metal")
THREADS = 256
# 32-block tiles per simdgroup of the score kernel.
SCORE_TILES = 8


def select_blocks(indexer: Any, hidden_BLD: Any, cache: Any, positions: Any) -> Any:
    """
    Purpose: Replace Qwen4ExpQSAIndexer.select for one unpadded training segment.
    Input: The indexer, hidden rows (1, L, D), a fresh QSAKVCache, and positions.
    Output: SimpleNamespace(selected_blocks, query_ends, complete_counts,
        left_padding, key_len, zero_padding), as upstream returns it.
    Note: The cache gets the indexer keys of this segment, as upstream does.
    """
    batch, length, _ = hidden_BLD.shape
    ratio, top_k = indexer.compress_ratio, indexer.block_topk
    past_index = getattr(cache, "index_offset", cache.offset)
    padding = getattr(cache, "left_padding", None)
    padded = isinstance(padding, mx.array) and padding.ndim > 0 and padding.size > 0
    blocks = (past_index + length) // ratio
    # Upstream returns no selection or pads rows in these cases.
    if batch != 1 or padded or blocks == 0 or (length == 1 and blocks <= top_k):
        return indexer.select(hidden_BLD, cache, positions)
    qk = indexer.index_qk_proj(hidden_BLD).reshape(
        batch, length, indexer.n_heads + indexer.kv_heads, indexer.head_dim)
    query_BNLD = indexer.q_layernorm(qk[:, :, :indexer.n_heads]).transpose(0, 2, 1, 3)
    raw_keys, full_positions = cache.update_indexer(
        qk[:, :, indexer.n_heads:].squeeze(2), positions)
    assert raw_keys.shape[1] // ratio == blocks
    query_BNLD = indexer._apply_rope(query_BNLD, positions)
    pooled = raw_keys[:, : blocks * ratio].reshape(batch, blocks, ratio, indexer.head_dim)
    pooled = indexer.k_layernorm(
        mx.mean(pooled.astype(mx.float32), axis=2).astype(raw_keys.dtype))[:, None]
    block_ids_B = mx.arange(blocks, dtype=mx.int32)
    pooled = indexer._apply_rope(pooled, full_positions[..., block_ids_B * ratio])
    cache.index_block_keys = pooled
    cache.index_block_ratio = ratio
    query_ends_L = past_index + mx.arange(length, dtype=mx.int32) + 1
    complete_L = query_ends_L // ratio
    scores_LB = block_scores(query_BNLD[0], pooled[0, 0], complete_L,
                             width=max(blocks, top_k))
    selected_LK = top_blocks(scores_LB, top_k)
    selected_BLK = mx.where(selected_LK < complete_L[:, None], selected_LK, -1)[None]
    return SimpleNamespace(
        selected_blocks=selected_BLK,
        query_ends=query_ends_L[None],
        complete_counts=complete_L[None],
        left_padding=mx.zeros((batch,), dtype=mx.int32),
        key_len=raw_keys.shape[1],
        zero_padding=True,
    )


def block_scores(query_NLD: Any, pooled_BD: Any, complete_L: Any, *, width: int) -> Any:
    """
    Purpose: Make the masked QSA block scores of every query row in one kernel.
    Input: Rotated query heads (N, L, D) and pooled block keys (B, D), both
        BF16, the complete block count of each row, and the output width.
    Output: FP32 scores (L, width): the sum over heads of ReLU(q . k) / sqrt(D)
        for a visible block, and -inf elsewhere.
    """
    heads, length, head_width = query_NLD.shape
    blocks = pooled_BD.shape[0]
    assert heads == 4 and head_width == 128 and width >= blocks
    assert query_NLD.dtype == pooled_BD.dtype == mx.bfloat16
    tiles = -(-width // 32)
    per_group = SCORE_TILES * 4
    return _kernel("scores")(
        inputs=[query_NLD, pooled_BD, complete_L.astype(mx.int32),
                mx.array([length, blocks, width, SCORE_TILES], mx.int32)],
        grid=(-(-tiles // per_group) * 128, -(-length // 4), 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(length, width)],
        output_dtypes=[mx.float32],
    )[0]


def top_blocks(scores_LB: Any, top_k: int) -> Any:
    """Return the indices (L, K) int32 of the K largest scores of each row, ascending."""
    length, blocks = scores_LB.shape
    assert scores_LB.dtype == mx.float32 and blocks >= top_k
    return _kernel("topk")(
        inputs=[mx.contiguous(scores_LB), mx.array([length, blocks, top_k], mx.int32)],
        grid=(length * THREADS, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(length, top_k)],
        output_dtypes=[mx.int32],
    )[0]


@lru_cache(maxsize=2)
def _kernel(phase: str) -> Any:
    """Compile one repository-owned selection shader, keyed by its source hash."""
    header = """
inline uint order_key(float value) {
    const uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}
"""
    if phase == "scores":
        header = (SHADER_ROOT / "gdn_nax_common.metal").read_text() + """
typedef metal::vec<float, 8> ffrag;
typedef metal::vec<bfloat, 8> bfrag;
inline ffrag zero_ffrag() { return ffrag(0.0f); }
"""
    names = {"scores": (["q", "k", "complete", "dims"], ["scores"]),
             "topk": (["scores", "dims"], ["selected"])}[phase]
    source = (SHADER_ROOT / f"qsa_{phase}.metal").read_text()
    identity = hashlib.sha256((header + source).encode()).hexdigest()[:12]
    return mx.fast.metal_kernel(
        name=f"shellsmith_qsa_{phase}_{identity}",
        input_names=names[0],
        output_names=names[1],
        header=header,
        source=source,
        ensure_row_contiguous=True,
    )
