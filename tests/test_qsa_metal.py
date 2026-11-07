"""
Purpose: Check the sparse QSA Metal kernels against dense masked attention.
Input: Random BF16 tensors and synthetic indexer selections.
Output: Close values and close Q, K, and V derivatives.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpQSAIndexer

from studio_compute.numerics.qsa_metal import selection_slots, sparse_attention

# L: queries, H: heads, J: KV heads, K: keys, D: width, S: selected blocks.


def _selection(length: int, prior: int, top_k: int, seed: int) -> SimpleNamespace:
    """Select random distinct complete blocks per query, like a sparse indexer."""
    mx.random.seed(seed)
    ends_L = prior + mx.arange(length, dtype=mx.int32) + 1
    complete_L = ends_L // 4
    blocks = (prior + length) // 4
    scores_LB = mx.random.uniform(shape=(length, max(blocks, top_k)))
    valid_LB = mx.arange(scores_LB.shape[1])[None] < complete_L[:, None]
    scores_LB = mx.where(valid_LB, scores_LB, -mx.inf)
    selected_LS = mx.argpartition(scores_LB, kth=-top_k, axis=-1)[:, -top_k:].astype(mx.int32)
    selected_LS = mx.where(selected_LS < complete_L[:, None], selected_LS, -1)
    return SimpleNamespace(
        selected_blocks=selected_LS.astype(mx.int32)[None],
        query_ends=ends_L[None], complete_counts=complete_L[None],
        left_padding=mx.zeros((1,), mx.int32), key_len=prior + length,
        zero_padding=True,
    )


def _cosine(a, b) -> float:
    a, b = a.astype(mx.float32).reshape(-1), b.astype(mx.float32).reshape(-1)
    return float((mx.sum(a * b) / mx.sqrt(mx.sum(a * a) * mx.sum(b * b))).item())


@pytest.mark.parametrize("heads, kv_heads", [(24, 2), (4, 1)])
@pytest.mark.parametrize(
    "length, prior, top_k",
    # 4500 prior keys span three key windows of the backward kernel.
    [(64, 0, 32), (48, 4500, 64), (64, 600, 32), (37, 300, 16), (4, 8, 2), (1, 12, 2)],
)
def test_sparse_attention_matches_dense_masked_attention(
    heads: int, kv_heads: int, length: int, prior: int, top_k: int
) -> None:
    mx.random.seed(3)
    keys = prior + length
    q_BLHD = mx.random.normal((1, length, heads, 256)).astype(mx.bfloat16)
    k_BJKD = mx.random.normal((1, kv_heads, keys, 256)).astype(mx.bfloat16)
    v_BJKD = mx.random.normal((1, kv_heads, keys, 256)).astype(mx.bfloat16)
    dy_BLHD = mx.random.normal((1, length, heads, 256)).astype(mx.bfloat16)
    selection = _selection(length, prior, top_k, seed=length + prior)
    indexer = SimpleNamespace(compress_ratio=4, block_topk=top_k)
    mask_B1LK = Qwen4ExpQSAIndexer.build_mask(indexer, selection)
    slots_LS, ends_L = selection_slots(selection)

    def dense(q, k, v):
        q_BHLD = q.transpose(0, 2, 1, 3)
        return mx.fast.scaled_dot_product_attention(
            q_BHLD, k, v, scale=0.0625, mask=mask_B1LK).transpose(0, 2, 1, 3)

    def sparse(q, k, v):
        return sparse_attention(q, k, v, slots_LS, ends_L, 0.0625)

    expected, expected_grads = mx.vjp(dense, [q_BLHD, k_BJKD, v_BJKD], [dy_BLHD])
    actual, actual_grads = mx.vjp(sparse, [q_BLHD, k_BJKD, v_BJKD], [dy_BLHD])
    mx.eval(expected, expected_grads, actual, actual_grads)
    error = mx.abs(actual[0].astype(mx.float32) - expected[0].astype(mx.float32))
    assert float(error.max().item()) < 0.05
    assert _cosine(actual[0], expected[0]) > 0.9999
    for name, a, b in zip("qkv", actual_grads, expected_grads):
        assert _cosine(a, b) > 0.9995, name
