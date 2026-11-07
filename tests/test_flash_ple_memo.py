"""
Purpose: Check that the replay PLE path with memo rows equals the upstream PLE layer.
Input: A tiny Qwen4ExpPLELayer and random BF16 hidden rows.
Output: Close residuals, equal convolution states, and the memo keep/take rules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
language = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")
cache_module = pytest.importorskip("mlx_vlm.models.cache")

from studio_compute.numerics.flash_ops import SegmentMemo, _ple_residual


def _layer():
    config = SimpleNamespace(
        hidden_size=64, hc_count=4, ple_embed_dim=64, ngram_size=3, heads_per_ngram=2,
        vocab_size=50, seed=3, eos_token_id=0, ngram_vocab_size_base=97,
        make_ngram_vocab_size_divisible_by=8, ple_storage=None, split_ngram_parts=4,
        rms_norm_eps=1e-6, ple_conv_kernel_size=4,
    )
    layer = language.Qwen4ExpPLELayer(config, layer_idx=1, ple_layer_index=0)
    layer.set_dtype(mx.bfloat16)
    return layer


def _cache(state, previous):
    cache = cache_module.ArraysCache(size=4)
    cache.cache = [None, None, state, previous]
    return cache


def test_ple_residual_with_memo_rows_matches_upstream() -> None:
    mx.random.seed(2)
    layer = _layer()
    rows, width = 11, 64 * 4
    hidden = mx.random.normal((1, rows, width)).astype(mx.bfloat16)
    ids = mx.random.randint(1, 50, (1, rows))
    state = mx.random.normal((1, layer.short_conv_state_len, width)).astype(mx.bfloat16)
    previous = mx.array([[7, 9]], mx.int64)
    upstream_cache, memo_cache = _cache(state, previous), _cache(state, previous)
    expected = layer(hidden, ids, upstream_cache, None)
    embeddings = layer.ple_embedding(ids, _cache(state, previous))
    actual = _ple_residual(layer, hidden, embeddings, memo_cache)
    # The fused conv kernel rounds once, the upstream conv and SiLU twice.
    left, right = actual.astype(mx.float32), expected.astype(mx.float32)
    cosine = mx.sum(left * right) / mx.sqrt(mx.sum(left**2) * mx.sum(right**2))
    assert cosine.item() > 0.9999
    assert bool(mx.array_equal(memo_cache[2], upstream_cache[2]).item())


def test_segment_memo_keeps_once_and_takes_once() -> None:
    store, made = {}, []
    make = lambda: made.append(1) or len(made)
    assert SegmentMemo(store, (1, 0), "keep").get("rows", make) == 1
    assert SegmentMemo(store, (1, 0), "take").get("rows", make) == 1
    assert not store and len(made) == 1
    # A take that misses makes the value again and does not keep it.
    assert SegmentMemo(store, (1, 0), "take").get("rows", make) == 2
    assert not store
