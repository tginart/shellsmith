"""
Purpose: Preserve BF16 and packed 8-bit inference caches in single and batched decode.
Input: Equal-length native requests and the BF16 decoder implementation.
Output: Batch equivalence, cache offsets, storage precision, and BF16 carry checks.
"""

import pytest

mx = pytest.importorskip("mlx.core")
from studio_compute.numerics.bf16 import decode_hidden
from test_saved_layers import tiny_saved_model


@pytest.mark.parametrize("precision", ["bf16", "8bit"])
def test_bf16_decode_preserves_packed_cache_and_batch_equivalence(precision):
    lm, adapter = tiny_saved_model()
    lm.eval()

    def decode(ids, next_ids):
        caches = lm.make_cache()
        if precision == "8bit":
            caches = [
                value.to_quantized(bits=8, group_size=64)
                if hasattr(value, "to_quantized")
                else value
                for value in caches
            ]
        batch, length = ids.shape
        positions = mx.broadcast_to(
            mx.arange(length, dtype=mx.int32)[None, None], (3, batch, length)
        )
        hidden = decode_hidden(lm, ids, positions, caches)
        mx.eval(hidden, [cache.state for cache in caches])
        hidden = decode_hidden(
            lm, next_ids, mx.full((3, batch, 1), length, mx.int32), caches
        )
        mx.eval(hidden, [cache.state for cache in caches])
        assert caches[3].offset == length + 1
        assert all(
            value.dtype == mx.bfloat16 for cache in caches[:3] for value in cache.state
        )
        if precision == "8bit":
            assert caches[3].bits == 8
            assert caches[3].state[0][0].dtype == mx.uint32
        return hidden

    ids = mx.array([list(range(17)), list(range(7, 24))], mx.int32)
    next_ids = mx.array([[31], [32]], mx.int32)
    expected = mx.concatenate(
        [decode(ids[i : i + 1], next_ids[i : i + 1]) for i in range(2)], axis=0
    )
    actual = decode(ids, next_ids)
    difference = mx.linalg.norm((actual - expected).astype(mx.float32))
    norm = mx.linalg.norm(expected.astype(mx.float32))
    assert float(difference.item()) / float(norm.item()) < 0.03
