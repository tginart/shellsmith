"""
Purpose: Check that the joined Flash n-gram table returns the sharded rows bitwise.
Input: A small upstream ShardedEmbedding with affine 4-bit quantized shards.
Output: Exact comparisons at decode and segment index shapes.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
language = pytest.importorskip("mlx_vlm.models.qwen4_exp.language")

from studio_compute.numerics.flash_ple import JoinedNGramTable


@pytest.mark.parametrize("shape", [(1, 1, 8), (8, 1, 8), (1, 64, 8)])
def test_joined_table_matches_sharded_lookup_bitwise(shape: tuple) -> None:
    mx.random.seed(5)
    sharded = language.ShardedEmbedding(1003, 64, 4)
    nn.quantize(sharded, group_size=64, bits=4)
    mx.eval(sharded.parameters())
    joined = JoinedNGramTable(sharded)
    indices = mx.random.randint(0, 1003, shape).astype(mx.int64)
    # Cover the first and last row of every shard.
    edges = mx.array([[offset, offset - 1] for offset in sharded.shard_offsets[1:-1]]
                     + [[0, 1002]])
    for values in (indices, edges.astype(mx.int64)):
        assert bool(mx.array_equal(sharded(values), joined(values)).item())
