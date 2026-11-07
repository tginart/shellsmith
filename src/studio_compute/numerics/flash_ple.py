"""
Purpose: Replace the sharded Flash n-gram table with one quantized device table.
Input: The loaded ShardedEmbedding of the Flash PLE layer.
Output: A module with the same lookup values and no host synchronization.
Note: The upstream lookup reads every index on the host and gathers once per shard.
    One joined table keeps the same quantized rows and the same dequantization.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn


class JoinedNGramTable(nn.Module):
    """
    Purpose: Gather quantized n-gram rows from one table on the device.
    Input: A ShardedEmbedding whose shards are affine QuantizedEmbedding modules.
    Output: BF16 rows identical to the sharded lookup for every valid index.
    """

    def __init__(self, sharded: Any):
        super().__init__()
        shards = sharded.shards
        first = shards[0]
        assert all(
            isinstance(shard, nn.QuantizedEmbedding)
            and (shard.group_size, shard.bits, shard.mode)
            == (first.group_size, first.bits, first.mode)
            for shard in shards
        ), "Every n-gram shard must share one affine quantization."
        assert tuple(shard.weight.shape[0] for shard in shards) == sharded.shard_sizes
        self.group_size, self.bits, self.mode = first.group_size, first.bits, first.mode
        self.dims, self.rows = sharded.dims, sharded.shard_offsets[-1]
        self.weight = mx.concatenate([shard.weight for shard in shards], axis=0)
        self.scales = mx.concatenate([shard.scales for shard in shards], axis=0)
        self.biases = mx.concatenate([shard.biases for shard in shards], axis=0)
        self.freeze()

    def __call__(self, indices: Any) -> Any:
        flat = indices.reshape(-1)
        rows = mx.dequantize(
            self.weight[flat], scales=self.scales[flat], biases=self.biases[flat],
            group_size=self.group_size, bits=self.bits, mode=self.mode,
        )
        return rows.reshape(*indices.shape, self.dims)


def join_ngram_tables(language_model: Any) -> int:
    """
    Purpose: Install joined n-gram tables in every PLE layer of a Flash decoder.
    Input: The Flash language model after loading and before parameter inventory.
    Output: The number of replaced tables. Shard arrays are released afterwards.
    """
    replaced = 0
    for layer in language_model.layers:
        if "ple" not in layer:
            continue
        embedding = layer.ple.ple_embedding
        joined = JoinedNGramTable(embedding.ngram_embedding)
        mx.eval(joined.parameters())
        embedding.ngram_embedding = joined
        replaced += 1
    return replaced
