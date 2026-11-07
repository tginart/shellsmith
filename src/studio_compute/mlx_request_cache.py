"""
Purpose: Share immutable KV prefixes and keep private request state.
Input: Native Qwen attention and recurrent caches. Output: Independent cache writers.
Note: Adapted from High Command mlx_request_cache.py at 1f51929bf55c.
"""

from copy import copy

import mlx.core as mx
from mlx_vlm.models.cache import ArraysCache, KVCache, QuantizedKVCache

# B: batch, H: heads, L: tokens, D: head width.


def fork_cache(source: object) -> object:
    """
    Purpose: Fork a snapshot without a shared mutable KV capacity buffer.
    Input: One supported native cache. Output: A private cache with the same position.
    Note: Packed quantized buffers require a copy. BF16 prefixes remain shared.
    """
    if isinstance(source, ArraysCache):
        result = copy(source)
        result.cache = [_clone(value) for value in source.cache]
        for name in ("left_padding", "lengths"):
            value = getattr(source, name, None)
            if value is not None:
                setattr(result, name, _clone(value))
        return result
    if isinstance(source, QuantizedKVCache):
        return type(source).from_state(_clone(source.state), source.meta_state)
    if isinstance(source, KVCache) and hasattr(source, "index_keys"):
        from .numerics.flash_ops import fork_qsa_cache
        return fork_qsa_cache(source)
    if not isinstance(source, KVCache):
        raise TypeError("Unsupported MLX request cache: " + type(source).__name__)
    result = OwnedKV()
    result.prefix_keys, result.prefix_values = source.state
    result.offset = source.offset
    return result


def state_arrays(cache: object) -> list:
    if isinstance(cache, OwnedKV):
        return [x for x in (cache.suffix_keys, cache.suffix_values) if x is not None]
    return cache.state


class OwnedKV(KVCache):
    """
    Purpose: Append private KV suffixes without writes into a shared prefix.
    Input: Native attention updates. Output: A complete KV view for one request.
    """

    def __init__(self):
        super().__init__()
        self.prefix_keys = self.prefix_values = None
        self.suffix_keys = self.suffix_values = None

    def update_and_fetch(self, keys_BHLD, values_BHLD):
        self.suffix_keys = _append(self.suffix_keys, keys_BHLD)
        self.suffix_values = _append(self.suffix_values, values_BHLD)
        self.offset += keys_BHLD.shape[2]
        return self.state

    @property
    def state(self):
        return (
            _append(self.prefix_keys, self.suffix_keys),
            _append(self.prefix_values, self.suffix_values),
        )

    @state.setter
    def state(self, values):
        self.prefix_keys, self.prefix_values = values
        self.suffix_keys = self.suffix_values = None
        self.offset = 0 if self.prefix_keys is None else self.prefix_keys.shape[2]

    @property
    def nbytes(self):
        return sum(
            value.nbytes
            for value in (
                self.prefix_keys,
                self.prefix_values,
                self.suffix_keys,
                self.suffix_values,
            )
            if value is not None
        )

    def empty(self) -> bool:
        return self.offset == 0


def _append(first: object, second: object) -> object:
    if first is None:
        return second
    if second is None:
        return first
    return mx.concatenate((first, second), axis=2)


def _clone(value: object) -> object:
    if isinstance(value, mx.array):
        return value + mx.zeros_like(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(item) for item in value)
    return value
