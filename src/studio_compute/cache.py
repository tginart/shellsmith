"""Bounded upstream inference cache snapshots, with no training graph ownership."""

from collections import OrderedDict


class PrefixCache:
    def __init__(self, mx, limit_bytes, *, capacity=None):
        self.mx, self.limit = mx, limit_bytes
        self.capacity = capacity
        self.entries, self.bytes = OrderedDict(), 0
        self.hits = self.misses = self.hit_tokens = 0
        self.evictions = self.stored = self.skipped = 0

    def _copy(self, cache):
        mx = self.mx

        def clone(value):
            if isinstance(value, mx.array):
                # Materialize an independent array owner; later upstream in-place
                # cache assignments must never mutate a published snapshot.
                return mx.stop_gradient(value + mx.zeros_like(value))
            if isinstance(value, (list, tuple)):
                return type(value)(clone(x) for x in value)
            return value

        result = [
            type(item).from_state(clone(item.state), item.meta_state) for item in cache
        ]
        mx.eval([item.state for item in result])
        return result

    def size(self, cache):
        def count(value):
            if isinstance(value, self.mx.array):
                return value.nbytes
            return (
                sum(count(x) for x in value) if isinstance(value, (tuple, list)) else 0
            )

        return sum(count(item.state) for item in cache)

    def _match(self, identity, policy, precision, tokens):
        best = None
        for key in self.entries:
            if (
                key[:3] == (identity, policy, precision)
                and len(key[3]) <= len(tokens)
                and tuple(tokens[: len(key[3])]) == key[3]
            ):
                if best is None or len(key[3]) > len(best[3]):
                    best = key
        return best

    def contains(self, identity, policy, precision, tokens):
        return self._match(identity, policy, precision, tokens) is not None

    def get(self, identity, policy, precision, tokens, *, fork=False):
        best = self._match(identity, policy, precision, tokens)
        if best is None:
            self.misses += 1
            return None
        self.hits += 1
        self.hit_tokens += len(best[3])
        cache, logits, _ = self.entries[best]
        self.entries.move_to_end(best)
        if fork:
            from .mlx_request_cache import fork_cache
            result = [fork_cache(item) for item in cache]
            self.mx.eval([item.state for item in result])
        else:
            result = self._copy(cache)
        return result, logits, len(best[3])

    def put(self, identity, policy, precision, tokens, cache, logits):
        size = self.size(cache) + logits.nbytes
        # Capacity includes the temporary detached copy's actual allocation.
        # Active inference and differentiable replay always take priority.
        capacity = self.limit if self.capacity is None else self.capacity(size)
        capacity = max(0, min(self.limit, capacity))
        if size > capacity:
            self.skipped += 1
            return
        key = (identity, policy, precision, tuple(tokens))
        old = self.entries.pop(key, None)
        if old:
            self.bytes -= old[2]
        while self.bytes + size > capacity and self.entries:
            _, (_, _, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1
        self.entries[key] = (self._copy(cache), self.mx.stop_gradient(logits), size)
        self.bytes += size
        self.stored += 1

    def trim(self, limit_bytes):
        target = max(0, min(self.limit, limit_bytes))
        while self.bytes > target and self.entries:
            _, (_, _, removed) = self.entries.popitem(last=False)
            self.bytes -= removed
            self.evictions += 1

    def metrics(self):
        return {
            "bytes": self.bytes, "limit_bytes": self.limit,
            "entries": len(self.entries), "hits": self.hits,
            "misses": self.misses, "hit_tokens": self.hit_tokens,
            "evictions": self.evictions, "stored": self.stored,
            "skipped": self.skipped,
        }

    def clear(self):
        self.entries.clear()
        self.bytes = 0
