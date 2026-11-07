"""Memory scheduling around the qualified numerical engine.

Cache/admission controls extend the resident engine without changing layer math.
Policy factor residency and KV snapshot residency have separate lifecycles.
Runtime source hashes bind checkpoint identity; memory settings are reported in
capabilities. Cache
contents are disposable and never part of adapter/optimizer checkpoints.
"""

import hashlib
import math
from pathlib import Path

from .cache import PrefixCache
from .engine import ResidentEngine


def runtime_sources(root=None):
    root = Path(root) if root is not None else Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "runtime.py",
            "cache.py",
            "service.py",
            "cli.py",
            "storage.py",
            "protocol.py",
            "http.py",
            "inference_metrics.py",
            "activation_budget.py",
            "inference.py",
            "mlx_racing.py",
            "mlx_decode.py",
            "mlx_request_cache.py",
        )
    }


class RuntimeEngine(ResidentEngine):
    def __init__(self, *args, prefix_cache_gib=None, inference_concurrency=8, **kwargs):
        if type(inference_concurrency) is not int or inference_concurrency < 1:
            raise ValueError("Inference concurrency must be a positive integer.")
        self.inference_concurrency = inference_concurrency
        if prefix_cache_gib is not None and (
            not math.isfinite(prefix_cache_gib) or prefix_cache_gib < 0
        ):
            raise ValueError("Prefix cache cap must be finite and nonnegative.")
        super().__init__(*args, prefix_cache_gib=0, **kwargs)
        limit = (
            self.resident.budget
            if prefix_cache_gib is None
            else min(self.resident.budget, int(prefix_cache_gib * 1024**3))
        )
        self.prefixes = PrefixCache(self.mx, limit, capacity=self._cache_capacity)
        sources = runtime_sources()
        self.identity = {**self.identity, "runtime_source_sha256": sources}
        self.capabilities = {
            **self.capabilities,
            "max_inference_batch": inference_concurrency,
            "continuous_batching": True,
            "runtime": {
                "prefix_cache_mode": "available_memory"
                if prefix_cache_gib is None
                else "capped",
                "prefix_cache_limit_bytes": limit,
                "source_sha256": sources,
            },
        }

    def _cache_capacity(self, snapshot_bytes=0, *, additional=0):
        retained = self.prefixes.bytes
        noncache = self.persistent_bytes() - retained
        budget = self.admission_budget_bytes()
        return max(
            0,
            min(
                self.prefixes.limit,
                budget - noncache - sum(self.reserved.values()) - additional,
                retained
                + budget
                - self.mx.get_active_memory()
                - snapshot_bytes
                - additional,
            ),
        )

    def admit(self, additional):
        # Evict only detached inference snapshots. Live replay carries and
        # borrowed working caches remain owned by their operations.
        self.prefixes.trim(self._cache_capacity(additional=additional))
        super().admit(additional)

    def _sample(self, body, started):
        maximum = (
            len(body["target_token_ids"])
            if body.get("target_token_ids")
            else body["max_tokens"]
        )
        precision = body.get("cache_precision", "bf16")
        peak = self.resident.estimate_inference(
            len(body["prompt_token_ids"]) + maximum, cache_precision=precision
        )
        if body.get("use_prefix_cache", True):
            peak += min(
                self.prefixes.limit,
                self.resident.estimate_inference(
                    len(body["prompt_token_ids"]), cache_precision=precision
                ),
            )
        self.admit(peak)
        reservation = ("inference", body.get("operation_id", id(body)))
        self.reserved[reservation] = peak
        try:
            return (yield from self._sample_units(body, started))
        finally:
            self.reserved.pop(reservation, None)

    def inference_batch_capacity(self, body, requested):
        noncache = self.persistent_bytes() - self.prefixes.bytes
        active = self.mx.get_active_memory() - self.prefixes.bytes
        reserved = sum(self.reserved.values())
        length = len(body["prompt_token_ids"]) + body["max_tokens"]
        budget = self.admission_budget_bytes()
        for count in range(
            min(getattr(self, "inference_concurrency", requested), requested), 1, -1
        ):
            peak = self.resident.estimate_inference(
                length, count, cache_precision=body.get("cache_precision", "bf16")
            )
            if max(active + peak, noncache + reserved + peak) <= budget:
                return count
        return 1

    def inference_scheduler(self, *, create, finish, fail):
        from . import inference
        from .mlx_racing import ContinuousBatcher

        return ContinuousBatcher(
            self.inference_concurrency,
            create=create,
            prefill=lambda request: inference.prefill_request(self, request),
            advance=lambda requests: inference.advance_requests(self, requests),
            sample=lambda requests: inference.sample_requests(self, requests),
            finish=finish,
            fail=fail,
            release=lambda request: inference.release_request(self, request),
        )

    def metrics(self, started):
        return {
            **super().metrics(started),
            "prefix_cache": {
                **self.prefixes.metrics(),
                "available_capacity_bytes": self._cache_capacity(),
            },
            "runtime": self.capabilities["runtime"],
        }
