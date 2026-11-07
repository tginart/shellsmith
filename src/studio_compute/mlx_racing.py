"""
Purpose: Keep a persistent inference queue with token-boundary admission and retirement.
Input: Request callbacks and a positive concurrency limit.
Output: Materialized progress and terminal callbacks for each request.
Note: Adapted from High Command mlx_racing.py at 1f51929bf55c.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecodeRequest:
    """
    Purpose: Own one request's cache, position, RNG, and output record.
    Input: A pinned policy and private request resources. Output: Mutable scheduler state.
    """

    request_id: str
    body: dict
    cache: list
    factors: Any
    lease: Any
    rng: Any
    started: float
    offset: int = 0
    cached_tokens: int = 0
    logits: Any = None
    needs_decode: bool = False
    batch_size: int = 1
    prefill_seconds: float = 0.0
    decode_started: float = 0.0
    tokens: list = field(default_factory=list)
    logprobs: list = field(default_factory=list)
    reason: str | None = None


class ContinuousBatcher:
    """
    Purpose: Refill free slots without a finite batch barrier.
    Input: Backend callbacks. A create callback returns None on temporary pressure.
    Output: One bounded prefill chunk and one decode boundary per step.
    Note: This object and all callbacks execute on the GPU owner thread.
    """

    def __init__(
        self,
        concurrency: int,
        *,
        create: Any,
        prefill: Any,
        advance: Any,
        sample: Any,
        finish: Any,
        fail: Any,
        release: Any,
    ) -> None:
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Inference concurrency must be a positive integer.")
        self.concurrency = concurrency
        self.create, self.prefill = create, prefill
        self.advance, self.sample = advance, sample
        self.finish, self.fail, self.release = finish, fail, release
        self.pending, self.active = deque(), {}
        self.prefill_cursor = 0

    def enqueue(self, request_id: str, body: dict) -> None:
        self.pending.append((request_id, body))

    def cancel(self, request_id: str) -> None:
        self.pending = deque(item for item in self.pending if item[0] != request_id)
        request = self.active.pop(request_id, None)
        if request is not None:
            self.release(request)

    def step(self) -> dict:
        """
        Purpose: Admit, prefill, decode, and retire at a materialized token boundary.
        Input: Pending and active requests. Output: Request IDs and cumulative counts.
        Note: A prefill error affects one request. A shared kernel error affects its rows.
        """
        while self.pending and len(self.active) < self.concurrency:
            request_id, body = self.pending[0]
            try:
                request = self.create(request_id, body)
            except Exception as error:  # noqa: BLE001 -- isolate request failures
                self.pending.popleft()
                self.fail(request_id, error)
                continue
            if request is None:
                break
            self.pending.popleft()
            self.active[request_id] = request

        prefills = [r for r in self.active.values() if r.logits is None]
        prefilled = None
        if prefills:
            request = prefills[self.prefill_cursor % len(prefills)]
            self.prefill_cursor += 1
            try:
                self.prefill(request)
                prefilled = (request.request_id, request.offset)
            except Exception as error:  # noqa: BLE001 -- isolate prefill failures
                self._retire(request, error=error)

        groups = {}
        for request in self.active.values():
            if request.logits is not None:
                groups.setdefault(request.body["policy"], []).append(request)
        ids, counts, ended, sizes, decode_sizes = [], [], [], [], []
        for requests in groups.values():
            try:
                advancing = [r for r in requests if r.needs_decode]
                if advancing:
                    self.advance(advancing)
                    decode_sizes.append(len(advancing))
                self.sample(requests)
            except Exception as error:  # noqa: BLE001 -- release every affected row
                for request in requests:
                    self._retire(request, error=error)
                continue
            sizes.append(sum(not r.body.get("target_token_ids") for r in requests))
            for request in requests:
                ids.append(request.request_id)
                counts.append(len(request.tokens))
                ended.append(request.reason is not None)
                if request.reason is not None:
                    self._retire(request)
        return {
            "phase": "continuous_decode",
            "request_ids": ids,
            "generated_tokens": counts,
            "finished_requests": ended,
            "batch_sizes": sizes,
            "decode_batch_sizes": decode_sizes,
            "prefill": prefilled,
        }

    def close(self) -> None:
        for request in list(self.active.values()):
            self.cancel(request.request_id)
        self.pending.clear()

    def _retire(self, request: DecodeRequest, *, error=None) -> None:
        self.active.pop(request.request_id)
        try:
            if error is None:
                self.finish(request)
            else:
                self.fail(request.request_id, error)
        finally:
            self.release(request)
