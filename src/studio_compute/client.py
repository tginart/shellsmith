"""Portable RPC client. MLX and Studio filesystem paths never cross this module."""

from __future__ import annotations
import asyncio
import json
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen
import uuid

from .protocol import Conflict, encoded_request, identifier, update_body


class OperationError(RuntimeError):
    def __init__(self, operation):
        self.operation = operation
        super().__init__(f"{operation['status']}: {operation['error']}")


class Client:
    def __init__(
        self, endpoint="http://127.0.0.1:8766", client_id="default", timeout=43200,
        sampling_timeout=None,
    ):
        identifier(client_id)
        self.sampling_timeout = sampling_timeout
        self.endpoint, self.client_id, self.timeout = (
            endpoint.rstrip("/"),
            client_id,
            timeout,
        )

    def _request(self, path, body=None, *, timeout_seconds=45):
        request = Request(
            self.endpoint + path,
            data=None if body is None else encoded_request(body),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                return json.load(response)
        except HTTPError as error:
            value = json.load(error)
            raise (Conflict if error.code == 409 else ValueError)(
                value.get("error", str(error))
            ) from error

    def capabilities(self):
        return self._request("/v1/capabilities")

    def client_info(self):
        return self._request("/v1/clients/" + quote(self.client_id, safe=""))

    def policy_info(self, reference):
        return self._request("/v1/policies/" + quote(reference, safe=""))

    def metrics(self, *, client_id: str | None = None, timeout_seconds: float = 2) -> dict:
        """Read CPU counters with a deadline independent of training RPCs."""
        path = "/v1/metrics"
        if client_id is not None:
            path += "?client=" + quote(identifier(client_id), safe="")
        return self._request(path, timeout_seconds=timeout_seconds)

    def tokenizer_bundle(self):
        return self._request("/v1/tokenizer")

    def validate_submission(self, kind, body, *, request_id):
        identifier(request_id)
        envelope = {"id": request_id, "client": self.client_id, "kind": kind, "body": body}
        return len(encoded_request(envelope))

    def validate_update(self, sequences, *, learning_rate, clip_low, clip_high, request_id):
        self.validate_submission(
            "forward_backward", update_body(sequences, clip_low, clip_high),
            request_id=request_id + ":backward",
        )
        self.validate_submission(
            "optim_step",
            {"learning_rate": learning_rate, "backward_operation": request_id + ":backward"},
            request_id=request_id + ":optimizer",
        )

    def submit(self, kind, body, *, request_id):
        identifier(request_id)
        return self._request(
            "/v1/operations",
            {"id": request_id, "client": self.client_id, "kind": kind, "body": body},
        )

    def operation(self, request_id, *, wait=0):
        return self._request(
            "/v1/operations/" + quote(request_id, safe="") + f"?wait={wait}"
        )

    def cancel(self, request_id: str) -> dict:
        return self._request(
            "/v1/operations/" + quote(identifier(request_id), safe="") + "/cancel", {}
        )

    def call(self, kind, body, *, request_id=None):
        request_id = request_id or uuid.uuid4().hex
        op = self.submit(kind, body, request_id=request_id)
        deadline = self.sampling_timeout if kind in ("sample", "logprobs") else None
        until = time.monotonic() + (self.timeout if deadline is None else deadline)
        while op["status"] in ("queued", "running"):
            if time.monotonic() >= until:
                raise TimeoutError(
                    f"Operation {request_id} remains {op['status']}; poll or retry the same durable ID."
                )
            op = self.operation(request_id, wait=30)
        if op["status"] != "succeeded":
            raise OperationError(op)
        return op["result"]

    async def acall(self, kind, body, *, request_id=None):
        return await asyncio.to_thread(self.call, kind, body, request_id=request_id)

    async def update(
        self, sequences, *, learning_rate, clip_low=0.0, clip_high=4.0, request_id,
        on_phase=None,
    ):
        await asyncio.to_thread(
            self.validate_update, sequences, learning_rate=learning_rate,
            clip_low=clip_low, clip_high=clip_high, request_id=request_id,
        )
        # Retry a completed update without depending on volatile backward state.
        # Submission still verifies that the complete original body matches.
        try:
            committed = await asyncio.to_thread(
                self.operation, request_id + ":optimizer"
            )
        except ValueError:
            committed = None
        if committed and committed["status"] == "succeeded":
            forward = await self.acall(
                "forward_backward",
                update_body(sequences, clip_low, clip_high),
                request_id=request_id + ":backward",
            )
            optimizer = await self.acall(
                "optim_step",
                {
                    "learning_rate": learning_rate,
                    "backward_operation": request_id + ":backward",
                },
                request_id=request_id + ":optimizer",
            )
            return combine_update_receipts(forward, optimizer)
        if on_phase is not None:
            on_phase("forward_backward")
        forward = await self.acall(
            "forward_backward",
            update_body(sequences, clip_low, clip_high),
            request_id=request_id + ":backward",
        )
        if on_phase is not None:
            on_phase("optimizer")
        optimizer = await self.acall(
            "optim_step",
            {
                "learning_rate": learning_rate,
                "backward_operation": request_id + ":backward",
            },
            request_id=request_id + ":optimizer",
        )
        return combine_update_receipts(forward, optimizer)


def combine_update_receipts(forward: dict, optimizer: dict) -> dict:
    """
    Purpose: Keep each operation measurement when an update has two receipts.
    Input: The backward receipt and the optimizer receipt.
    Output: The combined receipt with distinct operation measurements.
    Note: Legacy top-level fields retain the optimizer values.
    """
    return {
        **forward, **optimizer,
        "backward_execution_metrics": forward.get("execution_metrics", {}),
        "optimizer_execution_metrics": optimizer.get("execution_metrics", {}),
    }
