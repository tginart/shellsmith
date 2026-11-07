"""OpenAI-compatible Player endpoint backed by Tinker, recording every sample.

Architecture map
----------------
    Pi (one per episode)  --base_url=http://<host>:<port>/e/<episode_id>/v1-->  PlayerEndpoint
        GET  /e/<id>/v1/models            -> the one served model id
        POST /e/<id>/v1/chat/completions  -> TinkerPlayerSampler.sample
                                             -> records[<id>].append(record)
                                             -> <records_dir>/<id>.jsonl gets record.protocol_row
                                             -> JSON body, or SSE when "stream": true
    trainer / harvest  --take_records(<id>)-->  the episode's records, in request order

Invariants
----------
- The episode id lives in the URL path, so concurrent episodes sharing one
  endpoint never mix records (no header Pi would have to send).
- Every request needs ``Authorization: Bearer <api_key>``; the key is created
  per endpoint and handed to Pi through an environment variable.
- A record is stored before its response is written. If Pi drops the
  connection the record stays; ``policy.harvest`` keeps only records Pi's vat
  log shows it consumed.

Why this exists
---------------
Replaces the original Verifiers intercept and its 6k-line shared LLM proxy:
captures come straight from the endpoint that sampled them.

Main callers:
    ``shellsmith episode --player tinker`` and (M4) the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any

from shellsmith.policy.records import TinkerRolloutRecord
from shellsmith.policy.sampler import PlayerRequestError, TinkerPlayerSampler
from shellsmith.train.backend import PolicySampler
from shellsmith.process_io import fsync_directory

PATH_PATTERN = re.compile(r"^/e/(?P<episode>[A-Za-z0-9._-]+)/v1/(?P<route>models|chat/completions)$")
SERVED_MODEL_ID = "shellsmith-player"


def sse_body(request_id: str, model: str, assistant: dict[str, Any], finish_reason: str,
             prompt_tokens: int, completion_tokens: int) -> bytes:
    """Render one assistant message as an OpenAI ``chat.completion.chunk`` stream.

    Pi accumulates ``delta.reasoning_content`` into a ThinkingPart, so the GM
    can see and score the Player's thinking. The trailing usage chunk matters:
    Pi tracks context growth from ``usage.prompt_tokens`` to decide when to
    compact.
    """
    created = int(time.time())

    def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

    chunks = [chunk({"role": "assistant"})]
    if assistant.get("reasoning_content"):
        chunks.append(chunk({"reasoning_content": assistant["reasoning_content"]}))
    if assistant.get("content"):
        chunks.append(chunk({"content": assistant["content"]}))
    for index, call in enumerate(assistant.get("tool_calls") or []):
        function = call["function"]
        chunks.append(chunk({"tool_calls": [{"index": index, "id": call["id"], "type": "function",
                                             "function": {"name": function["name"], "arguments": ""}}]}))
        if function.get("arguments"):
            chunks.append(chunk({"tool_calls": [{"index": index, "function": {"arguments": function["arguments"]}}]}))
    chunks.append(chunk({}, finish_reason))
    chunks.append({"id": request_id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                            "total_tokens": prompt_tokens + completion_tokens}})
    return b"".join(f"data: {json.dumps(item)}\n\n".encode() for item in chunks) + b"data: [DONE]\n\n"


@dataclass
class _EpisodeRequests:
    active: int = 0
    closed: bool = False


@dataclass
class PlayerEndpoint:
    """A running Player endpoint. Use ``start()``/``stop()`` or a ``with`` block."""

    sampler: PolicySampler
    records_dir: Path
    host: str = "127.0.0.1"
    port: int = 0
    api_key: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    _records: dict[str, list[TinkerRolloutRecord]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _server: ThreadingHTTPServer | None = None
    _counter: int = 0
    _managed_episodes: bool = False
    _episodes: dict[str, _EpisodeRequests] = field(default_factory=dict)

    def open_episode(self, episode_id: str) -> None:
        """Register a runner-owned episode; unregistered HTTP requests then fail closed."""
        with self._lock:
            if episode_id in self._episodes:
                raise RuntimeError("Episode already has an owner or unfinished HTTP requests.")
            self._managed_episodes = True
            self._episodes[episode_id] = _EpisodeRequests()

    def _release_episode(self, episode_id: str) -> None:
        if hasattr(self.sampler, "release"):
            self.sampler.release(episode_id)

    def close_episode(self, episode_id: str) -> None:
        """Fence new work and drain memory; active handlers retain ownership until exit."""
        with self._lock:
            state = self._episodes.get(episode_id)
            if state is not None:
                state.closed = True
            self._records.pop(episode_id, None)
            if state is None or state.active == 0:
                self._release_episode(episode_id)
                self._episodes.pop(episode_id, None)

    def _begin_episode_request(self, episode_id: str):
        with self._lock:
            if not self._managed_episodes:
                return None  # Standalone CLI endpoints retain ad-hoc routing.
            state = self._episodes.get(episode_id)
            if state is None or state.closed:
                raise PlayerRequestError("Episode is closed or unregistered.", status=410)
            state.active += 1
            return state

    def _request_open(self, state) -> bool:
        with self._lock:
            return state is None or not state.closed

    def _end_episode_request(self, episode_id: str, state) -> None:
        if state is None:
            return
        with self._lock:
            state.active -= 1
            if state.closed and state.active == 0:
                self._release_episode(episode_id)
                self._episodes.pop(episode_id, None)

    def base_url(self, episode_id: str, *, host: str | None = None) -> str:
        """The OpenAI base URL Pi uses for one episode (``host`` overrides for Docker)."""
        if self._server is None:
            raise RuntimeError("PlayerEndpoint is not started.")
        return f"http://{host or self.host}:{self._server.server_address[1]}/e/{episode_id}/v1"

    def take_records(self, episode_id: str) -> list[TinkerRolloutRecord]:
        """Remove and return one episode's records in request order."""
        with self._lock:
            if not self._managed_episodes:
                self._release_episode(episode_id)
            return self._records.pop(episode_id, [])

    def _store(self, episode_id: str, record: TinkerRolloutRecord) -> bool:
        if getattr(record, "backend", "tinker") in {"mlx","studio"}:
            row = {**record.protocol_row, "capture_version": 2, "backend": record.backend,
                   "policy_ref": record.policy_ref, "policy_identity": record.policy_identity,
                   "session_id": record.session_id, "created_at": record.created_at,
                   "payload": record.payload, "finish_reason": record.finish_reason,
                   "prompt_token_ids": record.prompt.to_ints(), "prompt_tokens": record.prompt_tokens,
                   "sampled_tokens": list(record.sampled_tokens), "sampled_logprobs": list(record.sampled_logprobs),
                   "forced_close_token_span": record.forced_close_token_span, "request_id": record.request_id,
                   "diagnostic_generation_attempts": list(record.diagnostic_generation_attempts)}
            record = replace(record, protocol_row=row)
        with self._lock:
            state = self._episodes.get(episode_id)
            closed = self._managed_episodes and (state is None or state.closed)
            if closed:
                record = replace(record, protocol_row={**record.protocol_row, "discarded_after_episode_close": True})
            else:
                self._records.setdefault(episode_id, []).append(record)
            self.records_dir.mkdir(parents=True, exist_ok=True)
            with (self.records_dir / f"{episode_id}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.protocol_row, sort_keys=True) + "\n")
                if getattr(record, "backend", "tinker") in {"mlx","studio"}:
                    handle.flush()
                    os.fsync(handle.fileno())
            if record.protocol_row.get("protocol_error"):
                with (self.records_dir / "protocol_failures.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record.protocol_row, sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            if getattr(record, "backend", "tinker") in {"mlx","studio"}:
                fsync_directory(self.records_dir)
            return not closed

    def _next_request_id(self, episode_id: str) -> str:
        with self._lock:
            self._counter += 1
            return f"{episode_id}-req-{self._counter:06d}"

    def start(self) -> "PlayerEndpoint":
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
                return

            def _send(self, status: int, body: bytes, content_type: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _error(self, status: int, message: str) -> None:
                self._send(status, json.dumps({"error": {"message": message}}).encode())

            def _route(self) -> tuple[str, str] | None:
                if self.headers.get("Authorization", "") != f"Bearer {endpoint.api_key}":
                    self._error(401, "Missing or wrong bearer token.")
                    return None
                match = PATH_PATTERN.match(self.path.split("?", 1)[0])
                if match is None:
                    self._error(404, f"Unknown path {self.path}.")
                    return None
                return match["episode"], match["route"]

            def do_GET(self) -> None:  # noqa: N802 - stdlib naming
                routed = self._route()
                if routed is None:
                    return
                if routed[1] != "models":
                    self._error(405, "Use POST for chat/completions.")
                    return
                self._send(200, json.dumps({"object": "list", "data": [{"id": SERVED_MODEL_ID, "object": "model"}]}).encode())

            def do_POST(self) -> None:  # noqa: N802 - stdlib naming
                routed = self._route()
                if routed is None:
                    return
                episode_id, route = routed
                if route != "chat/completions":
                    self._error(405, "Use GET for models.")
                    return
                try:
                    episode_state = endpoint._begin_episode_request(episode_id)
                except PlayerRequestError as error:
                    self.close_connection = True  # The rejected request body is unread.
                    self._error(error.status, str(error))
                    return
                try:
                    try:
                        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                    except json.JSONDecodeError as error:
                        self._error(400, f"Request body is not JSON: {error}")
                        return
                    if not endpoint._request_open(episode_state):
                        self._error(410, "Episode closed before sampling.")
                        return
                    request_id = endpoint._next_request_id(episode_id)
                    try:
                        record, assistant = endpoint.sampler.sample(payload, request_id=request_id, session_id=episode_id)
                    except PlayerRequestError as error:
                        if error.record is not None:
                            if not endpoint._store(episode_id, error.record):
                                self._error(410, "Episode closed during sampling.")
                                return
                        self._error(error.status, str(error))
                        return
                    if not endpoint._store(episode_id, record):
                        self._error(410, "Episode closed during sampling.")
                        return
                    finish = "tool_calls" if assistant.get("tool_calls") else record.finish_reason
                    usage = (record.prompt_tokens, len(record.sampled_tokens))
                    if payload.get("stream"):
                        self._send(200, sse_body(request_id, SERVED_MODEL_ID, assistant, finish, *usage), "text/event-stream")
                        return
                    body = {"id": request_id, "object": "chat.completion", "created": int(time.time()), "model": SERVED_MODEL_ID,
                            "choices": [{"index": 0, "message": assistant, "finish_reason": finish}],
                            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}}
                    self._send(200, json.dumps(body).encode())
                finally:
                    endpoint._end_episode_request(episode_id, episode_state)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=self._server.serve_forever, name="shellsmith-player-endpoint", daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> "PlayerEndpoint":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
