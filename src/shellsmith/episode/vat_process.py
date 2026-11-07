"""Python host for the persistent Node Pi-SDK vat runner.

Architecture map:
    1. Validate that the temporary workspace contains only `AGENTS.md`.
    2. Start one Node child with stdin/stdout/stderr pipes.
    3. Initialize one in-memory Pi session bound to that unique `cwd`.
    4. For each prompt, service every `tool_request` through an injected async
       handler while Pi's JavaScript tool Promise remains suspended.
    5. Accept `prompt_complete` only after all tool calls have resolved.
    6. Re-check the one-file workspace invariant and shut down explicitly.

The subprocess is persistent but the session is not crash-recoverable: Pi keeps
its conversation in memory so it never writes into the vat. A Node crash makes
the episode untrainable and requires a new rollout.

Why this exists:
    `pi --mode rpc` reports tool execution but does not by itself prevent the
    real built-ins from running. This adapter hosts a purpose-built SDK process
    whose tool executors are wire requests rather than filesystem operations.
"""

from __future__ import annotations

import asyncio
import os
import json
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from shellsmith.types import JsonValue
from shellsmith.episode.failure import (
    ExecutionTimeoutError,
)
from shellsmith.episode.vat_protocol import (
    DEFAULT_VIRTUAL_TOOLS,
    VatAgentEvent,
    VatInitialized,
    VatPromptComplete,
    VatRunnerError,
    VatShutdownComplete,
    VatToolRequest,
    VatToolResponse,
    decode_runner_message,
    encode_jsonl,
    initialize_command,
    prompt_command,
    runner_message_kind,
    shutdown_command,
)
from shellsmith.episode.vat_workspace import (
    EphemeralVatWorkspace,
)

# asyncio.StreamReader defaults to 64 KiB per readline(). The host↔vat wire is
# one JSON object per LF-delimited line, and a forced-close thinking dump already
# exceeds that (~66 KiB once JSON-escaped). Raise the pipe limit so long
# assistant/tool messages do not crash the episode with:
#   ValueError: Separator is found, but chunk is longer than limit
# 8 MiB leaves headroom above model_max_tokens text plus JSON wrapping.
VAT_SUBPROCESS_STREAM_LIMIT_BYTES = 8 * 1024 * 1024


class VatToolHandler(Protocol):
    """Resolve suspended Pi tool calls through the simulated world."""

    async def handle_sibling_batch(
        self,
        requests: list[VatToolRequest],
        trajectory_delta: tuple[VatAgentEvent, ...],
    ) -> list[VatToolResponse]:
        """Grade one assistant multi-tool message as a single world turn."""
        ...


@dataclass(frozen=True)
class PiSdkVatConfig:
    """Everything Python needs to launch one isolated Node vat process."""

    node_executable: Path
    runner_script: Path
    episode_id: str
    workspace: EphemeralVatWorkspace
    model_provider: str
    model_id: str
    model_base_url: str
    thinking_level: str
    raw_log_path: Path
    model_api: str = "openai-completions"
    model_context_window: int = 224 * 1024
    model_max_tokens: int = 32_768
    compaction_reserve_tokens: int = 65536
    compaction_keep_recent_tokens: int = 20000
    retry: dict | None = None
    model_reasoning: bool = True
    enabled_tools: tuple[str, ...] = DEFAULT_VIRTUAL_TOOLS
    api_key_env: str | None = None
    # In-memory override is delivered only to the trusted Player subprocess.
    # It never enters the initialize wire log, argv, or process-wide environment.
    api_key: str | None = field(default=None, repr=False)
    pi_sdk_module_path: Path | None = None
    initialize_timeout_seconds: float = 30.0
    prompt_timeout_seconds: float = 900.0
    shutdown_timeout_seconds: float = 10.0


@dataclass(frozen=True)
class VatPromptResult:
    """One fully settled Pi prompt plus normalized private audit events."""

    session_id: str
    final_text: str
    message_count: int
    terminated: bool
    events: tuple[VatAgentEvent, ...]
    raw_log_path: Path


class PiSdkVatProcess:
    """Persistent JSONL subprocess adapter for the Node Pi-SDK runner.

    Main callers:
        `EpisodeOrchestrator` after the Game Master has written and
        boot validation has accepted the vat's `AGENTS.md`.

    Failure semantics:
        Malformed JSON, mismatched IDs, an unexpected message order, child exit,
        a tool-handler failure, timeout, or workspace mutation raises. No method
        manufactures a tool result or final response as a fallback.
    """

    def __init__(self, config: PiSdkVatConfig) -> None:
        self.config = config
        self._process: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._prompt_lock = asyncio.Lock()
        self._stderr_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        if self._session_id is None:
            raise RuntimeError("Pi SDK vat process has not been initialized.")
        return self._session_id

    async def launch(self) -> str:
        """Start Node, construct one Pi session, and return its session ID."""
        if self._process is not None:
            raise RuntimeError("Pi SDK vat process may only be launched once.")
        self.config.workspace.assert_agents_only()
        if not self.config.runner_script.is_file():
            raise RuntimeError(
                f"Pi SDK vat runner does not exist: {self.config.runner_script}."
            )
        environment = dict(os.environ)
        if self.config.api_key is not None:
            if not self.config.api_key_env or not self.config.api_key.strip():
                raise RuntimeError("A private Player credential requires a nonempty key and environment name.")
            environment[self.config.api_key_env] = self.config.api_key
        self.config.raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._process = await asyncio.create_subprocess_exec(
            str(self.config.node_executable),
            str(self.config.runner_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.config.workspace.root),
            env=environment,
            # Applies to stdout/stderr StreamReaders used by readline() in
            # `_receive` / `_drain_stderr`. Default 64 KiB is too small for
            # long thinking or tool-result JSONL lines.
            limit=VAT_SUBPROCESS_STREAM_LIMIT_BYTES,
        )
        self._stderr_task = asyncio.create_task(self._drain_stderr())

        request_id = self._new_request_id("initialize")
        await self._send(
            initialize_command(
                request_id=request_id,
                episode_id=self.config.episode_id,
                workspace_path=str(self.config.workspace.workspace_dir),
                agent_home_path=str(self.config.workspace.agent_home_dir),
                model_provider=self.config.model_provider,
                model_id=self.config.model_id,
                model_base_url=self.config.model_base_url,
                model_api=self.config.model_api,
                model_context_window=self.config.model_context_window,
                model_max_tokens=self.config.model_max_tokens,
                compaction_reserve_tokens=self.config.compaction_reserve_tokens,
                compaction_keep_recent_tokens=self.config.compaction_keep_recent_tokens,
                retry=self.config.retry,
                model_reasoning=self.config.model_reasoning,
                thinking_level=self.config.thinking_level,
                enabled_tools=self.config.enabled_tools,
                api_key_env=self.config.api_key_env,
                pi_sdk_module_path=(
                    str(self.config.pi_sdk_module_path)
                    if self.config.pi_sdk_module_path is not None
                    else None
                ),
            )
        )
        try:
            message = await self._receive(
                timeout_seconds=self.config.initialize_timeout_seconds
            )
        except TimeoutError as error:
            await self._terminate_after_failure()
            raise ExecutionTimeoutError(
                component="player",
                stage="player_initialize",
                timeout_seconds=self.config.initialize_timeout_seconds,
                provider_attributable=False,
            ) from error
        except BaseException:
            await self._terminate_after_failure()
            raise
        if isinstance(message, VatRunnerError):
            await self._terminate_after_failure()
            raise RuntimeError(f"Vat runner initialization failed: {message.message}")
        if not isinstance(message, VatInitialized):
            await self._terminate_after_failure()
            raise RuntimeError(
                "Vat runner emitted "
                f"{runner_message_kind(message)!r} before initialized."
            )
        if message.request_id != request_id:
            await self._terminate_after_failure()
            raise RuntimeError(
                "Vat initialized response request ID mismatch: "
                f"expected {request_id!r}, got {message.request_id!r}."
            )
        if message.enabled_tools != self.config.enabled_tools:
            await self._terminate_after_failure()
            raise RuntimeError(
                "Vat runner enabled an unexpected tool set: "
                f"expected {self.config.enabled_tools}, got {message.enabled_tools}."
            )
        self._session_id = message.session_id
        self.config.workspace.assert_agents_only()
        return message.session_id

    async def prompt(
        self,
        text: str,
        *,
        tool_handler: VatToolHandler,
    ) -> VatPromptResult:
        """Run one Pi prompt while synchronously servicing virtual tool calls.

        Main callers:
            Initial task delivery and any later explicit user/harness message.

        The Pi agent may make many model turns and tool calls before this method
        returns. The lock rejects overlapping prompts because the Game Master
        and its private action ledger are serialized in the prototype.
        """
        if self._process is None or self._session_id is None:
            raise RuntimeError("Pi SDK vat process must be launched before prompting.")
        async with self._prompt_lock:
            self.config.workspace.assert_agents_only()
            request_id = self._new_request_id("prompt")
            await self._send(prompt_command(request_id=request_id, text=text))
            try:
                async with asyncio.timeout(self.config.prompt_timeout_seconds):
                    result = await self._run_prompt_loop(
                        request_id=request_id,
                        tool_handler=tool_handler,
                    )
            except ExecutionTimeoutError:
                # A nested GM deadline already carries the owning stage
                # and provider attribution. Do not relabel it as a vat timeout.
                await self._terminate_after_failure()
                raise
            except TimeoutError as error:
                await self._terminate_after_failure()
                raise ExecutionTimeoutError(
                    component="player",
                    stage="player_prompt",
                    timeout_seconds=self.config.prompt_timeout_seconds,
                    provider_attributable=False,
                ) from error
            except BaseException:
                await self._terminate_after_failure()
                raise
            self.config.workspace.assert_agents_only()
            return result

    async def _run_prompt_loop(
        self,
        *,
        request_id: str,
        tool_handler: VatToolHandler,
    ) -> VatPromptResult:
        """Service complete assistant action batches without partial feedback.

        Pi may execute sibling tool calls concurrently, but the simulated world
        has a serialized state ledger. We therefore wait until every call from
        one assistant message is on the wire, evaluate them in message order,
        and only then release their observations. A terminal sibling marks the
        final response in the batch so Pi cannot abort the remaining calls.

        Main callers:
            ``prompt`` for every Player prompt.

        Why this exists:
            Sending the first sibling observation immediately lets Pi continue
            or abort before the remaining siblings are recorded. That produces
            an incomplete action in the world artifacts and invalid training
            data in the strict Tinker bridge.
        """
        events: list[VatAgentEvent] = []
        events_since_action: list[VatAgentEvent] = []
        shared_context_by_tool_id: dict[str, tuple[VatAgentEvent, ...]] = {}
        batch_ids_by_tool_id: dict[str, tuple[str, ...]] = {}
        pending_requests: dict[str, VatToolRequest] = {}
        handled_tool_ids: set[str] = set()
        while True:
            message = await self._receive(timeout_seconds=None)
            if isinstance(message, VatAgentEvent):
                events.append(message)
                events_since_action.append(message)
                tool_ids = self._assistant_tool_call_ids(message)
                if tool_ids:
                    shared_context = tuple(events_since_action)
                    shared_context_by_tool_id.update(
                        {tool_id: shared_context for tool_id in tool_ids}
                    )
                    batch_ids_by_tool_id.update(
                        {tool_id: tool_ids for tool_id in tool_ids}
                    )
                continue
            if isinstance(message, VatToolRequest):
                if message.tool_call_id in handled_tool_ids:
                    raise RuntimeError(
                        f"Vat runner repeated tool_call_id {message.tool_call_id!r}."
                    )
                handled_tool_ids.add(message.tool_call_id)
                batch_ids = batch_ids_by_tool_id.get(message.tool_call_id)
                if batch_ids is None:
                    raise RuntimeError(
                        "Vat runner emitted a tool request before declaring its "
                        f"assistant action batch: {message.tool_call_id!r}."
                    )
                pending_requests[message.tool_call_id] = message
                if not all(tool_id in pending_requests for tool_id in batch_ids):
                    continue

                responses: list[VatToolResponse] = []
                sibling_requests = [
                    pending_requests.pop(tool_id) for tool_id in batch_ids
                ]
                # One shared pre-action trajectory for the whole sibling set.
                trajectory_delta = shared_context_by_tool_id.pop(batch_ids[0])
                for tool_id in batch_ids[1:]:
                    shared_context_by_tool_id.pop(tool_id, None)
                if hasattr(tool_handler, "handle_sibling_batch"):
                    responses = await tool_handler.handle_sibling_batch(
                        sibling_requests,
                        trajectory_delta,
                    )
                else:
                    # Legacy single-call handlers (unit fakes) still work one-by-one.
                    for sibling_request in sibling_requests:
                        responses.append(
                            await tool_handler(  # type: ignore[operator]
                                sibling_request,
                                trajectory_delta,
                            )
                        )
                if len(responses) != len(batch_ids):
                    raise RuntimeError(
                        "Vat tool handler returned the wrong sibling response count: "
                        f"expected {len(batch_ids)}, got {len(responses)}."
                    )
                for tool_id, response in zip(batch_ids, responses, strict=True):
                    if response.tool_call_id != tool_id:
                        raise RuntimeError(
                            "Vat tool handler response correlation mismatch: "
                            f"expected {tool_id!r}, "
                            f"got {response.tool_call_id!r}."
                        )

                terminate_after_batch = any(
                    response.terminate for response in responses
                )
                for index, response in enumerate(responses):
                    await self._send(
                        replace(
                            response,
                            terminate=(
                                terminate_after_batch
                                and index == len(responses) - 1
                            ),
                        ).to_wire()
                    )
                for tool_id in batch_ids:
                    batch_ids_by_tool_id.pop(tool_id)
                events_since_action.clear()
                continue
            if isinstance(message, VatPromptComplete):
                if message.request_id != request_id:
                    raise RuntimeError(
                        "Vat prompt response request ID mismatch: "
                        f"expected {request_id!r}, got {message.request_id!r}."
                    )
                return VatPromptResult(
                    session_id=self.session_id,
                    final_text=message.final_text,
                    message_count=message.message_count,
                    terminated=message.terminated,
                    events=tuple(events),
                    raw_log_path=self.config.raw_log_path,
                )
            if isinstance(message, VatRunnerError):
                raise RuntimeError(f"Vat runner failed: {message.message}")
            raise RuntimeError(
                f"Unexpected {runner_message_kind(message)!r} during vat prompt."
            )


    @staticmethod
    def _assistant_tool_call_ids(event: VatAgentEvent) -> tuple[str, ...]:
        """Return tool IDs declared by one completed assistant message.

        Main callers:
            ``_run_prompt_loop`` when it snapshots the common reasoning context
            for a batch of sibling tool calls.
        """
        if event.event_type != "message_end":
            return ()
        message = event.payload.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            return ()
        content = message.get("content")
        if not isinstance(content, list):
            return ()
        return tuple(
            str(block["id"])
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "toolCall"
            and isinstance(block.get("id"), str)
            and block["id"]
        )

    async def close(self) -> None:
        """Gracefully dispose Pi; terminate only when graceful shutdown fails."""
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            request_id = self._new_request_id("shutdown")
            try:
                await self._send(shutdown_command(request_id=request_id))
                message = await self._receive(
                    timeout_seconds=self.config.shutdown_timeout_seconds
                )
                if not isinstance(message, VatShutdownComplete):
                    raise RuntimeError(
                        "Vat runner did not acknowledge shutdown; got "
                        f"{runner_message_kind(message)!r}."
                    )
                if message.request_id != request_id:
                    raise RuntimeError("Vat shutdown response request ID mismatch.")
                await asyncio.wait_for(
                    process.wait(), timeout=self.config.shutdown_timeout_seconds
                )
            except BaseException:
                await self._terminate_after_failure()
                raise
        await self._finish_stderr_task()
        self._process = None

    async def _send(self, message: dict[str, JsonValue]) -> None:
        process = self._require_live_process()
        if process.stdin is None:
            raise RuntimeError("Vat runner stdin pipe is unavailable.")
        self._append_audit(direction="host_to_runner", message=message)
        process.stdin.write(encode_jsonl(message))
        await process.stdin.drain()

    async def _receive(self, *, timeout_seconds: float | None):
        process = self._require_live_process()
        if process.stdout is None:
            raise RuntimeError("Vat runner stdout pipe is unavailable.")
        read = process.stdout.readline()
        line = (
            await asyncio.wait_for(read, timeout=timeout_seconds)
            if timeout_seconds is not None
            else await read
        )
        if not line:
            returncode = await process.wait()
            raise RuntimeError(
                f"Vat runner exited before its next protocol message ({returncode=})."
            )
        message = decode_runner_message(line)
        self._append_audit(
            direction="runner_to_host",
            message=json.loads(line.decode("utf-8")),
        )
        return message

    def _require_live_process(self) -> asyncio.subprocess.Process:
        process = self._process
        if process is None or process.returncode is not None:
            raise RuntimeError("Vat runner process is not alive.")
        return process

    async def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        stderr_path = self.config.raw_log_path.with_suffix(".stderr.log")
        while line := await process.stderr.readline():
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            with stderr_path.open("ab") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

    async def _finish_stderr_task(self) -> None:
        if self._stderr_task is not None:
            await self._stderr_task
            self._stderr_task = None

    async def _terminate_after_failure(self) -> None:
        process = self._process
        if process is None:
            return
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except TimeoutError:
                process.kill()
                await process.wait()
        await self._finish_stderr_task()

    def _append_audit(
        self, *, direction: str, message: dict[str, JsonValue]
    ) -> None:
        record = {"direction": direction, "message": message}
        with self.config.raw_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _new_request_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"
