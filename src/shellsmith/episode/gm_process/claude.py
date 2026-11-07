"""Claude CLI adapter for the shellsmith simulator agent."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shellsmith.config import DEFAULT_GM_REASONING_EFFORT
from shellsmith.types import JsonValue
from shellsmith.episode.failure import (
    ExecutionTimeoutError,
)
from shellsmith.episode.gm_process.shared import (
    AgentEvent,
    ProcessTurn,
    WorldTurnContext,
    validate_gm_runtime,
)


@dataclass(frozen=True)
class ClaudeWorldConfig:
    """Launch configuration for one private persistent Claude world session."""

    executable: Path
    model: str
    workspace_dir: Path
    log_dir: Path
    bootstrap_response_schema_path: Path
    additional_writable_dirs: tuple[Path, ...] = ()
    reasoning_effort: str = DEFAULT_GM_REASONING_EFFORT
    timeout_seconds: float = 900.0
    provider_retry_delays_seconds: tuple[float, ...] = (1.0, 3.0)


class ClaudeTransientProviderError(RuntimeError):
    """One retryable Claude transport/service failure with optional session."""

    def __init__(self, message: str, *, session_id: str | None) -> None:
        super().__init__(message)
        self.session_id = session_id


_CLAUDE_TRANSIENT_MARKERS = (
    "connection closed mid-response",
    "connection reset",
    "connection refused",
    "service unavailable",
    "temporarily unavailable",
    "overloaded",
    "internal server error",
    "gateway timeout",
    "bad gateway",
    "http 502",
    "http 503",
    "http 504",
)


def _claude_transient_provider_failure(
    result_event: dict[str, Any] | None,
    *,
    stderr: str,
) -> bool:
    """Distinguish retryable provider outages from model/contract failures."""
    if result_event is not None and result_event.get("terminal_reason") == "api_error":
        return True
    text = " ".join(
        (
            stderr,
            str((result_event or {}).get("result", "")),
            " ".join(str(item) for item in (result_event or {}).get("errors") or []),
        )
    ).casefold()
    return any(marker in text for marker in _CLAUDE_TRANSIENT_MARKERS)


def _last_codex_error_message(stdout: bytes) -> str:
    """Extract the last error/turn.failed message from Codex JSONL stdout.

    Main callers:
        ``CodexWorldProcess`` nonzero-exit handling when stderr is empty.
    """
    last = ""
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        message = event.get("message")
        if event.get("type") == "error" and isinstance(message, str):
            last = message
            continue
        error = event.get("error")
        if (
            event.get("type") == "turn.failed"
            and isinstance(error, dict)
            and isinstance(error.get("message"), str)
        ):
            last = error["message"]
    return last or "(no stderr; no error event in Codex stdout)"


def _artifact_response(path: Path) -> dict[str, JsonValue]:
    """Load one world response artifact when it already contains valid JSON."""
    if not path.is_file():
        raise RuntimeError(f"Expected world artifact is absent: {path}.")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"World artifact is malformed JSON: {path}.") from error
    if not isinstance(parsed, dict):
        raise RuntimeError(f"World artifact must be a JSON object: {path}.")
    return {str(key): value for key, value in parsed.items()}


def _artifact_is_valid(path: Path) -> bool:
    """Return whether one durable response artifact is present and parseable."""
    try:
        _artifact_response(path)
    except RuntimeError:
        return False
    return True


def _idempotent_recovery_prompt(
    *,
    idempotency_key: str | None,
    validation_error: str,
) -> str:
    key_line = (
        f"Idempotency key: {idempotency_key}\n"
        if isinstance(idempotency_key, str) and idempotency_key
        else ""
    )
    return (
        "Your previous world turn may have partially completed. Inspect the "
        "existing private episode state before repeating any mutation. This is "
        "a recovery of the same logical turn, not a new episode.\n"
        f"{key_line}"
        f"Validation error: {validation_error}"
    )
class ClaudeWorldProcess:
    """Legacy Claude adapter retained for offline configuration and audit helpers.

    Construction and transcript inspection remain inert. All execution entry
    points fail closed, including direct turn calls and artifact recovery; an
    explicit model or a previously recorded session cannot enable Claude GM.
    """

    def __init__(self, config: ClaudeWorldConfig) -> None:
        self.config = config
        self._next_turn_index = 0
        self._events_by_session: dict[str, list[AgentEvent]] = {}
        self._closed = False

    async def start(
        self,
        prompt: str,
        *,
        turn_context: WorldTurnContext | None = None,
    ) -> ProcessTurn:
        validate_gm_runtime("claude")
        if self._closed:
            raise RuntimeError("Claude world process adapter is closed.")
        agents_path: Path = self.config.workspace_dir.joinpath("AGENTS.md")
        if not agents_path.is_file():
            raise RuntimeError(
                "Claude world workspace is missing canonical AGENTS.md instructions."
            )
        claude_path: Path = self.config.workspace_dir.joinpath("CLAUDE.md")
        if claude_path.exists():
            raise RuntimeError(
                "Claude world workspace unexpectedly already contains CLAUDE.md."
            )
        # Claude Code intentionally does not discover AGENTS.md. Keep one
        # canonical instruction document and install its documented native
        # import shim at the harness boundary.
        claude_path.write_text("@AGENTS.md\n", encoding="utf-8")
        context = turn_context or WorldTurnContext(
            stage="bootstrap",
            output_schema_path=None,
        )
        return await self._run_turn(
            prompt=prompt,
            expected_session_id=None,
            turn_context=context,
        )

    async def resume(
        self,
        session_id: str,
        prompt: str,
        *,
        turn_context: WorldTurnContext | None = None,
        output_schema_path: Path | None = None,
    ) -> ProcessTurn:
        validate_gm_runtime("claude")
        if self._closed:
            raise RuntimeError("Claude world process adapter is closed.")
        if session_id not in self._events_by_session:
            raise RuntimeError(f"Unknown Claude world session ID: {session_id!r}.")
        context = turn_context or WorldTurnContext(
            stage="resume",
            output_schema_path=output_schema_path,
        )
        return await self._run_turn(
            prompt=prompt,
            expected_session_id=session_id,
            turn_context=context,
        )

    async def transcript(self, session_id: str) -> tuple[AgentEvent, ...]:
        try:
            return tuple(self._events_by_session[session_id])
        except KeyError as error:
            raise RuntimeError(
                f"Unknown Claude world session ID: {session_id!r}."
            ) from error

    async def close(self) -> None:
        self._closed = True

    async def _run_turn(
        self,
        *,
        prompt: str,
        expected_session_id: str | None,
        turn_context: WorldTurnContext,
    ) -> ProcessTurn:
        """Run one logical Claude turn with artifact-first acceptance or repair.

        Main callers:
            ``start`` for boot and ``resume`` for world actions/final judgment.

        Why this exists:
            Normal turns reuse a valid on-disk ``response.json`` even when the CLI
            transport fails. Explicit repairs disable that reuse because the artifact
            can be structurally valid but semantically rejected by the orchestrator.
            Blind identical-prompt provider retries are forbidden.
        """
        validate_gm_runtime("claude")
        artifact_path = turn_context.expected_artifact_path
        if (
            turn_context.reuse_valid_artifact
            and artifact_path is not None
            and _artifact_is_valid(artifact_path)
        ):
            session_id = expected_session_id or "artifact-recovered"
            return ProcessTurn(
                session_id=session_id,
                events=(),
                final_response=_artifact_response(artifact_path),
                raw_log_path=self.config.log_dir.joinpath(
                    f"{self._next_turn_index:06d}_{turn_context.stage}_artifact.json"
                ),
            )

        turn_index = self._next_turn_index
        self._next_turn_index += 1
        try:
            turn = await self._run_turn_once(
                prompt=prompt,
                expected_session_id=expected_session_id,
                output_schema_path=turn_context.output_schema_path,
                stage=turn_context.stage,
                turn_index=turn_index,
                attempt_index=0,
            )
        except ClaudeTransientProviderError as error:
            if (
                turn_context.reuse_valid_artifact
                and artifact_path is not None
                and _artifact_is_valid(artifact_path)
            ):
                session_id = error.session_id or expected_session_id or "artifact-recovered"
                return ProcessTurn(
                    session_id=session_id,
                    events=(),
                    final_response=_artifact_response(artifact_path),
                    raw_log_path=self.config.log_dir.joinpath(
                        f"{turn_index:06d}_{turn_context.stage}_artifact.json"
                    ),
                )
            raise RuntimeError(
                f"Claude world {turn_context.stage} transient provider failure: {error}"
            ) from error

        if artifact_path is None:
            return turn

        if _artifact_is_valid(artifact_path):
            return ProcessTurn(
                session_id=turn.session_id,
                events=turn.events,
                final_response=_artifact_response(artifact_path),
                raw_log_path=turn.raw_log_path,
            )

        if artifact_path.is_file():
            repair_prompt = _idempotent_recovery_prompt(
                idempotency_key=turn_context.idempotency_key,
                validation_error=f"Existing artifact failed validation at {artifact_path}.",
            )
            return await self._run_turn_once(
                prompt=repair_prompt,
                expected_session_id=turn.session_id,
                output_schema_path=turn_context.output_schema_path,
                stage=f"{turn_context.stage}_repair",
                turn_index=turn_index,
                attempt_index=1,
            )

        recovery_prompt = _idempotent_recovery_prompt(
            idempotency_key=turn_context.idempotency_key,
            validation_error=f"Expected artifact is still absent: {artifact_path}.",
        )
        return await self._run_turn_once(
            prompt=recovery_prompt,
            expected_session_id=turn.session_id,
            output_schema_path=turn_context.output_schema_path,
            stage=f"{turn_context.stage}_recovery",
            turn_index=turn_index,
            attempt_index=1,
        )

    async def _run_turn_once(
        self,
        *,
        prompt: str,
        expected_session_id: str | None,
        output_schema_path: Path | None,
        stage: str,
        turn_index: int,
        attempt_index: int,
    ) -> ProcessTurn:
        """Reject physical invocation too, before creating logs or reading schema."""
        validate_gm_runtime("claude")
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{turn_index:06d}_{stage}"
        if attempt_index:
            stem += f"_provider_retry_{attempt_index:02d}"
        stdout_path: Path = self.config.log_dir.joinpath(f"{stem}.jsonl")
        stderr_path: Path = self.config.log_dir.joinpath(f"{stem}.stderr.log")

        command = [
            str(self.config.executable),
            "-p",
            "--model",
            self.config.model,
            "--effort",
            self.config.reasoning_effort,
            "--output-format",
            "stream-json",
            "--verbose",
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "Bash,Read,Write,Edit,Glob,Grep,WebSearch,WebFetch",
        ]
        for path in self.config.additional_writable_dirs:
            command.extend(["--add-dir", str(path)])
        if expected_session_id is not None:
            command.extend(["--resume", expected_session_id])
        if output_schema_path is not None:
            schema = json.loads(output_schema_path.read_text(encoding="utf-8"))
            if not isinstance(schema, dict):
                raise RuntimeError("World response schema must be a JSON object.")
            # Claude CLI validates the supported schema vocabulary itself but
            # currently rejects the otherwise-valid Draft 2020-12 declaration.
            schema.pop("$schema", None)
            command.extend(
                ["--json-schema", json.dumps(schema, separators=(",", ":"))]
            )

        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.config.workspace_dir),
        )
        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                stdout, stderr = await process.communicate(prompt.encode("utf-8"))
        except TimeoutError as error:
            if process.returncode is None:
                process.kill()
            stdout, stderr = await process.communicate()
            stdout_path.write_bytes(stdout)
            stderr_path.write_bytes(stderr)
            raise ExecutionTimeoutError(
                component="gm",
                stage=stage,
                timeout_seconds=self.config.timeout_seconds,
                provider_attributable=True,
            ) from error
        except BaseException:
            if process.returncode is None:
                process.kill()
            stdout, stderr = await process.communicate()
            stdout_path.write_bytes(stdout)
            stderr_path.write_bytes(stderr)
            raise
        stdout_path.write_bytes(stdout)
        stderr_path.write_bytes(stderr)
        raw_events: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            stdout.decode("utf-8").splitlines(), start=1
        ):
            try:
                raw_event = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Claude emitted malformed JSONL at {stdout_path}:{line_number}."
                ) from error
            if not isinstance(raw_event, dict) or not isinstance(
                raw_event.get("type"), str
            ):
                raise RuntimeError(
                    f"Claude emitted a malformed event at {stdout_path}:{line_number}."
                )
            raw_events.append(raw_event)
        results = [event for event in raw_events if event.get("type") == "result"]
        raw = results[0] if len(results) == 1 else None
        stderr_text = stderr.decode("utf-8", errors="replace")
        session_id = raw.get("session_id") if raw is not None else None
        if not isinstance(session_id, str) or not session_id:
            session_id = None
        if _claude_transient_provider_failure(raw, stderr=stderr_text):
            if session_id is not None:
                events = tuple(
                    AgentEvent(
                        type=f"claude.{event['type']}",
                        payload={str(key): value for key, value in event.items()},
                    )
                    for event in raw_events
                )
                self._events_by_session.setdefault(session_id, []).extend(events)
            detail = (
                str((raw or {}).get("result", "")).strip()
                or " ".join(str(item) for item in (raw or {}).get("errors") or [])
                or stderr_text.strip()
                or f"process exited with {process.returncode}"
            )
            raise ClaudeTransientProviderError(detail, session_id=session_id)
        if process.returncode != 0:
            detail = (
                str((raw or {}).get("terminal_reason", "")).strip()
                or " ".join(str(item) for item in (raw or {}).get("errors") or [])
                or stderr_text.strip()
                or "no error detail emitted"
            )
            raise RuntimeError(
                f"Claude world {stage} exited with {process.returncode}: {detail}"
            )
        if raw is None:
            raise RuntimeError(
                f"Claude emitted {len(results)} result events at {stdout_path}; expected one."
            )
        if session_id is None:
            raise RuntimeError("Claude result omitted its session_id.")
        if expected_session_id is not None and session_id != expected_session_id:
            raise RuntimeError(
                "Claude resume emitted a mismatched session ID: "
                f"expected {expected_session_id!r}, got {session_id!r}."
            )
        if expected_session_id is None:
            self._events_by_session[session_id] = []
        events = tuple(
            AgentEvent(
                type=f"claude.{event['type']}",
                payload={str(key): value for key, value in event.items()},
            )
            for event in raw_events
        )
        self._events_by_session[session_id].extend(events)
        result = raw.get("structured_output", raw.get("result", ""))
        if isinstance(result, dict):
            final_response = {str(key): value for key, value in result.items()}
        else:
            text = str(result).strip()
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                final_response = {"text": text} if text else {}
            else:
                final_response = parsed if isinstance(parsed, dict) else {"text": text}
        return ProcessTurn(
            session_id=session_id,
            events=events,
            final_response=final_response,
            raw_log_path=stdout_path,
        )
