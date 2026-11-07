"""Disabled Pi GM adapter with offline configuration and audit-log helpers.

Pi's approved shell/read tools have no enforced filesystem or process boundary.
Filtering a child shell's environment cannot protect credentials still available
through the parent environment or auth files. Until confinement and a credential
broker exist, every execution entry point fails before reading credentials,
writing turn artifacts, or spawning a process. A workspace cwd is not a sandbox.

The Pi Player SDK is a separate, supported path in ``vat_process.py``; this
restriction applies only to the GM adapter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shellsmith.config import DEFAULT_GM_REASONING_EFFORT
from shellsmith.types import JsonValue
from shellsmith.episode.gm_process.shared import (
    AgentEvent,
    PI_GM_DISABLED_MESSAGE,
    ProcessTurn,
    WorldTurnContext,
)


@dataclass(frozen=True)
class PiWorldConfig:
    """Inputs for a persistent Pi session backed by an API model."""

    executable: Path
    model: str
    workspace_dir: Path
    log_dir: Path
    bootstrap_response_schema_path: Path
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    base_url: str | None = None
    api: str = "openai-completions"
    context_window: int = 224 * 1024
    max_tokens: int = 16_384
    reasoning_effort: str = DEFAULT_GM_REASONING_EFFORT
    timeout_seconds: float = 900.0
    enabled_tools: tuple[str, ...] = (
        "read",
        "bash",
        "edit",
        "write",
        "grep",
        "find",
        "ls",
    )


class PiWorldProcess:
    """Retain offline Pi helpers, but reject all GM execution unconditionally.

    Construction is inert so configuration and historical transcript tests stay
    usable. ``start``, ``resume``, and even direct ``_run_turn`` calls fail closed;
    there is no unsafe override or tool/provider combination that enables them.
    """

    def __init__(self, config: PiWorldConfig) -> None:
        self.config = config
        self._events_by_session: dict[str, list[AgentEvent]] = {}
        self._closed = False

    async def start(
        self,
        prompt: str,
        *,
        turn_context: WorldTurnContext | None = None,
    ) -> ProcessTurn:
        """Reject GM bootstrap before any artifact, credential, or process access."""
        raise RuntimeError(PI_GM_DISABLED_MESSAGE)

    async def resume(
        self,
        session_id: str,
        prompt: str,
        *,
        turn_context: WorldTurnContext | None = None,
        output_schema_path: Path | None = None,
    ) -> ProcessTurn:
        """Reject resuming even a previously recorded Pi GM session."""
        raise RuntimeError(PI_GM_DISABLED_MESSAGE)

    async def transcript(self, session_id: str) -> tuple[AgentEvent, ...]:
        try:
            return tuple(self._events_by_session[session_id])
        except KeyError as error:
            raise RuntimeError(f"Unknown Pi world session ID: {session_id!r}.") from error

    async def close(self) -> None:
        self._closed = True

    def _command(self, session_id: str) -> list[str]:
        """Describe the legacy invocation for offline compatibility tests only.

        This command is not confined and must not be executed. All execution
        entry points in this adapter reject it, regardless of enabled tools.
        """
        session_dir: Path = self.config.log_dir.joinpath("sessions")
        session_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.config.executable),
            "--provider",
            self.config.provider,
            "--model",
            self.config.model,
            "--thinking",
            self.config.reasoning_effort,
            "--mode",
            "json",
            "--print",
            "--session-dir",
            str(session_dir),
            "--session-id",
            session_id,
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--approve",
            "--tools",
            ",".join(self.config.enabled_tools),
        ]
        return command

    def _prepare_agent_home(self) -> Path | None:
        """Materialize an episode-local Pi provider for custom OpenAI endpoints."""
        if self.config.base_url is None:
            return None
        agent_home: Path = self.config.log_dir.joinpath("agent_home")
        agent_home.mkdir(parents=True, exist_ok=True)
        models = {
            "providers": {
                self.config.provider: {
                    "baseUrl": self.config.base_url,
                    "api": self.config.api,
                    "apiKey": f"${self.config.api_key_env}",
                    "authHeader": True,
                    "compat": {
                        "supportsDeveloperRole": False,
                        "supportsReasoningEffort": False,
                    },
                    "models": [
                        {
                            "id": self.config.model,
                            "name": self.config.model,
                            "reasoning": True,
                            "input": ["text"],
                            "contextWindow": self.config.context_window,
                            "maxTokens": self.config.max_tokens,
                            "cost": {
                                "input": 0,
                                "output": 0,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            }
        }
        models_path: Path = agent_home.joinpath("models.json")
        models_path.write_text(
            json.dumps(models, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return agent_home

    async def _run_turn(self, *, session_id: str, prompt: str, stage: str) -> ProcessTurn:
        """Fail closed for direct callers too; never assemble a credential env."""
        raise RuntimeError(PI_GM_DISABLED_MESSAGE)

    @staticmethod
    def _parse_output(path: Path) -> tuple[tuple[AgentEvent, ...], dict[str, JsonValue]]:
        """Normalize Pi JSONL and extract the last successful assistant text."""
        events: list[AgentEvent] = []
        last_assistant_text = ""
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"Pi emitted malformed JSONL at {path}:{line_number}.") from error
            if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
                raise RuntimeError(f"Pi event is malformed at {path}:{line_number}.")
            events.append(
                AgentEvent(
                    type=f"pi.{raw['type']}",
                    payload={str(key): value for key, value in raw.items()},
                )
            )
            message = raw.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                error_message = message.get("errorMessage")
                if isinstance(error_message, str) and error_message:
                    raise RuntimeError(f"Pi model request failed: {error_message}")
                text = PiWorldProcess._message_text(message)
                if text:
                    last_assistant_text = text
        if not events:
            raise RuntimeError(f"Pi emitted no JSONL events: {path}.")
        if not last_assistant_text:
            return tuple(events), {}
        json_text = PiWorldProcess._unwrap_json_fence(last_assistant_text)
        try:
            parsed: Any = json.loads(json_text)
        except json.JSONDecodeError:
            return tuple(events), {"text": last_assistant_text}
        if not isinstance(parsed, dict):
            return tuple(events), {"text": last_assistant_text}
        return tuple(events), {str(key): value for key, value in parsed.items()}

    @staticmethod
    def _unwrap_json_fence(text: str) -> str:
        """Unwrap one complete Markdown JSON fence without guessing at prose.

        Main callers:
            `_parse_output`, immediately before strict `json.loads` validation.

        Why this exists:
            Some OpenRouter models faithfully emit the requested JSON object but
            wrap it in a single Markdown fence. Accepting that exact transport
            wrapper keeps the adapter composable without extracting JSON from
            arbitrary conversational text.
        """
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0].strip().lower() == "```json":
            if lines[-1].strip() == "```":
                return "\n".join(lines[1:-1]).strip()
        return text

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if not isinstance(content, list):
            return ""
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ).strip()
