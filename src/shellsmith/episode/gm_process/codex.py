"""Codex CLI adapter for the shellsmith simulator agent."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from shellsmith.config import (
    DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS,
    DEFAULT_GM_REASONING_EFFORT,
    DEFAULT_GM_TURN_TIMEOUT_SECONDS,
)
from shellsmith.process_io import run_captured_process, write_json
from shellsmith.types import JsonValue
from shellsmith.episode.gm_process.python_runtime import PythonRuntime
from shellsmith.episode.failure import (
    ExecutionTimeoutError,
)
from shellsmith.episode.gm_process.shared import (
    AgentEvent,
    ProcessTurn,
    WorldTurnContext,
    normalized_event,
)


# How Codex confines the Game Master's shell commands and file edits.
#
# ``permission_profile``: Codex's own OS sandbox (Seatbelt on macOS, bubblewrap
# on Linux) scoped to an episode-only permission profile. This is the mode for
# host runs, where the Game Master shares a machine with everything else.
#
# ``danger-full-access``: explicitly unsafe opt-in for controlled experiments.
# It is never selected by the Docker launcher: a container holding credentials
# is not an isolation boundary between those credentials and model tools.
# Docker runs require nested user namespaces and retain the restricted profile.
SANDBOX_PERMISSION_PROFILE = "permission_profile"
SANDBOX_FULL_ACCESS = "danger-full-access"
CODEX_SANDBOX_MODES = (SANDBOX_PERMISSION_PROFILE, SANDBOX_FULL_ACCESS)

# Marker bubblewrap prints when the kernel/container refuses a new user
# namespace; seen in ``command_execution`` output of a sandboxed Codex turn.
BWRAP_NAMESPACE_FAILURE = "bwrap: No permissions to create a new namespace"


@dataclass(frozen=True)
class CodexWorldConfig:
    """Launch configuration for one private persistent Codex world thread."""

    executable: Path
    model: str
    workspace_dir: Path
    log_dir: Path
    bootstrap_response_schema_path: Path
    additional_writable_dirs: tuple[Path, ...] = ()
    reasoning_effort: str = DEFAULT_GM_REASONING_EFFORT
    permission_profile: str = "game_master"
    sandbox_mode: str = SANDBOX_PERMISSION_PROFILE
    timeout_seconds: float = DEFAULT_GM_TURN_TIMEOUT_SECONDS
    api_idle_timeout_seconds: float = DEFAULT_GM_API_IDLE_TIMEOUT_SECONDS
    ignore_user_config: bool = True
    extra_args: tuple[str, ...] = ()
    python_runtime: PythonRuntime | None = None


class CodexWorldProcess:
    """Live adapter for `codex exec` and `codex exec resume`.

    Main callers:
        `WorldBridge` for per-tool observations and
        `EpisodeOrchestrator` for bootstrap and final judging.

    Behavior:
        - Run `codex exec --json -` in the private world directory and parse
          exactly one `thread.started` ID.
        - Preserve the complete JSONL stream, including command execution and
          file-change events, under the episode log directory.
        - Resume the exact thread for each serialized Player action.
        - Treat the requested `turns/<index>/response.json` or
          `final/response.json` artifact as authoritative, not Codex's final
          conversational message.
        - Let the orchestrator validate turn/final artifacts after Codex exits.
        - Fail loudly on timeout, nonzero exit, missing thread ID, mismatched
          thread ID, missing response artifact, mismatched action ID, or malformed
          JSON.
    """

    def __init__(self, config: CodexWorldConfig) -> None:
        if config.python_runtime is not None:
            config.python_runtime.validate()
        self.config = config
        self._next_turn_index = 0
        self._events_by_session: dict[str, list[AgentEvent]] = {}
        self._closed = False

    async def start(self, prompt: str, *, turn_context: WorldTurnContext | None = None) -> ProcessTurn:
        """Start the private world thread and require one thread.started event.

        Boot is artifact-first: when ``turn_context.output_schema_path`` is None
        (the orchestrator default), Codex does not pass ``--output-schema``.
        The durable ``world/bootstrap/response.json`` is authoritative.
        """
        if self._closed:
            raise RuntimeError("Codex world process adapter is closed.")
        context = turn_context or WorldTurnContext(
            stage="bootstrap",
            output_schema_path=None,
        )
        command = self._base_command(resume=False)
        turn = await self._run_turn(
            command=command,
            prompt=prompt,
            expected_session_id=None,
            resume_session_id=None,
            output_schema_path=context.output_schema_path,
            stage=context.stage,
        )
        artifact_path = context.expected_artifact_path
        if artifact_path is not None and artifact_path.is_file():
            try:
                parsed = json.loads(artifact_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return turn
            if isinstance(parsed, dict):
                return ProcessTurn(
                    session_id=turn.session_id,
                    events=turn.events,
                    final_response={str(key): value for key, value in parsed.items()},
                    raw_log_path=turn.raw_log_path,
                )
        return turn

    async def resume(
        self,
        session_id: str,
        prompt: str,
        *,
        output_schema_path: Path | None = None,
        turn_context: WorldTurnContext | None = None,
    ) -> ProcessTurn:
        """Resume exactly one known world thread for an action or final judge."""
        if self._closed:
            raise RuntimeError("Codex world process adapter is closed.")
        if session_id not in self._events_by_session:
            raise RuntimeError(f"Unknown Codex world session ID: {session_id!r}.")
        schema_path = (
            turn_context.output_schema_path
            if turn_context is not None
            else output_schema_path
        )
        stage = turn_context.stage if turn_context is not None else "resume"
        command = self._base_command(resume=True)
        turn = await self._run_turn(
            command=command,
            prompt=prompt,
            expected_session_id=session_id,
            resume_session_id=session_id,
            output_schema_path=schema_path,
            stage=stage,
        )
        artifact_path = (
            turn_context.expected_artifact_path if turn_context is not None else None
        )
        if artifact_path is not None and artifact_path.is_file():
            try:
                parsed = json.loads(artifact_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return turn
            if isinstance(parsed, dict):
                return ProcessTurn(
                    session_id=turn.session_id,
                    events=turn.events,
                    final_response={str(key): value for key, value in parsed.items()},
                    raw_log_path=turn.raw_log_path,
                )
        return turn

    async def transcript(self, session_id: str) -> tuple[AgentEvent, ...]:
        """Return normalized events accumulated by this adapter instance."""
        try:
            return tuple(self._events_by_session[session_id])
        except KeyError as error:
            raise RuntimeError(
                f"Unknown Codex world session ID: {session_id!r}."
            ) from error

    async def close(self) -> None:
        """Prevent additional turns; each Codex child already exits per turn."""
        self._closed = True

    def _sandbox_args(self) -> list[str]:
        """Return the Codex flags that select the confinement for this run.

        See ``CODEX_SANDBOX_MODES`` for why two modes exist.
        """
        mode = self.config.sandbox_mode
        if mode == SANDBOX_FULL_ACCESS:
            # Config form, not ``--sandbox``: ``codex exec resume`` rejects the
            # flag ("unexpected argument '--sandbox'") while ``-c`` is accepted
            # by both the initial exec and every resume.
            return ["-c", f'sandbox_mode="{SANDBOX_FULL_ACCESS}"']
        if mode != SANDBOX_PERMISSION_PROFILE:
            raise RuntimeError(
                f"Unknown Codex world sandbox mode {mode!r}; "
                f"expected one of {CODEX_SANDBOX_MODES}."
            )
        if not self.config.permission_profile.replace("_", "").isalnum():
            raise RuntimeError(
                "Codex world permission profile must contain only letters, "
                "numbers, and underscores."
            )
        profile = self.config.permission_profile
        runtime_rule = ""
        if self.config.python_runtime is not None:
            self.config.python_runtime.validate()
            runtime_rule = f', {json.dumps(str(self.config.python_runtime.prefix))}="read"'
        permission_config = (
            "{ description=\"Episode-only Game Master\", "
            "filesystem={ \":minimal\"=\"read\", "
            "\":workspace_roots\"={ \".\"=\"write\" }" + runtime_rule + " }, "
            "network={ enabled=true } }"
        )
        return [
            "-c",
            f'default_permissions="{profile}"',
            "-c",
            f"permissions.{profile}={permission_config}",
        ]

    def _base_command(self, *, resume: bool) -> list[str]:
        from .auth_identity import verify_guard

        command = [str(self.config.executable), "exec"]
        for path in self.config.additional_writable_dirs:
            command.extend(["--add-dir", str(path)])
        if resume:
            command.append("resume")
        # exec/resume has its own config overrides. Passing provider settings
        # before the subcommand leaves resumed turns on the default provider.
        command.extend(verify_guard(self.config.executable))
        command.extend(
            [
                "--json",
                "--model",
                self.config.model,
                "-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"',
            ]
        )
        command.extend(self._sandbox_args())
        command.extend(
            [
                "-c",
                'shell_environment_policy.inherit="core"',
                "--skip-git-repo-check",
                "--ignore-rules",
            ]
        )
        if self.config.ignore_user_config:
            command.append("--ignore-user-config")
        if self.config.python_runtime is not None:
            command.extend(["-c", "shell_environment_policy.set={ "
                            f'PATH={json.dumps(self.config.python_runtime.shell_path)}, '
                            f'TMPDIR={json.dumps(str(self.config.workspace_dir / "world" / "runtime_scratch"))}, '
                            f'PIP_CACHE_DIR={json.dumps(str(self.config.workspace_dir / "world" / "runtime_scratch" / "pip_cache"))}, '
                            'PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1" }'])
        if not resume:
            command.extend(
                [
                    "--cd",
                    str(self.config.workspace_dir),
                ]
            )
        command.extend(self.config.extra_args)
        return command

    async def _run_turn(self, **kwargs) -> ProcessTurn:
        """Purpose: Preserve usage across successful and unsuccessful CLI calls.
        Input: The private invocation arguments. Output: The original turn result.
        """
        from .usage import save_usage

        path = self.config.log_dir / f"{self._next_turn_index:06d}_{kwargs['stage']}.jsonl"
        started_at = time.time()
        fields = {"started_at": started_at, "stage": kwargs["stage"],
                  "thread_id": kwargs.get("expected_session_id")}
        save_usage(path, **fields, finished_at=None, outcome="running")
        outcome = "failed"
        try:
            result = await self._execute_turn(**kwargs)
            outcome = "succeeded"
            return result
        finally:
            save_usage(path, **fields, finished_at=time.time(), outcome=outcome)

    async def _execute_turn(
        self,
        *,
        command: list[str],
        prompt: str,
        expected_session_id: str | None,
        resume_session_id: str | None,
        output_schema_path: Path | None,
        stage: str,
    ) -> ProcessTurn:
        """Execute one Codex child and preserve its complete stdout/stderr audit.

        Main callers:
            `start` for boot and `resume` for each world action/final judgment.

        Why this exists:
            Codex is process-per-turn. We therefore make each child invocation a
            durable transaction and accept no conversational fallback when the
            CLI exits unsuccessfully or omits its thread identity.
        """
        turn_index = self._next_turn_index
        self._next_turn_index += 1
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{turn_index:06d}_{stage}"
        stdout_path: Path = self.config.log_dir.joinpath(f"{stem}.jsonl")
        stderr_path: Path = self.config.log_dir.joinpath(f"{stem}.stderr.log")
        last_message_path: Path = self.config.log_dir.joinpath(f"{stem}.last_message.txt")
        if output_schema_path is not None:
            command.extend(["--output-schema", str(output_schema_path)])
        command.extend(["--output-last-message", str(last_message_path)])
        if resume_session_id is not None:
            command.append(resume_session_id)
        command.append("-")

        if self.config.python_runtime is not None:
            simulation = self.config.workspace_dir / "world" / "simulation_root"
            scratch = self.config.workspace_dir / "world" / "runtime_scratch"
            prompt += (
                "\n\nApproved host Python for private world computation: "
                f"{self.config.python_runtime.executable}. Use this absolute executable "
                "instead of /usr/bin/python3 or relying on a login shell's PATH. "
                "The installation is read-only; do not modify it or activate the trainer environment. "
                f"Run world computations with cwd and durable world state under {simulation}. "
                f"The orchestrator has already created {scratch}; it exists even when empty. "
                f"If a private virtual environment is needed, run the approved Python with "
                f"-m venv --copies {scratch / 'venv'} and use that venv's absolute bin/python. "
                "Keep dependency and build caches in runtime_scratch, never in durable world state. "
                "This runtime permission does not authorize access to other host files or credentials."
            )

        # Publish before spawning: failed audit I/O must not orphan a live CLI.
        write_json(stdout_path.with_suffix(".request.json"), {"stage": stage, "prompt": prompt})
        try:
            from .api_idle import api_idle_command
            with api_idle_command(command, idle_seconds=self.config.api_idle_timeout_seconds,
                                  turn_seconds=self.config.timeout_seconds) as (api_command, transport):
                try:
                    returncode, stdout, stderr = await run_captured_process(
                        api_command, cwd=self.config.workspace_dir, stdout_path=stdout_path, stderr_path=stderr_path,
                        timeout=self.config.timeout_seconds, input_data=prompt.encode("utf-8"))
                finally:
                    write_json(stdout_path.with_suffix(".api-transport.json"), {
                        "idle_timeout_seconds": self.config.api_idle_timeout_seconds,
                        "turn_timeout_seconds": self.config.timeout_seconds,
                        "api_idle_timeouts": list(transport.timeouts),
                    })
        except TimeoutError as error:
            raise ExecutionTimeoutError(
                component="gm", stage=stage,
                timeout_seconds=self.config.timeout_seconds,
                provider_attributable=True,
            ) from error
        if returncode != 0 and transport.timeouts:
            raise ExecutionTimeoutError(component="gm", stage=stage + ":api_idle",
                                        timeout_seconds=self.config.api_idle_timeout_seconds,
                                        provider_attributable=True)
        if returncode != 0:
            tail = stderr.decode("utf-8", errors="replace")[-2_000:]
            if not tail.strip():
                # Codex reports refusals (e.g. safety-filter turn.failed) on
                # stdout as JSONL events with an empty stderr; surface the last
                # error message so the trainer can classify the failure.
                tail = _last_codex_error_message(stdout)
            raise RuntimeError(
                f"Codex world {stage} exited with {returncode}: {tail}"
            )

        events = self._parse_events(stdout_path)
        _require_working_sandbox(events, stage=stage, log_path=stdout_path)
        started_ids = {
            str(event.payload["thread_id"])
            for event in events
            if event.type == "thread.started"
            and isinstance(event.payload.get("thread_id"), str)
        }
        if expected_session_id is None:
            if len(started_ids) != 1:
                raise RuntimeError(
                    "Codex bootstrap must emit exactly one thread.started ID; "
                    f"found {sorted(started_ids)}."
                )
            session_id = next(iter(started_ids))
            self._events_by_session[session_id] = []
        else:
            if started_ids and started_ids != {expected_session_id}:
                raise RuntimeError(
                    "Codex resume emitted a mismatched thread ID: "
                    f"expected {expected_session_id!r}, got {sorted(started_ids)}."
                )
            session_id = expected_session_id
        self._events_by_session[session_id].extend(events)
        final_response = self._load_last_message(
            last_message_path,
            allow_empty=output_schema_path is None,
        )
        return ProcessTurn(
            session_id=session_id,
            events=events,
            final_response=final_response,
            raw_log_path=stdout_path,
        )

    @staticmethod
    def _parse_events(path: Path) -> tuple[AgentEvent, ...]:
        events: list[AgentEvent] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Codex emitted malformed JSONL at {path}:{line_number}."
                ) from error
            if not isinstance(raw, dict):
                raise RuntimeError(
                    f"Codex JSONL event must be an object at {path}:{line_number}."
                )
            events.append(normalized_event(raw))
        if not events:
            raise RuntimeError(f"Codex emitted no JSONL events: {path}.")
        return tuple(events)

    @staticmethod
    def _load_last_message(
        path: Path, *, allow_empty: bool = False
    ) -> dict[str, JsonValue]:
        if not path.is_file():
            raise RuntimeError(f"Codex did not write its last message: {path}.")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            if allow_empty:
                return {}
            raise RuntimeError(f"Codex wrote an empty last message: {path}.")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}
        if not isinstance(parsed, dict):
            raise RuntimeError("Codex structured last message must be a JSON object.")
        return {str(key): value for key, value in parsed.items()}


def _require_working_sandbox(
    events: tuple[AgentEvent, ...], *, stage: str, log_path: Path
) -> None:
    """Fail loudly when Codex's own sandbox could not start inside this host.

    Main callers:
        ``CodexWorldProcess._run_turn`` after each child exits 0.

    Why this exists:
        When bubblewrap cannot create a user namespace, Codex still completes
        the turn with exit 0; only each ``command_execution`` carries the bwrap
        error and the world never writes its artifact. Without this check the
        orchestrator reports a misleading "artifact missing" three times over.
    """
    for event in events:
        item = event.payload.get("item")
        if not isinstance(item, dict) or item.get("type") != "command_execution":
            continue
        output = item.get("aggregated_output")
        if isinstance(output, str) and BWRAP_NAMESPACE_FAILURE in output:
            raise RuntimeError(
                f"Codex world {stage}: the Codex Linux sandbox cannot create "
                "user namespaces on this host, so every command and file write "
                "fails. Use a system that supports the restricted sandbox or "
                f"protected host execution; do not disable confinement. See {log_path}."
            )


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
