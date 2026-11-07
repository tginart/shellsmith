"""World-simulator harness adapters (Codex, Claude, Pi).

Public import path stays package-rooted so callers can write::

    from shellsmith.episode.gm_process import (
        WorldHarnessLauncher,
        ProcessTurn,
        ...
    )

Harness modules live beside shared types:
    shared.py  — ProcessTurn, protocol, launcher
    codex.py   — CodexWorldProcess
    claude.py  — ClaudeWorldProcess
    pi.py      — PiWorldProcess
"""

from shellsmith.episode.gm_process.claude import (
    ClaudeTransientProviderError,
    ClaudeWorldConfig,
    ClaudeWorldProcess,
)
from shellsmith.episode.gm_process.codex import (
    CodexWorldConfig,
    CodexWorldProcess,
)
from shellsmith.episode.gm_process.pi import (
    PiWorldConfig,
    PiWorldProcess,
)
from shellsmith.episode.gm_process.shared import (
    AgentEvent,
    ProcessTurn,
    ResumableAgentProcess,
    WorldHarnessConfig,
    WorldHarnessLauncher,
    WorldTurnContext,
    normalized_event,
)

__all__ = [
    "AgentEvent",
    "ClaudeTransientProviderError",
    "ClaudeWorldConfig",
    "ClaudeWorldProcess",
    "CodexWorldConfig",
    "CodexWorldProcess",
    "PiWorldConfig",
    "PiWorldProcess",
    "ProcessTurn",
    "ResumableAgentProcess",
    "WorldHarnessConfig",
    "WorldHarnessLauncher",
    "WorldTurnContext",
    "normalized_event",
]
