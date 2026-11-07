"""Purpose: Preserve GM token observations independently of episode acceptance.
Input: A private Codex stdout log and invocation metadata.
Output: A sanitized, atomic receipt with cumulative thread counters.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from shellsmith.process_io import write_json

TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "reasoning_output_tokens",
)


def save_usage(path: Path, *, started_at: float, finished_at: float | None,
               stage: str, thread_id: str | None, outcome: str) -> None:
    """Purpose: Retain available usage even when the CLI or artifact check fails.
    Input: The stdout path, invocation times, stage, thread and outcome.
    Output: An adjacent usage receipt. Receipt errors cannot alter the invocation.
    """
    try:
        usage = {}
        malformed = False
        if path.is_file():
            with path.open() as stream:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        malformed = True
                        continue
                    if not isinstance(event, dict):
                        malformed = True
                        continue
                    if event.get("type") == "thread.started":
                        thread_id = event.get("thread_id") or thread_id
                    values = event.get("usage")
                    if event.get("type") == "turn.completed" and isinstance(values, dict):
                        usage = {key: value for key, value in values.items()
                                 if key in TOKEN_FIELDS and type(value) is int and value >= 0}
        write_json(path.with_suffix(".usage.json"), {
            "version": 1, "invocation_id": path.stem, "thread_id": thread_id,
            "stage": stage, "started_at": started_at, "finished_at": finished_at,
            "outcome": outcome, "cumulative_usage": usage,
            "usage_known": bool(usage and thread_id), "malformed_log": malformed,
        })
    except (OSError, ValueError, TypeError) as error:
        print(f"shellsmith: GM usage receipt failed: {error}", file=sys.stderr, flush=True)
