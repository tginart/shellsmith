"""Checkpoint-aligned metrics with a one-row write-ahead record in trainer state.

The lifetime run lock is held by the caller. State/checkpoint publication commits
both an exact metrics prefix and its last row *before* appending that row. Each
append is fsynced before another transaction can replace the write-ahead record.
On resume the restored checkpoint, not the last attempted step, chooses the
prefix. Removed/repaired bytes and any discarded write-ahead row are archived.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from shellsmith.process_io import fsync_directory, write_json

HEAD_KEY = "metrics_head"


def _position(value: dict[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    return value["step"], value["next_world_offset"], tuple(value.get("settled_world_offsets", []))


def _head(prefix: bytes, tail: bytes) -> dict[str, Any]:
    return {"version": 1, "prefix_bytes": len(prefix),
            "prefix_sha256": hashlib.sha256(prefix).hexdigest(), "tail": tail.decode("utf-8")}


def _recover_head(data: bytes, head: dict[str, Any]) -> bytes:
    """Only the final row may be missing/torn; preceding durable bytes must agree."""
    if (not isinstance(head, dict) or head.get("version") != 1
            or type(head.get("prefix_bytes")) is not int or head["prefix_bytes"] < 0
            or not isinstance(head.get("tail"), str)):
        raise RuntimeError("Invalid durable metrics boundary; refusing to rewrite metrics.")
    size = head["prefix_bytes"]
    prefix = data[:size]
    if len(prefix) != size or hashlib.sha256(prefix).hexdigest() != head.get("prefix_sha256"):
        raise RuntimeError("Committed metrics prefix is missing or corrupt; refusing to rewrite metrics.")
    tail = head["tail"].encode("utf-8")
    if tail:
        try:
            if not tail.endswith(b"\n") or not isinstance(json.loads(tail), dict):
                raise ValueError("not a complete metrics row")
        except ValueError as error:
            raise RuntimeError("Invalid durable metrics last row; refusing to rewrite metrics.") from error
    elif size:
        raise RuntimeError("Invalid durable metrics last row; refusing to rewrite metrics.")
    return prefix + tail


def _covered(position: tuple[int, int, tuple[int, ...]], target: tuple[int, int, tuple[int, ...]]) -> bool:
    step, offset, sparse = position
    target_step, target_offset, target_sparse = target
    return (step <= target_step and offset <= target_offset
            and all(value < target_offset or value in target_sparse for value in sparse))


def _legacy_head(data: bytes, target: dict[str, Any]) -> dict[str, Any]:
    """Adopt an unambiguous old-format prefix, including sparse/no-datum progress.

    Legacy durable-input checkpoints may have no historical metrics at all. Do
    not invent those rows. Nonempty logs must reach the restored position, and
    already interleaved/replayed histories or damaged committed rows fail closed.
    """
    if not data:
        return _head(b"", b"")
    position = _position(target)
    prefix = bytearray()
    tail = b""
    previous = None
    rolled_back = False
    for line in data.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break  # only an uncommitted final fragment may be discarded below
        try:
            row = json.loads(line)
            if (not isinstance(row, dict) or type(row.get("step")) is not int or row["step"] < 0
                    or type(row.get("datums")) is not int or row["datums"] < 0
                    or type(row.get("next_world_offset")) is not int or row["next_world_offset"] < 0
                    or not isinstance(row.get("settled_world_offsets", []), list)
                    or any(type(value) is not int or value < 0 for value in row.get("settled_world_offsets", []))):
                raise ValueError("invalid metrics position")
            after = (row["step"] + int(row["datums"] > 0), row["next_world_offset"],
                     tuple(row.get("settled_world_offsets", [])))
        except (ValueError, UnicodeError) as error:
            raise RuntimeError("Cannot reconcile corrupt legacy metrics; original log left intact.") from error
        if not _covered(after, position):
            rolled_back = True
            continue
        if rolled_back or (previous is not None and
                           (row["step"] != previous[0] or not _covered(previous, after))):
            raise RuntimeError("Cannot reconcile ambiguous replayed legacy metrics; original log left intact.")
        previous = after
        prefix.extend(line)
        tail = line
    if previous != position and not (previous is None and position == (0, 0, ())):
        raise RuntimeError("Legacy metrics do not cover the committed cursor; original log left intact.")
    return _head(bytes(prefix[:-len(tail)]) if tail else b"", tail)


def _replace_metrics(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class MetricsJournal:
    """Materialize metrics.jsonl from the trainer's durable publication boundary."""

    def __init__(self, path: Path, data: bytes):
        self.path = path
        self._size = len(data)
        self._digest = hashlib.sha256(data)

    @classmethod
    def recover(cls, path: Path, *, state: dict[str, Any], checkpoint: dict[str, Any] | None) -> MetricsJournal:
        target = checkpoint if checkpoint is not None else state
        original = path.read_bytes() if path.exists() else b""
        previous_head = state.get(HEAD_KEY)
        head = target.get(HEAD_KEY)
        if head is None:
            if checkpoint is not None and previous_head is not None:
                raise RuntimeError("Full checkpoint has no durable metrics boundary; refusing to guess.")
            head = _legacy_head(original, target)
        recovered = _recover_head(original, head)
        if original != recovered or (previous_head is not None and previous_head != head):
            # Archive first, then atomically replace. A failed/repeated recovery
            # cannot destroy evidence or leave a half-truncated authoritative log.
            record = {"reason": "reconcile metrics with restored optimizer state and world cursor",
                      "checkpoint": checkpoint["path"] if checkpoint is not None else None,
                      "original_metrics_hex": original.hex(), "previous_state_head": previous_head,
                      "restored_head": head}
            identity = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
            archive = path.parent / "metrics_recovery" / f"{identity}.json"
            write_json(archive, record)
            if original != recovered:
                _replace_metrics(path, recovered)
            print(f"shellsmith: reconciled metrics; original evidence saved in {archive}", flush=True)
        if path.exists():
            # Even an already-complete write-ahead row might have been interrupted
            # before its append fsync. Establish durability before the next state.
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
            fsync_directory(path.parent)
        target[HEAD_KEY] = head
        state[HEAD_KEY] = head
        return cls(path, recovered)

    def preview_head(self, row: dict[str, Any]) -> dict[str, Any]:
        """Describe this exact next row before its full recovery envelope is saved."""
        data = (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        return {"version": 1, "prefix_bytes": self._size,
                "prefix_sha256": self._digest.hexdigest(), "tail": data.decode("utf-8")}

    def publish(self, *, row: dict[str, Any], state_path: Path, state: dict[str, Any],
                checkpoint: dict[str, Any] | None) -> None:
        head = self.preview_head(row)
        data = head["tail"].encode("utf-8")
        state[HEAD_KEY] = head
        if checkpoint is not None and _position(checkpoint) == _position(state):
            # A skipped update may advance only the cursor (or merely record a
            # halt). Step numbers alone cannot identify its publication boundary.
            checkpoint[HEAD_KEY] = head
        write_json(state_path, state)
        with self.path.open("ab") as handle:
            if handle.write(data) != len(data):
                raise OSError("Short metrics append")
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(self.path.parent)
        self._size += len(data)
        self._digest.update(data)
