"""
Purpose: Retain recoverable revisions and retire historical payloads without aliases.
Input: An exclusively owned Store and an optional trainer recovery state file.
Output: Durable retirement markers. Original manifests and receipts remain available.
Note: The service owner calls this module only when retention is explicitly enabled.
"""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import digest
from .storage import atomic_json, sync_directory


class CheckpointUnavailable(ValueError):
    """A retired checkpoint keeps its identity, but no longer has weight payloads."""


def plan(store, keep: int, recovery_state: Path | None) -> tuple[set[str], set[str]]:
    """
    Purpose: Select old revisions after all durable recovery references are protected.
    Input: The Store, revision count, and optional atomic trainer state file.
    Output: Candidate references and retained references, grouped by client revision.
    """
    documents = {}
    for folder in store.checkpoints.iterdir():
        if folder.name.startswith(".") or (folder / "retired.json").exists():
            continue
        reference = "studio-state:" + folder.name
        _, documents[reference] = store.manifest(reference)
    groups = {}
    for reference, document in documents.items():
        key = (document["client"], document["state"]["revision"])
        groups.setdefault(key, set()).add(reference)
    protected = {state["checkpoint"] for state in store.states("clients").values()}
    policies = store.states("policies")
    for row in store.db.execute(
        "SELECT body FROM operations WHERE status IN ('queued','running')"
    ):
        body = json.loads(row[0])
        if body.get("checkpoint") in documents:
            protected.add(body["checkpoint"])
        if (
            body.get("policy") in policies
            and policies[body["policy"]]["checkpoint"] in documents
        ):
            protected.add(policies[body["policy"]]["checkpoint"])
    if recovery_state is not None and recovery_state.exists():
        state = json.loads(recovery_state.read_text())
        checkpoints = [c for c in state.get("checkpoints", []) if c["kind"] == "state"]
        if checkpoints:
            protected.add(checkpoints[-1]["path"])
        policy = policies.get(state.get("active_policy_ref"))
        if policy:
            protected.add(policy["checkpoint"])
        pending = state.get("pending_studio_update")
        if pending:
            operation = store.db.execute(
                "SELECT result FROM operations WHERE id=? AND status='succeeded'",
                (pending["id"] + ":optimizer",),
            ).fetchone()
            if operation:
                protected.add(json.loads(operation[0])["checkpoint"])
    for reference in protected:
        if reference not in documents:
            raise CheckpointUnavailable(
                "Recovery reference is unavailable: " + reference
            )
    retained = set(protected)
    for client in {key[0] for key in groups}:
        revisions = sorted((r for c, r in groups if c == client), reverse=True)[:keep]
        for revision in revisions:
            retained.update(groups[(client, revision)])
    for references in groups.values():
        if references & retained:
            retained.update(references)
    return set(documents) - retained, retained


def mark(store, references: set[str], *, retained: set[str]) -> None:
    """
    Purpose: Make each retired reference explicitly unavailable before any unlink.
    Input: Verified candidates and the retained recovery references.
    Output: Fsynced markers beside unchanged manifests. No receipt is rewritten.
    """
    for reference in sorted(references):
        folder, document = store.manifest(reference)
        atomic_json(
            folder / "retired.json",
            {
                "checkpoint": reference,
                "available": False,
                "reason": "rolling_checkpoint_retention",
                "manifest_sha256": digest(document),
                "retained_checkpoints": sorted(retained),
            },
        )


def cleanup(store) -> None:
    """
    Purpose: Complete payload removal after a retirement marker, including crash recovery.
    Input: The exclusively owned Store. Output: Retired folders contain metadata only.
    Note: A missing payload is valid only after its durable retirement marker exists.
    """
    for folder in store.checkpoints.iterdir():
        if folder.name.startswith(".") or not (folder / "retired.json").exists():
            continue
        reference = "studio-state:" + folder.name
        _, document = store.manifest(reference)
        marker = json.loads((folder / "retired.json").read_text())
        if marker.get("checkpoint") != reference or marker.get(
            "manifest_sha256"
        ) != digest(document):
            raise ValueError("Checkpoint retirement marker corruption.")
        if marker.get("available") is not False:
            raise ValueError(
                "Checkpoint retirement marker must deny weight availability."
            )
        allowed = set(document["files"]) | {"manifest.json", "retired.json"}
        if {p.name for p in folder.iterdir()} - allowed:
            raise ValueError("Unexpected retired checkpoint inventory.")
        for name in document["files"]:
            path = folder / name
            if Path(name).name != name or path.is_symlink():
                raise ValueError("Unsafe retired checkpoint payload.")
            path.unlink(missing_ok=True)
        sync_directory(folder)
