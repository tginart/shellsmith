"""Append-only ledger of public facts the Game Master has committed to.

What this is (plain English)
----------------------------
GM is an LLM. Its chat history is not a reliable database: it can forget,
contradict itself, or invent earlier state. This module owns one small JSON file
per episode that records the facts we treat as already "on the record" for Player.

    Think of it as a tiny git log of world-truth, not a full world model:
        boot:  "these facts are true at episode start"
        turn 0: Player saw observation O0; world claimed C0; new facts N0
        turn 1: Player saw observation O1; world claimed C1; new facts N1
        ...

Only the orchestrator / bridge may write the file. GM may *read* it (path
pinned in request.json) and must *report* a consistency review in response.json.
Python then appends a receipt — GM never edits the ledger directly.

On-disk shape (world/consistency_ledger.json)
---------------------------------------------
::

    {
      "schema": "shellsmith.consistency_ledger/v1",
      "revision": 2,                    # == len(entries); starts at 0
      "boot_commitments": [             # from boot response; fixed thereafter
        "repo root is /workspace/app",
        "tests/ exists and is incomplete"
      ],
      "entries": [
        {
          "turn_index": 0,
          "action_id": "call_abc",
          "observation_hash": "<sha256 of Player-facing tool text>",
          "claims": ["ls listed tests/ and src/"],
          "new_commitments": ["src/main.py exists"],
          "cross_checks": ["checked boot: repo root still /workspace/app"]
        },
        { ... turn 1 ... }
      ]
    }

Related shapes GM writes (NOT this file — for orientation)
----------------------------------------------------------------
Per-turn receipt inside ``world/turns/<NNNNNN>/response.json``
(parsed by ``world_artifacts.load_turn_response``; feeds ``append_entry``)::

    "consistency_review": {
      "reviewed_revision": 2,              # must equal ledger.revision
      "observation_claims": ["..."],       # facts this observation asserts
      "new_commitments": ["..."],          # facts newly pinned for later turns
      "conflicts": [],                     # nonempty => untrainable / terminate
      "executed_cross_checks": ["..."]     # checks actually run this turn
    }

Episode-level summary inside ``world/final/response.json``
(parsed by ``world_artifacts.load_final_response``; does NOT append the ledger)::

    "factual_consistency": {
      "passed": true,                      # false if the world contradicted itself
      "contradictions": []                 # human-readable contradiction list
    }

How they relate::

    consistency_ledger.json     = durable append-only store (Python-owned)
    consistency_review          = per-turn form GM fills → becomes a ledger entry
    factual_consistency         = final judge's yes/no over the whole trajectory

Visual lifecycle (one episode)
------------------------------
::

    BOOT
      GM returns private_initial_commitments
           |
           v
      create_ledger(path, boot_commitments=...)
           |
           |  revision=0, entries=[]
           v
      +---------------------------+
      | consistency_ledger.json   |
      +---------------------------+

    EACH PLAYER TOOL TURN (bridge.step)
      1. load_ledger → current revision R, sha256 of file
      2. write request.json pinning path + R + sha256
      3. GM resumes, must review THAT revision
      4. GM writes response.json with consistency_review
      5. world_artifacts validates response.json
      6. if review.reviewed_revision != R → fail
      7. append_entry(expected_revision=R, claims, new_commitments, ...)
           |
           |  revision becomes R+1; one new entry
           v
      +---------------------------+
      | consistency_ledger.json   |  (atomic replace under file lock)
      +---------------------------+
      8. only then Player gets tool_response

    CONFLICTS
      If consistency_review.conflicts is non-empty, bridge marks the rollout
      untrainable / terminate. The ledger still records what was claimed; the
      conflict is a product signal, not a silent rewrite.

Who talks to whom
-----------------
::

    GM (Codex/Claude/Pi)          Python (this module + bridge)
    --------------------------          ------------------------------
    reads ledger path from request      owns create / load / append
    reports consistency_review          validates revision match
    does NOT write the ledger file      atomically appends receipt

API map
-------
    create_ledger   — revision 0 after boot (refuses if file already exists)
    load_ledger     — strict parse; unknown fields rejected
    append_entry    — optimistic concurrency on expected_revision;
                      identical retry is a no-op; same turn/action with
                      different content raises LedgerAppendConflictError
    observation_sha256 — hash of the exact Player-facing observation string

Important invariants
--------------------
    - Schema id is exact; unknown JSON fields are rejected.
    - ``revision`` equals ``len(entries)``.
    - A turn_index and an action_id may each appear in at most one entry.
    - Replaying an identical append is harmless (appended=False).
    - Reusing turn/action identity with different content is an error.
    - A genuinely new append must pass ``expected_revision == current revision``.
    - Writes: temp file → fsync → atomic replace; readers always see a full doc.

Why this exists
---------------
    Natural-language simulator context is not a database. This ledger gives the
    orchestrator a small, auditable record of facts that later observations must
    respect, without coupling persistence to prompt text or bridge internals.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping


LEDGER_SCHEMA = "shellsmith.consistency_ledger/v1"
_ROOT_KEYS = frozenset({"schema", "revision", "boot_commitments", "entries"})
_ENTRY_KEYS = frozenset(
    {
        "turn_index",
        "action_id",
        "observation_hash",
        "claims",
        "new_commitments",
        "cross_checks",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ConsistencyLedgerError(RuntimeError):
    """Base class for ledger contract and persistence failures."""


class LedgerValidationError(ConsistencyLedgerError):
    """The stored document does not satisfy the exact ledger contract."""


class LedgerRevisionError(ConsistencyLedgerError):
    """A new append was attempted against a stale or future revision."""


class LedgerAppendConflictError(ConsistencyLedgerError):
    """A committed turn or action was reused with different content."""


@dataclass(frozen=True)
class ConsistencyEntry:
    """One immutable public-observation commitment."""

    turn_index: int
    action_id: str
    observation_hash: str
    claims: tuple[str, ...]
    new_commitments: tuple[str, ...]
    cross_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the exact JSON representation used for equality and persistence."""

        return {
            "turn_index": self.turn_index,
            "action_id": self.action_id,
            "observation_hash": self.observation_hash,
            "claims": list(self.claims),
            "new_commitments": list(self.new_commitments),
            "cross_checks": list(self.cross_checks),
        }


@dataclass(frozen=True)
class ConsistencyLedger:
    """A fully validated snapshot of the append-only ledger."""

    revision: int
    boot_commitments: tuple[str, ...]
    entries: tuple[ConsistencyEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete canonical JSON document."""

        return {
            "schema": LEDGER_SCHEMA,
            "revision": self.revision,
            "boot_commitments": list(self.boot_commitments),
            "entries": [entry.to_dict() for entry in self.entries],
        }


@dataclass(frozen=True)
class AppendResult:
    """Describe whether an append created a revision or replayed one safely."""

    ledger: ConsistencyLedger
    entry: ConsistencyEntry
    appended: bool


def observation_sha256(observation: str) -> str:
    """Hash the exact UTF-8 public observation shown to the policy."""

    if not isinstance(observation, str):
        raise TypeError("observation must be a string")
    return hashlib.sha256(observation.encode("utf-8")).hexdigest()


def create_ledger(
    path: Path | str,
    *,
    boot_commitments: Iterable[str],
) -> ConsistencyLedger:
    """Create revision zero without replacing an existing ledger.

    Main callers:
        Episode boot orchestration, after the simulator returns its private initial
        commitments.

    Why this exists:
        Creation is deliberately distinct from append so a restarted process cannot
        silently erase a prior episode's factual history.
    """

    ledger_path = Path(path)
    if not ledger_path.parent.is_dir():
        raise FileNotFoundError(
            f"ledger parent directory does not exist: {ledger_path.parent}"
        )
    commitments = _gm_string_sequence(
        list(boot_commitments), "boot_commitments"
    )
    ledger = ConsistencyLedger(revision=0, boot_commitments=commitments, entries=())
    with _exclusive_lock(ledger_path):
        _atomic_write_new(ledger_path, ledger.to_dict())
    return ledger


def load_ledger(path: Path | str) -> ConsistencyLedger:
    """Load and strictly validate the complete ledger document."""

    ledger_path = Path(path)
    try:
        raw = json.loads(
            ledger_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except json.JSONDecodeError as exc:
        raise LedgerValidationError(
            f"invalid ledger JSON at {ledger_path}: {exc}"
        ) from exc
    return _parse_ledger(raw)


def append_entry(
    path: Path | str,
    *,
    expected_revision: int,
    turn_index: int,
    action_id: str,
    observation: str,
    claims: Iterable[str],
    new_commitments: Iterable[str],
    cross_checks: Iterable[str],
) -> AppendResult:
    """Append one observation, or recognize an identical retry.

    Main callers:
        The orchestrator immediately before forwarding a validated world response.

    A new identity requires ``expected_revision`` to equal the current revision.
    An exact retry returns the already committed snapshot even when its caller still
    holds the pre-append revision; this is the narrow exception that makes recovery
    from an uncertain write result safe. Any content change for an existing turn or
    action raises a conflict instead of creating a second history.
    """

    if not _is_plain_int(expected_revision) or expected_revision < 0:
        raise LedgerRevisionError("expected_revision must be a non-negative integer")
    entry = _make_entry(
        turn_index=turn_index,
        action_id=action_id,
        observation=observation,
        claims=claims,
        new_commitments=new_commitments,
        cross_checks=cross_checks,
    )
    ledger_path = Path(path)
    with _exclusive_lock(ledger_path):
        return _append_entry_locked(
            ledger_path,
            expected_revision=expected_revision,
            entry=entry,
        )


def _append_entry_locked(
    ledger_path: Path,
    *,
    expected_revision: int,
    entry: ConsistencyEntry,
) -> AppendResult:
    """Perform the read-check-write cycle while the sidecar lock is held."""

    ledger = load_ledger(ledger_path)

    identity_matches = [
        current
        for current in ledger.entries
        if current.turn_index == entry.turn_index
        or current.action_id == entry.action_id
    ]
    if identity_matches:
        if len(identity_matches) == 1 and identity_matches[0] == entry:
            return AppendResult(
                ledger=ledger, entry=identity_matches[0], appended=False
            )
        raise LedgerAppendConflictError(
            "turn_index or action_id is already committed with different content"
        )

    if expected_revision != ledger.revision:
        raise LedgerRevisionError(
            f"expected revision {expected_revision}, found {ledger.revision}"
        )

    updated = ConsistencyLedger(
        revision=ledger.revision + 1,
        boot_commitments=ledger.boot_commitments,
        entries=(*ledger.entries, entry),
    )
    _atomic_replace(ledger_path, updated.to_dict())
    return AppendResult(ledger=updated, entry=entry, appended=True)


def _make_entry(
    *,
    turn_index: int,
    action_id: str,
    observation: str,
    claims: Iterable[str],
    new_commitments: Iterable[str],
    cross_checks: Iterable[str],
) -> ConsistencyEntry:
    if not _is_plain_int(turn_index) or turn_index < 0:
        raise LedgerValidationError("turn_index must be a non-negative integer")
    if not isinstance(action_id, str) or not action_id.strip():
        raise LedgerValidationError("action_id must be a non-empty string")
    return ConsistencyEntry(
        turn_index=turn_index,
        action_id=action_id,
        observation_hash=observation_sha256(observation),
        claims=_gm_string_sequence(list(claims), "claims"),
        new_commitments=_gm_string_sequence(
            list(new_commitments), "new_commitments"
        ),
        cross_checks=_gm_string_sequence(list(cross_checks), "cross_checks"),
    )


def _parse_ledger(raw: Any) -> ConsistencyLedger:
    _require_exact_mapping(raw, _ROOT_KEYS, "ledger")
    if raw["schema"] != LEDGER_SCHEMA:
        raise LedgerValidationError(
            f"unsupported ledger schema: {raw['schema']!r}; expected {LEDGER_SCHEMA!r}"
        )
    revision = raw["revision"]
    if not _is_plain_int(revision) or revision < 0:
        raise LedgerValidationError("revision must be a non-negative integer")
    boot = _validate_string_sequence(raw["boot_commitments"], "boot_commitments")
    if not isinstance(raw["entries"], list):
        raise LedgerValidationError("entries must be an array")

    entries = tuple(
        _parse_entry(value, index=index) for index, value in enumerate(raw["entries"])
    )
    if revision != len(entries):
        raise LedgerValidationError(
            f"revision {revision} does not equal entry count {len(entries)}"
        )
    turns = [entry.turn_index for entry in entries]
    actions = [entry.action_id for entry in entries]
    if len(turns) != len(set(turns)):
        raise LedgerValidationError("entry turn_index values must be unique")
    if len(actions) != len(set(actions)):
        raise LedgerValidationError("entry action_id values must be unique")
    if turns != sorted(turns):
        raise LedgerValidationError("entry turn_index values must be increasing")
    return ConsistencyLedger(revision=revision, boot_commitments=boot, entries=entries)


def _parse_entry(raw: Any, *, index: int) -> ConsistencyEntry:
    context = f"entries[{index}]"
    _require_exact_mapping(raw, _ENTRY_KEYS, context)
    turn_index = raw["turn_index"]
    if not _is_plain_int(turn_index) or turn_index < 0:
        raise LedgerValidationError(
            f"{context}.turn_index must be non-negative integer"
        )
    action_id = raw["action_id"]
    if not isinstance(action_id, str) or not action_id.strip():
        raise LedgerValidationError(f"{context}.action_id must be a non-empty string")
    observation_hash = raw["observation_hash"]
    if not isinstance(observation_hash, str) or not _SHA256_RE.fullmatch(
        observation_hash
    ):
        raise LedgerValidationError(
            f"{context}.observation_hash must be lowercase SHA-256"
        )
    return ConsistencyEntry(
        turn_index=turn_index,
        action_id=action_id,
        observation_hash=observation_hash,
        claims=_validate_string_sequence(raw["claims"], f"{context}.claims"),
        new_commitments=_validate_string_sequence(
            raw["new_commitments"], f"{context}.new_commitments"
        ),
        cross_checks=_validate_string_sequence(
            raw["cross_checks"], f"{context}.cross_checks"
        ),
    )


def drop_blank_gm_slots(raw: Any) -> Any:
    """Treat empty/whitespace strings in a GM list as absent.

    Why this exists:
        The advertised JSON schema allows ``[""]``. Create/append already drop
        those slots. The acceptance audit must use the same filter or a valid
        episode looks inconsistent and ``accepted_for_training`` goes false.

    Main callers:
        ``create_ledger`` / ``append_entry`` via ``_gm_string_sequence``,
        and ``acceptance_audit._consistency_ledger_report``.
    """

    if not isinstance(raw, list):
        return raw
    if any(not isinstance(value, str) for value in raw):
        return raw
    return [value for value in raw if value.strip()]


def check_gm_lists(**lists: Any) -> None:
    """Apply the ledger's list rules to GM-reported lists before they reach the ledger.

    Each keyword names the field as the GM wrote it (``observation_claims``,
    ``private_initial_commitments``, ...), so the error tells the GM exactly
    what to fix: blank slots are ignored, remaining strings must be unique.

    Why this exists:
        ``append_entry`` / ``create_ledger`` enforce these rules, but they run
        after the GM's same-thread repair loop. A duplicate claim found there
        ended the whole episode (about 9% of canary episodes). Checking inside
        the repair loop sends the GM a repair prompt instead; nothing is
        deduplicated on the GM's behalf.

    Main callers:
        ``bridge.WorldBridge`` (turn stage 5) and the orchestrator's boot repair loop.
    """

    for name, raw in lists.items():
        _gm_string_sequence(raw, name)


def _gm_string_sequence(raw: Any, context: str) -> tuple[str, ...]:
    """Drop blank slots from a GM-reported list, then store the rest.

    The advertised turn schema allows empty strings. Create/append ingest that
    fill as "no claim" so the sandbox does not crash. Load/parse stays strict
    via ``_validate_string_sequence`` so a persisted ``[""]`` is still corrupt.
    """

    if not isinstance(raw, list):
        raise LedgerValidationError(f"{context} must be an array")
    if any(not isinstance(value, str) for value in raw):
        raise LedgerValidationError(f"{context} must contain only strings")
    return _validate_string_sequence(drop_blank_gm_slots(raw), context)


def _validate_string_sequence(raw: Any, context: str) -> tuple[str, ...]:
    """Strict on-disk rule: arrays of unique non-empty strings only."""

    if not isinstance(raw, list):
        raise LedgerValidationError(f"{context} must be an array")
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise LedgerValidationError(f"{context} must contain only non-empty strings")
    if len(raw) != len(set(raw)):
        raise LedgerValidationError(f"{context} must not contain duplicate strings")
    return tuple(raw)


def _require_exact_mapping(
    raw: Any, expected_keys: frozenset[str], context: str
) -> None:
    if not isinstance(raw, Mapping):
        raise LedgerValidationError(f"{context} must be an object")
    keys = frozenset(raw.keys())
    if keys != expected_keys:
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        raise LedgerValidationError(
            f"{context} has wrong fields; missing={missing}, extra={extra}"
        )


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise LedgerValidationError(f"duplicate JSON object key: {key!r}")
        document[key] = value
    return document


def _atomic_write_new(path: Path, document: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _write_document(handle, document)
        os.link(temporary_path, path)
        temporary_path.unlink()
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    _fsync_directory(path.parent)


def _atomic_replace(path: Path, document: dict[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            _write_document(handle, document)
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_document(handle: Any, document: dict[str, Any]) -> None:
    json.dump(document, handle, indent=2, sort_keys=True, ensure_ascii=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(ledger_path: Path):
    """Serialize revision checks across processes without locking a replaced inode."""

    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
