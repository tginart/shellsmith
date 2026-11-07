"""Focused contract tests for the orchestrator-owned consistency ledger."""

from __future__ import annotations

import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

from shellsmith.episode.consistency_ledger import (
    LEDGER_SCHEMA,
    LedgerAppendConflictError,
    LedgerRevisionError,
    LedgerValidationError,
    append_entry,
    create_ledger,
    load_ledger,
    observation_sha256,
)


def _append(path: Path, *, expected_revision: int = 0, observation: str = "rows=10"):
    return append_entry(
        path,
        expected_revision=expected_revision,
        turn_index=2,
        action_id="call-read-counts",
        observation=observation,
        claims=["The run landed 10 rows."],
        new_commitments=["run-10 landed exactly 10 rows"],
        cross_checks=["Compared run-10 with the boot chronology."],
    )


def test_create_load_and_append_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "consistency_ledger.json"
    created = create_ledger(path, boot_commitments=["Release v2 starts on July 10."])

    assert created.revision == 0
    assert path.stat().st_mode & 0o777 == 0o600
    result = _append(path)
    loaded = load_ledger(path)

    assert result.appended is True
    assert loaded == result.ledger
    assert loaded.revision == 1
    assert loaded.entries[0].observation_hash == observation_sha256("rows=10")
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == LEDGER_SCHEMA


def test_identical_retry_is_idempotent_even_with_pre_append_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])
    first = _append(path, expected_revision=0)
    before = path.read_bytes()
    retry = _append(path, expected_revision=0)

    assert first.appended is True
    assert retry.appended is False
    assert retry.ledger.revision == 1
    assert path.read_bytes() == before


def test_reused_identity_with_changed_content_is_a_conflict(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])
    _append(path)

    with pytest.raises(LedgerAppendConflictError):
        _append(path, expected_revision=1, observation="rows=11")


def test_new_append_requires_exact_current_revision(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])
    _append(path)

    with pytest.raises(LedgerRevisionError, match="expected revision 0, found 1"):
        append_entry(
            path,
            expected_revision=0,
            turn_index=3,
            action_id="call-next",
            observation="next",
            claims=[],
            new_commitments=[],
            cross_checks=[],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda raw: raw.update(extra=True),
        lambda raw: raw.update(schema="future/v2"),
        lambda raw: raw.update(revision=True),
        lambda raw: raw.update(revision=1),
        lambda raw: raw.update(boot_commitments=[""]),
        lambda raw: raw.update(entries="not-an-array"),
    ],
)
def test_load_rejects_malformed_or_unknown_root_data(tmp_path: Path, mutation) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])
    raw = json.loads(path.read_text(encoding="utf-8"))
    mutation(raw)
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(LedgerValidationError):
        load_ledger(path)


def test_load_rejects_duplicate_identity_and_bad_entry_hash(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])
    _append(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    duplicate = dict(raw["entries"][0])
    duplicate["observation_hash"] = "x" * 64
    raw["entries"].append(duplicate)
    raw["revision"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(LedgerValidationError, match="lowercase SHA-256"):
        load_ledger(path)


def test_load_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text(
        '{"schema":"shellsmith.consistency_ledger/v1",'
        '"revision":0,"revision":0,"boot_commitments":[],"entries":[]}',
        encoding="utf-8",
    )

    with pytest.raises(LedgerValidationError, match="duplicate JSON object key"):
        load_ledger(path)


def test_create_refuses_to_overwrite_and_missing_parent(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])

    with pytest.raises(FileExistsError):
        create_ledger(path, boot_commitments=[])
    with pytest.raises(FileNotFoundError):
        create_ledger(tmp_path / "missing" / "ledger.json", boot_commitments=[])


def test_revision_check_is_serialized_across_concurrent_appenders(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ledger.json"
    create_ledger(path, boot_commitments=[])

    def append(turn_index: int):
        return append_entry(
            path,
            expected_revision=0,
            turn_index=turn_index,
            action_id=f"call-{turn_index}",
            observation=f"observation-{turn_index}",
            claims=[],
            new_commitments=[],
            cross_checks=[],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(append, index) for index in (1, 2)]
    results = []
    errors = []
    for future in futures:
        try:
            results.append(future.result())
        except LedgerRevisionError as exc:
            errors.append(exc)

    assert len(results) == 1
    assert len(errors) == 1
    assert load_ledger(path).revision == 1


def test_check_gm_lists_names_the_gm_field_and_ignores_blank_slots() -> None:
    from shellsmith.episode.consistency_ledger import LedgerValidationError, check_gm_lists

    check_gm_lists(private_initial_commitments=["a", "", "b"], new_commitments=[])
    with pytest.raises(LedgerValidationError, match="private_initial_commitments must not contain duplicate"):
        check_gm_lists(private_initial_commitments=["a", "a"])
