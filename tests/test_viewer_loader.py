"""Viewer loader rules that need no Flask."""

from __future__ import annotations

import json
from pathlib import Path

from shellsmith.viewer.loader import load_catalog, load_run


def _episode(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    run_dir.joinpath("episode.json").write_text(json.dumps({"episode_id": "e1", "status": "complete"}))


def test_root_behind_a_symlinked_ancestor_loads(tmp_path: Path) -> None:
    """macOS reaches the system temp dir through /var -> /private/var."""
    _episode(tmp_path / "real" / "episodes" / "run1")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    catalog = load_catalog(tmp_path / "link")
    (run,) = catalog["runs"]
    assert run["status"] == "complete" and run["errors"] == []


def test_symlinks_inside_the_root_are_still_refused(tmp_path: Path) -> None:
    _episode(tmp_path / "outside" / "run1")
    (tmp_path / "root").mkdir()
    (tmp_path / "root" / "run1").symlink_to(tmp_path / "outside" / "run1")
    assert load_catalog(tmp_path / "root")["runs"] == []


def test_generation_diagnostics_include_limits_and_rejected_responses(tmp_path):
    run = tmp_path / "run1"
    _episode(run)
    (run / "logs").mkdir()
    diagnostics = {"requests": [{"request_id": "rejected", "protocol_error": "incomplete tool call",
                                 "generation_notices": [{"code": "response_limit_exceeded"}]}]}
    path = run / "logs/generation_diagnostics.json"
    path.write_text(json.dumps(diagnostics))
    record, error = load_run(tmp_path, "run1")
    assert error is None and record["generation_diagnostics"] == diagnostics
    path.write_text("broken json")
    record, _ = load_run(tmp_path, "run1")
    assert any("generation_diagnostics.json" in value for value in record["artifact_errors"])
