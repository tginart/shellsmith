"""Focused campaign-catalog tests for large shellsmith authoring batches."""

from __future__ import annotations

from pathlib import Path

from shellsmith.campaign.draft_catalog import (
    build_catalog,
    campaign_gate,
    render_markdown,
)


def _write(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _spec(*, world: str, embedded: str, reward: str) -> str:
    return f"""
# Example

## World

{world}

## Player

{embedded}

## Reward

{reward}

## Termination conditions

Successful after a supported answer. Failed for destructive behavior. Neutral for
a real blocker. Simulator failures are untrainable.
"""


def test_catalog_classifies_flavors_balance_and_ask_user(tmp_path: Path) -> None:
    _write(
        tmp_path / "legacy_api.md",
        _spec(
            world="A legacy API requires one useful ask_user clarification.",
            embedded="Investigate through terminal tools and ask_user only if needed.",
            reward="Reward a correct recovery.",
        ),
    )
    _write(
        tmp_path / "public_repo_fix.md",
        _spec(
            world=(
                "Use https://github.com/example/repo at "
                "1111111111111111111111111111111111111111; oracle "
                "2222222222222222222222222222222222222222."
            ),
            embedded="Repair the checked-out repository.",
            reward="Emit task_outcomes.public_repo_repair with evidence.",
        ),
    )
    _write(
        tmp_path / "paper_control.md",
        _spec(
            world="Audit class: no-error control. Primary source: arXiv:2601.00001.",
            embedded="Use webFetch to audit the paper and report any supported errors.",
            reward="Emit task_outcomes.paper_audit with evidence.",
        ),
    )

    report = build_catalog(tmp_path)

    assert report["counts"] == {"agentic": 1, "public_repo": 1, "paper": 1}
    assert report["paper_counts"] == {"error-bearing": 0, "no-error control": 1}
    assert report["ask_user"]["count"] == 1
    assert report["problem_count"] == 0
    markdown = render_markdown(report)
    assert "1 error-bearing / 0 no-error controls" not in markdown
    assert "0 error-bearing / 1 no-error controls" in markdown
    assert "[`legacy_api.md`](legacy_api.md)" in markdown


def test_campaign_gate_requires_scale_balance_and_clean_records(tmp_path: Path) -> None:
    _write(
        tmp_path / "paper_bad.md",
        _spec(
            world="Audit class: error-bearing. This mentions arXiv but no outcome contract.",
            embedded="The private oracle says the answer is known.",
            reward="Reward an answer.",
        ),
    )

    report = build_catalog(tmp_path)
    failures = campaign_gate(
        report,
        minimum_per_flavor=1,
        paper_balance_tolerance=0,
    )

    assert any("private provenance" in failure for failure in failures)
    assert any("task_outcomes.paper_audit" in failure for failure in failures)
    assert any("public_repo has 0" in failure for failure in failures)
    assert any("paper error/control imbalance" in failure for failure in failures)
