"""Campaign hygiene for the large ``worlds/drafts/`` pile.

What this is (plain English)
----------------------------
``contracts.py`` parses **one** task. This module walks **many** draft Markdown
files and asks: is the catalog balanced, labeled, and leak-free enough to run a
multi-world campaign?

It is **authoring / QA tooling**, not episode runtime. A single ``smoke`` run
does not need this. Trainers and hero slates often do.

Pipeline
--------
::

    worlds/drafts/*.md
            |
            v
    contracts.load_world_file   # four-section parse
            |
            v
    classify flavor
      - agentic
      - public_repo
      - paper
            |
            v
    checks (examples)
      - private authoring markers must not leak into Player text
      - paper: error-bearing vs no-error control balance
      - ask_user prevalence
      - optional gate: enough worlds per flavor (e.g. 50)
            |
            v
    JSON campaign report (+ CLI exit status)

What it does NOT do
-------------------
    - run Player or GM
    - write episode artifacts
    - decide accepted_for_training (that is acceptance_audit)

Where this sits
---------------
::

    contracts.py       grammar for one .md
    draft_catalog.py   inventory + QA over all drafts   (this file)
    hero_slate.py      fixed subset + GM assignment for trainer
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shellsmith.paths import DRAFTS_DIR
from shellsmith.world_file import (
    WorldFileSpec,
    load_world_file,
)


DEFAULT_DRAFT_ROOT = DRAFTS_DIR
NON_RUNNABLE = {
    "README.md",
    "CATALOG.md",
    "AGENTIC_WORLD_META_SPEC.md",
    "PAPER_AUDIT_META_SPEC.md",
    "PUBLIC_REPO_REPLAY_META_SPEC.md",
    # Recursive simulator-training experiments remain directly runnable, but
    # must not silently enter an immutable hero slate through flavor discovery.
    "meta_world_simulator_click_ansi_001.md",
    "meta_world_simulator_gcs_checksum_001.md",
    "meta_world_simulator_mongodb_resume_001.md",
}
SHA = re.compile(r"\b[0-9a-f]{40}\b")
URL = re.compile(r"https?://", re.I)
PRIVATE_LEAK = re.compile(
    r"\b(?:private oracle|hidden oracle|oracle (?:SHA|commit|diff|answer)|"
    r"world file|Game Master|reward rubric|audit class|error-bearing|"
    r"no-error control|GHSA-[\w-]+|CVE-\d{4}-\d+)\b|"
    r"\b[0-9a-f]{40}\b",
    re.I,
)
PAPER_CLASS = re.compile(
    r"Audit class:\s*(error-bearing|no-error control)\s*\.",
    re.I,
)

Flavor = Literal["agentic", "public_repo", "paper"]
PaperClass = Literal["error-bearing", "no-error control"]


@dataclass(frozen=True)
class DraftRecord:
    """One parsed draft plus campaign-only classification and validation."""

    path: Path
    spec: WorldFileSpec
    flavor: Flavor
    paper_class: PaperClass | None
    uses_ask_user: bool
    problems: tuple[str, ...]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "world_id": self.spec.id,
            "title": self.spec.title,
            "path": str(self.path),
            "flavor": self.flavor,
            "paper_class": self.paper_class,
            "uses_ask_user": self.uses_ask_user,
            "problems": list(self.problems),
        }


def classify(spec: WorldFileSpec, *, path: Path) -> Flavor:
    """Classify from stable authoring signals without changing runtime Markdown."""
    if path.name.startswith("public_repo_") or "task_outcomes.public_repo_repair" in spec.reward:
        return "public_repo"
    if path.name.startswith("paper_") or "task_outcomes.paper_audit" in spec.reward:
        return "paper"
    return "agentic"


def validate_record(path: Path) -> DraftRecord:
    """Parse and validate one runnable draft.

    Main callers:
        ``build_catalog`` and focused tests.

    Why this exists:
        Runtime validation checks only section privacy and shape. Campaign
        validation additionally proves that repository worlds pin immutable
        evidence and that paper controls remain balanced and privately labeled.
    """
    spec = load_world_file(path)
    flavor = classify(spec, path=path)
    problems: list[str] = []
    paper_class: PaperClass | None = None

    if PRIVATE_LEAK.search(spec.player):
        problems.append("Player prompt contains private provenance or grading language")

    if flavor == "public_repo":
        if URL.search(spec.world) is None:
            problems.append("public-repository world has no canonical URL")
        if len(SHA.findall(spec.world)) < 2:
            problems.append("public-repository world must pin base and oracle SHAs")
        if "task_outcomes.public_repo_repair" not in spec.markdown:
            problems.append("public-repository reward lacks task_outcomes.public_repo_repair")

    if flavor == "paper":
        match = PAPER_CLASS.search(spec.world)
        if match is None:
            problems.append("paper world lacks a private Audit class marker")
        else:
            paper_class = match.group(1).casefold()  # type: ignore[assignment]
        if not any(token in spec.world.casefold() for token in ("arxiv", "doi", "openreview")):
            problems.append("paper world lacks primary-source provenance")
        if "task_outcomes.paper_audit" not in spec.markdown:
            problems.append("paper reward lacks task_outcomes.paper_audit")

    combined = f"{spec.world}\n{spec.player}\n{spec.reward}"
    return DraftRecord(
        path=path,
        spec=spec,
        flavor=flavor,
        paper_class=paper_class,
        uses_ask_user="ask_user" in combined,
        problems=tuple(problems),
    )


def build_catalog(root: Path = DEFAULT_DRAFT_ROOT) -> dict[str, Any]:
    """Return the complete deterministic campaign catalog."""
    root = root.expanduser().resolve()
    records = [
        validate_record(path)
        for path in sorted(root.glob("*.md"))
        if path.name not in NON_RUNNABLE
    ]
    counts = {
        flavor: sum(record.flavor == flavor for record in records)
        for flavor in ("agentic", "public_repo", "paper")
    }
    paper_counts = {
        paper_class: sum(record.paper_class == paper_class for record in records)
        for paper_class in ("error-bearing", "no-error control")
    }
    problems = [
        {"world_id": record.spec.id, "problem": problem}
        for record in records
        for problem in record.problems
    ]
    return {
        "schema": "shellsmith.draft_catalog.v1",
        "draft_root": str(root),
        "total": len(records),
        "counts": counts,
        "paper_counts": paper_counts,
        "paper_balance_delta": abs(
            paper_counts["error-bearing"] - paper_counts["no-error control"]
        ),
        "ask_user": {
            "count": sum(record.uses_ask_user for record in records),
            "by_flavor": {
                flavor: sum(
                    record.flavor == flavor and record.uses_ask_user
                    for record in records
                )
                for flavor in ("agentic", "public_repo", "paper")
            },
        },
        "problem_count": len(problems),
        "problems": problems,
        "drafts": [record.to_jsonable() for record in records],
    }


def campaign_gate(
    report: dict[str, Any],
    *,
    minimum_per_flavor: int,
    paper_balance_tolerance: int,
    maximum_ask_user_fraction: float = 0.2,
) -> list[str]:
    """Return explicit reasons the authoring campaign is not ready for rollouts."""
    failures = [
        f"{problem['world_id']}: {problem['problem']}"
        for problem in report["problems"]
    ]
    for flavor, count in report["counts"].items():
        if count < minimum_per_flavor:
            failures.append(
                f"{flavor} has {count} worlds; requires at least {minimum_per_flavor}"
            )
    if report["paper_balance_delta"] > paper_balance_tolerance:
        failures.append(
            "paper error/control imbalance is "
            f"{report['paper_balance_delta']}; tolerance is {paper_balance_tolerance}"
        )
    ask_user_count = int(report["ask_user"]["count"])
    agentic_count = int(report["counts"]["agentic"])
    if ask_user_count < 1:
        failures.append("campaign contains no ask_user world")
    if agentic_count and ask_user_count / agentic_count > maximum_ask_user_fraction:
        failures.append(
            f"ask_user prevalence is {ask_user_count}/{agentic_count}; maximum "
            f"fraction is {maximum_ask_user_fraction}"
        )
    non_agentic_ask_user = sum(
        int(count)
        for flavor, count in report["ask_user"]["by_flavor"].items()
        if flavor != "agentic"
    )
    if non_agentic_ask_user:
        failures.append(
            f"{non_agentic_ask_user} ask_user worlds are outside the agentic flavor"
        )
    return failures


def render_markdown(report: dict[str, Any]) -> str:
    """Render a compact generated index suitable for a 150+ world campaign."""
    lines = [
        "# Shellsmith draft catalog",
        "",
        "Generated by `shellsmith.campaign.draft_catalog`; do not edit by hand.",
        "",
        f"- Total worlds: {report['total']}",
        f"- Agentic: {report['counts']['agentic']}",
        f"- Public repository: {report['counts']['public_repo']}",
        f"- Paper audit: {report['counts']['paper']}",
        "- Paper split: "
        f"{report['paper_counts']['error-bearing']} error-bearing / "
        f"{report['paper_counts']['no-error control']} no-error controls",
        f"- `ask_user` worlds: {report['ask_user']['count']}",
        f"- Validation problems: {report['problem_count']}",
        "",
    ]
    labels = {
        "agentic": "Agentic API/tool/terminal worlds",
        "public_repo": "Public-repository repair worlds",
        "paper": "Recent-paper audit worlds",
    }
    for flavor in ("agentic", "public_repo", "paper"):
        lines.extend(
            [
                f"## {labels[flavor]}",
                "",
                "| Task environment | Paper class | ask_user |",
                "| --- | --- | --- |",
            ]
        )
        for draft in report["drafts"]:
            if draft["flavor"] != flavor:
                continue
            paper_class = draft["paper_class"] or "-"
            ask_user = "yes" if draft["uses_ask_user"] else "no"
            path = Path(draft["path"])
            lines.append(
                f"| [`{path.name}`]({path.name}) | {paper_class} | {ask_user} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Catalog shellsmith drafts.")
    parser.add_argument("--draft-root", type=Path, default=DEFAULT_DRAFT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--campaign-gate", action="store_true")
    parser.add_argument("--minimum-per-flavor", type=int, default=50)
    parser.add_argument("--paper-balance-tolerance", type=int, default=1)
    parser.add_argument("--maximum-ask-user-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    report = build_catalog(args.draft_root)
    failures = (
        campaign_gate(
            report,
            minimum_per_flavor=args.minimum_per_flavor,
            paper_balance_tolerance=args.paper_balance_tolerance,
            maximum_ask_user_fraction=args.maximum_ask_user_fraction,
        )
        if args.campaign_gate
        else [
            f"{problem['world_id']}: {problem['problem']}"
            for problem in report["problems"]
        ]
    )
    report["gate_failures"] = failures
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    if args.markdown_output is not None:
        markdown_output = args.markdown_output.expanduser().resolve()
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(rendered, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
