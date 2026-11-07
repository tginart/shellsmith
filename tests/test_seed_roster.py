"""Offline integrity checks for the authoring-only seed roster.

The JSON is the structured source of truth; Markdown is its readable companion.
These checks do not fetch repositories, execute candidate code, or certify any
seed for runtime admission. They use unittest so documentation checks also run
without installing the project's optional pytest dependency.
"""

from __future__ import annotations

import json
import re
import unittest
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROSTER_PATH = ROOT / "docs" / "seed_roster.json"
MARKDOWN_PATH = ROOT / "docs" / "SEED_ROSTER.md"
DISCUSSED_REPOSITORIES = {
    "kelseyhightower/envconfig",
    "dbader/schedule",
    "sharkdp/hexyl",
    "jazzband/tablib",
    "simonw/sqlite-utils",
    "pypa/sampleproject",
    "duckdb/duckdb",
    "tidyverse/nycflights13",
    "scikit-learn/scikit-learn",
    "google/or-tools",
    "rfordatascience/tidytuesday",
    "stefanprodan/podinfo",
    "fivethirtyeight/data",
    "fastapi/full-stack-fastapi-template",
    "axllent/mailpit",
    "casey/just",
    "scikit-build/scikit-build-core",
    "lmfit/lmfit-py",
    "jupyter/nbconvert",
    "sharkdp/hyperfine",
    "gitpython-developers/GitPython",
}
QUALIFICATION_FLAGS = {
    "snapshot_materialized",
    "footprint_measured",
    "license_reviewed",
    "dependencies_frozen",
    "fixtures_selected",
    "baseline_tests_run",
    "validators_calibrated",
    "rollouts_run",
}


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class SeedRosterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.roster = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
        cls.markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
        cls.seeds = cls.roster["seeds"]
        cls.by_id = {seed["id"]: seed for seed in cls.seeds}
        cls.categories = {category["id"]: category for category in cls.roster["categories"]}

    def test_roster_has_unique_seeds_and_all_discussed_repositories(self) -> None:
        self.assertEqual(self.roster["schema"], "shellsmith.seed_roster.v1")
        self.assertEqual(len(self.seeds), len(self.by_id))
        repos = {seed["repository"] for seed in self.seeds}
        self.assertEqual(len(repos), len(self.seeds))
        self.assertTrue(DISCUSSED_REPOSITORIES.issubset(repos))
        self.assertEqual(set(self.categories), {f"C{i:02}" for i in range(1, 11)})
        self.assertEqual(len(self.roster["categories"]), len(self.categories))
        self.assertTrue(self.roster["curation_note"])

    def test_metadata_and_candidate_commit_provenance_are_consistent(self) -> None:
        collection = self.roster["metadata_collection"]
        start = _timestamp(collection["started_at_utc"])
        end = _timestamp(collection["completed_at_utc"])
        self.assertIsNotNone(start.tzinfo)
        self.assertLessEqual(start, end)
        self.assertEqual(self.roster["as_of_date"], end.date().isoformat())
        for seed in self.seeds:
            with self.subTest(seed=seed["id"]):
                meta = seed["github_metadata"]
                repo = seed["repository"]
                commit = meta["latest_default_branch_commit"]
                snapshot = seed["candidate_snapshot"]
                self.assertEqual(meta["repository"], repo)
                self.assertEqual(meta["repository_url"], f"https://github.com/{repo}")
                self.assertIs(type(meta["stars"]), int)
                self.assertGreaterEqual(meta["stars"], 0)
                self.assertIs(type(meta["archived"]), bool)
                self.assertIs(type(meta["github_size_kb"]), int)
                self.assertTrue(meta["default_branch"])
                self.assertRegex(commit["sha"], r"^[0-9a-f]{40}$")
                self.assertEqual(snapshot["commit_sha"], commit["sha"])
                self.assertEqual(commit["url"], f"https://github.com/{repo}/commit/{commit['sha']}")
                self.assertEqual(snapshot["commit_url"], commit["url"])
                self.assertLessEqual(_timestamp(commit["committer_date"]), end)
                self.assertEqual(meta["sources"]["repository_api"], f"https://api.github.com/repos/{repo}")
                self.assertEqual(meta["sources"]["commit_api"], f"https://api.github.com/repos/{repo}/commits/{commit['sha']}")
                release = meta["latest_github_release"]
                if release is not None:
                    self.assertFalse(release["prerelease"])
                    self.assertTrue(release["tag"])
                    self.assertLessEqual(_timestamp(release["published_at"]), end)

    def test_research_status_does_not_claim_materialization_or_validation(self) -> None:
        for seed in self.seeds:
            with self.subTest(seed=seed["id"]):
                q = seed["qualification"]
                self.assertEqual(q["status"], "research_candidate")
                self.assertIs(q["metadata_verified"], True)
                for flag in QUALIFICATION_FLAGS:
                    self.assertIs(q[flag], False)
                snapshot = seed["candidate_snapshot"]
                self.assertEqual(snapshot["status"], "reference_only_not_materialized")
                self.assertEqual(snapshot["selected_paths"], [])
                self.assertEqual(snapshot["file_hashes"], {})
        self.assertIn("candidates, not runnable worlds", self.markdown)
        self.assertIn("manual review pending", self.markdown)

    def test_category_roles_are_explicit_and_conditional_work_is_not_hidden(self) -> None:
        seen = set()
        for seed in self.seeds:
            with self.subTest(seed=seed["id"]):
                roles = seed["roles"]
                self.assertTrue(any(role["fit"] == "primary" for role in roles))
                self.assertEqual(len(roles), len({role["category"] for role in roles}))
                self.assertTrue(seed["bounded_scope"])
                for role in roles:
                    category = role["category"]
                    self.assertIn(category, self.categories)
                    self.assertIn(role["fit"], {"primary", "secondary"})
                    self.assertIn(role["support"], {"direct", "paired", "authored_workflow", "authored_overlay"})
                    self.assertTrue(role["world_idea"])
                    self.assertTrue(role["acceptance"])
                    self.assertTrue(role["player_start"])
                    if role["support"] != "direct":
                        self.assertTrue(role["requires"])
                    if category == "C09":
                        self.assertEqual(role["support"], "authored_workflow")
                        self.assertEqual(role["player_start"], "evolving_lab")
                    if category == "C10":
                        self.assertEqual(role["fit"], "secondary")
                        self.assertEqual(role["support"], "authored_overlay")
                        self.assertEqual(role["player_start"], "untrusted_content_overlay")
                    if category == "C01":
                        self.assertEqual(role["player_start"], "empty_workspace")
                    if category == "C02":
                        self.assertEqual(role["player_start"], "existing_checkout")
                    seen.add(category)
        self.assertEqual(seen, set(self.categories))
        self.assertIn("overlay-only", self.markdown)

    def test_recommendations_and_pairings_reference_known_seeds(self) -> None:
        for category in self.categories.values():
            self.assertTrue(category["recommended_seed_ids"])
            for seed_id in category["recommended_seed_ids"]:
                self.assertIn(seed_id, self.by_id)
                self.assertTrue(any(role["category"] == category["id"] for role in self.by_id[seed_id]["roles"]))
        for seed in self.seeds:
            self.assertIn(seed["selection_priority"], {"pilot", "expansion", "conditional"})
            self.assertEqual(len(seed["pair_with"]), len(set(seed["pair_with"])))
            for partner in seed["pair_with"]:
                self.assertIn(partner, self.by_id)
                self.assertNotEqual(partner, seed["id"])

    def test_pilots_cover_categories_without_claiming_built_environments(self) -> None:
        pilots = self.roster["pilot_compositions"]
        self.assertEqual(len(pilots), len({pilot["id"] for pilot in pilots}))
        self.assertEqual({pilot["category"] for pilot in pilots}, set(self.categories))
        for pilot in pilots:
            self.assertEqual(pilot["status"], "planned_not_materialized")
            self.assertTrue(pilot["world_idea"])
            self.assertTrue(pilot["additional_work"])
            self.assertTrue(pilot["seed_ids"])
            for seed_id in pilot["seed_ids"]:
                self.assertIn(seed_id, self.by_id)
                self.assertTrue(any(role["category"] == pilot["category"] for role in self.by_id[seed_id]["roles"]))
        self.assertTrue(self.roster["qualification_gates"])

    def test_markdown_mirrors_inventory_roles_and_provenance(self) -> None:
        lines = self.markdown.splitlines()
        self.assertIn(f"**{len(self.seeds)} seed repositories**", self.markdown)
        self.assertIn(f"**{len(self.categories)} possible task categories**", self.markdown)
        for seed in self.seeds:
            with self.subTest(seed=seed["id"]):
                prefix = f"| [{seed['repository']}](#{seed['id']}) |"
                rows = [line for line in lines if line.startswith(prefix)]
                self.assertEqual(len(rows), 1)
                cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
                meta = seed["github_metadata"]
                self.assertEqual(cells[2], f"{meta['stars']:,}")
                self.assertEqual(cells[3], meta["latest_default_branch_commit"]["committer_date"][:10])
                primary = ", ".join(role["category"] for role in seed["roles"] if role["fit"] == "primary")
                secondary = ", ".join(role["category"] for role in seed["roles"] if role["fit"] == "secondary") or "—"
                self.assertEqual(cells[4], primary)
                self.assertEqual(cells[5], secondary)
                self.assertEqual(cells[6], seed["selection_priority"])
                self.assertIn(f"### {seed['id']}\n", self.markdown)
                self.assertIn(meta["latest_default_branch_commit"]["sha"], self.markdown)
                self.assertIn(seed["bounded_scope"], self.markdown)
                for role in seed["roles"]:
                    self.assertIn(role["world_idea"], self.markdown)
                    self.assertIn(role["acceptance"], self.markdown)
        for category in self.categories:
            rows = [line for line in lines if line.startswith(f"| **{category}** |")]
            self.assertEqual(len(rows), 1)
            cells = [cell.strip() for cell in rows[0].strip("|").split("|")]
            for index, fit in ((2, "primary"), (3, "secondary")):
                seeds = [seed for seed in self.seeds if any(role["category"] == category and role["fit"] == fit for role in seed["roles"])]
                expected = f"{len(seeds)}: " + (", ".join(f"[{seed['id']}](#{seed['id']})" for seed in seeds) or "None")
                self.assertEqual(cells[index], expected)

    def test_markdown_local_links_and_anchors_resolve(self) -> None:
        headings = set(re.findall(r"^### ([a-z0-9_]+)$", self.markdown, re.MULTILINE))
        for target in re.findall(r"\]\(([^)]+)\)", self.markdown):
            if target.startswith("https://"):
                continue
            with self.subTest(target=target):
                if target.startswith("#"):
                    self.assertIn(target[1:], headings)
                else:
                    self.assertTrue((MARKDOWN_PATH.parent / target).resolve().is_file())
        for index_path in (ROOT / "docs" / "README.md", ROOT / "worlds" / "README.md"):
            self.assertIn("SEED_ROSTER.md", index_path.read_text(encoding="utf-8"))

    def test_roster_is_outside_the_runtime_allowlist(self) -> None:
        manifest = ROOT / "worlds" / "manifest.json"
        tasks = json.loads(manifest.read_text(encoding="utf-8"))["tasks"]
        approved = {(manifest.parent / task["path"]).resolve() for task in tasks}
        self.assertNotIn(ROSTER_PATH.resolve(), approved)
        self.assertNotIn(MARKDOWN_PATH.resolve(), approved)
        self.assertEqual(ROSTER_PATH.parent, ROOT / "docs")
        self.assertIn("not consumed by runtime", self.roster["runtime_admission"])


if __name__ == "__main__":
    unittest.main()
