"""Drive promoted seed files, canonical loading, projection, and coverage CLI.

No seed code, environments, GM episodes, or network requests are executed.
Negative fixtures copy the shipped files and corrupt their input contracts; the
coverage checker and canonical parser themselves are never mocked or copied.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shellsmith.campaign.seed_world_catalog import (
    build_seed_world_coverage,
    load_seed_world_specs,
)
from shellsmith.world_file import WorldFileSpec, load_world_file

ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "worlds" / "seeds"
INDEX = COLLECTION / "coverage.json"
ROSTER = ROOT / "docs" / "seed_roster.json"
BASELINE = ROOT / "tests" / "fixtures" / "seed_worlds_existing_hashes.json"
CONSUMER = ROOT / "tests" / "fixtures" / "seed_world_consumer.py"


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class SeedWorldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = json.loads(INDEX.read_text(encoding="utf-8"))
        cls.roster = json.loads(ROSTER.read_text(encoding="utf-8"))

    def setUp(self) -> None:
        # The goal runner supplies a private artifact directory. Ordinary local
        # runs retain the system's normal temporary-directory default.
        parent = os.environ.get("SHELLSMITH_TEST_SCRATCH")
        self.temp = tempfile.TemporaryDirectory(prefix="seed-world-test-", dir=parent)
        self.addCleanup(self.temp.cleanup)
        fixture_worlds = Path(self.temp.name) / "worlds"
        shutil.copytree(ROOT / "worlds", fixture_worlds)
        self.fixture = fixture_worlds / "seeds"
        self.fixture_manifest = fixture_worlds / "manifest.json"
        self.fixture_index = self.fixture / "coverage.json"
        self.mutable = json.loads(self.fixture_index.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.fixture_index.write_text(json.dumps(self.mutable, indent=2) + "\n", encoding="utf-8")

    def _report(self) -> dict:
        self._save()
        return build_seed_world_coverage(self.fixture_index, ROSTER)

    def _refresh_hashes(self, row: dict) -> None:
        # Preserve valid identities while testing a different validation rule.
        # Hashes come from the actual canonical loader, not a substitute parser.
        spec = load_world_file(self.fixture / row["path"])
        row["content_sha256"] = spec.content_hash()
        row["player_sha256"] = hashlib.sha256(spec.player_projection().encode()).hexdigest()

    def test_actual_contracts_load_with_only_player_projected(self) -> None:
        specs = load_seed_world_specs(INDEX, ROSTER)
        self.assertEqual([spec.id for spec in specs], [row["id"] for row in self.index["worlds"]])
        for spec, row in zip(specs, self.index["worlds"], strict=True):
            with self.subTest(world=spec.id):
                self.assertIsInstance(spec, WorldFileSpec)
                self.assertEqual(spec.title, row["title"])
                self.assertTrue(all((spec.world, spec.player, spec.reward, spec.termination_conditions)))
                self.assertEqual(spec.content_hash(), row["content_sha256"])
                player = spec.player_projection()
                self.assertEqual(player, spec.player)
                self.assertEqual(hashlib.sha256(player.encode()).hexdigest(), row["player_sha256"])
                for use in row["seed_uses"]:
                    self.assertNotIn(use["commit_sha"], player)
                    self.assertNotIn(use["source_url"], player)
                    self.assertNotIn(use["role_evidence"], player)
                    self.assertNotIn(" ".join(use["source_scope"].split()), " ".join(player.split()))
                self.assertNotIn("## World", player)
                self.assertNotIn("## Reward", player)
                for ending in ("Successful", "Failed", "Neutral", "Simulator integrity"):
                    self.assertIn(ending, spec.termination_conditions)
                self.assertIn("untrainable", spec.termination_conditions.lower())

    def test_shipped_index_covers_actual_seeds_categories_and_size_range(self) -> None:
        report = build_seed_world_coverage(INDEX, ROSTER)
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["world_count"], len(self.index["worlds"]))
        self.assertEqual(set(report["covered_seed_ids"]), {seed["id"] for seed in self.roster["seeds"]})
        self.assertEqual(set(report["covered_category_ids"]), {category["id"] for category in self.roster["categories"]})
        for field in ("missing_seed_ids", "unknown_seed_ids", "missing_category_ids", "unknown_category_ids", "unindexed_world_paths"):
            self.assertEqual(report[field], [])
        self.assertEqual(set(report["task_sizes"]), {"bounded_subset", "full_small_project"})
        self.assertTrue(all(count > 0 for count in report["task_sizes"].values()))

    def test_existing_contracts_and_promoted_seed_paths_match_reviewed_baseline(self) -> None:
        baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
        for relative, expected in baseline.items():
            with self.subTest(path=relative):
                self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        manifest = ROOT / "worlds" / "manifest.json"
        tasks = json.loads(manifest.read_text(encoding="utf-8"))["tasks"]
        existing = [load_world_file(manifest.parent / row["path"]) for row in tasks]
        self.assertEqual([spec.id for spec in existing], [row["id"] for row in tasks])
        promoted = {row["id"]: (manifest.parent / row["path"]).resolve() for row in tasks}
        for row in self.index["worlds"]:
            self.assertEqual(promoted[row["id"]], (COLLECTION / row["path"]).resolve())
        self.assertEqual(len(existing), len({spec.id for spec in existing}))
        self.assertEqual(self.index["status"], "promoted")
        self.assertTrue(self.index["runtime_promoted"])
        self.assertEqual(self.index["execution_qualification"], "deferred")

    def test_two_fresh_consumers_return_expected_hashes_and_public_prompts(self) -> None:
        results = []
        for _ in range(2):
            process = subprocess.run(
                [sys.executable, str(CONSUMER), "--index", str(INDEX)],
                cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            results.append(json.loads(process.stdout))
        self.assertNotEqual(results[0]["consumer_pid"], results[1]["consumer_pid"])
        self.assertEqual(results[0]["worlds"], results[1]["worlds"])
        for result in results:
            self.assertEqual([row["id"] for row in result["worlds"]], [row["id"] for row in self.index["worlds"]])
            for actual, expected in zip(result["worlds"], self.index["worlds"], strict=True):
                self.assertEqual(actual["content_sha256"], expected["content_sha256"])
                self.assertEqual(actual["player_sha256"], expected["player_sha256"])
                self.assertEqual(set(actual), {"id", "content_sha256", "player_sha256", "player"})

    def test_documented_cli_drives_the_real_coverage_entry_point(self) -> None:
        process = subprocess.run(
            [sys.executable, "-m", "shellsmith.campaign.seed_world_catalog", "--index", str(INDEX), "--roster", str(ROSTER)],
            cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), build_seed_world_coverage(INDEX, ROSTER))

    def test_missing_seed_and_unindexed_actual_file_are_detected(self) -> None:
        removed = self.mutable["worlds"].pop(0)
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertIn("envconfig", report["missing_seed_ids"])
        self.assertIn(removed["path"], report["unindexed_world_paths"])

    def test_unknown_seed_and_category_are_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        row["seed_uses"][0]["seed_id"] = "not_in_roster"
        row["category_ids"] = ["C99"]
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertIn("not_in_roster", report["unknown_seed_ids"])
        self.assertIn("C99", report["unknown_category_ids"])

    def test_repo_mention_without_actual_private_role_is_not_coverage(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        evidence = row["seed_uses"][0]["role_evidence"]
        text = path.read_text(encoding="utf-8")
        self.assertIn(evidence, text)
        path.write_text(text.replace(evidence, "", 1), encoding="utf-8")
        self._refresh_hashes(row)
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("seed role absent" in issue for issue in report["issues"]))

    def test_provenance_and_selected_scope_must_match_actual_private_world(self) -> None:
        use = self.mutable["worlds"][0]["seed_uses"][0]
        use["source_scope"] = "uninspected scope that does not occur in the contract"
        use["commit_sha"] = "0" * 40
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("selected scope absent" in issue for issue in report["issues"]))
        self.assertTrue(any("provenance differs" in issue for issue in report["issues"]))

    def test_other_roster_seed_provenance_leaks_are_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        used = {use["seed_id"] for use in row["seed_uses"]}
        seed = next(seed for seed in self.roster["seeds"] if seed["id"] not in used)
        path = self.fixture / row["path"]
        original = path.read_text(encoding="utf-8")
        for leaked in ("https://github.com/" + seed["repository"], seed["candidate_snapshot"]["commit_sha"]):
            with self.subTest(leaked=leaked):
                path.write_text(original.replace("## Player\n", "## Player\n\n" + leaked + "\n", 1), encoding="utf-8")
                self._refresh_hashes(row)
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("leaks into Player" in issue for issue in report["issues"]))

    def test_bare_repository_provenance_leaks_are_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        original = path.read_text(encoding="utf-8")
        for seed in self.roster["seeds"]:
            with self.subTest(seed=seed["id"]):
                leaked = seed["repository"]
                path.write_text(original.replace("## Player\n", "## Player\n\n" + leaked + "\n", 1), encoding="utf-8")
                self._refresh_hashes(row)
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("private repository leaks into Player" in issue for issue in report["issues"]))
                with self.assertRaisesRegex(ValueError, "repository leaks into Player"):
                    load_seed_world_specs(self.fixture_index, ROSTER)

    def test_mixed_case_markdown_extensions_are_detected_as_unindexed(self) -> None:
        original = self.fixture / self.mutable["worlds"][0]["path"]
        for suffix in (".MD", ".Md", ".mD"):
            with self.subTest(suffix=suffix):
                orphan = self.fixture / ("orphan" + suffix)
                shutil.copy2(original, orphan)
                self.assertEqual(load_world_file(orphan).id, "orphan")
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertIn(orphan.name, report["unindexed_world_paths"])
                orphan.unlink()

    def test_case_altered_provenance_is_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        original = path.read_text(encoding="utf-8")
        for seed in self.roster["seeds"]:
            for value in (seed["repository"], "https://github.com/" + seed["repository"], seed["candidate_snapshot"]["commit_sha"]):
                with self.subTest(seed=seed["id"], value=value):
                    path.write_text(original.replace("## Player\n", "## Player\n\n" + value.upper() + "\n", 1), encoding="utf-8")
                    self._refresh_hashes(row)
                    report = self._report()
                    self.assertFalse(report["valid"])
                    self.assertTrue(any("leaks into Player" in issue for issue in report["issues"]))

    def test_abbreviated_commit_provenance_is_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        original = path.read_text(encoding="utf-8")
        for seed in self.roster["seeds"]:
            for length in (12, 16, 39):
                with self.subTest(seed=seed["id"], length=length):
                    leaked = seed["candidate_snapshot"]["commit_sha"][:length].upper()
                    path.write_text(original.replace("## Player\n", "## Player\n\nRevision `" + leaked + "`.\n", 1), encoding="utf-8")
                    self._refresh_hashes(row)
                    report = self._report()
                    self.assertFalse(report["valid"])
                    self.assertTrue(any("abbreviated commit leaks" in issue for issue in report["issues"]))

    def test_case_altered_private_role_and_scope_are_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        original = path.read_text(encoding="utf-8")
        for field, message in (("role_evidence", "private seed role leaks"), ("source_scope", "private selected scope leaks")):
            with self.subTest(field=field):
                leaked = "  ".join(row["seed_uses"][0][field].upper().split())
                path.write_text(original.replace("## Player\n", "## Player\n\n" + leaked + "\n", 1), encoding="utf-8")
                self._refresh_hashes(row)
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any(message in issue for issue in report["issues"]))

    def test_unindexed_private_seed_declarations_are_rejected(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        original = path.read_text(encoding="utf-8")
        used = {use["seed_id"] for use in row["seed_uses"]}
        other = next(seed for seed in self.roster["seeds"] if seed["id"] not in used)
        for seed_id in (other["id"], "unknown_seed"):
            with self.subTest(seed=seed_id):
                declaration = f"Seed `{seed_id}`: additional private source."
                path.write_text(original.replace("## Player\n", declaration + "\n\n## Player\n", 1), encoding="utf-8")
                self._refresh_hashes(row)
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("seed declarations differ" in issue for issue in report["issues"]))

    def test_indexed_title_must_match_contract(self) -> None:
        row = self.mutable["worlds"][0]
        for value in ("Stale title", self.mutable["worlds"][1]["title"], "", None):
            with self.subTest(title=value):
                row["title"] = value
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("title" in issue for issue in report["issues"]))
        del row["title"]
        self.assertFalse(self._report()["valid"])

    def test_role_and_scope_cannot_be_swapped_between_seeds(self) -> None:
        for row in self.mutable["worlds"]:
            if len(row["seed_uses"]) < 2:
                continue
            first, second = row["seed_uses"][:2]
            for field, message in (("role_evidence", "seed role absent"), ("source_scope", "selected scope absent")):
                with self.subTest(world=row["id"], field=field):
                    first[field], second[field] = second[field], first[field]
                    report = self._report()
                    self.assertFalse(report["valid"])
                    self.assertTrue(any(message in issue for issue in report["issues"]))
                    first[field], second[field] = second[field], first[field]

    def test_swapped_classifications_are_rejected_without_hash_changes(self) -> None:
        for field in ("category_ids", "task_size"):
            with self.subTest(field=field):
                first = self.mutable["worlds"][0]
                second = next(row for row in self.mutable["worlds"] if row[field] != first[field])
                first[field], second[field] = second[field], first[field]
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("classification differs" in issue for issue in report["issues"]))
                first[field], second[field] = second[field], first[field]

    def test_loader_returns_the_snapshot_it_validated(self) -> None:
        row = self.mutable["worlds"][0]
        target = (self.fixture / row["path"]).resolve()
        original = load_world_file(target)
        reads = []

        def load_then_edit(path: Path) -> WorldFileSpec:
            spec = load_world_file(path)
            reads.append(path)
            if path == target and reads.count(target) == 1:
                target.write_text(spec.markdown.replace("## Player\n", "## Player\n\nChanged after read.\n", 1), encoding="utf-8")
            return spec

        # Only the filesystem timing is controlled; every read uses the real parser.
        with patch("shellsmith.campaign.seed_world_catalog.load_world_file", side_effect=load_then_edit):
            specs = load_seed_world_specs(self.fixture_index, ROSTER)
        self.assertEqual(specs[0], original)
        self.assertEqual(reads.count(target), 1)
        self.assertNotEqual(load_world_file(target), original)

    def test_indexed_source_urls_must_match_roster(self) -> None:
        use = self.mutable["worlds"][0]["seed_uses"][0]
        for field in ("source_url", "commit_url"):
            original = use[field]
            for value in ("https://github.com/incorrect/repository", "", None):
                with self.subTest(field=field, value=value):
                    use[field] = value
                    report = self._report()
                    self.assertFalse(report["valid"])
                    self.assertTrue(any("provenance differs" in issue for issue in report["issues"]))
                    with self.assertRaisesRegex(ValueError, "provenance differs"):
                        load_seed_world_specs(self.fixture_index, ROSTER)
            with self.subTest(field=field, missing=True):
                del use[field]
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("provenance differs" in issue for issue in report["issues"]))
            use[field] = original
        self.assertTrue(self._report()["valid"])

    def test_private_provenance_in_actual_player_is_rejected_even_with_fresh_hashes(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        leaked = row["seed_uses"][0]["commit_sha"]
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("## Player\n", "## Player\n\n" + leaked + "\n", 1), encoding="utf-8")
        self._refresh_hashes(row)
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("leaks into Player" in issue for issue in report["issues"]))

    def test_private_scope_leak_is_rejected_after_valid_hash_refresh(self) -> None:
        self.assertTrue(build_seed_world_coverage(self.fixture_index, ROSTER)["valid"])
        row = self.mutable["worlds"][0]
        scope = row["seed_uses"][0]["source_scope"]
        path = self.fixture / row["path"]
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("## Player\n", "## Player\n\n" + scope + "\n", 1), encoding="utf-8")
        self._refresh_hashes(row)
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("private selected scope leaks" in issue for issue in report["issues"]))
        process = subprocess.run(
            [sys.executable, str(CONSUMER), "--index", str(self.fixture_index)],
            cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
        )
        self.assertNotEqual(process.returncode, 0)

    def test_malformed_roster_cli_returns_invalid_json_without_traceback(self) -> None:
        cases = []
        for kind in ("null_seed", "missing_categories", "bad_repository", "null_snapshot", "bad_commit", "missing_commit_url", "bad_commit_url", "nonobject_category", "nonarray_seeds"):
            data = json.loads(json.dumps(self.roster))
            if kind == "null_seed":
                data["seeds"] = [None]
            elif kind == "missing_categories":
                del data["categories"]
            elif kind == "bad_repository":
                data["seeds"][0]["repository"] = None
            elif kind == "null_snapshot":
                data["seeds"][0]["candidate_snapshot"] = None
            elif kind == "bad_commit":
                data["seeds"][0]["candidate_snapshot"]["commit_sha"] = []
            elif kind == "missing_commit_url":
                del data["seeds"][0]["candidate_snapshot"]["commit_url"]
            elif kind == "bad_commit_url":
                data["seeds"][0]["candidate_snapshot"]["commit_url"] = None
            elif kind == "nonobject_category":
                data["categories"] = ["C01"]
            else:
                data["seeds"] = {"envconfig": {}}
            cases.append((kind, data))
        for kind, data in cases:
            with self.subTest(case=kind):
                path = self.fixture / (kind + "-roster.json")
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    build_seed_world_coverage(INDEX, path)
                process = subprocess.run(
                    [sys.executable, "-m", "shellsmith.campaign.seed_world_catalog", "--index", str(INDEX), "--roster", str(path)],
                    cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
                )
                self.assertEqual(process.returncode, 1)
                self.assertNotIn("Traceback", process.stderr)
                report = json.loads(process.stdout)
                self.assertFalse(report["valid"])
                self.assertTrue(report["issues"])

    def test_malformed_actual_contract_is_rejected_by_canonical_parser(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        path.write_text(path.read_text().replace("## Reward", "### Reward", 1))
        with self.assertRaises(RuntimeError):
            load_world_file(path)
        self.assertFalse(self._report()["valid"])

    def test_changed_contract_hash_is_rejected_by_collection_loader(self) -> None:
        self.mutable["worlds"][0]["content_sha256"] = "0" * 64
        self._save()
        with self.assertRaises(ValueError):
            load_seed_world_specs(self.fixture_index, ROSTER)
        self.assertTrue(any("content hash differs" in issue for issue in self._report()["issues"]))

    def test_duplicate_identity_and_escaping_path_are_rejected(self) -> None:
        self.mutable["worlds"].append(dict(self.mutable["worlds"][0]))
        self.mutable["worlds"][1]["path"] = "../../checkout_idempotency_001.md"
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("duplicate world identity" in issue for issue in report["issues"]))
        self.assertTrue(any("matching Markdown basename" in issue for issue in report["issues"]))

    def test_symlink_cannot_make_an_external_contract_part_of_the_collection(self) -> None:
        row = self.mutable["worlds"][0]
        path = self.fixture / row["path"]
        path.unlink()
        path.symlink_to(COLLECTION / row["path"])
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("path escapes" in issue for issue in report["issues"]))

    def test_size_range_and_unqualified_promotion_are_real_constraints(self) -> None:
        for row in self.mutable["worlds"]:
            row["task_size"] = "bounded_subset"
        self.mutable["runtime_promoted"] = False
        self.mutable["execution_qualification"] = "passed"
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertTrue(any("full_small_project" in issue for issue in report["issues"]))
        self.assertTrue(any("promotion" in issue for issue in report["issues"]))
        self.assertTrue(any("deferred" in issue for issue in report["issues"]))

    def test_unknown_or_incoherent_index_status_is_rejected(self) -> None:
        states = (
            ("authoring_only", True), ("promoted", False),
            ("authoring_only", None), ("promoted", 1),
            ("qualified", True), ("unknown", False), (None, False),
            ([], False), ({}, False),
        )
        for status, runtime_promoted in states:
            with self.subTest(status=status, runtime_promoted=runtime_promoted):
                self.mutable["status"] = status
                self.mutable["runtime_promoted"] = runtime_promoted
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("index status" in issue for issue in report["issues"]))

    def test_legacy_authoring_collection_loads_without_any_core_manifest(self) -> None:
        """Purpose: Preserve the authoring API before a collection is promoted.
        Input: A complete authoring index with no catalog manifest.
        Output: Matching successful API and CLI coverage reports.
        """
        self.mutable["status"] = "authoring_only"
        self.mutable["runtime_promoted"] = False
        del self.mutable["catalog_manifest"]
        del self.mutable["promoted_on"]
        self.fixture_manifest.unlink()

        report = self._report()
        self.assertTrue(report["valid"], report["issues"])
        self.assertEqual(report["status"], "authoring_only")
        self.assertFalse(report["runtime_promoted"])
        self.assertEqual(report["world_count"], len(self.mutable["worlds"]))
        specs = load_seed_world_specs(self.fixture_index, ROSTER)
        self.assertEqual([spec.id for spec in specs], [row["id"] for row in self.mutable["worlds"]])
        process = subprocess.run(
            [sys.executable, "-m", "shellsmith.campaign.seed_world_catalog",
             "--index", str(self.fixture_index), "--roster", str(ROSTER)],
            cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), report)

    def test_wrong_or_missing_manifest_reference_is_rejected(self) -> None:
        for reference in ("missing.json", str(ROOT / "worlds" / "manifest.json"), None):
            with self.subTest(reference=reference):
                self.mutable["catalog_manifest"] = reference
                report = self._report()
                self.assertFalse(report["valid"])
                self.assertTrue(any("catalog manifest" in issue for issue in report["issues"]))

    def test_seed_missing_from_promoted_manifest_is_rejected(self) -> None:
        manifest = json.loads(self.fixture_manifest.read_text())
        world_id = self.mutable["worlds"][0]["id"]
        manifest["tasks"] = [row for row in manifest["tasks"] if row["id"] != world_id]
        self.fixture_manifest.write_text(json.dumps(manifest))
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertIn(f"{world_id}: absent from promoted catalog", report["issues"])

    def test_promoted_seed_path_must_match_indexed_contract(self) -> None:
        manifest = json.loads(self.fixture_manifest.read_text())
        row = self.mutable["worlds"][0]
        duplicate = self.fixture_manifest.parent / "alternate" / row["path"]
        duplicate.parent.mkdir()
        shutil.copy2(self.fixture / row["path"], duplicate)
        promoted = next(item for item in manifest["tasks"] if item["id"] == row["id"])
        promoted["path"] = "alternate/" + row["path"]
        self.fixture_manifest.write_text(json.dumps(manifest))
        report = self._report()
        self.assertFalse(report["valid"])
        self.assertIn(
            f"{row['id']}: promoted catalog path differs from indexed contract",
            report["issues"],
        )

    def test_invalid_cli_input_reports_failure_not_success(self) -> None:
        self.mutable["worlds"][0]["seed_uses"][0]["seed_id"] = "not_in_roster"
        self._save()
        process = subprocess.run(
            [sys.executable, "-m", "shellsmith.campaign.seed_world_catalog", "--index", str(self.fixture_index), "--roster", str(ROSTER)],
            cwd=ROOT, env=_environment(), capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(process.returncode, 1, process.stderr)
        self.assertFalse(json.loads(process.stdout)["valid"])

    def test_index_documentation_points_to_actual_contracts(self) -> None:
        readme = (COLLECTION / "README.md").read_text(encoding="utf-8")
        for row in self.index["worlds"]:
            self.assertIn(f"({row['path']})", readme)
            self.assertTrue((COLLECTION / row["path"]).is_file())
            for use in row["seed_uses"]:
                self.assertIn(use["seed_id"], readme)
        self.assertIn("DEFERRED", readme)
        self.assertIn("one potential example", readme)


if __name__ == "__main__":
    unittest.main()
