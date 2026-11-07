"""Purpose: Check seed coverage, provenance, and declared catalog membership.
Input: The seed index, source roster, and a manifest for promoted indexes.
Output: A structural audit report and immutable parsed world contracts.
Note: These checks do not execute episodes or certify execution qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from shellsmith.catalog import load_catalog_entries
from shellsmith.paths import PROJECT_ROOT, WORLDS_DIR
from shellsmith.world_file import WorldFileSpec, load_world_file

DEFAULT_INDEX = WORLDS_DIR / "seeds" / "coverage.json"
DEFAULT_ROSTER = PROJECT_ROOT / "docs" / "seed_roster.json"
INDEX_SCHEMA = "shellsmith.seed_worlds.v1"
TASK_SIZES = {"bounded_subset", "full_small_project"}


def _object(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ValueError(f"{path}: expected schema {schema!r}")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: expected a nonempty string")
    return value


def _strings(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{context}: expected a nonempty string list")
    values = [_text(item, context) for item in value]
    if len(values) != len(set(values)):
        raise ValueError(f"{context}: duplicate values")
    return values


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _seed_evidence_blocks(world: str) -> dict[str, str]:
    """Associate private evidence with its explicit seed declaration.

    Declarations may span paragraphs; the next declaration ends a seed's block.
    These are collection-local metadata conventions, not a second world parser.
    """
    declarations = list(re.finditer(
        r"^(?:Seed |Private seed provenance: |Private provenance: seed )`([^`]+)`",
        world, re.MULTILINE,
    ))
    blocks: dict[str, str] = {}
    for index, declaration in enumerate(declarations):
        seed_id = declaration.group(1)
        if seed_id in blocks:
            raise ValueError(f"duplicate private seed declaration: {seed_id}")
        end = declarations[index + 1].start() if index + 1 < len(declarations) else len(world)
        blocks[seed_id] = _normalized(world[declaration.start():end])
    return blocks


def build_seed_world_coverage(
    index_path: Path = DEFAULT_INDEX,
    roster_path: Path = DEFAULT_ROSTER,
) -> dict[str, Any]:
    """Purpose: Check the index against contracts, sources, and declared promotion.
    Input: Paths to the seed index and source roster.
    Output: A structural coverage and promotion report.
    Note: A valid report does not certify solvability or execution quality.
    """
    report, _ = _audit_seed_worlds(index_path, roster_path)
    return report


def _audit_seed_worlds(
    index_path: Path, roster_path: Path,
) -> tuple[dict[str, Any], tuple[WorldFileSpec, ...]]:
    """Purpose: Audit one collection and retain its parsed contract snapshots.
    Input: Paths to an authoring or promoted seed index and source roster.
    Output: The audit report and immutable parsed contracts.
    Note: Only promoted collections require core manifest membership.
    """
    index_path = index_path.expanduser().resolve()
    roster_path = roster_path.expanduser().resolve()
    index = _object(index_path, INDEX_SCHEMA)
    roster = _object(roster_path, "shellsmith.seed_roster.v1")
    seed_rows = roster.get("seeds")
    category_rows = roster.get("categories")
    for values, label in ((seed_rows, "seeds"), (category_rows, "categories")):
        if not isinstance(values, list) or not values or any(not isinstance(row, dict) for row in values):
            raise ValueError(f"roster {label}: expected a nonempty list of objects")
    seeds = {_text(row.get("id"), "roster seed id"): row for row in seed_rows}
    categories = {_text(row.get("id"), "roster category id") for row in category_rows}
    if len(seeds) != len(seed_rows) or len(categories) != len(category_rows):
        raise ValueError("roster contains duplicate seed or category identities")
    for seed_id, seed in seeds.items():
        _text(seed.get("repository"), f"roster seed {seed_id} repository")
        snapshot = seed.get("candidate_snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError(f"roster seed {seed_id}: candidate_snapshot must be an object")
        _text(snapshot.get("commit_url"), f"roster seed {seed_id} commit_url")
        sha = _text(snapshot.get("commit_sha"), f"roster seed {seed_id} commit_sha")
        if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
            raise ValueError(f"roster seed {seed_id}: commit_sha must be 40 lowercase hex characters")
    rows = index.get("worlds")
    if not isinstance(rows, list) or not rows:
        raise ValueError("index worlds must be a nonempty list")

    issues: list[str] = []
    status = index.get("status")
    runtime_promoted = index.get("runtime_promoted")
    if status not in ("authoring_only", "promoted"):
        issues.append("index status must be authoring_only or promoted")
    elif runtime_promoted is not (status == "promoted"):
        issues.append("index status and runtime promotion must agree")
    if index.get("execution_qualification") != "deferred":
        issues.append("execution qualification must remain deferred")
    promoted_paths: dict[str, Path] = {}
    if status == "promoted":
        try:
            reference = _text(index.get("catalog_manifest"), "catalog manifest reference")
            if Path(reference).is_absolute():
                raise ValueError("catalog manifest reference must be relative to the index")
            manifest_path = (index_path.parent / reference).resolve()
            promoted_paths = {
                entry.id: entry.path for entry in load_catalog_entries(manifest_path)
            }
        except (OSError, RuntimeError, ValueError) as error:
            issues.append(f"catalog manifest: {error}")

    used_seeds: set[str] = set()
    used_categories: set[str] = set()
    seen_ids: set[str] = set()
    indexed_paths: set[Path] = set()
    reports: list[dict[str, Any]] = []
    specs: list[WorldFileSpec] = []
    sizes: Counter[str] = Counter()
    for row in rows:
        context = "world entry"
        try:
            if not isinstance(row, dict):
                raise ValueError("world entry must be an object")
            world_id = _text(row.get("id"), "world id")
            context = world_id
            if world_id in seen_ids:
                raise ValueError(f"{world_id}: duplicate world identity")
            seen_ids.add(world_id)
            relative = _text(row.get("path"), world_id + " path")
            if relative != world_id + ".md" or Path(relative).name != relative:
                raise ValueError(f"{world_id}: path must be the matching Markdown basename")
            path = (index_path.parent / relative).resolve()
            if path.parent != index_path.parent:
                raise ValueError(f"{world_id}: path escapes the collection")
            if path in indexed_paths:
                raise ValueError(f"{world_id}: duplicate world path")
            indexed_paths.add(path)
            if status == "promoted":
                if world_id not in promoted_paths:
                    issues.append(f"{world_id}: absent from promoted catalog")
                elif promoted_paths[world_id] != path:
                    issues.append(f"{world_id}: promoted catalog path differs from indexed contract")
            category_ids = _strings(row.get("category_ids"), world_id + " categories")
            used_categories.update(category_ids)
            size = _text(row.get("task_size"), world_id + " task size")
            if size not in TASK_SIZES:
                raise ValueError(f"{world_id}: unknown task size {size!r}")
            sizes[size] += 1
            uses = row.get("seed_uses")
            if not isinstance(uses, list) or not uses:
                raise ValueError(f"{world_id}: seed_uses must be nonempty")
            seed_ids = _strings([use.get("seed_id") for use in uses if isinstance(use, dict)], world_id + " seeds")
            if len(seed_ids) != len(uses):
                raise ValueError(f"{world_id}: invalid seed use")
            used_seeds.update(seed_ids)

            spec = load_world_file(path)
            if spec.id != world_id:
                raise ValueError(f"{world_id}: parser identity differs")
            title = _text(row.get("title"), world_id + " title")
            if title != spec.title:
                issues.append(f"{world_id}: indexed title differs from parsed contract")
            if spec.content_hash() != row.get("content_sha256"):
                issues.append(f"{world_id}: content hash differs from indexed contract")
            player = spec.player_projection()
            player_hash = hashlib.sha256(player.encode("utf-8")).hexdigest()
            if player_hash != row.get("player_sha256"):
                issues.append(f"{world_id}: Player projection hash differs from index")
            evidence_blocks = _seed_evidence_blocks(spec.world)
            if set(evidence_blocks) != set(seed_ids):
                issues.append(f"{world_id}: private seed declarations differ from indexed seed IDs")
            folded_player = player.casefold()
            normalized_player = _normalized(folded_player)
            # Require at least 12 hex digits and a complete token to avoid
            # treating short numbers or prefixes of unrelated hashes as revisions.
            abbreviated_commits = re.findall(r"\b[0-9a-f]{12,39}\b", folded_player)
            # The first private paragraph is the authoritative classification.
            # This convention is collection-local; the canonical parser is unchanged.
            category_text = ", ".join(f"`{category}`" for category in category_ids)
            declaration = (
                f"Private authoring identity: `{world_id}`; "
                f"category_ids: {category_text}; task_size: `{size}`."
            )
            if _normalized(spec.world.split("\n\n", 1)[0]) != declaration:
                issues.append(f"{world_id}: indexed classification differs from private World")
            for roster_id, roster_seed in seeds.items():
                sha = roster_seed["candidate_snapshot"]["commit_sha"]
                if any(sha.startswith(prefix) for prefix in abbreviated_commits):
                    issues.append(f"{world_id}/{roster_id}: private abbreviated commit leaks into Player")
                provenance = (
                    (roster_seed["repository"], "repository"),
                    ("https://github.com/" + roster_seed["repository"], "canonical source"),
                    (roster_seed["candidate_snapshot"]["commit_sha"], "commit"),
                )
                for value, label in provenance:
                    if value.casefold() in folded_player:
                        issues.append(f"{world_id}/{roster_id}: private {label} leaks into Player")
            for use in uses:
                seed_id = use["seed_id"]
                if seed_id not in seeds:
                    continue
                seed = seeds[seed_id]
                private = evidence_blocks.get(seed_id, "")
                url = "https://github.com/" + seed["repository"]
                sha = seed["candidate_snapshot"]["commit_sha"]
                if (
                    use.get("repository") != seed["repository"]
                    or use.get("commit_sha") != sha
                    or use.get("source_url") != url
                    or use.get("commit_url") != seed["candidate_snapshot"]["commit_url"]
                ):
                    issues.append(f"{world_id}/{seed_id}: indexed provenance differs from roster")
                for value, label in ((url, "canonical source"), (sha, "commit")):
                    if value not in private:
                        issues.append(f"{world_id}/{seed_id}: private World omits {label}")
                evidence = _text(use.get("role_evidence"), world_id + " seed role")
                scope = _text(use.get("source_scope"), world_id + " source scope")
                if len(evidence.split()) < 12 or len(evidence) < 80:
                    issues.append(f"{world_id}/{seed_id}: seed role is not substantive")
                if _normalized(evidence) not in private:
                    issues.append(f"{world_id}/{seed_id}: seed role absent from private World seed block")
                if _normalized(scope) not in private:
                    issues.append(f"{world_id}/{seed_id}: selected scope absent from private World seed block")
                if _normalized(evidence).casefold() in normalized_player:
                    issues.append(f"{world_id}/{seed_id}: private seed role leaks into Player")
                if _normalized(scope).casefold() in normalized_player:
                    issues.append(f"{world_id}/{seed_id}: private selected scope leaks into Player")
            specs.append(spec)
            reports.append({
                "id": spec.id, "path": relative, "title": spec.title,
                "content_sha256": spec.content_hash(), "player_sha256": player_hash,
                "seed_ids": seed_ids, "category_ids": category_ids, "task_size": size,
            })
        except (OSError, RuntimeError, ValueError) as error:
            issues.append(f"{context}: {error}")

    missing_seeds = sorted(set(seeds) - used_seeds)
    unknown_seeds = sorted(used_seeds - set(seeds))
    missing_categories = sorted(categories - used_categories)
    unknown_categories = sorted(used_categories - categories)
    if missing_seeds:
        issues.append("missing seeds: " + ", ".join(missing_seeds))
    if unknown_seeds:
        issues.append("unknown seeds: " + ", ".join(unknown_seeds))
    if missing_categories:
        issues.append("missing categories: " + ", ".join(missing_categories))
    if unknown_categories:
        issues.append("unknown categories: " + ", ".join(unknown_categories))
    if set(sizes) != TASK_SIZES:
        issues.append("collection must include bounded_subset and full_small_project")
    unindexed = sorted(
        path.name for path in index_path.parent.iterdir()
        if path.suffix.lower() == ".md"
        and path.name != "README.md" and path.resolve() not in indexed_paths
    )
    if unindexed:
        issues.append("unindexed worlds: " + ", ".join(unindexed))
    return {
        "schema": "shellsmith.seed_world_coverage.v1",
        "valid": not issues, "status": index.get("status"),
        "runtime_promoted": index.get("runtime_promoted"),
        "catalog_manifest": index.get("catalog_manifest"),
        "execution_qualification": index.get("execution_qualification"),
        "world_count": len(reports), "seed_count": len(seeds), "category_count": len(categories),
        "covered_seed_ids": sorted(used_seeds & set(seeds)),
        "missing_seed_ids": missing_seeds, "unknown_seed_ids": unknown_seeds,
        "covered_category_ids": sorted(used_categories & categories),
        "missing_category_ids": missing_categories, "unknown_category_ids": unknown_categories,
        "task_sizes": dict(sorted(sizes.items())), "unindexed_world_paths": unindexed,
        "issues": issues, "worlds": reports,
    }, tuple(specs)


def load_seed_world_specs(
    index_path: Path = DEFAULT_INDEX,
    roster_path: Path = DEFAULT_ROSTER,
) -> tuple[WorldFileSpec, ...]:
    """Load seed contracts after coverage and declared promotion checks pass."""
    report, specs = _audit_seed_worlds(index_path, roster_path)
    if not report["valid"]:
        raise ValueError("invalid seed world collection: " + "; ".join(report["issues"]))
    return specs


def main(argv: list[str] | None = None) -> int:
    """Purpose: Run the seed coverage audit from the command line.
    Input: Optional index and source-roster paths.
    Output: A JSON report and a nonzero exit code for an invalid collection.
    """
    parser = argparse.ArgumentParser(
        description="Audit seed coverage and catalog membership without episodes."
    )
    parser.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX,
        help="Seed index; promoted indexes require a relative core manifest reference.",
    )
    parser.add_argument(
        "--roster", type=Path, default=DEFAULT_ROSTER,
        help="Source roster for private seed provenance checks.",
    )
    args = parser.parse_args(argv)
    try:
        report = build_seed_world_coverage(args.index, args.roster)
    except (OSError, RuntimeError, ValueError) as error:
        report = {"valid": False, "issues": [str(error)]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
