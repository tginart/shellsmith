"""Canonical training IDs through real selection, collection, HTTP and Node.

Only external Tinker/Cookbook/torch, Codex and Pi SDK interfaces are fixtures.
The current Markdown contract derives IDs from filenames: same-basename regular
files are duplicates, not distinct declared-ID worlds. Same-basename symlink
entries can resolve to distinct canonical files and exercise the real pipeline.
"""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from shellsmith.campaign.draft_catalog import DEFAULT_DRAFT_ROOT, NON_RUNNABLE
from shellsmith.campaign.hero_slate import load_hero_slate
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config
from shellsmith.train.world_selection import WorldStream, select_world_stream
from shellsmith.world_file import load_world_file
from test_player_credentials import _assert_endpoint_closed
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def _world(path: Path, label: str = "custom") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# {label}\n\n## World\nPrivate {label} world.\n\n"
        f"## Player\nCheck the {label} service.\n\n"
        "## Reward\nScore the final result.\n\n"
        "## Termination conditions\nStop after the final answer.\n",
        encoding="utf-8",
    )
    return path


def _hero(tmp_path: Path, paths: list[Path], *, world_order: list[str] | None = None) -> Path:
    # Exclude the built-in drafts without replacing the canonical slate loader.
    # Invalid-input tests can declare the intended order without pre-validating
    # their files in this fixture; the shipped entry must reject those inputs.
    payload = {
        "schema": "shellsmith.hero_slate.v1",
        "groups_per_update": 2,
        "shuffle_seed": 0,
        "catalog": {
            "include_flavors": ["agentic"],
            "exclude": [path.stem for path in DEFAULT_DRAFT_ROOT.glob("*.md")
                        if path.name not in NON_RUNNABLE],
            "additional_worlds": [str(path) for path in paths],
        },
        "paper_worlds": [],
        "world_order": (world_order if world_order is not None
                        else list(dict.fromkeys(load_world_file(path.resolve()).id for path in paths))),
        "simulators": {
            "terra": {"runtime": "codex", "model": "gpt-5.6-terra", "provider": "codex-cli",
                      "reasoning_effort": "high"},
        },
        "default_simulator": "terra",
        "assignments": {},
        "expected_simulator_counts": {"terra": len(paths)},
    }
    path = tmp_path / "custom-hero.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _config(tmp_path: Path, paths: list[Path], *, world_order: list[str] | None = None):
    config = load_train_config(None, log_dir=tmp_path / "not-started")
    return replace(config, worlds=replace(config.worlds, select="hero",
                                          hero_slate=str(_hero(tmp_path, paths, world_order=world_order))))


class _RefusingSDK:
    """No external SDK attribute may be needed to reject invalid identities."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        self.calls.append(name)
        raise AssertionError(f"Invalid world ID reached the external SDK: {name}")


def test_selected_world_pins_canonical_id_and_episode_args(tmp_path):
    config = load_train_config(None)
    selected = select_world_stream(config).worlds[0]
    assert selected.world_id == load_world_file(selected.world_file).id
    assert trainer._episode_args(config, selected).world_id == selected.world_id
    with pytest.raises(FrozenInstanceError):
        selected.world_id = "different"
    alternate = _world(tmp_path / "new-canonical.01.md")
    changed = replace(selected, world_file=alternate)
    assert changed.world_id == load_world_file(alternate).id
    assert trainer._episode_args(config, changed).world_id == changed.world_id
    assert selected.world_id != changed.world_id


def test_actual_selected_world_ids_must_be_unique(tmp_path):
    config = load_train_config(None)
    original = select_world_stream(config).worlds[0]
    paths = [_world(tmp_path / side / "same.md", side) for side in ("left", "right")]
    worlds = tuple(replace(original, world_file=path) for path in paths)
    assert paths[0] != paths[1]
    assert load_world_file(paths[0]).id == load_world_file(paths[1]).id
    with pytest.raises(ValueError, match="Duplicate selected world ID"):
        WorldStream(worlds, {"select": "hero"})


def test_same_basename_regular_hero_worlds_reject_before_auth_or_output(tmp_path):
    paths = [_world(tmp_path / side / "same.md", side) for side in ("left", "right")]
    config = _config(tmp_path, paths)
    deps = _RefusingSDK()
    with pytest.raises(RuntimeError, match="Duplicate hero world ID: same"):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert deps.calls == []
    assert not config.resolved_log_dir().exists()


@pytest.mark.parametrize("world_id", [
    ".", "..", "bad world", "bad?world", "bad#world", "bad%2fworld", "bad\\world",
    "bad:world", "caf\u00e9", "bad\nworld", "bad\tworld",
])
def test_unsafe_canonical_hero_id_rejects_before_auth_or_output(tmp_path, world_id):
    path = _world(tmp_path / "inputs" / f"{world_id}.md")
    # Python 3.14 considers all-leading-dot names suffixless; those inputs are
    # rejected by the real extension guard, while older versions reach the ID
    # guard. Keep both negative cases, without failing during fixture creation.
    if path.suffix == ".md":
        assert load_world_file(path).id == world_id
        message = "Selected world ID.*endpoint-safe"
    else:
        assert world_id in {".", ".."}
        message = r"World files must use the \.md extension"
    config = _config(tmp_path, [path], world_order=[world_id])
    deps = _RefusingSDK()
    with pytest.raises((ValueError, RuntimeError), match=message):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert deps.calls == []
    assert not config.resolved_log_dir().exists()


def test_case_colliding_world_directories_reject_before_auth_or_output(tmp_path):
    paths = [_world(tmp_path / "left" / "Case-ID.md"), _world(tmp_path / "right" / "case-id.md")]
    config = _config(tmp_path, paths)
    deps = _RefusingSDK()
    with pytest.raises(ValueError, match="Selected world IDs.*case-insensitive"):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert deps.calls == []
    assert not config.resolved_log_dir().exists()


@pytest.mark.parametrize("world_id", [
    "campaign_manifest.json", "campaign_report.json", "campaign_report.md", "capture_index.json", "player_records",
])
def test_world_id_cannot_claim_campaign_storage_before_auth_or_output(tmp_path, world_id):
    config = _config(tmp_path, [_world(tmp_path / "inputs" / f"{world_id}.md")])
    deps = _RefusingSDK()
    with pytest.raises(ValueError, match="Selected world ID.*reserved campaign path"):
        asyncio.run(trainer.run_train(config, deps=deps))
    assert deps.calls == []
    assert not config.resolved_log_dir().exists()


def test_real_custom_hero_canonical_ids_keep_concurrent_endpoints_paths_and_captures_distinct(
    tmp_path, monkeypatch, training_sdk,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    targets = [_world(tmp_path / side / f"{world_id}.md", side)
               for side, world_id in (("left", "Canonical-A.01"), ("right", "canonical_B-02"))]
    aliases = []
    for target in targets:
        alias = target.with_name("world.md")
        alias.symlink_to(target.name)
        aliases.append(alias)
    assert aliases[0].name == aliases[1].name
    canonical_ids = [load_world_file(path).id for path in targets]
    config = replace(
        config,
        worlds=replace(config.worlds, select="hero", hero_slate=str(_hero(tmp_path, aliases)), count=1),
        train=replace(config.train, steps=1, groups_per_update=2),
        adaptive=replace(config.adaptive, parallelism=4, parallelism_per_provider=4),
    )

    # The external SDK waits for all four real Node children to authenticate.
    # This proves worlds overlap, rather than accidentally testing serial runs.
    sdk = Path(config.player.pi_sdk_module_path)
    source = sdk.read_text().replace("import { appendFileSync }", "import { appendFileSync, readFileSync }")
    source = source.replace("stage, status: response.status, model: payload.data[0].id,",
                            "stage, baseUrl: runtime.config.baseUrl, status: response.status, model: payload.data[0].id,")
    source = source.replace("await authenticate(modelRuntime, 'prompt');", """
              await authenticate(modelRuntime, 'prompt');
              const deadline = Date.now() + 10000;
              while (true) {
                const rows = readFileSync(auditPath, 'utf8').trim().split('\\n').map(JSON.parse);
                if (rows.filter(row => row.stage === 'initialize').length === 4) break;
                require(Date.now() < deadline, 'four concurrent Node episodes did not initialize');
                await new Promise(resolve => setTimeout(resolve, 10));
              }
              appendFileSync(auditPath, JSON.stringify({
                stage: 'sampling', baseUrl: modelRuntime.config.baseUrl, pid: process.pid,
              }) + '\\n');
    """)
    sdk.write_text(source)

    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert state["status"] == "complete" and state["step"] == 1 and state["next_world_offset"] == 2
    assert sum(call[0] == "optim_step" for call in training_sdk.calls) == 1
    root = config.resolved_log_dir()
    segment = root / "segments/000"
    identities = {row["world_id"]: row for row in state["group_identities"].values()}
    assert set(identities) == set(canonical_ids)
    assert len({row["key"] for row in identities.values()}) == 2
    snapshot = json.loads((root / trainer.SNAPSHOT_FILE).read_text())
    assert snapshot["worlds"] == canonical_ids
    assert [row["world_file"] for row in snapshot["world_identities"]] == [str(path) for path in targets]
    metrics = json.loads((root / trainer.METRICS_FILE).read_text().splitlines()[0])
    assert [row["world_id"] for row in metrics["worlds"]] == canonical_ids
    assert metrics["datums"] == 4
    assert all(row["accepted"] == 2 and row["rewards"] == [0.2, 0.8] for row in metrics["worlds"])

    expected_episodes = {f"{state['run_id']}-s000-{world_id}-{attempt:02d}":
                         (world_id, segment / world_id / f"rollout_{attempt:02d}")
                         for world_id in canonical_ids for attempt in (1, 2)}
    index = json.loads((segment / "capture_index.json").read_text())["episodes"]
    assert set(index) == set(expected_episodes)
    assert len({row["relative_rollout_dir"] for row in index.values()}) == 4
    assert {path.stem for path in (segment / "player_records").glob("*.jsonl")} == set(expected_episodes)
    for episode_id, (world_id, rollout) in expected_episodes.items():
        episode = json.loads((rollout / "episode.json").read_text())
        outcome = json.loads((rollout / "logs/durable_outcome.json").read_text())["outcome"]
        capture = json.loads((rollout / "logs/tinker_trainable_capture.json").read_text())
        provenance = json.loads((rollout / "policy_provenance.json").read_text())
        assert episode["world_id"] == outcome["world_id"] == world_id
        assert episode["episode_id"] == outcome["episode_id"] == capture["episode_id"] == episode_id
        assert outcome["accepted_count"] == 1 and outcome["error"] is None
        assert outcome["group_identity_key"] == capture["group_identity_key"] == identities[world_id]["key"]
        assert provenance["episode_id"] == episode_id and provenance["group_identity_key"] == capture["group_identity_key"]
        assert index[episode_id]["accepted"] is True
        assert index[episode_id]["group_identity_key"] == capture["group_identity_key"]
        assert segment / index[episode_id]["relative_rollout_dir"] == rollout
        assert Path(capture["rollout_root"]) == rollout
        rows = [json.loads(line) for line in (segment / "player_records" / f"{episode_id}.jsonl").read_text().splitlines()]
        assert len(rows) == len(capture["records"]) == 1
        assert rows[0]["session_id"] == episode_id
        assert rows[0]["request_id"] == capture["records"][0]["request_id"]
        assert rows[0]["request_id"].startswith(episode_id + "-req-")
        protocol = [json.loads(line) for line in (rollout / "logs/vat.jsonl").read_text().splitlines()]
        initialize = next(row["message"] for row in protocol if row.get("message", {}).get("type") == "initialize")
        assert initialize["episode_id"] == episode_id
        assert urlparse(initialize["model"]["base_url"]).path == f"/e/{episode_id}/v1"

    audit = [json.loads(line) for line in pi_audit.read_text().splitlines()]
    urls = {row["baseUrl"] for row in audit}
    assert len(urls) == len({row["pid"] for row in audit}) == 4
    assert {urlparse(url).path for url in urls} == {f"/e/{episode_id}/v1" for episode_id in expected_episodes}
    first_sample = next(index for index, row in enumerate(audit) if row["stage"] == "sampling")
    assert sum(row["stage"] == "initialize" for row in audit[:first_sample]) == 4
    for url in urls:
        rows = [row for row in audit if row["baseUrl"] == url]
        assert [row["stage"] for row in rows] == ["initialize", "prompt", "sampling"]
        assert all(row["status"] == 200 for row in rows[:2])
        _assert_endpoint_closed(url)

    # Check the public selection/task-argument handoff too, after exercising the
    # complete runtime. The old selector/args had no canonical identity field.
    selected = select_world_stream(config)
    assert [world.world_id for world in selected.worlds] == canonical_ids
    assert [trainer._episode_args(config, world).world_id for world in selected.worlds] == canonical_ids
    assert [world.world_id for world in load_hero_slate(Path(config.worlds.hero_slate)).worlds] == canonical_ids
