"""Campaign preflight through the shipped config, selection and entry point.

No collector, policy, report or snapshot is replaced. Invalid local inputs must
fail before any external SDK attribute is accessed or output is created.
Runtime/HTTP/Node coverage lives in test_campaign_inference_integration.py.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from shellsmith.campaign import inference
from shellsmith.train.config import load_train_config
from shellsmith.train.world_selection import select_world_stream
from test_trainer_world_ids import _hero, _world


class _NoSDK:
    def __getattr__(self, name):
        pytest.fail(f"Campaign preflight reached external SDK attribute {name}")


def _config(tmp_path: Path):
    return load_train_config(None, log_dir=tmp_path / "log", campaign_mode="inference")


def _reject(config, match: str, error=(ValueError, RuntimeError)) -> None:
    with pytest.raises(error, match=match):
        asyncio.run(inference.run_inference_campaign(config, deps=_NoSDK(), sampler_path=None))
    assert not config.resolved_log_dir(kind="campaigns").exists()


def test_campaign_refuses_a_used_log_dir(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = config.resolved_log_dir(kind="campaigns")
    root.mkdir()
    previous = root / "old.json"
    previous.write_text("{}")
    with pytest.raises(RuntimeError, match="not empty"):
        asyncio.run(inference.run_inference_campaign(config, deps=_NoSDK(), sampler_path=None))
    assert list(root.iterdir()) == [previous] and previous.read_text() == "{}"


def test_duplicate_requested_ids_reject_before_auth_or_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    world = select_world_stream(config, campaign_mode="inference").worlds[0]
    config = replace(config, worlds=replace(config.worlds, select="ids", ids=(world.world_id,) * 2, count=2))
    _reject(config, "Duplicate.*ids")


def test_duplicate_actual_selected_ids_reject_before_auth_or_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = [_world(tmp_path / side / "same.md", side) for side in ("left", "right")]
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(_hero(tmp_path, paths))))
    _reject(config, "Duplicate hero world ID: same")


def test_case_colliding_selected_ids_reject_before_auth_or_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = [_world(tmp_path / "left/Case-ID.md"), _world(tmp_path / "right/case-id.md")]
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(_hero(tmp_path, paths))))
    _reject(config, "Selected world IDs.*case-insensitive")


@pytest.mark.parametrize("world_id", [".", "..", "bad world", "bad%2fworld", "player_records", "campaign_report.json"])
def test_unsafe_selected_ids_reject_before_auth_or_output(tmp_path: Path, world_id: str) -> None:
    config = _config(tmp_path)
    path = _world(tmp_path / "inputs" / f"{world_id}.md")
    # Declare invalid inputs without deriving/validating IDs in the fixture.
    # Python 3.14 rejects all-leading-dot filenames at the extension guard;
    # earlier versions reach the endpoint-safe ID guard. Both must stay offline.
    slate = _hero(tmp_path, [path], world_order=[world_id])
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(slate)))
    if path.suffix == ".md":
        message = "Selected world ID.*(endpoint-safe|reserved)"
    else:
        assert world_id in {".", ".."}
        message = r"World files must use the \.md extension"
    _reject(config, message)


@pytest.mark.parametrize("overrides,match", [
    ({"maximum_attempts": 2}, "initial_attempts"),
    ({"minimum_trainable": 9}, "minimum_trainable"),
    ({"minimum_reward_range": float("inf")}, "finite"),
    ({"minimum_trainable": 0}, "minimum_trainable"),
])
def test_invalid_merged_world_policy_rejects_before_auth_or_output(tmp_path: Path, overrides, match: str) -> None:
    config = _config(tmp_path)
    world = select_world_stream(config, campaign_mode="inference").worlds[0]
    config = replace(config, world_overrides={world.world_id: overrides})
    _reject(config, match)


def test_unknown_world_override_rejects_before_auth_or_output(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), world_overrides={"not-selected": {"initial_attempts": 1}})
    _reject(config, "not in this run")


def test_campaign_composed_name_preflight_precedes_sdk_and_output(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = replace(config, run=replace(config.run, run_id="r" * 240))
    _reject(config, "(Player record filename|Episode staging directory).*byte")


@pytest.mark.parametrize("field,value", [("runtime", "claude"), ("provider", "openrouter")])
def test_campaign_rejects_unconfined_hero_assignment_before_auth(tmp_path: Path, field: str, value: str) -> None:
    config = _config(tmp_path)
    path = _hero(tmp_path, [_world(tmp_path / "inputs/world.md")])
    payload = json.loads(path.read_text())
    payload["simulators"]["terra"][field] = value
    path.write_text(json.dumps(payload))
    config = replace(config, worlds=replace(config.worlds, select="hero", hero_slate=str(path)))
    _reject(config, "(confined Codex GM|codex-cli)")


def test_campaign_import_does_not_import_trainer() -> None:
    subprocess.run([sys.executable, "-c", "import sys; import shellsmith.campaign.inference; "
                    "assert 'shellsmith.train.trainer' not in sys.modules"],
                   check=True, capture_output=True, text=True, timeout=10)


def test_inference_config_allows_minimum_one_and_ignores_train_settings(tmp_path: Path) -> None:
    path = tmp_path / "one.toml"
    path.write_text('[adaptive]\ninitial_attempts = 1\nmaximum_attempts = 1\nminimum_trainable = 1\n'
                    '[train]\nsteps = 0\ngroups_per_update = 0\ncheckpoint_every = 0\nlearning_rate = -1.0\n')
    config = load_train_config(path, campaign_mode="inference")
    assert config.adaptive.minimum_trainable == 1 and config.train.steps == 0
    assert select_world_stream(config, campaign_mode="inference").worlds
    with pytest.raises(ValueError):
        load_train_config(path)  # Training's stricter group-relative contract is unchanged.


def test_fallback_to_a_claude_gm_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "fallback.toml"
    path.write_text('[adaptive]\nprovider_outage = "fallback"\n')
    with pytest.raises(ValueError, match="must be 'halt'"):
        load_train_config(path, campaign_mode="inference")
