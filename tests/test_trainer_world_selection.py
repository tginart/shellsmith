"""M4 config and immutable world selection through the real loaders and trainer.

No selector, slate loader, identity builder, or trainer function is replaced.
The SDK boundary deliberately refuses service creation: invalid configurations
must fail before authenticating, collecting episodes, or doing paid work.
"""

from __future__ import annotations

import asyncio
import builtins
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from shellsmith.campaign.hero_slate import load_hero_slate
from shellsmith.catalog import load_catalog_entries
from shellsmith.config import SUPPORTED_TINKER_BASE_MODELS
from shellsmith.paths import CATALOG_MANIFEST, PROJECT_ROOT
from shellsmith.train.config import RunSection, TrainConfig, load_train_config, validate_train_config
from shellsmith.train.identity import group_identity
from shellsmith.train.world_selection import SelectedWorld, WorldStream, select_world_stream


@pytest.fixture(autouse=True)
def forbid_optional_sdk_imports(monkeypatch):
    """Guard only the external import boundary, never the functions under test."""
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"tinker", "tinker_cookbook", "torch"}:
            pytest.fail(f"Config/selection validation must remain offline: imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _toml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _direct_config() -> TrainConfig:
    # Direct callers must now supply an explicit identity; do not let that
    # independent guard mask the setting each validation regression exercises.
    return TrainConfig(run=RunSection(run_id="direct-validation-fixture"))


def _changed(config: TrainConfig, section: str, key: str, value: object) -> TrainConfig:
    return replace(config, **{section: replace(getattr(config, section), **{key: value})})


@pytest.fixture
def hero_path(tmp_path: Path) -> Path:
    """Ten real worlds: draft flavors, an additional world, and a paper world."""
    ids = [
        "geospatial_antimeridian_tile_gap_001",
        "public_repo_go_jose_multisig_key_selection_001",
        "insurance_edi_835_plb_balance_001",
        "public_repo_packaging_error_placeholder_integrity_001",
        "sap_odata_delta_token_001",
        "s3_parquet_decimal_schema_001",
        "public_repo_attrs_nested_validator_disable_001",
        "public_repo_regex_unicode_property_negation_001",
    ]
    paper_id = "paper_control_dpo_loss_001"
    payload = {
        "schema": "shellsmith.hero_slate.v1",
        "groups_per_update": 2,
        "shuffle_seed": 20260716,
        "catalog": {
            "include_flavors": ["agentic", "public_repo"],
            "include": ids,
            "exclude": [],
            "additional_worlds": ["worlds/checkout_idempotency_001.md"],
        },
        "paper_worlds": [paper_id],
        "world_order": [ids[1], "checkout_idempotency_001", ids[0], paper_id, *ids[2:]],
        "simulators": {
            "sonnet": {"runtime": "claude", "model": "sonnet", "provider": "claude-cli",
                       "reasoning_effort": "medium"},
            "terra": {"runtime": "codex", "model": "gpt-5.6-terra", "provider": "codex-cli",
                      "reasoning_effort": "high"},
            "sol": {"runtime": "codex", "model": "gpt-6.1-sol", "provider": "codex-cli",
                    "reasoning_effort": "xhigh"},
            "pi": {"runtime": "pi", "model": "qwen/qwen3.8-27b", "provider": "openrouter",
                   "reasoning_effort": "low"},
        },
        "default_simulator": "sonnet",
        "assignments": {"sol": [ids[1], ids[7]], "terra": [ids[5], ids[6]], "pi": [paper_id]},
        "expected_simulator_counts": {"sonnet": 5, "terra": 2, "sol": 2, "pi": 1},
    }
    path = tmp_path / "representative-hero.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _hero_config(tmp_path: Path, path: Path, extra: str = "") -> TrainConfig:
    return load_train_config(_toml(
        tmp_path, f'[worlds]\nselect = "hero"\nhero_slate = {json.dumps(str(path))}\n{extra}',
    ))


def test_catalog_manifest_keeps_exact_existing_paths_and_global_assignments(tmp_path: Path) -> None:
    config = load_train_config(_toml(tmp_path, '[worlds]\ncount = 3\n[gm]\nreasoning_effort = "high"\n'))
    entries = load_catalog_entries()
    stream = select_world_stream(config)
    assert isinstance(stream, WorldStream)
    assert tuple(world.world_file for world in stream.worlds) == tuple(entry.path for entry in entries[:3])
    assert all(world.world_file.is_absolute() and world.world_file.is_file() for world in stream.worlds)
    assert all((world.gm_runtime, world.gm_model, world.gm_provider, world.gm_reasoning_effort) ==
               (config.gm.runtime, config.gm.model, config.gm.provider, "high") for world in stream.worlds)
    assert stream.provenance == {
        "select": "manifest", "manifest_path": str(CATALOG_MANIFEST.resolve()),
        "manifest_sha256": hashlib.sha256(CATALOG_MANIFEST.read_bytes()).hexdigest(),
    }


def test_catalog_random_is_seeded_before_count_and_does_not_touch_global_rng(tmp_path: Path) -> None:
    config = load_train_config(_toml(tmp_path, '[worlds]\nselect = "random"\ncount = 5\nseed = 37\n'))
    expected = [entry.path for entry in load_catalog_entries()]
    random.Random(37).shuffle(expected)
    rng_state = random.getstate()
    first = select_world_stream(config)
    assert first == select_world_stream(config)
    assert [world.world_file for world in first.worlds] == expected[:5]
    assert random.getstate() == rng_state


def test_catalog_ids_keep_requested_order_and_count(tmp_path: Path) -> None:
    entries = load_catalog_entries()
    chosen = (entries[5], entries[1], entries[3])
    config = load_train_config(_toml(
        tmp_path, f'[worlds]\nselect = "ids"\nids = {json.dumps([entry.id for entry in chosen])}\ncount = 2\n',
    ))
    assert [world.world_file for world in select_world_stream(config).worlds] == [entry.path for entry in chosen[:2]]
    all_entries = select_world_stream(_changed(config, "worlds", "count", 10_000))
    assert len(all_entries.worlds) == len(chosen)  # No duplication to fill count.


def test_unknown_catalog_id_is_rejected_even_beyond_count(tmp_path: Path) -> None:
    known = load_catalog_entries()[0].id
    config = load_train_config(_toml(
        tmp_path, f'[worlds]\nselect = "ids"\ncount = 1\nids = {json.dumps([known, "not-a-catalog-world"])}\n',
    ))
    with pytest.raises(ValueError, match="not in the manifest.*not-a-catalog-world"):
        select_world_stream(config)


@pytest.mark.parametrize("extra", ["", "count = 1\nseed = 917\n", "count = 10000\nseed = -6\n"])
def test_hero_keeps_entire_order_and_every_simulator_assignment(
    tmp_path: Path, hero_path: Path, extra: str,
) -> None:
    config = _hero_config(tmp_path, hero_path, extra)
    before = hero_path.read_bytes()
    slate = load_hero_slate(hero_path)
    stream = select_world_stream(config)
    assert len(stream.worlds) == 10 > TrainConfig().worlds.count
    assert stream.worlds == tuple(
        SelectedWorld(world.world_file, world.simulator.runtime, world.simulator.model,
                      world.simulator.provider, world.simulator.reasoning_effort)
        for world in slate.worlds
    )
    assert {world.gm_runtime for world in stream.worlds} == {"codex", "claude", "pi"}
    assert {world.gm_reasoning_effort for world in stream.worlds} == {"low", "medium", "high", "xhigh"}
    assert stream.provenance == {
        "select": "hero", "hero_slate_path": str(hero_path.resolve()),
        "hero_slate_sha256": hashlib.sha256(before).hexdigest(), "groups_per_update": 2,
    }
    assert hero_path.read_bytes() == before
    assert config.train.groups_per_update == 2
    with pytest.raises(FrozenInstanceError):
        stream.worlds = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        stream.worlds[0].gm_model = "silently-remapped"  # type: ignore[misc]


def test_hero_uses_the_actual_slate_loader_exactly_once(tmp_path: Path, hero_path: Path) -> None:
    config = _hero_config(tmp_path, hero_path)
    calls = []

    def observe(frame, event, arg):  # type: ignore[no-untyped-def]
        if event == "call" and frame.f_code is load_hero_slate.__code__:
            calls.append(frame.f_locals["path"])

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        stream = select_world_stream(config)
    finally:
        sys.setprofile(previous)
    assert calls == [hero_path]
    assert len(stream.worlds) == 10


def test_checked_in_hero_slate_keeps_its_existing_models_and_absolute_provenance(tmp_path: Path) -> None:
    relative = Path("hero_runs/first_hero_baby_v1.json")
    slate = load_hero_slate(PROJECT_ROOT / relative)
    config = _hero_config(tmp_path, relative)
    stream = select_world_stream(config)
    assert [world.world_file for world in stream.worlds] == [world.world_file for world in slate.worlds]
    assert [world.gm_model for world in stream.worlds] == [world.simulator.model for world in slate.worlds]
    assert "gpt-5.6-sol" in {world.gm_model for world in stream.worlds}
    assert stream.provenance["hero_slate_path"] == str((PROJECT_ROOT / relative).resolve())


def test_hero_group_mismatch_is_rejected_not_rewritten(tmp_path: Path, hero_path: Path) -> None:
    config = _hero_config(tmp_path, hero_path, "[train]\ngroups_per_update = 1\n")
    with pytest.raises(ValueError, match="groups_per_update.*match.*hero slate"):
        select_world_stream(config)
    assert config.train.groups_per_update == 1


@pytest.mark.parametrize("corruption,message", [
    ("duplicate_order", "world_order contains duplicate"),
    ("duplicate_assignment", "multiple simulator assignments"),
    ("wrong_provider", "must declare provider"),
    ("wrong_count", "simulator counts changed"),
])
def test_hero_loader_validation_is_not_bypassed(
    tmp_path: Path, hero_path: Path, corruption: str, message: str,
) -> None:
    payload = json.loads(hero_path.read_text(encoding="utf-8"))
    if corruption == "duplicate_order":
        payload["world_order"][1] = payload["world_order"][0]
    elif corruption == "duplicate_assignment":
        payload["assignments"]["terra"].append(payload["assignments"]["sol"][0])
    elif corruption == "wrong_provider":
        payload["simulators"]["sol"]["provider"] = "openrouter"
    else:
        payload["expected_simulator_counts"]["sonnet"] += 1
    hero_path.write_text(json.dumps(payload), encoding="utf-8")
    config = _hero_config(tmp_path, hero_path)
    with pytest.raises(RuntimeError, match=message):
        select_world_stream(config)


def test_hero_identity_keeps_simulator_fields_distinct(tmp_path: Path, hero_path: Path) -> None:
    config = _hero_config(tmp_path, hero_path)
    stream = select_world_stream(config)

    def identity(world: SelectedWorld):
        return group_identity(
            world.world_file, gm_runtime=world.gm_runtime, gm_model=world.gm_model,
            gm_provider=world.gm_provider, gm_reasoning_effort=world.gm_reasoning_effort,
            max_world_steps=config.gm.max_world_steps, player_model=config.player.base_model,
            renderer_name=config.player.thinking_renderer, max_tokens=config.player.max_tokens,
            temperature=config.player.temperature, top_p=config.player.top_p,
        )

    identities = [identity(world) for world in stream.worlds]
    assert len({item.key for item in identities}) == len(stream.worlds)
    for world, item in zip(stream.worlds, identities, strict=True):
        assert (item.gm_runtime, item.gm_model, item.gm_provider, item.gm_reasoning_effort) == (
            world.gm_runtime, world.gm_model, world.gm_provider, world.gm_reasoning_effort,
        )
    # Isolate the GM identity: identical world bytes still must not mix groups.
    same_world = stream.worlds[0].world_file
    assert len({identity(replace(world, world_file=same_world)).key for world in stream.worlds}) == 4


INVALID_SETTINGS = [
    ("worlds", "count", 0),
    ("worlds", "seed", True),
    ("gm", "max_world_steps", 0),
    ("gm", "timeout_seconds", 0),
    ("gm", "timeout_seconds", float("inf")),
    ("gm", "api_idle_timeout_seconds", 0),
    ("gm", "api_idle_timeout_seconds", float("nan")),
    ("gm", "api_idle_timeout_seconds", True),
    ("player", "lora_rank", 0),
    ("player", "max_tokens", 0),
    ("player", "max_tokens", True),
    ("player", "prompt_timeout_seconds", -1),
    ("player", "prompt_timeout_seconds", 1.0),
    ("player", "prompt_timeout_seconds", 900.0),
    ("player", "prompt_timeout_seconds", float("nan")),
    ("player", "temperature", -0.1),
    ("player", "temperature", 2.1),
    ("player", "temperature", float("nan")),
    ("player", "top_p", 0),
    ("player", "top_p", 1.01),
    ("player", "top_p", float("inf")),
    ("train", "steps", 0),
    ("train", "steps", 1.5),
    ("train", "steps", True),
    ("train", "steps", "3"),
    ("train", "groups_per_update", 0),
    ("train", "checkpoint_every", 0),
    ("train", "checkpoint_every", -1),
    ("train", "checkpoint_every", True),
    ("train", "checkpoint_every", 1.5),
    ("train", "learning_rate", 0),
    ("train", "learning_rate", 1.01),
    ("train", "learning_rate", float("inf")),
    ("train", "cispo_clip_low", -1),
    ("train", "cispo_clip_low", float("nan")),
    ("train", "cispo_clip_high", 0),
    ("train", "cispo_clip_high", float("inf")),
    ("train", "kl_penalty_coef", -1),
    ("train", "kl_penalty_coef", float("nan")),
    ("train", "kl_discount_factor", -0.1),
    ("train", "kl_discount_factor", 1.1),
    ("train", "kl_discount_factor", float("nan")),
    ("adaptive", "initial_attempts", 0),
    ("adaptive", "maximum_attempts", 13),
    ("adaptive", "minimum_trainable", 9),
    ("adaptive", "minimum_reward_range", -0.1),
    ("adaptive", "minimum_reward_range", 1.1),
    ("adaptive", "minimum_reward_range", float("nan")),
    ("adaptive", "low_spread_top_up", 5),
    ("adaptive", "parallelism", 0),
    ("adaptive", "parallelism_per_provider", 9),
    ("adaptive", "provider_outage", "unknown"),
]


@pytest.mark.parametrize("section,key,value", INVALID_SETTINGS)
@pytest.mark.parametrize("source", ["toml", "dataclass"])
def test_invalid_numeric_settings_fail_for_loader_and_direct_configs(
    tmp_path: Path, section: str, key: str, value: object, source: str,
) -> None:
    with pytest.raises(ValueError, match=key):
        if source == "toml":
            literal = str(value) if isinstance(value, float) else json.dumps(value)
            load_train_config(_toml(tmp_path, f"[{section}]\n{key} = {literal}\n"))
        else:
            validate_train_config(_changed(_direct_config(), section, key, value))


def test_valid_boundaries_and_integer_float_coercion(tmp_path: Path) -> None:
    config = load_train_config(_toml(tmp_path,
        "[train]\nsteps = 1\ngroups_per_update = 1\ncheckpoint_every = 1\nlearning_rate = 1\n"
        "kl_discount_factor = 1\n[player]\ntemperature = 0\ntop_p = 1\nmax_tokens = 1\n"
        "[adaptive]\ninitial_attempts = 2\nmaximum_attempts = 2\nminimum_trainable = 2\n"
        "low_spread_top_up = 1\nparallelism = 1\nparallelism_per_provider = 1\n"))
    assert config.train.learning_rate == 1.0
    assert type(config.train.learning_rate) is float
    validate_train_config(config)
    validate_train_config(_direct_config())
    with pytest.raises(ValueError, match="cispo_clip_high"):
        validate_train_config(_changed(config, "train", "cispo_clip_low", 5.0))


@pytest.mark.parametrize("text,message", [
    ('[worlds]\nselect = "ids"\nids = "abc"\n', "array of strings"),
    ('[worlds]\nselect = "ids"\nids = 1\n', "array of strings"),
    ('[worlds]\nselect = "ids"\nids = {a = "a"}\n', "array of strings"),
    ('[worlds]\nselect = "ids"\nids = []\n', "needs"),
    ('[worlds]\nselect = "ids"\nids = ["a", 1]\n', "only strings"),
    ('[worlds]\nselect = "ids"\nids = ["a", true]\n', "only strings"),
    ('[worlds]\nselect = "ids"\nids = [["a"]]\n', "only strings"),
    ('[worlds]\nselect = "ids"\nids = [""]\n', "non-empty"),
    ('[worlds]\nselect = "ids"\nids = [" "]\n', "non-empty"),
    ('[worlds]\nselect = "ids"\nids = [" a"]\n', "whitespace"),
    ('[worlds]\nselect = "ids"\nids = ["a", "a"]\ncount = 1\n', "Duplicate"),
    ('[worlds]\nselect = "hero"\n', "hero_slate"),
    ('[worlds]\nhero_slate = "a.json"\n', "select = 'hero'"),
    ('[worlds]\nselect = "typo"\n', "select must be"),
])
def test_real_toml_loader_rejects_malformed_selection_early(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_train_config(_toml(tmp_path, text))


@pytest.mark.parametrize("ids", ["abc", ["a"], (1,), (True,), ("",), (" ",), ("a", "a")])
def test_direct_world_ids_are_typed_nonempty_and_unique(ids: object) -> None:
    config = _changed(_direct_config(), "worlds", "ids", ids)
    with pytest.raises(ValueError, match="ids"):
        select_world_stream(config)


@pytest.mark.parametrize("name", SUPPORTED_TINKER_BASE_MODELS)
def test_exact_m3_allowlist_is_accepted_offline(tmp_path: Path, name: str) -> None:
    config = load_train_config(_toml(tmp_path, f"[player]\nbase_model = {json.dumps(name)}\n"))
    validate_train_config(config)
    assert config.player.base_model == name
    assert select_world_stream(config).worlds


@pytest.mark.parametrize("section,key,value,message", [
    ("player", "base_model", "Qwen/Qwen3.8-27B:peft:65536", "Unsupported Tinker base model"),
    ("player", "base_model", "Qwen/Qwen3.8-27B ", "Unsupported Tinker base model"),
    ("player", "base_model", "Qwen/Qwen3.8-235B", "Unsupported Tinker base model"),
    ("player", "thinking_renderer", "qwen3", "thinking_renderer"),
    ("player", "plain_renderer", "qwen3", "plain_renderer"),
    ("gm", "runtime", "claude", "confined Codex"),
    ("gm", "runtime", "pi", "confined Codex"),
    ("gm", "provider", "openrouter", "codex-cli"),
    ("gm", "model", " ", "non-empty"),
    ("gm", "reasoning_effort", "typo", "reasoning_effort"),
])
def test_model_renderer_and_confinement_contracts_fail_offline(
    tmp_path: Path, section: str, key: str, value: str, message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        load_train_config(_toml(tmp_path, f"[{section}]\n{key} = {json.dumps(value)}\n"))
    with pytest.raises(ValueError, match=message):
        validate_train_config(_changed(_direct_config(), section, key, value))


def test_run_id_origin_is_explicit_nonserialized_invocation_metadata(tmp_path: Path) -> None:
    implicit = load_train_config(None, log_dir=tmp_path / "run")
    assert implicit.run_id_generated is True
    explicit = load_train_config(_toml(tmp_path, f'[run]\nrun_id = "{implicit.run.run_id}"\n'),
                                 log_dir=tmp_path / "run")
    assert explicit.run_id_generated is False
    assert implicit == explicit
    assert "run_id_generated" not in repr(implicit)
    assert implicit.to_jsonable() == explicit.to_jsonable()
    assert set(implicit.to_jsonable()) == {"run", "worlds", "gm", "player", "train", "adaptive", "compute", "mlx", "world_overrides", "optimizer", "credit", "retry", "telemetry"}
    assert "run_id_generated" not in json.dumps(implicit.to_jsonable())
    assert TrainConfig().run_id_generated is False
    assert load_train_config(_toml(tmp_path, '[run]\nlog_dir = "somewhere"\n')).run_id_generated is True


@pytest.mark.parametrize("text,message", [
    ('run_id_generated = true\n', "Unknown section"),
    ('[run_id_generated]\n', "Unknown section"),
    ('[run]\nrun_id_generated = true\n', "Unknown key"),
    ('[run]\nrun_id = ""\n', "non-empty"),
    ('[run]\nrun_id = " "\n', "non-empty"),
])
def test_run_id_marker_cannot_be_set_in_toml(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_train_config(_toml(tmp_path, text))


@pytest.fixture
def refusing_sdk():
    """Only the external SDK boundary is supplied; no production function is mocked."""
    calls = []

    class ServiceClient:
        def __init__(self):
            calls.append("ServiceClient")
            raise AssertionError("Service authentication must not run for invalid config/worlds")

    return SimpleNamespace(tinker=SimpleNamespace(ServiceClient=ServiceClient)), calls


@pytest.mark.parametrize("section,key,value,message", [
    ("train", "checkpoint_every", 0, "checkpoint_every"),
    ("train", "steps", 0, "steps"),
    ("player", "base_model", "unsupported/model", "Unsupported Tinker base model"),
    ("player", "thinking_renderer", "wrong", "thinking_renderer"),
    ("gm", "runtime", "claude", "confined Codex"),
])
def test_real_trainer_validates_direct_configs_before_service(
    tmp_path: Path, refusing_sdk, section: str, key: str, value: object, message: str,
) -> None:
    from shellsmith.train.trainer import run_train

    deps, calls = refusing_sdk
    config = load_train_config(None, log_dir=tmp_path / "not-started")
    config = _changed(config, section, key, value)
    with pytest.raises(ValueError, match=message):
        asyncio.run(run_train(config, deps=deps))
    assert calls == []
    assert not config.resolved_log_dir().exists()


def test_duplicate_ids_fail_before_collection_or_external_sdk(tmp_path: Path, refusing_sdk) -> None:
    from shellsmith.train.trainer import run_train

    deps, calls = refusing_sdk
    world_id = load_catalog_entries()[0].id
    config = load_train_config(None, log_dir=tmp_path / "not-started")
    config = replace(config, worlds=replace(config.worlds, select="ids", ids=(world_id, world_id), count=1))
    with pytest.raises(ValueError, match="Duplicate.*ids"):
        asyncio.run(run_train(config, deps=deps))
    assert calls == []
    assert not config.resolved_log_dir().exists()


def test_real_trainer_rejects_unsupported_hero_assignments_without_remapping(
    tmp_path: Path, hero_path: Path, refusing_sdk,
) -> None:
    from shellsmith.train.trainer import run_train

    deps, calls = refusing_sdk
    config = _hero_config(tmp_path, hero_path)
    config = _changed(config, "run", "log_dir", str(tmp_path / "not-started"))
    with pytest.raises((ValueError, RuntimeError), match="[Cc]odex|[Cc]onfin|[Uu]nsupported.*runtime"):
        asyncio.run(run_train(config, deps=deps))
    assert calls == []
    assert not (config.resolved_log_dir() / "trainer_state.json").exists()
