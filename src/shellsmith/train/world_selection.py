"""Resolve a training queue offline without discarding per-world GM identities.

Catalog selection retains manifest order, seeded random order, or explicit ID
order, limited by ``worlds.count``. Hero selection delegates exactly once to the
canonical slate loader and uses its entire immutable queue and simulator map;
``worlds.count`` and ``worlds.seed`` apply only to catalog selection. Training
must match the slate's grouping rather than silently rewriting its config.

This module never creates a service client or remaps an unsupported simulator.
The trainer checks selected runtimes before authenticating with Tinker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import random
import re
from typing import Any

from shellsmith.campaign.hero_slate import load_hero_slate
from shellsmith.catalog import load_catalog_entries
from shellsmith.episode.world_prompt_profile import FrozenPromptInputs
from shellsmith.paths import CATALOG_MANIFEST
from shellsmith.process_io import fsync_directory, write_json
from shellsmith.train.config import TrainConfig, validate_train_config
from shellsmith.train.name_budget import validate_component_name
from shellsmith.world_file import WorldFileSpec, load_world_file

# World directories share a segment root with campaign bookkeeping and the
# endpoint's sample records. They must not claim those existing paths.
_RESERVED_WORLD_IDS = frozenset({
    "campaign_manifest.json", "campaign_report.json", "campaign_report.md",
    "capture_index.json", "player_records",
})


@dataclass(frozen=True)
class SelectedWorld:
    world_file: Path
    gm_runtime: str
    gm_model: str
    gm_provider: str
    gm_reasoning_effort: str
    expected_sha256: str | None = None
    expected_manifest_sha256: str | None = None
    world_id: str = field(init=False)

    def __post_init__(self) -> None:
        # Pin the canonical contract identity, not an independent filename
        # guess. dataclasses.replace also recomputes it for a changed path.
        object.__setattr__(self, "world_id", load_world_file(self.world_file).id)


@dataclass(frozen=True)
class WorldStream:
    worlds: tuple[SelectedWorld, ...]
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        # Validate the actual resolved queue, including hero additional_worlds,
        # before the trainer authenticates or creates episode/output paths.
        seen: dict[str, SelectedWorld] = {}
        for world in self.worlds:
            world_id = world.world_id
            if (not isinstance(world_id, str) or world_id in {".", ".."}
                    or re.fullmatch(r"[A-Za-z0-9._-]+", world_id) is None):
                raise ValueError(
                    f"Selected world ID {world_id!r} from {str(world.world_file)!r} must be an "
                    "endpoint-safe non-dot path segment: letters, digits, '.', '_' or '-'."
                )
            path_key = world_id.casefold()
            if path_key in _RESERVED_WORLD_IDS:
                raise ValueError(f"Selected world ID {world_id!r} uses a reserved campaign path.")
            if path_key in seen:
                previous = seen[path_key]
                if previous.world_id == world_id:
                    raise ValueError(
                        f"Duplicate selected world ID {world_id!r}: {previous.world_file} and {world.world_file}."
                    )
                raise ValueError(
                    f"Selected world IDs {previous.world_id!r} and {world_id!r} collide on case-insensitive paths."
                )
            seen[path_key] = world


@dataclass(frozen=True)
class FrozenWorld:
    """One pre-auth read, shared by snapshot identity, classification and launch.

    Keep the original canonical source path as provenance, not as a later read
    instruction. Raw bytes preserve the file hash (including line endings); the
    spec uses the same universal-newline decoding as load_world_file.
    """

    selected: SelectedWorld
    source_path: Path
    contents: bytes = field(repr=False)
    spec: WorldFileSpec = field(repr=False)
    prompt_inputs: FrozenPromptInputs = field(repr=False)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.contents).hexdigest()

    def group_identity(self, **settings: Any) -> Any:
        # Use the same schema as identity.group_identity, without reopening any
        # world, template, AGENTS or profile-tail source after preflight.
        from shellsmith.train import identity

        binding = self.prompt_inputs.for_model(runtime=settings["gm_runtime"], model=settings["gm_model"])
        return identity.RolloutGroupIdentity(
            world_id=self.spec.id, world_sha256=self.sha256, gm_prompt_profile=binding.profile.name,
            gm_prompt_sha256=binding.sha256, **settings)

    def launch_path(self, root: Path) -> Path:
        # A per-world subdirectory cannot collide with another selected world
        # or campaign bookkeeping, and keeps the canonical filename-derived ID.
        return root / self.spec.id / "input" / f"{self.spec.id}.md"

    def validate_names(self, root: Path) -> None:
        path = self.launch_path(root)
        validate_component_name(path.name, label="Frozen world input filename", directory=path.parent)
        validate_component_name("provenance.json", label="Frozen world provenance filename", directory=path.parent)
        validate_component_name("prompt_provenance.json", label="Frozen prompt provenance filename", directory=path.parent)
        for prompt in self.prompt_inputs.files:
            validate_component_name(prompt.path.name, label="Frozen prompt filename", directory=path.parent / "prompts")

    def stage(self, root: Path) -> Path:
        """Durably publish a private read-only copy after startup state exists.

        No hardlinks/symlinks to sources, no repository temporary files. A failed
        staging operation is ordinary segment work, never a snapshot-only startup
        artifact. The next trainer retry owns a fresh segment under its run lock.
        """
        path = self.launch_path(root)
        path.parent.mkdir(parents=True, exist_ok=False)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(self.contents)
            handle.flush()
            os.fsync(handle.fileno())
        write_json(path.parent / "provenance.json", {
            "world_id": self.spec.id, "world_sha256": self.sha256,
            "source_world_file": str(self.source_path), "frozen_world_file": str(path),
        })
        prompt_root = path.parent / "prompts"
        prompt_root.mkdir()
        for prompt in self.prompt_inputs.files:
            descriptor = os.open(prompt_root / prompt.path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(prompt.contents)
                handle.flush()
                os.fsync(handle.fileno())
        write_json(path.parent / "prompt_provenance.json", {
            "files": [{"name": prompt.path.name, "source_path": str(prompt.source_path),
                       "sha256": hashlib.sha256(prompt.contents).hexdigest()}
                      for prompt in self.prompt_inputs.files],
            "profiles": [{"runtime": binding.runtime, "model": binding.model,
                          "gm_prompt_profile": binding.profile.name, "gm_prompt_sha256": binding.sha256}
                         for binding in self.prompt_inputs.profiles],
        })
        fsync_directory(prompt_root)
        fsync_directory(path.parent)
        fsync_directory(path.parent.parent)
        fsync_directory(root)
        return path


def freeze_worlds(worlds: tuple[SelectedWorld, ...]) -> tuple[FrozenWorld, ...]:
    """Capture world AND prompt inputs once, without output or SDK clients.

    Include known backup profiles without enabling any backup runtime. Existing
    Codex-only validation and outage policy still govern executable GMs.
    """
    from shellsmith.campaign.provider_outage import resolve_backup_world
    from shellsmith.train.identity import PROMPT_PATHS

    models = [(world.gm_runtime, world.gm_model) for world in worlds]
    for world in worlds:
        try:
            backup = resolve_backup_world(runtime=world.gm_runtime, model=world.gm_model, provider=world.gm_provider)
        except RuntimeError:  # Unknown models have no configured fallback.
            continue
        models.append((backup.runtime, backup.model))
    prompt_inputs = FrozenPromptInputs.capture(PROMPT_PATHS, models=tuple(models))
    frozen = []
    manifest_sha = hashlib.sha256(CATALOG_MANIFEST.read_bytes()).hexdigest() if any(
        world.expected_manifest_sha256 for world in worlds) else None
    for world in worlds:
        path = world.world_file.expanduser().resolve()
        contents = path.read_bytes()
        if (world.expected_sha256 is not None
                and hashlib.sha256(contents).hexdigest() != world.expected_sha256):
            raise RuntimeError(f"Pinned hero world changed before freezing: {world.world_id}")
        if world.expected_manifest_sha256 is not None and world.expected_manifest_sha256 != manifest_sha:
            raise RuntimeError("Pinned hero catalog manifest changed before freezing.")
        text = contents.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        spec = WorldFileSpec.from_markdown(task_id=path.stem, markdown=text)
        if spec.id != world.world_id:
            raise RuntimeError(f"Selected world identity changed before freezing: {world.world_file}.")
        frozen.append(FrozenWorld(selected=world, source_path=path, contents=contents, spec=spec,
                                  prompt_inputs=prompt_inputs))
    return tuple(frozen)


def select_world_stream(config: TrainConfig, *, campaign_mode: str = "tinker_train") -> WorldStream:
    """Resolve exact paths, order, assignments, and source provenance offline."""
    validate_train_config(config, campaign_mode=campaign_mode)
    worlds = config.worlds
    if worlds.select == "hero":
        slate = load_hero_slate(Path(worlds.hero_slate))
        if campaign_mode != "inference" and config.train.groups_per_update != slate.groups_per_update:
            raise ValueError(
                "[train].groups_per_update must match the hero slate: "
                f"config={config.train.groups_per_update}, slate={slate.groups_per_update}."
            )
        if not slate.worlds:
            raise ValueError("The hero slate must select at least one world.")
        return WorldStream(
            worlds=tuple(
                SelectedWorld(
                    world_file=world.world_file,
                    gm_runtime=world.simulator.runtime,
                    gm_model=world.simulator.model,
                    gm_provider=world.simulator.provider,
                    gm_reasoning_effort=world.simulator.reasoning_effort,
                    expected_sha256=(slate.pinned_assets["world_sha256"][world.world_id]
                                     if slate.pinned_assets else None),
                    expected_manifest_sha256=(slate.pinned_assets["manifest_sha256"]
                                              if slate.pinned_assets else None),
                )
                for world in slate.worlds
            ),
            provenance={
                "select": "hero",
                "hero_slate_path": str(slate.path),
                "hero_slate_sha256": slate.sha256,
                "groups_per_update": slate.groups_per_update,
                **({"pinned_assets": slate.pinned_assets} if slate.pinned_assets else {}),
            },
        )

    entries = load_catalog_entries()
    if worlds.select == "ids":
        by_id = {entry.id: entry.path for entry in entries}
        missing = [world_id for world_id in worlds.ids if world_id not in by_id]
        if missing:
            raise ValueError(f"[worlds].ids not in the manifest: {', '.join(missing)}")
        chosen = [by_id[world_id] for world_id in worlds.ids]
    else:
        chosen = [entry.path for entry in entries]
        if worlds.select == "random":
            random.Random(worlds.seed).shuffle(chosen)
    gm = config.gm
    return WorldStream(
        worlds=tuple(
            SelectedWorld(
                world_file=path,
                gm_runtime=gm.runtime,
                gm_model=gm.model,
                gm_provider=gm.provider,
                gm_reasoning_effort=gm.reasoning_effort,
            )
            for path in chosen[:worlds.count]
        ),
        provenance={
            "select": worlds.select,
            "manifest_path": str(CATALOG_MANIFEST.resolve()),
            "manifest_sha256": hashlib.sha256(CATALOG_MANIFEST.read_bytes()).hexdigest(),
        },
    )
