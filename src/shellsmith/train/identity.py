"""The identity a GRPO group must share: same world, GM setup, prompts and sampling.

Why this exists:
    Group-relative advantages compare rewards across episodes. That comparison
    is only meaningful when every episode used the same world file, GM model,
    GM prompts and Player sampling settings. ``RolloutGroupIdentity.key`` is the
    hash of all of them; the capture store and credit bridge refuse to mix keys,
    which also enforces "never mix GM models in one group" after an outage
    fallback. Ported from the legacy conductor.

Main callers:
    ``train.runner`` (stamps each capture) and ``train.trainer`` (records the
    identities a run used).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from shellsmith.episode.world_prompt_profile import resolve_world_prompt_profile
from shellsmith.paths import PROMPTS_DIR
from shellsmith.world_file import load_world_file

PROMPT_PATHS = (
    PROMPTS_DIR / "game_master" / "AGENTS.md",
    PROMPTS_DIR / "WORLD_BOOTSTRAP_PROMPT.md",
    PROMPTS_DIR / "WORLD_TURN_PROMPT.md",
    PROMPTS_DIR / "WORLD_FINAL_PROMPT.md",
    # Repair is executable GM input too, and must invalidate capture identity.
    PROMPTS_DIR / "WORLD_ARTIFACT_REPAIR_PROMPT.md",
)


@dataclass(frozen=True)
class RolloutGroupIdentity:
    """Fields that must remain identical for group-relative reward comparison."""

    world_id: str
    world_sha256: str
    gm_runtime: str
    gm_model: str
    gm_provider: str
    gm_reasoning_effort: str
    gm_prompt_profile: str
    gm_prompt_sha256: str
    max_world_steps: int
    player_model: str
    renderer_name: str
    max_tokens: int
    temperature: float
    top_p: float
    backend: str = "tinker"
    profile_sha256: str = ""

    @property
    def key(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_jsonable(self) -> dict[str, Any]:
        return {"key": self.key, **asdict(self)}


def prompt_bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def group_identity(
    world_file: Path, *, gm_runtime: str, gm_model: str, gm_provider: str, gm_reasoning_effort: str,
    max_world_steps: int, player_model: str, renderer_name: str, max_tokens: int, temperature: float, top_p: float,
    backend: str = "tinker", profile_sha256: str = "",
) -> RolloutGroupIdentity:
    """Build the identity for one world under one GM and Player configuration."""
    profile = resolve_world_prompt_profile(runtime=gm_runtime, model=gm_model)
    prompt_paths = PROMPT_PATHS + ((profile.tail_path,) if profile.tail_path else ())
    return RolloutGroupIdentity(
        world_id=load_world_file(world_file).id,
        world_sha256=hashlib.sha256(world_file.read_bytes()).hexdigest(),
        gm_runtime=gm_runtime,
        gm_model=gm_model,
        gm_provider=gm_provider,
        gm_reasoning_effort=gm_reasoning_effort,
        gm_prompt_profile=profile.name,
        gm_prompt_sha256=prompt_bundle_sha256(prompt_paths),
        max_world_steps=max_world_steps,
        player_model=player_model,
        renderer_name=renderer_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        backend=backend, profile_sha256=profile_sha256,
    )
