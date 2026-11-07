"""Select small model-specific tails for the universal world-simulator prompt.

Sol, Astra, and other untailed models share the canonical prompt, including
the surmountable-roadblocks policy. Other world models receive one short
additive profile that corrects observed tendencies without changing task truth
or creating model-specific reward groups accidentally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import stat

# Explicit files only, with bounded reads before authentication. No prompt tree
# walks, mutable caches, or late fallback to the original source files.
MAX_PROMPT_FILE_BYTES = 1_048_576
MAX_PROMPT_BUNDLE_BYTES = 8 * MAX_PROMPT_FILE_BYTES
MAX_PROMPT_FILES = 32


@dataclass(frozen=True)
class FrozenPromptFile:
    path: Path
    source_path: Path
    contents: bytes = field(repr=False)

    def text(self) -> str:
        # Match Path.read_text's universal-newline handling, without losing raw
        # bytes from identity/provenance (in particular CRLF versus LF).
        return self.contents.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class FrozenPromptProfile:
    runtime: str
    model: str
    profile: WorldPromptProfile
    sha256: str


@dataclass(frozen=True)
class FrozenPromptInputs:
    """One immutable source binding shared by identity and all episode renders."""

    files: tuple[FrozenPromptFile, ...]
    profiles: tuple[FrozenPromptProfile, ...]

    @classmethod
    def capture(cls, paths: tuple[Path, ...], *, models: tuple[tuple[str, str], ...]) -> FrozenPromptInputs:
        profiles = tuple((runtime, model, resolve_world_prompt_profile(runtime=runtime, model=model))
                         for runtime, model in dict.fromkeys(models))
        all_paths = tuple(dict.fromkeys((*paths, *(profile.tail_path for _, _, profile in profiles
                                                 if profile.tail_path is not None))))
        if len(all_paths) > MAX_PROMPT_FILES or len({path.name for path in all_paths}) != len(all_paths):
            raise ValueError("Frozen prompt inputs require bounded, uniquely named files.")
        files = []
        total = 0
        for path in all_paths:
            # Nonblocking open also refuses a FIFO without hanging preflight.
            descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(descriptor, "rb") as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError(f"Prompt input must be a regular file: {path}.")
                content = handle.read(MAX_PROMPT_FILE_BYTES + 1)
            total += len(content)
            if len(content) > MAX_PROMPT_FILE_BYTES or total > MAX_PROMPT_BUNDLE_BYTES:
                raise ValueError(f"Prompt input exceeds the frozen prompt byte budget: {path}.")
            content.decode("utf-8")  # Fail invalid templates locally, before any SDK authentication.
            files.append(FrozenPromptFile(path=path.absolute(), source_path=path.resolve(), contents=content))
        by_name = {item.path.name: item.contents for item in files}
        bindings = []
        for runtime, model, profile in profiles:
            digest = hashlib.sha256()
            for path in paths + ((profile.tail_path,) if profile.tail_path else ()):
                # Keep identity.prompt_bundle_sha256's ordered basename/NUL/raw
                # byte schema exactly; each model hashes only its own tail.
                digest.update(path.name.encode() + b"\0" + by_name[path.name] + b"\0")
            bindings.append(FrozenPromptProfile(runtime, model, profile, digest.hexdigest()))
        return cls(tuple(files), tuple(bindings))

    def for_model(self, *, runtime: str, model: str) -> FrozenPromptProfile:
        for binding in self.profiles:
            if (binding.runtime, binding.model) == (runtime, model):
                return binding
        raise RuntimeError(f"GM prompt profile was not frozen before authentication: {runtime}/{model}.")

    def read_text(self, path: Path) -> str:
        for item in self.files:
            if item.path.name == path.name:
                return item.text()
        raise RuntimeError(f"Prompt input was not frozen before authentication: {path.name}.")


# Prompt tails live at the shellsmith package root (sibling of episode/).
TAIL_ROOT = (
    Path(__file__)
    .resolve()
    .parent.parent.joinpath("prompts", "game_master", "tails")
)


@dataclass(frozen=True)
class WorldPromptProfile:
    name: str
    tail_path: Path | None

    def tail(self) -> str:
        return "" if self.tail_path is None else self.tail_path.read_text(encoding="utf-8").strip()


def resolve_world_prompt_profile(*, runtime: str, model: str) -> WorldPromptProfile:
    """Return a deterministic profile from the configured world harness/model."""
    normalized = model.lower()
    if "muse-spark" in normalized:
        return WorldPromptProfile("muse_difficulty_v2", TAIL_ROOT.joinpath("MUSE.md"))
    if "tencent/hy3" in normalized:
        return WorldPromptProfile("hy3_artifact_discipline_v1", TAIL_ROOT.joinpath("HY3.md"))
    if "grok" in normalized:
        return WorldPromptProfile("grok_calibration_v1", TAIL_ROOT.joinpath("GROK.md"))
    if runtime.lower() == "claude" and "sonnet" in normalized:
        return WorldPromptProfile("sonnet_difficulty_v2", TAIL_ROOT.joinpath("SONNET.md"))
    if runtime.lower() == "pi" and "qwen" in normalized:
        return WorldPromptProfile("qwen_difficulty_v1", TAIL_ROOT.joinpath("QWEN.md"))
    return WorldPromptProfile("canonical", None)


def render_world_agents(*, canonical_path: Path, profile: WorldPromptProfile,
                        prompt_inputs: FrozenPromptInputs | None = None) -> str:
    canonical = (prompt_inputs.read_text(canonical_path) if prompt_inputs is not None
                 else canonical_path.read_text(encoding="utf-8")).rstrip() + "\n"
    tail = (prompt_inputs.read_text(profile.tail_path).strip() if profile.tail_path else "") \
        if prompt_inputs is not None else profile.tail()
    return canonical if not tail else canonical + "\n" + tail + "\n"
