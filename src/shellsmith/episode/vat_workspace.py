"""Lifecycle for the real, deliberately empty Player workspace.

Architecture map:
    create()        -> unique temporary root with empty `workspace/` and
                       isolated `agent-home/`
    world boot      -> writes exactly `workspace/AGENTS.md`
    Pi startup      -> reads that one file into model context
    every boundary  -> assert that no other workspace entry appeared
    close()         -> recursively delete the complete ephemeral root

The simulated filesystem does not live here. A successful simulated `write`,
`edit`, or `bash` action changes world-simulator state and returns an observation;
it must never mutate this directory.

Why this exists:
    A unique `cwd` prevents accidental project context discovery. The explicit
    one-file invariant makes an unintended real Pi tool execution immediately
    observable instead of allowing it to silently contaminate training data.
"""

from __future__ import annotations

import shutil
import tempfile
import hashlib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EphemeralVatWorkspace:
    """Paths and invariant checks for one temporary Player process."""

    root: Path
    workspace_dir: Path
    agent_home_dir: Path
    _agents_sha256: str | None = None

    @classmethod
    def create(cls, *, temp_parent: Path | None = None) -> "EphemeralVatWorkspace":
        """Create a unique empty vat without materializing public instructions.

        Main callers:
            The no-training episode launcher before world-simulator boot.

        The Game Master, not this method, later writes `AGENTS.md`.
        """
        if temp_parent is not None:
            temp_parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix="shellsmith-vat-",
                dir=str(temp_parent) if temp_parent is not None else None,
            )
        ).resolve()
        workspace_dir: Path = root.joinpath("workspace")
        agent_home_dir: Path = root.joinpath("agent-home")
        workspace_dir.mkdir()
        agent_home_dir.mkdir()
        vat = cls(
            root=root,
            workspace_dir=workspace_dir,
            agent_home_dir=agent_home_dir,
        )
        vat.assert_empty()
        return vat

    @property
    def agents_path(self) -> Path:
        return self.workspace_dir.joinpath("AGENTS.md")

    def workspace_entries(self) -> tuple[Path, ...]:
        """Return deterministic direct children for audit and invariant checks."""
        if not self.workspace_dir.is_dir():
            raise RuntimeError(
                f"Vat workspace disappeared or is not a directory: {self.workspace_dir}."
            )
        return tuple(sorted(self.workspace_dir.iterdir(), key=lambda path: path.name))

    def assert_empty(self) -> None:
        """Fail unless world boot has not materialized any public file yet."""
        entries = self.workspace_entries()
        if entries:
            names = [entry.name for entry in entries]
            raise RuntimeError(f"Vat workspace must be empty before boot; found {names}.")

    def assert_agents_only(self) -> None:
        """Fail unless the vat retains its original regular `AGENTS.md`.

        Main callers:
            World boot validation, Pi startup, every completed prompt, and final
            episode cleanup.

        Why this exists:
            Extra physical files are inert because all Player tools are
            virtual. `AGENTS.md` is different: Pi consumes it as real context, so
            replacement or mutation must fail immediately.
        """
        entries = self.workspace_entries()
        agents_path = self.agents_path
        if not agents_path.exists():
            names = [entry.name for entry in entries]
            raise RuntimeError(
                "Vat workspace must contain AGENTS.md after boot; "
                f"found {names}."
            )
        if agents_path.is_symlink() or not agents_path.is_file():
            raise RuntimeError("Vat AGENTS.md must be a regular non-symlink file.")
        content = agents_path.read_bytes()
        if not content.strip():
            raise RuntimeError("Vat AGENTS.md must not be empty.")
        digest = hashlib.sha256(content).hexdigest()
        if self._agents_sha256 is None:
            self._agents_sha256 = digest
        elif digest != self._agents_sha256:
            raise RuntimeError("Vat AGENTS.md changed after Pi initialization.")

    def close(self) -> None:
        """Delete the complete ephemeral root; repeated cleanup is harmless."""
        if self.root.exists():
            shutil.rmtree(self.root)

    def __enter__(self) -> "EphemeralVatWorkspace":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
