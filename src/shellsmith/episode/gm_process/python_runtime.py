"""An explicit, read-only Python installation for confined MLX world tools."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys


@dataclass(frozen=True)
class PythonRuntime:
    executable: Path
    prefix: Path

    def validate(self) -> None:
        """Reject broad roots, symlink escapes and another account's runtime."""
        prefix, executable = self.prefix, self.executable
        home = Path.home().resolve()
        if not prefix.is_absolute() or prefix.resolve(strict=True) != prefix:
            raise ValueError("GM Python runtime prefix must be an absolute, canonical installation directory")
        if prefix == home or prefix in home.parents or prefix == Path(prefix.anchor):
            raise ValueError("GM Python runtime prefix cannot grant access to the home directory or a broad filesystem root")
        resolved = executable.resolve(strict=True)
        if (not executable.is_absolute() or not resolved.is_relative_to(prefix / "bin")
                or resolved.parent != prefix / "bin" or not resolved.name.startswith("python")):
            raise ValueError("GM Python executable must belong to the approved runtime's bin directory")
        stdlib = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
        if (not stdlib.is_dir() or not stdlib.resolve(strict=True).is_relative_to(prefix)
                or not (stdlib / "os.py").is_file()):
            raise ValueError("GM Python runtime prefix must contain its Python standard library")
        for path in (prefix, prefix / "bin", stdlib, resolved):
            metadata = path.stat()
            if metadata.st_uid != os.getuid() or metadata.st_mode & 0o022:
                raise ValueError("GM Python runtime must be owned by this account and not writable by other accounts")
        if not os.access(resolved, os.X_OK):
            raise ValueError("GM Python runtime executable is not executable")

    @property
    def shell_path(self) -> str:
        return str(self.prefix / "bin") + os.pathsep + "/usr/bin:/bin:/usr/sbin:/sbin"

    def probe(self) -> None:
        """Verify world-computation modules without importing the training stack."""
        self.validate()
        result = subprocess.run(
            [str(self.executable), "-I", "-B", "-c", "import json, sqlite3, venv"],
            env={"PATH": self.shell_path}, capture_output=True, timeout=15,
        )
        if result.returncode:
            raise ValueError("GM Python runtime cannot import json, sqlite3 and venv; repair the dedicated Python installation")


def current_python_runtime() -> PythonRuntime:
    """Use the current trainer's actual installation, resolving venv aliases.

    Conda environments have their own standard library and base prefix. A venv
    aliases a base installation; granting only that installation keeps unrelated
    editable packages and sibling environments outside the GM read permission.
    """
    runtime = PythonRuntime(Path(sys.executable).resolve(strict=True), Path(sys.base_prefix).resolve(strict=True))
    runtime.validate()
    return runtime
