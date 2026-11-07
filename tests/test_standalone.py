"""Shellsmith must stand alone: no imports from the original monorepo, Verifiers, Prime or High Command.

Why this exists
---------------
Shellsmith was extracted from a larger monorepo. A single leftover import would only fail at
runtime on a machine without that checkout. These tests parse every Python file
and fail on any forbidden import, including lazy imports inside functions and
``importlib`` module strings.
"""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (REPO / "src", REPO / "tests")

# Top-level modules that belong to the original monorepo or to the dropped integrations.
FORBIDDEN_TOP_LEVEL = {
    "backend",
    "rl",
    "simulator",
    "merlin",
    "research",
    "verifiers",
    "prime",
    "prime_rl",
    "prime_sandboxes",
    "high_command",
}


def _python_files() -> list[Path]:
    return sorted(path for root in PYTHON_ROOTS for path in root.rglob("*.py"))


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """Return (line, module) for every absolute import and import_module("...") call."""

    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "attr", getattr(node.func, "id", None)) == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
    return found


def test_python_sources_exist() -> None:
    assert len(_python_files()) > 50


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO)))
def test_no_forbidden_imports(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [
        f"{path.relative_to(REPO)}:{line}: {module}"
        for line, module in _imported_modules(tree)
        if module.split(".")[0] in FORBIDDEN_TOP_LEVEL
    ]
    assert not offenders, "Forbidden imports:\n" + "\n".join(offenders)


def test_no_monorepo_module_paths_in_source() -> None:
    """Catch ``python -m backend.research...`` strings and monorepo file paths."""

    offenders = []
    for path in _python_files():
        if path.name == "test_standalone.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "backend.research" in line or "backend/research" in line:
                offenders.append(f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert not offenders, "Monorepo paths:\n" + "\n".join(offenders)


def test_default_log_roots_do_not_collide_between_posix_users(tmp_path, monkeypatch) -> None:
    from shellsmith import paths

    monkeypatch.setattr(paths.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(paths.os, "getuid", lambda: 1001, raising=False)
    first = paths._default_log_root()
    monkeypatch.setattr(paths.os, "getuid", lambda: 1002)
    second = paths._default_log_root()
    assert first == tmp_path / "shellsmith-1001"
    assert second == tmp_path / "shellsmith-1002"
    first.mkdir(mode=0o700)
    second.mkdir(mode=0o700)
    assert first != second


def test_default_log_root_on_platform_without_uid(tmp_path, monkeypatch) -> None:
    from shellsmith import paths

    monkeypatch.setattr(paths.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.delattr(paths.os, "getuid", raising=False)
    assert paths._default_log_root() == tmp_path / "shellsmith"


def test_readme_model_matches_runnable_defaults() -> None:
    from shellsmith.episode.smoke import _parser

    model = _parser().parse_args([]).gm_model
    assert f"Default: Codex CLI `{model}`." in (REPO / "README.md").read_text()


def test_hero_slates_do_not_advertise_removed_conductor() -> None:
    guide = (REPO / "hero_runs/README.md").read_text()
    assert "Historical" in guide
    assert "not shipped" in guide
    assert "python -m shellsmith.conductor" not in guide
    assert "backend/research" not in guide


def test_architecture_vat_commands_target_existing_package() -> None:
    guide = (REPO / "docs/ARCHITECTURE.md").read_text()
    for command in ("npm test", "npm run check"):
        assert f"{command} --prefix src/shellsmith/episode/vat_runner" in guide
    assert (REPO / "src/shellsmith/episode/vat_runner/package.json").is_file()


def test_import_shellsmith_loads_no_forbidden_modules() -> None:
    script = (
        "import sys, shellsmith, shellsmith.paths, shellsmith.credit.to_tinker_bridge, "
        "shellsmith.episode.smoke, shellsmith.train.stream_engine\n"
        f"bad = sorted(m for m in sys.modules if m.split('.')[0] in {sorted(FORBIDDEN_TOP_LEVEL)!r})\n"
        "print(','.join(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""
