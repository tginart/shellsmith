"""Codex provenance and binding through real PATH/filesystem and offline training.

Only external Tinker/Cookbook/torch, Codex and Pi are fixtures. In particular,
resolution, snapshots, HTTP/Node episodes, harvesting and training are real.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from shellsmith.episode import smoke
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.process_io import write_json
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def _config(tmp_path):
    sdk = tmp_path / "external-sdk.js"
    sdk.write_text("export {};\n")
    config = load_train_config(None, log_dir=tmp_path / "run")
    return replace(
        config,
        worlds=replace(config.worlds, select="ids", ids=("checkout_idempotency_001",), count=1),
        player=replace(config.player, pi_sdk_module_path=str(sdk)),
    )


def _snapshot(config):
    return trainer.snapshot(config, trainer.select_world_stream(config))


def _tools(tmp_path, monkeypatch):
    """Keep real git/Node available, without leaking the host's Codex into PATH."""
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("git", "node"):
        selected = shutil.which(name)
        if selected:
            (tools / name).symlink_to(Path(selected).resolve())
    monkeypatch.setenv("PATH", str(tools))
    return tools


def _executable(path, contents=b"#!/bin/sh\n# original Codex bytes\nexit 97\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    path.chmod(0o755)
    return path


def _file_identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _save(config):
    saved = _snapshot(config)
    state = {
        "run_id": config.run.run_id,
        "checkpoints": [{"kind": "state", "path": "tinker://full", "sampler_path": "tinker://sampler",
                         "step": 1, "next_world_offset": 1}],
    }
    write_json(config.resolved_log_dir() / trainer.STATE_FILE, state)
    write_json(config.resolved_log_dir() / trainer.SNAPSHOT_FILE, saved)
    return state, saved


class _NoExternalSDK:
    def __getattr__(self, name):
        pytest.fail(f"Codex identity checks must precede external dependency access: {name}")


def _assert_resume_rejected(config, state, saved):
    current = _snapshot(config)
    assert current["codex"] != saved["codex"]
    assert {key: value for key, value in current.items() if key != "codex"} == {
        key: value for key, value in saved.items() if key != "codex"
    }, "Only actual Codex filesystem identity may explain this rejection"
    with pytest.raises(RuntimeError, match="changed since"):
        trainer._resume_config(config=config, state=state, saved=saved, current=current,
                               log_dir=config.resolved_log_dir())
    before = {path: path.read_bytes() for path in config.resolved_log_dir().rglob("*") if path.is_file()}
    with pytest.raises(RuntimeError, match="changed since"):
        asyncio.run(trainer.run_train(config, deps=_NoExternalSDK()))
    assert before == {path: path.read_bytes() for path in config.resolved_log_dir().rglob("*") if path.is_file()}
    return current


def test_snapshot_resolves_the_smoke_parser_executable_without_executing_it(tmp_path, monkeypatch):
    tools = _tools(tmp_path, monkeypatch)
    selected = _executable(tmp_path / "installed" / "codex", b"#!/bin/sh\n# binary comment \xff\nexit 97\n")
    (tools / "codex").symlink_to(selected)
    identity = _snapshot(_config(tmp_path))["codex"]
    assert identity["executable_path"] == str(Path(smoke._parser().parse_args([]).codex_bin).resolve())
    assert identity == {
        "executable_path": str(selected.resolve()),
        "executable_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
        "npm": None,
    }


def test_resume_rejects_changed_codex_bytes_with_same_path_size_and_mtime(tmp_path, monkeypatch):
    tools = _tools(tmp_path, monkeypatch)
    selected = _executable(tools / "codex")
    config = _config(tmp_path)
    state, saved = _save(config)
    previous = selected.stat()
    selected.write_bytes(selected.read_bytes().replace(b"original", b"modified"))
    os.utime(selected, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert selected.stat().st_size == previous.st_size
    current = _assert_resume_rejected(config, state, saved)
    assert current["codex"]["executable_path"] == saved["codex"]["executable_path"]
    assert current["codex"]["executable_sha256"] != saved["codex"]["executable_sha256"]


@pytest.mark.parametrize("changed", ["path", "file-symlink", "directory-symlink", "relative-path-cwd"])
def test_resume_rejects_reselected_codex_even_with_identical_bytes(tmp_path, monkeypatch, changed):
    tools = _tools(tmp_path, monkeypatch)
    first = _executable(tmp_path / "first" / "bin" / "codex")
    second = _executable(tmp_path / "second" / "bin" / "codex", first.read_bytes())
    selected = tools / "selected"
    if changed == "path":
        monkeypatch.setenv("PATH", str(first.parent) + os.pathsep + str(tools))
    elif changed == "relative-path-cwd":
        monkeypatch.chdir(first.parent.parent)
        monkeypatch.setenv("PATH", "bin" + os.pathsep + str(tools))
    elif changed == "file-symlink":
        selected = tools / "codex"
        selected.symlink_to(first)
    else:
        selected.symlink_to(first.parent, target_is_directory=True)
        monkeypatch.setenv("PATH", str(selected) + os.pathsep + str(tools))
    config = _config(tmp_path)
    state, saved = _save(config)
    if changed == "path":
        monkeypatch.setenv("PATH", str(second.parent) + os.pathsep + str(tools))
    elif changed == "relative-path-cwd":
        monkeypatch.chdir(second.parent.parent)
    else:
        selected.unlink()
        selected.symlink_to(second if changed == "file-symlink" else second.parent,
                            target_is_directory=changed == "directory-symlink")
    current = _assert_resume_rejected(config, state, saved)
    assert saved["codex"]["executable_path"] == str(first.resolve())
    assert current["codex"]["executable_path"] == str(second.resolve())
    assert current["codex"]["executable_sha256"] == saved["codex"]["executable_sha256"]


@pytest.mark.parametrize("unusable", ["missing", "directory", "broken-symlink", "not-executable"])
def test_missing_default_codex_is_snapshot_only_and_training_fails_before_auth(tmp_path, monkeypatch, unusable):
    tools = _tools(tmp_path, monkeypatch)
    selected = tools / "codex"
    if unusable == "directory":
        selected.mkdir()
    elif unusable == "broken-symlink":
        selected.symlink_to(tmp_path / "missing")
    elif unusable == "not-executable":
        _executable(selected).chmod(0o644)
    config = _config(tmp_path)
    assert _snapshot(config)["codex"] == {"executable_path": None, "executable_sha256": None, "npm": None}
    with pytest.raises(RuntimeError, match="Codex.*(not installed|not found|executable)"):
        asyncio.run(trainer.run_train(config, deps=_NoExternalSDK()))
    assert not config.resolved_log_dir().exists()


@pytest.mark.parametrize("installed_before", [False, True])
def test_resume_rejects_codex_installation_or_removal(tmp_path, monkeypatch, installed_before):
    tools = _tools(tmp_path, monkeypatch)
    selected = tools / "codex"
    if installed_before:
        _executable(selected)
    config = _config(tmp_path)
    state, saved = _save(config)
    if installed_before:
        selected.unlink()
    else:
        _executable(selected)
    _assert_resume_rejected(config, state, saved)


def _npm_install(tmp_path, tools, *, layout="bin", location="nested"):
    if shutil.which("node") is None:
        pytest.skip("Node is needed to resolve the external npm Codex wrapper")
    platform, arch = json.loads(subprocess.check_output(
        [shutil.which("node"), "-p", "JSON.stringify([process.platform, process.arch])"], text=True))
    target = {
        ("darwin", "arm64"): "aarch64-apple-darwin", ("darwin", "x64"): "x86_64-apple-darwin",
        ("linux", "arm64"): "aarch64-unknown-linux-musl", ("linux", "x64"): "x86_64-unknown-linux-musl",
    }[(platform, arch)]
    root = tmp_path / "npm" / "node_modules" / "@openai" / "codex"
    package = root / "package.json"
    launcher = _executable(root / "bin" / "codex.js", (
        "#!/usr/bin/env node\n"
        "import path from 'node:path';\n"
        "const vendorRoot = 'vendor', targetTriple = 'target';\n"
        + (f"const binaryPath = path.join(vendorRoot, targetTriple, '{layout}', 'codex');\n" if layout == "bin"
           else "const archRoot = path.join(vendorRoot, targetTriple);\n"
                "const binaryPath = path.join(archRoot, 'codex', 'codex');\n")
        + "throw new Error('snapshot must not execute the npm launcher');\n"
    ).encode())
    package.write_text(json.dumps({"name": "@openai/codex", "version": "test-1", "type": "module",
                                   "bin": {"codex": "bin/codex.js"}}))
    (tools / "codex").symlink_to(launcher)
    native_package = None
    if location == "vendor":
        vendor = root / "vendor"
    else:
        modules = root / "node_modules" if location == "nested" else root.parent.parent
        native_package = modules / "@openai" / f"codex-{platform}-{arch}" / "package.json"
        native_package.parent.mkdir(parents=True)
        native_package.write_text(json.dumps({"name": "@openai/codex", "version": "test-native-1"}))
        vendor = native_package.parent / "vendor"
    native = _executable(vendor / target / layout / "codex")
    return launcher, package, native, native_package


@pytest.mark.parametrize("layout", ["bin", "codex"])
@pytest.mark.parametrize("location", ["nested", "sibling", "vendor"])
def test_npm_native_upgrade_is_detected_without_launcher_or_version_change(tmp_path, monkeypatch, layout, location):
    tools = _tools(tmp_path, monkeypatch)
    launcher, package, native, native_package = _npm_install(tmp_path, tools, layout=layout, location=location)
    config = _config(tmp_path)
    state, saved = _save(config)
    assert saved["codex"]["executable_path"] == str(launcher.resolve())
    assert saved["codex"]["npm"] == {
        "package_json": _file_identity(package),
        "platform_package_json": _file_identity(native_package) if native_package else None,
        "native_executable": _file_identity(native),
    }
    native.write_bytes(native.read_bytes() + b"# upgraded native bytes, unchanged version\n")
    current = _assert_resume_rejected(config, state, saved)
    assert current["codex"]["executable_sha256"] == saved["codex"]["executable_sha256"]
    assert current["codex"]["npm"]["native_executable"] != saved["codex"]["npm"]["native_executable"]


@pytest.mark.parametrize("changed", ["launcher-metadata", "native-metadata", "native-symlink"])
def test_npm_metadata_and_native_canonical_path_are_provenance(tmp_path, monkeypatch, changed):
    tools = _tools(tmp_path, monkeypatch)
    _, package, native, native_package = _npm_install(tmp_path, tools)
    config = _config(tmp_path)
    state, saved = _save(config)
    if changed == "native-symlink":
        target = _executable(tmp_path / "another-native", native.read_bytes())
        native.unlink()
        native.symlink_to(target)
    else:
        metadata = package if changed == "launcher-metadata" else native_package
        metadata.write_text(metadata.read_text().replace("test-", "changed-"))
    _assert_resume_rejected(config, state, saved)


@pytest.mark.parametrize("broken", ["missing-native", "not-executable", "unknown-layout", "missing-node"])
def test_incomplete_npm_install_fails_locally_before_external_sdk(tmp_path, monkeypatch, broken):
    tools = _tools(tmp_path, monkeypatch)
    launcher, _, native, _ = _npm_install(tmp_path, tools)
    if broken == "missing-native":
        native.unlink()
        # A present optional package with no binary must NOT fall back to vendor/.
        fallback = launcher.parent.parent / "vendor" / native.parent.parent.name / "bin" / "codex"
        _executable(fallback)
    elif broken == "not-executable":
        native.chmod(0o644)
    elif broken == "unknown-layout":
        launcher.write_bytes(launcher.read_bytes().replace(b"'bin'", b"'unknown-layout'"))
    else:
        (tools / "node").unlink()
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="Codex"):
        _snapshot(config)
    with pytest.raises(RuntimeError, match="Codex"):
        asyncio.run(trainer.run_train(config, deps=_NoExternalSDK()))
    assert not config.resolved_log_dir().exists()


def test_generic_wrapper_does_not_claim_transitive_dependency_identity(tmp_path, monkeypatch):
    tools = _tools(tmp_path, monkeypatch)
    _executable(tools / "codex", b"#!/bin/sh\nexec ./other-tool\n")
    helper = _executable(tools / "other-tool")
    config = _config(tmp_path)
    before = _snapshot(config)
    helper.write_bytes(helper.read_bytes() + b"# not within the bounded launcher fingerprint\n")
    assert _snapshot(config) == before


@pytest.mark.parametrize("changed", ["path", "file-symlink", "directory-symlink", "relative-path-cwd"])
def test_actual_runner_uses_snapshotted_codex_after_resolution_changes(tmp_path, monkeypatch, training_sdk, changed):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    tools = tmp_path / "external-tools"
    selected = tools / "codex"
    audit = tmp_path / "codex-invocations.jsonl"
    original = _executable(tmp_path / "installed" / "codex", selected.read_bytes().replace(
        f"#!{sys.executable}\n".encode(),
        (f"#!{sys.executable}\nimport json, pathlib, sys\n"
         f"with pathlib.Path({str(audit)!r}).open('a') as log: log.write(json.dumps(sys.argv[0]) + '\\n')\n").encode(),
        1,
    ))
    selected.unlink()
    selected.symlink_to(original)
    other = tmp_path / "other"
    wrong = _executable(other / "external-tools" / "codex", (
        f"#!{sys.executable}\nraise SystemExit('Codex was reselected after snapshot')\n"
    ).encode())
    initial_path = os.environ["PATH"]
    directory_link = tmp_path / "selected-tools"
    if changed == "directory-symlink":
        directory_link.symlink_to(tools, target_is_directory=True)
        monkeypatch.setenv("PATH", str(directory_link) + os.pathsep + initial_path)
    elif changed == "relative-path-cwd":
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PATH", "external-tools" + os.pathsep + initial_path)
    calls = []

    def change_resolution_after_snapshot():
        if calls:
            return
        calls.append("external-service-boundary")
        if changed == "path":
            monkeypatch.setenv("PATH", str(wrong.parent) + os.pathsep + initial_path)
        elif changed == "relative-path-cwd":
            monkeypatch.chdir(other)
        else:
            link = selected if changed == "file-symlink" else directory_link
            link.unlink()
            link.symlink_to(wrong if changed == "file-symlink" else wrong.parent,
                            target_is_directory=changed == "directory-symlink")

    training_sdk.observe = change_resolution_after_snapshot
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert calls and state["status"] == "complete" and state["step"] == 1
    saved = json.loads((config.resolved_log_dir() / trainer.SNAPSHOT_FILE).read_text())
    assert saved["codex"]["executable_path"] == str(original.resolve())
    assert pi_audit.is_file() and pi_audit.read_text().strip()
    invocations = [json.loads(row) for row in audit.read_text().splitlines()]
    assert invocations and set(invocations) == {str(original.resolve())}
    assert sum(call[0] == "optim_step" for call in training_sdk.calls) == 1


def test_real_completed_training_rejects_codex_upgrade_before_resume_auth(tmp_path, monkeypatch, training_sdk):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    selected = Path(shutil.which("codex")).resolve()
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert state["status"] == "complete" and state["step"] == 1 and pi_audit.is_file()
    calls = list(training_sdk.calls)
    before = {path: path.read_bytes() for path in config.resolved_log_dir().rglob("*") if path.is_file()}
    selected.write_bytes(selected.read_bytes() + b"\n# external Codex upgrade\n")
    with pytest.raises(RuntimeError, match="changed since"):
        asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert training_sdk.calls == calls
    assert before == {path: path.read_bytes() for path in config.resolved_log_dir().rglob("*") if path.is_file()}
