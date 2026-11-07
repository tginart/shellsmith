"""Pin the selected Pi SDK's real file identity before any external SDK work."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import threading

import pytest

from shellsmith.episode import smoke
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.process_io import write_json
from shellsmith.train import trainer
from shellsmith.train.config import load_train_config
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


def _config(tmp_path, module_path=""):
    config = load_train_config(None, log_dir=tmp_path / "run")
    return replace(
        config,
        worlds=replace(config.worlds, select="ids", ids=("checkout_idempotency_001",), count=1),
        player=replace(config.player, pi_sdk_module_path=str(module_path)),
    )


def _snapshot(config):
    return trainer.snapshot(config, trainer.select_world_stream(config))


def _save(config):
    saved = _snapshot(config)
    checkpoint = {
        "kind": "state", "path": "tinker://full", "sampler_path": "tinker://sampler", "step": 1,
        "next_world_offset": 1, "settled_world_offsets": [],
    }
    state = {
        "run_id": config.run.run_id, "status": "complete", "step": 1, "rft_updates_completed": 1,
        "next_world_offset": 1, "settled_world_offsets": [], "checkpoints": [checkpoint],
        "segments": [], "group_identities": {},
    }
    write_json(config.resolved_log_dir() / trainer.STATE_FILE, state)
    write_json(config.resolved_log_dir() / trainer.SNAPSHOT_FILE, saved)
    return state, saved


class _NoExternalSDK:
    def __getattr__(self, name):
        pytest.fail(f"SDK identity checks must precede external dependency access: {name}")


def _assert_resume_rejected(config, state, saved):
    current = _snapshot(config)
    assert current["config"] == saved["config"], "The configured path string has not changed"
    assert current["pi_sdk"] != saved["pi_sdk"]
    assert {key: value for key, value in current.items() if key != "pi_sdk"} == {
        key: value for key, value in saved.items() if key != "pi_sdk"
    }, "Only real SDK filesystem identity may explain this rejection"
    with pytest.raises(RuntimeError, match="changed since"):
        trainer._resume_config(config=config, state=state, saved=saved, current=current,
                               log_dir=config.resolved_log_dir())
    before = {path: path.read_bytes() for path in config.resolved_log_dir().rglob("*") if path.is_file()}
    with pytest.raises(RuntimeError, match="changed since"):
        asyncio.run(trainer.run_train(config, deps=_NoExternalSDK()))
    assert before == {path: path.read_bytes() for path in config.resolved_log_dir().rglob("*") if path.is_file()}
    return current


def _entry(root, contents=b"export const sdk = 'original';\n"):
    entry = root / "dist" / "index.js"
    entry.parent.mkdir(parents=True)
    (root / "package.json").write_text('{"name":"test-pi-sdk","type":"module"}\n')
    entry.write_bytes(contents)
    return entry


def test_resume_rejects_changed_custom_file_bytes_at_same_path(tmp_path):
    sdk = tmp_path / "custom-sdk.js"
    sdk.write_bytes(b"// binary comment \xff\nexport const sdk = 1;\n")
    config = _config(tmp_path, sdk)
    state, saved = _save(config)
    assert saved["pi_sdk"]["module_path"] == str(sdk.resolve())
    assert saved["pi_sdk"]["module_sha256"] == hashlib.sha256(sdk.read_bytes()).hexdigest()
    previous_stat = sdk.stat()
    sdk.write_bytes(sdk.read_bytes().replace(b"sdk = 1", b"sdk = 2"))
    os.utime(sdk, ns=(previous_stat.st_atime_ns, previous_stat.st_mtime_ns))
    current = _assert_resume_rejected(config, state, saved)
    assert current["pi_sdk"]["module_path"] == saved["pi_sdk"]["module_path"]
    assert current["pi_sdk"]["module_sha256"] != saved["pi_sdk"]["module_sha256"]


@pytest.mark.parametrize("same_bytes", [False, True])
def test_resume_rejects_same_relative_sdk_path_from_different_cwd(tmp_path, monkeypatch, same_bytes):
    first = _entry(tmp_path / "first" / "sdk")
    second = _entry(tmp_path / "second" / "sdk", first.read_bytes() if same_bytes else b"export const sdk = 2;\n")
    config = _config(tmp_path, "sdk/dist/index.js")
    monkeypatch.chdir(tmp_path / "first")
    state, saved = _save(config)
    monkeypatch.chdir(tmp_path / "second")
    current = _assert_resume_rejected(config, state, saved)
    assert saved["pi_sdk"]["module_path"] == str(first.resolve())
    assert current["pi_sdk"]["module_path"] == str(second.resolve())
    assert (current["pi_sdk"]["module_sha256"] == saved["pi_sdk"]["module_sha256"]) == same_bytes


@pytest.mark.parametrize("directory_link", [False, True])
def test_resume_rejects_retargeted_custom_sdk_symlink(tmp_path, directory_link):
    first = _entry(tmp_path / "first")
    second = _entry(tmp_path / "second", first.read_bytes())
    selected = tmp_path / "selected"
    selected.symlink_to(first.parent if directory_link else first, target_is_directory=directory_link)
    config = _config(tmp_path, selected / "index.js" if directory_link else selected)
    state, saved = _save(config)
    selected.unlink()
    selected.symlink_to(second.parent if directory_link else second, target_is_directory=directory_link)
    current = _assert_resume_rejected(config, state, saved)
    assert saved["pi_sdk"]["module_path"] == str(first.resolve())
    assert current["pi_sdk"]["module_path"] == str(second.resolve())
    assert saved["pi_sdk"]["module_sha256"] == current["pi_sdk"]["module_sha256"]


@pytest.mark.parametrize("kind", ["missing", "directory", "broken-symlink"])
def test_missing_custom_sdk_fails_snapshot_and_run_before_external_sdk(tmp_path, kind):
    sdk = tmp_path / "custom-sdk.js"
    if kind == "directory":
        sdk.mkdir()
    elif kind == "broken-symlink":
        sdk.symlink_to(tmp_path / "missing-target.js")
    config = _config(tmp_path, sdk)
    with pytest.raises(RuntimeError, match="Pi SDK.*(exist|file)"):
        _snapshot(config)
    with pytest.raises(RuntimeError, match="Pi SDK.*(exist|file)"):
        asyncio.run(trainer.run_train(config, deps=_NoExternalSDK()))
    assert not config.resolved_log_dir().exists()


def _pinned_install(tmp_path, monkeypatch, *, installed):
    runner = tmp_path / "vat_runner"
    runner.mkdir()
    lockfile = runner / "package-lock.json"
    lockfile.write_bytes((smoke.VAT_RUNNER / "package-lock.json").read_bytes())
    package = runner / "node_modules" / "@earendil-works" / "pi-coding-agent"
    sdk = _entry(package) if installed else package / "dist" / "index.js"
    # Only relocate real filesystem inputs; never replace an identity result.
    monkeypatch.setattr(smoke, "VAT_RUNNER", runner)
    monkeypatch.setattr(smoke, "PINNED_PI_SDK", sdk)
    return sdk, lockfile


def test_snapshot_of_missing_default_sdk_records_expected_path_without_installing(tmp_path, monkeypatch):
    sdk, lockfile = _pinned_install(tmp_path, monkeypatch, installed=False)
    snap = _snapshot(_config(tmp_path))
    assert snap["pi_sdk"]["module_path"] == str(sdk.resolve())
    assert snap["pi_sdk"]["module_sha256"] is None
    assert snap["pi_sdk"]["lockfile"] == {
        "path": str(lockfile.resolve()), "sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest(),
    }
    assert not (smoke.VAT_RUNNER / "node_modules").exists()
    # Snapshotting is pure/offline, but the shipped episode preflight still
    # refuses the exact snapshotted path instead of installing or falling back.
    with pytest.raises(RuntimeError, match="pi-sdk-module-path does not exist"):
        smoke._resolve_pi_sdk(argparse.Namespace(pi_sdk_module_path=Path(snap["pi_sdk"]["module_path"])))
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="pinned Pi SDK is not installed"):
        asyncio.run(trainer.run_train(config, deps=_NoExternalSDK()))
    assert not config.resolved_log_dir().exists()


@pytest.mark.parametrize("changed", ["entry", "lockfile", "installation"])
def test_default_sdk_snapshots_actual_entry_and_lockfile_provenance(tmp_path, monkeypatch, changed):
    sdk, lockfile = _pinned_install(tmp_path, monkeypatch, installed=changed != "installation")
    config = _config(tmp_path)
    state, saved = _save(config)
    if changed == "installation":
        _entry(sdk.parent.parent)
    elif changed == "entry":
        assert saved["pi_sdk"]["module_sha256"] == hashlib.sha256(sdk.read_bytes()).hexdigest()
        sdk.write_bytes(sdk.read_bytes() + b"// changed selected SDK bytes\n")
    else:
        lockfile.write_bytes(lockfile.read_bytes() + b"\n")
    current = _assert_resume_rejected(config, state, saved)
    assert current["pi_sdk"]["module_path"] == str(sdk.resolve())
    assert current["pi_sdk"]["module_sha256"] == hashlib.sha256(sdk.read_bytes()).hexdigest()
    assert current["pi_sdk"]["lockfile"] == {
        "path": str(lockfile.resolve()), "sha256": hashlib.sha256(lockfile.read_bytes()).hexdigest(),
    }


@pytest.mark.parametrize("changed", ["helper", "package-metadata", "added-runtime-file", "deleted-runtime-file"])
def test_sdk_package_runtime_changes_cannot_hide_behind_unchanged_entry(tmp_path, changed):
    sdk = _entry(tmp_path / "sdk", b"export { sdk } from './core/helper.js';\n")
    helper = sdk.parent / "core" / "helper.js"
    helper.parent.mkdir()
    helper.write_text("export const sdk = 1;\n")
    config = _config(tmp_path, sdk)
    state, saved = _save(config)
    if changed == "helper":
        helper.write_text("export const sdk = 2;\n")
    elif changed == "package-metadata":
        (sdk.parent.parent / "package.json").write_text('{"name":"test-pi-sdk","type":"commonjs"}\n')
    elif changed == "added-runtime-file":
        (helper.parent / "another-helper.cjs").write_text("module.exports = {};\n")
    else:
        helper.unlink()
    current = _assert_resume_rejected(config, state, saved)
    assert current["pi_sdk"]["module_sha256"] == saved["pi_sdk"]["module_sha256"]
    assert current["pi_sdk"]["runtime"] != saved["pi_sdk"]["runtime"]


def test_sdk_runtime_fingerprint_does_not_walk_nested_node_modules(tmp_path):
    sdk = _entry(tmp_path / "sdk")
    dependency = _entry(sdk.parent / "node_modules" / "nested-dependency")
    source_map = sdk.with_suffix(".js.map")
    source_map.write_text('{}\n')
    config = _config(tmp_path, sdk)
    before = _snapshot(config)
    dependency.write_text("throw new Error('outside the bounded runtime fingerprint');\n")
    source_map.write_text('{"sourceRoot":"different-build-directory"}\n')
    assert _snapshot(config) == before


@pytest.mark.parametrize("changed", ["cwd", "symlink"])
def test_actual_runner_uses_snapshotted_canonical_sdk_after_resolution_changes(
    tmp_path, monkeypatch, training_sdk, changed,
):
    config, pi_audit = _runtime(tmp_path, monkeypatch)
    sdk = Path(config.player.pi_sdk_module_path).resolve()
    other = tmp_path / "other-cwd"
    other.mkdir()
    wrong_sdk = _entry(other / "external-pi-sdk", b"throw new Error('SDK selected after snapshot');\n")
    monkeypatch.chdir(tmp_path)
    selected = tmp_path / "selected-sdk.js"
    selected.symlink_to(sdk)
    configured_path = str(sdk.relative_to(tmp_path)) if changed == "cwd" else str(selected)
    config = replace(config, player=replace(config.player, pi_sdk_module_path=configured_path))
    calls = []
    resolution_lock = threading.Lock()

    def change_resolution_after_snapshot():
        # The external fixture invokes this hook from concurrent sampling
        # threads too. Change resolution exactly once at its first boundary,
        # rather than racing unlink/symlink operations in unrelated callbacks.
        with resolution_lock:
            if calls:
                return
            calls.append("external-service-boundary")
            if changed == "cwd":
                monkeypatch.chdir(other)
            else:
                selected.unlink()
                selected.symlink_to(wrong_sdk)

    # The fixture's external service boundary is reached after real snapshot().
    # No Shellsmith snapshot, identity, runner or collection result is mocked.
    training_sdk.observe = change_resolution_after_snapshot
    state = asyncio.run(trainer.run_train(config, deps=load_tinker_deps()))
    assert calls == ["external-service-boundary"]
    assert state["status"] == "complete" and state["step"] == 1
    saved = json.loads((config.resolved_log_dir() / trainer.SNAPSHOT_FILE).read_text())
    assert saved["pi_sdk"]["module_path"] == str(sdk)
    assert pi_audit.is_file() and pi_audit.read_text().strip()
    initialized_paths = []
    for log in config.resolved_log_dir().rglob("vat.jsonl"):
        for line in log.read_text().splitlines():
            message = json.loads(line).get("message", {})
            if message.get("type") == "initialize":
                initialized_paths.append(message["pi_sdk_module_path"])
    assert initialized_paths and set(initialized_paths) == {str(sdk)}
