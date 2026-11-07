"""Qualified host memory is checked before resident models or live rollouts."""

import hashlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shellsmith import cli
from shellsmith.campaign import inference
import shellsmith.jobs.readiness
import shellsmith.mlx
from shellsmith.mlx import backend, readiness
from shellsmith.train import harness_identity, trainer
from shellsmith.train.config import MLXSection
from test_mlx_contracts import _artifact


STUDIO_MEMORY_BYTES = 256 * 1024**3


@pytest.fixture
def qualified_model(tmp_path, monkeypatch):
    folder = tmp_path / "model"
    folder.mkdir()
    _artifact(folder)
    config = json.loads((folder / "config.json").read_text())
    config["quantization"]["mode"] = "affine"
    config["text_config"].update(hidden_size=5120, num_hidden_layers=64,
                               layer_types=["full_attention" if (index + 1) % 4 == 0 else "linear_attention"
                                            for index in range(64)])
    (folder / "config.json").write_text(json.dumps(config))
    # This tiny artifact substitutes only the approved asset fingerprint;
    # the production metadata parser and byte hashing still run unchanged.
    monkeypatch.setattr(readiness, "APPROVED_CONFIG_SHA256",
                        hashlib.sha256((folder / "config.json").read_bytes()).hexdigest(), raising=False)
    return folder


def _host(monkeypatch, memory):
    # Only the external machine/dependency probe is replaced. Model/profile
    # admission and the memory comparison run through their real public APIs.
    monkeypatch.setattr(readiness, "capabilities", lambda **kwargs: {
        "blockers": [], "memory_bytes": memory, "os": "Darwin", "architecture": "arm64", "metal": True,
    })


@pytest.mark.parametrize("memory", [None, True, 0, -1, 64 * 1024**3, 128 * 1024**3,
                                   256 * 1000**3, STUDIO_MEMORY_BYTES - 1],
                         ids=["unknown", "bool", "zero", "negative", "64-gib", "128-gib", "decimal-256-gb", "one-byte-short"])
def test_readiness_rejects_unqualified_capacity_even_when_model_shards_fit(monkeypatch, qualified_model, memory):
    _host(monkeypatch, memory)
    report = readiness.readiness(MLXSection(model_path=str(qualified_model)))
    assert report["model"]["weight_bytes"] == len(b"test frozen weights")
    assert not report["ready"]
    assert any("256 GiB" in blocker and "128 GiB" in blocker for blocker in report["blockers"])


@pytest.mark.parametrize("memory", [STUDIO_MEMORY_BYTES, 2 * STUDIO_MEMORY_BYTES], ids=["qualified-studio", "larger"])
def test_readiness_admits_qualified_physical_memory(monkeypatch, qualified_model, memory):
    _host(monkeypatch, memory)
    report = readiness.require_ready(MLXSection(model_path=str(qualified_model)))
    assert report["ready"] and report["blockers"] == []
    assert report["capabilities"]["memory_bytes"] == memory


@pytest.mark.parametrize("failure", ["sysctl-error", "malformed-output"])
def test_failed_physical_memory_probe_blocks_readiness_without_loading_mlx(monkeypatch, qualified_model, failure):
    monkeypatch.setattr(readiness.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(readiness.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(readiness.platform, "mac_ver", lambda: ("27.0", (), ""))
    monkeypatch.setattr(readiness.importlib.metadata, "version", readiness.PINS.__getitem__)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")

    def sysctl(command, **kwargs):
        assert command == ["/usr/sbin/sysctl", "-n", "hw.memsize"]
        if failure == "sysctl-error":
            raise OSError("fixture physical memory probe failed")
        return SimpleNamespace(stdout="unavailable")

    monkeypatch.setattr(readiness.subprocess, "run", sysctl)
    before = set(sys.modules)
    report = readiness.readiness(MLXSection(model_path=str(qualified_model)), probe_metal=False)
    assert not report["ready"] and report["capabilities"]["memory_bytes"] is None
    assert any("sysctl hw.memsize" in blocker for blocker in report["blockers"])
    assert any("256 GiB" in blocker for blocker in report["blockers"])
    assert "mlx.core" not in set(sys.modules) - before


@pytest.mark.parametrize("memory", [None, 64 * 1024**3], ids=["unknown", "smaller-host"])
def test_session_rejects_memory_before_identity_or_model_initialization(monkeypatch, tmp_path, qualified_model, memory):
    _host(monkeypatch, memory)
    effects = []

    def forbidden(*args, **kwargs):
        effects.append("identity-or-model")
        pytest.fail("Unqualified host reached identity hashing or model initialization")

    monkeypatch.setattr(backend, "resolve_identity", forbidden)
    monkeypatch.setattr(backend.MLXSession, "_initialize", forbidden)
    output = tmp_path / "uncreated-checkpoints"
    with pytest.raises(RuntimeError, match="256 GiB"):
        backend.MLXSession(SimpleNamespace(mlx=MLXSection(model_path=str(qualified_model))), checkpoint_root=output)
    assert not effects and not output.exists()


@pytest.mark.parametrize("command", ["train", "campaign", "episode"])
@pytest.mark.parametrize("memory", [None, 64 * 1024**3], ids=["unknown", "smaller-host"])
def test_public_cli_rejects_memory_before_session_rollouts_or_output(
    monkeypatch, tmp_path, qualified_model, command, memory,
):
    _host(monkeypatch, memory)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    monkeypatch.setattr(shellsmith.jobs.readiness, "probe", lambda: {"ready": True, "blockers": []})
    monkeypatch.setattr(harness_identity, "pi_sdk_identity", lambda path: {"module_sha256": "fixture-sdk"})
    from shellsmith.train import preflight

    monkeypatch.setattr(preflight.shutil, "which", lambda executable: "/fixture/codex")

    def authenticated(command, **kwargs):
        assert command == ["/fixture/codex", "login", "status"]
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(preflight, "subprocess", SimpleNamespace(run=authenticated, TimeoutExpired=subprocess.TimeoutExpired))
    effects = []

    def forbidden(*args, **kwargs):
        effects.append("session-or-rollouts")
        pytest.fail("Unqualified host reached session or rollout work")

    monkeypatch.setattr(shellsmith.mlx, "create_session", forbidden)
    monkeypatch.setattr(trainer, "run_train", forbidden)
    monkeypatch.setattr(inference, "run_inference_campaign", forbidden)
    config = tmp_path / "run.toml"
    config.write_text(f'[compute]\nbackend="mlx"\ngpu_lease_root="{tmp_path / "lease"}"\n'
                      f'[mlx]\nmodel_path="{qualified_model}"\n')
    output = tmp_path / "uncreated-output"
    argv = [command, "--config", str(config), "--log-dir", str(output)]
    if command == "episode":
        argv += ["--player", "mlx"]
    with pytest.raises(SystemExit, match="256 GiB"):
        cli.main(argv)
    assert not effects and not output.exists() and not (tmp_path / "lease").exists()
