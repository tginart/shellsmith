"""Portable optional-backend admission and objective contracts."""

import importlib
import json
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from shellsmith.mlx.cispo import reference_loss_and_gradient
from shellsmith.mlx.identity import model_identity
from shellsmith.mlx.readiness import CANONICAL_TARGETS, model_metadata, readiness


def _artifact(tmp_path):
    config = {"model_type": "qwen3_5", "quantization": {"bits": 8, "group_size": 64}, "text_config": {
        "num_hidden_layers": 4, "full_attention_interval": 4, "max_position_embeddings": 262144,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"]}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "model-1.safetensors"}}))
    (tmp_path / "model-1.safetensors").write_bytes(b"test frozen weights")
    (tmp_path / "tokenizer_config.json").write_text("{}")
    (tmp_path / "chat_template.jinja").write_text("test native template")
    return tmp_path


def test_cispo_is_sum_of_detached_token_surrogates():
    new = [-2.0, -1.0, -4.0, -2.0]
    old = [-3.0, -4.0, -1.0, -2.0]
    advantages = [0.1, -0.25, 0.5, 1.0]
    loss, gradient = reference_loss_and_gradient(new, old, advantages, [1, 1, 1, 0])
    expected = [-math.e * 0.1, 4.0 * 0.25, -math.exp(-3) * 0.5, 0.0]
    assert gradient == pytest.approx(expected)
    assert loss == pytest.approx(sum(g * value for g, value in zip(expected, new)))
    doubled, _ = reference_loss_and_gradient(new * 2, old * 2, advantages * 2, [1, 1, 1, 0] * 2)
    assert doubled == pytest.approx(loss * 2)


def test_cispo_zero_advantage_and_floor():
    assert reference_loss_and_gradient([-2], [-2], [0]) == (0.0, (0.0,))
    _, gradient = reference_loss_and_gradient([-1000], [-1], [1], clip_low=0.2)
    assert gradient == (-0.2,)


@pytest.mark.parametrize("low,high", [(1, 0), (-1, 4), (0, float("nan")), (0, 0)])
def test_invalid_cispo_clipping(low, high):
    with pytest.raises(ValueError):
        reference_loss_and_gradient([-1], [-1], [1], clip_low=low, clip_high=high)


def test_asset_identity_hashes_actual_weights_and_template(tmp_path):
    folder = _artifact(tmp_path)
    before = model_identity(folder)
    assert before["quantization"] == {"bits": 8, "group_size": 64}
    (folder / "model-1.safetensors").write_bytes(b"different frozen weights")
    after = model_identity(folder)
    assert before["asset_sha256"] != after["asset_sha256"]
    assert before["files"]["chat_template.jinja"] == after["files"]["chat_template.jinja"]


def test_model_index_rejects_nonlocal_shards(tmp_path):
    folder = _artifact(tmp_path)
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "../escape.safetensors"}}))
    with pytest.raises(ValueError, match="nonlocal"):
        model_metadata(folder)


def test_readiness_rejects_moe_and_wrong_quantization(tmp_path):
    folder = _artifact(tmp_path)
    path = folder / "config.json"
    config = json.loads(path.read_text())
    config["text_config"]["num_experts"] = 8
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="dense"):
        model_metadata(folder)
    config["text_config"].pop("num_experts")
    config["quantization"]["bits"] = 4
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="8-bit"):
        model_metadata(folder)


def test_readiness_is_explicit_about_execution_host(tmp_path):
    settings = SimpleNamespace(model_path=str(_artifact(tmp_path)), lora_targets=CANONICAL_TARGETS)
    report = readiness(settings, probe_metal=False)
    assert report["model"]["layers"] == 4
    assert report["capabilities"]["python"] == sys.executable
    assert isinstance(report["blockers"], list)


def test_readiness_rejects_tiny_dense_model_under_qualified_27b_profile(tmp_path):
    settings = SimpleNamespace(model_path=str(_artifact(tmp_path)), lora_targets=CANONICAL_TARGETS)
    report = readiness(settings, probe_metal=False)
    assert any("hidden_size=5120" in reason for reason in report["blockers"])


def test_readiness_requires_affine_quantization_even_with_qualified_dimensions(tmp_path):
    folder = _artifact(tmp_path)
    config = json.loads((folder / "config.json").read_text())
    config["text_config"].update(hidden_size=5120, num_hidden_layers=64,
                               layer_types=["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(64)])
    config["quantization"]["mode"] = "mxfp8"
    (folder / "config.json").write_text(json.dumps(config))
    report = readiness(SimpleNamespace(model_path=str(folder), lora_targets=CANONICAL_TARGETS), probe_metal=False)
    assert any("affine" in reason for reason in report["blockers"])
    assert not any("hidden_size=5120" in reason for reason in report["blockers"])


def test_package_import_does_not_import_mlx(monkeypatch):
    # The contracts modules and package can load on Linux with no numerical deps.
    before = set(sys.modules)
    importlib.reload(importlib.import_module("shellsmith.mlx"))
    assert "mlx.core" not in set(sys.modules) - before


@pytest.mark.parametrize("dependency", ["mlx/render.py", "mlx/readiness.py", "policy/chat_adapter.py",
                                         "policy/sampler.py", "policy/reasoning.py", "train/backend.py"])
def test_renderer_source_change_invalidates_adapter_checkpoint(monkeypatch, tmp_path, dependency):
    from shellsmith.mlx import identity
    from shellsmith.mlx.checkpoint import publish_checkpoint, verify_checkpoint
    from shellsmith.train.config import MLXSection

    package = Path(identity.__file__).resolve().parent.parent
    copied = tmp_path / "source" / "shellsmith"
    shutil.copytree(package / "mlx", copied / "mlx", ignore=shutil.ignore_patterns("__pycache__"))
    (copied / "policy").mkdir()
    for name in ("chat_adapter.py", "sampler.py", "reasoning.py"):
        shutil.copyfile(package / "policy" / name, copied / "policy" / name)
    (copied / "train").mkdir()
    shutil.copyfile(package / "train" / "backend.py", copied / "train" / "backend.py")
    monkeypatch.setattr(identity, "__file__", str(copied / "mlx" / "identity.py"))
    monkeypatch.setattr(identity.importlib.metadata, "version", identity.PINS.__getitem__)
    assets = tmp_path / "model"
    assets.mkdir()
    model = model_identity(_artifact(assets))
    settings = MLXSection(model_path=str(assets))
    original = identity.policy_identity(settings, model)
    assert "policy/chat_adapter.py" in original["source_files"]
    assert "policy/sampler.py" in original["source_files"]
    assert "train/backend.py" in original["source_files"]

    def writer(folder):
        (folder / "adapters.safetensors").write_bytes(b"fixture adapter")
    checkpoint = publish_checkpoint(tmp_path / "sampler", writer=writer, kind="sampler",
                                    policy_identity=original, policy_version=1)
    verify_checkpoint(checkpoint, kind="sampler", expected_identity=original)
    changed = copied / dependency
    if dependency == "train/backend.py":
        changed.write_text(changed.read_text().replace("return list(self.tokens)", "return list(reversed(self.tokens))"))
    else:
        changed.write_text(changed.read_text() + "\n# changed renderer contract\n")
    current = identity.policy_identity(settings, model)
    assert current["source_sha256"] != original["source_sha256"]
    assert current["identity_sha256"] != original["identity_sha256"]
    assert current["model"] == original["model"]
    with pytest.raises(ValueError, match="frozen.*source identity"):
        verify_checkpoint(checkpoint, kind="sampler", expected_identity=current)


@pytest.mark.parametrize("setting,value", [("compaction_reserve_tokens", 13312),
                                          ("compaction_keep_recent_tokens", 4096)])
def test_compaction_policy_change_invalidates_adapter_checkpoint(monkeypatch, tmp_path, setting, value):
    from dataclasses import replace
    from shellsmith.mlx import identity
    from shellsmith.mlx.checkpoint import publish_checkpoint, verify_checkpoint
    from shellsmith.train.config import MLXSection

    monkeypatch.setattr(identity.importlib.metadata, "version", identity.PINS.__getitem__)
    model = model_identity(_artifact(tmp_path))
    settings = MLXSection(model_path=str(tmp_path))
    original = identity.policy_identity(settings, model)
    def writer(folder):
        (folder / "adapters.safetensors").write_bytes(b"fixture adapter")
    checkpoint = publish_checkpoint(tmp_path / "sampler", writer=writer, kind="sampler",
                                    policy_identity=original, policy_version=1)
    verify_checkpoint(checkpoint, kind="sampler", expected_identity=original)
    current = identity.policy_identity(replace(settings, **{setting: value}), model)
    assert current["source_sha256"] == original["source_sha256"]
    assert current["identity_sha256"] != original["identity_sha256"]
    assert current["sampling"][setting] == value
    with pytest.raises(ValueError, match="frozen.*source identity"):
        verify_checkpoint(checkpoint, kind="sampler", expected_identity=current)
