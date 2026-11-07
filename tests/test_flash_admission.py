"""
Purpose: Reject unqualified Flash artifacts before any MLX import or GPU allocation.
Input: The installed mixed-precision model configuration and incompatible variants.
Output: Exact fingerprint admission or an explicit profile error.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from studio_compute.flash_model import (
    APPROVED_FLASH_CONFIG_SHA256,
    QwenFlashModel,
    admitted_flash_config,
)

FIXTURE = Path(__file__).parent / "fixtures/studio_compute/qwen-flash-config.json"


def test_flash_profile_matches_installed_artifact(tmp_path: Path) -> None:
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == APPROVED_FLASH_CONFIG_SHA256
    (tmp_path / "config.json").write_bytes(raw)
    _, text = admitted_flash_config(tmp_path)
    assert text["num_hidden_layers"] == 48
    assert text["num_experts"] == 512
    assert text["hc_count"] == 4


@pytest.mark.parametrize(
    "key,value",
    [
        ("hidden_size", 5120),
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("hc_count", 1),
        ("max_position_embeddings", 65536),
    ],
)
def test_flash_rejects_incompatible_architecture_before_mlx_import(
    tmp_path: Path,
    key: str,
    value: int,
) -> None:
    config = json.loads(FIXTURE.read_bytes())
    config["text_config"][key] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="approved Qwen3.8-Flash"):
        QwenFlashModel(tmp_path)


def test_flash_rejects_unqualified_quantization_before_mlx_import(
    tmp_path: Path,
) -> None:
    config = json.loads(FIXTURE.read_bytes())
    config["quantization"]["bits"] = 2
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="approved 4-bit-expert"):
        QwenFlashModel(tmp_path)


def test_flash_service_profile_is_explicit_and_requires_bf16() -> None:
    from studio_compute.cli import parse_args

    assert parse_args(["serve", "--model", "tiny"]).model_profile == "qwen27"
    args = parse_args(
        ["serve", "--model", "tiny", "--model-profile", "flash-experimental"]
    )
    assert args.model_profile == "flash-experimental"
    with pytest.raises(SystemExit):
        parse_args(
            [
                "serve",
                "--model",
                "tiny",
                "--model-profile",
                "flash-experimental",
                "--execution-profile",
                "legacy",
            ]
        )


def test_flash_inference_budget_accounts_for_quantized_metadata() -> None:
    model = QwenFlashModel.__new__(QwenFlashModel)
    attention = SimpleNamespace(num_key_value_heads=2, head_dim=256)
    recurrent = SimpleNamespace(num_v_heads=48, head_v_dim=128, head_k_dim=128)
    model.lm = SimpleNamespace(
        args=SimpleNamespace(indexer_head_dim=128),
        layers=[SimpleNamespace(is_linear=False, self_attn=attention)] * 12
        + [SimpleNamespace(is_linear=True, linear_attn=recurrent)] * 36,
    )
    length, batch = 1000, 8
    fixed_bytes = batch * 36 * 48 * 128 * 128 * 2 + (2 << 30)
    assert (
        model.estimate_inference(length, batch) == fixed_bytes + batch * length * 29232
    )
    assert (
        model.estimate_inference(length, batch, cache_precision="8bit")
        == fixed_bytes + batch * length * 17712
    )
    with pytest.raises(ValueError, match="cache precision"):
        model.estimate_inference(length, cache_precision="4bit")
