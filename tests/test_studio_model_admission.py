"""CPU-only admission against the approved artifact's actual config bytes."""

import hashlib
import json
from pathlib import Path

import pytest

from studio_compute.model import APPROVED_CONFIG_SHA256, Qwen27Model, admitted_config

FIXTURE = Path(__file__).parent / "fixtures/studio_compute/qwen27-config.json"


def test_supplied_27b_profile_matches_qualified_fingerprint(tmp_path):
    data = FIXTURE.read_bytes()
    assert hashlib.sha256(data).hexdigest() == APPROVED_CONFIG_SHA256
    (tmp_path / "config.json").write_bytes(data)
    config, text = admitted_config(tmp_path)
    assert text["hidden_size"] == 5120 and len(text["layer_types"]) == 64
    assert config["quantization"] == {"bits": 8, "group_size": 64, "mode": "affine"}


@pytest.mark.parametrize("section,key,value,message", [
    ("text_config", "hidden_size", 2048, "width 5120"),
    ("text_config", "num_hidden_layers", 32, "64 decoder layers"),
    ("text_config", "num_experts", 128, "dense"),
    ("quantization", "bits", 4, "8-bit/group-64"),
    ("quantization", "group_size", 128, "8-bit/group-64"),
    ("quantization", "mode", "mxfp8", "affine"),
    ("text_config", "vocab_size", 1024, "approved config SHA256"),
    ("text_config", "max_position_embeddings", 32768, "approved config SHA256"),
])
def test_other_dense_profiles_rejected_before_mlx_import_or_loading(tmp_path, section, key, value, message):
    config = json.loads(FIXTURE.read_bytes())
    config[section][key] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        Qwen27Model(tmp_path)  # This host need not have MLX installed.


def test_even_compatible_shape_requires_the_exact_supplied_config(tmp_path):
    config = json.loads(FIXTURE.read_bytes())
    (tmp_path / "config.json").write_text(json.dumps(config, sort_keys=True))
    with pytest.raises(ValueError, match="approved config SHA256"):
        Qwen27Model(tmp_path)
