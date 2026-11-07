"""The fixed profile admits only the saved Studio model's complete configuration.

Synthetic model metadata substitutes only the approved fingerprint. Actual
metadata parsing and file hashing stay real; no MLX or model weights are loaded.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from shellsmith.mlx import backend, readiness
from shellsmith.train.config import MLXSection
from test_mlx_contracts import _artifact
from test_mlx_memory_admission import STUDIO_MEMORY_BYTES, _host, qualified_model


ARCHITECTURE_MUTATIONS = {
    "vocabulary": {"vocab_size": 1},
    "mlp": {"intermediate_size": 1},
    "attention-heads": {"num_attention_heads": 1},
    "kv-heads": {"num_key_value_heads": 1},
    "attention-dimension": {"head_dim": 1},
    "gdn-value-heads": {"linear_num_value_heads": 1},
    "gdn-key-heads": {"linear_num_key_heads": 1},
    "gdn-key-dimension": {"linear_key_head_dim": 1},
    "gdn-value-dimension": {"linear_value_head_dim": 1},
    "gdn-convolution": {"linear_conv_kernel_dim": 1},
    "rotary": {"rope_parameters": {"rope_theta": 1.0}},
    "normalization": {"rms_norm_eps": 0.1},
    "tied-embeddings": {"tie_word_embeddings": True},
    "attention-bias": {"attention_bias": True},
    "activation": {"hidden_act": "relu"},
    "position-capacity": {"max_position_embeddings": 524288},
    "layer-schedule": {"full_attention_interval": 8,
                       "layer_types": ["full_attention" if (i + 1) % 8 == 0 else "linear_attention"
                                       for i in range(64)]},
    "additional-dimension": {"additional_decoder_dimension": 1},
}


def _mutate_config(folder, mutation):
    path = folder / "config.json"
    config = json.loads(path.read_text())
    config["text_config"].update(ARCHITECTURE_MUTATIONS[mutation])
    path.write_text(json.dumps(config))


def test_approved_fingerprint_matches_shipped_actual_studio_model_evidence():
    evidence_path = Path(__file__).resolve().parents[1] / "docs/mlx-qualification/studio-r3.json"
    evidence = json.loads(evidence_path.read_text())
    recorded = evidence["policy_identity"]["model"]["files"]["config.json"]
    assert recorded == "8f80874ac3ad8fa386d3f6dc0ea85377f703376e009a03dee0360e08e289a25d"
    assert readiness.APPROVED_CONFIG_SHA256 == recorded


def test_readiness_admits_exact_config_and_reports_actual_fingerprint(monkeypatch, qualified_model):
    _host(monkeypatch, STUDIO_MEMORY_BYTES)
    report = readiness.require_ready(MLXSection(model_path=str(qualified_model)))
    assert report["ready"]
    assert report["model"]["config_sha256"] == hashlib.sha256((qualified_model / "config.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("mutation", ARCHITECTURE_MUTATIONS)
def test_matching_width_and_layer_count_do_not_admit_other_architectures(monkeypatch, qualified_model, mutation):
    _host(monkeypatch, STUDIO_MEMORY_BYTES)
    _mutate_config(qualified_model, mutation)
    # Each mutation remains admissible under the generic dense metadata
    # contract, demonstrating that the fixed-profile fingerprint is the gate.
    _, text, kinds = readiness.model_metadata(qualified_model)
    assert text["hidden_size"] == 5120 and text["num_hidden_layers"] == len(kinds) == 64
    report = readiness.readiness(MLXSection(model_path=str(qualified_model)))
    assert not report["ready"]
    assert any("approved config.json SHA256" in blocker for blocker in report["blockers"])


@pytest.mark.parametrize("mutation", ["vocabulary", "mlp", "gdn-value-dimension"])
def test_unqualified_architecture_fails_before_session_identity_or_model(
    monkeypatch, tmp_path, qualified_model, mutation,
):
    _host(monkeypatch, STUDIO_MEMORY_BYTES)
    _mutate_config(qualified_model, mutation)
    effects = []

    def forbidden(*args, **kwargs):
        effects.append("identity-or-model")
        pytest.fail("Unqualified architecture reached identity hashing or model initialization")

    monkeypatch.setattr(backend, "resolve_identity", forbidden)
    monkeypatch.setattr(backend.MLXSession, "_initialize", forbidden)
    output = tmp_path / "uncreated-checkpoints"
    with pytest.raises(RuntimeError, match="approved config.json SHA256"):
        backend.MLXSession(SimpleNamespace(mlx=MLXSection(model_path=str(qualified_model))), checkpoint_root=output)
    assert not effects and not output.exists()


def test_generic_tiny_dense_metadata_remains_available(tmp_path):
    _, _, kinds = readiness.model_metadata(_artifact(tmp_path))
    assert len(kinds) == 4
