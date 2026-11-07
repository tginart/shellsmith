"""
Purpose: Own one mixed-precision Flash base for resident LoRA and inference.
Input: The approved Flash artifact and the allocator budget.
Output: Architecture-specific factors, replay operations, and memory estimates.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

from .model import Qwen27Model, numerical_sources, tokenizer_bundle
from .protocol import digest
from .storage import file_hash


APPROVED_FLASH_CONFIG_SHA256 = (
    "3afe6d373350634985a06e383937d23308b0871f838e2e05ff844f38eee8d417"
)
# One VJP over a checkpoint group keeps the backward state of every layer in it.
# Measured on the approved config: 12.0 GB for ten layers of a 2048-token segment,
# 2.25 GB for one layer, so about 532 KiB per token for each added layer.
GROUP_VJP_BYTES_PER_TOKEN_LAYER = 576 << 10


def admitted_flash_config(path: Path) -> tuple:
    """Require the exact mixed-precision Flash profile before any GPU import."""
    raw = (path / "config.json").read_bytes()
    config = json.loads(raw)
    text = config.get("text_config", {})
    expected = {
        "hidden_size": 2560,
        "num_hidden_layers": 48,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "max_position_embeddings": 262144,
        "hc_count": 4,
    }
    if config.get("model_type") != "qwen4_exp" or any(
        text.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Flash requires the approved Qwen3.8-Flash-Next decoder.")
    if hashlib.sha256(raw).hexdigest() != APPROVED_FLASH_CONFIG_SHA256:
        raise ValueError(
            "Flash requires the approved 4-bit-expert/8-bit-dense/8-bit-PLE config."
        )
    return config, text


class QwenFlashModel(Qwen27Model):
    """
    Purpose: Share resident ownership and publication mechanics with the dense model.
    Input: The approved Flash artifact.
    Output: Flash-specific model math with the same client and factor lifecycle.
    Note: This class replaces dense admission, adapters, estimates, and decoder math.
    """

    supports_saved_activations = False
    # Zero keeps sorted per-row expert products for prompt prefill.
    prefill_moe_chunk = 0

    def __init__(
        self,
        path: Path,
        *,
        headroom_gib: float = 32,
        allocator_gib: float | None = None,
    ):
        path = Path(path).expanduser().resolve()
        _, text = admitted_flash_config(path)
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_vlm.utils import load_model
        from .numerics.flash_adapter import FlashAdapter
        from .numerics.flash_ops import FlashOps
        from .numerics.flash_ple import join_ngram_tables

        physical = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]))
        budget = (
            physical - int(headroom_gib * 1024**3)
            if allocator_gib is None
            else int(allocator_gib * 1024**3)
        )
        if headroom_gib < 0 or not 0 < budget <= physical - int(headroom_gib * 1024**3):
            raise ValueError(
                "Flash allocator must preserve the configured host headroom."
            )
        self.mx, self.budget = mx, budget
        mx.set_memory_limit(budget)
        mx.set_cache_limit(2 << 30)
        self.device = mx.device_info()
        self.model = load_model(path, lazy=False, strict=True)
        self.model.freeze()
        self.lm = self.model.language_model
        # One device table gives bitwise-equal rows without a host index sync.
        assert join_ngram_tables(self.lm) > 0, "Flash PLE layers have no n-gram table."
        mx.clear_cache()
        self.adapter = FlashAdapter(self.lm, text["layer_types"])
        self.replay_ops = FlashOps()
        self.selected, self.training = None, None
        mx.eval(self.model.parameters())
        self.frozen = [
            (name, value)
            for name, value in tree_flatten(self.model.parameters())
            if not name.endswith((".a", ".b"))
        ]
        self.base_allocation = tuple(
            (name, id(value), value.nbytes) for name, value in self.frozen
        )
        self.bundle = tokenizer_bundle(path)
        assets = {
            file.name: file_hash(file) for file in sorted(path.glob("*.safetensors"))
        }
        assets["config.json"] = file_hash(path / "config.json")
        code = numerical_sources()
        code["flash_model.py"] = file_hash(Path(__file__))
        self.identity = {
            "model": digest(assets),
            "model_files": assets,
            "tokenizer": self.bundle["hash"],
            "numerics": digest(code),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("mlx", "mlx-lm", "mlx-vlm", "transformers")
            },
        }
        self.identity["kernel_build"] = {
            "numerical_sources": self.identity["numerics"],
            "runtime_versions": self.identity["versions"],
            "architecture": self.device.get("architecture"),
            "os_build": subprocess.check_output(
                ["sw_vers", "-buildVersion"], text=True
            ).strip(),
            "mlx_library_sha256": file_hash(
                Path(mx.__file__).parent / "lib/libmlx.dylib"
            ),
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "MLX default"),
        }
        self.identity["kernel_build_id"] = digest(self.identity["kernel_build"])
        self.capabilities = {
            "model": "Qwen/Qwen3.8-Flash-Next",
            "context_window": 262144,
            "vocabulary_size": text["vocab_size"],
            "losses": ["cispo", "weighted_cross_entropy"],
            "precision": {
                "activations": "bf16",
                "lora": "bf16",
                "base": "8-bit dense/shared/PLE; 4-bit routed experts",
                "optimizer": "bf16",
                "recurrent_state": "bf16",
                "training_kv": "bf16",
            },
            "cache_precisions": ["bf16", "8bit"],
            "max_inference_batch": 8,
            "physical_memory_bytes": physical,
            "allocator_budget_bytes": budget,
            "host_headroom_bytes": int(headroom_gib * 1024**3),
            "recommended_working_set_bytes": self.device.get(
                "max_recommended_working_set_size"
            ),
            "tokenizer_hash": self.bundle["hash"],
            "frozen_tensors": len(self.frozen),
            "resident_base_bytes": sum(value.nbytes for _, value in self.frozen),
            "lora_targets": "all decoder mixer/shared-expert/routed-expert projections",
            "qualification": "experimental_flash_resident_lora",
        }

    def estimate_replay(
        self,
        sequences: list,
        checkpoint_layers: int = 1,
        *,
        segment_tokens: int = 2048,
        head_rows: int = 128,
    ) -> dict:
        length = max(
            len(row["prompt_token_ids"]) + len(row["sampled_tokens"]) - 1
            for row in sequences
        )
        segments = math.ceil(length / segment_tokens)
        width = self.lm.args.hidden_size * self.lm.args.hc_count
        boundaries = (
            length * width * math.ceil(len(self.lm.layers) / checkpoint_layers) * 2
        )
        recurrent = sum(
            layer.linear_attn.num_v_heads
            * layer.linear_attn.head_v_dim
            * layer.linear_attn.head_k_dim
            * 2
            for layer in self.lm.layers
            if layer.is_linear
        )
        attention = sum(
            length
            * (
                layer.self_attn.num_key_value_heads * layer.self_attn.head_dim * 4
                + self.lm.args.indexer_head_dim * 2
            )
            for layer in self.lm.layers
            if not layer.is_linear
        )
        carries = segments * recurrent + attention
        # A recorded forward keeps the n-gram rows of each PLE layer for its VJP.
        ple_rows = length * self.lm.args.ple_embed_dim * 2 * sum(
            "ple" in layer for layer in self.lm.layers
        )
        adjoints = recurrent + attention
        branches = max(0, len(sequences) - 1) * length * (width * 2 + 12)
        rows = min(length, segment_tokens)
        scratch = (
            rows * width * checkpoint_layers * 2
            + rows * (checkpoint_layers - 1) * GROUP_VJP_BYTES_PER_TOKEN_LAYER
            + (4 << 30)
        )
        return {
            "additional_bytes": boundaries
            + carries
            + ple_rows
            + adjoints * 3
            + branches
            + scratch,
            "boundary_bytes": boundaries,
            "carry_bytes": carries,
            "carry_adjoint_bytes": adjoints * 3,
            "ple_row_bytes": ple_rows,
        }

    def estimate_inference(
        self,
        length: int,
        batch: int = 1,
        *,
        cache_precision: str = "bf16",
    ) -> int:
        """Bound sampling state with the selected KV format and BF16 indexer state."""
        if cache_precision not in ("bf16", "8bit"):
            raise ValueError("Unsupported Flash inference cache precision.")
        kv_bytes_per_element = 2.0 if cache_precision == "bf16" else 1.0 + 4 / 64
        per_row = length * sum(
            math.ceil(
                layer.self_attn.num_key_value_heads
                * layer.self_attn.head_dim
                * 2
                * kv_bytes_per_element
            )
            + self.lm.args.indexer_head_dim * 3
            + 4
            for layer in self.lm.layers
            if not layer.is_linear
        )
        per_row += sum(
            layer.linear_attn.num_v_heads
            * layer.linear_attn.head_v_dim
            * layer.linear_attn.head_k_dim
            * 2
            for layer in self.lm.layers
            if layer.is_linear
        )
        return int(batch * per_row + (2 << 30))

    def decode_hidden(self, ids_BL: object, positions: object, caches: list) -> object:
        """Use the same Flash layer math and resident factors for real inference."""
        from .numerics.flash_ops import decode_hidden

        return decode_hidden(self.lm, ids_BL, positions, caches,
                             moe_chunk=self.prefill_moe_chunk or None)

    def decode_rows(self, engine: object, requests: list) -> None:
        from .numerics.flash_ops import decode_rows

        decode_rows(engine, requests)
