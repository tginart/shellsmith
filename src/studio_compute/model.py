"""Model-specific ownership and admission; scheduler has no layer/layout assumptions."""

from __future__ import annotations
import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from .protocol import digest
from .storage import file_hash

# The supplied artifact's fingerprint, also recorded by historical MLX readiness
# and docs/mlx-qualification/studio-r3.json. Keep this package standalone.
APPROVED_CONFIG_SHA256 = (
    "8f80874ac3ad8fa386d3f6dc0ea85377f703376e009a03dee0360e08e289a25d"
)


def admitted_config(path):
    """Fail before numerical imports/allocation for any other model profile."""
    data = (path / "config.json").read_bytes()
    config = json.loads(data)
    text = config.get("text_config", config) if isinstance(config, dict) else None
    if (
        not isinstance(text, dict)
        or config.get("model_type") != "qwen3_5"
        or text.get("model_type") not in ("qwen3_5", "qwen3_5_text")
        or text.get("num_experts", 0)
    ):
        raise ValueError("Studio v1 requires the approved dense Qwen3.8-27B decoder.")
    if text.get("hidden_size") != 5120 or text.get("num_hidden_layers") != 64:
        raise ValueError(
            "Studio Qwen3.8-27B requires width 5120 and 64 decoder layers."
        )
    quantization = config.get("quantization", config.get("quantization_config"))
    if not isinstance(quantization, dict) or any(
        quantization.get(key) != value
        for key, value in {"bits": 8, "group_size": 64, "mode": "affine"}.items()
    ):
        raise ValueError("Studio Qwen3.8-27B requires affine 8-bit/group-64 weights.")
    actual = hashlib.sha256(data).hexdigest()
    if actual != APPROVED_CONFIG_SHA256:
        raise ValueError(
            f"Studio Qwen3.8-27B requires approved config SHA256 {APPROVED_CONFIG_SHA256}; detected {actual}."
        )
    return config, text


TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)


def tokenizer_bundle(path):
    files = {}
    for name in TOKENIZER_FILES:
        asset = path / name
        if asset.is_file() and not asset.is_symlink():
            data = asset.read_bytes()
            files[name] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "base64": base64.b64encode(data).decode(),
            }
    if not files.get("tokenizer.json") or not files.get("tokenizer_config.json"):
        raise ValueError("Model must supply a portable tokenizer bundle.")
    return {
        "hash": digest({name: item["sha256"] for name, item in files.items()}),
        "files": files,
    }


def numerical_sources(root: Path | None = None) -> dict[str, str]:
    """
    Purpose: Bind the numerical graph and activation admission to one source inventory.
    Input: The package directory. The default is the installed package directory.
    Output: Source paths and SHA256 digests for the compute identity.
    """
    root = Path(root) if root is not None else Path(__file__).parent
    sources = {
        str(file.relative_to(root)): file_hash(file)
        for file in sorted((root / "numerics").rglob("*"))
        if file.is_file() and file.suffix in (".py", ".metal")
    }
    sources.update(
        {
            name: file_hash(root / name)
            for name in ("model.py", "engine.py", "activation_budget.py")
        }
    )
    return sources


class Qwen27Model:
    """V1 dense profile. Flash gets its own implementation of this boundary."""

    def __init__(self, path, *, headroom_gib=16, allocator_gib=None):
        path = Path(path).expanduser().resolve()
        config, text = admitted_config(path)
        import mlx.core as mx
        from mlx_vlm.utils import load_model
        from .numerics.adapter import Adapter
        from mlx.utils import tree_flatten

        physical = int(
            subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.memsize"])
        )
        budget = (
            physical - int(headroom_gib * 1024**3)
            if allocator_gib is None
            else int(allocator_gib * 1024**3)
        )
        if headroom_gib < 0 or not 0 < budget <= physical - int(headroom_gib * 1024**3):
            raise ValueError(
                "Allocator must fit physical memory minus configured host/GM headroom."
            )
        self.mx, self.budget = mx, budget
        mx.set_memory_limit(budget)
        mx.set_cache_limit(2 << 30)
        self.device = mx.device_info()
        self.model = load_model(path, lazy=False, strict=True)
        self.model.freeze()
        self.lm = self.model.language_model
        self.adapter = Adapter(self.lm, text["layer_types"])
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
            name.name: file_hash(name) for name in sorted(path.glob("*.safetensors"))
        }
        assets["config.json"] = file_hash(path / "config.json")
        code = numerical_sources()
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
                ["/usr/bin/sw_vers", "-buildVersion"], text=True
            ).strip(),
            "mlx_library_sha256": file_hash(
                Path(mx.__file__).parent / "lib/libmlx.dylib"
            ),
            "mlx_enable_tf32": os.environ.get("MLX_ENABLE_TF32", "MLX default"),
        }
        self.identity["kernel_build_id"] = digest(self.identity["kernel_build"])
        self.capabilities = {
            "model": "Qwen/Qwen3.8-27B",
            "context_window": text["max_position_embeddings"],
            "vocabulary_size": text["vocab_size"],
            "losses": ["cispo", "weighted_cross_entropy"],
            "precision": {
                "activations": "bf16",
                "lora": "bf16",
                "base": "8-bit affine/group64",
                "optimizer": "fp32",
                "recurrent_state": "fp32",
                "training_kv": "bf16",
            },
            "cache_precisions": ["bf16", "8bit"],
            "max_inference_batch": 4,
            "physical_memory_bytes": physical,
            "allocator_budget_bytes": budget,
            "host_headroom_bytes": int(headroom_gib * 1024**3),
            "recommended_working_set_bytes": self.device.get(
                "max_recommended_working_set_size"
            ),
            "tokenizer_hash": self.bundle["hash"],
            "frozen_tensors": len(self.frozen),
            "resident_base_bytes": sum(value.nbytes for _, value in self.frozen),
            "flash_requirement": {
                "context_window": 262144,
                "dense_bits": 8,
                "expert_bits": 4,
                "ngram_bits": [4, 6],
                "execution": "deferred",
            },
        }

    def select(self, factors):
        # Reference substitution only, after all factors were validated on load.
        # Never merge adapters into or replace the underlying quantized base.
        if factors is self.selected:
            return
        for name, module in self.adapter.modules.items():
            module.a, module.b = factors[name + ".a"], factors[name + ".b"]
        self.selected = factors

    def mode(self, training):
        if training != self.training:
            self.model.train(training)
            self.training = training

    def estimate_replay(
        self, sequences, checkpoint_layers=8, *, segment_tokens=2048, head_rows=128
    ):
        # Architecture-specific layouts are confined to this model implementation.
        width = self.lm.args.hidden_size
        length = max(
            len(x["prompt_token_ids"]) + len(x["sampled_tokens"]) - 1 for x in sequences
        )
        segments = (length + segment_tokens - 1) // segment_tokens
        boundaries = (
            length
            * width
            * ((len(self.lm.layers) + checkpoint_layers - 1) // checkpoint_layers)
            * 2
        )
        carries, adjoints = 0, 0
        for layer in self.lm.layers:
            if layer.is_linear:
                linear = layer.linear_attn
                size = (
                    linear.num_v_heads * linear.head_v_dim * linear.head_k_dim * 4
                    + (linear.conv_kernel_size - 1) * linear.conv_dim * 2
                )
                carries += segments * size
                adjoints += size
            else:
                attention = layer.self_attn
                size = length * attention.num_key_value_heads * attention.head_dim * 4
                carries += size
                adjoints += size
        # Branch forward/backward work is sequential. The prompt boundaries and
        # carries are shared; FP32 summed boundary adjoints overlap a branch's
        # BF16 adjoints. Full embedding inputs exist for each completion.
        branches = len(sequences)
        if length > segment_tokens:
            adjoints *= 3
        additional = (
            boundaries
            + carries
            + adjoints
            + (branches - 1) * length * (width * 2 + 12)
            + min(length, segment_tokens) * width * checkpoint_layers * 2
            + (4 << 30)
            + max(0, head_rows - 128) * self.lm.args.vocab_size * 16
        )
        return {
            "additional_bytes": additional,
            "boundary_bytes": boundaries,
            "carry_bytes": carries,
            "carry_adjoint_bytes": adjoints,
        }

    def estimate_inference(self, length, batch=1, *, cache_precision="bf16"):
        """Keep the qualified dense profile's conservative BF16 bound for both caches."""
        if cache_precision not in ("bf16", "8bit"):
            raise ValueError("Unsupported inference cache precision.")
        cache = sum(
            length * layer.self_attn.num_key_value_heads * layer.self_attn.head_dim * 4
            for layer in self.lm.layers
            if not layer.is_linear
        )
        recurrent = sum(
            layer.linear_attn.num_v_heads
            * layer.linear_attn.head_v_dim
            * layer.linear_attn.head_k_dim
            * 4
            for layer in self.lm.layers
            if layer.is_linear
        )
        return batch * (cache + recurrent) + (2 << 30)

    def admit(self, additional, reserved=0, persistent=None):
        active = self.mx.get_active_memory()
        # Live replay storage already occupies part of its peak reservation.
        # Do not charge that storage twice when admitting another GM's sample.
        needed = max(
            active + additional,
            (active if persistent is None else persistent) + reserved + additional,
        )
        if needed > self.budget:
            raise MemoryError(
                f"Admission needs {needed} bytes; budget is {self.budget}."
            )

    def frozen_digest(self):
        from mlx.utils import tree_flatten

        current = {
            name: value
            for name, value in tree_flatten(self.model.parameters())
            if not name.endswith((".a", ".b"))
        }
        return self.tensor_digest(current)

    def tensor_digest(self, tensors):
        import numpy as np

        result = hashlib.sha256()
        for name, value in sorted(tensors.items()):
            result.update(
                (
                    name + "\0" + str(value.shape) + "\0" + str(value.dtype) + "\0"
                ).encode()
            )
            raw = (
                value.view(self.mx.uint16) if value.dtype == self.mx.bfloat16 else value
            )
            result.update(np.asarray(raw).tobytes())
        return result.hexdigest()
