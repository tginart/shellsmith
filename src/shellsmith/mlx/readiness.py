"""Read-only admission checks; importing this module is portable to Linux."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

from .profiles import PROFILE_CONTEXTS, QUALIFICATION_PROFILES

PINS = {"mlx": "0.32.3", "mlx-lm": "0.32.0", "mlx-vlm": "0.7.4", "transformers": "5.17.0"}
MIN_MACOS_MAJOR = 26
MIN_PHYSICAL_MEMORY_BYTES = 256 * 1024**3
PROFILE = "qwen38_dense_8bit_32k"
# Exact supplied config.json fingerprint recorded in the actual Studio native
# report (docs/mlx-qualification/studio-r3.json, policy_identity.model.files)
# and both saved live canaries. This pins every architecture/quantization field.
APPROVED_CONFIG_SHA256 = "8f80874ac3ad8fa386d3f6dc0ea85377f703376e009a03dee0360e08e289a25d"
GDN_TARGETS = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")
ATTENTION_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")
CANONICAL_TARGETS = GDN_TARGETS + ATTENTION_TARGETS + MLP_TARGETS


def _setting(settings: Any, name: str, default: Any = None) -> Any:
    return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)


def capabilities(*, probe_metal: bool = True) -> dict[str, Any]:
    """Report host/package blockers without loading model weights."""
    blockers: list[str] = []
    versions: dict[str, str | None] = {}
    system, architecture = platform.system(), platform.machine()
    if system != "Darwin" or architecture != "arm64":
        blockers.append("MLX training requires Apple Silicon macOS on the execution host.")
    else:
        try:
            major = int(platform.mac_ver()[0].split(".")[0])
        except ValueError:
            major = 0
        if major < MIN_MACOS_MAJOR:
            blockers.append(f"The pinned MLX runtime requires macOS {MIN_MACOS_MAJOR} or later.")
    for name, wanted in PINS.items():
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != wanted:
            blockers.append(f"Install {name}=={wanted} in the MLX execution environment.")
    metal = False
    precision = os.environ.setdefault("MLX_ENABLE_TF32", "0")
    if precision != "0":
        blockers.append("MLX_ENABLE_TF32 must be 0 before loading the pinned numerical runtime.")
    if probe_metal and not blockers:
        try:
            # MLX otherwise permits reduced-precision FP32 matrix operations.
            # This must be selected before the first numerical runtime import.
            import mlx.core as mx

            metal = bool(mx.metal.is_available())
            if not metal:
                blockers.append("Metal is unavailable in this execution environment.")
        except Exception as exc:
            blockers.append(f"MLX runtime failed to load: {type(exc).__name__}: {exc}")
    memory = None
    if system == "Darwin":
        try:
            value = subprocess.run(["/usr/sbin/sysctl", "-n", "hw.memsize"], capture_output=True,
                                   text=True, check=True, timeout=5).stdout.strip()
            memory = int(value)
        except (OSError, ValueError, subprocess.SubprocessError):
            blockers.append("Unable to determine physical memory with sysctl hw.memsize.")
    return {"os": system, "architecture": architecture, "python": sys.executable,
            "versions": versions, "metal": metal, "memory_bytes": memory,
            "numeric_precision": {"MLX_ENABLE_TF32": precision}, "blockers": blockers}


def model_metadata(model_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    """Validate the admitted local dense artifact before any numerical import."""
    folder = Path(model_path).expanduser().resolve()
    raw = json.loads((folder / "config.json").read_text())
    if not isinstance(raw, dict):
        raise ValueError("MLX model config.json must be an object.")
    text = raw.get("text_config", raw)
    if not isinstance(text, dict) or raw.get("model_type") != "qwen3_5" or text.get("num_experts", 0):
        raise ValueError("MLX v1 admits only dense qwen3_5 text decoders.")
    kinds = text.get("layer_types")
    if not (isinstance(kinds, list) and kinds and all(k in {"linear_attention", "full_attention"} for k in kinds)):
        raise ValueError("Model text_config.layer_types must specify every dense decoder layer.")
    if type(text.get("num_hidden_layers")) is not int or text["num_hidden_layers"] != len(kinds):
        raise ValueError("Model layer count disagrees with layer_types.")
    interval = text.get("full_attention_interval")
    if type(interval) is not int or interval < 1 or any(
            kind != ("full_attention" if (i + 1) % interval == 0 else "linear_attention")
            for i, kind in enumerate(kinds)):
        raise ValueError("Model full_attention_interval disagrees with layer_types.")
    quantization = raw.get("quantization", raw.get("quantization_config"))
    if not isinstance(quantization, dict) or quantization.get("bits") != 8 or quantization.get("group_size") != 64:
        raise ValueError("MLX v1 requires the supplied 8-bit, group-size-64 model artifact.")
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map or any(
            not isinstance(name, str) or not name or Path(name).name != name
            for name in weight_map.values()):
        raise ValueError("Model weight index is missing, malformed, or contains nonlocal assets.")
    for name in set(weight_map.values()):
        path = folder / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked model weight shard: {name}")
    for name in ("tokenizer_config.json", "chat_template.jinja"):
        if not (folder / name).is_file():
            raise ValueError(f"MLX model is missing {name}.")
    return raw, text, tuple(kinds)


def readiness(settings: Any, *, probe_metal: bool = True) -> dict[str, Any]:
    """Inspect the fixed v1 profile and local model, without creating files."""
    caps = capabilities(probe_metal=probe_metal)
    blockers = list(caps["blockers"])
    memory = caps["memory_bytes"]
    if type(memory) is not int or memory < MIN_PHYSICAL_MEMORY_BYTES:
        detected = f"{memory} bytes" if type(memory) is int and memory > 0 else "unknown"
        blockers.append("MLX v1 requires at least 256 GiB physical unified memory (274877906944 bytes) "
                        "for the fixed 128 GiB MLX budget and host headroom; "
                        f"detected {detected}. Use a qualified Studio with sufficient physical memory.")
    profile = _setting(settings, "profile", PROFILE)
    if profile not in PROFILE_CONTEXTS:
        blockers.append(f"MLX supports {PROFILE!r} and explicit context qualification candidates only.")
    if (_setting(settings, "lora_rank", 16), _setting(settings, "lora_scale", 2.0),
            _setting(settings, "lora_layers", -1)) != (16, 2.0, -1):
        blockers.append("MLX v1 requires the canonical all-layer rank-16, scale-2 adapter.")
    targets = _setting(settings, "lora_targets", CANONICAL_TARGETS)
    if set(targets) != set(CANONICAL_TARGETS):
        blockers.append("MLX v1 requires the complete canonical GDN, attention, and MLP target set.")
    context = _setting(settings, "context_window", 32768)
    if type(context) is not int or context != PROFILE_CONTEXTS.get(profile):
        blockers.append(f"The selected MLX profile context window is exactly {PROFILE_CONTEXTS.get(profile)} tokens.")
    model = Path(_setting(settings, "model_path", "")).expanduser()
    info: dict[str, Any] = {}
    try:
        raw, text, kinds = model_metadata(model)
        config_sha256 = hashlib.sha256((model / "config.json").read_bytes()).hexdigest()
        if config_sha256 != APPROVED_CONFIG_SHA256:
            blockers.append("The qualified Qwen3.8-27B profile requires the approved config.json SHA256 "
                            f"{APPROVED_CONFIG_SHA256}; detected {config_sha256}. "
                            "Restore the untouched config.json supplied with the qualified model.")
        if text.get("hidden_size") != 5120 or text.get("num_hidden_layers") != 64:
            blockers.append("The qualified Qwen3.8-27B profile requires hidden_size=5120 and exactly 64 dense decoder layers.")
        quantization = raw.get("quantization", raw.get("quantization_config"))
        if quantization.get("mode") != "affine":
            blockers.append("The qualified Qwen3.8-27B artifact requires affine 8-bit quantization.")
        if type(text.get("max_position_embeddings")) is not int or context > text["max_position_embeddings"]:
            blockers.append("Configured MLX context exceeds the artifact's positional capacity.")
        index = json.loads((model / "model.safetensors.index.json").read_text())
        size = sum((model / name).stat().st_size for name in set(index["weight_map"].values()))
        info = {"path": str(model.resolve()), "model_type": raw["model_type"], "layers": len(kinds),
                "config_sha256": config_sha256,
                "weight_bytes": size, "quantization": {"bits": 8, "group_size": 64}}
        if caps["memory_bytes"] is not None and size >= caps["memory_bytes"]:
            blockers.append("Model weight assets exceed physical memory on this host.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        blockers.append(f"Local MLX model admission failed: {exc}")
    return {"ready": not blockers, "blockers": blockers, "capabilities": caps, "model": info,
            "profile": profile, "qualification_candidate": profile in QUALIFICATION_PROFILES}


def require_ready(settings: Any) -> dict[str, Any]:
    report = readiness(settings)
    if report["blockers"]:
        raise RuntimeError("MLX execution is unavailable:\n- " + "\n- ".join(report["blockers"]))
    return report
