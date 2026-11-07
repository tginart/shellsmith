"""Content identities for local model assets and native backend semantics."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .readiness import CANONICAL_TARGETS, PINS, model_metadata
from .profiles import MEMORY_GIB, checkpoint_layers
from .generation import completion_budget, reasoning_effort

IMPLEMENTATION = "shellsmith-mlx-native-sequential-v1"
HIGH_COMMAND_REFERENCE = "10e830e9a5becb9c2387bd37a378fd7dad1d3590"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def model_identity(folder: str | Path) -> dict[str, Any]:
    """Hash actual local assets, including tokenizer/template and every shard.

    Local metadata's claimed upstream revision never substitutes for hashes.
    Reject a shard being changed while hashed rather than admitting mixed assets.
    """
    folder = Path(folder).expanduser().resolve()
    raw, text, kinds = model_metadata(folder)
    paths = sorted(path for path in folder.iterdir() if path.is_file() and not path.name.startswith("."))
    files: dict[str, str] = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Model identity rejects symlinked asset: {path.name}")
        before = path.stat()
        files[path.name] = file_digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise RuntimeError(f"MLX model asset changed while hashing: {path.name}")
    return {"path": str(folder), "asset_sha256": digest(files), "files": files,
            "model_type": raw["model_type"], "layer_types": list(kinds),
            "max_position_embeddings": text["max_position_embeddings"],
            "quantization": {"bits": 8, "group_size": 64}}


def policy_identity(settings: Any, model: dict[str, Any]) -> dict[str, Any]:
    """Freeze model, renderer, numerical runtime and approved adapter semantics."""
    versions = {name: importlib.metadata.version(name) for name in PINS}
    package = Path(__file__).resolve().parent.parent
    # Native rendering uses shared message/tool normalization and TokenPrompt.
    # Package-relative names keep identities independent of checkout location.
    paths = sorted([*(package / "mlx").glob("*.py"), package / "policy" / "chat_adapter.py",
                    package / "policy" / "sampler.py", package / "policy" / "reasoning.py",
                    package / "train" / "backend.py"])
    source = {path.relative_to(package).as_posix(): file_digest(path) for path in paths}
    template_sha = model["files"]["chat_template.jinja"]
    tokenizer_files = {name: sha for name, sha in model["files"].items()
                       if name.startswith("tokenizer") or name in {"vocab.json", "merges.txt", "special_tokens_map.json"}}
    result = {"backend": "mlx", "implementation": IMPLEMENTATION, "source_sha256": digest(source),
              "source_files": source,
              "reference": HIGH_COMMAND_REFERENCE, "runtime": versions, "model": model,
              "numeric_precision": {"MLX_ENABLE_TF32": "0"},
              "model_id": settings.model_id, "profile": settings.profile,
              "renderer": {"implementation": "native-qwen38-template-v1", "template_sha256": template_sha,
                           "tokenizer_sha256": digest(tokenizer_files), "thinking_effort": reasoning_effort(settings)},
              "adapter": {"rank": 16, "scale": 2.0, "layers": "all", "targets": list(CANONICAL_TARGETS)},
              "optimizer": {"implementation": "fp32-master-adam-v1", "beta1": 0.9, "beta2": 0.95,
                            "eps": 1e-8, "weight_decay": 0.0, "gradient_clip": None},
              "replay": {"segment_tokens": 2048, "checkpoint_layers": checkpoint_layers(settings.profile),
                         "memory_limit_gib": MEMORY_GIB, "storage": "bf16",
                         "reduction": "sum", "attention": "native-tiled-causal-sdpa", "gdn": "native-autodiff"},
              "sampling": {"temperature": 1.0, "top_p": 1.0, "context_window": settings.context_window,
                           "max_tokens": completion_budget(settings), "thinking_tokens": settings.thinking_tokens,
                           "response_tokens": settings.response_tokens, "plain_tokens": settings.plain_tokens,
                           "compaction_tokens": settings.compaction_tokens,
                           "compaction_reserve_tokens": settings.compaction_reserve_tokens,
                           "compaction_keep_recent_tokens": settings.compaction_keep_recent_tokens}}
    result["identity_sha256"] = digest(result)
    return result


def resolve_identity(settings: Any) -> dict[str, Any]:
    """Resolve immutable assets before trainer resume checks or GPU allocation."""
    return policy_identity(settings, model_identity(settings.model_path))
