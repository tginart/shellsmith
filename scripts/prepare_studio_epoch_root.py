"""
Purpose: Prepare a fresh Studio epoch root from a previous root and a new frozen source.
Input: The previous root, a new root path and run ID, a source tar archive, and the
    planned scheduling and replay settings.
Output: A root with source/, runtime_manifest.json, train.toml, service.toml,
    codex-account.json, and epoch-plan.json. The previous root is only read.
Note: Only scheduling, replay execution, and host headroom settings change.
    Attempts, budgets, LoRA, precision, GM settings, and loss settings are copied
    unchanged.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tomllib
from datetime import UTC, datetime
from pathlib import Path

NODE_MODULES = str(Path.home() / ".local/share/shellsmith/releases/89bfcffdfb72"
                   / "src/shellsmith/episode/vat_runner/node_modules")
STUDIO_MEMORY_GIB = 256


def main() -> None:
    args = _parse_args()
    previous, root = args.previous.resolve(), args.root
    assert not root.exists(), root
    plan = json.loads((previous / "epoch-plan.json").read_text())
    root.mkdir(mode=0o700)
    source = root / "source"
    source.mkdir()
    subprocess.run(["tar", "-x", "-C", str(source), "-f", str(args.archive)], check=True)
    source_sha256 = _write_manifest(source)
    (source / "src/shellsmith/episode/vat_runner/node_modules").symlink_to(NODE_MODULES)
    old_id = plan["run_id"]
    root = root.resolve()
    train = _rewrite((previous / "train.toml").read_text(), previous, root, old_id,
                     args.run_id, args.port)
    train = _set(train, "parallelism", args.episode_parallelism)
    train = _set(train, "parallelism_per_provider", args.episode_parallelism)
    if args.steps is not None:
        train = _set(train, "steps", args.steps)
    service = _rewrite((previous / "service.toml").read_text(), previous, root, old_id,
                       args.run_id, args.port)
    service = _set(service, "inference_concurrency", args.inference_concurrency)
    service = _put(service, "replay_layout", f'"{args.replay_layout}"')
    service = _put(service, "training_moe_chunk", str(args.training_moe_chunk))
    service = _put(service, "prefill_moe_chunk", str(args.prefill_moe_chunk))
    service = _set(service, "prefill_tokens", args.prefill_tokens)
    service = _put(service, "headroom_gib", f"{args.headroom_gib:.1f}")
    service = re.sub(r"headroom: \d+ GiB on Studio",
                     f"headroom: {STUDIO_MEMORY_GIB - args.headroom_gib:g} GiB on Studio",
                     service)
    service = service.replace("eight active slots",
                              f"{args.inference_concurrency} active slots")
    (root / "train.toml").write_text(train)
    (root / "service.toml").write_text(service)
    tomllib.loads(train), tomllib.loads(service)
    shutil.copy2(previous / "codex-account.json", root / "codex-account.json")
    usage = shutil.disk_usage(root)
    plan.update({
        "root": str(root), "run_id": args.run_id, "runtime_root": str(source),
        "source_sha256": source_sha256,
        "train_config": str(root / "train.toml"),
        "service_config": str(root / "service.toml"),
        "train_config_sha256": _file_hash(root / "train.toml"),
        "service_config_sha256": _file_hash(root / "service.toml"),
        "hero_slate_sha256": _file_hash(source / "hero_runs/studio_catalog_epoch_v1.json"),
        "backprop_url": plan["backprop_url"].replace(old_id, args.run_id),
        "endpoint": f"http://127.0.0.1:{args.port}",
        "episode_parallelism": args.episode_parallelism,
        "inference_concurrency": args.inference_concurrency,
        "replay_layout": args.replay_layout,
        "training_moe_chunk": args.training_moe_chunk,
        "prefill_moe_chunk": args.prefill_moe_chunk,
        "prefill_tokens": args.prefill_tokens,
        "headroom_gib": args.headroom_gib,
        "numerical_sources_unchanged": False,
        "optimization_baseline_source": str(previous / "source"),
        "optimization_benchmark": args.evidence,
        "replacement_for": str(previous),
        "replacement_reason": args.reason,
        "initialization": ("fresh optimizer and factors from the normal configured "
                           "seed; full epoch starts at first world"),
        "prepared_at_utc": datetime.now(UTC).isoformat(),
        "disk_free_bytes_at_prepare": usage.free,
        "status": "ready_to_dispatch",
    })
    for key in ("optimization_validation_sha256", "disk_free_bytes_before_launch",
                "inference_optimization_validated"):
        plan.pop(key, None)
    (root / "epoch-plan.json").write_text(json.dumps(plan, sort_keys=True))
    print(json.dumps({"root": str(root), "source_sha256": source_sha256}))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True, help="The previous root.")
    parser.add_argument("--root", type=Path, required=True, help="The new root.")
    parser.add_argument("--run-id", required=True, help="The new run ID.")
    parser.add_argument("--archive", type=Path, required=True, help="Source tar file.")
    parser.add_argument("--port", type=int, default=8767, help="Service port.")
    # Smoke 2 measured 16 episodes 1.9x slower per episode than r3's 8.
    parser.add_argument("--episode-parallelism", type=int, default=8,
                        help="Concurrent episodes.")
    # Smoke 1 showed 16 decode slots starve the prefix cache of reserved memory.
    parser.add_argument("--inference-concurrency", type=int, default=8,
                        help="Concurrent GPU decode slots.")
    parser.add_argument("--replay-layout", default="segment_trie",
                        help="Service replay layout.")
    parser.add_argument("--training-moe-chunk", type=int, default=32,
                        help="Grouped training expert tile rows, or 0.")
    # Gate numerics-v2 and bench prefill-v2: tiles pass at 1024-token chunks.
    parser.add_argument("--prefill-moe-chunk", type=int, default=32,
                        help="Grouped prompt-prefill expert tile rows, or 0.")
    parser.add_argument("--prefill-tokens", type=int, default=1024,
                        help="Prompt tokens per prefill chunk between decode steps.")
    # perf4 hung the Studio at 240 GiB of 256 GiB with 16 GiB of headroom.
    parser.add_argument("--headroom-gib", type=float, default=32.0,
                        help="Host memory outside the MLX allocator, in GiB.")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override train steps (smoke roots only).")
    parser.add_argument("--evidence", required=True, help="Benchmark evidence path.")
    parser.add_argument("--reason", required=True, help="Replacement reason.")
    return parser.parse_args()


def _rewrite(text: str, previous: Path, root: Path, old_id: str, new_id: str,
             port: int) -> str:
    """Move root paths first, then the remaining run ID strings and the port."""
    text = text.replace(str(previous), str(root))
    return text.replace(old_id, new_id).replace("127.0.0.1:8767", f"127.0.0.1:{port}") \
        .replace("port = 8767", f"port = {port}")


def _put(text: str, key: str, literal: str) -> str:
    """Set one TOML value. Insert it after head_rows when the previous root has none."""
    text, count = re.subn(rf"(?m)^{key} = .*$", f"{key} = {literal}", text)
    assert count <= 1, key
    if count == 0:
        assert text.count("head_rows = 128\n") == 1, key
        text = text.replace("head_rows = 128\n", f"head_rows = 128\n{key} = {literal}\n")
    return text


def _set(text: str, key: str, value: int) -> str:
    text, count = re.subn(rf"(?m)^{key} = \d+", f"{key} = {value}", text)
    assert count == 1, key
    return text


def _write_manifest(root: Path) -> str:
    files = {}
    for path in sorted(root.rglob("*")):
        assert not path.is_symlink(), path
        if path.is_file():
            files[path.relative_to(root).as_posix()] = _file_hash(path)
    source_sha256 = hashlib.sha256(_canonical(files)).hexdigest()
    (root / "runtime_manifest.json").write_bytes(
        _canonical({"source_sha256": source_sha256, "files": files}))
    return source_sha256


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
