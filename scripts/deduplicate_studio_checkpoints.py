"""
Purpose: Share byte-identical immutable checkpoint payloads in inactive Studio stores.
Input: Explicit service roots. Output: An audit receipt with hashes and physical bytes saved.
Note: Each Store lock stays held. All paths, manifests, and distinct bytes remain intact.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import uuid
from pathlib import Path

from studio_compute.storage import Store, atomic_json, file_hash, sync_directory


def deduplicate(roots: list[Path]) -> dict:
    """
    Purpose: Replace verified duplicate payload inodes with atomic hard links.
    Input: Inactive service roots on one filesystem. Output: A physical-space audit.
    Method:
        1. Hold every Store lock and verify every checkpoint manifest.
        2. Hash each distinct payload inode and check the manifest digest.
        3. Replace every duplicate path after its bytes match the retained payload.
    """
    stores = []
    replacements = []
    freed = 0
    try:
        for root in sorted(set(roots)):
            stores.append(Store(root))
        groups = {}
        for store in stores:
            for folder in store.checkpoints.iterdir():
                if folder.name.startswith(".") or (folder / "retired.json").exists():
                    continue
                _, doc = store.verify("studio-state:" + folder.name)
                for name in ("adapters.safetensors", "optimizer.safetensors"):
                    if name not in doc["files"]:
                        continue
                    path = folder / name
                    info = path.lstat()
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError("Payload must be a regular file: " + str(path))
                    key = (doc["files"][name], info.st_size, info.st_dev, info.st_mode)
                    groups.setdefault(key, {}).setdefault(info.st_ino, []).append(path)
        for (expected, size, device, mode), inodes in groups.items():
            if len(inodes) < 2:
                continue
            for paths in inodes.values():
                if file_hash(paths[0]) != expected:
                    raise ValueError("Payload hash differs: " + str(paths[0]))
            canonical_inode, canonical_paths = next(iter(inodes.items()))
            source = canonical_paths[0]
            for inode, paths in inodes.items():
                if inode == canonical_inode:
                    continue
                # Account for space only when all hard links belong to this audit.
                owned_all_links = paths[0].stat().st_nlink == len(paths)
                for target in paths:
                    info = target.lstat()
                    if (
                        info.st_ino != inode
                        or info.st_dev != device
                        or info.st_mode != mode
                    ):
                        raise RuntimeError(
                            "Payload changed during the exclusive audit."
                        )
                    temporary = target.with_name(".dedup-" + uuid.uuid4().hex)
                    try:
                        os.link(source, temporary)
                        os.replace(temporary, target)
                        sync_directory(target.parent)
                    finally:
                        temporary.unlink(missing_ok=True)
                    replacements.append(
                        {
                            "path": str(target),
                            "source": str(source),
                            "sha256": expected,
                            "size_bytes": size,
                        }
                    )
                if owned_all_links:
                    freed += size
        return {"payload_bytes_released": freed, "replacements": replacements}
    finally:
        for store in reversed(stores):
            store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--service-root",
        type=Path,
        action="append",
        required=True,
        help="An inactive service root to lock and audit.",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="A new audit receipt path outside the service roots.",
    )
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("Use a fresh receipt path.")
    receipt = deduplicate(args.service_root)
    atomic_json(args.receipt, receipt)
    print(
        json.dumps(
            {
                "payload_bytes_released": receipt["payload_bytes_released"],
                "paths_replaced": len(receipt["replacements"]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
