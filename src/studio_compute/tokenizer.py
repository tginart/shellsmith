"""Hash-verified, atomically published CPU tokenizer assets."""

import base64
import errno
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

from .protocol import digest
from .storage import sync_directory


def materialize(bundle, root=None):
    assets = bundle["files"]
    if digest({name: item["sha256"] for name, item in assets.items()}) != bundle["hash"]:
        raise ValueError("Tokenizer bundle identity differs.")
    decoded = {}
    for name, item in assets.items():
        if Path(name).name != name or name.startswith("."):
            raise ValueError("Invalid tokenizer asset name.")
        data = base64.b64decode(item["base64"], validate=True)
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Tokenizer asset hash differs: " + name)
        decoded[name] = data
    # Separate immutable publications from old runtimes' in-place cache writes.
    parent = Path(root or Path(tempfile.gettempdir()) / "studio-compute-tokenizers" / "atomic-v1")
    destination = parent / bundle["hash"]
    parent.mkdir(parents=True, exist_ok=True)

    def verify():
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("Tokenizer bundle must be a regular local directory.")
        for name, item in assets.items():
            file = destination / name
            if file.is_symlink() or not file.is_file():
                raise ValueError("Tokenizer assets must be regular local files.")
            if hashlib.sha256(file.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("Cached tokenizer asset hash differs: " + name)

    if destination.exists() or destination.is_symlink():
        verify()
        # Complete a publication whose parent sync failed after the rename.
        sync_directory(parent)
        return destination
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=parent))
    try:
        for name, data in decoded.items():
            asset = staging / name
            asset.write_bytes(data)
            with asset.open("rb") as file:
                os.fsync(file.fileno())
        sync_directory(staging)
        try:
            staging.rename(destination)
        except OSError as error:
            if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            # A concurrent publisher won. Never overwrite a visible asset.
        sync_directory(parent)
        verify()
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
