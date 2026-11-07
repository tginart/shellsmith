"""One cooperative GPU owner per account, shared by all MLX entrypoints."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path

from .storage import lock, now, private_dir, process_identity, write


class GpuBusyError(RuntimeError):
    pass


class GpuLease:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or os.environ.get("SHELLSMITH_GPU_LEASE_DIR") or
                         "~/.local/state/shellsmith/gpu-lease").expanduser()
        self.stack = ExitStack()

    def __enter__(self):
        private_dir(self.root)
        try:
            self.stack.enter_context(lock(self.root / "owner.lock", nonblocking=True))
        except BlockingIOError as error:
            self.stack.close()
            raise GpuBusyError("Another Shellsmith process owns the Studio GPU lease") from error
        try:
            write(self.root / "owner.json", {"identity": process_identity(os.getpid()), "acquired": now()})
        except BaseException as error:
            try:
                self.stack.close()
            except BaseException as cleanup_error:
                error.add_note(f"GPU lease cleanup also failed: {cleanup_error}")
            raise
        return self

    def __exit__(self, *arguments):
        primary_error = arguments[1]
        try:
            try:
                write(self.root / "owner.json", {"identity": None, "released": now()})
            except BaseException as error:
                if primary_error is None:
                    primary_error = error
                    raise
                primary_error.add_note(f"GPU lease release metadata also failed: {error}")
        finally:
            try:
                released = self.stack.__exit__(*arguments)
            except BaseException as cleanup_error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"GPU lease cleanup also failed: {cleanup_error}")
                released = False
        return released
