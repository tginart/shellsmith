"""Private atomic records and identity checks for the job owner."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import platform
import stat
import subprocess
import tempfile
import time


def identifier(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise ValueError("Job and request IDs must be safe identifiers of at most 96 characters")
    return value


def private_dir(path: Path) -> Path:
    path = path.expanduser().absolute()
    # Refuse symlink aliases anywhere in managed storage, including an existing leaf.
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError("Managed job storage must not contain symlinks")
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        directory.mkdir(mode=0o700, exist_ok=True)
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
        raise PermissionError("Job storage must be account-owned and private (0700)")
    return path


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read(path: Path):
    if path.is_symlink():
        raise ValueError("Refusing a symlinked job record")
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    if path.is_symlink():
        raise ValueError("Refusing a symlinked job record")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        os.fchmod(stream.fileno(), 0o600)
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def lock(path: Path, *, nonblocking=False):
    private_dir(path.parent)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_uid != os.getuid():
            raise PermissionError("Unsafe job lock")
        fcntl.flock(descriptor, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        yield
    finally:
        os.close(descriptor)


def boot_identity() -> str:
    linux = Path("/proc/sys/kernel/random/boot_id")
    if linux.exists():
        return linux.read_text().strip()
    raw = subprocess.check_output(["sysctl", "-n", "kern.boottime"], text=True, timeout=5).strip()
    found = re.search(r"sec\s*=\s*(\d+),\s*usec\s*=\s*(\d+)", raw)
    if not found:
        raise RuntimeError("Cannot establish kernel boot identity")
    return ":".join(found.groups())


def _birth_identity(pid: int) -> list[int] | None:
    """Use kernel birth precision, rather than ps's one-second timestamp."""
    if platform.system() == "Darwin":
        # SDK sys/proc_info.h: proc_bsdinfo, PROC_PIDTBSDINFO=3.
        class BsdInfo(ctypes.Structure):
            _fields_ = [("header", ctypes.c_uint32 * 12), ("comm", ctypes.c_char * 16),
                        ("name", ctypes.c_char * 32), ("details", ctypes.c_uint32 * 6),
                        ("start_seconds", ctypes.c_uint64), ("start_microseconds", ctypes.c_uint64)]
        try:
            function = ctypes.CDLL("/usr/lib/libproc.dylib").proc_pidinfo
            function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
            function.restype = ctypes.c_int
            info = BsdInfo()
            if function(pid, 3, 0, ctypes.byref(info), ctypes.sizeof(info)) != ctypes.sizeof(info):
                return None
            if info.header[3] != pid or info.header[5] != os.getuid():
                return None
            return [info.start_seconds, info.start_microseconds]
        except (OSError, AttributeError):
            return None
    try:
        row = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return [int(row[19])]
    except (OSError, ValueError, IndexError):
        return None


def owner_identity() -> dict:
    linux = Path("/etc/machine-id")
    if linux.is_file():
        machine = linux.read_text().strip()
    else:
        raw = subprocess.check_output(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"], text=True, timeout=5)
        found = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', raw)
        if not found:
            raise RuntimeError("Cannot establish the selected machine's stable identity")
        machine = found.group(1)
    return {"uid": os.getuid(), "machine": machine}


def process_identity(pid: int) -> dict | None:
    try:
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart=", "-o", "stat=", "-o", "command="],
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    pieces = value.split(maxsplit=6)
    if result.returncode or len(pieces) != 7 or "Z" in pieces[5]:
        return None
    birth = _birth_identity(pid)
    if birth is None:
        return None
    # Keep only process birth, not changing scheduler status flags.
    command = pieces[6]
    # macOS's Python launcher can replace its executable pathname immediately
    # after Popen; the module and its immutable arguments retain their identity.
    marker = command.find(" -m ")
    arguments = command[marker + 1:] if marker >= 0 else command.split(" ", 1)[-1]
    return {"pid": pid, "boot": boot_identity(), "birth": birth,
            "started": " ".join(pieces[:5]), "command": arguments}


def alive(identity: dict | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


def now() -> float:
    return time.time()
