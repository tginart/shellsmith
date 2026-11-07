"""Coordinator-side immutable requests and a small SSH transport."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tempfile
import tomllib

from .bundle import pack
from .storage import canonical, digest, identifier, lock, private_dir, read, write

# Defaults are relative to the execution account's home, so no user name is baked in.
DEFAULT_EXECUTION_HOME = "~/.local/state/shellsmith"
DEFAULT_EXECUTION_PYTHON = "~/.local/share/shellsmith/studio-v1/env/bin/python"


@dataclass(frozen=True)
class Endpoint:
    host: str | None
    home: str
    python: str

    def __post_init__(self):
        if self.host and (self.host.startswith("-") or any(c.isspace() for c in self.host)):
            raise ValueError("Use one SSH hostname or account@hostname, without options")
        if not self.home or not self.python:
            raise ValueError("Execution home and Python interpreter must be explicit")


# This bootstrap uses only the destination's stdlib; project imports occur after
# checking every file against the controller's content-addressed manifest.
BOOTSTRAP = r'''
import base64, fcntl, hashlib, io, json, os, pathlib, shutil, sys, tarfile, tempfile
e = json.load(sys.stdin)
home = pathlib.Path(e['home']).expanduser().absolute()
sha = e['source_sha256']
if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha): raise ValueError('Invalid runtime identity')
for p in (home, *home.parents):
    if p.is_symlink(): raise ValueError('Symlinked execution home refused')
home.mkdir(mode=0o700, parents=True, exist_ok=True)
if home.stat().st_uid != os.getuid() or home.stat().st_mode & 0o077: raise ValueError('Execution home must be private')
runtime = home / 'runtime' / sha
for p in (runtime, runtime.parent):
    if p.is_symlink(): raise ValueError('Symlinked runtime refused')
runtime.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
stage_lock = os.open(home/'source-stage.lock', os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW, 0o600)
fcntl.flock(stage_lock, fcntl.LOCK_EX)
# Only an abandoned bootstrap can leave pending directories once this lock is held.
for pending in runtime.parent.glob('.pending-*'):
    if pending.is_dir() and not pending.is_symlink():
        shutil.rmtree(pending)
if not (runtime / 'runtime_manifest.json').exists():
    raw = base64.b64decode(e['archive'], validate=True)
    if len(raw) > 33554432: raise ValueError('Oversized source bundle')
    staging = pathlib.Path(tempfile.mkdtemp(prefix='.pending-', dir=runtime.parent))
    try:
        total, names = 0, set()
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            for item in archive:
                name = pathlib.PurePosixPath(item.name)
                total += item.size
                if not item.isfile() or name.is_absolute() or '..' in name.parts or item.name in names or total > 33554432:
                    raise ValueError('Unsafe source archive')
                names.add(item.name)
                target = staging / item.name
                for p in (target, *target.parents):
                    if p.is_symlink(): raise ValueError('Symlinked staging refused')
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                if target.exists(): raise ValueError('Partial source staging; remove only after verifying no job references it')
                with target.open('xb') as out: out.write(archive.extractfile(item).read())
                target.chmod(0o600)
        candidate = json.loads((staging/'runtime_manifest.json').read_text())
        if names != set(candidate['files']) | {'runtime_manifest.json'}:
            raise ValueError('Unmanifested source archive entries before publication')
        encoded = json.dumps(candidate['files'], sort_keys=True, separators=(',',':')).encode()
        if candidate.get('source_sha256') != sha or hashlib.sha256(encoded).hexdigest() != sha:
            raise ValueError('Source identity mismatch before publication')
        for name, wanted in candidate['files'].items():
            p = pathlib.PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or hashlib.sha256((staging/name).read_bytes()).hexdigest() != wanted:
                raise ValueError('Source contents mismatch before publication')
        os.rename(staging, runtime)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

os.close(stage_lock)
manifest = json.loads((runtime / 'runtime_manifest.json').read_text())
canonical = json.dumps(manifest['files'], sort_keys=True, separators=(',',':')).encode()
if manifest['source_sha256'] != sha or hashlib.sha256(canonical).hexdigest() != sha: raise ValueError('Source identity mismatch')
for name, wanted in manifest['files'].items():
    p = pathlib.PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe runtime manifest')
    target = runtime / name
    if target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest() != wanted: raise ValueError('Frozen source changed: '+name)
sys.path.insert(0, str(runtime / 'src'))
prefix = pathlib.Path(sys.executable).parent.parent.parent
os.environ['PATH'] = os.pathsep.join([str(pathlib.Path(sys.executable).parent), str(prefix/'node'/'bin'), str(prefix/'tools'/'node_modules'/'.bin'), os.environ.get('PATH','')])
os.environ['MLX_ENABLE_TF32'] = '0'
from shellsmith.jobs.service import dispatch
from shellsmith.jobs.bundle import verify
verify(runtime, sha)
print(json.dumps(dispatch(home, runtime, e), allow_nan=False))
'''


def ssh_transport(endpoint: Endpoint, envelope: dict) -> dict:
    if endpoint.host:
        remote_command = f"{_shell_path(endpoint.python)} -c {shlex.quote(BOOTSTRAP)}"
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=yes",
                "-o", "ControlMaster=no", "-o", "ControlPath=none", "--", endpoint.host, remote_command]
    else:
        argv = [os.path.expanduser(endpoint.python), "-c", BOOTSTRAP]
    result = subprocess.run(argv, input=canonical(envelope), capture_output=True, timeout=180)
    if result.returncode:
        # No environment or credential-bearing output is persisted in receipts.
        lines = result.stderr.decode(errors="replace").strip().splitlines()
        detail = lines[-1] if lines else f"exit {result.returncode}"
        raise RuntimeError(f"Job transport failed: {detail}")
    try:
        return json.loads(result.stdout)
    except (ValueError, UnicodeError) as error:
        raise RuntimeError("Uncertain job response; retry with the same request ID") from error


def _toml_scalar(value):
    if isinstance(value, str):
        return json.dumps(value)
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) in (int, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_scalar(item) for item in value) + "]"
    raise ValueError("Run configuration contains an unsupported value")


def _toml(document: dict) -> bytes:
    lines = []
    def table(values, names):
        if names:
            lines.append("[" + ".".join(json.dumps(name) for name in names) + "]")
        for key, value in values.items():
            if not isinstance(value, dict):
                lines.append(json.dumps(key) + " = " + _toml_scalar(value))
        lines.append("")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, [*names, key])
    table(document, [])
    return "\n".join(lines).encode()


def _without_credentials(document: dict) -> None:
    for key, value in document.items():
        if key.lower() in {"api_key", "password", "token", "secret"}:
            raise ValueError("Credentials must remain outside run configurations")
        if isinstance(value, dict):
            _without_credentials(value)


class JobClient:
    def __init__(self, endpoint: Endpoint, cache_dir: Path | None = None, transport=None):
        from shellsmith.paths import DEFAULT_LOG_ROOT
        self.endpoint = endpoint
        self.cache = private_dir(Path(cache_dir or DEFAULT_LOG_ROOT / "job-client").expanduser().resolve())
        self.transport = transport or ssh_transport

    def _call(self, envelope):
        return self.transport(self.endpoint, {"home": self.endpoint.home, **envelope})

    def _saved(self, job_id):
        row = read(self.cache / (identifier(job_id) + ".json"))
        if row["endpoint"] != asdict(self.endpoint):
            raise ValueError("Job endpoint differs from its recorded owner; use the original destination")
        return row

    def _control(self, job_id, action, **values):
        row = self._saved(job_id)
        return self._call({"action": action, "job_id": job_id, "source_sha256": row["source_sha256"], **values})

    def submit(self, config_path: Path, source_root: Path, request_id: str, *,
               command: str = "train", adapter_path: str | Path | None = None) -> dict:
        if command not in {"train", "campaign"}:
            raise ValueError("Jobs support train or campaign commands")
        if adapter_path is not None and command != "campaign":
            raise ValueError("Adapter selection is only supported for campaign jobs")
        adapter = str(adapter_path) if adapter_path is not None else None
        if adapter is not None and (not Path(adapter).is_absolute() or ".." in Path(adapter).parts):
            raise ValueError("Use an absolute execution-host sampler checkpoint path")
        request_id = identifier(request_id)
        job_id = "job-" + digest(request_id.encode())[:24]
        path = self.cache / (job_id + ".json")
        original = Path(config_path).read_bytes()
        with lock(self.cache / "client.lock"):
            if path.exists():
                row = self._saved(job_id)
                if row["original_config_sha256"] != digest(original):
                    raise ValueError("Submission request ID was already used with different configuration")
                if (row["envelope"].get("command", "train") != command or
                        row["envelope"].get("adapter_path") != adapter):
                    raise ValueError("Submission request ID was already used with different command or adapter")
            else:
                document = tomllib.loads(original.decode())
                if document.get("compute", {}).get("backend", "tinker") != "mlx":
                    raise ValueError("Studio jobs require compute.backend = \"mlx\"; Tinker remains available through train/campaign")
                _without_credentials(document)
                document.setdefault("run", {})["run_id"] = document.get("run", {}).get("run_id") or job_id
                document["run"]["log_dir"] = ""  # worker chooses the immutable job-owned directory
                slate = document.get("worlds", {}).get("hero_slate")
                if slate:
                    checkout = Path(source_root).expanduser().resolve()
                    candidate = Path(slate).expanduser()
                    if not candidate.is_absolute():
                        candidate = checkout / candidate
                    candidate = candidate.resolve()
                    try:
                        relative = candidate.relative_to(checkout)
                    except ValueError as exc:
                        raise ValueError("Hero slate must belong to the frozen source checkout") from exc
                    if relative.parts[0] != "hero_runs" or not candidate.is_file():
                        raise ValueError("Hero slate must be a checked-in file under hero_runs")
                    document["worlds"]["hero_slate"] = relative.as_posix()
                # Connection fields are coordinator settings, never recursively delegated.
                if "compute" in document:
                    document["compute"].update(host="", home="", python="")
                config = _toml(document)
                source = pack(Path(source_root))
                row = {"endpoint": asdict(self.endpoint), "original_config_sha256": digest(original),
                       "source_sha256": source["source_sha256"], "envelope": {
                           "action": "submit", "job_id": job_id, "request_id": request_id,
                           "command": command, "adapter_path": adapter,
                           "destination": asdict(self.endpoint), **source,
                           "config": base64.b64encode(config).decode(), "config_sha256": digest(config)}}
                write(path, row)
        return self._call(row["envelope"])

    def status(self, job_id):
        return self._control(job_id, "status")

    def logs(self, job_id, offset=0, limit=65536):
        return self._control(job_id, "logs", offset=offset, limit=limit)

    def stop(self, job_id, request_id):
        return self._control(job_id, "stop", request_id=identifier(request_id))

    def resume(self, job_id, request_id):
        return self._control(job_id, "resume", request_id=identifier(request_id))

    def check(self, config: dict | None = None):
        from shellsmith.paths import PROJECT_ROOT
        return self._call({"action": "check", **pack(PROJECT_ROOT), "config_document": config})

    def sync(self, job_id: str, local_dir: Path, *, page_bytes: int = 4 * 1024 * 1024) -> dict:
        """Publish a verified mirror only after every bounded metadata page arrives."""
        if type(page_bytes) is not int or not 1 <= page_bytes <= 4 * 1024 * 1024:
            raise ValueError("Invalid artifact page size")
        root = private_dir(Path(local_dir).expanduser().absolute())
        if not root.is_dir():
            raise ValueError("Local artifact cache must be a directory")
        if any(root.iterdir()):
            marker = root / "studio-job.json"
            managed = read(marker) if marker.is_file() else None
            if not isinstance(managed, dict) or managed.get("job_id") != job_id:
                raise ValueError("Nonempty local directory is not a managed mirror for this job")
            if any(path.is_symlink() for path in root.rglob("*")):
                raise ValueError("Local artifact cache refuses symlinks")
        for attempt in range(3):
            staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.sync-", dir=root.parent))
            try:
                previous_mirror = _mirror_stamp(root)
                # Keep account-owned notes and previously mirrored artifacts.
                # Only files in the new remote inventory are replaced below.
                shutil.copytree(root, staging, dirs_exist_ok=True)
                result = self._sync_into(job_id, staging, page_bytes)
                if _mirror_stamp(root) != previous_mirror:
                    raise ValueError("Local mirror changed during sync; retry after local edits settle")
                backup = None
                if root.exists():
                    backup = Path(tempfile.mkdtemp(prefix=f".{root.name}.previous-", dir=root.parent))
                    backup.rmdir()
                    root.rename(backup)
                try:
                    staging.rename(root)
                except BaseException:
                    if backup is not None:
                        backup.rename(root)
                    raise
                if backup is not None:
                    shutil.rmtree(backup, ignore_errors=True)
                return {"job_id": job_id, "local_dir": str(root), **result}
            except (ValueError, RuntimeError) as error:
                if "Artifacts changed during sync" not in str(error) or attempt == 2:
                    raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        raise AssertionError("Artifact retry loop exhausted")

    def _sync_into(self, job_id: str, staging: Path, page_bytes: int) -> dict:
        response = self._control(job_id, "artifacts", artifacts_version=2, limit=page_bytes, cursor=None)
        # Historical frozen runtimes ignore the new fields and return one shot.
        if response.get("format") != "shellsmith.job-artifacts.v2":
            for name, item in response["files"].items():
                target = _artifact_target(staging, name)
                data = base64.b64decode(item["data"], validate=True)
                if digest(data) != item["sha256"]:
                    raise ValueError("Remote artifact integrity mismatch")
                target.write_bytes(data)
                target.chmod(0o600)
            write(staging / "studio-job.json", response["job"])
            write(staging / "studio-artifact-sync.json", {"format": "shellsmith.job-artifact-sync.v1",
                  "files": {name: item["sha256"] for name, item in response["files"].items()}})
            return {"file_count": len(response["files"]), "artifacts_version": 1}
        inventory, seen, pages, total = response.get("inventory_sha256"), {}, 0, 0
        active = None
        previous_cursor = None
        try:
            while True:
                pages += 1
                chunks = response.get("chunks")
                cursor, done = response.get("next_cursor"), response.get("done")
                if (response.get("format") != "shellsmith.job-artifacts.v2"
                        or response.get("inventory_sha256") != inventory or not _sha256(inventory)
                        or type(done) is not bool or done != (cursor is None)
                        or not isinstance(chunks, list) or len(chunks) > 256 or (not chunks and not done)):
                    raise ValueError("Invalid remote artifact page")
                page_size = 0
                for chunk in chunks:
                    if (not isinstance(chunk, dict) or type(chunk.get("size")) is not int or chunk["size"] < 0
                            or type(chunk.get("offset")) is not int or chunk["offset"] < 0
                            or type(chunk.get("eof")) is not bool or not _sha256(chunk.get("file_sha256"))):
                        raise ValueError("Invalid remote artifact chunk")
                    name, size, file_sha = chunk.get("path"), chunk["size"], chunk["file_sha256"]
                    data = base64.b64decode(chunk["data"], validate=True)
                    page_size += len(data)
                    if page_size > page_bytes or digest(data) != chunk.get("sha256"):
                        raise ValueError("Remote artifact integrity mismatch or page exceeds limit")
                    if active is None:
                        target = _artifact_target(staging, name)
                        if name in seen or chunk["offset"] != 0:
                            raise ValueError("Remote artifact chunks repeat or skip a file")
                        if target.exists():
                            if not target.is_file():
                                raise ValueError("Remote artifact conflicts with an existing local directory")
                            target.unlink()
                        stream = target.open("xb")
                        target.chmod(0o600)
                        active = {"name": name, "size": size, "sha": file_sha, "offset": 0,
                                  "stream": stream, "checksum": hashlib.sha256()}
                    if (name != active["name"] or size != active["size"] or file_sha != active["sha"]
                            or chunk["offset"] != active["offset"] or active["offset"] + len(data) > size
                            or chunk["eof"] != (active["offset"] + len(data) == size)
                            or (not data and size != 0)):
                        raise ValueError("Remote artifact chunk sequence is inconsistent")
                    active["stream"].write(data)
                    active["checksum"].update(data)
                    active["offset"] += len(data)
                    total += len(data)
                    if chunk["eof"]:
                        active["stream"].flush()
                        os.fsync(active["stream"].fileno())
                        active["stream"].close()
                        if active["checksum"].hexdigest() != file_sha:
                            raise ValueError("Remote artifact full-file integrity mismatch")
                        seen[name] = file_sha
                        active = None
                if done:
                    if active is not None:
                        raise ValueError("Remote artifact transfer ended with an incomplete file")
                    break
                if not isinstance(cursor, dict) or cursor == previous_cursor:
                    raise ValueError("Invalid remote artifact continuation cursor")
                previous_cursor = cursor
                response = self._control(job_id, "artifacts", artifacts_version=2, limit=page_bytes, cursor=cursor)
            write(staging / "studio-job.json", response["job"])
            write(staging / "studio-artifact-sync.json", {"format": "shellsmith.job-artifact-sync.v2",
                  "inventory_sha256": inventory, "file_count": len(seen), "bytes": total, "pages": pages,
                  "files": seen})
            return {"file_count": len(seen), "artifacts_version": 2, "pages": pages, "bytes": total}
        finally:
            if active is not None:
                active["stream"].close()


def _sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _mirror_stamp(root: Path) -> dict:
    entries = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Local artifact cache refuses symlinks")
        info = path.stat()
        entries[path.relative_to(root).as_posix()] = (info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return entries


def _artifact_target(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\0" in name:
        raise ValueError("Unsafe remote artifact path")
    parts = PurePosixPath(name)
    if (parts.is_absolute() or ".." in parts.parts or parts.as_posix() != name
            or parts.suffix not in {".json", ".jsonl", ".md", ".txt"}
            or any(part.startswith(".") or part.lower() in {"credentials", "auth", "secrets", "node_modules"}
                   for part in parts.parts) or parts.name.lower() in {"auth.json", "credentials.json", "secrets.json"}
            or name in {"studio-job.json", "studio-artifact-sync.json"}):
        raise ValueError("Unsafe remote artifact path")
    target = root.joinpath(*parts.parts)
    private_dir(target.parent)
    return target


def _shell_path(path: str) -> str:
    # Leave a leading "~/" unquoted so that the remote login shell expands it.
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)
