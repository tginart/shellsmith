"""Exercise the stdlib bootstrap before any staged project import occurs."""
import base64
import io
import json
import subprocess
import sys
import tarfile

import pytest

from shellsmith.jobs.bundle import pack
from shellsmith.jobs.client import BOOTSTRAP


@pytest.mark.parametrize("extra", [None, "src/mlx/__init__.py", "outside.txt", "src/./shellsmith/extra.py"])
def test_bootstrap_requires_exact_archive_inventory(tmp_path, extra):
    source = tmp_path / "source"
    worker = source / "src/shellsmith/jobs/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# pinned worker\n")
    payload = pack(source)
    if extra:
        raw = base64.b64decode(payload["archive"])
        output = io.BytesIO()
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as old:
            with tarfile.open(fileobj=output, mode="w:gz") as new:
                for entry in old:
                    new.addfile(entry, old.extractfile(entry))
                content = b"raise RuntimeError('untrusted module imported')\n"
                entry = tarfile.TarInfo(extra)
                entry.size = len(content)
                new.addfile(entry, io.BytesIO(content))
        payload["archive"] = base64.b64encode(output.getvalue()).decode()
    home = tmp_path / "execution"
    payload["home"] = str(home)
    abandoned = home / "runtime" / ".pending-abandoned"
    abandoned.mkdir(parents=True)
    home.chmod(0o700)
    (abandoned / "partial-source").write_text("interrupted extraction")
    # Run the actual bootstrap staging and integrity checks, stopping at its
    # first project-import boundary. No job or numerical runtime is started.
    script = BOOTSTRAP.split("sys.path.insert(0", 1)[0]
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                            text=True, capture_output=True, timeout=15)
    runtime = home / "runtime" / payload["source_sha256"]
    assert not abandoned.exists()
    if extra:
        assert result.returncode != 0
        assert "Unmanifested source archive entries before publication" in result.stderr
        assert not runtime.exists()
        assert not list(runtime.parent.glob(".pending-*"))
        # Rejected retries must not accumulate abandoned extraction directories.
        retry = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                               text=True, capture_output=True, timeout=15)
        assert retry.returncode != 0
        assert not list(runtime.parent.glob(".pending-*"))
    else:
        assert result.returncode == 0, result.stderr
        assert (runtime / "runtime_manifest.json").is_file()
        assert (runtime / "src/shellsmith/jobs/worker.py").read_text() == "# pinned worker\n"
        # Normal retries can reuse the successfully published runtime.
        retry = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                               text=True, capture_output=True, timeout=15)
        assert retry.returncode == 0, retry.stderr


def test_bootstrap_cleans_failed_manifest_validation(tmp_path):
    source = tmp_path / "source"
    worker = source / "src/shellsmith/jobs/worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("# fixture\n")
    payload = pack(source)
    payload.update(home=str(tmp_path / "execution"), source_sha256="0" * 64)
    script = BOOTSTRAP.split("sys.path.insert(0", 1)[0]
    result = subprocess.run([sys.executable, "-c", script], input=json.dumps(payload),
                            text=True, capture_output=True, timeout=15)
    assert result.returncode != 0 and "Source identity mismatch before publication" in result.stderr
    assert not list((tmp_path / "execution/runtime").glob(".pending-*"))
    assert not (tmp_path / "execution/runtime" / ("0" * 64)).exists()
