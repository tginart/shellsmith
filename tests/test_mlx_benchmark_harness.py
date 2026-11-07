"""Verify benchmark process ownership and the exact worker source identity."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("custom_script", [False, True])
def test_supervisor_runs_and_records_the_selected_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, custom_script: bool
) -> None:
    """Purpose: Prevent a shared supervisor from launching its own wrong CLI."""
    module = _load_harness()
    lease = tmp_path / "lease"
    lease.touch()
    output = tmp_path / "evidence"
    script = tmp_path / "selected.py"
    script.write_text("print('selected worker')\n")
    expected = script if custom_script else Path(module.__file__).resolve()
    child = Mock(pid=98231, returncode=0)
    popen = Mock(return_value=child)
    monkeypatch.setattr(module.subprocess, "Popen", popen)
    monkeypatch.setattr(module.fcntl, "flock", lambda *_: None)
    monkeypatch.setattr(module, "_command", lambda _: "unavailable in unit test")
    monkeypatch.setattr(module.sys, "argv", [str(expected), "--output", str(output)])
    args = SimpleNamespace(output=output, lease=lease, timeout_seconds=7)

    module.supervise(args, script=script if custom_script else None)

    command = popen.call_args.args[0]
    assert Path(command[2]) == expected
    assert command[-1] == "--worker"
    assert popen.call_args.kwargs["start_new_session"] is True
    child.wait.assert_called_once_with(timeout=7)
    receipt = json.loads((output / "execution.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["worker_pid"] == child.pid
    assert receipt["source_sha256"] == hashlib.sha256(expected.read_bytes()).hexdigest()
    assert (output / "source.py").read_bytes() == expected.read_bytes()
    assert (output / "supervisor-source.py").read_bytes() == Path(
        module.__file__
    ).read_bytes()


def test_supervisor_timeout_terminates_only_its_owned_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Purpose: Verify the hard deadline stops the worker without a broad kill."""
    module = _load_harness()
    lease = tmp_path / "lease"
    lease.touch()
    output = tmp_path / "evidence"
    child = Mock(pid=98231, returncode=-15)
    child.wait.side_effect = [module.subprocess.TimeoutExpired("worker", 7), None]
    kill = Mock()
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(module.fcntl, "flock", lambda *_: None)
    monkeypatch.setattr(module.os, "killpg", kill)
    monkeypatch.setattr(module, "_command", lambda _: "unavailable in unit test")
    monkeypatch.setattr(module.sys, "argv", [module.__file__])

    with pytest.raises(SystemExit) as failure:
        module.supervise(SimpleNamespace(output=output, lease=lease, timeout_seconds=7))
    assert failure.value.code == 124

    kill.assert_called_once_with(child.pid, module.signal.SIGTERM)
    receipt = json.loads((output / "execution.json").read_text())
    assert receipt["status"] == "failed" and receipt["timed_out"]
    assert receipt["returncode"] == -15


@pytest.mark.parametrize("returncode", [1, 7, -6])
def test_supervisor_propagates_worker_failure(tmp_path, monkeypatch, returncode):
    """Purpose: Prevent a failed qualification from appearing successful to a shell."""
    module = _load_harness()
    lease = tmp_path / "lease"
    lease.touch()
    output = tmp_path / "evidence"
    child = Mock(pid=98231, returncode=returncode)
    monkeypatch.setattr(module.subprocess, "Popen", Mock(return_value=child))
    monkeypatch.setattr(module.fcntl, "flock", lambda *_: None)
    monkeypatch.setattr(module, "_command", lambda _: "unavailable in unit test")
    monkeypatch.setattr(module.sys, "argv", [module.__file__])
    with pytest.raises(SystemExit) as failure:
        module.supervise(SimpleNamespace(output=output, lease=lease, timeout_seconds=7))
    assert failure.value.code == max(1, returncode)
    receipt = json.loads((output / "execution.json").read_text())
    assert receipt["status"] == "failed" and receipt["returncode"] == returncode



def test_supervisor_never_overwrites_existing_evidence(tmp_path, monkeypatch):
    """Purpose: Preserve a prior report when a command reuses its output directory."""
    module = _load_harness()
    output = tmp_path / "evidence"
    output.mkdir()
    report = output / "report.json"
    report.write_text("preserved evidence")
    popen = Mock()
    monkeypatch.setattr(module.subprocess,"Popen",popen)
    with pytest.raises(FileExistsError):
        module.supervise(SimpleNamespace(output=output,lease=tmp_path/"lease",timeout_seconds=7))
    assert report.read_text() == "preserved evidence"
    popen.assert_not_called()


def _load_harness():
    path = Path(__file__).resolve().parents[1] / "scripts/mlx_hardware_roofline.py"
    spec = importlib.util.spec_from_file_location("mlx_hardware_roofline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
