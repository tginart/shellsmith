"""The bounded-run owner leaves receipts and reaps its owned descendants."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPT=Path(__file__).resolve().parents[1]/"scripts/run_bounded_studio.py"


def _process_is_running(pid):
    status = subprocess.run(["ps", "-p", str(pid), "-o", "stat="],
                            capture_output=True, text=True, timeout=5)
    assert status.returncode in (0, 1), status.stderr
    state = status.stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_watchdog_assertion_distinguishes_live_process_from_unreaped_zombie():
    child = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        assert _process_is_running(child.pid)
        child.kill()
        # Keep this child's wait status unconsumed to exercise a real zombie,
        # even on hosts whose PID 1 normally reaps orphans promptly.
        for _ in range(50):
            if not _process_is_running(child.pid):
                break
            time.sleep(.02)
        else:
            pytest.fail("Killed child remained live")
        os.kill(child.pid, 0)  # PID existence alone would give a false failure.
        state = subprocess.check_output(["ps", "-p", str(child.pid), "-o", "stat="], text=True)
        assert state.strip().startswith("Z")
    finally:
        child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("normal_exit",[False,True])
def test_watchdog_reaps_owned_descendants_and_writes_receipt(tmp_path,normal_exit):
    child=tmp_path/"child.py"
    child.write_text("import subprocess,sys,time\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"open({str(tmp_path/'descendant')!r},'w').write(str(p.pid))\n"
        + ("time.sleep(.1)\n" if normal_exit else "time.sleep(60)\n"))
    receipt=tmp_path/"receipt.json"
    run=subprocess.run([sys.executable,str(SCRIPT),"--ready-immediate","--seconds",".5",
        "--receipt",str(receipt),"--",sys.executable,str(child)],capture_output=True,timeout=10)
    result=json.loads(receipt.read_text())
    assert run.returncode==(0 if normal_exit else 1)
    assert result["stop_reason"]==(None if normal_exit else "execution_deadline")
    pid=int((tmp_path/"descendant").read_text())
    # A Linux PID 1 may retain killed descendants as zombies. Such a process
    # cannot execute; require cleanup of live work, not unrelated host reaping.
    for _ in range(50):
        if not _process_is_running(pid):
            break
        time.sleep(.02)
    else:pytest.fail("Owned descendant survived bounded-run cleanup")
    again=subprocess.run([sys.executable,str(SCRIPT),"--ready-immediate","--receipt",str(receipt),
        "--",sys.executable,"-c","pass"],capture_output=True,timeout=5)
    assert again.returncode!=0
    assert json.loads(receipt.read_text())==result


@pytest.mark.parametrize("status", [0, 7])
def test_pre_readiness_exit_is_always_startup_failure(tmp_path, status):
    receipt = tmp_path / "receipt.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--ready-file", str(tmp_path / "ready"),
        "--receipt", str(receipt), "--", sys.executable, "-c", f"raise SystemExit({status})"],
        capture_output=True, timeout=10)
    row = json.loads(receipt.read_text())
    assert result.returncode == 1 and row["returncode"] == status
    assert row["stop_reason"] == "startup_exit" and row["execution_seconds"] is None


def test_fast_exit_after_readiness_is_success(tmp_path):
    ready, receipt = tmp_path / "ready", tmp_path / "receipt.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--ready-file", str(ready),
        "--receipt", str(receipt), "--", sys.executable, "-c",
        f"from pathlib import Path;Path({str(ready)!r}).write_text('ready')"], capture_output=True, timeout=10)
    row = json.loads(receipt.read_text())
    assert result.returncode == 0 and row["stop_reason"] is None and row["execution_seconds"] is not None
