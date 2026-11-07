"""Bounded CPU fault injection: monitoring is never training authority."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from shellsmith.jobs.studio_monitor import effective_run_state, monitor_loop, spawn_logged, wait_for_training

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts/studio_micro_supervisor.py"


def database(root):
    (root / "state").mkdir()
    connection = sqlite3.connect(root / "state/journal.sqlite3")
    try:
        connection.execute("create table operations (id text,client text,kind text,status text,error text,body text,result text,progress text)")
        connection.execute("insert into operations values ('one','client','sample','succeeded',null,?,?,null)",
                           (json.dumps({"prompt_token_ids": [1], "max_tokens": 8}),
                            json.dumps({"prompt_tokens": 1, "completion_tokens": 1})))
        connection.commit()
    finally:
        connection.close()


def environment():
    return {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}


def test_failed_child_start_closes_log_descriptor(tmp_path, monkeypatch):
    captured = []
    def fail_start(*args, **kwargs):
        captured.append(kwargs["stdout"])
        raise OSError(24, "Too many open files")
    monkeypatch.setattr(subprocess, "Popen", fail_start)
    with pytest.raises(OSError):
        spawn_logged(["unused"], tmp_path / "monitor.log", cwd=tmp_path, env=environment())
    assert len(captured) == 1 and captured[0].closed


def test_fd_stress_reproduces_old_bug_and_fixed_polling_survives_without_gc(tmp_path):
    database(tmp_path)
    code = r'''
import gc,json,os,resource,sqlite3,sys
from pathlib import Path
from shellsmith.jobs.studio_monitor import OperationCollector
root=Path(sys.argv[1]);gc.disable()
soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE,(min(64,hard),hard))
failed=False
for count in range(200):
 try:
  with sqlite3.connect(root/'state/journal.sqlite3') as conn:
   conn.execute('select * from operations').fetchall()
 except OSError:failed=True;break
 except sqlite3.OperationalError:failed=True;break
assert failed,'Legacy connection context did not reproduce descriptor exhaustion'
del conn;gc.collect()
collector=OperationCollector(root)
fd_root='/dev/fd' if sys.platform=='darwin' else '/proc/self/fd'
before=len(os.listdir(fd_root));maximum=before
for count in range(2000):
 assert collector.collect()[0]['status']=='succeeded'
 maximum=max(maximum,len(os.listdir(fd_root)))
assert maximum<=before+1,(before,maximum)
print(json.dumps({'legacy_exhaustion_reproduced':True,'fixed_polls':2000,'fd_before':before,'fd_max':maximum,'gc_enabled':gc.isenabled()}))
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=environment(), capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["fixed_polls"] == 2000 and evidence["gc_enabled"] is False


def test_snapshot_write_failure_degrades_collector_without_stopping_it(tmp_path):
    database(tmp_path)
    # A genuine filesystem write failure, without mocking the collector.
    (tmp_path / "compute/micro-operations.json").mkdir(parents=True)
    monitor_loop(tmp_path, iterations=3, interval=.001)
    health = json.loads((tmp_path / "monitor-health.json").read_text())
    assert health["status"] == "degraded" and health["failure_count"] == 3
    assert "IsADirectoryError" in health["error"]


def test_monitor_health_write_failure_is_also_nonfatal(tmp_path):
    database(tmp_path)
    (tmp_path / "monitor-health.json").mkdir()
    monitor_loop(tmp_path, iterations=3, interval=.001)
    assert json.loads((tmp_path / "compute/micro-operations.json").read_text())["operations"][0]["status"] == "succeeded"


def test_missing_monitor_and_failed_failure_notification_do_not_abort_trainer(tmp_path):
    marker = tmp_path / "training-completed"
    service = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(10)"])
    trainer = subprocess.Popen([sys.executable, "-c", "import pathlib,sys,time;time.sleep(.15);pathlib.Path(sys.argv[1]).write_text('done')", str(marker)])
    def failed_notification(code):
        raise OSError(24, "Too many open files")
    try:
        assert wait_for_training(trainer, service, None, on_monitor_failure=failed_notification, interval=.01) == 0
        assert marker.read_text() == "done" and service.poll() is None
    finally:
        service.terminate(); service.wait(timeout=3)
        if trainer.poll() is None:
            trainer.kill(); trainer.wait(timeout=3)


def supervised(root, workflow, *, seconds=15):
    root.mkdir()
    now = time.time()
    (root / "instance.json").write_text(json.dumps({"python": sys.executable, "original_launch_at": now,
        "absolute_deadline_at": now + seconds, "status": "starting"}))
    (root / "workflow.py").write_text(textwrap.dedent(workflow))
    result = subprocess.run([sys.executable, str(SUPERVISOR), "--run-root", str(root)],
                            env=environment(), capture_output=True, text=True, timeout=seconds + 8)
    assert (root / "owner-limit.json").exists(), result.stderr
    return result, json.loads((root / "instance.json").read_text()), json.loads((root / "owner-limit.json").read_text())


def test_killed_monitor_cannot_kill_training_under_real_supervisor(tmp_path):
    result, state, receipt = supervised(tmp_path / "isolated", '''
        from pathlib import Path
        import json,subprocess,sys,time
        from shellsmith.jobs.studio_monitor import wait_for_training
        root=Path(__file__).parent
        state=json.loads((root/'instance.json').read_text());state['status']='running'
        (root/'instance.json').write_text(json.dumps(state))
        service=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        trainer=subprocess.Popen([sys.executable,'-c',"import pathlib,sys,time;time.sleep(1.2);pathlib.Path(sys.argv[1]).write_text('completed')",str(root/'training-completed')],start_new_session=True)
        monitor=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        time.sleep(.2);monitor.kill();monitor.wait()
        def degraded(code):state['monitoring_degraded']=True
        code=wait_for_training(trainer,service,monitor,on_monitor_failure=degraded,interval=.02)
        assert code==0 and service.poll() is None
        service.terminate();service.wait()
        state.update(status='finished',finished=time.time(),trainer_returncode=code)
        (root/'instance.json').write_text(json.dumps(state))
    ''')
    assert result.returncode == 0, result.stderr
    assert state["status"] == "finished" and state["monitoring_degraded"] is True
    assert (tmp_path / "isolated/training-completed").read_text() == "completed"
    assert receipt["owned_surviving_pids"] == []


def test_owner_crash_gets_terminal_status_without_owner_cleanup(tmp_path):
    result, state, receipt = supervised(tmp_path / "crash", '''
        from pathlib import Path
        import json,os,time
        root=Path(__file__).parent
        state=json.loads((root/'instance.json').read_text());state['status']='running'
        (root/'instance.json').write_text(json.dumps(state))
        time.sleep(.2);os._exit(7)
    ''')
    assert result.returncode != 0 and receipt["returncode"] == 7
    assert state["status"] == "failed" and state["finished"]
    assert state["terminal_status_source"] == "owner-limit.json"
    assert receipt["owned_surviving_pids"] == []


def test_deadline_still_stops_owned_work_and_records_terminal_status(tmp_path):
    result, state, receipt = supervised(tmp_path / "deadline", '''
        from pathlib import Path
        import json,signal,subprocess,sys,time
        root=Path(__file__).parent
        state=json.loads((root/'instance.json').read_text());state['status']='running'
        (root/'instance.json').write_text(json.dumps(state))
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
        time.sleep(30)
    ''', seconds=3)
    assert result.returncode != 0 and receipt["stop_reason"] == "absolute_deadline"
    assert state["status"] == "interrupted" and state["finished"]
    assert receipt["owned_surviving_pids"] == []


@pytest.mark.parametrize("returncode", [0, 1, -signal.SIGKILL])
def test_terminal_receipt_never_leaves_stale_running_owner(returncode):
    state = effective_run_state({"status": "running"}, {"finished": 100, "returncode": returncode, "owned_surviving_pids": []})
    assert state["status"] == "failed" and state["finished"] == 100
    assert state["raw_owner_status"] == "running"


def test_verified_owner_success_is_preserved():
    owner = {"status": "finished", "finished": 99, "workflow_status": "passed"}
    assert effective_run_state(owner, {"finished": 100, "returncode": 0, "owned_surviving_pids": []}) == owner


def test_current_recipes_use_medium_gm_reasoning():
    from shellsmith.config import DEFAULT_GM_REASONING_EFFORT
    from shellsmith.train.config import TrainConfig, load_train_config
    from shellsmith.episode.gm_process.codex import CodexWorldConfig
    assert DEFAULT_GM_REASONING_EFFORT == TrainConfig().gm.reasoning_effort == "medium"
    assert CodexWorldConfig.__dataclass_fields__["reasoning_effort"].default == "medium"
    for name in ("studio-micro-train.toml",):
        config = load_train_config(ROOT / "examples" / name, campaign_mode="inference" if "eval" in name else "studio_train")
        assert config.gm.reasoning_effort == "medium" and config.studio.response_tokens == 32768


def test_staging_uses_canonical_controls_and_refuses_historical_run(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage_controls", ROOT / "scripts/stage_studio_micro_controls.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    fresh = tmp_path / "fresh"
    receipt = module.stage(fresh)
    assert receipt["status"] == "staged_not_launched"
    assert (fresh / "workflow.py").read_bytes() == (ROOT / "scripts/studio_micro_workflow.py").read_bytes()
    assert module.stage(fresh) == receipt  # Identical preparation is safe.
    (fresh / "launch.json").write_text("{}")
    with pytest.raises(ValueError, match="historical"):
        module.stage(fresh)


def test_final_owner_snapshot_preserves_monitor_times_and_does_not_repeat_metrics(tmp_path, monkeypatch):
    from shellsmith.jobs.studio_monitor import OperationCollector
    import shellsmith.jobs.studio_monitor as module
    database(tmp_path)

    def update(status, result):
        connection = sqlite3.connect(tmp_path / "state/journal.sqlite3")
        try:
            connection.execute("update operations set status=?,result=? where id='one'", (status, json.dumps(result)))
            connection.commit()
        finally:
            connection.close()

    update("running", {})
    monkeypatch.setattr(module.time, "time", lambda: 100.)
    child = OperationCollector(tmp_path)
    child.collect()
    update("succeeded", {"execution_metrics": {"allocator_peak_bytes": 123}})
    monkeypatch.setattr(module.time, "time", lambda: 200.)
    child.collect()
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == 1

    monkeypatch.setattr(module.time, "time", lambda: 300.)
    owner = OperationCollector(tmp_path)
    final = owner.collect(persist=False)[0]
    assert final["first_observed_at"] == final["started_at"] == 100.
    assert final["finished_at"] == 200.
    assert final["status"] == "succeeded"
    assert final["result"]["execution_metrics"]["allocator_peak_bytes"] == 123
    owner.collect()
    assert len((tmp_path / "telemetry.jsonl").read_text().splitlines()) == 1



def _cleanup_owned_fixture(root):
    from shellsmith.jobs.storage import process_identity
    path = root / "fixture-owned.json"
    if not path.exists():
        return
    for pid in json.loads(path.read_text()).values():
        if type(pid) is int:
            identity = process_identity(pid)
            if identity and str(root) in identity["command"]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


_REPARENT_WORKFLOW = """
from pathlib import Path
import json,os,sys,time
from shellsmith.jobs.studio_monitor import spawn_logged
root=Path(__file__).parent
# The first supervisor poll precedes this spawn; the owner exits before the
# next one. Only the stable session enclosure can recover the reparented child.
time.sleep(.15)
child=spawn_logged([sys.executable,'-c','import time;time.sleep(30)',str(root)],root/'child.log',cwd=root,env=os.environ.copy())
(root/'fixture-owned.json').write_text(json.dumps({'owner':os.getpid(),'child':child.pid,'sid':os.getsid(child.pid),'pgid':os.getpgid(child.pid)}))
os._exit(7)
"""


def test_owner_crash_between_polls_preserves_child_ownership(tmp_path):
    from shellsmith.jobs.storage import process_identity
    root = tmp_path / "reparent"
    try:
        result, state, receipt = supervised(root, _REPARENT_WORKFLOW)
        owned = json.loads((root / "fixture-owned.json").read_text())
        assert owned["sid"] == owned["owner"]
        assert owned["pgid"] == owned["child"]
        assert result.returncode != 0 and state["status"] == "failed"
        assert receipt["cleanup_verified"] and receipt["owned_surviving_pids"] == []
        assert process_identity(owned["child"]) is None
    finally:
        _cleanup_owned_fixture(root)


def test_launch_receipt_write_failure_cleans_real_owner_and_children(tmp_path):
    from shellsmith.jobs.storage import process_identity
    root = tmp_path / "launch-write-failure"
    root.mkdir()
    now = time.time()
    (root / "instance.json").write_text(json.dumps({"python": sys.executable,
        "original_launch_at": now, "absolute_deadline_at": now + 15, "status": "starting"}))
    workflow = _REPARENT_WORKFLOW.replace("os._exit(7)", "time.sleep(30)")
    (root / "workflow.py").write_text(workflow)
    # Only receipt publication fails. Popen, OS process identities, signalling,
    # and cleanup run unchanged; the child is live before injecting ENOSPC.
    injection = r"""
import os,runpy,sys,time
from pathlib import Path
script,root=sys.argv[1],Path(sys.argv[2])
original=os.replace
def fail_launch(source,destination,*args,**kwargs):
    if Path(destination).name=='launch.json':
        until=time.monotonic()+3
        while not (root/'fixture-owned.json').exists() and time.monotonic()<until:time.sleep(.01)
        assert (root/'fixture-owned.json').exists()
        raise OSError(28,'injected launch receipt write failure')
    return original(source,destination,*args,**kwargs)
os.replace=fail_launch
sys.argv=[script,'--run-root',str(root)]
runpy.run_path(script,run_name='__main__')
"""
    try:
        result = subprocess.run([sys.executable, "-c", injection, str(SUPERVISOR), str(root)],
                                env=environment(), capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and "injected launch receipt" in result.stderr
        owned = json.loads((root / "fixture-owned.json").read_text())
        assert process_identity(owned["owner"]) is None
        assert process_identity(owned["child"]) is None
        receipt = json.loads((root / "owner-limit.json").read_text())
        assert receipt["cleanup_verified"] and receipt["owned_surviving_pids"] == []
        assert json.loads((root / "instance.json").read_text())["status"] == "failed"
    finally:
        _cleanup_owned_fixture(root)


def test_staging_refuses_symlinked_receipt_without_touching_victim(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage_controls", ROOT / "scripts/stage_studio_micro_controls.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "fresh"
    root.mkdir()
    victim = tmp_path / "victim"
    victim.write_text("keep this file unchanged")
    (root / "control-template.json").symlink_to(victim)
    with pytest.raises(ValueError, match="symlink"):
        module.stage(root)
    assert victim.read_text() == "keep this file unchanged"
    assert not (root / "workflow.py").exists()
