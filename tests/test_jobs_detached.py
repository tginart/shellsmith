"""Real stdlib subprocesses prove jobs outlive the submitting transport."""
import os
from contextlib import nullcontext
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

import pytest

from shellsmith.jobs import Endpoint, JobClient
from shellsmith.jobs import service
from shellsmith.jobs.storage import alive, owner_identity, read, write


def fixture_client(tmp_path, seconds, *, ignore_interrupt=False):
    original = Path(__file__).resolve().parents[1] / "src/shellsmith/jobs"
    root = tmp_path / "fixture-source"
    package = root / "src/shellsmith"
    shutil.copytree(original, package / "jobs", ignore=shutil.ignore_patterns("__pycache__"))
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(
        "import pathlib,sys,time,signal\n"
        + ("signal.signal(signal.SIGINT,signal.SIG_IGN)\n" if ignore_interrupt else "") +
        "root=pathlib.Path(sys.argv[-1]);root.mkdir(mode=0o700)\n"
        "(root/'started.json').write_text('{\"started\":true}')\n"
        "print('fixture trainer started',flush=True)\n"
        f"time.sleep({seconds})\n"
        "print('fixture trainer completed',flush=True)\n")
    (root / "README.md").write_text("Detached process qualification fixture\n")
    config = tmp_path / "fixture.toml"
    config.write_text('[compute]\nbackend="mlx"\n')
    client = JobClient(Endpoint(None, str(tmp_path / "owner"), sys.executable), tmp_path / "cache")
    return client, config, root


def await_status(client, job_id, allowed, timeout=15):
    until = time.monotonic() + timeout
    last = None
    while time.monotonic() < until:
        last = client.status(job_id)
        if last["status"] in allowed:
            return last
        time.sleep(.15)
    pytest.fail(f"Job did not reach {allowed}: {last}")


@pytest.mark.parametrize("identity_available", [True, False])
def test_worker_publishes_running_with_child_identity_atomically(tmp_path, monkeypatch, identity_available):
    from shellsmith.jobs import worker

    folder = tmp_path / "job"
    folder.mkdir()
    config = folder / "run.toml"
    config.write_bytes(b'[compute]\nbackend="mlx"\n')
    record = {"status": "starting", "owner_identity": {"owner": "fixture"}, "dependencies": {},
              "job_id": "fixture-job", "attempt": 1,
              "runtime": str(tmp_path / "runtime"), "source_sha256": "fixture",
              "config_sha256": worker.digest(config.read_bytes()), "python": "fixture-python",
              "worker": None, "child": None, "stop_requested": False}
    # The dispatcher died after Popen and before publishing its worker identity.
    worker_identity = {"pid": os.getpid(), "boot": "fixture", "birth": [1, 2]}
    prepare_identity = {"pid": 98765431, "boot": "fixture", "birth": [3, 4]}
    child_identity = {"pid": 98765432, "boot": "fixture", "started": [1, 2]} if identity_available else None
    clock = [1000.0]
    vat = Path(record["runtime"]) / "src/shellsmith/episode/vat_runner"
    (vat / "node_modules").mkdir(parents=True)
    (vat / "package-lock.json").write_text('{}')
    events = []
    monkeypatch.setattr(worker, "job_folder", lambda home, job_id: folder)
    monkeypatch.setattr(worker, "lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(worker, "read", lambda path: dict(record))
    monkeypatch.setattr(worker, "owner_identity", lambda: {"owner": "fixture"})
    monkeypatch.setattr(worker, "now", lambda: clock[0])
    monkeypatch.setattr(worker, "installed_dependencies", lambda: {})
    monkeypatch.setattr(worker.signal, "signal", lambda *args: None)
    monkeypatch.setattr(worker, "process_identity",
                        lambda pid: child_identity if pid == 98765432 else (
                            prepare_identity if pid == 98765431 else worker_identity))

    def persist(path, value):
        if value.get("status") == "running":
            assert child_identity is not None and value.get("child") == child_identity
        events.append(("publish", dict(value)))
        record.update(value)

    def publish(folder, **changes):
        persist(folder / "job.json", {**record, **changes})
        return dict(record)

    monkeypatch.setattr(worker, "write", persist)
    monkeypatch.setattr(worker, "update", publish)
    monkeypatch.setattr(service, "job_folder", lambda home, job_id: folder)
    monkeypatch.setattr(service, "lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(service, "read", lambda path: dict(record))
    monkeypatch.setattr(service, "write", persist)
    monkeypatch.setattr(service, "owner_identity", lambda: {"owner": "fixture"})
    monkeypatch.setattr(service, "now", lambda: clock[0])
    monkeypatch.setattr(service, "alive", lambda identity: identity in (worker_identity, prepare_identity))

    def verify_source(*args):
        assert record["worker"] == worker_identity
        # Slow preflight or a dependency lock can outlast the status age limit.
        clock[0] += 11
        assert service.status(tmp_path, "fixture-job")["status"] == "starting"

    monkeypatch.setattr(worker, "verify", verify_source)

    def spawn(*args, **kwargs):
        assert record["status"] != "running"
        events.append(("spawn", None))
        if args[0][0] == "npm":
            def finish_preparation():
                clock[0] += 11
                state = service.status(tmp_path, "fixture-job")
                assert state["status"] == "starting" and state["child"] == prepare_identity
                return 0
            return SimpleNamespace(pid=98765431, poll=finish_preparation, wait=lambda: 0)
        # A short-lived child may exit before a kernel identity can be read.
        return SimpleNamespace(pid=98765432, poll=lambda: 0, wait=lambda: 0)

    monkeypatch.setattr(worker.subprocess, "Popen", spawn)
    assert worker.run(tmp_path, "fixture-job") == 0
    assert record["status"] == "completed"
    running = [value for kind, value in events if kind == "publish" and value.get("status") == "running"]
    assert len(running) == int(identity_available)
    assert record["child"] == (child_identity if identity_available else prepare_identity)


def test_dependency_preparation_records_child_without_publishing_running(tmp_path, monkeypatch):
    from shellsmith.jobs import worker

    record = {"status": "starting", "phase": "preparing_dependencies", "stop_requested": False}
    child_identity = {"pid": 98765432, "boot": "fixture", "started": [1, 2]}
    monkeypatch.setattr(worker, "process_identity", lambda pid: child_identity)
    monkeypatch.setattr(worker, "lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(worker, "read", lambda path: dict(record))
    monkeypatch.setattr(worker, "write", lambda path, value: record.update(value))
    monkeypatch.setattr(worker, "update", lambda folder, **changes: record.update(changes))
    child = SimpleNamespace(pid=98765432, poll=lambda: 0, wait=lambda: 0)
    assert worker._wait_child(child, tmp_path, lambda: False, timeout_seconds=600) == (0, False)
    assert record == {"status": "starting", "phase": "preparing_dependencies", "stop_requested": False,
                      "child": child_identity, "updated": record["updated"]}


@pytest.mark.parametrize("unavailable", [None, {"pid": -1}])
def test_worker_self_identity_retries_without_adopting_unknown_process(monkeypatch, unavailable):
    from shellsmith.jobs import worker

    exact = {"pid": os.getpid(), "boot": "fixture", "birth": [1, 2]}
    identities = iter([unavailable, unavailable, exact])
    pauses = []
    monkeypatch.setattr(worker, "process_identity", lambda pid: next(identities))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: pauses.append(seconds))
    assert worker._self_identity() == exact
    assert pauses == [.05, .05]
    monkeypatch.setattr(worker, "process_identity", lambda pid: unavailable)
    with pytest.raises(RuntimeError, match="process birth identity"):
        worker._self_identity()


@pytest.mark.parametrize("executing", [False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_worker_child_publication_preserves_concurrent_stop(tmp_path, monkeypatch, executing, bound):
    from shellsmith.jobs import worker

    record = {"status": "starting", "stop_requested": False, "attempt": 1}
    child_identity = {"pid": 98765432}
    locked = False

    class RecordLock:
        def __enter__(self):
            nonlocal locked
            # A stop wins record.lock before the worker can publish its child.
            record.update(status="stopping", stop_requested=True)
            locked = True

        def __exit__(self, *args):
            nonlocal locked
            locked = False

    monkeypatch.setattr(worker, "lock", lambda path: RecordLock())

    def load(path):
        assert locked
        return dict(record)

    def persist(path, value):
        assert locked
        assert value["status"] == "stopping" and value["stop_requested"]
        record.update(value)

    monkeypatch.setattr(worker, "read", load)
    monkeypatch.setattr(worker, "write", persist)
    assert not worker._record_child(tmp_path, child_identity, executing=executing,
                                    binding={"attempt": 1} if bound else None)
    assert record["child"] == child_identity
    assert record["status"] == "stopping"


@pytest.mark.parametrize("stop_requested", [False, True])
def test_worker_retries_child_identity_and_honors_pending_stop(tmp_path, monkeypatch, stop_requested):
    from shellsmith.jobs import worker

    record = {"status": "starting", "phase": "launching", "stop_requested": stop_requested}
    child_identity = {"pid": 98765432}
    identities = iter([None, None, child_identity])
    polls = iter([None, None, 0])
    published = []
    signals = []
    monkeypatch.setattr(worker, "lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(worker, "process_identity", lambda pid: next(identities))
    monkeypatch.setattr(worker, "read", lambda path: dict(record))

    def persist(path, value):
        published.append(dict(value))
        record.update(value)

    monkeypatch.setattr(worker, "write", persist)
    monkeypatch.setattr(worker, "update", lambda folder, **changes: record.update(changes))
    monkeypatch.setattr(worker, "alive", lambda value: bool(value))
    monkeypatch.setattr(worker.os, "killpg", lambda pid, signum: signals.append((pid, signum)))
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    child = SimpleNamespace(pid=98765432, poll=lambda: next(polls), wait=lambda: 0)
    assert worker._wait_child(child, tmp_path, lambda: False, executing=True) == (0, stop_requested)
    assert record["child"] == child_identity
    assert record["status"] == ("stopping" if stop_requested else "running")
    assert signals == ([(child.pid, signal.SIGINT)] if stop_requested else [])
    assert all(row["child"] is not None for row in published if row["status"] == "running")


@pytest.mark.parametrize("command", ["train", "campaign"])
def test_detached_job_finishes_after_submission_process_exits(tmp_path, command):
    client, config, root = fixture_client(tmp_path, 1)
    result = client.submit(config, root, "detached-completion", command=command)
    job_id = result["job_id"]
    try:
        final = await_status(client, job_id, {"completed", "failed"})
        assert final["status"] == "completed", final
        assert final["worker"]["pid"] != os.getpid()
        assert final["child"]["pid"] != final["worker"]["pid"]
        assert final["command"][3] == command
        assert "fixture trainer completed" in client.logs(job_id)["text"]
    finally:
        client.stop(job_id, "cleanup-completed")


def test_stop_owns_only_recorded_process_and_is_idempotent(tmp_path):
    client, config, root = fixture_client(tmp_path, 60)
    result = client.submit(config, root, "detached-stop")
    job_id = result["job_id"]
    try:
        await_status(client, job_id, {"running"})
        client.stop(job_id, "stop-once")
        stopped = await_status(client, job_id, {"stopped", "failed"})
        assert stopped["status"] == "stopped", stopped
        assert client.stop(job_id, "stop-once")["attempt"] == 1
    finally:
        client.stop(job_id, "cleanup-stop")


def test_stop_reconciles_worker_terminated_before_handler_installation(tmp_path):
    client, config, root = fixture_client(tmp_path, 60)
    # Freeze the pre-handler startup window rather than relying on import speed.
    # This real subprocess keeps SIGTERM's default action and creates no trainer.
    (root / "src/shellsmith/jobs/worker.py").write_text(
        "import argparse,json,pathlib,signal,time\n"
        "parser=argparse.ArgumentParser();parser.add_argument('--home');parser.add_argument('--job-id')\n"
        "args=parser.parse_args()\n"
        "assert signal.getsignal(signal.SIGTERM)==signal.SIG_DFL\n"
        "folder=pathlib.Path(args.home)/'jobs'/args.job_id\n"
        "(folder/'before-handler.json').write_text(json.dumps({'default_sigterm':True}))\n"
        "time.sleep(60)\n")
    submitted = client.submit(config, root, "stop-before-handler")
    job_id = submitted["job_id"]
    folder = Path(client.endpoint.home) / "jobs" / job_id
    try:
        until = time.monotonic() + 10
        while not (folder / "before-handler.json").exists() and time.monotonic() < until:
            time.sleep(.05)
        assert read(folder / "before-handler.json")["default_sigterm"]
        starting = client.status(job_id)
        assert starting["status"] == "starting" and starting["child"] is None
        assert alive(starting["worker"])
        client.stop(job_id, "stop-default-worker")
        stopped = await_status(client, job_id, {"stopped", "interrupted"})
        assert stopped["status"] == "stopped", stopped
        assert stopped["stop_requested"] and not alive(starting["worker"])
        assert not (folder / "run").exists()
        assert client.stop(job_id, "stop-default-worker")["status"] == "stopped"
        assert stopped["attempt"] == 1
    finally:
        client.stop(job_id, "cleanup-default-worker")


def test_reboot_or_missing_worker_requires_explicit_resume(tmp_path, monkeypatch):
    from shellsmith.jobs.storage import private_dir, write
    home = private_dir(tmp_path / "owner")
    folder = service.job_folder(home, "recorded-job")
    write(folder / "job.json", {"job_id": "recorded-job", "status": "running", "updated": 0,
                              "owner_identity": owner_identity(),
                              "worker": {"pid": 12345, "boot": "old-boot", "started": "old-start"}})
    monkeypatch.setattr(service, "alive", lambda value: False)
    state = service.status(home, "recorded-job")
    assert state["status"] == "interrupted"
    assert "explicit resume" in state["reason"]


def test_stop_reconciles_surviving_child_after_supervisor_loss(tmp_path):
    client, config, root = fixture_client(tmp_path, 60, ignore_interrupt=True)
    result = client.submit(config, root, "detached-orphan")
    job_id = result["job_id"]
    try:
        running = await_status(client, job_id, {"running"})
        # Running means durable child admission; establish actual CLI entry and
        # its ignored-interrupt handler before testing a surviving orphan.
        marker = Path(client.endpoint.home) / "jobs" / job_id / "run/started.json"
        until = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < until:
            time.sleep(.05)
        assert marker.exists()
        os.kill(running["worker"]["pid"], signal.SIGKILL)
        until = time.monotonic() + 5
        while alive(running["worker"]) and time.monotonic() < until:
            time.sleep(.05)
        assert not alive(running["worker"])
        assert alive(running["child"])
        stopped = client.stop(job_id, "stop-orphan")
        assert stopped["status"] == "stopped"
        until = time.monotonic() + 5
        while alive(running["child"]) and time.monotonic() < until:
            time.sleep(.05)
        assert not alive(running["child"])
        assert client.stop(job_id, "stop-orphan")["status"] == "stopped"
    finally:
        client.stop(job_id, "cleanup-orphan")


def test_stop_interrupts_dependency_preparation_before_trainer_launch(tmp_path, monkeypatch):
    client, config, root = fixture_client(tmp_path, 1)
    vat = root / "src/shellsmith/episode/vat_runner"
    vat.mkdir(parents=True)
    (vat / "package-lock.json").write_text('{}')
    tools = tmp_path / "tools"
    tools.mkdir()
    npm = tools / "npm"
    npm.write_text(f'#!{sys.executable}\nimport time\nprint("preparing fixture dependencies",flush=True)\ntime.sleep(60)\n')
    npm.chmod(0o700)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    result = client.submit(config, root, "stop-preparation")
    job_id = result["job_id"]
    try:
        until = time.monotonic() + 15
        while time.monotonic() < until:
            current = client.status(job_id)
            if current.get("phase") == "preparing_dependencies" and current.get("child") is not None:
                break
            time.sleep(.1)
        assert current.get("phase") == "preparing_dependencies", current
        client.stop(job_id, "stop-preparation-once")
        final = await_status(client, job_id, {"stopped", "failed"})
        assert final["status"] == "stopped", final
        assert not (Path(client.endpoint.home) / "jobs" / job_id / "run/started.json").exists()
    finally:
        client.stop(job_id, "cleanup-preparation")


def test_worker_death_between_trainer_spawn_and_publication_cannot_start_training(tmp_path):
    client, config, root = fixture_client(tmp_path, 60)
    worker_path = root / "src/shellsmith/jobs/worker.py"
    source = worker_path.read_text()
    boundary = "            code, stopped = _wait_child(child, folder, lambda: interrupted, executing=True"
    assert boundary in source
    worker_path.write_text(source.replace(boundary,
        "            (folder / 'crash-gap.json').write_text(str(child.pid))\n"
        "            os._exit(91)\n" + boundary, 1))
    job = client.submit(config, root, "trainer-publication-crash")
    folder = Path(client.endpoint.home) / "jobs" / job["job_id"]
    trainer_pid = None
    try:
        until = time.monotonic() + 15
        while not (folder / "crash-gap.json").exists() and time.monotonic() < until:
            time.sleep(.05)
        trainer_pid = int((folder / "crash-gap.json").read_text())
        from shellsmith.jobs.storage import process_identity
        until = time.monotonic() + 5
        while process_identity(trainer_pid) is not None and time.monotonic() < until:
            time.sleep(.05)
        assert process_identity(trainer_pid) is None, "Unadmitted trainer survived supervisor death"
        assert not (folder / "run/started.json").exists()
        state = read(folder / "job.json")
        state["updated"] -= 11
        write(folder / "job.json", state)
        assert client.status(job["job_id"])["status"] == "interrupted"
        assert client.stop(job["job_id"], "stop-unadmitted-trainer")["status"] == "interrupted"
        with pytest.raises((ValueError, RuntimeError), match="checkpoint"):
            client.resume(job["job_id"], "no-checkpoint-second-trainer")
        assert client.status(job["job_id"])["attempt"] == 1
        assert not (folder / "run/started.json").exists()
    finally:
        client.stop(job["job_id"], "cleanup-unadmitted-trainer")
        if trainer_pid is not None:
            from shellsmith.jobs.storage import process_identity
            identity = process_identity(trainer_pid)
            if alive(identity):
                os.killpg(trainer_pid, signal.SIGKILL)


@pytest.mark.parametrize("change", [
    {"status": "interrupted"}, {"status": "stopped", "stop_requested": True},
    {"status": "stopping", "stop_requested": True}, {"attempt": 2},
    {"worker": {"pid": 12345}}, {"source_sha256": "changed"},
    {"config_sha256": "changed"},
])
@pytest.mark.parametrize("after_publication", [False, True])
def test_trainer_admission_rejects_late_or_stopped_attempts(tmp_path, monkeypatch, change, after_publication):
    from shellsmith.jobs import trainer_startup as startup

    folder = service.job_folder(tmp_path, "admission-job")
    config = folder / "run.toml"
    config.write_text('[compute]\nbackend="mlx"\n')
    record = {"job_id": "admission-job", "attempt": 1, "owner_identity": owner_identity(),
              "worker": {"pid": 98765}, "child": None, "status": "starting", "stop_requested": False,
              "runtime": str(Path(startup.__file__).resolve().parents[3]), "python": sys.executable,
              "source_sha256": "frozen-source", "config_sha256": startup.digest(config.read_bytes()),
              "command": [sys.executable, "-m", "shellsmith", "train", "--config", str(config),
                          "--log-dir", str(folder / "run")]}
    binding = startup.launch_binding(record)
    if not after_publication:
        record.update(change)
    write(folder / "job.json", record)
    identity = {"pid": os.getpid(), "boot": "fixture", "birth": [1, 2]}
    monkeypatch.setattr(startup, "process_identity", lambda pid: identity)
    monkeypatch.setattr(startup, "alive", lambda value: False)
    monkeypatch.setattr(startup, "verify", lambda *args: pytest.fail("Rejected attempt reached source/model initialization"))
    monkeypatch.setattr(startup.runpy, "run_module", lambda *args, **kwargs: pytest.fail("Rejected trainer initialized"))
    reader, writer = os.pipe()
    os.write(writer, b"G")
    os.close(writer)
    if after_publication:
        def arrive_before_release(*args):
            current = read(folder / "job.json")
            assert current["child"] == identity
            current.update(status="running")
            current.update(change)
            write(folder / "job.json", current)
            return [reader], [], []
        monkeypatch.setattr(startup.select, "select", arrive_before_release)
    assert startup.run(tmp_path, "admission-job", binding, reader) == 1
    final = read(folder / "job.json")
    assert all(final[key] == value for key, value in change.items())
    assert final["child"] == (identity if after_publication else None)


def test_trainer_publication_failure_closes_parent_gate_without_admission(tmp_path, monkeypatch):
    from shellsmith.jobs import worker

    reader, writer = os.pipe()
    monkeypatch.setattr(worker, "process_identity", lambda pid: {"pid": pid})
    monkeypatch.setattr(worker, "_record_child", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("publication failed")))
    child = SimpleNamespace(pid=12345)
    try:
        with pytest.raises(OSError, match="publication failed"):
            worker._wait_child(child, tmp_path, lambda: False, executing=True, start_gate=writer, binding={})
        assert os.read(reader, 1) == b""
    finally:
        os.close(reader)


@pytest.mark.parametrize("executing", [False, True])
def test_late_worker_publication_and_failure_preserve_terminal_or_new_attempt(tmp_path, monkeypatch, executing):
    from shellsmith.jobs import worker

    record = {"status": "running", "attempt": 1, "child": {"pid": 12345}, "stop_requested": False}
    binding = {"attempt": 1}
    monkeypatch.setattr(worker, "lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(worker, "read", lambda path: dict(record))
    monkeypatch.setattr(worker, "write", lambda path, value: record.update(value))
    # An unavailable ps observation cannot erase the wrapper's durable birth.
    assert not worker._record_child(tmp_path, None, executing=executing, binding=binding)
    assert record["child"] == {"pid": 12345}
    for change in ({"status": "stopped"}, {"status": "starting", "attempt": 2}):
        record.update(change)
        before = dict(record)
        assert not worker._record_child(tmp_path, {"pid": 54321}, executing=executing, binding=binding)
        worker._update_attempt(tmp_path, binding, status="failed", reason="late failure")
        assert record == before


@pytest.mark.parametrize("change", [{"attempt": 2}, {"status": "interrupted"},
                                    {"status": "stopping", "stop_requested": True},
                                    {"worker": {"pid": 12345}}])
def test_worker_initial_claim_cannot_adopt_changed_attempt_or_stop(tmp_path, monkeypatch, change):
    from shellsmith.jobs import worker

    record = {"job_id": "job", "attempt": 1, "status": "starting", "stop_requested": False,
              "owner_identity": owner_identity(), "worker": None}
    binding = {"job_id": "job", "attempt": 1}
    record.update(change)
    before = dict(record)
    monkeypatch.setattr(worker, "lock", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(worker, "read", lambda path: dict(record))
    monkeypatch.setattr(worker, "write", lambda *args: pytest.fail("Stale worker overwrote the current attempt"))
    assert worker._claim_worker(tmp_path, binding, {"pid": os.getpid()}) is None
    assert record == before
