"""Portable job transactions, frozen inputs and account-private artifact mirrors."""
import base64
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

from shellsmith.jobs import Endpoint, JobClient
from shellsmith.jobs.bundle import pack, unpack, verify
from shellsmith.jobs.lease import GpuBusyError, GpuLease
from shellsmith.jobs import service
from shellsmith.jobs.storage import digest, read, write
from shellsmith.mlx.checkpoint import publish_checkpoint


def source(tmp_path):
    root = tmp_path / "source"
    (root / "src/shellsmith/jobs").mkdir(parents=True)
    (root / "src/shellsmith/jobs/worker.py").write_text("# frozen fixture\n")
    (root / "worlds").mkdir()
    (root / "worlds/world.md").write_text("world\n")
    (root / "README.md").write_text("fixture\n")
    return root


def client_fixture(tmp_path, monkeypatch):
    home = tmp_path / "owner"
    root = source(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text('[compute]\nbackend="mlx"\nhost="studio"\n[train]\nsteps=3\n')
    endpoint = Endpoint(None, str(home), sys.executable)
    launches = []
    def launch(folder, record):
        launches.append(record["job_id"])
        record.update(status="running", updated=service.now())
        write(folder / "job.json", record)
        return record
    monkeypatch.setattr(service, "_launch", launch)
    def transport(selected, envelope):
        runtime = unpack(home, envelope) if "archive" in envelope else home / "runtime" / envelope["source_sha256"]
        return service.dispatch(home, runtime, envelope)
    return JobClient(endpoint, tmp_path / "cache", transport), root, config, launches, home


def test_dropped_submission_reply_reconciles_original_job(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    transport = client.transport
    def dropped(endpoint, envelope):
        transport(endpoint, envelope)
        raise ConnectionError("response lost")
    client.transport = dropped
    with pytest.raises(ConnectionError):
        client.submit(config, root, "stable-submit")
    client.transport = transport
    result = client.submit(config, root, "stable-submit")
    assert launches == [result["job_id"]]
    assert result["attempt"] == 1
    config.write_text('[train]\nsteps=4\n')
    with pytest.raises(ValueError, match="different configuration"):
        client.submit(config, root, "stable-submit")


def test_retries_use_frozen_source_after_checkout_changes(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    first = client.submit(config, root, "source-pin")
    (root / "src/shellsmith/jobs/worker.py").write_text("# changed later\n")
    second = client.submit(config, root, "source-pin")
    assert first["source_sha256"] == second["source_sha256"]
    assert len(launches) == 1


def test_campaign_submission_freezes_command_and_adapter_selection(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    base = client.submit(config, root, "campaign-base", command="campaign")
    assert base["command_kind"] == "campaign" and base["adapter"] is None
    with pytest.raises(ValueError, match="different command"):
        client.submit(config, root, "campaign-base")
    service.update(home / "jobs" / base["job_id"], status="completed", worker=None)
    with pytest.raises(ValueError, match="Campaign jobs do not resume"):
        client.resume(base["job_id"], "campaign-resume")
    sampler = {"path": "/studio/checkpoints/sampler", "manifest_sha256": "a" * 64,
               "policy_identity": {"model": "pinned"}, "policy_version": 2}
    monkeypatch.setattr(service, "_adapter_checkpoint", lambda config, path: dict(sampler))
    trained = client.submit(config, root, "campaign-trained", command="campaign", adapter_path=sampler["path"])
    assert trained["adapter"] == sampler
    with pytest.raises(ValueError, match="different command or adapter"):
        client.submit(config, root, "campaign-trained", command="campaign", adapter_path="/studio/other")


def test_campaign_adapter_validates_tensor_identity_and_frozen_manifest(tmp_path, monkeypatch):
    from shellsmith.mlx import identity
    monkeypatch.setattr(identity, "resolve_identity", lambda settings: {"model": "fixture"})
    config = tmp_path / "campaign.toml"
    config.write_text('[compute]\nbackend="mlx"\n')
    sampler = publish_checkpoint(tmp_path / "sampler", kind="sampler", policy_identity={"model": "fixture"},
                                 policy_version=2, writer=lambda folder: (folder / "adapters.safetensors").write_bytes(b"adapters"))
    pinned = service._adapter_checkpoint(config, sampler)
    assert pinned["policy_version"] == 2
    manifest = Path(sampler) / "manifest.json"
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest changed"):
        service._adapter_checkpoint(config, sampler, pinned["manifest_sha256"])
    (Path(sampler) / "adapters.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity failure"):
        service._adapter_checkpoint(config, sampler)


def test_hero_slate_is_frozen_relative_to_checkout(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    (root / "hero_runs").mkdir()
    slate = root / "hero_runs/qualification.json"
    slate.write_text('{"schema":"fixture"}')
    config.write_text(f'[compute]\nbackend="mlx"\n[worlds]\nselect="hero"\nhero_slate="{slate}"\n')
    submitted = client.submit(config, root, "slate-path")
    frozen = (home / "jobs" / submitted["job_id"] / "run.toml").read_text()
    assert 'hero_runs/qualification.json' in frozen and str(slate) not in frozen
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    config.write_text(f'[compute]\nbackend="mlx"\n[worlds]\nhero_slate="{outside}"\n')
    with pytest.raises(ValueError, match="frozen source checkout"):
        client.submit(config, root, "outside-slate")


def test_receipt_before_launch_recovers_failed_dispatch(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    original = service._launch
    monkeypatch.setattr(service, "_launch", lambda *args: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        client.submit(config, root, "retry-launch")
    monkeypatch.setattr(service, "_launch", original)
    result = client.submit(config, root, "retry-launch")
    assert result["status"] == "running" and launches == [result["job_id"]]


def real_launch_client(tmp_path, monkeypatch):
    launch = service._launch
    client, root, config, _, home = client_fixture(tmp_path, monkeypatch)
    clock, boot, spawned = [1000.0], ["first-boot"], []
    monkeypatch.setattr(service, "_launch", launch)
    monkeypatch.setattr(service, "now", lambda: clock[0])
    monkeypatch.setattr(service, "boot_identity", lambda: boot[0])
    monkeypatch.setattr(service, "owner_identity", lambda: {"owner": "fixture"})
    monkeypatch.setattr(service, "installed_dependencies", lambda: {})
    monkeypatch.setattr(service, "process_identity", lambda pid: {"pid": pid, "boot": boot[0]})
    monkeypatch.setattr(service, "alive", lambda identity: bool(
        identity and identity.get("pid") in spawned and identity.get("boot") == boot[0]))

    def spawn(*args, **kwargs):
        pid = 7000000 + len(spawned)
        spawned.append(pid)
        return SimpleNamespace(pid=pid)

    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    return client, root, config, home, clock, boot, spawned, spawn


@pytest.mark.parametrize("failure_stage", ["log-open", "launch-publication", "spawn"])
def test_aged_queued_submission_retries_only_its_unlaunched_request(tmp_path, monkeypatch, failure_stage):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    original_open = Path.open
    original_write = service.write

    def open_log(path, *args, **kwargs):
        if path.name == "worker.log":
            raise OSError("log open failed")
        return original_open(path, *args, **kwargs)

    if failure_stage == "log-open":
        monkeypatch.setattr(Path, "open", open_log)
    elif failure_stage == "launch-publication":
        def fail_launch_publication(path, value):
            original_write(path, value)
            if path.name == "job.json" and value["status"] == "starting":
                raise OSError("launch publication failed")
        monkeypatch.setattr(service, "write", fail_launch_publication)
    else:
        monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        client.submit(config, root, "aged-unlaunched")
    monkeypatch.setattr(Path, "open", original_open)
    monkeypatch.setattr(service, "write", original_write)
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    job_id = "job-" + digest(b"aged-unlaunched")[:24]
    queued = read(home / "jobs" / job_id / "job.json")
    assert queued["status"] == "queued" and queued["worker"] is None and queued["child"] is None
    assert not spawned and (home / "receipts/aged-unlaunched.json").is_file()
    clock[0] += 11
    assert client.status(job_id)["status"] == "queued"
    result = client.submit(config, root, "aged-unlaunched")
    assert result["status"] == "starting" and result["attempt"] == 1
    assert len(spawned) == 1
    assert client.submit(config, root, "aged-unlaunched")["attempt"] == 1
    assert len(spawned) == 1


@pytest.mark.parametrize("field", ["worker", "child", "stop_requested"])
def test_queued_receipt_cannot_override_recorded_process_or_stop(tmp_path, monkeypatch, field):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        client.submit(config, root, "queued-recorded-process")
    job_id = "job-" + digest(b"queued-recorded-process")[:24]
    value = True if field == "stop_requested" else {"pid": 6000000, "boot": boot[0]}
    service.update(home / "jobs" / job_id, **{field: value})
    clock[0] += 11
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    expected = "stopped" if field == "stop_requested" else "interrupted"
    assert client.submit(config, root, "queued-recorded-process")["status"] == expected
    assert not spawned


def test_legacy_unlaunched_queue_uses_original_boot_and_request(tmp_path, monkeypatch):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        client.submit(config, root, "queued-legacy")
    job_id = "job-" + digest(b"queued-legacy")[:24]
    path = home / "jobs" / job_id / "job.json"
    queued = read(path)
    queued.pop("queued_boot")
    queued.pop("queued_request_id")
    write(path, queued)
    clock[0] += 11
    assert client.status(job_id)["status"] == "queued"
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    assert client.submit(config, root, "queued-legacy")["attempt"] == 1
    assert len(spawned) == 1


def test_failed_queued_launch_retry_preserves_a_later_stop(tmp_path, monkeypatch):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        client.submit(config, root, "queued-then-stop")
    job_id = "job-" + digest(b"queued-then-stop")[:24]
    clock[0] += 11
    assert client.stop(job_id, "stop-unlaunched")["status"] == "stopped"
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    assert client.submit(config, root, "queued-then-stop")["status"] == "stopped"
    assert not spawned


@pytest.mark.parametrize("stop_requested", [False, True])
@pytest.mark.parametrize("child_alive", [False, True])
def test_lost_worker_status_reconciles_stop_only_after_child_exit(tmp_path, monkeypatch, stop_requested, child_alive):
    monkeypatch.setattr(service, "boot_identity", lambda: "fixture-boot")
    monkeypatch.setattr(service, "owner_identity", lambda: {"owner": "fixture"})
    monkeypatch.setattr(service, "installed_dependencies", lambda: {})
    monkeypatch.setattr(service, "now", lambda: 1000.0)
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    submitted = client.submit(config, root, "lost-worker-stop")
    child = {"pid": 6000000, "boot": "fixture-boot"}
    live = [child_alive]
    signals = []
    monkeypatch.setattr(service, "alive", lambda identity: live[0] and identity == child)
    service.update(home / "jobs" / submitted["job_id"], updated=0,
                   status="stopping" if stop_requested else "running", stop_requested=stop_requested,
                   worker={"pid": 7000000, "boot": "old-boot"}, child=child)
    expected = "stopped" if stop_requested and not child_alive else "interrupted"
    assert client.status(submitted["job_id"])["status"] == expected
    assert not signals and launches == [submitted["job_id"]]
    if stop_requested and child_alive:
        def signal_child(pid, signum):
            assert (pid, signum) == (child["pid"], signal.SIGINT)
            signals.append((pid, signum))
            live[0] = False
        monkeypatch.setattr(service.os, "killpg", signal_child)
        assert client.stop(submitted["job_id"], "stop-surviving-child")["status"] == "stopped"
        assert signals == [(child["pid"], signal.SIGINT)]


def test_queued_resume_is_bound_to_its_request_attempt_and_current_boot(tmp_path, monkeypatch):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    initial = client.submit(config, root, "queued-resume-original")
    folder = home / "jobs" / initial["job_id"]
    service.update(folder, status="interrupted", child={"pid": 6000000, "boot": boot[0]})
    full_checkpoint(folder)
    boot[0] = "second-boot"
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        client.resume(initial["job_id"], "queued-resume-new-attempt")
    queued = read(folder / "job.json")
    assert queued["attempt"] == 2 and queued["submitted_boot"] == "first-boot"
    assert queued["worker"] is None and queued["child"] is None
    clock[0] += 11
    assert client.status(initial["job_id"])["status"] == "queued"
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    monkeypatch.setattr(service, "_full_checkpoint", lambda *args: pytest.fail("A receipt retry must not repeat resume admission"))
    assert client.submit(config, root, "queued-resume-original")["status"] == "queued"
    assert len(spawned) == 1
    resumed = client.resume(initial["job_id"], "queued-resume-new-attempt")
    assert resumed["status"] == "starting" and resumed["attempt"] == 2
    assert len(spawned) == 2
    assert client.resume(initial["job_id"], "queued-resume-new-attempt")["attempt"] == 2
    assert len(spawned) == 2


@pytest.mark.parametrize("action", ["submit", "resume"])
def test_queued_receipt_retry_cannot_launch_across_reboot(tmp_path, monkeypatch, action):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    request_id = "queued-reboot-original"
    if action == "resume":
        initial = client.submit(config, root, request_id)
        folder = home / "jobs" / initial["job_id"]
        service.update(folder, status="interrupted", worker=None)
        full_checkpoint(folder)
        operation = lambda: client.resume(initial["job_id"], "queued-reboot-resume")
    else:
        operation = lambda: client.submit(config, root, request_id)
    monkeypatch.setattr(service.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("spawn failed")))
    with pytest.raises(OSError):
        operation()
    successful_before = len(spawned)
    monkeypatch.setattr(service.subprocess, "Popen", spawn)
    boot[0] = "rebooted"
    clock[0] += 11
    # Retry directly, before polling status: reconciliation itself must reject
    # an incompatible boot, including an older submit receipt for a resume.
    result = client.submit(config, root, request_id)
    assert result["status"] == "interrupted"
    assert operation()["status"] == "interrupted"
    assert len(spawned) == successful_before


@pytest.mark.parametrize("failure_stage", ["identity", "publication"])
def test_post_spawn_fault_cannot_relaunch_without_explicit_resume(tmp_path, monkeypatch, failure_stage):
    client, root, config, home, clock, boot, spawned, spawn = real_launch_client(tmp_path, monkeypatch)
    original_write = service.write
    if failure_stage == "identity":
        monkeypatch.setattr(service, "process_identity", lambda *args: (_ for _ in ()).throw(OSError("identity lookup failed")))
    else:
        def fail_started_publication(path, value):
            if path.name == "job.json" and value.get("worker"):
                raise OSError("worker identity publication failed")
            return original_write(path, value)
        monkeypatch.setattr(service, "write", fail_started_publication)
    with pytest.raises(OSError):
        client.submit(config, root, "post-spawn-fault")
    monkeypatch.setattr(service, "write", original_write)
    monkeypatch.setattr(service, "process_identity", lambda pid: {"pid": pid, "boot": boot[0]})
    assert len(spawned) == 1
    job_id = "job-" + digest(b"post-spawn-fault")[:24]
    assert read(home / "jobs" / job_id / "job.json")["status"] == "starting"
    clock[0] += 11
    assert client.submit(config, root, "post-spawn-fault")["status"] == "interrupted"
    assert len(spawned) == 1
    with pytest.raises(ValueError, match="full optimizer checkpoint"):
        client.resume(job_id, "post-spawn-explicit-resume")


@pytest.mark.parametrize("action", ["submit", "resume", "stop"])
def test_failed_receipt_publication_reconciles_exact_durable_mutation(tmp_path, monkeypatch, action):
    monkeypatch.setattr(service, "boot_identity", lambda: "fixture-boot")
    monkeypatch.setattr(service, "owner_identity", lambda: {"owner": "fixture"})
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    request_id = f"receipt-fault-{action}"
    if action == "submit":
        operation = lambda: client.submit(config, root, request_id)
        job_id = "job-" + digest(request_id.encode())[:24]
    else:
        submitted = client.submit(config, root, "receipt-fault-original")
        job_id = submitted["job_id"]
        if action == "resume":
            service.update(home / "jobs" / job_id, status="interrupted", worker=None)
            full_checkpoint(home / "jobs" / job_id)
        operation = lambda: getattr(client, action)(job_id, request_id)
    original = service.write
    failure = OSError("interrupted receipt publication")

    def fail_receipt(path, value):
        if path == home / "receipts" / f"{request_id}.json":
            raise failure
        return original(path, value)

    monkeypatch.setattr(service, "write", fail_receipt)
    with pytest.raises(OSError) as caught:
        operation()
    assert caught.value is failure
    queued = read(home / "jobs" / job_id / "job.json")
    assert queued["attempt"] == (2 if action == "resume" else 1)
    assert queued["request_transactions"][request_id]["action"] == action
    monkeypatch.setattr(service, "write", original)

    saved = client._saved(job_id)
    envelope = ({**saved["envelope"], "home": str(home)} if action == "submit" else {
        "home": str(home), "action": action, "job_id": job_id, "request_id": request_id,
        "source_sha256": saved["source_sha256"]})
    with pytest.raises(ValueError, match="different inputs"):
        service.dispatch(home, Path(queued["runtime"]), envelope | {"source_sha256": "b" * 64})
    recovered = operation()
    assert recovered["attempt"] == queued["attempt"]
    assert recovered["status"] == ("stopped" if action == "stop" else "running")
    assert launches == [job_id] * (2 if action == "resume" else 1)
    assert operation()["attempt"] == recovered["attempt"]
    assert launches == [job_id] * (2 if action == "resume" else 1)


def test_submission_receipt_recovery_preserves_a_later_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "boot_identity", lambda: "fixture-boot")
    monkeypatch.setattr(service, "owner_identity", lambda: {"owner": "fixture"})
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    original = service.write
    request_id = "receipt-fault-then-stop"

    def fail_receipt(path, value):
        if path == home / "receipts" / f"{request_id}.json":
            raise OSError("interrupted receipt publication")
        return original(path, value)

    monkeypatch.setattr(service, "write", fail_receipt)
    with pytest.raises(OSError):
        client.submit(config, root, request_id)
    monkeypatch.setattr(service, "write", original)
    job_id = "job-" + digest(request_id.encode())[:24]
    assert client.stop(job_id, "stop-before-submit-retry")["status"] == "stopped"
    assert client.submit(config, root, request_id)["status"] == "stopped"
    assert not launches


def test_source_identity_rejects_modified_and_injected_code(tmp_path):
    original = source(tmp_path)
    payload = pack(original)
    runtime = unpack(tmp_path / "owner", payload)
    verify(runtime, payload["source_sha256"])
    unexpected = runtime / "src/shellsmith/unknown.py"
    unexpected.write_text("# unknown\n")
    with pytest.raises(ValueError, match="Unexpected source"):
        verify(runtime, payload["source_sha256"])
    unexpected.unlink()
    (runtime / "src/shellsmith/jobs/worker.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="source changed"):
        verify(runtime, payload["source_sha256"])


def test_bundle_refuses_symlinks_and_excludes_credentials(tmp_path):
    root = source(tmp_path)
    (root / "src/shellsmith/.env").write_text("credential=never-copy")
    (root / "src/shellsmith/auth.json").write_text("credential=never-copy")
    (root / "src/shellsmith/node_modules").mkdir()
    (root / "src/shellsmith/node_modules/secret.json").write_text("secret")
    payload = pack(root)
    runtime = unpack(tmp_path / "owner", payload)
    assert not (runtime / "src/shellsmith/.env").exists()
    assert not (runtime / "src/shellsmith/auth.json").exists()
    assert not (runtime / "src/shellsmith/node_modules").exists()
    (root / "src/shellsmith/link.py").symlink_to(root / "README.md")
    with pytest.raises(ValueError, match="symlinks"):
        pack(root)


def test_gpu_lease_contends_and_releases(tmp_path):
    root = tmp_path / "gpu"
    with GpuLease(root):
        with pytest.raises(GpuBusyError):
            with GpuLease(root):
                pytest.fail("second GPU owner")
    with GpuLease(root):
        assert read(root / "owner.json")["identity"]["pid"] > 0
    assert read(root / "owner.json")["identity"] is None


def test_gpu_lease_acquisition_metadata_failure_releases_lock(tmp_path, monkeypatch):
    import shellsmith.jobs.lease as lease_module
    monkeypatch.setattr(lease_module, "process_identity", lambda pid: {"pid": pid})
    root = tmp_path / "gpu"
    lease = GpuLease(root)
    failure = OSError("acquisition metadata failed")
    original = lease_module.write
    monkeypatch.setattr(lease_module, "write", lambda *args: (_ for _ in ()).throw(failure))
    with pytest.raises(OSError) as caught:
        lease.__enter__()
    assert caught.value is failure
    monkeypatch.setattr(lease_module, "write", original)
    with GpuLease(root):
        pass  # Keep the failed lease alive: garbage collection must not release its lock.


@pytest.mark.parametrize("body_fails", [False, True])
def test_gpu_lease_release_metadata_failure_releases_lock_and_preserves_error(tmp_path, monkeypatch, body_fails):
    import shellsmith.jobs.lease as lease_module
    monkeypatch.setattr(lease_module, "process_identity", lambda pid: {"pid": pid})
    root = tmp_path / "gpu"
    lease = GpuLease(root)
    original = lease_module.write
    publication_error = OSError("release metadata failed")
    body_error = ValueError("training failed")
    expected = body_error if body_fails else publication_error
    with pytest.raises(type(expected)) as caught:
        with lease:
            monkeypatch.setattr(lease_module, "write", lambda *args: (_ for _ in ()).throw(publication_error))
            if body_fails:
                raise body_error
    assert caught.value is expected
    if body_fails:
        assert any("release metadata" in note for note in body_error.__notes__)
    monkeypatch.setattr(lease_module, "write", original)
    with GpuLease(root):
        pass


def test_endpoint_changes_do_not_redirect_old_job(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "endpoint-pin")
    client.endpoint = Endpoint("other-studio", str(home), sys.executable)
    with pytest.raises(ValueError, match="recorded owner"):
        client.status(result["job_id"])


def test_resume_requires_full_state_and_matching_environment(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "resume-pin")
    job_id = result["job_id"]
    folder = home / "jobs" / job_id
    service.update(folder, status="interrupted", worker=None)
    with pytest.raises(ValueError, match="full optimizer"):
        client.resume(job_id, "resume-1")
    run = folder / "run"
    run.mkdir()
    checkpoint = run / "checkpoint"
    checkpoint.mkdir()
    write(checkpoint / "manifest.json", {"complete": True})
    write(run / "trainer_state.json", {"checkpoints": [{"kind": "state", "path": str(checkpoint), "sampler_path": "adapter"}]})
    monkeypatch.setattr(service, "installed_dependencies", lambda: {"changed": "1"})
    with pytest.raises(ValueError, match="dependency identity"):
        client.resume(job_id, "resume-1")


def test_metadata_sync_is_private_and_excludes_auth_and_tensor_weights(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    (run / "episode.json").write_text('{"status":"complete"}')
    (run / "weights.safetensors").write_bytes(b"large tensor stays remote")
    (run / ".env").write_text("secret")
    (run / "auth.json").write_text("secret")
    (run / "credentials.json").write_text("secret")
    (run / "auth").mkdir()
    (run / "auth/credentials.json").write_text("secret")
    mirror = tmp_path / "mirror"
    synced = client.sync(result["job_id"], mirror)
    assert synced["file_count"] == 1
    assert (mirror / "episode.json").stat().st_mode & 0o077 == 0
    assert not (mirror / "weights.safetensors").exists()
    assert not (mirror / "auth").exists()


@pytest.mark.parametrize("sizes", [(33 * 1024 * 1024 + 7,), (12 * 1024 * 1024,) * 3])
def test_chunked_sync_exceeds_legacy_total_and_single_file_limit(tmp_path, monkeypatch, sizes):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "large-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    for index, size in enumerate(sizes):
        with (run / f"capture{index}.jsonl").open("wb") as stream:
            block = bytes([65 + index]) * (1024 * 1024)
            remaining = size
            while remaining:
                stream.write(block[:min(remaining, len(block))])
                remaining -= min(remaining, len(block))
    original = client.transport
    pages = []
    def bounded(endpoint, envelope):
        response = original(endpoint, envelope)
        if envelope["action"] == "artifacts":
            pages.append(response)
            assert sum(len(base64.b64decode(chunk["data"])) for chunk in response["chunks"]) <= 4 * 1024 * 1024
        return response
    client.transport = bounded
    mirror = tmp_path / "mirror"
    synced = client.sync(result["job_id"], mirror)
    assert synced["file_count"] == len(sizes) and synced["bytes"] == sum(sizes)
    assert synced["pages"] == len(pages) > 1
    for index in range(len(sizes)):
        assert digest((mirror / f"capture{index}.jsonl").read_bytes()) == digest((run / f"capture{index}.jsonl").read_bytes())
        assert (mirror / f"capture{index}.jsonl").stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("remote_shape", [False, True])
def test_chunked_sync_restarts_on_inventory_change_and_preserves_previous_mirror(tmp_path, monkeypatch, remote_shape):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "changing-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    path = run / "capture.jsonl"
    path.write_bytes(b"initial capture bytes")
    original = client.transport
    def shaped(endpoint, envelope):
        try:
            return original(endpoint, envelope)
        except ValueError as error:
            if remote_shape:
                raise RuntimeError(f"Job transport failed: ValueError: {error}") from error
            raise
    starts = []
    def changes_once(endpoint, envelope):
        if envelope["action"] == "artifacts" and envelope.get("cursor") is None:
            starts.append(1)
        response = shaped(endpoint, envelope)
        if envelope["action"] == "artifacts" and len(starts) == 1 and envelope.get("cursor") is None:
            path.write_bytes(b"replacement capture bytes")
        return response
    client.transport = changes_once
    mirror = tmp_path / "mirror"
    mirror.mkdir(mode=0o700)
    write(mirror / "studio-job.json", result)
    (mirror / "old.json").write_text("prior verified mirror")
    client.sync(result["job_id"], mirror, page_bytes=4)
    assert len(starts) == 2 and (mirror / "capture.jsonl").read_bytes() == path.read_bytes()
    before = {p.name: p.read_bytes() for p in mirror.iterdir()}
    def never_settles(endpoint, envelope):
        response = shaped(endpoint, envelope)
        if envelope["action"] == "artifacts" and envelope.get("cursor") is None:
            path.write_bytes(path.read_bytes() + b"+")
        return response
    client.transport = never_settles
    with pytest.raises((ValueError, RuntimeError), match="Artifacts changed during sync"):
        client.sync(result["job_id"], mirror, page_bytes=4)
    assert {p.name: p.read_bytes() for p in mirror.iterdir()} == before
    assert not list(tmp_path.glob(".mirror.sync-*"))


@pytest.mark.parametrize("damage", ["traversal", "credentials", "offset", "chunk_hash", "file_hash", "truncated", "repeat_cursor"])
def test_chunked_sync_rejects_malformed_pages_without_publishing(tmp_path, monkeypatch, damage):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "malicious-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    (run / "capture.jsonl").write_bytes(b"exact capture")
    original = client.transport
    first_cursor = None
    def corrupt(endpoint, envelope):
        nonlocal first_cursor
        response = original(endpoint, envelope)
        if envelope["action"] != "artifacts":
            return response
        chunk = response["chunks"][0]
        if damage == "traversal":
            chunk["path"] = "../escape.json"
        elif damage == "credentials":
            chunk["path"] = "auth/credentials.json"
        elif damage == "offset":
            chunk["offset"] += 1
        elif damage == "chunk_hash":
            chunk["sha256"] = "0" * 64
        elif damage == "file_hash":
            chunk["file_sha256"] = "0" * 64
        elif damage == "truncated":
            response.update(done=True, next_cursor=None)
        elif first_cursor is None:
            first_cursor = response["next_cursor"]
        else:
            response["next_cursor"] = first_cursor
        return response
    client.transport = corrupt
    mirror = tmp_path / "mirror"
    mirror.mkdir(mode=0o700)
    write(mirror / "studio-job.json", result)
    (mirror / "old.json").write_text("prior verified mirror")
    with pytest.raises(ValueError):
        client.sync(result["job_id"], mirror, page_bytes=4)
    assert {p.name for p in mirror.iterdir()} == {"old.json", "studio-job.json"}
    assert not list(tmp_path.glob(".mirror.sync-*"))
    assert not (tmp_path / "escape.json").exists()


def test_new_client_accepts_historical_one_shot_artifact_server(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "legacy-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    (run / "episode.json").write_bytes(b'{"complete":true}')
    original = client.transport
    def old_server(endpoint, envelope):
        if envelope["action"] == "artifacts":
            return service.artifacts(home / "jobs" / result["job_id"])
        return original(endpoint, envelope)
    client.transport = old_server
    mirror = tmp_path / "mirror"
    synced = client.sync(result["job_id"], mirror)
    assert synced["artifacts_version"] == 1 and synced["file_count"] == 1
    assert (mirror / "episode.json").read_bytes() == (run / "episode.json").read_bytes()


def test_sync_publish_failure_restores_previous_mirror_and_cleans_staging(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "publish-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    (run / "episode.json").write_text("new artifact")
    mirror = tmp_path / "mirror"
    mirror.mkdir(mode=0o700)
    write(mirror / "studio-job.json", result)
    (mirror / "old.json").write_text("prior verified mirror")
    rename = Path.rename
    def fail_publication(path, target):
        if path.name.startswith(".mirror.sync-"):
            raise OSError("mirror publication failed")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_publication)
    with pytest.raises(OSError, match="publication failed"):
        client.sync(result["job_id"], mirror)
    assert (mirror / "old.json").read_text() == "prior verified mirror"
    assert not list(tmp_path.glob(".mirror.sync-*"))
    assert not list(tmp_path.glob(".mirror.previous-*"))


def test_sync_preserves_managed_mirror_notes_and_refuses_foreign_directory(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "preserve-notes-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    (run / "episode.json").write_text("first artifact")
    mirror = tmp_path / "mirror"
    client.sync(result["job_id"], mirror)
    (mirror / "user-note.txt").write_text("keep my research notes")
    (run / "episode.json").write_text("new artifact")
    client.sync(result["job_id"], mirror)
    assert (mirror / "episode.json").read_text() == "new artifact"
    assert (mirror / "user-note.txt").read_text() == "keep my research notes"
    assert "user-note.txt" not in read(mirror / "studio-artifact-sync.json")["files"]
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    (foreign / "user-note.txt").write_text("unrelated content")
    with pytest.raises(ValueError, match="not a managed mirror"):
        client.sync(result["job_id"], foreign)
    assert (foreign / "user-note.txt").read_text() == "unrelated content"
    write(foreign / "studio-job.json", {"job_id": "another-job"})
    with pytest.raises(ValueError, match="not a managed mirror"):
        client.sync(result["job_id"], foreign)


def test_sync_refuses_parent_symlink_before_staging(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "parent-symlink-sync")
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        client.sync(result["job_id"], alias / "mirror")
    assert not (actual / "mirror").exists()


def test_sync_preserves_notes_written_during_transfer(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "concurrent-note-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    (run / "episode.json").write_text("artifact")
    mirror = tmp_path / "mirror"
    client.sync(result["job_id"], mirror)
    original = client.transport
    def edit_note(endpoint, envelope):
        response = original(endpoint, envelope)
        (mirror / "user-note.txt").write_text("written during transfer")
        return response
    client.transport = edit_note
    with pytest.raises(ValueError, match="Local mirror changed during sync"):
        client.sync(result["job_id"], mirror)
    assert (mirror / "user-note.txt").read_text() == "written during transfer"


def test_paginated_sync_handles_empty_files_and_bounds_file_count(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "empty-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    for index in range(300):
        (run / f"empty{index:03d}.jsonl").write_bytes(b"")
    mirror = tmp_path / "mirror"
    synced = client.sync(result["job_id"], mirror)
    assert synced["file_count"] == 300 and synced["bytes"] == 0 and synced["pages"] == 2


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_chunked_server_rejects_symlinked_metadata(tmp_path, monkeypatch, kind):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "symlink-sync")
    run = home / "jobs" / result["job_id"] / "run"
    run.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.json").write_text("secret")
    if kind == "file":
        (run / "capture.json").symlink_to(outside / "secret.json")
    else:
        (run / "nested").symlink_to(outside, target_is_directory=True)
        # Inventory must reject a directory before trusting its children.
        (run / "nested.json").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        client.sync(result["job_id"], tmp_path / "mirror")


def test_chunked_reader_refuses_parent_symlink_swap_after_inventory(tmp_path):
    run = tmp_path / "run"
    original = run / "logs"
    original.mkdir(parents=True)
    path = original / "capture.jsonl"
    path.write_text("allowed capture")
    stamp = service._artifact_stamp(path.stat())
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "capture.jsonl").write_text("outside credential content")
    original.rename(run / "old-logs")
    original.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        service._artifact_open(path, stamp, run)


def full_checkpoint(folder):
    run = folder / "run"
    run.mkdir()
    identity = {"model": "fixture", "source": "frozen"}
    def writer(kind):
        def write_tensors(staging):
            (staging / "adapters.safetensors").write_bytes(b"same adapters")
            if kind == "state":
                (staging / "optimizer.safetensors").write_bytes(b"optimizer moments")
                (staging / "rng.safetensors").write_bytes(b"rng")
        return write_tensors
    sampler = publish_checkpoint(run / "sampler", writer=writer("sampler"), kind="sampler",
                                 policy_identity=identity, policy_version=1)
    checkpoint = publish_checkpoint(run / "state", writer=writer("state"), kind="state",
                                    policy_identity=identity, policy_version=1,
                                    sampler_path=sampler, optimizer={"steps": 1}, metadata={"step": 1})
    row = {"kind": "state", "path": checkpoint, "sampler_path": sampler, "policy_identity": identity, "step": 1}
    write(run / "trainer_state.json", {"checkpoints": [row]})
    return Path(checkpoint), row


def test_full_resume_verifies_committed_checkpoint_and_matching_sampler(tmp_path, monkeypatch):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "resume-complete")
    folder = home / "jobs" / result["job_id"]
    service.update(folder, status="interrupted", worker=None)
    checkpoint, row = full_checkpoint(folder)
    assert service._full_checkpoint(folder) == row
    resumed = client.resume(result["job_id"], "resume-confirmed")
    assert resumed["attempt"] == 2 and launches == [result["job_id"], result["job_id"]]
    assert client.resume(result["job_id"], "resume-confirmed")["attempt"] == 2


@pytest.mark.parametrize("damage", ["tensor", "sampler", "row", "identity"])
def test_resume_refuses_corrupt_or_uncommitted_checkpoint(tmp_path, monkeypatch, damage):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    result = client.submit(config, root, "resume-damaged")
    folder = home / "jobs" / result["job_id"]
    service.update(folder, status="interrupted", worker=None)
    checkpoint, row = full_checkpoint(folder)
    if damage == "tensor":
        (checkpoint / "optimizer.safetensors").write_bytes(b"corrupt")
    elif damage == "sampler":
        (Path(row["sampler_path"]) / "adapters.safetensors").write_bytes(b"corrupt")
    elif damage == "row":
        row["step"] = 2
        write(folder / "run/trainer_state.json", {"checkpoints": [row]})
    else:
        row["policy_identity"]["source"] = "changed"
        write(folder / "run/trainer_state.json", {"checkpoints": [row]})
    with pytest.raises(ValueError):
        client.resume(result["job_id"], "resume-refused")
    assert len(launches) == 1


def test_unknown_destination_option_and_artifact_traversal_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        Endpoint("-oProxyCommand=bad", "home", "python")
    client = JobClient(Endpoint(None, str(tmp_path / "home"), sys.executable), tmp_path / "cache")
    write(client.cache / "safe.json", {"endpoint": asdict(client.endpoint), "source_sha256": "a" * 64})
    data = b"outside"
    client.transport = lambda *args: {"job": {}, "files": {"../escape.json": {"data": base64.b64encode(data).decode(), "sha256": digest(data)}}}
    with pytest.raises(ValueError, match="Unsafe remote"):
        client.sync("safe", tmp_path / "mirror")


@pytest.mark.parametrize("compute", ["", '[compute]\nbackend="tinker"\n', '[compute]\nbackend="typo"\n'])
@pytest.mark.parametrize("command", ["train", "campaign"])
def test_studio_submission_rejects_non_mlx_before_receipt_or_transport(tmp_path, monkeypatch, compute, command):
    client, root, config, launches, home = client_fixture(tmp_path, monkeypatch)
    config.write_text(compute + '[train]\nsteps=3\n')
    calls = []
    client.transport = lambda *args: calls.append(args)
    with pytest.raises(ValueError, match="Studio jobs require compute.backend"):
        client.submit(config, root, "wrong-backend", command=command)
    assert not calls and not launches and not home.exists()
    assert not list(client.cache.glob("job-*.json"))


def test_ssh_transport_lets_remote_shell_expand_home_relative_python(monkeypatch):
    from shellsmith.jobs import client

    calls = []
    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=b"{}", stderr=b"")
    monkeypatch.setattr(client.subprocess, "run", run)
    client.ssh_transport(Endpoint("user@studio-host", "~/state", "~/env dir/bin/python"), {})
    client.ssh_transport(Endpoint("", "~/state", "~/env/bin/python"), {})
    assert calls[0][-2] == "user@studio-host"
    assert calls[0][-1].startswith("~/'env dir/bin/python' -c ")
    assert calls[1][0] == str(Path.home() / "env/bin/python")


def test_job_command_requires_an_explicit_execution_host(tmp_path, monkeypatch):
    from shellsmith.cli import main

    monkeypatch.delenv("SHELLSMITH_STUDIO_HOST", raising=False)
    with pytest.raises(SystemExit, match="No execution host"):
        main(["job", "check", "--cache-dir", str(tmp_path)])
