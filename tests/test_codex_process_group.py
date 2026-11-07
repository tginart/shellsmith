"""Real Codex-adapter descendant lifecycle tests on POSIX.

Only the external Codex executable is a local Python fixture. The shipped start
method, command construction, OS spawn, capture, timeout and cancellation paths
run unchanged. No process, signal, wait, PID or captured result is mocked.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from shellsmith.episode.failure import ExecutionTimeoutError
from shellsmith.episode.gm_process.codex import CodexWorldConfig, CodexWorldProcess


_FIXTURE = r'''
import json, os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(__file__).parent
settings = json.loads((root / 'settings.json').read_text())
role = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ('child', 'grandchild') else 'leader'
if role != 'leader' and settings['resist']:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
info = {'role': role, 'pid': os.getpid(), 'pgid': os.getpgrp(), 'sid': os.getsid(0)}
(root / (role + '.json')).write_text(json.dumps(info))
if role == 'grandchild':
    while True:
        (root / 'heartbeat').write_text(str(time.monotonic_ns()))
        time.sleep(.01)
if role == 'child':
    child = subprocess.Popen([sys.executable, __file__, 'grandchild'])
else:
    prompt = sys.stdin.read()
    (root / 'consumed-prompt').write_text(prompt)
    io = {'stdout': subprocess.DEVNULL, 'stderr': subprocess.DEVNULL} if settings['closed_pipes'] else {}
    child = subprocess.Popen([sys.executable, __file__, 'child'], **io)
while not (root / 'grandchild.json').exists() or not (root / 'heartbeat').exists():
    time.sleep(.01)
if role == 'leader':
    print(json.dumps({'type': 'fixture.ready', 'pid': os.getpid()}), flush=True)
    print('durable-stderr-before-interruption', file=sys.stderr, flush=True)
    (root / 'ready').touch()
    if settings['leader_exits']:
        sys.exit(0)
    def terminate(_signum, _frame):
        print(json.dumps({'type': 'fixture.terminating'}), flush=True)
        print('durable-stderr-during-termination', file=sys.stderr, flush=True)
        (root / 'term-observed').touch()
        if not settings['resist']:
            child.wait(timeout=2)
        sys.exit(0)
    signal.signal(signal.SIGTERM, terminate)
elif not settings['resist']:
    def terminate(_signum, _frame):
        child.wait(timeout=2)
        sys.exit(0)
    signal.signal(signal.SIGTERM, terminate)
while True:
    time.sleep(.02)
'''


def _running(pid: int) -> bool:
    # kill(pid, 0) includes zombies. A killed orphan awaiting init reaping is not
    # executing. Read the real kernel-visible state, never fake os.kill results.
    process = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                             capture_output=True, text=True, timeout=5)
    state = process.stdout.strip()
    return bool(state) and not state.startswith('Z')


def _pids(root):
    return [json.loads(path.read_text()) for path in
            (root / 'leader.json', root / 'child.json', root / 'grandchild.json') if path.exists()]


def _cleanup_fixture(root):
    # The intentionally broken baseline leaves descendants; clean those exact
    # fixture PIDs individually, never signal pytest's inherited process group.
    for row in reversed(_pids(root)):
        try:
            os.kill(row['pid'], signal.SIGKILL)
        except ProcessLookupError:
            pass


def _adapter(tmp_path, *, resist, closed_pipes=False, leader_exits=False, timeout=5.0):
    executable = tmp_path / 'external-codex'
    executable.write_text(f'#!{sys.executable}\n' + _FIXTURE)
    executable.chmod(0o700)
    (tmp_path / 'settings.json').write_text(json.dumps({
        'resist': resist, 'closed_pipes': closed_pipes, 'leader_exits': leader_exits,
    }))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    return CodexWorldProcess(CodexWorldConfig(
        executable=executable, model='external-fixture', workspace_dir=workspace,
        log_dir=tmp_path / 'logs', bootstrap_response_schema_path=tmp_path / 'schema.json',
        timeout_seconds=timeout,
    ))


async def _ready(root, task):
    async with asyncio.timeout(10):
        while not (root / 'ready').exists():
            if task.done():
                await task
                pytest.fail('Codex adapter ended before its real descendants were ready')
            await asyncio.sleep(.01)


@pytest.mark.parametrize('interruption', ['timeout', 'cancel', 'repeat-cancel'])
@pytest.mark.parametrize('resist,closed_pipes,leader_exits', [
    (False, False, False),
    (True, False, False),
    (True, True, False),
    (True, False, True),
], ids=['cooperative', 'resistant-pipes', 'resistant-closed-pipes', 'leader-already-exited'])
def test_codex_interrupt_stops_real_descendants_and_preserves_output(
    tmp_path, interruption, resist, closed_pipes, leader_exits,
):
    adapter = _adapter(tmp_path, resist=resist, closed_pipes=closed_pipes, leader_exits=leader_exits,
                       timeout=5.0 if interruption == 'timeout' else 30.0)
    prompt = 'actual adapter input; preserve the durable output\n'

    async def run():
        task = asyncio.create_task(adapter.start(prompt))
        try:
            await _ready(tmp_path, task)
            rows = _pids(tmp_path)
            assert {row['role'] for row in rows} == {'leader', 'child', 'grandchild'}
            assert all(_running(row['pid']) for row in rows if row['role'] != 'leader')
            if interruption != 'timeout':
                task.cancel('original cancellation')
            if interruption == 'repeat-cancel':
                await asyncio.sleep(.05)
                task.cancel('second cancellation during cleanup')
            started = time.monotonic()
            expected = ExecutionTimeoutError if interruption == 'timeout' else asyncio.CancelledError
            with pytest.raises(expected) as caught:
                async with asyncio.timeout(12):
                    await task
            assert time.monotonic() - started < 10
            if interruption == 'timeout':
                assert caught.value.component == 'gm' and caught.value.stage == 'bootstrap'
                assert caught.value.timeout_seconds == 5.0
            else:
                assert caught.value.args == ('original cancellation',)
            # Give the kernel/reaper a short bounded observation window; this is
            # not a grace period in the implementation under test.
            async with asyncio.timeout(1):
                while any(_running(row['pid']) for row in rows):
                    await asyncio.sleep(.02)
            assert not [row for row in rows if _running(row['pid'])]
            assert (tmp_path / 'consumed-prompt').read_text() == prompt
            stdout = (tmp_path / 'logs/000000_bootstrap.jsonl').read_text()
            stderr = (tmp_path / 'logs/000000_bootstrap.stderr.log').read_text()
            events = [json.loads(line)['type'] for line in stdout.splitlines()]
            assert events == (['fixture.ready'] if leader_exits else ['fixture.ready', 'fixture.terminating'])
            assert stderr == ('durable-stderr-before-interruption\n' +
                              ('' if leader_exits else 'durable-stderr-during-termination\n'))
            assert json.loads((tmp_path / 'logs/000000_bootstrap.request.json').read_text())['prompt'] == prompt
            # Spawn owns a new session/group; all descendants are in that owned
            # group, never the test runner's or an unrelated command's group.
            leader = next(row for row in rows if row['role'] == 'leader')
            assert all(row['pgid'] == row['sid'] == leader['pid'] for row in rows)
            assert leader['pgid'] != os.getpgrp()
            heartbeat = (tmp_path / 'heartbeat').read_bytes()
            await asyncio.sleep(.06)
            assert (tmp_path / 'heartbeat').read_bytes() == heartbeat
            assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task()]
        finally:
            _cleanup_fixture(tmp_path)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())


def test_successful_codex_start_and_resume_use_separate_owned_groups(tmp_path):
    executable = tmp_path / 'successful-codex'
    executable.write_text(f'#!{sys.executable}\n' + r'''
import json, os, pathlib, sys
root = pathlib.Path(__file__).parent
args = sys.argv[1:]
with (root / 'success-audit.jsonl').open('a') as audit:
    audit.write(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp(), 'sid': os.getsid(0),
                           'resume': 'resume' in args, 'prompt': sys.stdin.read()}) + '\n')
if 'resume' not in args:
    print(json.dumps({'type': 'thread.started', 'thread_id': 'actual-fixture-thread'}), flush=True)
print(json.dumps({'type': 'turn.completed'}), flush=True)
pathlib.Path(args[args.index('--output-last-message') + 1]).write_text('{"status":"ready"}')
''')
    executable.chmod(0o700)
    adapter = CodexWorldProcess(CodexWorldConfig(
        executable=executable, model='external-fixture', workspace_dir=tmp_path,
        log_dir=tmp_path / 'logs', bootstrap_response_schema_path=tmp_path / 'schema.json',
    ))

    async def run():
        boot = await adapter.start('bootstrap actual content')
        turn = await adapter.resume(boot.session_id, 'resume actual content')
        assert boot.session_id == turn.session_id == 'actual-fixture-thread'
        assert boot.final_response == turn.final_response == {'status': 'ready'}
        await adapter.close()

    asyncio.run(run())
    rows = [json.loads(line) for line in (tmp_path / 'success-audit.jsonl').read_text().splitlines()]
    assert [row['prompt'] for row in rows] == ['bootstrap actual content', 'resume actual content']
    assert [row['resume'] for row in rows] == [False, True]
    assert all(row['pid'] == row['pgid'] == row['sid'] != os.getpgrp() for row in rows)
    assert not any(_running(row['pid']) for row in rows)


def test_failed_request_publication_never_spawns_codex(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, resist=True)
    original = Path.replace
    error = OSError('request publication refused')

    def fail_request(path, target):
        if str(target).endswith('.request.json'):
            raise error
        return original(path, target)

    monkeypatch.setattr(Path, 'replace', fail_request)
    try:
        with pytest.raises(OSError) as caught:
            asyncio.run(adapter.start('must not spawn'))
        assert caught.value is error
        assert not _pids(tmp_path) and not (tmp_path / 'consumed-prompt').exists()
        assert not (tmp_path / 'logs/000000_bootstrap.jsonl').exists()
    finally:
        _cleanup_fixture(tmp_path)


@pytest.mark.parametrize('input_backpressure', [False, True])
def test_capture_fsync_error_kills_owned_group_without_becoming_provider_timeout(
    tmp_path, monkeypatch, input_backpressure,
):
    adapter = _adapter(tmp_path, resist=True)
    if input_backpressure:
        executable = adapter.config.executable
        executable.write_text(executable.read_text().replace('prompt = sys.stdin.read()',
                                                             "prompt = 'stdin deliberately unconsumed'"))
    stdout_path = tmp_path / 'logs/000000_bootstrap.jsonl'
    original = os.fsync
    error = OSError('capture stdout fsync failed')
    faults = []

    def fail_stdout(fd):
        if stdout_path.exists() and os.fstat(fd).st_ino == stdout_path.stat().st_ino and not faults:
            faults.append(True)
            raise error
        return original(fd)

    monkeypatch.setattr(os, 'fsync', fail_stdout)

    async def run():
        try:
            with pytest.raises(OSError) as caught:
                await adapter.start('x' * (4 * 1024 * 1024) if input_backpressure
                                    else 'capture fault after real descendants started')
            assert caught.value is error and faults
            async with asyncio.timeout(1):
                while any(_running(row['pid']) for row in _pids(tmp_path)):
                    await asyncio.sleep(.02)
            assert len(_pids(tmp_path)) == 3
            assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
        finally:
            _cleanup_fixture(tmp_path)
            await adapter.close()

    asyncio.run(run())


def test_cancellation_during_real_spawn_retains_handle_until_group_cleanup(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path, resist=True, timeout=30)
    executable = adapter.config.executable
    # Only the external fixture is adjusted so it can create descendants before
    # the parent has obtained the OS process handle or sent prompt input.
    executable.write_text(executable.read_text().replace('prompt = sys.stdin.read()', "prompt = 'spawn-race'"))
    real_spawn = asyncio.create_subprocess_exec

    async def run():
        spawned = asyncio.Event()
        release = asyncio.Event()

        async def delayed_handle(*args, **kwargs):
            process = await real_spawn(*args, **kwargs)
            async with asyncio.timeout(10):
                while not (tmp_path / 'ready').exists():
                    await asyncio.sleep(.01)
            spawned.set()
            await release.wait()
            return process

        monkeypatch.setattr(asyncio, 'create_subprocess_exec', delayed_handle)
        task = asyncio.create_task(adapter.start('cancel during spawn'))
        try:
            async with asyncio.timeout(10):
                await spawned.wait()
            task.cancel('original spawn cancellation')
            await asyncio.sleep(.02)
            task.cancel('second spawn cancellation')
            await asyncio.sleep(.02)
            assert not task.done(), 'Owned spawn must finish acquiring its handle before cancellation returns'
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                async with asyncio.timeout(10):
                    await task
            assert caught.value.args == ('original spawn cancellation',)
            async with asyncio.timeout(1):
                while any(_running(row['pid']) for row in _pids(tmp_path)):
                    await asyncio.sleep(.02)
            assert len(_pids(tmp_path)) == 3
            assert not [other for other in asyncio.all_tasks() if other is not asyncio.current_task()]
        finally:
            release.set()
            _cleanup_fixture(tmp_path)
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await adapter.close()

    asyncio.run(run())
