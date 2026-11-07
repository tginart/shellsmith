"""Real local HTTP exchanges; no paid APIs, credentials, or Studio workloads."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import threading
import time
from urllib.parse import urlsplit

import pytest

from shellsmith.episode.gm_process.api_idle import ApiIdleRelay, api_idle_command, provider_origin


@contextmanager
def upstream(mode):
    stop, received = threading.Event(), []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append((self.path, dict(self.headers), self.rfile.read(int(self.headers.get('Content-Length', 0)))))
            if mode == 'headers_stall':
                stop.wait(3)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            if mode in {'stream_stall', 'stream_silent'}:
                if mode == 'stream_stall':
                    self.wfile.write(b': first heartbeat\n\n')
                    self.wfile.flush()
                stop.wait(3)
                return
            for _ in range(6 if mode in {'heartbeats', 'native_heartbeats'} else 1):
                self.wfile.write(b': keepalive\n\n')
                self.wfile.flush()
                if mode in {'heartbeats', 'native_heartbeats'}:
                    stop.wait(.06)
            if mode == 'native_heartbeats':
                item = {'type': 'message', 'id': 'fixture-message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'OK'}]}
                events = [
                    {'type': 'response.output_item.done', 'output_index': 0, 'item': item},
                    {'type': 'response.completed', 'response': {'id': 'fixture', 'status': 'completed',
                     'output': [item], 'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2}}},
                ]
                for event in events:
                    self.wfile.write(('event: '+event['type']+'\ndata: '+json.dumps(event)+'\n\n').encode())
                self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.handle_error = lambda *args: None
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', received
    finally:
        stop.set()
        server.shutdown()
        server.server_close()
        thread.join(1)


def request(relay):
    url = urlsplit(relay.base_url)
    client = http.client.HTTPConnection(url.hostname, url.port, timeout=2)
    client.request('POST', url.path + '/responses?fixture=1', b'input', {'Authorization': 'Bearer fixture-secret'})
    return client


@pytest.mark.parametrize('mode,phase,status', [('headers_stall', 'response_headers', 504), ('stream_stall', 'response_stream', 200), ('stream_silent', 'response_stream', 200)])
def test_idle_covers_first_headers_and_midstream(mode, phase, status):
    with upstream(mode) as (origin, received), ApiIdleRelay(origin, .15) as relay:
        started = time.monotonic()
        client = request(relay)
        response = client.getresponse()
        assert response.status == status
        response.read()
        client.close()
        assert .12 <= time.monotonic() - started < 1.5
        assert relay.timeouts == [{'phase': phase, 'idle_seconds': .15}]
        assert received[0][0] == '/v1/responses?fixture=1'
        assert received[0][1]['Authorization'] == 'Bearer fixture-secret'
        assert received[0][2] == b'input'
        assert 'fixture-secret' not in json.dumps(relay.timeouts)


def test_raw_keepalives_reset_inactivity_and_no_request_has_no_timer():
    with upstream('heartbeats') as (origin, _), ApiIdleRelay(origin, .15) as relay:
        # Local tool time: no API exchange exists during this wait.
        time.sleep(.25)
        started = time.monotonic()
        client = request(relay)
        assert client.getresponse().read() == b': keepalive\n\n' * 6
        client.close()
        assert time.monotonic() - started > .30
        assert relay.timeouts == []


@pytest.mark.parametrize('mode', ['headers_stall', 'stream_stall', 'stream_silent'])
def test_close_interrupts_inflight_upstream_and_downstream_sockets(mode):
    with upstream(mode) as (origin, received):
        relay = ApiIdleRelay(origin, 30)
        relay.__enter__()
        client = request(relay)
        deadline = time.monotonic() + 2
        while not received and time.monotonic() < deadline:
            time.sleep(.01)
        assert received
        if mode == 'stream_stall':
            assert client.getresponse().read1() == b': first heartbeat\n\n'
        started = time.monotonic()
        relay.__exit__()
        assert time.monotonic() - started < 1
        deadline = time.monotonic() + 1
        while relay.sockets and time.monotonic() < deadline:
            time.sleep(.01)
        assert not relay.sockets
        assert not relay.thread.is_alive()
        assert relay.timeouts == []
        client.close()


def test_private_endpoint_rejects_other_paths():
    with upstream('ok') as (origin, received), ApiIdleRelay(origin, .15) as relay:
        url = urlsplit(relay.base_url)
        client = http.client.HTTPConnection(url.hostname, url.port, timeout=1)
        client.request('POST', '/responses', b'')
        assert client.getresponse().status == 404
        assert not received
        client.close()


@pytest.mark.parametrize('invalid', [0, -1, True, float('inf'), float('nan')])
def test_invalid_idle_limit_is_rejected(invalid):
    with pytest.raises(ValueError):
        ApiIdleRelay('http://127.0.0.1', invalid)


@pytest.mark.parametrize('resume', [False, True])
def test_custom_provider_and_override_precedence_with_resume(resume):
    command = ['codex', 'exec'] + (['resume'] if resume else []) + [
        '--ignore-user-config', '-c', 'model_provider="fixture"', '-c',
        'model_providers.fixture={name="Fixture",base_url="http://127.0.0.1:1234/v1",wire_api="responses",request_max_retries=0}',
        '-c', 'model_providers.fixture.base_url="http://127.0.0.1:5678/v1"',
    ] + (['fixture-session'] if resume else []) + ['-']
    assert provider_origin(command) == ('fixture', 'http://127.0.0.1:5678/v1')
    with api_idle_command(command, idle_seconds=.15, turn_seconds=900) as (actual, relay):
        assert actual[-1] == '-'
        assert actual[:len(command)-1] == command[:-1]
        assert provider_origin(actual) == ('fixture', relay.base_url)
        assert 'model_providers.fixture.stream_idle_timeout_ms=901000' in actual
        assert 'model_providers.fixture.supports_websockets=false' in actual
        assert 'request_max_retries=0' in ''.join(actual)


def test_default_provider_reads_only_auth_mode_and_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    (tmp_path/'auth.json').write_text('{"auth_mode":"chatgpt","tokens":{"access_token":"fixture-secret"}}')
    assert provider_origin(['codex', 'exec', '--ignore-user-config', '-']) == ('openai', 'https://chatgpt.com/backend-api/codex')
    monkeypatch.setenv('OPENAI_BASE_URL', 'http://127.0.0.1:1234/v1')
    assert provider_origin(['codex', 'exec', '-'])[1] == 'http://127.0.0.1:1234/v1'
    with pytest.raises(ValueError, match='explicit custom provider'):
        provider_origin(['codex', 'exec', '-c', 'model_provider="unknown"', '-'])

@pytest.mark.parametrize('mode', ['headers_stall', 'stream_stall', 'stream_silent'])
def test_pinned_codex_adapter_reports_api_idle_before_turn_deadline(tmp_path, monkeypatch, mode):
    """Installed Codex against a credential-free loopback Responses fixture."""
    import asyncio
    from pathlib import Path
    import shutil
    from shellsmith.episode.gm_process.codex import CodexWorldConfig, CodexWorldProcess
    from shellsmith.episode.failure import ExecutionTimeoutError

    executable = shutil.which('codex')
    if executable is None:
        pytest.skip('Native Codex CLI is not installed')
    home, workspace = tmp_path/'home', tmp_path/'workspace'
    home.mkdir()
    workspace.mkdir()
    monkeypatch.setenv('CODEX_HOME', str(home))
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'SHELLSMITH_GM_AUTH_GUARD'):
        monkeypatch.delenv(key, raising=False)
    with upstream(mode) as (origin, received):
        command = [executable, 'exec', '--ignore-user-config', '--skip-git-repo-check', '--json',
                   '--model', 'gpt-6.1-sol', '-c', 'features.plugins=false',
                   '-c', 'model_provider="fixture"', '-c', 'model_providers.fixture={name="Local fixture",base_url='+json.dumps(origin)+',wire_api="responses",requires_openai_auth=false,request_max_retries=0,stream_max_retries=0,supports_websockets=false}']
        adapter = CodexWorldProcess(CodexWorldConfig(
            executable=Path(executable), model='gpt-6.1-sol', workspace_dir=workspace,
            log_dir=tmp_path/'logs', bootstrap_response_schema_path=tmp_path/'schema.json',
            timeout_seconds=5, api_idle_timeout_seconds=.15,
        ))
        started = time.monotonic()
        with pytest.raises(ExecutionTimeoutError) as error:
            asyncio.run(adapter._run_turn(command=command, prompt='Local test only; reply OK.',
                                         expected_session_id=None, resume_session_id=None,
                                         output_schema_path=None, stage='bootstrap'))
        assert error.value.stage == 'bootstrap:api_idle'
        assert error.value.timeout_seconds == .15
        assert time.monotonic() - started < 4
        assert received
        audit = json.loads((tmp_path/'logs/000000_bootstrap.api-transport.json').read_text())
        assert audit['api_idle_timeouts'] == [{'phase': 'response_headers' if mode == 'headers_stall' else 'response_stream', 'idle_seconds': .15}]
        assert audit['turn_timeout_seconds'] == 5
        # All remote/model requests are the fixture provider. No login exists.
        assert all(row[0] == '/v1/responses' for row in received)
        assert 'Authorization' not in received[0][1]


def test_user_config_and_profile_routing_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path))
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    (tmp_path/'config.toml').write_text('''model_provider="fixture"
[model_providers.fixture]
name="Local fixture"
base_url="http://127.0.0.1:1234/v1"
[profiles.other]
model_provider="other"
[model_providers.other]
name="Other fixture"
base_url="http://127.0.0.1:5678/v1"
''')
    assert provider_origin(['codex', 'exec', '-']) == ('fixture', 'http://127.0.0.1:1234/v1')
    assert provider_origin(['codex', 'exec', '--profile', 'other', '-']) == ('other', 'http://127.0.0.1:5678/v1')
    assert provider_origin(['codex', 'exec', '--ignore-user-config', '-'])[0] == 'openai'


@pytest.mark.parametrize('factory', ['train', 'smoke', 'episode', 'campaign', 'batch'])
def test_timeout_defaults_and_explicit_configuration(factory):
    from shellsmith.train.config import TrainConfig
    from shellsmith.episode.smoke import _parser as smoke_parser
    from shellsmith.cli import build_parser
    from shellsmith.campaign.rollout_campaign import _parser as campaign_parser
    from shellsmith.campaign.checkout_batch import _parser as batch_parser
    if factory == 'train':
        assert TrainConfig().gm.timeout_seconds == 900
        assert TrainConfig().gm.api_idle_timeout_seconds == 180
        return
    parser = {'smoke': smoke_parser, 'episode': build_parser, 'campaign': campaign_parser, 'batch': batch_parser}[factory]()
    initial = ['episode', '--host'] if factory == 'episode' else []
    args = parser.parse_args(initial)
    assert args.gm_api_idle_timeout_seconds == 180
    if factory != 'batch':
        assert args.timeout_seconds == 900
    args = parser.parse_args(initial + ['--gm-api-idle-timeout-seconds', '42'])
    assert args.gm_api_idle_timeout_seconds == 42


@pytest.mark.parametrize('command_auth', [False, True])
def test_native_codex_keepalives_and_fresh_relay_on_resume(tmp_path, monkeypatch, command_auth):
    import asyncio
    from pathlib import Path
    import shutil
    from shellsmith.episode.gm_process.codex import CodexWorldConfig, CodexWorldProcess
    executable = shutil.which('codex')
    if executable is None:
        pytest.skip('Native Codex CLI is not installed')
    home, workspace = tmp_path/'home', tmp_path/'workspace'
    home.mkdir()
    workspace.mkdir()
    monkeypatch.setenv('CODEX_HOME', str(home))
    for key in ('OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL', 'SHELLSMITH_GM_AUTH_GUARD'):
        monkeypatch.delenv(key, raising=False)
    with upstream('native_heartbeats') as (origin, received):
        auth = ',http_headers={"ChatGPT-Account-Id"="fixture-account"},auth={command="/usr/bin/printf",args=["fixture-auth-token"],timeout_ms=1000,refresh_interval_ms=60000}' if command_auth else ''
        provider = '{name="Local fixture",base_url='+json.dumps(origin)+',wire_api="responses",requires_openai_auth=false,request_max_retries=0,stream_max_retries=0,supports_websockets=false'+auth+'}'
        adapter = CodexWorldProcess(CodexWorldConfig(
            executable=Path(executable), model='gpt-6.1-sol', workspace_dir=workspace,
            log_dir=tmp_path/'logs', bootstrap_response_schema_path=tmp_path/'schema.json',
            timeout_seconds=5, api_idle_timeout_seconds=.15,
            extra_args=('-c', 'features.plugins=false', '-c', 'model_provider="fixture"', '-c', 'model_providers.fixture='+provider),
        ))

        async def run():
            first = await adapter.start('Local fixture only; reply OK.')
            # More than the API idle limit spent outside of an API exchange.
            await asyncio.sleep(.25)
            second = await adapter.resume(first.session_id, 'Local fixture only; reply OK again.')
            assert first.session_id == second.session_id
            assert first.final_response == second.final_response == {'text': 'OK'}
            await adapter.close()
        asyncio.run(run())
        assert len(received) == 2
        if command_auth:
            assert all(row[1].get('authorization', row[1].get('Authorization')) == 'Bearer fixture-auth-token' for row in received)
            assert all(row[1].get('chatgpt-account-id', row[1].get('ChatGPT-Account-Id')) == 'fixture-account' for row in received)
        for audit in (tmp_path/'logs').glob('*.api-transport.json'):
            assert json.loads(audit.read_text())['api_idle_timeouts'] == []
            assert 'fixture-auth-token' not in audit.read_text()
