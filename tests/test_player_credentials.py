"""Private Player credentials through shipped CLI, HTTP, and Node boundaries.

Only external Tinker/Cookbook, Pi SDK, and Codex interfaces are fixtures. These
are local credential/lifecycle regressions, not live Tinker or GM model runs.
The real macOS Codex sandbox probes remain in test_sandbox_confinement.py.
"""

from __future__ import annotations

import asyncio
import builtins
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import textwrap
from types import ModuleType, SimpleNamespace
import urllib.parse
import urllib.request

import pytest

from shellsmith import cli
from shellsmith.config import DEFAULT_TINKER_BASE_MODEL, PLAYER_ENDPOINT_KEY_ENV
from shellsmith.episode import smoke
from shellsmith.episode.vat_process import PiSdkVatConfig, PiSdkVatProcess
from shellsmith.episode.vat_workspace import EphemeralVatWorkspace
from shellsmith.paths import PACKAGE_ROOT
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.policy.sampler import PlayerSamplingConfig, TinkerPlayerSampler
from shellsmith.policy.server import PlayerEndpoint, SERVED_MODEL_ID
from shellsmith.policy.tinker_setup import base_sampling_client


RUNNER = PACKAGE_ROOT / "episode/vat_runner/runner.js"
PRESERVED_ENV = "SHELLSMITH_CREDENTIAL_TEST_PRESERVED"


def _assert_environment(expected, actual=None):
    # Report names, not the caller's potentially real credential values.
    actual = os.environ if actual is None else actual
    changed = {name for name in set(expected) | set(actual)
               if expected.get(name) != actual.get(name)}
    assert not changed, f"Environment changed: {sorted(changed)}"


@pytest.fixture(params=[None, "previous-ambient-player-token"], ids=["unset", "prior-key"])
def ambient_key(request, monkeypatch):
    if request.param is None:
        monkeypatch.delenv(PLAYER_ENDPOINT_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(PLAYER_ENDPOINT_KEY_ENV, request.param)
    monkeypatch.setenv(PRESERVED_ENV, "unrelated-setting")
    return request.param


@pytest.fixture
def forbidden_external_execution(monkeypatch):
    """Trip on external I/O/imports, without replacing any Shellsmith target."""
    def forbidden(*args, **kwargs):
        pytest.fail("Credential guard must precede external network/process execution")

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"tinker", "tinker_cookbook", "torch"}:
            pytest.fail("Credential guard must precede external SDK loading")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden)


@pytest.mark.parametrize("backend", ["--host", "--docker"])
@pytest.mark.parametrize("runtime", ["claude", "pi"])
def test_tinker_cli_rejects_unconfined_gm_before_external_execution(
    tmp_path, forbidden_external_execution, backend, runtime,
):
    output = tmp_path / "must-not-exist"
    with pytest.raises((RuntimeError, SystemExit)) as error:
        cli.main(["episode", backend, "--player", "tinker", "--gm-runtime", runtime,
                  "--gm-model", "fixture-model", "--log-dir", str(output)])
    if isinstance(error.value, RuntimeError):
        assert "confined Codex GM" in str(error.value)
    else:
        assert error.value.code == 2 or "use --host" in str(error.value)
    assert not output.exists()


@pytest.mark.parametrize("host", [True, False], ids=["host", "docker"])
@pytest.mark.parametrize("runtime", ["claude", "pi"])
def test_programmatic_tinker_entry_rejects_mutated_gm_before_sdk(
    tmp_path, forbidden_external_execution, host, runtime,
):
    args = cli.build_parser().parse_args(["episode", "--player", "tinker"])
    args.host, args.gm_runtime = host, runtime
    with pytest.raises(RuntimeError, match="confined Codex GM"):
        cli._run_with_tinker_player(args, tmp_path / "must-not-exist")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("indicator", ["provider", "reserved-env", "explicit-key"])
@pytest.mark.parametrize("unsafe", ["claude", "pi", "danger-full-access"])
def test_smoke_rejects_private_player_before_preflight_or_process(
    tmp_path, monkeypatch, forbidden_external_execution, indicator, unsafe,
):
    args = smoke._parser().parse_args(["--output-dir", str(tmp_path / "must-not-exist")])
    args.player_base_url = "http://127.0.0.1:9/v1"
    args.player_api_key_env = "CREDENTIAL_GUARD_TEST_KEY"
    monkeypatch.setenv(args.player_api_key_env, "guard-fixture-key")
    explicit_key = None
    if indicator == "provider":
        args.player_provider = "shellsmith-tinker"
    elif indicator == "reserved-env":
        args.player_api_key_env = PLAYER_ENDPOINT_KEY_ENV
        monkeypatch.setenv(PLAYER_ENDPOINT_KEY_ENV, "guard-fixture-key")
    else:
        explicit_key = "in-memory-guard-fixture-key"
    if unsafe == "danger-full-access":
        args.codex_sandbox = unsafe
    else:
        args.gm_runtime = unsafe  # exercise mutated programmatic namespaces too
    expected = (f"{unsafe.title()} GM execution is disabled"
                if unsafe in {"claude", "pi"} else "confined Codex GM")
    with pytest.raises(RuntimeError, match=expected):
        asyncio.run(smoke.run_live_smoke(args, player_api_key=explicit_key))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("indicator", ["--player-provider", "--player-api-key-env"])
@pytest.mark.parametrize("unsafe_flags", [
    ["--gm-runtime", "claude", "--gm-model", "fixture-model"],
    ["--codex-sandbox", "danger-full-access"],
])
def test_smoke_cli_rejects_private_endpoint_before_preflight(
    tmp_path, forbidden_external_execution, indicator, unsafe_flags,
):
    value = "shellsmith-tinker" if indicator == "--player-provider" else PLAYER_ENDPOINT_KEY_ENV
    expected_error = SystemExit if "claude" in unsafe_flags else RuntimeError
    expected_message = "2" if "claude" in unsafe_flags else "confined Codex GM"
    with pytest.raises(expected_error, match=expected_message):
        smoke.main([indicator, value, *unsafe_flags, "--output-dir", str(tmp_path / "output")])
    assert not list(tmp_path.iterdir())


@pytest.fixture
def external_tinker_sdk(monkeypatch):
    """Let the real dependency loader/setup/sampler run with external SDK modules."""
    state = SimpleNamespace(calls=[], observe=lambda: None)

    class Prompt:
        def __init__(self, tokens):
            self.tokens = list(tokens)
            self.length = len(self.tokens)

        def to_ints(self):
            return list(self.tokens)

    class Renderer:
        def create_conversation_prefix_with_tools(self, tool_specs, system_prompt):
            assert tool_specs == [] and system_prompt == ""
            return []

        def build_generation_prompt(self, messages):
            return Prompt([ord(c) for message in messages for c in message.content])

        def get_stop_sequences(self):
            return []

        def parse_response(self, tokens):
            return (SimpleNamespace(role="assistant", content="".join(map(chr, tokens))),
                    SimpleNamespace(is_clean=True, value="stop"))

    def sample(*, prompt, num_samples, sampling_params):
        state.observe()
        state.calls.append("sample")
        assert num_samples == 1 and prompt.length > 0
        tokens = list(map(ord, "local-authenticated"))
        sequence = SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason="stop")
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[sequence]))

    class ServiceClient:
        def __init__(self):
            state.observe()
            state.calls.append("service")

        def get_server_capabilities(self):
            state.observe()
            state.calls.append("capabilities")
            return SimpleNamespace(supported_models=[SimpleNamespace(
                model_name=DEFAULT_TINKER_BASE_MODEL, max_context_length=262144,
            )])

        def create_sampling_client(self, *, base_model):
            state.observe()
            assert base_model == DEFAULT_TINKER_BASE_MODEL
            state.calls.append("sampling-client")
            return SimpleNamespace(sample=sample)

    modules = {name: ModuleType(name) for name in (
        "tinker", "tinker.types", "tinker.types.tensor_data", "torch",
        "tinker_cookbook", "tinker_cookbook.renderers", "tinker_cookbook.tokenizer_utils",
    )}
    modules["tinker"].ServiceClient = ServiceClient
    modules["tinker"].types = modules["tinker.types"]
    modules["tinker.types"].SamplingParams = SimpleNamespace
    modules["tinker.types.tensor_data"].TensorData = object
    renderers = modules["tinker_cookbook.renderers"]
    for name in ("Message", "TextPart", "ThinkingPart", "ToolCall"):
        setattr(renderers, name, SimpleNamespace)
    renderers.get_renderer = lambda *_: Renderer()
    modules["tinker_cookbook"].renderers = renderers
    modules["tinker_cookbook.tokenizer_utils"].get_tokenizer = lambda _: object()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


def _endpoint(tmp_path):
    deps = load_tinker_deps()
    sampler = TinkerPlayerSampler(
        config=PlayerSamplingConfig(base_model=DEFAULT_TINKER_BASE_MODEL), deps=deps,
        sampling_client=base_sampling_client(deps, model_name=DEFAULT_TINKER_BASE_MODEL),
    )
    return PlayerEndpoint(sampler=sampler, records_dir=tmp_path / "endpoint-records")


def _pi_sdk(tmp_path, *, failure=None):
    """An external SDK that actually authenticates to the registered HTTP server."""
    root = tmp_path / "external-pi-sdk"
    core = root / "dist/core"
    core.mkdir(parents=True)
    (root / "package.json").write_text('{"type":"module"}\n')
    (core / "auth-storage.js").write_text(
        "export class AuthStorage { static inMemory() { return {}; } }\n"
    )
    (core / "http-dispatcher.js").write_text(
        "export function configureHttpDispatcher() {}\n"
    )
    audit = tmp_path / "external-pi-evidence.jsonl"
    source = "\n".join([
        f"const auditPath = {json.dumps(str(audit))};",
        f"const failure = {json.dumps(failure)};",
        f"const keyEnv = {json.dumps(PLAYER_ENDPOINT_KEY_ENV)};",
        f"const preservedEnv = {json.dumps(PRESERVED_ENV)};",
        f"const modelId = {json.dumps(SERVED_MODEL_ID)};",
    ]) + "\n" + textwrap.dedent("""
        import { appendFileSync } from 'node:fs';
        function require(value, message) { if (!value) throw new Error(message); }
        export const SettingsManager = { inMemory: () => ({}) };
        export const SessionManager = { inMemory: () => ({}) };
        export class DefaultResourceLoader { async reload() {} }
        export class ModelRuntime {
          static async create(options) {
            require(options.allowModelNetwork === false, 'unexpected model catalog network');
            return {
              async setRuntimeApiKey(provider, key, options) {
                require(options.allowNetwork === false, 'unexpected key catalog network');
                require(key && key === process.env[keyEnv], 'missing child credential');
                require(process.env[preservedEnv] === 'unrelated-setting', 'lost inherited environment');
                require(!process.argv.some(arg => arg.includes(key)), 'credential entered argv');
                this.key = key;
                this.provider = provider;
              },
              registerProvider(provider, config) {
                require(provider === this.provider && config.apiKey === this.key,
                        'provider did not receive runtime key');
                this.config = config;
                this.model = config.models[0];
              },
              getModel() { return this.model; },
            };
          }
        }
        async function authenticate(runtime, stage) {
          const url = new URL(runtime.config.baseUrl + '/models');
          require(url.hostname === '127.0.0.1', 'credential fixture must stay on loopback');
          const response = await fetch(url, {
            headers: { Authorization: 'Bearer ' + runtime.key },
            signal: AbortSignal.timeout(3000),
          });
          require(response.status === 200, 'local endpoint rejected child credential');
          const payload = await response.json();
          require(payload.data[0].id === modelId, 'unexpected served model');
          appendFileSync(auditPath, JSON.stringify({
            stage, status: response.status, model: payload.data[0].id,
            pid: process.pid, argv: process.argv,
          }) + '\\n');
          if (failure === stage) throw new Error('injected Pi SDK ' + stage + ' failure');
        }
        export async function createAgentSession({ modelRuntime }) {
          await authenticate(modelRuntime, 'initialize');
          let listener = () => {};
          const session = {
            sessionId: 'authenticated-credential-fixture',
            messages: [],
            subscribe(callback) { listener = callback; return () => {}; },
            dispose() {},
            async abort() {},
            async prompt() {
              await authenticate(modelRuntime, 'prompt');
              const response = await fetch(modelRuntime.config.baseUrl + '/chat/completions', {
                method: 'POST',
                headers: { Authorization: 'Bearer ' + modelRuntime.key, 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: modelId, reasoning_effort: 'none',
                  messages: [{ role: 'user', content: 'Check local authentication.' }] }),
                signal: AbortSignal.timeout(3000),
              });
              require(response.status === 200, 'local endpoint rejected sampling credential');
              const payload = await response.json();
              const message = { role: 'assistant', content: [{ type: 'text', text: payload.choices[0].message.content }] };
              this.messages.push(message);
              listener({ type: 'message_end', message });
            },
          };
          return { session, extensionsResult: { errors: [] } };
        }
    """)
    for tool in ("Read", "Bash", "Edit", "Write", "Grep", "Find", "Ls"):
        source += f"export function create{tool}Tool() {{ return {{ name: '{tool.lower()}' }}; }}\n"
    entry = root / "dist/index.js"
    entry.write_text(source)
    return entry, audit


@dataclass
class _ChildObservation:
    command: tuple[str, ...]
    # Pytest failure diagnostics must not dump unrelated inherited credentials.
    environment: dict[str, str] = field(repr=False)
    cwd: Path
    process: asyncio.subprocess.Process


def _observe_processes(monkeypatch, parent_environment):
    """Observe the external spawn boundary while still executing every child."""
    original = asyncio.create_subprocess_exec
    calls = []

    async def spawn(*command, **kwargs):
        _assert_environment(parent_environment)
        environment = dict(kwargs.get("env", os.environ))
        process = await original(*command, **kwargs)
        calls.append(_ChildObservation(command=command, environment=environment,
                                       cwd=Path(kwargs["cwd"]), process=process))
        _assert_environment(parent_environment)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    return calls


def _assert_private_spawn(call, parent_environment, key):
    assert key not in " ".join(call.command)
    assert call.environment[PLAYER_ENDPOINT_KEY_ENV] == key
    changed = {name for name in set(parent_environment) | set(call.environment)
               if parent_environment.get(name) != call.environment.get(name)}
    assert changed == {PLAYER_ENDPOINT_KEY_ENV}
    assert call.environment[PRESERVED_ENV] == "unrelated-setting"
    assert call.process.returncode is not None, "trusted Node child was not reaped"
    assert not call.cwd.exists(), "ephemeral Player workspace was not cleaned"


def _assert_not_saved(root, key):
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            leaked = key.encode() in path.read_bytes()
            assert not leaked, f"Private Player credential persisted in {path}"


def _assert_endpoint_closed(base_url):
    address = urllib.parse.urlparse(base_url)
    with pytest.raises(OSError):
        with socket.create_connection((address.hostname, address.port), timeout=0.5):
            pytest.fail("Player endpoint is still serving after cleanup")


@pytest.mark.parametrize("failure", [None, "initialize", "prompt"], ids=["success", "initialize-failure", "prompt-failure"])
def test_real_node_runner_uses_private_key_without_changing_parent(
    tmp_path, monkeypatch, capsys, ambient_key, external_tinker_sdk, failure,
):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the shipped vat-runner credential regression")
    sdk, audit = _pi_sdk(tmp_path, failure=failure)
    parent_environment = dict(os.environ)
    calls = _observe_processes(monkeypatch, parent_environment)
    endpoint = _endpoint(tmp_path)
    records_path = tmp_path / "run/player_records.json"

    async def run():
        with EphemeralVatWorkspace.create(temp_parent=tmp_path / "vats") as vat:
            vat.agents_path.write_text("# Local credential boundary fixture.\n")
            config = PiSdkVatConfig(
                node_executable=Path(node), runner_script=RUNNER,
                episode_id="credential-direct", workspace=vat,
                model_provider="shellsmith-tinker", model_id=SERVED_MODEL_ID,
                model_base_url=endpoint.base_url("credential-direct"), thinking_level="off",
                api_key_env=PLAYER_ENDPOINT_KEY_ENV, api_key=endpoint.api_key,
                pi_sdk_module_path=sdk, raw_log_path=tmp_path / "run/vat.jsonl",
                enabled_tools=("read",), initialize_timeout_seconds=5,
                prompt_timeout_seconds=5,
            )
            assert config.api_key == endpoint.api_key
            assert endpoint.api_key not in repr(config)
            process = PiSdkVatProcess(config)
            try:
                assert await process.launch() == "authenticated-credential-fixture"
                _assert_environment(parent_environment)
                assert endpoint.api_key not in os.environ.values()

                async def no_tool_calls(*args):
                    pytest.fail("Credential SDK fixture must not execute tools")

                result = await process.prompt("Check local authentication.", tool_handler=no_tool_calls)
                assert result.final_text == "local-authenticated"
                _assert_environment(parent_environment)
            except RuntimeError:
                # Failure must reap the child itself, not rely on our fallback close.
                assert failure is not None
                assert calls[0].process.returncode is not None
                _assert_environment(parent_environment)
                raise
            finally:
                await process.close()

    def episode():
        with cli._recording_player_endpoint(endpoint, "credential-direct", records_path):
            _assert_environment(parent_environment)
            asyncio.run(run())

    if failure:
        with pytest.raises(RuntimeError, match=f"injected Pi SDK {failure} failure"):
            episode()
    else:
        episode()
    _assert_environment(parent_environment)
    assert os.environ.get(PLAYER_ENDPOINT_KEY_ENV) == ambient_key
    assert endpoint._server is None
    records = json.loads(records_path.read_text())
    assert len(records) == (0 if failure else 1)
    if records:
        assert records[0]["content"] == "local-authenticated"
        assert records[0]["sampled_tokens"] == list(map(ord, "local-authenticated"))
    assert len(calls) == 1
    _assert_private_spawn(calls[0], parent_environment, endpoint.api_key)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["stage"] for row in rows] == (["initialize"] if failure == "initialize" else ["initialize", "prompt"])
    assert all(row["status"] == 200 and row["model"] == SERVED_MODEL_ID for row in rows)
    assert all(row["pid"] == calls[0].process.pid for row in rows)
    protocol = [json.loads(line) for line in (tmp_path / "run/vat.jsonl").read_text().splitlines()]
    initialize = next(row["message"] for row in protocol if row["message"]["type"] == "initialize")
    assert initialize["model"]["api_key_env"] == PLAYER_ENDPOINT_KEY_ENV
    assert "api_key" not in initialize and "api_key" not in initialize["model"]
    _assert_not_saved(tmp_path, endpoint.api_key)
    captured = capsys.readouterr()
    assert endpoint.api_key not in captured.out + captured.err


CODEX_FIXTURE = r'''
import json, pathlib, sys
args = sys.argv[1:]
sys.stdin.read()
assert 'default_permissions="game_master"' in args
assert 'shell_environment_policy.inherit="core"' in args
assert not any('danger-full-access' in arg for arg in args)
root = pathlib.Path.cwd()
if 'resume' not in args:
    vat = pathlib.Path(args[args.index('--add-dir') + 1])
    (vat / 'workspace/AGENTS.md').write_text('# Work only through virtual tools.\n')
    response = root / 'world/bootstrap/response.json'
    payload = {'private_world_summary': 'Local credential fixture, not a model run.',
               'private_initial_commitments': []}
else:
    response = root / 'world/final/response.json'
    payload = {'final_reward': 0.25, 'completion_percentage': 25,
               'trainable': False, 'criteria': [], 'process_deductions': [],
               'commentary': 'Credential fixture only.', 'integrity_issues': [],
               'factual_consistency': {'passed': True, 'contradictions': []},
               'task_outcomes': {}}
response.write_text(json.dumps(payload))
pathlib.Path(args[args.index('--output-last-message') + 1]).write_text('{}\n')
print(json.dumps({'type': 'thread.started', 'thread_id': 'credential-fixture-gm'}))
'''


@pytest.mark.parametrize("failure", [None, "initialize", "prompt"], ids=["success", "initialize-failure", "prompt-failure"])
def test_real_tinker_cli_host_handoff_preflight_and_cleanup(
    tmp_path, monkeypatch, capsys, ambient_key, external_tinker_sdk, failure,
):
    if shutil.which("node") is None:
        pytest.skip("Node is required for the real CLI/Player credential path")
    sdk, audit = _pi_sdk(tmp_path, failure=failure)
    tools = tmp_path / "external-tools"
    tools.mkdir()
    codex = tools / "codex"
    codex.write_text(f"#!{sys.executable}\n" + CODEX_FIXTURE)
    codex.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    parent_environment = dict(os.environ)
    external_tinker_sdk.observe = lambda: _assert_environment(parent_environment)
    calls = _observe_processes(monkeypatch, parent_environment)
    original_urlopen = urllib.request.urlopen
    preflights = []

    def urlopen(request, *args, **kwargs):
        _assert_environment(parent_environment)
        assert request.full_url.startswith("http://127.0.0.1:")
        assert request.full_url.endswith("/models")
        response = original_urlopen(request, *args, **kwargs)
        preflights.append((request.full_url, request.get_header("Authorization"), response.status))
        return response

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    output = tmp_path / "episode output"
    argv = ["episode", "--host", "--player", "tinker", "--log-dir", str(output),
            "--pi-sdk-module-path", str(sdk), "--timeout-seconds", "10",
            "--player-prompt-timeout-seconds", "20"]
    if failure:
        with pytest.raises(RuntimeError, match=f"injected Pi SDK {failure} failure"):
            cli.main(argv)
        assert json.loads((output / "episode.json").read_text())["status"] == "failed"
    else:
        assert cli.main(argv) == 0
        summary = json.loads((output / "smoke_summary.json").read_text())
        assert summary["final_reward"] == 0.25 and summary["trainable"] is False
        assert json.loads((output / "capture_summary.json").read_text())["endpoint_records"] == 1
    assert external_tinker_sdk.calls == ["service", "capabilities", "sampling-client"] + ([] if failure else ["sample"])
    assert len(preflights) == 1 and preflights[0][2] == 200
    node_calls = [call for call in calls if str(RUNNER) in call.command]
    gm_calls = [call for call in calls if call.command[0] == str(codex)]
    assert len(node_calls) == 1
    assert len(gm_calls) == (1 if failure else 2)
    key = node_calls[0].environment[PLAYER_ENDPOINT_KEY_ENV]
    assert preflights[0][1] == f"Bearer {key}"
    assert key != ambient_key and key not in os.environ.values()
    _assert_private_spawn(node_calls[0], parent_environment, key)
    for call in gm_calls:
        _assert_environment(parent_environment, call.environment)
        assert key not in " ".join(call.command)
        assert call.process.returncode == 0
        assert not call.cwd.exists(), "private world staging was not cleaned"
    _assert_environment(parent_environment)
    assert os.environ.get(PLAYER_ENDPOINT_KEY_ENV) == ambient_key
    _assert_endpoint_closed(preflights[0][0])
    records = json.loads((output / "player_records.json").read_text())
    assert len(records) == (0 if failure else 1)
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["stage"] for row in rows] == (["initialize"] if failure == "initialize" else ["initialize", "prompt"])
    assert all(row["status"] == 200 for row in rows)
    _assert_not_saved(tmp_path, key)
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err


def test_real_tinker_cli_docker_handoff_and_failure_cleanup(
    tmp_path, monkeypatch, capsys, ambient_key, external_tinker_sdk,
):
    # Reuse the existing external git/Docker transport, not a launcher stub.
    from test_docker_launcher import _capture_engine

    monkeypatch.setenv("DOCKER_HOST", "unix:///synthetic-docker.sock")
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    parent_environment = {}
    observed = []
    forwarded = []

    def observe(command, kwargs):
        _assert_environment(parent_environment)
        child_environment = kwargs.get("env", os.environ)
        if command[:2] == ["docker", "create"]:
            key = child_environment[PLAYER_ENDPOINT_KEY_ENV]
            assert key != ambient_key and key not in os.environ.values()
            setting = next(arg for arg in command if arg.startswith("SHELLSMITH_PLAYER_BASE_URL="))
            base_url = setting.split("=", 1)[1]
            assert urllib.parse.urlparse(base_url).hostname == "host.docker.internal"
            local_url = base_url.replace("host.docker.internal", "127.0.0.1")
            smoke._preflight_player(base_url=local_url, model=SERVED_MODEL_ID,
                                    api_key_env=PLAYER_ENDPOINT_KEY_ENV, api_key=key)
            forwarded.append((local_url, key))
        else:
            assert child_environment.get(PLAYER_ENDPOINT_KEY_ENV) == ambient_key
        observed.append(command)

    calls, homes = _capture_engine(tmp_path, monkeypatch, outcome="failure", observe=observe)
    parent_environment.update(os.environ)
    external_tinker_sdk.observe = lambda: _assert_environment(parent_environment)
    output = tmp_path / "docker output"
    with pytest.raises(RuntimeError, match="Episode container exited 7"):
        cli.main(["episode", "--docker", "--image", "shellsmith:credential-fixture",
                  "--player", "tinker", "--log-dir", str(output)])
    assert external_tinker_sdk.calls == ["service", "capabilities", "sampling-client"]
    assert len(forwarded) == 1
    local_url, key = forwarded[0]
    assert calls == observed
    assert calls[-1][:3] == ["docker", "rm", "--force"]
    assert any(call[:2] == ["docker", "stop"] for call in calls)
    assert homes and all(not home.exists() for home in homes)
    assert not list(tmp_path.glob("shellsmith-codex-*"))
    assert not list(output.rglob("auth.json"))
    assert json.loads((output / "player_records.json").read_text()) == []
    _assert_environment(parent_environment)
    _assert_endpoint_closed(local_url)
    assert key not in " ".join(arg for command in calls for arg in command)
    _assert_not_saved(output, key)
    captured = capsys.readouterr()
    assert key not in captured.out + captured.err


def test_preflight_explicit_key_overrides_ambient_and_still_authenticates(
    tmp_path, ambient_key, external_tinker_sdk,
):
    parent_environment = dict(os.environ)
    with _endpoint(tmp_path) as endpoint:
        arguments = dict(base_url=endpoint.base_url("preflight"), model=SERVED_MODEL_ID,
                         api_key_env=PLAYER_ENDPOINT_KEY_ENV)
        smoke._preflight_player(**arguments, api_key=endpoint.api_key)
        with pytest.raises(RuntimeError, match="401"):
            smoke._preflight_player(**arguments, api_key="wrong-explicit-key")
        with pytest.raises(RuntimeError, match="absent|401"):
            smoke._preflight_player(**arguments)
        _assert_environment(parent_environment)
        assert endpoint.api_key not in os.environ.values()
    assert endpoint._server is None
