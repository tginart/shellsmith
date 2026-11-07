"""Frozen prompt bytes through the real trainer/inference HTTP -> Node -> GM path.

Only external Tinker/Cookbook/torch, Codex CLI and Pi SDK are fixtures. Prompt
sources are private copies selected via the actual identity/profile constants;
snapshotting, identity, collection, orchestration and rendering are not mocked.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import threading

import pytest

from shellsmith.episode import world_prompt_profile
from shellsmith.episode.bridge import format_player_turn_summary
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.train import identity, trainer
from test_frozen_world_inputs import _run, _setup
from test_trainer_resume import _no_sdk, external_tinker_sdk, training_sdk


def _prompts(tmp_path, monkeypatch):
    root = tmp_path / "prompt-sources"
    sources = {}
    paths = (*identity.PROMPT_PATHS, identity.PROMPTS_DIR / "WORLD_ARTIFACT_REPAIR_PROMPT.md")
    for path in dict.fromkeys((*paths, *world_prompt_profile.TAIL_ROOT.glob("*.md"))):
        target = root / path.name
        # Raw provenance must not be silently replaced by normalized text.
        target.parent.mkdir(exist_ok=True)
        contents = path.read_bytes().replace(b"\r\n", b"\n") + f"\nM5_ORIGINAL_{path.name}\n".encode()
        contents = contents.replace(b"\n", b"\r\n")
        target.write_bytes(contents)
        sources[target] = contents
    monkeypatch.setattr(identity, "PROMPT_PATHS", tuple(root / path.name for path in dict.fromkeys(paths)))
    monkeypatch.setattr(world_prompt_profile, "TAIL_ROOT", root)
    return sources


def _runtime(tmp_path, monkeypatch, *, top_up=False, worlds=2):
    config, paths, _, _ = _setup(tmp_path, monkeypatch, worlds=worlds, top_up=top_up)
    slate = Path(config.worlds.hero_slate)
    payload = json.loads(slate.read_text())
    payload["simulators"]["sol"]["model"] = "grok-fixture"
    payload["simulators"]["terra"]["model"] = "muse-spark-fixture"
    slate.write_text(json.dumps(payload))
    sources = _prompts(tmp_path, monkeypatch)
    # The explicit external Cookbook parser emits one real tool call. The
    # shipped HTTP adapter, Node virtual-tools bridge, WorldBridge and artifact
    # parser handle it unchanged, including a real same-thread turn repair.
    deps = load_tinker_deps()
    renderer_factory = deps.renderers.get_renderer

    def renderer(*args):
        result = renderer_factory(*args)
        parse = result.parse_response

        def parse_response(tokens):
            message, termination = parse(tokens)
            message.tool_calls = [{"id": "m5-call", "type": "function",
                                   "function": {"name": "bash", "arguments": '{"command":"pwd"}'}}]
            return message, termination

        result.parse_response = parse_response
        return result

    monkeypatch.setattr(deps.renderers, "get_renderer", renderer)
    sdk = Path(config.player.pi_sdk_module_path)
    source = sdk.read_text().replace("createAgentSession({ modelRuntime })",
                                     "createAgentSession({ modelRuntime, customTools })")
    source = source.replace("this.messages.push(message);", """
              const calls = payload.choices[0].message.tool_calls;
              require(calls.length === 1, 'expected one real HTTP tool call');
              message.content.push(...calls.map(call => ({ type: 'toolCall', id: call.id,
                name: call.function.name, arguments: JSON.parse(call.function.arguments) })));
              this.messages.push(message);""")
    source = source.replace("listener({ type: 'message_end', message });", """
              listener({ type: 'message_end', message });
              const call = calls[0];
              const result = await customTools.find(tool => tool.name === call.function.name)
                .execute(call.id, JSON.parse(call.function.arguments));
              listener({ type: 'tool_execution_end', toolCallId: call.id,
                toolName: call.function.name, isError: false, result });""")
    sdk.write_text(source)
    executable = tmp_path / "external-tools/codex"
    source = executable.read_text().replace("sys.stdin.read()", "prompt = sys.stdin.read()")
    needle = "episode = json.loads((root / 'episode.json').read_text())"
    source = source.replace(needle, needle +
        f"\nwith pathlib.Path({str(tmp_path / 'external-gm-prompts.jsonl')!r}).open('a') as audit:\n"
        "    audit.write(json.dumps({'episode_id': episode['episode_id'], 'world_id': episode['world_id'], "
        "'root': str(root), 'prompt': prompt, 'agents': (root / 'AGENTS.md').read_text(), "
        "'vat_workspace': str(pathlib.Path(args[args.index('--add-dir') + 1]) / 'workspace') "
        "if '--add-dir' in args else None}) + '\\n')")
    source = source.replace("else:\n    response = root / 'world/final/response.json'", r"""
elif list(root.glob('world/turns/*/request.json')) and not (root / 'world/final/request.json').exists():
    request_path = sorted(root.glob('world/turns/*/request.json'))[-1]
    request = json.loads(request_path.read_text())
    response = request_path.with_name('response.json')
    payload = {'action_id': request['action']['id'],
               'tool_response': {'content': '/workspace\\n', 'is_error': False},
               'turn_reward': reward, 'difficulty_rating': 0.5, 'completion_percentage': 50,
               'commentary': 'External frozen-prompt turn fixture.', 'reasoning_assessment': None,
               'consistency_review': {'reviewed_revision': request['consistency_ledger']['expected_revision'],
                   'observation_claims': [], 'new_commitments': [], 'conflicts': [], 'executed_cross_checks': []},
               'trainable': True, 'integrity_issues': [], 'terminate_episode': False, 'termination_reason': None}
else:
    response = root / 'world/final/response.json'""")
    # Exercise turn and final repair using real artifact parsers, not mocks.
    source = source.replace("response.write_text(json.dumps(payload))",
        "if 'resume' in args and not response.with_name('repair_01_prompt.md').exists():\n"
        "    payload = {}\nresponse.write_text(json.dumps(payload))")
    executable.write_text(source)
    return config, paths, sources


def _mutate(sources, kind="edit"):
    for path in sources:
        if kind == "delete":
            path.unlink()
        elif kind == "retarget":
            target = path.with_suffix(".replacement")
            target.write_bytes(sources[path].replace(b"ORIGINAL", b"CHANGED"))
            path.unlink()
            path.symlink_to(target.name)
        else:
            path.write_bytes(sources[path].replace(b"ORIGINAL", b"CHANGED"))


def _digest(sources, tail):
    digest = hashlib.sha256()
    by_name = {path.name: contents for path, contents in sources.items()}
    for name in [path.name for path in identity.PROMPT_PATHS] + [tail]:
        digest.update(name.encode() + b"\0" + by_name[name] + b"\0")
    return digest.hexdigest()


def _text(contents):
    return contents.decode().replace("\r\n", "\n").replace("\r", "\n")


def _expected_prompt(template, name, row, rollout):
    """Independently fill the shipped placeholders from actual input artifacts."""
    root = Path(row["root"])
    replacements = {"EPISODE_ID": row["episode_id"]}
    if name == "WORLD_BOOTSTRAP_PROMPT.md":
        replacements.update(
            WORLD_MARKDOWN=(rollout / "scenario/world_file.md").read_text(),
            VAT_WORKSPACE_PATH=row["vat_workspace"],
            SIMULATION_ROOT_PATH=str(root / "world/simulation_root"),
            BOOT_RESPONSE_PATH=str(root / "world/bootstrap/response.json"),
            BOOT_RESPONSE_SCHEMA_PATH=str(root / "schemas/world_boot_response.schema.json"))
    elif name == "WORLD_FINAL_PROMPT.md":
        replacements.update(
            FINAL_REQUEST_PATH=str(root / "world/final/request.json"),
            FINAL_RESPONSE_PATH=str(root / "world/final/response.json"),
            FINAL_RESPONSE_SCHEMA_PATH=str(root / "schemas/world_final_response.schema.json"))
    elif name == "WORLD_TURN_PROMPT.md":
        request_path, = (rollout / "world/turns").glob("*/request.json")
        request = json.loads(request_path.read_text())
        directory = root / "world/turns" / request_path.parent.name
        replacements.update(
            TURN_INDEX=str(request["step_index"]), TURN_DIRECTORY=str(directory),
            TURN_REQUEST_PATH=str(directory / "request.json"), TURN_RESPONSE_PATH=str(directory / "response.json"),
            TURN_RESPONSE_SCHEMA_PATH=str(root / "schemas/world_turn_response.schema.json"),
            SIMULATION_ROOT_PATH=str(root / "world/simulation_root"),
            CONSISTENCY_LEDGER_PATH=str(root / "world/consistency_ledger.json"),
            CONSISTENCY_LEDGER_REVISION=str(request["consistency_ledger"]["expected_revision"]),
            CONSISTENCY_LEDGER_SHA256=request["consistency_ledger"]["sha256"],
            PLAYER_TURN_SUMMARY=format_player_turn_summary(request))
    else:
        assert name == "WORLD_ARTIFACT_REPAIR_PROMPT.md"
        if "per-action world" in row["prompt"]:
            request_path, = (rollout / "world/turns").glob("*/request.json")
            directory = root / "world/turns" / request_path.parent.name
            stage, error = "per-action world", "turn response.action_id must be a non-empty string."
        else:
            directory = root / "world/final"
            stage, error = "final judgment", "final response.factual_consistency must be an object."
        replacements.update(STAGE=stage, REQUEST_PATH=str(directory / "request.json"),
                            RESPONSE_PATH=str(directory / "response.json"), VALIDATION_ERROR=error)
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    assert re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", template) is None
    return template


def _assert_frozen(config, sources, result, *, top_up=False, admitted=None):
    root = config.resolved_log_dir()
    snapshot = json.loads((root / trainer.SNAPSHOT_FILE).read_text())
    templates = {path.name: _text(content) for path, content in sources.items()}
    expected = {row["world_id"]: row for row in snapshot["world_identities"][:admitted]}
    identities = list(result["group_identities"].values())
    assert len(identities) == len(expected), "Source edits must not split a world's capture identity"
    assert len({row["gm_prompt_sha256"] for row in identities}) == 2
    for row in [*snapshot["world_identities"], *identities]:
        assert row["gm_model"] == ("grok-fixture" if row["world_id"] == "frozen_0" else "muse-spark-fixture")
        assert row["gm_reasoning_effort"] == ("low" if row["world_id"] == "frozen_0" else "high")
        tail = "GROK.md" if row["gm_model"] == "grok-fixture" else "MUSE.md"
        assert row["gm_prompt_sha256"] == _digest(sources, tail)
        assert row["gm_prompt_profile"] == ("grok_calibration_v1" if tail == "GROK.md" else "muse_difficulty_v2")
    outcomes = list(root.rglob("durable_outcome.json"))
    assert len(outcomes) == 2 * len(expected) + (2 if top_up else 0)
    rollouts = {}
    for path in outcomes:
        row = json.loads(path.read_text())["outcome"]
        rollouts[row["episode_id"]] = path.parent.parent
        assert row["error"] is None and row["accepted_count"] == 1
        assert row["group_identity_key"] == expected[row["world_id"]]["key"]
    consumed = [json.loads(line) for line in (root.parent / "external-gm-prompts.jsonl").read_text().splitlines()]
    assert len(consumed) == 5 * len(outcomes)
    for row in consumed:
        tail = "GROK.md" if expected[row["world_id"]]["gm_model"] == "grok-fixture" else "MUSE.md"
        assert row["agents"] == templates["AGENTS.md"].rstrip() + "\n\n" + templates[tail].strip() + "\n"
        assert "CHANGED" not in row["prompt"]
        names = [name for name in templates if f"M5_ORIGINAL_{name}" in row["prompt"]]
        assert len(names) == 1
        name = names[0]
        # Exact independently rendered content, not just sentinels or a digest.
        assert row["prompt"] == _expected_prompt(templates[name], name, row, rollouts[row["episode_id"]])
        requests = [json.loads(path.read_text()) for path in root.rglob("*.request.json")]
        assert any(request.get("prompt") == row["prompt"] for request in requests)
    assert result["status"] == "complete"
    if "step" in result:
        assert result["step"] >= 1 and trainer.latest_full_checkpoint(result)["step"] == result["step"]
    # Raw bytes and source provenance are archived only for admitted worlds.
    for path in root.rglob("prompt_provenance.json"):
        provenance = json.loads(path.read_text())
        for entry in provenance["files"]:
            content = sources[Path(entry["source_path"])]
            staged = path.parent / "prompts" / entry["name"]
            assert staged.read_bytes() == content
            assert not staged.is_symlink() and staged.stat().st_mode & 0o222 == 0
            assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    assert len(list(root.rglob("prompt_provenance.json"))) == len(expected)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("mutation", ["edit", "delete", "retarget"])
def test_prompt_sources_after_snapshot_do_not_change_identity_or_execution(
    tmp_path, monkeypatch, training_sdk, mode, mutation,
):
    config, _, sources = _runtime(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient
    mutations = []

    class ExternalService(base):
        def create_sampling_client(self, **kwargs):
            client = super().create_sampling_client(**kwargs)
            if not mutations:
                assert (config.resolved_log_dir() / trainer.SNAPSHOT_FILE).is_file()
                _mutate(sources, mutation)
                mutations.append(True)
            return client

    monkeypatch.setattr(deps.tinker, "ServiceClient", ExternalService)
    result = _run(config, mode, deps)
    assert mutations
    _assert_frozen(config, sources, result)


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("phase", ["top-ups", "later-worlds"])
def test_prompt_sources_changed_between_jobs_remain_bound(
    tmp_path, monkeypatch, training_sdk, mode, phase,
):
    top_up = phase == "top-ups"
    config, _, sources = _runtime(tmp_path, monkeypatch, top_up=top_up, worlds=2 if top_up else 4)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient
    samples = []
    mutations = []
    lock = threading.Lock()

    class ExternalService(base):
        def create_sampling_client(self, **kwargs):
            client = super().create_sampling_client(**kwargs)
            sample = client.sample

            def external_sample(**kw):
                with lock:
                    samples.append(True)
                    if not mutations and (top_up or len(samples) >= 8):
                        _mutate(sources)
                        mutations.append(True)
                return sample(**kw)

            client.sample = external_sample
            return client

    monkeypatch.setattr(deps.tinker, "ServiceClient", ExternalService)
    result = _run(config, mode, deps)
    assert mutations
    _assert_frozen(config, sources, result, top_up=top_up)


@pytest.mark.parametrize("mode", ["train", "inference"])
def test_prompt_binding_precedes_service_authentication(tmp_path, monkeypatch, training_sdk, mode):
    config, _, sources = _runtime(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient

    class ExternalService(base):
        def __init__(self):
            _mutate(sources, "delete")
            super().__init__()

    monkeypatch.setattr(deps.tinker, "ServiceClient", ExternalService)
    _assert_frozen(config, sources, _run(config, mode, deps))


def test_prompt_auth_failure_is_snapshot_only_and_retryable(tmp_path, monkeypatch, training_sdk):
    config, _, sources = _runtime(tmp_path, monkeypatch)
    deps = load_tinker_deps()
    base = deps.tinker.ServiceClient

    class AuthFailure(base):
        async def create_lora_training_client_async(self, **kwargs):
            assert (config.resolved_log_dir() / trainer.SNAPSHOT_FILE).is_file()
            raise RuntimeError("external authentication failed")

    monkeypatch.setattr(deps.tinker, "ServiceClient", AuthFailure)
    with pytest.raises(RuntimeError, match="external authentication failed"):
        _run(config, "train", deps)
    root = config.resolved_log_dir()
    assert {path.name for path in root.iterdir()} == {trainer.SNAPSHOT_FILE}
    before = (root / trainer.SNAPSHOT_FILE).read_bytes()
    monkeypatch.setattr(deps.tinker, "ServiceClient", base)
    _assert_frozen(config, sources, _run(config, "train", deps))
    assert (root / trainer.SNAPSHOT_FILE).read_bytes() == before


def test_prompt_staging_is_sparse_until_world_admission(tmp_path, monkeypatch, training_sdk):
    config, worlds, sources = _runtime(tmp_path, monkeypatch, worlds=4)
    config = replace(config, train=replace(config.train, steps=1))
    result = _run(config, "train", load_tinker_deps())
    _assert_frozen(config, sources, result, admitted=2)
    segment = config.resolved_log_dir() / "segments/000"
    assert all(not (segment / path.stem).exists() for path in worlds[2:])


@pytest.mark.parametrize("mode", ["train", "inference"])
@pytest.mark.parametrize("bad_input", ["oversized", "invalid-utf8", "fifo"])
def test_bad_prompt_inputs_fail_before_auth_or_output(tmp_path, monkeypatch, training_sdk, mode, bad_input):
    import os

    config, _, sources = _runtime(tmp_path, monkeypatch)
    path = next(iter(sources))
    if bad_input == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        path.write_bytes(b"x" * (world_prompt_profile.MAX_PROMPT_FILE_BYTES + 1)
                         if bad_input == "oversized" else b"\xff")
    with pytest.raises(ValueError):
        _run(config, mode, _no_sdk())
    assert not config.resolved_log_dir().exists()


def test_backup_profile_binding_is_frozen_without_enabling_unsafe_runtime(tmp_path, monkeypatch, training_sdk):
    from shellsmith.campaign.provider_outage import resolve_backup_world
    from shellsmith.train.world_selection import freeze_worlds, select_world_stream

    config, _, sources = _runtime(tmp_path, monkeypatch)
    # Exercise the actual Terra -> Sonnet lookup. Sonnet remains non-executable
    # under the independent Codex-only runtime guard; it is not remapped to Codex.
    selected = tuple(replace(world, gm_model="gpt-5.6-terra") for world in select_world_stream(config).worlds)
    frozen = freeze_worlds(selected)
    backup = resolve_backup_world(runtime="codex", model="gpt-5.6-terra", provider="codex-cli")
    settings = dict(gm_provider=backup.provider, gm_reasoning_effort="high", max_world_steps=100,
                    player_model=config.player.base_model, renderer_name=config.player.thinking_renderer,
                    max_tokens=256, temperature=1.0, top_p=1.0)
    before = frozen[0].group_identity(gm_runtime=backup.runtime, gm_model=backup.model, **settings)
    assert before.gm_prompt_profile == "sonnet_difficulty_v2"
    assert before.gm_prompt_sha256 == _digest(sources, "SONNET.md")
    _mutate(sources, "delete")
    after = frozen[0].group_identity(gm_runtime=backup.runtime, gm_model=backup.model, **settings)
    assert after == before and after.gm_runtime == "claude"
    primary = frozen[0].prompt_inputs.for_model(runtime="codex", model="gpt-5.6-terra")
    assert primary.profile.name == "canonical" and primary.sha256 != after.gm_prompt_sha256
    assert frozen[0].prompt_inputs.read_text(Path("SONNET.md")) == _text(next(
        content for path, content in sources.items() if path.name == "SONNET.md"))
