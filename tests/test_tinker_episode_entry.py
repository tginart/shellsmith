"""Real endpoint/sampler/CLI lifecycle; only the external Tinker SDK is a fixture.

These are not live Tinker model runs. They drive actual HTTP routing, native
message conversion, sampling bookkeeping, opaque ID selection, and durable
record publication through success and injected caller failure.
"""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import urllib.request

import pytest

from shellsmith import cli
from shellsmith.config import PLAYER_ENDPOINT_KEY_ENV
from shellsmith.policy.sampler import PlayerSamplingConfig, TinkerPlayerSampler
from shellsmith.policy.server import PlayerEndpoint, SERVED_MODEL_ID


class SDKPrompt:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.length = len(self.tokens)

    def to_ints(self):
        return list(self.tokens)


class SDKRenderer:
    """Small external Cookbook contract fixture, not a Shellsmith sampler fake."""

    def create_conversation_prefix_with_tools(self, tool_specs, system_prompt):
        assert tool_specs == [] and system_prompt == ""
        return []

    def build_generation_prompt(self, messages):
        text = "|".join(f"{message.role}:{message.content}" for message in messages)
        return SDKPrompt([ord(character) for character in text])

    def get_stop_sequences(self):
        return []

    def parse_response(self, tokens):
        return SimpleNamespace(role="assistant", content="".join(map(chr, tokens))), SimpleNamespace(is_clean=True, value="stop")


class SDKSamplingClient:
    def __init__(self):
        self.prompts = []

    def sample(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        self.prompts.append(prompt.to_ints())
        sequence = SimpleNamespace(tokens=[65, 66], logprobs=[-0.25, -0.5], stop_reason="stop")
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[sequence]))


def _endpoint(tmp_path):
    sdk_client = SDKSamplingClient()
    deps = SimpleNamespace(
        Message=SimpleNamespace, TextPart=SimpleNamespace, ThinkingPart=SimpleNamespace,
        ToolCall=SimpleNamespace, get_tokenizer=lambda _: object(),
        renderers=SimpleNamespace(get_renderer=lambda *_: SDKRenderer()),
        tinker=SimpleNamespace(types=SimpleNamespace(SamplingParams=SimpleNamespace)),
    )
    sampler = TinkerPlayerSampler(config=PlayerSamplingConfig(base_model="sdk-fixture"), deps=deps, sampling_client=sdk_client)
    return PlayerEndpoint(sampler=sampler, records_dir=tmp_path / "endpoint-journal"), sdk_client


def _request(url, key, payload=None):
    request = urllib.request.Request(
        url, data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


@pytest.mark.parametrize("basename", ["episode with spaces", "日本語 #% (trial)"])
@pytest.mark.parametrize("error_type", [None, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("previous_key", [None, "previous-endpoint-token"])
def test_real_endpoint_keeps_full_records_for_opaque_ids_and_caller_failures(tmp_path, monkeypatch, basename, error_type, previous_key):
    if previous_key is None:
        monkeypatch.delenv(PLAYER_ENDPOINT_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(PLAYER_ENDPOINT_KEY_ENV, previous_key)
    log_dir = tmp_path / basename
    args = cli.build_parser().parse_args(["episode", "--player", "tinker", "--log-dir", str(log_dir)])
    episode_id = cli._episode_id(args)
    assert re.fullmatch(r"episode-[0-9a-f]{32}", episode_id)
    assert cli._episode_id(args) == episode_id
    assert episode_id != basename
    endpoint, sdk_client = _endpoint(tmp_path)
    primary = error_type("injected late episode failure") if error_type else None
    expectation = pytest.raises(error_type) if error_type else nullcontext()
    with expectation as caught:
        with cli._recording_player_endpoint(endpoint, episode_id, log_dir / "player_records.json") as records:
            assert not log_dir.exists(), "do not pre-create the smoke output directory"
            assert os.environ.get(PLAYER_ENDPOINT_KEY_ENV) == previous_key
            base_url = endpoint.base_url(episode_id)
            assert _request(base_url + "/models", endpoint.api_key)["data"][0]["id"] == SERVED_MODEL_ID
            for attempt in range(2):
                payload = {"model": SERVED_MODEL_ID, "messages": [{"role": "user", "content": f"request {attempt}"}], "reasoning_effort": "none"}
                response = _request(base_url + "/chat/completions", endpoint.api_key, payload)
                assert response["choices"][0]["message"]["content"] == "AB"
                assert response["usage"]["completion_tokens"] == 2
            if primary is not None:
                raise primary
    if primary is not None:
        assert caught.value is primary
    assert endpoint._server is None
    assert os.environ.get(PLAYER_ENDPOINT_KEY_ENV) == previous_key
    saved = json.loads((log_dir / "player_records.json").read_text())
    assert len(saved) == len(records) == 2
    assert len({record["request_id"] for record in saved}) == 2
    assert [record["prompt_token_ids"] for record in saved] == sdk_client.prompts
    assert sdk_client.prompts == [[ord(c) for c in f"user:request {attempt}"] for attempt in range(2)]
    for record in saved:
        assert record["session_id"] == episode_id
        assert record["prompt_tokens"] == len(record["prompt_token_ids"])
        assert record["sampled_tokens"] == [65, 66]
        assert record["sampled_logprobs"] == [-0.25, -0.5]
        assert record["content"] == "AB"
        assert record["forced_close_token_span"] is None
        assert record["protocol_row"]["request_id"] == record["request_id"]
    assert endpoint.api_key not in (log_dir / "player_records.json").read_text()
    assert endpoint.take_records(episode_id) == []


def test_record_publication_error_does_not_replace_original_episode_failure(tmp_path, monkeypatch):
    monkeypatch.delenv(PLAYER_ENDPOINT_KEY_ENV, raising=False)
    endpoint, _client = _endpoint(tmp_path)
    destination = tmp_path / "records.json"
    destination.mkdir()  # a genuine atomic-publication failure, not a mocked writer
    primary = RuntimeError("original GM failure")
    with pytest.raises(RuntimeError) as caught:
        with cli._recording_player_endpoint(endpoint, "episode-safe", destination):
            raise primary
    assert caught.value is primary
    assert any("Could not persist Player records" in note for note in primary.__notes__)
    assert PLAYER_ENDPOINT_KEY_ENV not in os.environ


def test_retry_cannot_overwrite_a_previous_capture(tmp_path):
    log_dir = tmp_path / "previous"
    log_dir.mkdir()
    records = log_dir / "player_records.json"
    records.write_text('[{"previous":"capture"}]')
    args = cli.build_parser().parse_args(["episode", "--player", "tinker"])
    with pytest.raises(RuntimeError, match="output directory already exists"):
        cli._run_with_tinker_player(args, log_dir)
    endpoint, _client = _endpoint(tmp_path)
    with pytest.raises(RuntimeError, match="Player records already exist"):
        with cli._recording_player_endpoint(endpoint, "other", records):
            pytest.fail("existing captures must not be opened")
    assert records.read_text() == '[{"previous":"capture"}]'
    assert endpoint._server is None


def test_record_serialization_with_installed_tinker_model_input(tmp_path):
    tinker = pytest.importorskip("tinker", reason="optional real Tinker SDK serialization check")
    from shellsmith.policy.records import TinkerRolloutRecord

    prompt = tinker.types.ModelInput.from_ints([42, 31415, 0])
    record = TinkerRolloutRecord(
        request_id="sdk-request", created_at=1.0, prompt=prompt, prompt_tokens=3,
        sampled_tokens=[71, 72], sampled_logprobs=[-0.1, -0.2], content="GH",
        finish_reason="stop", payload={"messages": []}, forced_close_token_span=(1, 2),
    )
    destination = tmp_path / "actual-sdk/player_records.json"
    cli._write_player_records(destination, [record])
    saved = json.loads(destination.read_text())[0]
    assert saved["prompt_token_ids"] == [42, 31415, 0]
    assert saved["sampled_tokens"] == [71, 72]
    assert saved["sampled_logprobs"] == [-0.1, -0.2]
    assert saved["forced_close_token_span"] == [1, 2]


def test_opaque_ids_do_not_collide_for_the_same_log_basename(tmp_path):
    first = cli.build_parser().parse_args(["episode", "--log-dir", str(tmp_path / "a/same")])
    second = cli.build_parser().parse_args(["episode", "--log-dir", str(tmp_path / "b/same")])
    assert cli._episode_id(first) != cli._episode_id(second)
