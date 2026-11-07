"""Player endpoint and shipped sampler regressions (not live Tinker model runs).

HTTP routing tests use FakeSampler; sampler tests below execute the real sampler
and chat adapter with explicit external Tinker/Cookbook interface fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest

from shellsmith.policy import sampler as sampler_module
from shellsmith.policy.records import TinkerRolloutRecord
from shellsmith.policy.sampler import (
    CONTEXT_WINDOW_TOKENS,
    PlayerRequestError,
    clamp_completion_budget,
    payload_is_compaction_summary,
    sample_thinking_two_pass,
    thinking_requested,
)
from shellsmith.policy.server import SERVED_MODEL_ID, PlayerEndpoint


# --- HTTP endpoint ---------------------------------------------------------

class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def sample(self, payload: dict, *, request_id: str, session_id: str) -> tuple[TinkerRolloutRecord, dict]:
        self.calls.append((request_id, session_id))
        assistant = {"role": "assistant", "content": None, "reasoning_content": "think first",
                     "tool_calls": [{"id": "call_1", "type": "function",
                                     "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}
        record = TinkerRolloutRecord(
            request_id=request_id, created_at=0.0, prompt=None, prompt_tokens=7,
            sampled_tokens=[1, 2, 3], sampled_logprobs=[-0.1, -0.2, -0.3], content="",
            finish_reason="stop", payload=payload, reasoning_content="think first", session_id=session_id,
            protocol_row={"request_id": request_id, "tool_calls": [{"id": "call_1", "name": "bash", "arguments": {}}]},
        )
        return record, assistant


def _request(url: str, *, key: str | None, body: dict | None = None) -> tuple[int, bytes, str]:
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                     method="GET" if body is None else "POST")
    if key is not None:
        request.add_header("Authorization", f"Bearer {key}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read(), response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers.get("Content-Type", "")


def test_endpoint_routes_auth_streams_and_records(tmp_path: Path) -> None:
    fake = FakeSampler()
    with PlayerEndpoint(sampler=fake, records_dir=tmp_path / "records") as endpoint:  # type: ignore[arg-type]
        base = endpoint.base_url("ep-1")
        assert _request(base + "/models", key=None)[0] == 401
        assert _request(base.replace("/e/ep-1", "/x"), key=endpoint.api_key)[0] == 404
        status, body, _ = _request(base + "/models", key=endpoint.api_key)
        assert status == 200 and json.loads(body)["data"][0]["id"] == SERVED_MODEL_ID

        payload = {"model": SERVED_MODEL_ID, "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        status, body, content_type = _request(base + "/chat/completions", key=endpoint.api_key, body=payload)
        assert status == 200 and content_type == "text/event-stream"
        events = [json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: {")]
        deltas = [event["choices"][0]["delta"] for event in events if event["choices"]]
        assert {"reasoning_content": "think first"} in deltas
        assert any("tool_calls" in delta for delta in deltas)
        assert events[-1]["usage"] == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        assert body.endswith(b"data: [DONE]\n\n")

        status, body, _ = _request(base + "/chat/completions", key=endpoint.api_key, body={**payload, "stream": False})
        assert json.loads(body)["choices"][0]["finish_reason"] == "tool_calls"

        records = endpoint.take_records("ep-1")
        assert [record.session_id for record in records] == ["ep-1", "ep-1"]
        assert endpoint.take_records("ep-1") == []
        rows = (tmp_path / "records" / "ep-1.jsonl").read_text().splitlines()
        assert len(rows) == 2 and json.loads(rows[0])["request_id"] == records[0].request_id


def test_endpoint_separates_episodes(tmp_path: Path) -> None:
    with PlayerEndpoint(sampler=FakeSampler(), records_dir=tmp_path) as endpoint:  # type: ignore[arg-type]
        for episode in ("a", "b", "a"):
            _request(endpoint.base_url(episode) + "/chat/completions", key=endpoint.api_key,
                     body={"messages": [{"role": "user", "content": "x"}]})
        assert len(endpoint.take_records("a")) == 2
        assert len(endpoint.take_records("b")) == 1


def test_endpoint_preserves_exact_rejected_native_generation(tmp_path: Path) -> None:
    from dataclasses import replace
    from shellsmith.train.backend import TokenPrompt

    class BrokenSampler(FakeSampler):
        def sample(self, payload, *, request_id, session_id):
            record, _ = super().sample(payload, request_id=request_id, session_id=session_id)
            record = replace(record, prompt=TokenPrompt((10, 20)), prompt_tokens=2,
                backend="mlx", policy_ref="policy:one", policy_identity={"sha": "one"},
                protocol_row={"request_id": request_id, "protocol_error": "malformed tool call"},
                diagnostic_generation_attempts=({"attempt": 1, "sampled_tokens": [8],
                    "sampled_logprobs": [-.4], "disposition": "rejected_eos_inside_think"},))
            raise PlayerRequestError("malformed tool call", status=500, record=record)

    with PlayerEndpoint(sampler=BrokenSampler(), records_dir=tmp_path) as endpoint:
        status, body, _ = _request(endpoint.base_url("failed") + "/chat/completions", key=endpoint.api_key,
                                   body={"messages": [{"role": "user", "content": "x"}]})
        assert status == 500 and "choices" not in json.loads(body)
        record, = endpoint.take_records("failed")
        row = json.loads((tmp_path / "failed.jsonl").read_text())
        assert row["protocol_error"] == "malformed tool call"
        assert row["prompt_token_ids"] == [10, 20]
        assert row["sampled_tokens"] == record.sampled_tokens == [1, 2, 3]
        assert row["sampled_logprobs"] == record.sampled_logprobs
        assert row["policy_identity"] == {"sha": "one"} and row["policy_ref"] == "policy:one"
        assert json.loads((tmp_path / "protocol_failures.jsonl").read_text()) == row
        assert row["diagnostic_generation_attempts"] == list(record.diagnostic_generation_attempts)


# --- sampling helpers --------------------------------------------------------

CLOSE = 99  # token id standing in for "</think>"
FORCED = [97, 98]  # token ids standing in for "\n</think>\n\n"


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [CLOSE] if text == "</think>" else list(FORCED)

    def decode(self, tokens: list[int]) -> str:
        return "</think>" if CLOSE in tokens else "thinking..."


class FakePrompt:
    def __init__(self, tokens: list[int]) -> None:
        self.tokens = tokens
        self.length = len(tokens)

    def append(self, chunk: SimpleNamespace) -> "FakePrompt":
        return FakePrompt(self.tokens + list(chunk.tokens))


class FakeClient:
    """Replays scripted (tokens, stop_reason) samples; scores forced tokens at -1.5."""

    def __init__(self, samples: list[tuple[list[int], str]]) -> None:
        self.samples = list(samples)
        self.prompts: list[list[int]] = []
        self.sampling_params: list[dict | SimpleNamespace] = []

    def sample(self, *, prompt: FakePrompt, num_samples: int, sampling_params: dict | SimpleNamespace) -> SimpleNamespace:
        assert num_samples == 1
        self.prompts.append(prompt.tokens)
        self.sampling_params.append(sampling_params)
        tokens, stop = self.samples.pop(0)
        sequence = SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason=stop)
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[sequence]))

    def compute_logprobs(self, prompt: FakePrompt) -> SimpleNamespace:
        return SimpleNamespace(result=lambda: [None] + [-1.5] * (prompt.length - 1))


DEPS = SimpleNamespace(types=SimpleNamespace(EncodedTextChunk=lambda tokens: SimpleNamespace(tokens=tokens)))


def _two_pass(client: FakeClient) -> dict:
    return sample_thinking_two_pass(
        sampling_client=client, prompt=FakePrompt([1, 2]), tokenizer=FakeTokenizer(), deps=DEPS,
        params=lambda max_tokens, stop: {"max_tokens": max_tokens, "stop": stop},
        thinking_cap=8, response_cap=8, base_stop=["<|im_end|>"],
    )


def test_two_pass_natural_close_has_no_forced_span() -> None:
    client = FakeClient([([10, 11, CLOSE], "stop"), ([20, 21], "stop")])
    result = _two_pass(client)
    assert result["sampled_tokens"] == [10, 11, CLOSE, 20, 21]
    assert result["forced_close_token_span"] is None
    assert client.prompts[1] == [1, 2, 10, 11, CLOSE]


def test_two_pass_budget_overrun_forces_close_and_marks_span() -> None:
    client = FakeClient([([10, 11, 12], "length"), ([20], "stop")])
    result = _two_pass(client)
    assert result["sampled_tokens"] == [10, 11, 12, *FORCED, 20]
    assert result["sampled_logprobs"][3:5] == [-1.5, -1.5]
    assert result["forced_close_token_span"] == (3, 5)


def test_two_pass_refuses_repeated_end_of_turn_inside_thinking() -> None:
    client = FakeClient([([10], "stop")] * 3)
    with pytest.raises(PlayerRequestError, match="inside open thinking"):
        _two_pass(client)


def test_clamp_completion_budget() -> None:
    assert clamp_completion_budget(prompt_tokens=1000, max_tokens=2000) == 2000
    assert clamp_completion_budget(prompt_tokens=CONTEXT_WINDOW_TOKENS - 5000, max_tokens=28_000) == 5000
    with pytest.raises(PlayerRequestError) as caught:
        clamp_completion_budget(prompt_tokens=CONTEXT_WINDOW_TOKENS - 10, max_tokens=28_000)
    assert caught.value.status == 422


def test_thinking_detection() -> None:
    assert thinking_requested({"reasoning_effort": "high"})
    assert not thinking_requested({"reasoning_effort": "none"})
    assert not thinking_requested({})


class CookbookRendererFixture:
    """External Cookbook interface only; no Shellsmith function is replaced."""

    def __init__(self, *, stop, content, tool_calls=None) -> None:
        self.tokenizer = FakeTokenizer()
        self.stop = stop
        self.content = content
        self.tool_calls = tool_calls
        self.parsed_tokens: list[int] = []

    def create_conversation_prefix_with_tools(self, tool_specs, system_prompt):
        return [SimpleNamespace(role="system", content=system_prompt)] if system_prompt else []

    def build_generation_prompt(self, messages):
        text = "|".join(f"{message.role}:{message.content}" for message in messages)
        return FakePrompt([ord(character) for character in text])

    def get_stop_sequences(self):
        return self.stop

    def parse_response(self, tokens):
        self.parsed_tokens = list(tokens)
        return (
            SimpleNamespace(role="assistant", content=self.content, tool_calls=self.tool_calls),
            SimpleNamespace(is_clean=True, value="stop"),
        )


def _sampler_fixture(*, thinking=False, default_stop=("<|im_end|>",), content="done", tool_calls=None):
    """Real sampler with scripted external SDK responses, not a live provider."""
    renderer = CookbookRendererFixture(stop=default_stop, content=content, tool_calls=tool_calls)
    deps = SimpleNamespace(
        types=DEPS.types, Message=SimpleNamespace, TextPart=SimpleNamespace, ThinkingPart=SimpleNamespace,
        get_tokenizer=lambda _: renderer.tokenizer,
        renderers=SimpleNamespace(get_renderer=lambda *_: renderer),
        tinker=SimpleNamespace(types=SimpleNamespace(SamplingParams=SimpleNamespace)),
    )
    client = FakeClient(([([10, CLOSE], "stop")] if thinking else []) + [([20, 21], "stop")])
    sampler = sampler_module.TinkerPlayerSampler(
        config=sampler_module.PlayerSamplingConfig(
            base_model="external-sdk-fixture", plain_renderer="qwen3_5", thinking_renderer="qwen3_5",
        ),
        deps=deps, sampling_client=client,
    )
    return sampler, client, renderer


@pytest.mark.parametrize("thinking", [False, True], ids=["single-pass", "two-pass"])
@pytest.mark.parametrize("options,default_stop,expected", [
    ({"stop": "END"}, ["<|im_end|>"], ["END"]),
    ({"stop": ["END", "STOP"]}, ["<|im_end|>"], ["END", "STOP"]),
    ({"stop": ["END", "</think>"]}, ["<|im_end|>"], ["END", "</think>"]),
    ({}, ["<|im_end|>"], ["<|im_end|>"]),
    ({"stop": None}, ("END", "STOP"), ["END", "STOP"]),
    ({"stop": []}, ["<|im_end|>"], ["<|im_end|>"]),
    ({"stop": ""}, ["<|im_end|>"], ["<|im_end|>"]),
    ({}, "END", ["END"]),
    ({}, [7, 8], [7, 8]),
    ({}, [], []),
], ids=["scalar", "sequences", "existing-think-stop", "default", "null", "empty-list", "empty-string",
        "scalar-default", "token-default", "no-stops"])
def test_sampler_preserves_stop_sequences(options, default_stop, expected, thinking: bool) -> None:
    sampler, client, renderer = _sampler_fixture(thinking=thinking, default_stop=default_stop)
    payload = {"messages": [{"role": "user", "content": "continue"}], **options}
    if thinking:
        payload["reasoning_effort"] = "high"
    original_payload = json.dumps(payload)
    original_defaults = json.dumps(default_stop)

    record, assistant = sampler.sample(payload, request_id="stops", session_id="sampler")

    stops = [params.stop for params in client.sampling_params]
    if thinking:
        think_stop = expected.copy()
        close = CLOSE if expected and all(isinstance(item, int) for item in expected) else "</think>"
        if close not in think_stop:
            think_stop.append(close)
        assert stops == [think_stop, expected]
        assert client.prompts[1] == client.prompts[0] + [10, CLOSE]
    else:
        assert stops == [expected]
    assert record.sampled_tokens == ([10, CLOSE] if thinking else []) + [20, 21]
    assert record.sampled_logprobs == [-0.5] * len(record.sampled_tokens)
    assert renderer.parsed_tokens == record.sampled_tokens
    assert record.forced_close_token_span is None
    assert assistant["content"] == "done"
    assert json.dumps(payload) == original_payload
    assert json.dumps(default_stop) == original_defaults


@pytest.mark.parametrize("content", [
    "<summary_request> now",
    "Explain the literal <summary_request> tag in this source file.",
    "<summary_request>Summarize this task.</summary_request>",
    [{"type": "text", "text": "Document <summary_request> and </summary_request>."}],
])
def test_sampler_does_not_classify_user_summary_tags_as_compaction(content) -> None:
    sampler, _, _ = _sampler_fixture()
    payload = {"messages": [{"role": "user", "content": content}]}
    record, _ = sampler.sample(payload, request_id="ordinary-user", session_id="sampler")
    assert not payload_is_compaction_summary(payload)
    assert record.protocol_row["compaction_summary"] is False


def _pi_summary_payload(kind="history", *, system_role="system", content_parts=False, focus=None):
    """Pinned request fixture; the installed-Pi test independently checks these prompts."""
    prompt = "<conversation>\n[User]: Fix the regression.\n</conversation>\n\n"
    if kind == "update":
        prompt += "<previous-summary>\nEarlier checkpoint.\n</previous-summary>\n\n"
    prompt += {
        "history": sampler_module.PI_COMPACTION_SUMMARY_PROMPT,
        "update": sampler_module.PI_COMPACTION_UPDATE_PROMPT,
        "prefix": sampler_module.PI_COMPACTION_TURN_PREFIX_PROMPT,
    }[kind]
    if focus:
        assert kind != "prefix"  # Pi does not pass customInstructions to the prefix call.
        prompt += "\n\nAdditional focus: " + focus
    system = sampler_module.PI_COMPACTION_SUMMARY_SYSTEM_PROMPT
    return {"messages": [
        {"role": system_role, "content": [{"type": "text", "text": system}] if content_parts else system},
        {"role": "user", "content": [{"type": "text", "text": prompt}] if content_parts else prompt},
    ]}


@pytest.mark.parametrize("kind,focus", [
    ("history", None), ("update", None), ("prefix", None),
    ("history", "Keep paths exact.\nExplain <summary_request>."),
    ("update", "Keep paths exact.\nExplain <summary_request>."),
])
@pytest.mark.parametrize("system_role", ["system", "developer"])
@pytest.mark.parametrize("content_parts", [False, True], ids=["text", "content-parts"])
def test_sampler_recognizes_complete_pi_compaction_framing(kind, focus, system_role, content_parts) -> None:
    sampler, _, _ = _sampler_fixture()
    payload = _pi_summary_payload(kind, system_role=system_role, content_parts=content_parts, focus=focus)
    record, _ = sampler.sample(payload, request_id="summary", session_id="sampler")
    assert payload_is_compaction_summary(payload)
    assert record.protocol_row["compaction_summary"] is True


@pytest.mark.parametrize("changed", [
    "missing-system", "system-prefix-only", "changed-system", "ordinary-system", "user-system",
    "unframed-user", "quoted-request", "truncated-request", "changed-instructions", "missing-conversation",
    "appended-text", "earlier-summary", "assistant-request", "tools", "non-text-system-part",
])
def test_sampler_requires_complete_standalone_compaction_provenance(changed: str) -> None:
    payload = _pi_summary_payload()
    system, user = payload["messages"]
    if changed == "missing-system":
        payload["messages"] = [user]
    elif changed == "system-prefix-only":
        system["content"] = "You are a context summarization assistant. ..."
    elif changed == "changed-system":
        system["content"] += " Also answer the user."
    elif changed == "ordinary-system":
        system["content"] = "You are a coding assistant."
    elif changed == "user-system":
        system["role"] = "user"
    elif changed == "unframed-user":
        user["content"] = "Explain <summary_request>."
    elif changed == "quoted-request":
        user["content"] = "Explain this template:\n" + user["content"]
    elif changed == "truncated-request":
        user["content"] = user["content"][:-1]
    elif changed == "changed-instructions":
        user["content"] = user["content"].replace("## Goal", "## User Goal")
    elif changed == "missing-conversation":
        user["content"] = sampler_module.PI_COMPACTION_SUMMARY_PROMPT
    elif changed == "appended-text":
        user["content"] += "\nContinue the conversation."
    elif changed == "earlier-summary":
        payload["messages"] += [
            {"role": "assistant", "content": "Checkpoint"},
            {"role": "user", "content": "Now explain <summary_request>."},
        ]
    elif changed == "assistant-request":
        user["role"] = "assistant"
    elif changed == "tools":
        payload["tools"] = [{"type": "function", "function": {
            "name": "bash", "description": "Run a command", "parameters": {"type": "object"},
        }}]
    elif changed == "non-text-system-part":
        system["content"] = [{"type": "text", "text": system["content"]},
                             {"type": "thinking", "thinking": "ordinary history"}]
    sampler, _, _ = _sampler_fixture()
    record, _ = sampler.sample(payload, request_id="ordinary", session_id="sampler")
    assert payload_is_compaction_summary(payload) is False
    assert record.protocol_row["compaction_summary"] is False


@pytest.mark.parametrize("messages", [
    None, {}, [], [None, "<summary_request>"],
    [{"role": "user", "content": "<summary_request>"}, {}],
    [{"role": [], "content": ""}, {"role": "user", "content": ""}],
])
def test_compaction_detection_rejects_malformed_messages(messages) -> None:
    assert not payload_is_compaction_summary({"messages": messages})


def _installed_pi_compaction_cases():
    """Run installed Pi's compact + OpenAI converter; fixture only the external LLM response.

    No network call, Shellsmith unit, Pi compactor, or wire converter is mocked.
    Optional because the Python-only dev install need not have Node/Pi installed.
    """
    runner = Path(__file__).resolve().parents[1] / "src/shellsmith/episode/vat_runner"
    pi = runner / "node_modules/@earendil-works/pi-coding-agent"
    node = shutil.which("node")
    if not node or not (pi / "dist/core/compaction/compaction.js").is_file():
        pytest.skip("Optional installed-Pi compaction contract requires Node and npm ci in vat_runner")
    lock = json.loads((runner / "package-lock.json").read_text())
    assert json.loads((pi / "package.json").read_text())["version"] == "0.84.1"
    assert lock["packages"]["node_modules/@earendil-works/pi-coding-agent"]["version"] == "0.84.1"
    result = subprocess.run([node, "--input-type=module", "-e", r'''
import { pathToFileURL } from "node:url";
const pi = process.argv[1];
const load = path => import(pathToFileURL(`${pi}/${path}`));
const { compact } = await load("dist/core/compaction/compaction.js");
const { createFileOps } = await load("dist/core/compaction/utils.js");
const { convertMessages } = await load("node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js");
const model = { id: "external-llm-fixture", api: "openai-completions", provider: "fixture",
                reasoning: true, maxTokens: 16384, input: ["text"] };
const user = text => ({ role: "user", content: [{ type: "text", text }], timestamp: 0 });
const usage = { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2,
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const cases = [];
for (const name of ["history", "update", "split", "prefix-only"]) {
    const requests = [];
    const streamFn = async (_model, context) => {
        const text = `${name} checkpoint ${requests.length + 1}`;
        const compat = { supportsDeveloperRole: name === "update" || name === "split" };
        requests.push({ text, payload: { messages: convertMessages(model, context, compat) } });
        return { result: async () => ({ role: "assistant", content: [{ type: "text", text }],
                                       stopReason: "stop", usage }) };
    };
    const split = name === "split" || name === "prefix-only";
    const preparation = {
        firstKeptEntryId: `kept-${name}`, tokensBefore: 160000,
        messagesToSummarize: name === "prefix-only" ? [] : [user("Earlier discussion of the regression.")],
        turnPrefixMessages: split ? [user("Continue fixing the regression.")] : [],
        isSplitTurn: split, previousSummary: name === "update" ? "Previous checkpoint." : undefined,
        fileOps: createFileOps(), settings: { reserveTokens: 16384 },
    };
    const focus = name === "history" ? undefined : "Keep exact filenames.\nReport verified progress.";
    const compaction = await compact(preparation, model, "external-fixture-key", undefined,
        focus, undefined, "off", streamFn);
    cases.push({ name, requests, compaction });
}
console.log(JSON.stringify(cases));
''', str(pi)], cwd=runner, text=True, capture_output=True, timeout=30, check=True)
    return json.loads(result.stdout)


def test_cli_window_matches_sampler_window() -> None:
    from shellsmith import cli

    assert cli.TINKER_CONTEXT_WINDOW_TOKENS == CONTEXT_WINDOW_TOKENS
