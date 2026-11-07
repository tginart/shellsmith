"""Native protocol and two-pass budget tests that need no MLX installation."""

import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from shellsmith.mlx.render import NativeProtocolError, normalize_messages, parse_native_response
from shellsmith.mlx.sampler import MLXPlayerSampler
from shellsmith.policy.sampler import PlayerRequestError
from shellsmith.train.backend import TokenPrompt

TOOLS = [{"name": "shell_command", "description": "Run a command", "parameters": {"type": "object",
          "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
          "required": ["command"], "additionalProperties": False}}]


def test_native_tool_thinking_roundtrip_and_types():
    text = 'verify first</think>\n<tool_call>\n<function=shell_command>\n<parameter=command>\npwd\n</parameter>\n<parameter=timeout>5</parameter>\n</function>\n</tool_call><|im_end|>'
    assistant, finish = parse_native_response(text, thinking=True, tool_specs=TOOLS, request_id="r1", clean_stop=True)
    assert finish == "tool_calls"
    assert assistant["reasoning_content"] == "verify first"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"command": "pwd", "timeout": 5}
    history = normalize_messages({"messages": [{"role": "user", "content": "inspect"}, assistant,
        {"role": "tool", "tool_call_id": assistant["tool_calls"][0]["id"], "content": "/workspace"}]})
    assert history[1]["tool_calls"][0]["function"]["arguments"]["timeout"] == 5


@pytest.mark.parametrize("body", [
    "<tool_call><function=missing></function></tool_call>",
    "<tool_call><function=shell_command></function></tool_call>",
    "<tool_call><function=shell_command><parameter=command>pwd</parameter></function></tool_call>suffix",
    "<tool_call><function=shell_command><parameter=command>pwd</parameter><parameter=command>ls</parameter></function></tool_call>",
    "<tool_call><function=shell_command><parameter=command>pwd</parameter><parameter=timeout>bad</parameter></function></tool_call>",
    "<tool_call><function=shell_command><parameter=command>pwd</parameter>",
])
def test_native_parser_refuses_malformed_calls_without_text_recovery(body):
    with pytest.raises(NativeProtocolError):
        parse_native_response(body, thinking=False, tool_specs=TOOLS, request_id="r1", clean_stop=True)


def test_text_only_renderer_rejects_media_and_unmatched_tool_results():
    with pytest.raises(NativeProtocolError):
        normalize_messages({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "secret"}]}]})
    with pytest.raises(NativeProtocolError, match="matching"):
        normalize_messages({"messages": [{"role": "user", "content": "hello"},
                                         {"role": "tool", "tool_call_id": "missing", "content": "x"}]})


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        mapping = {10: "reason", 11: "done", 99: "</think>", 98: "<|im_end|>", 97: "\n", 96: "\n\n"}
        return "".join(mapping[token] for token in tokens)


class FakeSession:
    def __init__(self, passes):
        self.settings = SimpleNamespace(model_id="Qwen/Qwen3.8-27B", reasoning_effort="xhigh", max_tokens=8, thinking_tokens=4,
            response_tokens=4, plain_tokens=4, compaction_tokens=2, context_window=32768)
        self.lock = threading.RLock()
        self.policy_ref = "mlx-sampler:one"
        self.policy_identity = {"identity_sha256": "policy-one"}
        self.passes = list(passes)
        self.calls = []
        self.renderer = SimpleNamespace(close_ids=(99,), forced_close_ids=(97, 99, 96), eos_ids={98},
                                        tokenizer=FakeTokenizer(), render=lambda payload, thinking: (TokenPrompt((1, 2)), [], "tools"))

    def require_published(self): pass
    def prepare_prompt(self, ids): return None
    def draw_pass(self, cursor, *, maximum, stop_ids):
        self.calls.append(maximum)
        tokens, reason = self.passes.pop(0)
        assert len(tokens) <= maximum
        return list(tokens), [-1.0] * len(tokens), reason, cursor
    def force_tokens(self, cursor, tokens): return [-3.0] * len(tokens), cursor


def test_two_pass_keeps_exact_natural_close_and_stop_ids():
    session = FakeSession([([10, 99], "stop"), ([11, 98], "stop")])
    record, assistant = MLXPlayerSampler(session).sample({"messages": [{"role": "user", "content": "hello"}],
                                                        "reasoning_effort": "xhigh"}, request_id="r1", session_id="s1")
    assert record.sampled_tokens == [10, 99, 11, 98]
    assert record.sampled_logprobs == [-1.] * 4
    assert record.forced_close_token_span is None
    assert assistant["reasoning_content"] == "reason"
    assert assistant["content"] == "done"
    assert record.backend == "mlx" and record.policy_ref == session.policy_ref
    assert record.protocol_row["renderer_name"] == "qwen3_8_xhigh_reasoning"
    assert record.diagnostic_generation_attempts[0]["sampled_tokens"] == [10, 99]


def test_forced_close_is_scored_masked_and_charged_to_total_budget():
    session = FakeSession([([10] * 4, "length"), ([11], "length")])
    record, _ = MLXPlayerSampler(session).sample({"messages": [{"role": "user", "content": "hello"}],
                                                "reasoning_effort": "xhigh"}, request_id="r1", session_id="s1")
    assert len(record.sampled_tokens) == 8
    assert record.forced_close_token_span == (4, 7)
    assert record.sampled_logprobs[4:7] == [-3.] * 3
    assert session.calls == [4, 1]


def test_independent_4k_limits_keep_full_response_and_report_reasoning_limit(caplog):
    session = FakeSession([([10] * 4096, "length"), ([11] * 4095 + [98], "stop")])
    session.settings.max_tokens = 0
    session.settings.thinking_tokens = session.settings.response_tokens = 4096
    session.settings.reasoning_effort = "medium"
    record, assistant = MLXPlayerSampler(session).sample(
        {"reasoning_effort": "medium"}, request_id="at-limit", session_id="s1")
    assert session.calls == [4096, 4096]
    assert len(record.sampled_tokens) == 8195
    assert record.forced_close_token_span == (4096, 4099)
    assert record.protocol_row["renderer_name"] == "qwen3_8_medium_reasoning"
    assert record.protocol_row["reasoning_effort"] == "medium"
    assert record.protocol_row["response_limit"] == 4096
    assert record.protocol_row["generation_notices"][0]["code"] == "reasoning_limit_exceeded"
    assert "reasoning_limit_exceeded" in caplog.text
    assert assistant["content"] and record.finish_reason == "stop"


def test_derived_reserve_admits_exact_context_and_rejects_one_token_overflow():
    from shellsmith.mlx.generation import completion_budget

    for overflow in (False, True):
        session = FakeSession([([10] * 4096, "length"), ([11, 98], "stop")])
        session.settings.max_tokens = 0
        session.settings.thinking_tokens = session.settings.response_tokens = 4096
        session.settings.context_window = 262144
        reserve = completion_budget(session.settings)
        assert reserve == 8195
        ids = tuple(range(262144 - reserve + int(overflow)))
        session.renderer.render = lambda payload, thinking: (TokenPrompt(ids), [], "tools")
        if overflow:
            with pytest.raises(PlayerRequestError, match="context_length_exceeded"):
                MLXPlayerSampler(session).sample({"reasoning_effort": "medium"}, request_id="overflow", session_id="s1")
            assert session.calls == []
        else:
            record, _ = MLXPlayerSampler(session).sample({"reasoning_effort": "medium"}, request_id="exact", session_id="s1")
            assert record.prompt_tokens + reserve == 262144


def test_derived_plain_budget_follows_response_limit_and_explicit_request_caps():
    session = FakeSession([([11, 98], "stop")])
    session.settings.max_tokens = session.settings.plain_tokens = 0
    session.settings.response_tokens = 6
    MLXPlayerSampler(session).sample({}, request_id="plain", session_id="s1")
    assert session.calls == [6]
    session = FakeSession([([10] * 4, "length"), ([11], "length")])
    session.settings.max_tokens = 0
    record, _ = MLXPlayerSampler(session).sample(
        {"reasoning_effort": "medium", "max_completion_tokens": 8}, request_id="explicit", session_id="s1")
    assert len(record.sampled_tokens) == 8 and session.calls == [4, 1]


@pytest.mark.parametrize("effort", ["low", "medium", "xhigh"])
def test_renderer_passes_configured_effort_to_the_native_template(tmp_path, effort):
    from shellsmith.mlx.render import Renderer

    class Tokenizer:
        def encode(self, text, **kwargs):
            return {"</think>": [99], "\n</think>\n\n": [97, 99, 96],
                    "<|im_end|>": [98], "<|endoftext|>": [95]}.get(text, [1, 2])

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["reasoning_effort"] == effort
            assert kwargs["chat_template"] == "artifact-template"
            assert messages[0] == {"role": "system", "content": "Keep the application prompt."}
            assert kwargs["tools"][0]["function"]["name"] == "shell_command"
            return "rendered"

    (tmp_path / "chat_template.jinja").write_text("artifact-template")
    renderer = Renderer(tmp_path, tokenizer=Tokenizer(), reasoning_effort=effort)
    # Generic client labels cannot override the run's native-model effort.
    renderer.render({"messages": [{"role": "system", "content": "Keep the application prompt."},
                                  {"role": "user", "content": "inspect"}],
                     "tools": [{"type": "function", "function": TOOLS[0]}],
                     "reasoning_effort": "high"}, thinking=True)


def test_sampling_rejects_temperature_that_changes_behavior_distribution():
    with pytest.raises(PlayerRequestError, match="temperature"):
        MLXPlayerSampler(FakeSession([])).sample({"temperature": 0.7}, request_id="r1", session_id="s1")


def test_context_profile_is_enforced_without_prompt_truncation():
    session = FakeSession([])
    session.renderer.render = lambda payload, thinking: (TokenPrompt(tuple(range(32767))), [], "tools")
    with pytest.raises(PlayerRequestError) as caught:
        MLXPlayerSampler(session).sample({"messages": [{"role": "user", "content": "hello"}]}, request_id="r1", session_id="s1")
    assert caught.value.status == 422
    assert "context_length_exceeded" in str(caught.value)


def test_exact_native_budget_rejects_even_when_estimate_would_fit():
    session = FakeSession([])
    session.settings.max_tokens = 8192
    session.settings.thinking_tokens = session.settings.response_tokens = 4096
    session.renderer.render = lambda payload, thinking: (TokenPrompt(tuple(range(30720))), [], "tools")
    # A caller's smaller estimate cannot override the actual rendered IDs.
    with pytest.raises(PlayerRequestError, match="context_length_exceeded") as caught:
        MLXPlayerSampler(session).sample({"reasoning_effort": "xhigh", "max_tokens": 8192,
                                         "estimated_prompt_tokens": 24000}, request_id="r1", session_id="s1")
    assert caught.value.status == 422
    assert "requested_completion_tokens=8192" in str(caught.value)
    assert session.calls == []


def test_explicit_two_token_request_cap_fits_exact_native_window():
    session = FakeSession([([11, 98], "stop")])
    session.renderer.render = lambda payload, thinking: (TokenPrompt(tuple(range(32766))), [], "tools")
    record, assistant = MLXPlayerSampler(session).sample({"max_completion_tokens": 2}, request_id="r1", session_id="s1")
    assert record.prompt_tokens == 32766 and record.sampled_tokens == [11, 98]
    assert assistant["content"] == "done"
    assert session.calls == [2]


def test_protocol_failure_preserves_exact_rejected_sample_for_diagnostics():
    session = FakeSession([([10, 98], "stop")])
    session.renderer.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: "<tool_call><function=bad>")
    with pytest.raises(PlayerRequestError) as caught:
        MLXPlayerSampler(session).sample({"messages": [{"role": "user", "content": "hello"}]},
                                         request_id="rejected", session_id="s1")
    record = caught.value.record
    assert caught.value.status == 500
    assert record.prompt.to_ints() == [1, 2]
    assert record.sampled_tokens == [10, 98]
    assert record.sampled_logprobs == [-1., -1.]
    assert record.policy_identity == session.policy_identity
    assert record.protocol_row["protocol_error"]
    assert record.protocol_row["prompt_token_ids"] == [1, 2]


def test_tool_results_consume_ids_before_subsequent_messages():
    call = {"role": "assistant", "content": None,
            "tool_calls": [{"id": "tool1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "tool1", "content": "done"}
    user = {"role": "user", "content": "next"}
    with pytest.raises(NativeProtocolError):
        normalize_messages({"messages": [user, call, result, result]})
    with pytest.raises(NativeProtocolError):
        normalize_messages({"messages": [user, call, user, result]})


def test_native_string_argument_preserves_its_own_whitespace():
    command = "  printf 'hello'  \n"
    body = "<tool_call><function=shell_command><parameter=command>\n" + command + "\n</parameter></function></tool_call>"
    assistant, _ = parse_native_response(body, thinking=False, tool_specs=TOOLS, request_id="r1", clean_stop=True)
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["command"] == command


def test_thinking_success_after_eos_retry_preserves_every_discarded_draw():
    session = FakeSession([([10, 98], "stop"), ([10, 99], "stop"), ([11, 98], "stop")])
    record, _ = MLXPlayerSampler(session).sample({"reasoning_effort": "xhigh"}, request_id="retry", session_id="s1")
    assert record.sampled_tokens == [10, 99, 11, 98]
    attempts = record.diagnostic_generation_attempts
    assert len(attempts) == 2
    assert attempts[0]["sampled_tokens"] == [10, 98]
    assert attempts[0]["sampled_logprobs"] == [-1., -1.]
    assert attempts[0]["disposition"] == "rejected_eos_inside_think"
    assert attempts[1]["disposition"] == "accepted"


def test_exhausted_thinking_retries_preserve_prompt_policy_and_all_attempts():
    session = FakeSession([([10, 98], "stop")] * 3)
    with pytest.raises(PlayerRequestError) as caught:
        MLXPlayerSampler(session).sample({"reasoning_effort": "xhigh"}, request_id="retry", session_id="s1")
    record = caught.value.record
    assert caught.value.status == 500
    assert record.prompt.to_ints() == [1, 2]
    assert record.policy_ref == session.policy_ref and record.policy_identity == session.policy_identity
    assert record.sampled_tokens == [10, 98]
    assert len(record.diagnostic_generation_attempts) == 3
    assert all(attempt["sampled_tokens"] == [10, 98] and attempt["sampled_logprobs"] == [-1., -1.]
               for attempt in record.diagnostic_generation_attempts)
    assert "protocol_error" in record.protocol_row


@pytest.mark.parametrize("effort,limit", [("low", 2048), ("medium", 4096), ("xhigh", 8192)])
def test_sampler_uses_effort_limit_and_keeps_response_independent(effort, limit):
    session = FakeSession([([10] * limit, "length"), ([11, 98], "stop")])
    session.settings.max_tokens = session.settings.thinking_tokens = 0
    session.settings.response_tokens = 4096
    session.settings.reasoning_effort = effort
    record, _ = MLXPlayerSampler(session).sample({"reasoning_effort": effort}, request_id="effort", session_id="s1")
    assert session.calls == [limit, 4096]
    assert record.protocol_row["reasoning_limit"] == limit


def test_configured_thinking_retry_limit_is_enforced():
    session = FakeSession([([10, 98], "stop")] * 3)
    session.settings.thinking_anomaly_attempts = 1
    with pytest.raises(PlayerRequestError) as caught:
        MLXPlayerSampler(session).sample({"reasoning_effort": "medium"}, request_id="retry", session_id="s1")
    assert len(caught.value.record.diagnostic_generation_attempts) == 1
    assert len(session.passes) == 2


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_native_nonfinite_json_number_keeps_rejected_exact_sample(number):
    tools = [{"name": "measure", "parameters": {"type": "object", "properties": {"value": {"type": "number"}},
                                               "required": ["value"], "additionalProperties": False}}]
    text = "<tool_call><function=measure><parameter=value>" + number + "</parameter></function></tool_call>"
    session = FakeSession([([10, 98], "stop")])
    session.renderer.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: text)
    session.renderer.render = lambda payload, thinking: (TokenPrompt((1, 2)), tools, "tools")
    with pytest.raises(PlayerRequestError, match="nonfinite") as caught:
        MLXPlayerSampler(session).sample({}, request_id="numeric-error", session_id="s1")
    assert caught.value.record.sampled_tokens == [10, 98]
    assert caught.value.record.sampled_logprobs == [-1., -1.]
