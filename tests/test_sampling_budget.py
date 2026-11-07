"""Completion-budget regressions through the real sampler and chat adapter.

Only the external Tinker/Cookbook interfaces are fixtures; the SDK fixture fills
its requested allowance so caps are checked against actual recorded tokens.
These tests do not call a live provider or replace any Shellsmith sampling code.
"""

from types import SimpleNamespace

import pytest

from shellsmith.policy.sampler import (
    CONTEXT_WINDOW_TOKENS,
    MIN_COMPLETION_TOKENS,
    RESPONSE_TOKEN_BUDGET,
    THINK_CLOSE_MARKER,
    THINK_CLOSE_TEXT,
    THINKING_TOKEN_BUDGET,
    PlayerRequestError,
    PlayerSamplingConfig,
    TinkerPlayerSampler,
)


NATURAL_CLOSE = 99
# Like the endpoint fixtures, encoding the full injected span is NOT the same
# length as encoding just </think>. Reserve the former, not the stop marker.
FORCED_CLOSE = (97, 98)
_DEFAULT = object()


class TokenizerFixture:
    def __init__(self, forced_ids=FORCED_CLOSE):
        self.forced_ids = list(forced_ids)

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        if text == THINK_CLOSE_MARKER:
            return [NATURAL_CLOSE]
        assert text == THINK_CLOSE_TEXT
        return list(self.forced_ids)

    def decode(self, tokens):
        return THINK_CLOSE_MARKER if NATURAL_CLOSE in tokens else "thinking"


class PromptFixture:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.length = len(self.tokens)

    def append(self, chunk):
        return PromptFixture(self.tokens + list(chunk.tokens))


class RendererFixture:
    def __init__(self, prompt_tokens, forced_ids):
        self.tokenizer = TokenizerFixture(forced_ids)
        self.prompt_tokens = prompt_tokens
        self.parsed_tokens = None
        self.loads = 0

    def create_conversation_prefix_with_tools(self, tool_specs, system_prompt):
        assert tool_specs == [] and system_prompt == ""
        return []

    def build_generation_prompt(self, messages):
        assert len(messages) == 1 and messages[0].role == "user"
        return PromptFixture([1] * self.prompt_tokens)

    def get_stop_sequences(self):
        return "END"

    def parse_response(self, tokens):
        self.parsed_tokens = list(tokens)
        content = [{"type": "text", "text": "answer"}]
        if NATURAL_CLOSE in tokens or 97 in tokens:
            content.insert(0, {"type": "thinking", "thinking": "reasoning"})
        return SimpleNamespace(role="assistant", content=content), SimpleNamespace(is_clean=True, value="stop")


class SamplingClientFixture:
    """External SDK fixture: exhaust budgets, optionally closing thinking early."""

    def __init__(self, *, thinking, natural_close, think_length, overrun_stage):
        self.thinking = thinking
        self.natural_close = natural_close
        self.think_length = think_length
        self.overrun_stage = overrun_stage
        self.calls = []
        self.scored_prompts = []

    def sample(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        stage = "think" if self.thinking and not self.calls else "response" if self.thinking else "single"
        self.calls.append((prompt, sampling_params))
        count = sampling_params.max_tokens
        assert isinstance(count, int) and count > 0, "Sampler must never request an empty/negative pass"
        if stage == "think" and self.think_length is not None:
            count = min(count, self.think_length)
        if stage == self.overrun_stage:
            count += 1
        if stage == "think":
            tokens = [10] * count
            if self.natural_close:
                tokens[-1] = NATURAL_CLOSE
            stop_reason = "stop" if self.natural_close else "length"
        else:
            tokens = [20] * count
            stop_reason = "length"
        sequence = SimpleNamespace(tokens=tokens, logprobs=[-0.5] * len(tokens), stop_reason=stop_reason)
        return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[sequence]))

    def compute_logprobs(self, prompt):
        self.scored_prompts.append(prompt)
        return SimpleNamespace(result=lambda: [None] + [-1.5] * (prompt.length - 1))


def _sampler(*, configured=_DEFAULT, thinking=True, natural_close=False, think_length=None,
             prompt_tokens=11, forced_ids=FORCED_CLOSE, overrun_stage=None):
    renderer = RendererFixture(prompt_tokens, forced_ids)

    def get_renderer(*_):
        renderer.loads += 1
        return renderer

    deps = SimpleNamespace(
        types=SimpleNamespace(EncodedTextChunk=lambda tokens: SimpleNamespace(tokens=tokens)),
        Message=SimpleNamespace, TextPart=SimpleNamespace, ThinkingPart=SimpleNamespace,
        get_tokenizer=lambda _: renderer.tokenizer,
        renderers=SimpleNamespace(get_renderer=get_renderer),
        tinker=SimpleNamespace(types=SimpleNamespace(SamplingParams=SimpleNamespace)),
    )
    client = SamplingClientFixture(thinking=thinking, natural_close=natural_close,
                                   think_length=think_length, overrun_stage=overrun_stage)
    config = PlayerSamplingConfig(base_model="external-sdk-fixture", **(
        {} if configured is _DEFAULT else {"max_tokens": configured}
    ))
    return TinkerPlayerSampler(config=config, deps=deps, sampling_client=client), client, renderer


def _sample(sampler, *, thinking=True, **options):
    return sampler.sample({"messages": [{"role": "user", "content": "continue"}],
                           "reasoning_effort": "high" if thinking else "none", **options},
                          request_id="budget", session_id="real-sampler")


def _assert_accounting(record, client, renderer, *, total):
    assert 0 < len(record.sampled_tokens) <= total
    assert record.prompt_tokens + len(record.sampled_tokens) <= CONTEXT_WINDOW_TOKENS
    assert len(record.sampled_tokens) == len(record.sampled_logprobs)
    assert renderer.parsed_tokens == record.sampled_tokens
    span = record.forced_close_token_span
    injected = span[1] - span[0] if span else 0
    assert record.protocol_row["thinking_tokens"] + record.protocol_row["response_tokens"] + injected == len(record.sampled_tokens)
    for prompt, params in client.calls:
        assert params.max_tokens > 0
        assert prompt.length + params.max_tokens <= CONTEXT_WINDOW_TOKENS
    for prompt in client.scored_prompts:
        assert prompt.length <= CONTEXT_WINDOW_TOKENS
    if span:
        assert span[0] == record.protocol_row["thinking_tokens"]
        assert record.sampled_tokens[span[0]:span[1]] == renderer.tokenizer.forced_ids
        assert record.sampled_logprobs[span[0]:span[1]] == [-1.5] * injected
        assert client.scored_prompts[0].tokens == client.calls[0][0].tokens + record.sampled_tokens[:span[1]]
    else:
        assert client.scored_prompts == []
    if len(client.calls) == 2:
        response_start = record.protocol_row["thinking_tokens"] + injected
        assert client.calls[1][0].tokens == client.calls[0][0].tokens + record.sampled_tokens[:response_start]
        assert client.calls[1][1].stop == ["END"]


@pytest.mark.parametrize("natural_close", [False, True], ids=["forced", "natural"])
@pytest.mark.parametrize("options,expected", [
    ({}, 4096),
    ({"max_tokens": 1024}, 1024),
    ({"max_completion_tokens": 1024}, 1024),
    ({"max_tokens": 50_000}, 4096),
    ({"max_completion_tokens": 50_000}, 4096),
    ({"max_tokens": 512, "max_completion_tokens": 2048}, 512),
    ({"max_tokens": 2048, "max_completion_tokens": 512}, 512),
    ({"max_tokens": None, "max_completion_tokens": 512}, 512),
    ({"max_tokens": 512, "max_completion_tokens": None}, 512),
])
def test_high_reasoning_obeys_configured_and_all_requested_total_caps(natural_close, options, expected):
    sampler, client, renderer = _sampler(configured=4096, natural_close=natural_close)
    record, assistant = _sample(sampler, **options)
    _assert_accounting(record, client, renderer, total=expected)
    assert len(record.sampled_tokens) == expected
    assert bool(record.forced_close_token_span) is not natural_close
    assert assistant["content"] == "answer" and assistant["reasoning_content"] == "reasoning"


@pytest.mark.parametrize("options,expected", [
    ({}, 1024), ({"max_tokens": 64}, 64), ({"max_completion_tokens": 64}, 64),
    ({"max_tokens": 8192}, 1024), ({"max_completion_tokens": 8192}, 1024),
    ({"max_tokens": 64, "max_completion_tokens": 128}, 64),
    ({"max_tokens": 128, "max_completion_tokens": 64}, 64),
])
def test_plain_sampling_uses_the_same_total_cap_policy(options, expected):
    sampler, client, renderer = _sampler(configured=1024, thinking=False)
    record, _ = _sample(sampler, thinking=False, **options)
    _assert_accounting(record, client, renderer, total=expected)
    assert len(record.sampled_tokens) == expected
    assert [params.max_tokens for _, params in client.calls] == [expected]


@pytest.mark.parametrize("thinking", [False, True], ids=["plain", "high-reasoning"])
@pytest.mark.parametrize("configured", [_DEFAULT, 32_768], ids=["sampler-default", "trainer-default"])
@pytest.mark.parametrize("options", [{}, {"max_tokens": None, "max_completion_tokens": None}], ids=["absent", "null"])
def test_absent_caps_preserve_ordinary_phase_budgets(thinking, configured, options):
    sampler, client, renderer = _sampler(configured=configured, thinking=thinking, natural_close=True)
    record, _ = _sample(sampler, thinking=thinking, **options)
    expected = ([THINKING_TOKEN_BUDGET] if thinking else []) + [RESPONSE_TOKEN_BUDGET]
    assert [params.max_tokens for _, params in client.calls] == expected
    _assert_accounting(record, client, renderer, total=sampler.config.max_tokens)


@pytest.mark.parametrize("configured", [_DEFAULT, 32_768], ids=["sampler-default", "trainer-default"])
def test_absent_caps_still_charge_injected_close_to_the_total(configured):
    sampler, client, renderer = _sampler(configured=configured)
    record, _ = _sample(sampler)
    _assert_accounting(record, client, renderer, total=sampler.config.max_tokens)
    assert client.calls[0][1].max_tokens == THINKING_TOKEN_BUDGET
    assert client.calls[1][1].max_tokens == min(
        RESPONSE_TOKEN_BUDGET, sampler.config.max_tokens - THINKING_TOKEN_BUDGET - len(FORCED_CLOSE),
    )


@pytest.mark.parametrize("natural_close", [False, True], ids=["forced", "natural"])
@pytest.mark.parametrize("think_length", [None, 3], ids=["exhausted", "early-close"])
def test_response_allowance_charges_only_actual_thinking_and_injected_span(natural_close, think_length):
    sampler, client, renderer = _sampler(configured=4096, natural_close=natural_close, think_length=think_length)
    record, _ = _sample(sampler)
    _assert_accounting(record, client, renderer, total=4096)
    assert client.calls[0][1].max_tokens == 256
    used_thinking = 256 if think_length is None else think_length
    injected = 0 if natural_close else len(FORCED_CLOSE)
    assert record.protocol_row["thinking_tokens"] == used_thinking
    assert client.calls[1][1].max_tokens == min(4096 - 256, 4096 - used_thinking - injected)
    if natural_close:
        assert record.sampled_tokens[used_thinking - 1] == NATURAL_CLOSE
        assert record.forced_close_token_span is None


@pytest.mark.parametrize("natural_close", [False, True], ids=["forced", "natural"])
@pytest.mark.parametrize("available", [MIN_COMPLETION_TOKENS, 5000, THINKING_TOKEN_BUDGET + RESPONSE_TOKEN_BUDGET])
def test_context_clamp_includes_every_recorded_token(natural_close, available):
    sampler, client, renderer = _sampler(configured=50_000, natural_close=natural_close,
                                        prompt_tokens=CONTEXT_WINDOW_TOKENS - available)
    record, _ = _sample(sampler, max_tokens=60_000)
    _assert_accounting(record, client, renderer, total=available)
    assert record.prompt_tokens + len(record.sampled_tokens) == CONTEXT_WINDOW_TOKENS


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("available", [-1, 0, 10, MIN_COMPLETION_TOKENS - 1])
def test_nearly_full_context_refuses_before_sampling(thinking, available):
    sampler, client, renderer = _sampler(thinking=thinking, prompt_tokens=CONTEXT_WINDOW_TOKENS - available)
    with pytest.raises(PlayerRequestError, match="Context budget exceeded") as error:
        _sample(sampler, thinking=thinking, max_completion_tokens=2048)
    assert error.value.status == 422
    assert client.calls == [] and client.scored_prompts == [] and renderer.parsed_tokens is None


@pytest.mark.parametrize("natural_close", [False, True], ids=["forced", "natural"])
@pytest.mark.parametrize("total", [4, 5, 8, 32])
def test_small_reasoning_budget_keeps_a_positive_visible_pass(natural_close, total):
    sampler, client, renderer = _sampler(configured=total, natural_close=natural_close,
                                        prompt_tokens=CONTEXT_WINDOW_TOKENS - total)
    record, _ = _sample(sampler)
    _assert_accounting(record, client, renderer, total=total)
    assert len(record.sampled_tokens) == total
    assert record.protocol_row["response_tokens"] >= 1


@pytest.mark.parametrize("total", [1, 2, 3])
def test_reasoning_budget_too_small_for_safe_closure_refuses_before_sampling(total):
    sampler, client, renderer = _sampler(configured=total)
    with pytest.raises(PlayerRequestError, match="budget") as error:
        _sample(sampler)
    assert error.value.status == 400
    assert client.calls == [] and client.scored_prompts == [] and renderer.parsed_tokens is None


def test_plain_one_token_budget_can_use_the_last_context_slot():
    sampler, client, renderer = _sampler(configured=1, thinking=False, prompt_tokens=CONTEXT_WINDOW_TOKENS - 1)
    record, _ = _sample(sampler, thinking=False)
    _assert_accounting(record, client, renderer, total=1)


@pytest.mark.parametrize("natural_close", [False, True], ids=["forced", "natural"])
def test_longer_injected_close_reserves_room_without_truncating_natural_close(natural_close):
    forced = (97,) * 6
    sampler, client, renderer = _sampler(configured=8, natural_close=natural_close, forced_ids=forced)
    record, _ = _sample(sampler)
    _assert_accounting(record, client, renderer, total=8)
    assert client.calls[0][1].max_tokens == 1
    assert record.protocol_row["response_tokens"] >= 1
    if natural_close:
        assert record.sampled_tokens[0] == NATURAL_CLOSE
        assert record.forced_close_token_span is None


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("field", ["configured", "max_tokens", "max_completion_tokens"])
@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.0, 1.5, "32", "bad", [], {}])
def test_invalid_caps_fail_before_loading_renderer_or_sampling(thinking, field, invalid):
    sampler, client, renderer = _sampler(configured=invalid if field == "configured" else 4096, thinking=thinking)
    options = {} if field == "configured" else {field: invalid}
    # A valid second field must not hide an invalid value through precedence.
    if field == "max_tokens":
        options["max_completion_tokens"] = 128
    elif field == "max_completion_tokens":
        options["max_tokens"] = 128
    with pytest.raises(PlayerRequestError, match="positive integer") as error:
        _sample(sampler, thinking=thinking, **options)
    assert error.value.status == 400
    assert renderer.loads == 0 and client.calls == [] and client.scored_prompts == []


def test_configured_null_is_not_an_unbounded_budget():
    sampler, client, renderer = _sampler(configured=None)
    with pytest.raises(PlayerRequestError, match="positive integer"):
        _sample(sampler)
    assert renderer.loads == 0 and client.calls == []


@pytest.mark.parametrize("stage", ["think", "response", "single"])
def test_sdk_over_budget_sequences_are_rejected_without_truncating_policy_tokens(stage):
    thinking = stage != "single"
    sampler, client, renderer = _sampler(configured=32, thinking=thinking, overrun_stage=stage)
    with pytest.raises(PlayerRequestError, match="budget") as error:
        _sample(sampler, thinking=thinking)
    assert error.value.status == 500 and renderer.parsed_tokens is None
    if stage == "think":
        assert len(client.calls) == 1 and client.scored_prompts == []


def test_empty_forced_close_encoding_fails_before_sampling():
    sampler, client, renderer = _sampler(forced_ids=())
    with pytest.raises(PlayerRequestError, match="empty </think>") as error:
        _sample(sampler)
    assert error.value.status == 500
    assert client.calls == [] and client.scored_prompts == [] and renderer.parsed_tokens is None
