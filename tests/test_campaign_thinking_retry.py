"""Campaign setup must pass retry settings to its actual prepared sampler.

External SDK/tokenizer/Codex/Pi interfaces are offline fixtures. Stop at the
rollout-stack boundary after exercising the shipped sampler's anomaly loop;
no world episodes, live provider calls, or optimizer operations run here.
"""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from shellsmith.campaign import inference
from shellsmith.policy.deps import load_tinker_deps
from shellsmith.policy.sampler import PlayerRequestError, TinkerPlayerSampler
from test_trainer_resume import _runtime, external_tinker_sdk, training_sdk


@pytest.mark.parametrize("attempts", [1, 5])
def test_tinker_campaign_prepared_sampler_honors_thinking_retry_limit(
    tmp_path, monkeypatch, training_sdk, attempts,
):
    config, _ = _runtime(tmp_path, monkeypatch)
    config = replace(config, retry=replace(config.retry, thinking_anomaly_attempts=attempts))
    calls = []

    class ReachedStack(Exception):
        pass

    def inspect_prepared_sampler(*args, **kwargs):
        sampler = kwargs["prepared_sampler"]
        assert isinstance(sampler, TinkerPlayerSampler)
        assert sampler.config.thinking_anomaly_attempts == attempts
        assert kwargs["sampling_client"] is sampler._sampling_client

        # A provider repeatedly stops inside open thinking. Count the actual
        # attempts rather than merely asserting configuration propagation.
        def malformed_thinking(**request):
            calls.append(request)
            assert "</think>" in request["sampling_params"].stop
            sequence = SimpleNamespace(tokens=[ord("x")], logprobs=[-0.5], stop_reason="stop")
            return SimpleNamespace(result=lambda: SimpleNamespace(sequences=[sequence]))

        sampler._sampling_client.sample = malformed_thinking
        with pytest.raises(PlayerRequestError, match="repeatedly ended the turn inside open thinking"):
            sampler.sample(
                {"messages": [{"role": "user", "content": "continue"}], "reasoning_effort": "medium"},
                request_id="campaign-retry", session_id="campaign-session",
            )
        assert len(calls) == attempts
        raise ReachedStack()

    monkeypatch.setattr(inference, "open_rollout_stack", inspect_prepared_sampler)
    with pytest.raises(ReachedStack):
        asyncio.run(inference.run_inference_campaign(config, deps=load_tinker_deps(), sampler_path=None))
    assert len(calls) == attempts
    assert training_sdk.updates == 0
