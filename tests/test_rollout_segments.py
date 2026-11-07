"""Independent response replay is the objective oracle for packed histories."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from shellsmith.credit.compaction import CompactionAwareTrajectory, CompactionTurn
from shellsmith.credit.compaction_datums import build_normalized_tinker_trajectory
from shellsmith.train.backend import TokenPrompt, TokenSequence
from shellsmith.train.segments import pack_rollout_segments


def response(prompt, tokens, name, advantages=None, mask=None):
    mask = mask or [1.0] * len(tokens)
    return TokenSequence(
        tuple(prompt),
        tuple(tokens),
        tuple(-2.0 - i / 10 for i in range(len(tokens))),
        tuple(a * m for a, m in zip(advantages or [0.2] * len(tokens), mask)),
        tuple(mask),
        name,
    )


def histories():
    first = response([1, 2, 3], [4, 5, 6], "think-and-act", [0.3, 0.1, 0.2], [1, 0, 1])
    second = response(
        [1, 2, 3, 4, 5, 6, 7, 8], [9, 10], "tool-continuation", [-0.2, 0.4]
    )
    return first, second


def test_exact_spans_shift_and_context():
    a, b = histories()
    packed = pack_rollout_segments([(0, a), (0, b)])
    assert not packed.rewrites
    (s,) = packed.sequences
    assert (
        s.prompt_token_ids + s.sampled_tokens == b.prompt_token_ids + b.sampled_tokens
    )
    assert s.loss_mask == (1, 0, 1, 0, 0, 1, 1)
    assert s.source_spans == ((a.request_id, 3, 6), (b.request_id, 8, 10))
    for r in (a, b):
        start = len(r.prompt_token_ids) - len(s.prompt_token_ids)
        for i in range(len(r.sampled_tokens)):
            # Predictor row for target j is j-1 in the complete history.
            assert (
                s.prompt_token_ids + s.sampled_tokens[: start + i]
                == r.prompt_token_ids + r.sampled_tokens[:i]
            )
            assert s.behavior_logprobs[start + i] == r.behavior_logprobs[i]
            assert s.advantages[start + i] == r.advantages[i]


def test_rewrites_fail_or_split_with_evidence_and_compaction_stays_separate():
    a, b = histories()
    rewrite = replace(b, prompt_token_ids=(1, 2, 42, 4, 5, 6, 7, 8))
    with pytest.raises(ValueError, match="Incompatible causal history"):
        pack_rollout_segments([(0, a), (0, rewrite)], on_incompatible="error")
    result = pack_rollout_segments([(0, a), (0, rewrite)])
    assert (
        len(result.sequences) == 2 and result.rewrites[0]["common_prefix_tokens"] == 2
    )
    compact = pack_rollout_segments([(0, a), (1, b)])
    assert len(compact.sequences) == 2 and not compact.rewrites
    assert compact.sequences[0].advantages == a.advantages


def test_rollout_denominator_once_across_compaction_and_injected_tokens():
    a, b = histories()
    b = replace(b, prompt_token_ids=(40, 41, 42))  # actual compacted context
    turns = []
    for i, r in enumerate((a, b)):
        record = SimpleNamespace(
            request_id=r.request_id,
            prompt=TokenPrompt(r.prompt_token_ids),
            sampled_tokens=r.sampled_tokens,
            sampled_logprobs=r.behavior_logprobs,
            forced_close_token_span=(1, 2) if i == 0 else None,
        )
        turns.append(
            CompactionTurn(
                record,
                "compaction_generation" if i == 0 else "ordinary_action",
                sequence_index=i,
            )
        )
    trajectory = CompactionAwareTrajectory("rollout", "test", tuple(turns))
    result = build_normalized_tinker_trajectory(
        deps=None,
        trajectory=trajectory,
        advantages_by_request_id={a.request_id: 2.0, b.request_id: -1.0},
        group_id="g",
        step=0,
        rollout_index=0,
        reward=1.0,
    )
    assert result.trainable_token_count == 4
    assert result.segments[0].advantages == (0.5, 0.0, 0.5)
    assert result.segments[1].advantages == (-0.25, -0.25)
    assert len(result.segments) == 2
    assert result.segments[1].prompt_token_ids == (40, 41, 42)
    assert result.segments[0].sampled_tokens == a.sampled_tokens


def test_numerical_weights_are_preserved_and_context_is_zero():
    a, b = histories()
    a = replace(a, weights=(0.4, 0.0, 0.2))
    b = replace(b, weights=(-0.1, 0.3))
    (s,) = pack_rollout_segments([(0, a), (0, b)]).sequences
    assert s.weights == (0.4, 0.0, 0.2, 0.0, 0.0, -0.1, 0.3)


def test_duplicate_requests_and_invalid_segment_order_are_rejected():
    a, b = histories()
    with pytest.raises(ValueError, match="Duplicate"):
        pack_rollout_segments([(0, a), (0, a)])
    with pytest.raises(ValueError, match="ordered"):
        pack_rollout_segments([(1, a), (0, b)])


def test_tinker_receives_same_shifted_masks_credit_and_one_adam():
    import asyncio

    from shellsmith.train.config import TrainConfig
    from shellsmith.train.tinker import TinkerBackend

    a, b = histories()
    (s,) = pack_rollout_segments([(0, a), (0, b)]).sequences

    class Tensor:
        @staticmethod
        def from_torch(x):
            return x

    deps = SimpleNamespace(
        torch=SimpleNamespace(tensor=lambda x: x),
        TensorData=Tensor,
        types=SimpleNamespace(
            ModelInput=SimpleNamespace(from_ints=tuple),
            Datum=lambda **kw: SimpleNamespace(**kw),
            AdamParams=lambda **kw: kw,
        ),
    )
    events = []

    class Future:
        async def result_async(self):
            events.append("backward_done")
            return SimpleNamespace(metrics={})

    class Client:
        async def forward_backward_async(self, values, **kw):
            events.append("backward")
            assert len(values) == 1 and "mask" not in values[0].loss_fn_inputs
            fields = values[0].loss_fn_inputs
            assert values[0].model_input == s.prompt_token_ids + s.sampled_tokens[:-1]
            assert fields["advantages"] == [0.0, 0.0] + list(s.advantages)
            assert fields["logprobs"] == [0.0, 0.0] + list(s.behavior_logprobs)
            assert fields["target_tokens"] == list(
                s.prompt_token_ids[1:] + s.sampled_tokens
            )
            assert kw["loss_fn"] == "cispo"
            return Future()

        async def optim_step_async(self, adam):
            events.append("adam")
            assert adam == {
                "learning_rate": 5e-5,
                "beta1": 0.9,
                "beta2": 0.95,
                "eps": 1e-8,
            }
            return Future()

    backend = TinkerBackend(deps=deps, training_client=Client())
    asyncio.run(backend.update_raw([], config=TrainConfig(), sequences=[s]))
    assert events == ["backward", "backward_done", "adam", "backward_done"]


def test_source_spans_reject_context_supervision():
    _a, b = histories()
    with pytest.raises(ValueError, match="Context outside"):
        replace(
            b,
            source_spans=(
                ("b", len(b.prompt_token_ids) + 1, len(b.prompt_token_ids) + 2),
            ),
        )


def test_optional_kl_keeps_response_boundaries_and_remasks_context(monkeypatch):
    import asyncio
    import sys
    from types import ModuleType

    from shellsmith.train.tinker import TinkerBackend

    a, b = histories()
    (s,) = pack_rollout_segments([(0, a), (0, b)]).sequences
    deps = SimpleNamespace(
        torch=SimpleNamespace(tensor=list),
        TensorData=SimpleNamespace(from_torch=lambda x: x),
        types=SimpleNamespace(
            ModelInput=SimpleNamespace(from_ints=tuple),
            Datum=lambda **kw: SimpleNamespace(**kw),
        ),
    )
    metrics = ModuleType("tinker_cookbook.rl.metrics")

    async def incorporate(values, reference, coef, discount):
        assert len(values) == 2 and coef == 0.1 and discount == 0.8
        assert [len(v.model_input) for v in values] == [5, 9]
        for datum in values:
            datum.loss_fn_inputs["advantages"] = [
                10.0 + i for i in range(len(datum.model_input))
            ]

    metrics.incorporate_kl_penalty = incorporate
    monkeypatch.setitem(sys.modules, metrics.__name__, metrics)
    backend = TinkerBackend(deps=deps, training_client=None)
    (shaped,) = asyncio.run(
        backend._shape_segment_credit(
            [s], SimpleNamespace(kl_penalty_coef=0.1, kl_discount_factor=0.8)
        )
    )
    assert shaped.advantages == (12.0, 0.0, 14.0, 0.0, 0.0, 17.0, 18.0)
    assert (
        shaped.loss_mask == s.loss_mask
        and shaped.behavior_logprobs == s.behavior_logprobs
    )


def test_hosted_failed_backward_never_submits_adam():
    import asyncio

    from shellsmith.train.config import TrainConfig
    from shellsmith.train.tinker import TinkerBackend

    class Future:
        async def result_async(self):
            raise RuntimeError("failed backward")

    class Client:
        async def forward_backward_async(self, *a, **kw):
            return Future()

        async def optim_step_async(self, *a):
            pytest.fail("Adam must follow successful backward")

    deps = SimpleNamespace(
        types=SimpleNamespace(Datum=lambda **kw: kw, AdamParams=lambda **kw: kw)
    )
    datum = SimpleNamespace(datum=SimpleNamespace(model_input="i", loss_fn_inputs={}))
    with pytest.raises(RuntimeError, match="failed backward"):
        asyncio.run(
            TinkerBackend(deps=deps, training_client=Client()).update_raw(
                [datum], config=TrainConfig()
            )
        )
