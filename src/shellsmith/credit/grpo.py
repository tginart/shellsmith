"""GRPO trainable datum contracts and Tinker datum conversion.

Architecture map:
    1. `RolloutGroupSpec` computes advantages from simulator rewards.
    2. `GRPOTrainableDatumSpec` binds one accepted rollout record to its group
       reward/advantage.
    3. `to_tinker_datum` converts the contract into the renderer-specific
       in-memory `tinker.types.Datum` used by CISPO or importance-sampling updates.
    4. `GRPOTrainableDatumProofSpec` is the JSONL-friendly audit row because
       the real datum can contain non-JSON prompt and image objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from shellsmith.credit.rollout import TrainableRolloutRecordSpec
from shellsmith.types import JsonValue, LossFnName, require_finite_float


@dataclass(frozen=True)
class GRPOTrainableDatumSpec:
    """One literal trainable datum candidate before Tinker object conversion."""

    schema: Literal["shellsmith.grpo_trainable_datum.v0"]
    datum_index: int
    group_id: str
    step: int
    rollout_index: int
    record_index: int
    request_id: str
    prompt: Any
    sampled_tokens: tuple[int, ...]
    sampled_logprobs: tuple[float, ...]
    reward: float
    advantage: float
    loss_fn: LossFnName = "importance_sampling"
    loss_mask_strategy: Literal["zero_observation_prefix"] = "zero_observation_prefix"
    # Half-open [start, end) span inside `sampled_tokens` covering tokens the
    # harness injected deterministically (forced `</think>` close). They carry
    # correct behavior logprobs for context, but were never policy actions:
    # their advantage and loss mask are zeroed to remove direct target loss.
    # Their use as later context remains differentiable.
    forced_close_token_span: tuple[int, int] | None = None
    # False for a record whose sampled tokens must stay in the batch as context
    # but train nothing — today that is a replayed context-compaction summary,
    # whose tokens were already trained once as the compaction *generation*.
    # Both the advantage vector and the loss mask are zeroed over the sampled
    # span, removing direct target loss while retaining context gradients.
    train_sampled_tokens: bool = True

    @staticmethod
    def from_rollout_record(
        *,
        datum_index: int,
        group_id: str,
        step: int,
        rollout_index: int,
        record_index: int,
        record: object,
        reward: float,
        advantage: float,
    ) -> "GRPOTrainableDatumSpec":
        parsed = TrainableRolloutRecordSpec.from_record(record)
        return GRPOTrainableDatumSpec(
            schema="shellsmith.grpo_trainable_datum.v0",
            datum_index=datum_index,
            group_id=group_id,
            step=step,
            rollout_index=rollout_index,
            record_index=record_index,
            request_id=parsed.request_id,
            prompt=parsed.prompt,
            sampled_tokens=parsed.sampled_tokens,
            sampled_logprobs=parsed.sampled_logprobs,
            reward=require_finite_float(reward, context="grpo.reward"),
            advantage=require_finite_float(advantage, context="grpo.advantage"),
            forced_close_token_span=parsed.forced_close_token_span,
        )

    def to_tinker_datum(self, deps: Any, *, include_action_mask: bool = False) -> Any:
        """Convert this spec into Tinker's renderer-specific `Datum`.

        Main callers:
            The online/no-train supervisor while building optimizer candidates.

        Pythonese flow:
            - Keep the original prompt as observation context.
            - Append sampled output tokens except the final next-token target.
            - Pad target/logprob/advantage arrays over the observation prefix
              with zero values.
            - Put the rollout advantage on sampled-output positions only.
            - Add the separate action mask only when a caller needs Cookbook
              KL bookkeeping; hosted policy losses receive the strict default.
        """
        if not math.isfinite(self.advantage):
            raise RuntimeError(
                f"Trainable datum {self.request_id!r} has a non-finite advantage "
                f"{self.advantage!r}."
            )
        if any(not math.isfinite(value) for value in self.sampled_logprobs):
            raise RuntimeError(
                f"Trainable datum {self.request_id!r} has a non-finite sampled logprob."
            )
        ob_len = self.prompt.length - 1
        model_input = self.prompt.append(deps.types.EncodedTextChunk(tokens=list(self.sampled_tokens[:-1])))
        target_tokens = [0] * ob_len + list(self.sampled_tokens)
        padded_logprobs = [0.0] * ob_len + list(self.sampled_logprobs)
        padded_advantages = [0.0] * ob_len + [self.advantage] * (model_input.length - ob_len)
        loss_mask = [0.0] * ob_len + [1.0] * (model_input.length - ob_len)
        if not self.train_sampled_tokens:
            # Context-only record: keep the sequence, train nothing on it.
            padded_advantages = [0.0] * model_input.length
            loss_mask = [0.0] * model_input.length
        if self.forced_close_token_span is not None:
            start, end = self.forced_close_token_span
            if not (0 <= start <= end <= len(self.sampled_tokens)):
                raise RuntimeError(
                    "Forced-close token span is out of bounds: "
                    f"{self.forced_close_token_span!r} for "
                    f"{len(self.sampled_tokens)} sampled tokens."
                )
            # These tokens were injected by the harness, not sampled from the
            # policy: keep them as context but train nothing on them.
            for index in range(start, end):
                padded_advantages[ob_len + index] = 0.0
                loss_mask[ob_len + index] = 0.0
        if not (
            model_input.length
            == len(target_tokens)
            == len(padded_logprobs)
            == len(padded_advantages)
            == len(loss_mask)
        ):
            raise RuntimeError(
                "RL datum length mismatch: "
                f"model_input={model_input.length} "
                f"target_tokens={len(target_tokens)} "
                f"logprobs={len(padded_logprobs)} "
                f"advantages={len(padded_advantages)}"
            )
        loss_fn_inputs = {
            "target_tokens": deps.TensorData.from_torch(deps.torch.tensor(target_tokens)),
            "logprobs": deps.TensorData.from_torch(deps.torch.tensor(padded_logprobs)),
            "advantages": deps.TensorData.from_torch(deps.torch.tensor(padded_advantages)),
        }
        if include_action_mask:
            loss_fn_inputs["mask"] = deps.TensorData.from_torch(deps.torch.tensor(loss_mask))
        return deps.types.Datum(
            model_input=model_input,
            loss_fn_inputs=loss_fn_inputs,
        )

    def to_token_sequence(self) -> Any:
        """Apply the same action masks before accelerator-specific conversion."""
        from shellsmith.train.backend import TokenSequence
        tokens = tuple(self.prompt.to_ints() if hasattr(self.prompt, "to_ints") else self.prompt.tokens)
        mask = [float(self.train_sampled_tokens)] * len(self.sampled_tokens)
        if self.forced_close_token_span is not None:
            start, end = self.forced_close_token_span
            if not 0 <= start <= end <= len(mask):
                raise RuntimeError("Forced-close span is out of bounds.")
            mask[start:end] = [0.0] * (end - start)
        return TokenSequence(prompt_token_ids=tokens, sampled_tokens=self.sampled_tokens,
            behavior_logprobs=self.sampled_logprobs,
            advantages=tuple(self.advantage if value else 0.0 for value in mask),
            loss_mask=tuple(mask), request_id=self.request_id)


@dataclass(frozen=True)
class GRPOTrainableDatumProofSpec:
    """JSON-safe proof that one accepted record became a datum candidate."""

    schema: Literal["shellsmith.grpo_trainable_datum_proof.v0"]
    datum_index: int
    group_id: str
    world_spec_id: str
    run_id: str
    rollout_path: str
    canonical_initial_state_id: str | None
    rollout_index: int
    record_index: int
    request_id: str
    reward: float
    advantage: float
    prompt_tokens: int
    sampled_token_count: int
    sampled_logprob_count: int
    finish_reason: str
    loss_fn: LossFnName
    no_train: bool
    optimizer_submitted: bool
    acceptance_audit_passed: bool

    def to_jsonable(self) -> dict[str, JsonValue]:
        return {
            "schema": self.schema,
            "datum_index": self.datum_index,
            "group_id": self.group_id,
            "world_spec_id": self.world_spec_id,
            "run_id": self.run_id,
            "rollout_path": self.rollout_path,
            "canonical_initial_state_id": self.canonical_initial_state_id,
            "rollout_index": self.rollout_index,
            "record_index": self.record_index,
            "request_id": self.request_id,
            "reward": self.reward,
            "advantage": self.advantage,
            "prompt_tokens": self.prompt_tokens,
            "sampled_token_count": self.sampled_token_count,
            "sampled_logprob_count": self.sampled_logprob_count,
            "finish_reason": self.finish_reason,
            "loss_fn": self.loss_fn,
            "no_train": self.no_train,
            "optimizer_submitted": self.optimizer_submitted,
            "acceptance_audit_passed": self.acceptance_audit_passed,
        }


def build_tinker_datum_from_record(*, deps: Any, record: object, advantage: float) -> Any:
    """Compatibility wrapper around `GRPOTrainableDatumSpec`.

    Main callers:
        Existing unit tests and any harness code that still expects the old
        `_rl_datum_from_sample`-style callable during the migration.
    """
    spec = GRPOTrainableDatumSpec.from_rollout_record(
        datum_index=0,
        group_id="compat",
        step=0,
        rollout_index=0,
        record_index=0,
        record=record,
        reward=0.0,
        advantage=advantage,
    )
    return spec.to_tinker_datum(deps)


def rl_datum_from_sample(
    deps: Any,
    prompt: Any,
    sampled_tokens: list[int],
    sampled_logprobs: list[float],
    advantage: float,
    *,
    include_action_mask: bool = False,
    forced_close_token_span: tuple[int, int] | None = None,
) -> Any:
    """Build a Tinker datum from raw prompt/tokens/logprobs.

    Main callers:
        The small standalone RFT harness and the simulator supervisor default
        datum builder.

    Why this exists:
        Some harnesses do not have a proxy record object. They still need the
        same trainable datum semantics as the generic GRPO contract.
    """
    spec = GRPOTrainableDatumSpec(
        schema="shellsmith.grpo_trainable_datum.v0",
        datum_index=0,
        group_id="raw_sample",
        step=0,
        rollout_index=0,
        record_index=0,
        request_id="raw_sample",
        prompt=prompt,
        sampled_tokens=tuple(sampled_tokens),
        sampled_logprobs=tuple(sampled_logprobs),
        reward=0.0,
        advantage=advantage,
        forced_close_token_span=forced_close_token_span,
    )
    if not spec.sampled_tokens or not spec.sampled_logprobs:
        raise RuntimeError("Trainable sample is missing sampled tokens/logprobs.")
    if len(spec.sampled_tokens) != len(spec.sampled_logprobs):
        raise RuntimeError(
            "Trainable sample has mismatched sampled token/logprob lengths: "
            f"{len(spec.sampled_tokens)} vs {len(spec.sampled_logprobs)}."
        )
    return spec.to_tinker_datum(deps, include_action_mask=include_action_mask)
