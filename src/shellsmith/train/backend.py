"""Portable, token-exact training contracts. No accelerator dependency is imported here."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class TokenPrompt:
    tokens: tuple[int, ...]

    @property
    def length(self) -> int:
        return len(self.tokens)

    def to_ints(self) -> list[int]:
        return list(self.tokens)

    def append(self, chunk: Any) -> TokenPrompt:
        return TokenPrompt(self.tokens + tuple(chunk.tokens))


@dataclass(frozen=True)
class TokenSequence:
    prompt_token_ids: tuple[int, ...]
    sampled_tokens: tuple[int, ...]
    behavior_logprobs: tuple[float, ...]
    advantages: tuple[float, ...]
    loss_mask: tuple[float, ...]
    request_id: str
    weights: tuple[float, ...] | None = None
    # Source sampled spans in the complete prompt+tail, before target shifting.
    source_spans: tuple[tuple[str, int, int], ...] = ()

    def __post_init__(self) -> None:
        n = len(self.sampled_tokens)
        if not self.prompt_token_ids or not n or not self.request_id:
            raise ValueError(
                "Token sequence requires prompt, sampled tokens and request identity."
            )
        if any(
            len(values) != n
            for values in (self.behavior_logprobs, self.advantages, self.loss_mask)
        ):
            raise ValueError(
                "Token sequence sampled arrays must have identical lengths."
            )
        if any(
            type(token) is not int or token < 0
            for token in self.prompt_token_ids + self.sampled_tokens
        ):
            raise ValueError("Token sequence IDs must be non-negative integers.")
        if any(
            not math.isfinite(value)
            for values in (self.behavior_logprobs, self.advantages, self.loss_mask)
            for value in values
        ):
            raise ValueError("Token sequence floats must be finite.")
        if any(value not in (0.0, 1.0) for value in self.loss_mask):
            raise ValueError("Token sequence mask must contain zero or one.")
        if any(
            advantage != 0
            for advantage, mask in zip(self.advantages, self.loss_mask)
            if not mask
        ):
            raise ValueError("Masked tokens must have zero advantage.")
        if self.source_spans:
            end, identities = len(self.prompt_token_ids), set()
            covered = [False] * n
            for name, start, stop in self.source_spans:
                if (not isinstance(name, str) or not name or name in identities
                        or type(start) is not int or type(stop) is not int
                        or not end <= start < stop <= len(self.prompt_token_ids) + n):
                    raise ValueError("Source spans must be unique, ordered, and within the target tail.")
                identities.add(name)
                covered[start - len(self.prompt_token_ids):stop - len(self.prompt_token_ids)] = [True] * (stop-start)
                end = stop
            if any(m and not c for m,c in zip(self.loss_mask, covered)):
                raise ValueError("Context outside sampled source spans cannot receive supervision.")
        if self.weights is not None and (
            len(self.weights) != n
            or any(not math.isfinite(value) for value in self.weights)
        ):
            raise ValueError("Cross-entropy weights must be finite and target-aligned.")


@dataclass(frozen=True)
class TokenBatch:
    sequences: tuple[TokenSequence, ...]
    reduction: str = "sum"

    def __post_init__(self) -> None:
        if self.reduction != "sum":
            raise ValueError(
                "CISPO uses sum reduction; rollout normalization is already in advantages."
            )


class PolicySampler(Protocol):
    config: Any

    def sample(
        self, payload: dict[str, Any], *, request_id: str, session_id: str
    ) -> tuple[Any, dict[str, Any]]: ...


class TrainingBackend(Protocol):
    """Resident policy plus optimizer; publication follows the durable checkpoint boundary."""

    sampler: PolicySampler
    policy_identity: dict[str, Any]
    policy_ref: str

    async def sample_tokens(
        self,
        *,
        policy: str,
        prompt_token_ids: list[int],
        max_tokens: int,
        request_id: str,
        **options,
    ) -> dict[str, Any]: ...

    async def token_logprobs(
        self,
        *,
        policy: str,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
        request_id: str,
    ) -> dict[str, Any]: ...

    async def forward_backward(
        self, batch: TokenBatch, *, request_id: str, **options
    ) -> dict[str, Any]: ...

    async def optimizer_step(
        self, *, learning_rate: float, backward_operation: str, request_id: str
    ) -> dict[str, Any]: ...

    async def restore_state(
        self, reference: str, *, request_id: str
    ) -> dict[str, Any]: ...

    async def operation_status(self, request_id: str) -> dict[str, Any]: ...

    async def update(
        self,
        batch: TokenBatch,
        *,
        learning_rate: float,
        clip_low: float,
        clip_high: float,
    ) -> dict[str, Any]: ...

    async def save_checkpoint(
        self, *, kind: str, name: str, metadata: dict[str, Any] | None = None
    ) -> str: ...

    async def publish_policy(self, *, revision: int) -> str: ...

    def load_adapter(self, path: Path) -> None: ...

    def close(self) -> None: ...


def player_profile(config: Any) -> dict[str, Any]:
    """The one resolved profile used by Pi, capture grouping and checkpoint snapshots."""
    from shellsmith.mlx.generation import completion_budget, reasoning_budget, reasoning_effort

    if config.compute.backend == "studio":
        studio = config.studio
        return {
            **asdict(studio),
            "backend": "studio",
            "base_model": studio.model_id,
            "max_tokens": completion_budget(studio),
            "resolved_reasoning_tokens": reasoning_budget(studio),
            "thinking_renderer": f"qwen3_8_{reasoning_effort(studio)}_reasoning",
            "plain_renderer": "qwen3_8_disable_thinking",
        }
    if config.compute.backend == "mlx":
        mlx = config.mlx
        return {
            **asdict(mlx),
            "backend": "mlx",
            "base_model": mlx.model_id,
            "max_tokens": completion_budget(mlx),
            "resolved_reasoning_tokens": reasoning_budget(mlx),
            "thinking_renderer": f"qwen3_8_{reasoning_effort(mlx)}_reasoning",
            "plain_renderer": "qwen3_8_disable_thinking",
        }
    return {
        "backend": "tinker",
        "base_model": config.player.base_model,
        "lora_rank": config.player.lora_rank,
        "thinking_renderer": config.player.thinking_renderer,
        "plain_renderer": config.player.plain_renderer,
        "context_window": 224 * 1024,
        "max_tokens": config.player.max_tokens,
        "compaction_reserve_tokens": 65536,
        "compaction_keep_recent_tokens": 20000,
    }


def profile_hash(profile: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
