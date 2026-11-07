"""Studio adapter and Shellsmith-owned native chat/tool orchestration."""

from __future__ import annotations
from dataclasses import asdict, replace
import threading
from types import SimpleNamespace
import uuid
import asyncio

from studio_compute.client import Client, OperationError
from studio_compute.protocol import digest, bounded_identifier, identifier
from studio_compute.tokenizer import materialize
from shellsmith.mlx.render import Renderer
from shellsmith.mlx.sampler import MLXPlayerSampler
from .studio_initialization import InitializationJournal


def validate_capabilities(settings, capabilities):
    """Reject a model/context mismatch during preflight and backend discovery."""
    if settings.model_id != capabilities.get("model"):
        raise ValueError("Configured model differs from Studio capability.")
    context = capabilities.get("context_window")
    if type(context) is not int or not 0 < settings.context_window <= context:
        raise ValueError("Configured context exceeds the served model capability.")
    if settings.cache_precision not in capabilities.get("cache_precisions", []):
        raise ValueError("Configured cache precision is unsupported by Studio.")


def compact_update_metrics(result):
    """Keep token-level RPC outputs in service receipts, outside trainer telemetry."""
    return {key: value for key, value in result.items() if key not in {"logprobs", "logprob_target_indices"}}


def load_renderer(client, capabilities, *, reasoning_effort="medium"):
    """Validate the served template on the CPU before any rollout starts."""
    bundle = client.tokenizer_bundle()
    if bundle["hash"] != capabilities["tokenizer_hash"]:
        raise ValueError("Tokenizer identity changed during client setup.")
    renderer = Renderer(materialize(bundle), reasoning_effort=reasoning_effort)
    payload = {
        "messages": [{"role": "user", "content": "Readiness check."}],
        "tools": [{"type": "function", "function": {
            "name": "readiness", "description": "Check tool formatting.",
            "parameters": {"type": "object", "properties": {},
                           "additionalProperties": False},
        }}],
    }
    for thinking in (True, False):
        renderer.render(payload, thinking=thinking)
    return renderer


class RemoteSamplingSession:
    """Exact-ID cursor over RPC; contains no GPU or model path."""

    def __init__(self, backend, policy, request_id):
        self.backend, self.policy_ref, self.request_id = backend, policy, request_id
        self.renderer, self.settings = backend.renderer, backend.settings
        self.policy_identity = backend.policy_identity
        self.lock, self.counter = threading.RLock(), 0
        self.owner_operation = getattr(backend, "_owner_operation", None)

    def require_published(self):
        if not self.policy_ref:
            raise RuntimeError("Studio has no published policy.")

    def prepare_prompt(self, ids):
        return list(ids)

    def _call(self, kind, body):
        self.counter += 1
        return self.backend.client.call(
            kind,
            {"policy": self.policy_ref, **body, **(
                {"expected_owner": self.owner_operation} if self.owner_operation is not None else {}
            )},
            request_id=self.backend.sampling_request_id(f"{self.request_id}:pass:{self.counter}"),
        )

    def draw_pass(self, cursor, *, maximum, stop_ids):
        result = self._call(
            "sample",
            {
                "prompt_token_ids": cursor,
                "max_tokens": maximum,
                "stop_token_ids": sorted(stop_ids),
                "cache_precision": self.settings.cache_precision,
            },
        )
        return (
            result["sampled_tokens"],
            result["behavior_logprobs"],
            result["finish_reason"],
            cursor + result["sampled_tokens"],
        )

    def force_tokens(self, cursor, tokens):
        result = self._call(
            "logprobs",
            {
                "prompt_token_ids": cursor,
                "target_token_ids": list(tokens),
                "cache_precision": self.settings.cache_precision,
            },
        )
        return result["logprobs"], cursor + list(tokens)


class StudioPlayerSampler:
    def __init__(self, backend):
        self.backend, self.lock, self.pins = backend, threading.RLock(), {}
        self.config = MLXPlayerSampler(
            RemoteSamplingSession(backend, backend.policy_ref, "validation")
        ).config

    def validate_thinking_budget(self):
        return MLXPlayerSampler(
            RemoteSamplingSession(self.backend, self.backend.policy_ref, "validation")
        ).validate_thinking_budget()

    def pin(self, session_id):
        with self.lock:
            return self.pins.setdefault(session_id, self.backend.policy_ref)

    def release(self, session_id):
        with self.lock:
            self.pins.pop(session_id, None)

    def sample(self, payload, *, request_id, session_id):
        policy = self.pin(session_id)
        from shellsmith.policy.sampler import PlayerRequestError

        try:
            record, assistant = MLXPlayerSampler(
                RemoteSamplingSession(self.backend, policy, request_id)
            ).sample(payload, request_id=request_id, session_id=session_id)
        except OperationError as error:
            raise PlayerRequestError(str(error), status=503) from error
        except PlayerRequestError as error:
            if getattr(error, "record", None):
                error.record = replace(
                    error.record,
                    backend="studio",
                    protocol_row={**error.record.protocol_row, "backend": "studio"},
                )
            raise
        return replace(
            record,
            backend="studio",
            protocol_row={**record.protocol_row, "backend": "studio"},
        ), assistant


class StudioBackend:
    def __init__(self, config, *, client_id=None):
        studio = config.studio
        selected_client = client_id or studio.training_client or config.run.run_id
        identifier(selected_client)
        self.client = Client(
            studio.endpoint,
            selected_client,
            timeout=studio.operation_timeout_seconds,
            sampling_timeout=config.player.prompt_timeout_seconds,
        )
        self.capabilities = self.client.capabilities()
        validate_capabilities(studio, self.capabilities)
        self.renderer = load_renderer(self.client, self.capabilities, reasoning_effort=studio.reasoning_effort)
        self.settings = SimpleNamespace(
            **{**asdict(studio), "model_id": self.capabilities["model"],
               "thinking_anomaly_attempts": config.retry.thinking_anomaly_attempts}
        )
        self.policy_identity = {
            **self.capabilities["identity"],
            "backend": "studio",
            "identity_sha256": digest(self.capabilities["identity"]),
        }
        self.policy_ref, self.revision = "", 0
        self._acquired = False
        self.sampler = StudioPlayerSampler(self)
        self.config = config
        self.operation_prefix = (
            config.run.run_id + ":client:" + digest({
                "client": selected_client, "output": str(config.resolved_log_dir().resolve()),
            })
        )
        self.initializations = InitializationJournal(config.resolved_log_dir(), self.operation_prefix, self.policy_identity)
        self._initialization_target = self._initialization_result = None

    async def initialize(self, checkpoint=None, *, inference=False):
        self.initializations.acquire()
        try:
            return await self._initialize(checkpoint, inference=inference)
        except BaseException:
            self._acquired = False
            self.initializations.release()
            raise

    async def _initialize(self, checkpoint=None, *, inference=False):
        target = checkpoint["path"] if checkpoint else None
        if self._acquired and target == self._initialization_target and self._initialization_result is not None:
            return self._initialization_result
        previous = self.initializations.read()
        if checkpoint is None and previous:
            if previous["phase"] == "pending":
                target = previous["checkpoint"]
            if previous["phase"] == "active":
                target = previous["result"]["checkpoint"]
        if target is not None:
            return await self._initialize_restore(target, previous, requested_target=checkpoint["path"] if checkpoint else None)
        from .config import TrainSection
        learning_rate = TrainSection().learning_rate if inference else self.config.train.learning_rate
        result = await self.client.acall(
            "create",
            {
                "seed": self.config.worlds.seed,
                "expected_identity": self.capabilities["identity"],
                "lora_rank": 16,
                "lora_scale": 2.0,
                "learning_rate": learning_rate,
                "optimizer": asdict(self.config.optimizer),
            },
            request_id=bounded_identifier(self.operation_prefix + ":create"),
        )
        self.revision = result["revision"]
        self._acquired = True
        self._owner_operation = bounded_identifier(self.operation_prefix + ":create")
        self.initializations.owned(self._owner_operation, result)
        self._initialization_target, self._initialization_result = None, result
        return result

    async def _initialize_restore(self, target, previous, *, requested_target):
        from studio_compute.client import OperationError
        if previous and previous["phase"] == "pending":
            attempt = self.initializations.begin(target)
        else:
            proof = previous.get("owner_operation") if previous and previous["phase"] == "active" else bounded_identifier(self.operation_prefix + ":create")
            try:
                current = await asyncio.to_thread(self.client.client_info)
            except ValueError:
                current = None
            owner = None
            if current and current["live"]:
                if current["owner_operation"] != proof:
                    raise RuntimeError("Studio client is live under another acquisition.")
                owner = proof
            attempt = self.initializations.begin(target, owner)
        try:
            if attempt.get("release_id"):
                await self.client.acall("suspend", {"expected_owner": attempt["previous_owner"], "expected_identity": self.capabilities["identity"]}, request_id=attempt["release_id"])
                self._acquired = False
            result = await self.restore_state(target, request_id=attempt["request_id"])
        except OperationError:
            self.initializations.write({**attempt, "phase": "failed"})
            raise
        self.initializations.owned(attempt["request_id"], result, target)
        self._initialization_target, self._initialization_result = requested_target, result
        return result

    @staticmethod
    def _sequences(batch):
        return [asdict(x) | {
            "prompt_token_ids": list(x.prompt_token_ids), "sampled_tokens": list(x.sampled_tokens),
            "behavior_logprobs": list(x.behavior_logprobs), "advantages": list(x.advantages),
            "loss_mask": list(x.loss_mask),
        } for x in batch.sequences]

    def _update_id(self, request_id=None):
        return request_id or getattr(self, "update_request_id", None) or bounded_identifier(
            f"{self.operation_prefix}:update:{self.revision + 1}", limit=160
        )

    def validate_update(self, batch, *, learning_rate, clip_low, clip_high):
        self.client.validate_update(
            self._sequences(batch), learning_rate=learning_rate,
            clip_low=clip_low, clip_high=clip_high, request_id=self._update_id(),
        )

    async def update(
        self, batch, *, learning_rate, clip_low, clip_high, request_id=None
    ):
        result = await self.client.update(
            self._sequences(batch),
            learning_rate=learning_rate,
            clip_low=clip_low,
            clip_high=clip_high,
            request_id=self._update_id(request_id),
            on_phase=getattr(self, "observation_phase", None),
        )
        self.revision = result["optim_metrics"]["optimizer_steps"]
        return result

    async def save_checkpoint(self, *, kind, name, metadata=None):
        result = await self.client.acall(
            "save",
            {"kind": kind, "metadata": metadata or {}},
            request_id=bounded_identifier(self.operation_prefix + ":save:" + name),
        )
        return result["policy"] if kind == "sampler" else result["checkpoint"]

    async def publish_policy(self, *, revision):
        result = await self.client.acall(
            "publish",
            {"revision": revision},
            request_id=bounded_identifier(
                self.operation_prefix + ":publish:" + str(revision) + ":" + uuid.uuid4().hex
            ),
        )
        self.policy_ref = result["policy"]
        return self.policy_ref

    def load_adapter(self, path):
        raise ValueError(
            "Studio policies are opaque remote references; select a published policy reference."
        )

    def close(self):
        # Release accelerator state while retaining the durable checkpoint/RNG
        # cursor. A later operation or coordinator resume can load it lazily.
        if not getattr(self, "_acquired", False):
            self.initializations.release()
            return
        try:
            self.client.call("suspend", {"expected_owner": self._owner_operation, "expected_identity": self.capabilities["identity"]}, request_id=bounded_identifier(self.operation_prefix + ":suspend:" + uuid.uuid4().hex))
        finally:
            self._acquired = False
            self.initializations.release()

    def sampling_request_id(self, request_id):
        # Service receipts have global IDs, not per-client primary keys.
        return bounded_identifier(self.operation_prefix + ":sample:" + request_id)

    async def sample_tokens(
        self, *, policy, prompt_token_ids, max_tokens, request_id, **options
    ):
        return await self.client.acall(
            "sample",
            {
                "policy": policy,
                "prompt_token_ids": prompt_token_ids,
                "max_tokens": max_tokens,
                **options,
                "expected_owner": self._owner_operation,
            },
            request_id=self.sampling_request_id(request_id),
        )

    async def token_logprobs(
        self, *, policy, prompt_token_ids, target_token_ids, request_id
    ):
        return await self.client.acall(
            "logprobs",
            {
                "policy": policy,
                "prompt_token_ids": prompt_token_ids,
                "target_token_ids": target_token_ids,
                "expected_owner": self._owner_operation,
            },
            request_id=self.sampling_request_id(request_id),
        )

    async def forward_backward(self, batch, *, request_id, **options):
        return await self.client.acall(
            "forward_backward",
            {
                "sequences": [asdict(x) for x in batch.sequences],
                "reduction": "sum",
                **options,
            },
            request_id=request_id,
        )

    async def optimizer_step(self, *, learning_rate, backward_operation, request_id):
        return await self.client.acall(
            "optim_step",
            {"learning_rate": learning_rate, "backward_operation": backward_operation},
            request_id=request_id,
        )

    async def restore_state(self, reference, *, request_id):
        identifier(request_id)
        self.initializations.acquire()
        if getattr(self, "_acquired", False):
            import asyncio
            try:
                operation = await asyncio.to_thread(self.client.operation, request_id)
            except ValueError:
                operation = None
            # A committed retry must not release the client as a new side effect.
            if not operation or operation["status"] != "succeeded":
                await self.client.acall("suspend", {"expected_owner": self._owner_operation, "expected_identity": self.capabilities["identity"]},
                                        request_id=bounded_identifier(self.operation_prefix + ":restore-release:" + digest({"restore": request_id})))
                self._acquired = False
        result = await self.client.acall(
            "restore", {"checkpoint": reference, "expected_identity": self.capabilities["identity"]}, request_id=request_id
        )
        self.revision = result["revision"]
        self._acquired = True
        self._owner_operation = request_id
        previous = self.initializations.read()
        if not previous or previous["phase"] != "pending":
            self.initializations.owned(request_id, result, reference)
        return result

    async def operation_status(self, request_id):
        import asyncio

        return await asyncio.to_thread(self.client.operation, request_id)
