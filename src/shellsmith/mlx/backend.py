"""Optional resident MLX session behind Shellsmith's portable training contract."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import gc
import math
from pathlib import Path
import threading
import time
from typing import Any

from .checkpoint import publish_checkpoint, verify_checkpoint
from .identity import digest, resolve_identity
from .readiness import model_metadata, require_ready
from .profiles import MEMORY_GIB, checkpoint_layers


@dataclass
class DecodeCursor:
    states: list[Any]
    logits: Any
    position: int


class MLXSession:
    """One resident model and one serialized inference/replay/optimizer stream.

    Between an update and its durable full checkpoint, sampling is disabled.
    The trainer publishes the pending sampler only after committing its cursor
    and metrics. All inference caches are request-local and discarded.
    """

    def __init__(self, config, *, checkpoint_root, checkpoint=None):
        require_ready(config.mlx)
        self.settings = config.mlx
        self.thinking_anomaly_attempts = config.retry.thinking_anomaly_attempts
        self.policy_identity = resolve_identity(self.settings)
        self.checkpoint_root = Path(checkpoint_root).expanduser().absolute()
        self.lock = threading.RLock()
        self.closed, self._pending, self._inference_only = False, False, False
        self.policy_version = 0
        self.policy_ref = "mlx-base:" + self.policy_identity["identity_sha256"]
        self._pending_ref = None
        self._durable_version = None
        self.mx = self.model = self.lm = self.adapter = self.optimizer = self.sampler = self.rng = None
        self.replay_progress = None
        # Verify recovery integrity and all frozen identities before GPU allocation.
        document = None
        if checkpoint is not None:
            document = verify_checkpoint(checkpoint["path"], kind="state", expected_identity=self.policy_identity)
            if checkpoint.get("policy_identity", self.policy_identity) != self.policy_identity:
                raise ValueError("Trainer checkpoint row has a different immutable MLX policy identity.")
            if checkpoint.get("sampler_path", document["sampler_path"]) != document["sampler_path"]:
                raise ValueError("Trainer checkpoint row and native checkpoint reference different sampler policies.")
            for name, value in (document.get("trainer_position") or {}).items():
                if checkpoint.get(name) != value:
                    raise ValueError(f"Trainer checkpoint row and native checkpoint differ at recovery field {name!r}.")
        try:
            self._initialize(config, document, checkpoint)
        except BaseException:
            # Model loading and adapter/restore validation can fail after GPU
            # allocation, before callers receive a session to register cleanup.
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _initialize(self, config, document, checkpoint):
        import mlx.core as mx
        from mlx_vlm import load

        from .adapter import Adapter
        from .optimizer import MasterAdam
        from .render import Renderer
        from .sampler import MLXPlayerSampler

        self.mx = mx
        mx.set_memory_limit(MEMORY_GIB * 1024 ** 3)
        mx.set_cache_limit(2 * 1024 ** 3)
        _, _, kinds = model_metadata(self.settings.model_path)
        self.renderer = Renderer(self.settings.model_path, reasoning_effort=self.settings.reasoning_effort)
        self.model, _ = load(str(Path(self.settings.model_path).expanduser()), lazy=False, trust_remote_code=False)
        self.model.freeze()
        self.lm = self.model.language_model
        self.adapter = Adapter(self.lm, kinds)
        self.adapter.load(self.adapter.initial(config.worlds.seed))
        self.optimizer = MasterAdam(self.adapter, config.train.learning_rate)
        self.rng = mx.random.key(config.worlds.seed)
        self.sampler = MLXPlayerSampler(self)
        self.sampler.validate_thinking_budget()
        mx.eval(self.model.parameters(), self.rng)
        if document is not None:
            self._restore(checkpoint["path"], document)

    def require_published(self):
        from shellsmith.policy.sampler import PlayerRequestError

        if self.closed:
            raise PlayerRequestError("MLX session is closed.", status=503)
        if self._pending:
            raise PlayerRequestError("MLX policy update is awaiting its durable publication boundary.", status=503)

    def _head(self, hidden):
        decoder = self.lm.model
        head = decoder.embed_tokens.as_linear if self.lm.args.tie_word_embeddings else self.lm.lm_head
        return head(decoder.norm(hidden)).astype(self.mx.float32)

    def _advance(self, ids, states, position):
        from .ops import segment_layer

        mx = self.mx
        ids = tuple(ids)
        if not ids or position + len(ids) > self.settings.context_window:
            raise ValueError("Native MLX decoder cannot advance beyond the admitted context.")
        hidden = self.lm.model.embed_tokens(mx.array([ids], dtype=mx.int32)).astype(mx.bfloat16)
        positions = mx.broadcast_to(mx.arange(position, position + len(ids), dtype=mx.int32)[None, None],
                                     (3, 1, len(ids)))
        carried_states = []
        for layer, incoming in zip(self.lm.layers, states):
            hidden, *carried = segment_layer(layer, hidden, [value for _, value in incoming], positions)
            if layer.is_linear:
                values = carried
            else:
                values = [mx.concatenate([*[value for _, value in incoming][0::2], carried[0]], axis=2),
                          mx.concatenate([*[value for _, value in incoming][1::2], carried[1]], axis=2)] if incoming else carried
            mx.eval(hidden, values)
            carried_states.append([(None, value) for value in values])
        logits = self._head(hidden[:, -1:])[:, -1]
        mx.eval(logits)
        return DecodeCursor(carried_states, logits, position + len(ids))

    def prepare_prompt(self, ids):
        from .ops import initial_states
        from .replay import SEGMENT_TOKENS

        self.model.eval()
        states = initial_states(self.lm.layers, self.mx.bfloat16)
        cursor = None
        for start in range(0, len(ids), SEGMENT_TOKENS):
            cursor = self._advance(ids[start:start + SEGMENT_TOKENS], states, start)
            states = cursor.states
        if cursor is None:
            raise ValueError("MLX generation prompt cannot be empty.")
        return cursor

    def _logprobs(self, logits):
        mx = self.mx
        return logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)

    def draw_pass(self, cursor, *, maximum, stop_ids):
        mx = self.mx
        if type(maximum) is not int or maximum < 1 or cursor.position + maximum > self.settings.context_window:
            raise ValueError("MLX sampling pass exceeds its admitted token budget.")
        tokens, probabilities, reason = [], [], "length"
        for _ in range(maximum):
            logprobs = self._logprobs(cursor.logits)
            self.rng, key = mx.random.split(self.rng)
            drawn = mx.random.categorical(logprobs, key=key)[0]
            selected = logprobs[0, drawn]
            mx.eval(drawn, selected, self.rng)
            token, probability = int(drawn.item()), float(selected.item())
            if not math.isfinite(probability):
                raise FloatingPointError("MLX sampled a nonfinite behavior log probability.")
            tokens.append(token)
            probabilities.append(probability)
            # The cursor includes every drawn token, including stop controls.
            # Continuation therefore never needs decoded-text re-tokenization.
            cursor = self._advance((token,), cursor.states, cursor.position)
            if token in stop_ids:
                reason = "stop"
                break
        return tokens, probabilities, reason, cursor

    def force_tokens(self, cursor, tokens):
        probabilities = []
        for token in tokens:
            selected = self._logprobs(cursor.logits)[0, token]
            probability = float(selected.item())
            if not math.isfinite(probability):
                raise FloatingPointError("MLX could not score a forced thinking boundary token.")
            probabilities.append(probability)
            cursor = self._advance((token,), cursor.states, cursor.position)
        return probabilities, cursor

    async def update(self, batch, *, learning_rate, clip_low, clip_high):
        return await asyncio.to_thread(self._update, batch, learning_rate, clip_low, clip_high)

    def _update(self, batch, learning_rate, clip_low, clip_high):
        from .cispo import cispo_loss, validate_clip
        from .replay import Replay

        validate_clip(clip_low, clip_high)
        with self.lock:
            self.require_published()
            if self._inference_only:
                raise RuntimeError("An adapter-only inference session cannot resume training; use a full checkpoint.")
            if batch.reduction != "sum" or not batch.sequences:
                raise ValueError("MLX update requires a nonempty sum-reduced token batch.")
            for sequence in batch.sequences:
                if len(sequence.prompt_token_ids) + len(sequence.sampled_tokens) > self.settings.context_window:
                    raise ValueError("Accepted MLX training sequence exceeds its explicit context profile.")
                if any(token >= self.lm.args.vocab_size for token in sequence.prompt_token_ids + sequence.sampled_tokens):
                    raise ValueError("Accepted MLX capture contains a token outside this artifact's vocabulary.")
            started = time.monotonic()
            self.model.train()
            self.optimizer.zero()
            loss, replay_seconds, predicted_tokens = 0.0, 0.0, 0
            try:
                replay = Replay(self.lm, memory_gib=MEMORY_GIB, progress=self.replay_progress,
                                checkpoint_layers=checkpoint_layers(self.settings.profile))
                for sequence in batch.sequences:
                    objective = lambda current: cispo_loss(current, sequence.behavior_logprobs, sequence.advantages,
                                                          sequence.loss_mask, clip_low=clip_low, clip_high=clip_high)
                    _, value, gradients, stats = replay.run(sequence.prompt_token_ids, sequence.sampled_tokens, objective)
                    scalar = float(value.item())
                    if not math.isfinite(scalar):
                        raise FloatingPointError("MLX update produced a nonfinite CISPO loss.")
                    self.optimizer.accumulate(gradients)
                    loss += scalar
                    replay_seconds += stats["replay_seconds"]
                    predicted_tokens += sum(sequence.loss_mask)
                    del gradients
                norm = self.optimizer.step(learning_rate=learning_rate)
                self._pending = True
                self._pending_ref, self._durable_version = None, None
            finally:
                self.optimizer.zero()
                self.model.eval()
                self.mx.clear_cache()
            return {"forward_metrics": {"cispo_loss": loss, "loss": loss, "reduction": "sum",
                                        "sequences": len(batch.sequences), "loss_tokens": int(predicted_tokens)},
                    "optim_metrics": {"gradient_norm": norm, "learning_rate": learning_rate,
                                      "optimizer_steps": self.optimizer.steps},
                    "mlx_metrics": {"replay_seconds": replay_seconds, "update_seconds": time.monotonic() - started,
                                    "peak_memory_bytes": self.mx.get_peak_memory(), "policy_version": self.optimizer.steps}}

    async def save_checkpoint(self, *, kind, name, metadata=None):
        return await asyncio.to_thread(self._save_checkpoint, kind, name, metadata)

    def _save_checkpoint(self, kind, name, metadata):
        if not isinstance(name, str) or not name or Path(name).name != name or name.startswith("."):
            raise ValueError("MLX checkpoint name must be a nonempty local artifact name.")
        with self.lock:
            if self.closed or self._inference_only:
                raise RuntimeError("MLX session cannot publish training checkpoints.")
            mx = self.mx
            tensors, optimizer = self.optimizer.state()

            def writer(folder):
                mx.save_safetensors(str(folder / "adapters.safetensors"), self.adapter.state(),
                                    metadata={"rank": "16", "scale": "2.0"})
                if kind == "state":
                    mx.save_safetensors(str(folder / "optimizer.safetensors"), tensors)
                    mx.save_safetensors(str(folder / "rng.safetensors"), {"key": self.rng})

            sampler_path = self._pending_ref or self.policy_ref
            if kind == "state" and not Path(sampler_path).is_absolute():
                raise RuntimeError("Full MLX checkpoints need their already-saved matching sampler artifact.")
            path = publish_checkpoint(self.checkpoint_root / name, writer=writer, kind=kind,
                policy_identity=self.policy_identity, policy_version=self.optimizer.steps,
                sampler_path=sampler_path if kind == "state" else None,
                optimizer=optimizer if kind == "state" else None, metadata=metadata)
            if kind == "sampler":
                self._pending_ref = path
            else:
                self._durable_version = self.optimizer.steps
            return path

    async def publish_policy(self, revision):
        with self.lock:
            if (type(revision) is not int or revision != self.optimizer.steps or not self._pending_ref
                    or self._durable_version != revision):
                raise RuntimeError("MLX publication requires matching sampler, full checkpoint, and committed update revision.")
            self.policy_version, self.policy_ref = revision, self._pending_ref
            self._pending_ref, self._pending = None, False
            self.mx.clear_cache()
            return self.policy_ref

    def _restore(self, path, document):
        mx = self.mx
        folder = Path(path)
        adapters = mx.load(str(folder / "adapters.safetensors"))
        optimizer = mx.load(str(folder / "optimizer.safetensors"))
        rng = mx.load(str(folder / "rng.safetensors"))
        self.adapter.validate(adapters)
        state = self.optimizer.validated_state(optimizer, document["optimizer"])
        if set(rng) != {"key"} or rng["key"].dtype != mx.uint32 or rng["key"].shape != (2,):
            raise ValueError("MLX checkpoint RNG state is invalid.")
        if any(not bool(mx.array_equal(value, state["master"][name].astype(mx.bfloat16)).item())
               for name, value in adapters.items()):
            raise ValueError("MLX checkpoint adapter differs from its FP32 optimizer master weights.")
        # Also compare the separate sampler artifact before publishing recovered weights.
        sampler = mx.load(str(Path(document["sampler_path"]) / "adapters.safetensors"))
        if set(sampler) != set(adapters) or any(not bool(mx.array_equal(sampler[name], value).item())
                                              for name, value in adapters.items()):
            raise ValueError("MLX full checkpoint and matching sampler contain different adapter weights.")
        self.optimizer.load(optimizer, document["optimizer"])
        self.rng = rng["key"]
        self.policy_version, self.policy_ref = document["policy_version"], document["sampler_path"]
        self._durable_version, self._pending, self._pending_ref = self.policy_version, False, None
        mx.eval(self.rng, self.adapter.state())

    def load_adapter(self, path):
        """Load an identity-verified sampler export for inference, never optimizer resume."""
        with self.lock:
            self.require_published()
            doc = verify_checkpoint(path, kind="sampler", expected_identity=self.policy_identity)
            weights = self.mx.load(str(Path(path) / "adapters.safetensors"))
            self.adapter.load(weights)
            self.policy_version = doc["policy_version"]
            self.policy_ref = "mlx-adapter:" + digest({"policy_identity": self.policy_identity,
                                                       "policy_version": self.policy_version, "files": doc["files"]})
            self._inference_only = True
            self.mx.clear_cache()
            return self.policy_ref

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.sampler = None
            self.model = self.lm = self.adapter = self.optimizer = None
            self.rng = None
            gc.collect()
            if self.mx is not None:
                self.mx.clear_cache()


async def create_session(config, *, checkpoint=None, checkpoint_root):
    construction = asyncio.create_task(
        asyncio.to_thread(MLXSession, config, checkpoint=checkpoint, checkpoint_root=checkpoint_root))
    try:
        return await asyncio.shield(construction)
    except BaseException:
        # A cancelled await cannot stop the construction thread. Keep the
        # caller's GPU lease until it settles and dispose any completed session.
        while not construction.done():
            try:
                await asyncio.shield(construction)
            except asyncio.CancelledError:
                continue
            except BaseException:
                break
        try:
            session = construction.result()
        except BaseException:
            pass
        else:
            try:
                session.close()
            except BaseException:
                pass
        raise
