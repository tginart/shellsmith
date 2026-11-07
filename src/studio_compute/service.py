"""HTTP-neutral service with one GPU owner and cooperative per-client replay."""

from __future__ import annotations

from collections import deque
import hashlib
from pathlib import Path
import shutil
import threading
import time
from typing import Protocol

from .protocol import Conflict, digest, identifier, validate_request, MAX_REQUEST_BYTES
from .storage import Store
from .inference_metrics import InferenceMetrics

CONTINUOUS_INFERENCE = ("inference",)
# Replay yields thousands of times per update. Commit progress durably on the
# first yield and then at this interval; polls read the in-memory value.
PROGRESS_COMMIT_SECONDS = 30.0


class Engine(Protocol):
    """All methods, including construction, execute on the GPU owner thread."""

    identity: dict
    capabilities: dict

    def create(self, client, body): ...
    def restore(self, client, folder, state): ...
    def write_state(self, client, folder): ...
    def work(self, client, kind, body): ...
    def policy(self, reference, folder, state, *, adapter_digest): ...
    def close_client(self, client): ...
    def abort(self, client): ...
    def close(self): ...


class ClientOwnershipConflict(Conflict):
    """A rejected precondition must never enter optimizer rollback."""


class Service:
    def __init__(self, root, engine_factory, *, checkpoint_retention=0,
                 recovery_state=None, checkpoint_reserve_gib=16):
        if type(checkpoint_retention) is not int or checkpoint_retention < 0 or checkpoint_retention == 1:
            raise ValueError("Checkpoint retention must be zero or at least two revisions.")
        self.checkpoint_retention = checkpoint_retention
        self.recovery_state = Path(recovery_state) if recovery_state else None
        self.checkpoint_reserve_bytes = int(checkpoint_reserve_gib * (1 << 30))
        self.store = Store(root)
        self.condition = threading.Condition(threading.RLock())
        self.queues, self.round_robin, self.active = {}, deque(), {}
        self.progress_commits = {}
        self.resident_clients = set()
        self.live_clients = set()
        self.cancellations = set()
        self.stopping, self.ready, self.start_error = False, threading.Event(), None
        self.fatal_error = None
        self.engine_factory = engine_factory
        self.owner = threading.Thread(
            target=self._owner, name="studio-gpu-owner", daemon=True
        )
        self.owner.start()
        self.ready.wait()
        if self.start_error:
            self.store.close()
            raise self.start_error
        self.inference_metrics.start_logging(
            self.store.root / "inference-telemetry.jsonl",
            model=self.engine.identity.get("model"),
            max_inference_batch=self.capabilities.get("max_inference_batch", 1),
        )

    def submit(self, request_id, client, kind, body):
        identifier(request_id)
        identifier(client)
        if not isinstance(body, dict):
            raise ValueError("Operation body must be an object.")
        with self.condition:
            if self.stopping:
                raise RuntimeError("Service is stopping.")
            validate_request(kind, body, self.capabilities)
            if "expected_identity" in body and body["expected_identity"] != self.engine.identity:
                raise Conflict("Studio compute identity changed before submission.")
            # Check deduplication before checking mutable client state.
            op, fresh = self.store.submit(request_id, client, kind, body)
            if fresh:
                if kind == "sample" and not body.get("target_token_ids"):
                    self.inference_metrics.submit(request_id, client=client)
                lane = (
                    CONTINUOUS_INFERENCE
                    if kind in ("sample", "logprobs") and self.capabilities.get("continuous_batching")
                    else "inference:" + client
                    if kind in ("sample", "logprobs")
                    else "training:" + client
                )
                if lane not in self.queues:
                    self.queues[lane] = deque()
                    self.round_robin.append(lane)
                self.queues[lane].append((request_id, client, kind, body))
                self.condition.notify_all()
            return op

    def cancel(self, request_id: str) -> dict:
        """
        Purpose: Cancel queued or active sampling at the next materialized boundary.
        Input: A durable sample ID. Output: Its current receipt.
        Note: Poll the same operation ID for the terminal interrupted receipt.
        """
        with self.condition:
            op = self.store.operation(identifier(request_id))
            if op["kind"] != "sample" or not self.capabilities.get("continuous_batching"):
                raise ValueError("Cancellation requires continuous sampling.")
            if op["status"] in ("queued", "running"):
                self.cancellations.add(request_id)
                self.condition.notify_all()
            return op

    def _create_continuous(self, request_id: str, item: dict):
        """
        Purpose: Prepare RNG once, then retry resource admission without another draw.
        Input: A queued sample or score. Output: A request, or a temporary memory deferral.
        """
        from .inference import create_request

        self._validate_inference_owner(item["client"], item["body"])
        if "prepared" not in item:
            if item["kind"] == "logprobs":
                item["body"] = {"seed": 0, **item["body"]}
            try:
                item["prepared"] = self._inference_body(request_id, item["client"], item["body"])
            except MemoryError:
                # A cold RNG owner can wait for live replay. Once resident,
                # key allocation errors must fail without a second RNG draw.
                if (hasattr(self.engine, "next_rng")
                    and item["client"] not in self.resident_clients
                    and self.engine.reserved):
                    return None
                raise
        request = create_request(self.engine, request_id, item["prepared"])
        if request is not None:
            with self.condition:
                self.store.running(request_id)
            if item["kind"] == "sample":
                self.inference_metrics.start(request_id)
        return request

    def _finish_continuous(self, request) -> None:
        from .inference import request_result

        result = request_result(self.engine, request)
        with self.condition:
            self.store.finish(request.request_id, result)
        self.inference_metrics.finish(request.request_id, tokens=len(request.tokens))

    def _fail_continuous(self, request_id: str, error: Exception) -> None:
        with self.condition:
            self.store.fail(request_id, f"{type(error).__name__}: {error}")
        self.inference_metrics.finish(request_id, failed=True)

    def _continuous_samples(self, first):
        """
        Purpose: Bridge durable operations to one persistent inference queue.
        Input: Existing sample RPCs. Output: Per-request receipts at token boundaries.
        Note: The owner alternates this lane with the existing training lanes.
        """
        outstanding = set()

        def finish(request):
            self._finish_continuous(request)
            outstanding.discard(request.request_id)

        def fail(request_id, error):
            self._fail_continuous(request_id, error)
            outstanding.discard(request_id)

        scheduler = self.engine.inference_scheduler(
            create=self._create_continuous, finish=finish, fail=fail,
        )
        items = [first]
        try:
            while True:
                with self.condition:
                    queue = self.queues[CONTINUOUS_INFERENCE]
                    items.extend(queue)
                    queue.clear()
                    for request_id, client, kind, body in items:
                        outstanding.add(request_id)
                        scheduler.enqueue(request_id, {"client": client, "kind": kind, "body": body})
                    items.clear()
                    canceled = self.cancellations & outstanding
                    self.cancellations.difference_update(canceled)
                for request_id in canceled:
                    scheduler.cancel(request_id)
                    with self.condition:
                        self.store.fail(request_id, "Sampling canceled at a token boundary", "interrupted")
                    self.inference_metrics.finish(request_id, failed=True)
                    outstanding.discard(request_id)
                if not scheduler.pending and not scheduler.active:
                    return
                progress = scheduler.step()
                for size in progress["batch_sizes"]:
                    if size:
                        self.inference_metrics.batch(size)
                with self.condition:
                    for request_id, count, ended in zip(
                        progress["request_ids"], progress["generated_tokens"],
                        progress["finished_requests"],
                    ):
                        if not ended:
                            self.store.observe_progress(request_id, {"phase": "decode", "tokens": count})
                    if progress["prefill"] is not None and progress["prefill"][0] in scheduler.active:
                        request_id, offset = progress["prefill"]
                        self.store.observe_progress(request_id, {"phase": "prefill", "tokens": offset})
                    self.condition.notify_all()
                    if not progress["request_ids"] and progress["prefill"] is None and not self.round_robin:
                        self.condition.wait(timeout=.05)
                yield progress
        finally:
            scheduler.close()
            for request_id in outstanding:
                with self.condition:
                    if self.store.operation(request_id)["status"] in ("queued", "running"):
                        self.store.fail(request_id, "Service stopped at a token boundary", "interrupted")
                self.inference_metrics.finish(request_id, failed=True)

    def policy_info(self, reference):
        """Read committed publication metadata without allocating or submitting work."""
        key = reference.removeprefix("studio-policy:")
        if not reference.startswith("studio-policy:") or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Malformed policy reference.")
        with self.condition:
            policy = self.store.states("policies").get(reference)
            if policy is None:
                raise KeyError("Unknown unpublished policy reference.")
            if self._retired(policy["checkpoint"]):
                return {"policy": reference, "revision": policy["revision"],
                        "identity": self.capabilities["identity"], "available": False,
                        "reason": "checkpoint weights were pruned"}
            return {"policy": reference, "revision": policy["revision"],
                    "identity": self.capabilities["identity"]}

    def client_info(self, client):
        with self.condition:
            state = self.clients.get(client)
            if state is None:
                raise KeyError(client)
            return {"client": client, "live": client in self.live_clients,
                    "owner_operation": state.get("owner_operation")}

    def telemetry(self, *, client_id: str | None = None) -> dict:
        """
        Purpose: Expose existing operation progress without a call on the GPU owner.
        Input: The active operation map and durable progress receipts.
        Output: Inference counters and bounded train-operation progress.
        Note: This response contains no operation input or token replay array.
        """
        with self.condition:
            operations = {}
            for request_id, client, kind, _ in self.active.values():
                if client_id is not None and client != client_id:
                    continue
                if kind in {"forward_backward", "optim_step", "save", "publish", "restore"}:
                    row = self.store.operation(request_id)
                    operations[request_id] = {
                        "kind": kind, "status": row["status"], "progress": row["progress"],
                    }
            value = {**self.inference_metrics.readout(), "training_operations": operations,
                     "scope": "service", "memory_peak_scope": "process_lifetime",
                     "max_inference_batch": self.capabilities.get("max_inference_batch", 1)}
            if client_id is not None:
                identifier(client_id)
                value["client_metrics"] = self.inference_metrics.client_readout(client_id)
            return value

    def operation(self, request_id):
        with self.condition:
            return self.store.operation(request_id)

    def wait(self, request_id, timeout=30):
        until = time.monotonic() + timeout
        with self.condition:
            while True:
                op = self.store.operation(request_id)
                if self.stopping and op["status"] in ("queued", "running"):
                    return {
                        **op,
                        "status": "interrupted",
                        "error": self.fatal_error or "Service is stopping",
                    }
                if (
                    op["status"] not in ("queued", "running")
                    or time.monotonic() >= until
                ):
                    return op
                self.condition.wait(max(0, until - time.monotonic()))

    def _checkpoint(self, request_id, client, state, result, *, writer=None):
        if self.checkpoint_retention and client in self.clients:
            folder, document = self.store.manifest(self.clients[client]["checkpoint"])
            required = sum((folder / name).stat().st_size for name in document["files"])
            if shutil.disk_usage(self.store.root).free < required + self.checkpoint_reserve_bytes:
                raise OSError("Insufficient disk for the next atomic checkpoint and reserve.")
        with self.condition:
            intent = self.store.checkpoint_intent(request_id)
        state, result = self.store.checkpoint(
            request_id,
            writer or (lambda folder: self.engine.write_state(client, folder)),
            identity=self.engine.identity,
            state=state,
            result=result,
            intent=intent,
        )
        with self.condition:
            self.store.finish(request_id, result, client_state=state)
            self.clients[client] = state
        self._retain_checkpoints()
        return result

    def _retired(self, reference):
        folder, _ = self.store.manifest(reference, self.engine.identity)
        return (folder / "retired.json").exists()

    def _retain_checkpoints(self):
        """
        Purpose: Prune old payloads after replacement revisions pass full verification.
        Input: Committed Store state and any trainer recovery reference.
        Output: Durable retirement markers and bounded full-state payloads.
        Note: Recheck queued work under the journal lock before each retirement batch.
        """
        if not self.checkpoint_retention:
            return
        from . import retention
        from .storage import atomic_json

        with self.condition:
            candidates, retained = retention.plan(
                self.store, self.checkpoint_retention, self.recovery_state
            )
        if not candidates:
            return
        for reference in retained:
            self.store.verify(reference, self.engine.identity)
        with self.condition:
            current, retained = retention.plan(
                self.store, self.checkpoint_retention, self.recovery_state
            )
            candidates &= current
            retention.mark(self.store, candidates, retained=retained)
        retention.cleanup(self.store)
        atomic_json(self.store.root / "retention.json", {
            "keep_revisions": self.checkpoint_retention,
            "retired_checkpoints": sorted(candidates),
            "retained_checkpoints": sorted(retained),
        })

    def _policy_reference(self, client, state):
        _, document = self.store.verify(state["checkpoint"], self.engine.identity)
        factors = document["files"].get(
            "adapters.safetensors", digest(document["files"])
        )
        return "studio-policy:" + digest(
            {
                "client": client,
                "revision": state["revision"],
                "factors": factors,
                "identity": self.engine.identity,
            }
        )

    def _ensure_client(self, client):
        if client not in self.resident_clients:
            state = self.clients[client]
            folder, _ = self.store.verify(state["checkpoint"], self.engine.identity)
            self.engine.restore(client, folder, state)
            self.resident_clients.add(client)

    def _validate_inference_owner(self, client: str, body: dict) -> None:
        if client not in self.clients or self.clients[client].get("closed"):
            raise ValueError("Unknown or closed training client.")
        if "expected_owner" in body and (
            client not in self.live_clients or body["expected_owner"] != self.clients[client].get("owner_operation")
        ):
            raise ClientOwnershipConflict("Sampling acquisition is no longer live.")

    def _inference_body(self, request_id, client, body):
        self._validate_inference_owner(client, body)
        with self.condition:
            policy = self.store.states("policies").get(body["policy"])
        if policy is None:
            raise ValueError("Unknown unpublished policy reference.")
        if self._retired(policy["checkpoint"]):
            raise ValueError("Published policy weights were pruned: " + body["policy"])
        body = {**body, "revision": policy["revision"], "operation_id": request_id}
        if "seed" not in body:
            if hasattr(self.engine, "next_rng"):
                self._ensure_client(client)
            with self.condition:
                state = self.clients[client]
                counter = state.get("rng_counter", 0)
                if hasattr(self.engine, "next_rng"):
                    next_key, draw_key = self.engine.next_rng(client)
                    body["rng_key"] = draw_key
                    state = {**state, "rng_key": next_key, "rng_counter": counter + 1}
                else:
                    body["seed"] = int.from_bytes(
                        hashlib.sha256(f"{state['seed']}:{counter}".encode()).digest()[
                            :4
                        ],
                        "big",
                    )
                    state = {**state, "rng_counter": counter + 1}
                self.store.assign_rng(request_id, client, state, body)
                self.clients[client] = state
        return body

    def _execute_batch(self, items):
        valid, bodies = [], []
        for request_id, client, _, body in items:
            try:
                bodies.append(self._inference_body(request_id, client, body))
                valid.append(request_id)
            except Exception as error:
                with self.condition:
                    self.store.fail(request_id, str(error))
                self.inference_metrics.finish(request_id, failed=True)
        if not valid:
            return
        generated = sum(not body.get("target_token_ids") for body in bodies)
        if generated:
            self.inference_metrics.batch(generated)
        try:
            results = yield from self._observe_samples(valid, self.engine.sample_batch(bodies))
            with self.condition:
                for request_id, result in zip(valid, results):
                    self.store.finish(request_id, result)
        except Exception as error:
            with self.condition:
                for request_id in valid:
                    self.store.fail(request_id, str(error))

    def _observe_samples(self, request_ids, generator):
        """Measure individual batch members, never multiply step count by B.

        Final receipts reconcile engines that emit coarser progress. Native
        decode progress supplies exact counts and EOS state for every member.
        """
        succeeded = False
        try:
            while True:
                try:
                    progress = next(generator)
                except StopIteration as done:
                    results = done.value if isinstance(done.value, list) else [done.value]
                    if len(results) != len(request_ids):
                        raise RuntimeError("Sampling result count differs from admitted requests.")
                    for request_id, result in zip(request_ids, results):
                        self.inference_metrics.finish(
                            request_id, tokens=result.get("completion_tokens", len(result.get("sampled_tokens", [])))
                        )
                    succeeded = True
                    return done.value
                counts = progress.get("generated_tokens")
                if counts is None and progress.get("phase") == "decode" and progress.get("tokens") is not None:
                    counts = [progress["tokens"]]
                if isinstance(counts, int):
                    counts = [counts]
                if counts is not None:
                    finished = progress.get("finished_requests", [False] * len(counts))
                    for request_id, tokens, ended in zip(request_ids, counts, finished):
                        self.inference_metrics.progress(request_id, tokens, finished=ended)
                yield progress
        finally:
            try:
                generator.close()
            finally:
                if not succeeded:
                    for request_id in request_ids:
                        self.inference_metrics.finish(request_id, failed=True)

    def _record_progress(self, request_id, progress):
        """Commit the first and then periodic progress; keep the rest in memory."""
        now = time.monotonic()
        last = self.progress_commits.get(request_id)
        if last is None or now - last >= PROGRESS_COMMIT_SECONDS:
            self.store.running(request_id, progress)
            self.progress_commits[request_id] = now
        else:
            self.store.observe_progress(request_id, progress)

    def _execute(self, request_id, client, kind, body):
        body = {**body, "operation_id": request_id}
        if kind == "create":
            if client in self.clients:
                raise Conflict(
                    "Training client already exists; use its original creation ID or restore into a new client."
                )
            metadata = self.engine.create(client, body)
            self.resident_clients.add(client)
            state = {
                "revision": 0,
                "seed": body.get("seed", 0),
                "closed": False,
                "owner_operation": request_id,
                **metadata,
            }
            result = self._checkpoint(
                request_id, client, state, {"client": client, "revision": 0}
            )
            self.live_clients.add(client)
            return result
        if kind == "restore":
            if client in self.live_clients:
                raise ClientOwnershipConflict("Training client is live; suspend or close it before restore.")
            folder, document = self.store.verify(
                body.get("checkpoint"), self.engine.identity
            )
            self.engine.restore(client, folder, document["state"])
            self.resident_clients.add(client)
            state = {**document["state"], "closed": False, "owner_operation": request_id}
            result = self._checkpoint(
                request_id,
                client,
                state,
                {"client": client, "revision": state["revision"]},
                writer=(lambda target: self.engine.write_snapshot(client, target, folder))
                if hasattr(self.engine, "write_snapshot") else None,
            )
            self.live_clients.add(client)
            return result
        if kind == "suspend" and (client not in self.clients or self.clients[client].get("closed")):
            result = {"suspended": True, "closed": self.clients.get(client, {}).get("closed", False)}
            with self.condition:
                self.store.finish(request_id, result)
            return result
        if client not in self.clients or self.clients[client].get("closed"):
            raise ValueError("Unknown or closed training client.")
        state = self.clients[client]
        if kind in ("close", "suspend"):
            if body.get("expected_owner") is not None and body["expected_owner"] != state.get("owner_operation"):
                raise ClientOwnershipConflict("Client acquisition changed before release.")
            if client in self.resident_clients:
                self.engine.close_client(client)
                self.resident_clients.discard(client)
            with self.condition:
                state = {**state, "closed": kind == "close"}
                result = {"closed": state["closed"], "suspended": kind == "suspend"}
                self.store.finish(request_id, result, client_state=state)
                self.clients[client] = state
                self.live_clients.discard(client)
            return result
        if kind == "publish":
            if body.get("revision") != state["revision"]:
                raise Conflict("Publication must match the durable optimizer revision.")
            folder, document = self.store.verify(
                state["checkpoint"], self.engine.identity
            )
            reference = self._policy_reference(client, state)
            policy = {
                "id": reference,
                "checkpoint": state["checkpoint"],
                "revision": state["revision"],
                "client": client,
            }
            self.engine.policy(
                reference, folder, document["state"],
                adapter_digest=document["files"].get("adapters.safetensors"),
            )
            result = {
                "policy": reference,
                "revision": state["revision"],
                "checkpoint": state["checkpoint"],
            }
            with self.condition:
                self.store.finish(request_id, result, policy=policy)
            return result
        if kind == "save":
            result = {
                "revision": state["revision"],
                "policy": self._policy_reference(client, state),
                "metadata": body.get("metadata", {}),
                "checkpoint": state["checkpoint"],
            }
            if body.get("kind", "state") == "sampler":
                # The durable optimizer checkpoint already owns immutable
                # factors. A sampler save needs only a journaled reference.
                with self.condition:
                    self.store.finish(request_id, result)
                return result
            self._ensure_client(client)
            writer = None
            if hasattr(self.engine, "write_snapshot"):
                source, _ = self.store.verify(state["checkpoint"], self.engine.identity)
                writer = lambda folder: self.engine.write_snapshot(client, folder, source)
            return self._checkpoint(request_id, client, state, result, writer=writer)
        if kind in ("sample", "logprobs"):
            if kind == "logprobs":
                body.setdefault("seed", 0)
            # Independent deterministic RNG streams; retries and scheduling cannot
            # consume another client's RNG. Explicit seeds support recovery probes.
            body = self._inference_body(request_id, client, body)
            if kind == "sample" and not body.get("target_token_ids"):
                self.inference_metrics.batch(1)
        else:
            self._ensure_client(client)
            if kind in ("forward", "forward_backward", "optim_step"):
                self.live_clients.add(client)
        work = self.engine.work(client, kind, body)
        if kind == "sample" and not body.get("target_token_ids"):
            work = self._observe_samples([request_id], work)
        result = yield from work
        if kind == "optim_step":
            state = {**state, "revision": result["optim_metrics"]["optimizer_steps"]}
            return self._checkpoint(request_id, client, state, result)
        with self.condition:
            self.store.finish(request_id, result)
        return result

    def _owner(self):
        try:
            self.engine = self.engine_factory()
            self.capabilities = {
                **self.engine.capabilities,
                "identity": self.engine.identity,
                "protocol_version": 1,
                "max_request_bytes": MAX_REQUEST_BYTES,
            }
            with self.condition:
                self.store.recover(self.engine.identity)
                self.clients = self.store.states("clients")
            # Recovery verifies durable state but leaves optimizer allocations
            # cold. Only an operation requiring a client brings it onto the GPU.
            for reference, state in self.store.states("policies").items():
                if self._retired(state["checkpoint"]):
                    continue
                folder, document = self.store.verify(
                    state["checkpoint"], self.engine.identity
                )
                self.engine.policy(
                    reference, folder, document["state"],
                    adapter_digest=document["files"].get("adapters.safetensors"),
                )
            self.inference_metrics = InferenceMetrics()
            if hasattr(self.engine, "set_inference_observer"):
                self.engine.set_inference_observer(self.inference_metrics.progress)
        except BaseException as error:
            self.start_error = error
            self.ready.set()
            return
        self.ready.set()
        try:
            while True:
                with self.condition:
                    while not self.round_robin and not self.stopping:
                        self.condition.wait()
                    if self.stopping:
                        break
                    lane = self.round_robin.popleft()
                    if lane not in self.active:
                        request_id, client, kind, body = self.queues[lane].popleft()
                        if lane != CONTINUOUS_INFERENCE:
                            self.store.running(request_id)
                        items = [(request_id, client, kind, body)]
                        if kind == "sample" and hasattr(self.engine, "sample_batch") and lane != CONTINUOUS_INFERENCE:
                            batch_limit = self.capabilities.get("max_inference_batch", 4)
                            capacity = getattr(self.engine, "inference_batch_capacity", None)
                            if capacity:
                                batch_limit = capacity(body, batch_limit)
                            signature = lambda value: (
                                value.get("policy"),
                                value.get("cache_precision", "bf16"),
                                len(value.get("prompt_token_ids", [])),
                                value.get("max_tokens"),
                            )
                            while (
                                len(items) < batch_limit
                                and self.queues[lane]
                                and self.queues[lane][0][2] == "sample"
                                and (not capacity or capacity(self.queues[lane][0][3], batch_limit) > 1)
                                and signature(self.queues[lane][0][3])
                                == signature(body)
                            ):
                                item = self.queues[lane].popleft()
                                items.append(item)
                                self.store.running(item[0])
                            for candidate in list(self.round_robin):
                                if len(items) == batch_limit:
                                    break
                                if (
                                    candidate in self.active
                                    or not self.queues[candidate]
                                ):
                                    continue
                                item = self.queues[candidate][0]
                                if item[2] == "sample" and (not capacity or capacity(item[3], batch_limit) > 1) and signature(
                                    item[3]
                                ) == signature(body):
                                    items.append(self.queues[candidate].popleft())
                                    self.store.running(item[0])
                                    if not self.queues[candidate]:
                                        self.round_robin.remove(candidate)
                                        del self.queues[candidate]
                        generator = (
                            self._continuous_samples(items[0])
                            if lane == CONTINUOUS_INFERENCE
                            else self._execute_batch(items)
                            if len(items) > 1
                            else self._execute(request_id, client, kind, body)
                        )
                        for item_id, _, item_kind, item_body in items:
                            if lane != CONTINUOUS_INFERENCE and item_kind == "sample" and not item_body.get("target_token_ids"):
                                self.inference_metrics.start(item_id)
                        self.active[lane] = (request_id, client, kind, generator)
                    request_id, client, kind, generator = self.active[lane]
                try:
                    progress = next(generator)
                    with self.condition:
                        if lane != CONTINUOUS_INFERENCE:
                            self._record_progress(request_id, progress)
                except StopIteration:
                    with self.condition:
                        self.active.pop(lane)
                        self.progress_commits.pop(request_id, None)
                except Exception as error:
                    self.progress_commits.pop(request_id, None)
                    if lane == CONTINUOUS_INFERENCE:
                        # Request kernel errors already have isolated receipts.
                        # A queue or journal failure stops the owner. Its leader
                        # may already have a durable successful receipt.
                        raise
                    if kind == "sample":
                        self.inference_metrics.finish(request_id, failed=True)
                    # A failed optimizer must not leave undurable weights live.
                    if not isinstance(error, ClientOwnershipConflict) and kind in ("forward_backward", "optim_step", "restore"):
                        self.engine.abort(client, request_id)
                    state = self.clients.get(client)
                    if not isinstance(error, ClientOwnershipConflict) and state and kind in ("optim_step", "restore"):
                        folder, _ = self.store.verify(
                            state["checkpoint"], self.engine.identity
                        )
                        if state.get("closed"):
                            if client in self.resident_clients:
                                self.engine.close_client(client)
                            self.resident_clients.discard(client)
                        else:
                            self.engine.restore(client, folder, state)
                            self.resident_clients.add(client)
                    # If rename succeeded, recover its embedded receipt now.
                    # Verification and GPU restore must not hold the poll lock.
                    candidate = "studio-state:" + digest(
                        {"operation": request_id, "identity": self.engine.identity}
                    )
                    if (self.store.checkpoints / candidate.split(":")[1]).exists():
                        folder, doc = self.store.verify(candidate, self.engine.identity)
                        self.engine.restore(client, folder, doc["state"])
                        with self.condition:
                            self.store.finish(request_id, doc["result"], client_state=doc["state"])
                            self.clients[client] = doc["state"]
                        self.resident_clients.add(client)
                        if kind in ("create", "restore"):
                            self.live_clients.add(client)
                    else:
                        with self.condition:
                            self.store.fail(request_id, f"{type(error).__name__}: {error}")
                        if kind in ("create", "restore") and state is None and client in self.resident_clients:
                            self.engine.close_client(client)
                            self.resident_clients.discard(client)
                    with self.condition:
                        self.active.pop(lane)
                with self.condition:
                    if lane in self.active or self.queues[lane]:
                        self.round_robin.append(lane)
                    else:
                        del self.queues[lane]
                    self.condition.notify_all()
        except Exception as error:
            self.fatal_error = f"{type(error).__name__}: {error}"
        finally:
            for request_id, client, _, generator in self.active.values():
                generator.close()
                self.engine.abort(client)
                with self.condition:
                    try:
                        if self.store.operation(request_id)["status"] in ("queued", "running"):
                            self.store.fail(
                                request_id,
                                "Service stopped at a materialized work boundary",
                                "interrupted",
                            )
                    except Exception:
                        pass  # restart recovers the durable intent/checkpoint
            self.engine.close()
            # Include queued/interleaved requests interrupted by shutdown. The
            # counters are volatile and restart under a new session identity.
            for request_id in list(self.inference_metrics.active):
                self.inference_metrics.finish(request_id, failed=True)
            for request_id in list(self.inference_metrics.queued):
                self.inference_metrics.finish(request_id, failed=True)
            with self.condition:
                self.stopping = True
                self.condition.notify_all()

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.owner.join()
        self.inference_metrics.close()
        self.store.close()
