"""
Purpose: Measure independent Flash request caches and run the normal rollout trainer.
Input: A fresh root, the pinned model, and a normal ShellSmith train configuration.
Output: Capacity receipts, real rollout outcomes, optimizer receipts, and memory evidence.
Note: An external supervisor must enforce the absolute process deadline.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from studio_compute import inference
from studio_compute.flash_model import QwenFlashModel
from studio_compute.http import server
from studio_compute.mlx_racing import ContinuousBatcher
from studio_compute.runtime import RuntimeEngine
from studio_compute.service import Service

GIB = 1 << 30
WINDOW = 262144


class ObservedBatcher(ContinuousBatcher):
    """Observe request-owned cache bytes at existing materialized boundaries."""

    def __init__(self, *args: object, engine: object, **kwargs: object):
        super().__init__(*args, **kwargs)
        self.engine = engine

    def step(self) -> dict:
        value = super().step()
        self.engine.observe_requests(list(self.active.values()))
        return value

    def _retire(self, request: object, *, error: Exception | None = None) -> None:
        self.engine.observe_requests(list(self.active.values()))
        super()._retire(request, error=error)


class ObservedFlashRuntime(RuntimeEngine):
    """
    Purpose: Record cache ownership without a second model or a different numerical graph.
    Input: Normal service operations and the normal cooperative scheduler.
    Output: Bounded CPU observations. No GPU-owner file or network calls occur here.
    """

    def __init__(self, *args: object, **kwargs: object):
        self.qualification_cache_precision = kwargs.pop("cache_precision", "bf16")
        super().__init__(*args, **kwargs)
        self.observed = {
            "peak_active_requests": 0,
            "peak_live_cache_bytes": 0,
            "peak_cached_tokens": 0,
            "peak_total_cache_bytes": 0,
            "eight_independent_caches_observed": False,
        }
        self.training = []

    def inference_scheduler(
        self, *, create: object, finish: object, fail: object
    ) -> object:
        return ObservedBatcher(
            self.inference_concurrency,
            engine=self,
            create=create,
            prefill=lambda request: inference.prefill_request(self, request),
            advance=lambda requests: inference.advance_requests(self, requests),
            sample=lambda requests: inference.sample_requests(self, requests),
            finish=finish,
            fail=fail,
            release=lambda request: inference.release_request(self, request),
        )

    def observe_requests(self, requests: list) -> None:
        """
        Purpose: Count live request state after the existing execution boundary.
        Input: At most eight active requests. Output: CPU high-water measurements.
        Note: Array byte properties do not evaluate tensors or synchronize the GPU.
        """
        rows = []
        caches = []
        for request in requests:
            size = self.prefixes.size(request.cache)
            offsets = [
                cache.offset for cache in request.cache if hasattr(cache, "index_keys")
            ]
            rows.append(
                {
                    "request": request.request_id,
                    "cache_bytes": size,
                    "cached_tokens": min(offsets, default=0),
                    "attention_cache_formats": sorted(
                        {
                            f"{cache.bits}bit/group{cache.group_size}"
                            if hasattr(cache, "bits")
                            else "bf16"
                            for cache in request.cache
                            if hasattr(cache, "index_keys")
                        }
                    ),
                }
            )
            caches.extend(id(cache) for cache in request.cache)
        value = {
            "active_requests": len(rows),
            "live_cache_bytes": sum(row["cache_bytes"] for row in rows),
            "cached_tokens": sum(row["cached_tokens"] for row in rows),
            "retained_prefix_cache_bytes": self.prefixes.bytes,
            "allocator_active_bytes": self.mx.get_active_memory(),
            "allocator_peak_bytes": self.mx.get_peak_memory(),
            "requests": rows,
        }
        previous = self.observed
        independent = (
            len(rows) == 8
            and all(row["cached_tokens"] > 0 for row in rows)
            and len(caches) == len(set(caches))
        )
        self.observed = {
            **value,
            "peak_active_requests": max(len(rows), previous["peak_active_requests"]),
            "peak_live_cache_bytes": max(
                value["live_cache_bytes"], previous["peak_live_cache_bytes"]
            ),
            "peak_cached_tokens": max(
                value["cached_tokens"], previous["peak_cached_tokens"]
            ),
            "peak_total_cache_bytes": max(
                value["live_cache_bytes"] + value["retained_prefix_cache_bytes"],
                previous["peak_total_cache_bytes"],
            ),
            "eight_independent_caches_observed": independent
            or previous["eight_independent_caches_observed"],
        }
        if independent and not previous["eight_independent_caches_observed"]:
            self.observed["first_eight_request_snapshot"] = value
        elif "first_eight_request_snapshot" in previous:
            self.observed["first_eight_request_snapshot"] = previous[
                "first_eight_request_snapshot"
            ]

    def metrics(self, started: float) -> dict:
        return {**super().metrics(started), "independent_request_caches": self.observed}

    def work(self, client: str, kind: str, body: dict) -> object:
        """Run normal operations and keep compact evidence outside raw token arrays."""
        if kind == "inspect" and body.get("qualification_probe") == "capacity":
            return self.capacity(client)
        value = yield from super().work(client, kind, body)
        if kind == "forward_backward":
            sequences = body["sequences"]
            differences = []
            for index, (row, probabilities) in enumerate(
                zip(sequences, value["logprobs"], strict=True)
            ):
                indices = value["logprob_target_indices"][index]
                expected = row["behavior_logprobs"]
                if indices is not None:
                    expected = [expected[i] for i in indices]
                differences.extend(
                    actual - prior
                    for actual, prior in zip(probabilities, expected, strict=True)
                )
            self.training.append(
                {
                    "sequences": len(sequences),
                    "distinct_prompts": len(
                        {tuple(row["prompt_token_ids"]) for row in sequences}
                    ),
                    "maximum_sequence_tokens": max(
                        len(row["prompt_token_ids"]) + len(row["sampled_tokens"])
                        for row in sequences
                    ),
                    "supervised_tokens": value["forward_metrics"].get(
                        "supervised_tokens"
                    ),
                    "maximum_logprob_error_nats": max(map(abs, differences), default=0),
                    "root_mean_square_logprob_error_nats": math.sqrt(
                        sum(x * x for x in differences) / max(1, len(differences))
                    ),
                    "execution_metrics": value["execution_metrics"],
                }
            )
        return value

    def capacity(self, client: str) -> dict:
        """
        Purpose: Measure independent cache allocations and query normal backward admission.
        Input: One resident qualification client. Output: Explicit allocation and admission facts.
        Note: Distinct cache contents prevent common-expression aliasing during this allocation-only test.
        """
        from studio_compute.numerics.flash_ops import FlashOps
        from studio_compute.activation_budget import SAFETY_BYTES

        mx = self.mx
        self.prefixes.clear()
        mx.clear_cache()
        resident_bytes = max(mx.get_active_memory(), self.persistent_bytes())
        layers = self.lm.layers
        text = self.lm.args
        precision = self.qualification_cache_precision
        kv_bytes_per_element = 2 if precision == "bf16" else 1 + 4 / 64
        per_token = sum(
            math.ceil(
                layer.self_attn.num_key_value_heads
                * layer.self_attn.head_dim
                * 2
                * kv_bytes_per_element
            )
            + text.indexer_head_dim * 2
            + 4
            for layer in layers
            if not layer.is_linear
        )
        fixed = sum(
            layer.linear_attn.num_v_heads
            * layer.linear_attn.head_v_dim
            * layer.linear_attn.head_k_dim
            * 2
            + (layer.linear_attn.conv_kernel_size - 1) * layer.linear_attn.conv_dim * 2
            + (
                layer.ple.short_conv_state_len * text.hidden_size * text.hc_count * 2
                + 16
                if "ple" in layer
                else 0
            )
            for layer in layers
            if layer.is_linear
        )
        scratch = 8 * (2 << 30)
        future_policy_bytes = sum(value.nbytes for value in self.idle_factors.values())
        available = max(
            0,
            self.admission_budget_bytes()
            - resident_bytes
            - SAFETY_BYTES
            - scratch
            - future_policy_bytes,
        )
        length = min(WINDOW, max(0, (available - 8 * fixed) // (8 * per_token)))
        result = {
            "inference_cache_precision": precision,
            "inference_kv_group_size": 64 if precision == "8bit" else None,
            "recurrent_and_indexer_precision": "bf16",
            "training_cache_precision": "bf16",
            "configured_global_kv_pool_bytes": None,
            "pool_policy": "dynamic byte admission; no paged token pool",
            "allocator_budget_bytes": self.resident.budget,
            "effective_working_set_budget_bytes": self.admission_budget_bytes(),
            "resident_bytes_before_pool": resident_bytes,
            "safety_bytes": SAFETY_BYTES,
            "eight_request_scratch_reserve_bytes": scratch,
            "policy_reserve_bytes": future_policy_bytes,
            "conservative_live_cache_capacity_bytes": available,
            "cache_bytes_per_context_token": per_token,
            "fixed_recurrent_bytes_per_request": fixed,
            "per_request_context_window": WINDOW,
            "eight_equal_context_tokens_by_budget": length,
        }
        if length:
            caches = []
            for request_index in range(8):
                cache = self.lm.make_cache()
                if precision == "8bit":
                    cache = [
                        item.to_quantized(bits=8, group_size=64)
                        if hasattr(item, "to_quantized")
                        else item
                        for item in cache
                    ]
                for layer_index, (layer, item, initial) in enumerate(
                    zip(
                        layers,
                        cache,
                        FlashOps.initial_states(layers, mx.bfloat16),
                        strict=True,
                    )
                ):
                    marker = 4 * (request_index * len(layers) + layer_index)
                    if layer.is_linear:
                        item.cache = [
                            _cache_fill(mx, value.shape, marker + index, value.dtype)
                            for index, (_, value) in enumerate(initial)
                        ]
                        if "ple" in layer:
                            item.cache.append(
                                mx.zeros(
                                    (1, layer.ple.ple_embedding.context_len), mx.int64
                                )
                            )
                    else:
                        heads, width = (
                            layer.self_attn.num_key_value_heads,
                            layer.self_attn.head_dim,
                        )
                        if precision == "8bit":
                            item.keys = _packed_cache_fill(
                                mx, heads, length, width, marker
                            )
                            item.values = _packed_cache_fill(
                                mx, heads, length, width, marker + 1
                            )
                        else:
                            item.keys = _cache_fill(
                                mx, (1, heads, length, width), marker, mx.bfloat16
                            )
                            item.values = _cache_fill(
                                mx, (1, heads, length, width), marker + 1, mx.bfloat16
                            )
                        item.offset = length
                        item.index_keys = _cache_fill(
                            mx,
                            (1, length, text.indexer_head_dim),
                            marker + 2,
                            mx.bfloat16,
                        )
                        item.index_position_ids = mx.zeros((1, length), mx.int32)
                caches.append(cache)
            mx.eval([item.state for cache in caches for item in cache])
            result.update(
                materialized_independent_caches=8,
                materialized_cache_bytes=sum(
                    self.prefixes.size(cache) for cache in caches
                ),
                materialized_total_context_tokens=8 * length,
                allocator_bytes_with_pool=mx.get_active_memory(),
                allocator_peak_bytes=mx.get_peak_memory(),
            )
            del caches, cache, item, initial
            mx.clear_cache()
        gradients = sum(
            value.nbytes for value in self.clients[client]["optimizer"].master.values()
        )

        def plan(total: int) -> dict:
            sequence = {
                "prompt_token_ids": [16] * (total - 16),
                "sampled_tokens": [17] * 16,
            }
            try:
                interval, peak, budgets = self._saved_admission(
                    {0: [sequence]}, gradients
                )
                return {
                    "admitted": True,
                    "checkpoint_layers": interval,
                    "additional_bytes": peak,
                }
            except MemoryError as error:
                return {"admitted": False, "error": str(error)}

        result["maximum_context_backward_admission"] = plan(WINDOW)
        low, high = 1, WINDOW // 2048
        while low < high:
            midpoint = (low + high + 1) // 2
            if plan(midpoint * 2048)["admitted"]:
                low = midpoint
            else:
                high = midpoint - 1
        result["largest_estimated_backward_context_multiple_of_2048"] = low * 2048
        result["largest_estimated_backward_plan"] = plan(low * 2048)
        result["after_capacity_metrics"] = self.metrics(time.monotonic())
        return result


def main() -> None:
    """
    Purpose: Own the service and normal rollout trainer in a fresh qualification directory.
    Input: CLI paths. Output: Local evidence and normal trainer artifacts.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Fresh prepared qualification directory",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Pinned mixed-precision Flash artifact",
    )
    parser.add_argument(
        "--port", type=int, default=8767, help="Private loopback service port"
    )
    parser.add_argument(
        "--cache-precision",
        choices=("bf16", "8bit"),
        default="bf16",
        help="Inference KV format; backward remains BF16",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if (root / "result.json").exists():
        raise FileExistsError("Use a fresh qualification result directory.")
    started = time.monotonic()
    result = {
        "status": "starting",
        "normal_trainer": True,
        "model_loads": 0,
        "stages": [],
        "target_optimizer_updates": 1,
        "inference_cache_precision": args.cache_precision,
        "training_cache_precision": "bf16",
    }
    service = http = trainer = None

    def record(stage: str, **fields: object) -> None:
        value = {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        result["stages"].append(value)
        result["stages"] = result["stages"][-200:]
        with (root / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        pending = root / "result.pending"
        pending.write_text(json.dumps(result, indent=2) + "\n")
        pending.replace(root / "result.json")
        print(json.dumps(value), flush=True)

    def factory(path: Path, **kwargs: object) -> object:
        result["model_loads"] += 1
        return QwenFlashModel(path, **kwargs)

    def call(client: str, kind: str, body: dict, request_id: str) -> dict:
        service.submit(request_id, client, kind, body)
        last = 0.0
        while True:
            operation = service.wait(request_id, timeout=1)
            if operation["status"] not in ("queued", "running"):
                if operation["status"] != "succeeded":
                    raise RuntimeError(
                        f"{operation['status']}: {operation.get('error')}"
                    )
                return operation["result"]
            if time.monotonic() - last >= 15:
                record(
                    "operation_progress",
                    request_id=request_id,
                    progress=operation.get("progress"),
                )
                last = time.monotonic()

    def interrupted(*_: object) -> None:
        raise InterruptedError("Qualification supervisor requested stop.")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    lease = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
    with lease.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            record("service_loading")
            service = Service(
                root / "service",
                lambda: ObservedFlashRuntime(
                    args.model,
                    model_factory=factory,
                    execution_profile="bf16-fast",
                    headroom_gib=16,
                    activation_budget_gib=16,
                    inference_concurrency=8,
                    segment_tokens=2048,
                    prefill_tokens=2048,
                    head_rows=128,
                    cache_precision=args.cache_precision,
                ),
            )
            result["capabilities"] = service.capabilities
            record("service_ready")
            probe_client = "flash-capacity-only"
            call(
                probe_client,
                "create",
                {
                    "seed": 20261019,
                    "lora_rank": 16,
                    "lora_scale": 2.0,
                    "learning_rate": 5e-5,
                    "optimizer": {"beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8},
                },
                "capacity:create",
            )
            result["capacity"] = call(
                probe_client,
                "inspect",
                {"qualification_probe": "capacity"},
                "capacity:pool-and-admission",
            )
            record("capacity_complete", capacity=result["capacity"])
            maximum = {
                "sequences": [
                    {
                        "prompt_token_ids": [16] * (WINDOW - 16),
                        "sampled_tokens": [17] * 16,
                        "behavior_logprobs": [-1.0] * 16,
                        "advantages": [1.0] * 16,
                        "loss_mask": [1.0] * 16,
                    }
                ],
                "loss": "cispo",
                "reduction": "sum",
                "clip_low": 0.0,
                "clip_high": 4.0,
            }
            try:
                check = call(
                    probe_client,
                    "forward_backward",
                    maximum,
                    "capacity:actual-max-context-backward",
                )
                result["maximum_context_backward"] = {
                    "status": "completed",
                    "metrics": check["execution_metrics"],
                }
            except RuntimeError as error:
                result["maximum_context_backward"] = {
                    "status": "rejected_or_failed",
                    "error": str(error),
                }
            call(probe_client, "suspend", {}, "capacity:suspend")
            record("max_context_checked", result=result["maximum_context_backward"])
            http = server(service, ("127.0.0.1", args.port))
            threading.Thread(target=http.serve_forever, daemon=True).start()
            command = [
                sys.executable,
                "-u",
                "-m",
                "shellsmith",
                "train",
                "--config",
                str(root / "pilot.toml"),
            ]
            with (root / "readiness.json").open("w") as out:
                subprocess.run(
                    [*command, "--dry-run"],
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            with (root / "trainer.log").open("w") as log:
                trainer = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True,
                )
            result["trainer_pid"] = trainer.pid
            result["status"] = "normal_rollouts_running"
            record("normal_trainer_started", trainer_pid=trainer.pid)
            while trainer.poll() is None:
                result["inference_metrics"] = service.telemetry()
                result["request_caches"] = service.engine.observed
                result["training_receipts"] = service.engine.training
                state_path = root / "train/trainer_state.json"
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    result["trainer_progress"] = {
                        key: state.get(key)
                        for key in (
                            "status",
                            "step",
                            "rft_updates_completed",
                            "active_policy_ref",
                            "next_world_offset",
                        )
                    }
                record(
                    "normal_trainer_progress",
                    inference=result["inference_metrics"],
                    caches=result["request_caches"],
                )
                time.sleep(15)
            result["trainer_returncode"] = trainer.returncode
            result["training_receipts"] = service.engine.training
            state_path = root / "train/trainer_state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            result["trainer_progress"] = {
                key: state.get(key)
                for key in (
                    "status",
                    "step",
                    "rft_updates_completed",
                    "active_policy_ref",
                    "next_world_offset",
                )
            }
            result["request_caches"] = service.engine.observed
            result["inference_metrics"] = service.telemetry()
            result["status"] = (
                "normal_rollout_update_completed"
                if trainer.returncode == 0 and state.get("rft_updates_completed") == 1
                else "normal_rollout_qualification_incomplete"
            )
            record("normal_trainer_finished", status=result["status"])
        except BaseException as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            record("qualification_failed", error=result["error"])
            raise
        finally:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            if trainer is not None and trainer.poll() is None:
                os.killpg(trainer.pid, signal.SIGINT)
                try:
                    trainer.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(trainer.pid, signal.SIGKILL)
                    trainer.wait(timeout=15)
            if http is not None:
                http.shutdown()
                http.server_close()
            if service is not None:
                service.close()
            record("owned_service_stopped")


def _cache_fill(mx: object, shape: tuple, number: int, dtype: object) -> object:
    """Use distinct exactly representable BF16 values for each capacity buffer."""
    scalar = (128 + number % 128) * 2.0 ** (number // 128 - 7)
    return mx.full(shape, scalar, dtype)


def _packed_cache_fill(
    mx: object, heads: int, length: int, width: int, number: int
) -> tuple:
    """Allocate native 8-bit KV payload and group metadata for the capacity probe."""
    assert width % 64 == 0
    return (
        mx.full((1, heads, length, width // 4), number + 1, mx.uint32),
        _cache_fill(mx, (1, heads, length, width // 64), number, mx.bfloat16),
        _cache_fill(mx, (1, heads, length, width // 64), number + 1, mx.bfloat16),
    )


if __name__ == "__main__":
    main()
