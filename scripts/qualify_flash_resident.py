"""
Purpose: Qualify real Flash LoRA through the resident service and its durable receipts.
Input: An approved mixed-precision model and a fresh output directory.
Output: One optimizer commit, frozen-base evidence, inference checks, and memory data.
Note: This qualification uses synthetic tokens. It does not measure world rewards.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import traceback


def main() -> None:
    """
    Purpose: Exercise one GPU owner across inference, replay, Adam, and publication.
    Input: CLI paths and bounded fixture sizes.
    Output: Local reports. The caller must also enforce a process deadline.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, required=True, help="Approved Flash model"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh qualification root"
    )
    parser.add_argument(
        "--prefix-tokens", type=int, default=2049, help="Shared training prefix length"
    )
    parser.add_argument(
        "--branches", type=int, default=8, help="Independent sampled continuations"
    )
    parser.add_argument(
        "--completion-tokens",
        type=int,
        default=16,
        help="Sampled tokens per continuation",
    )
    parser.add_argument(
        "--check-max-context",
        action="store_true",
        help="Attempt requested 262k-context training admission after the micro update",
    )
    args = parser.parse_args()
    if args.branches < 1 or args.completion_tokens < 1 or args.prefix_tokens < 1:
        parser.error("Fixture sizes must be positive.")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine
    from studio_compute.service import Service

    started = time.monotonic()
    service = None
    result = {
        "status": "running",
        "optimizer_updates_committed": 0,
        "model_loads": 0,
        "model": str(args.model),
        "stages": [],
        "fixture": {
            "prefix_tokens": args.prefix_tokens,
            "branches": args.branches,
            "completion_tokens": args.completion_tokens,
        },
        "settings": {
            "lora_rank": 16,
            "lora_scale": 2.0,
            "learning_rate": 5e-5,
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1e-8,
            "context_window": 262144,
            "inference_concurrency": 8,
            "execution_profile": "bf16-fast",
            "segment_tokens": 2048,
            "prefill_tokens": 2048,
            "head_rows": 128,
            "activation_budget_gib": 16,
        },
    }

    def record(stage: str, **fields: object) -> None:
        event = {
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        result["stages"].append(event)
        with (args.output / "events.jsonl").open("a") as target:
            target.write(json.dumps(event) + "\n")
        temporary = args.output / "result.pending"
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.output / "result.json")
        print(json.dumps(event), flush=True)

    def wait(request_id: str) -> dict:
        last = None
        last_report = 0.0
        while True:
            operation = service.wait(request_id, timeout=1)
            if operation["status"] not in ("queued", "running"):
                if operation["status"] != "succeeded":
                    raise RuntimeError(
                        f"{request_id}: {operation['status']}: {operation.get('error')}"
                    )
                return operation["result"]
            progress = operation.get("progress")
            now = time.monotonic()
            if progress != last and now - last_report > 15:
                record("operation_progress", request_id=request_id, progress=progress)
                last, last_report = progress, now

    def call(kind: str, body: dict, request_id: str) -> dict:
        record("operation_started", kind=kind, request_id=request_id)
        service.submit(request_id, "flash-qualification", kind, body)
        value = wait(request_id)
        record(
            "operation_finished",
            kind=kind,
            request_id=request_id,
            metrics=value.get("execution_metrics"),
        )
        return value

    def factory(path: Path, **kwargs: object) -> object:
        result["model_loads"] += 1
        return QwenFlashModel(path, **kwargs)

    def samples(
        policy: str, label: str, prefix: list, count: int, maximum: int
    ) -> list:
        ids = []
        for index in range(count):
            request_id = f"{label}:{index}"
            service.submit(
                request_id,
                "flash-qualification",
                "sample",
                {
                    "policy": policy,
                    "prompt_token_ids": prefix,
                    "max_tokens": maximum,
                    "seed": 100 + index,
                    "cache_precision": "bf16",
                    "temperature": 1.0,
                    "top_p": 1.0,
                },
            )
            ids.append(request_id)
        return [wait(request_id) for request_id in ids]

    lease_path = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
    with lease_path.open("r+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            record("service_load_started")
            service = Service(
                args.output / "service",
                lambda: RuntimeEngine(
                    args.model,
                    model_factory=factory,
                    execution_profile="bf16-fast",
                    headroom_gib=16,
                    segment_tokens=2048,
                    head_rows=128,
                    prefill_tokens=2048,
                    activation_budget_gib=16,
                    inference_concurrency=8,
                    prefix_cache_gib=None,
                ),
            )
            result["capabilities"] = service.capabilities
            record("service_ready")
            call(
                "create",
                {
                    "seed": 20261019,
                    "lora_rank": 16,
                    "lora_scale": 2.0,
                    "learning_rate": 5e-5,
                    "optimizer": {"beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8},
                },
                "create",
            )
            policy = call("publish", {"revision": 0}, "publish:0")["policy"]
            before = call(
                "inspect", {"hash_base": True, "hash_state": True}, "inspect:before"
            )
            result["before"] = before
            prefix = [16 + index % 80 for index in range(args.prefix_tokens)]
            record("initial_sampling_started")
            sampled = samples(
                policy, "sample:before", prefix, args.branches, args.completion_tokens
            )
            record("initial_sampling_complete", metrics=service.telemetry())
            score_body = {
                "policy": policy,
                "prompt_token_ids": prefix,
                "target_token_ids": sampled[0]["sampled_tokens"],
                "max_tokens": args.completion_tokens,
                "seed": 100,
                "cache_precision": "bf16",
                "use_prefix_cache": False,
            }
            old_score_before = call("logprobs", score_body, "score:old-before")
            sequences = [
                {
                    "prompt_token_ids": prefix,
                    "sampled_tokens": value["sampled_tokens"],
                    "behavior_logprobs": value["logprobs"],
                    "advantages": [1.0] * len(value["sampled_tokens"]),
                    "loss_mask": [1.0] * len(value["sampled_tokens"]),
                }
                for value in sampled
            ]
            backward_body = {
                "sequences": sequences,
                "loss": "cispo",
                "reduction": "sum",
                "clip_low": 0.0,
                "clip_high": 4.0,
            }
            service.submit(
                "update:backward",
                "flash-qualification",
                "forward_backward",
                backward_body,
            )
            interleave_ids = []
            last_report = 0.0
            while True:
                operation = service.wait("update:backward", timeout=1)
                progress = operation.get("progress") or {}
                if not interleave_ids and "backward" in str(progress.get("phase", "")):
                    for index in range(8):
                        request_id = f"sample:during:{index}"
                        service.submit(
                            request_id,
                            "flash-qualification",
                            "sample",
                            {
                                "policy": policy,
                                "prompt_token_ids": prefix[: 32 + index],
                                "max_tokens": 4,
                                "seed": index,
                                "cache_precision": "bf16",
                            },
                        )
                        interleave_ids.append(request_id)
                    record("inference_submitted_during_backward", progress=progress)
                if time.monotonic() - last_report > 15:
                    record(
                        "backward_progress",
                        status=operation["status"],
                        progress=progress,
                    )
                    last_report = time.monotonic()
                if operation["status"] not in ("queued", "running"):
                    break
            forward = wait("update:backward")
            for request_id in interleave_ids:
                wait(request_id)
            result["interleaved_inference_requests"] = len(interleave_ids)
            result["backward_metrics"] = forward
            differences = [
                actual - expected
                for replay, sample in zip(forward["logprobs"], sampled)
                for actual, expected in zip(replay, sample["logprobs"], strict=True)
            ]
            result["sampling_replay_logprob_error"] = {
                "maximum_absolute_nats": max(map(abs, differences)),
                "root_mean_square_nats": math.sqrt(
                    sum(value * value for value in differences) / len(differences)
                ),
                "acceptance_maximum_absolute_nats": 0.125,
                "passed": max(map(abs, differences)) <= 0.125,
            }
            record("backward_complete", metrics=forward.get("execution_metrics"))
            optimizer_body = {
                "learning_rate": 5e-5,
                "backward_operation": "update:backward",
            }
            committed = call("optim_step", optimizer_body, "update:optimizer")
            result["optimizer_receipt"] = committed
            result["optimizer_updates_committed"] = committed["optim_metrics"][
                "optimizer_steps"
            ]
            assert result["optimizer_updates_committed"] == 1
            assert (
                math.isfinite(committed["optim_metrics"]["gradient_norm"])
                and committed["optim_metrics"]["gradient_norm"] > 0
            )
            duplicate = call("optim_step", optimizer_body, "update:optimizer")
            assert duplicate == committed
            new_policy = call("publish", {"revision": 1}, "publish:1")["policy"]
            after = call(
                "inspect", {"hash_base": True, "hash_state": True}, "inspect:after"
            )
            result["after"] = after
            assert before["base_allocation"] == after["base_allocation"]
            assert before["frozen_base_hash"] == after["frozen_base_hash"]
            assert before["state_hashes"]["adapter"] != after["state_hashes"]["adapter"]
            assert result["model_loads"] == 1
            old_score_after = call("logprobs", score_body, "score:old-after")
            result["immutable_policy_matched_scores"] = (
                old_score_before["logprobs"] == old_score_after["logprobs"]
            )
            assert result["immutable_policy_matched_scores"]
            old_again = samples(
                policy, "sample:old-after", prefix, 1, args.completion_tokens
            )[0]
            result["stochastic_sample_changed_across_cache_and_batch_histories"] = (
                old_again["sampled_tokens"] != sampled[0]["sampled_tokens"]
            )
            updated = samples(
                new_policy, "sample:updated", prefix, 1, args.completion_tokens
            )[0]
            result["updated_policy_sample_finite"] = all(
                math.isfinite(value) for value in updated["logprobs"]
            )
            assert result["updated_policy_sample_finite"]
            result["final_checkpoint"] = committed["checkpoint"]
            result["published_policy"] = new_policy
            result["inference_metrics"] = service.telemetry()
            if args.check_max_context:
                long_prefix = [
                    16 + index % 80 for index in range(262144 - args.completion_tokens)
                ]
                long_sequences = [
                    {**value, "prompt_token_ids": long_prefix} for value in sequences
                ]
                try:
                    call(
                        "forward_backward",
                        {**backward_body, "sequences": long_sequences},
                        "max-context:backward",
                    )
                    result["max_context_backward"] = (
                        "completed_without_optimizer_update"
                    )
                except RuntimeError as error:
                    result["max_context_backward"] = "rejected_or_failed"
                    result["max_context_error"] = str(error)
            result["status"] = (
                "passed_resident_micro_update"
                if result["sampling_replay_logprob_error"]["passed"]
                else "mechanical_checks_passed_numerical_check_failed"
            )
            record("qualification_finished", status=result["status"])
        except BaseException as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            result["traceback"] = traceback.format_exc()
            record("qualification_failed", error=result["error"])
            raise
        finally:
            if service is not None:
                service.close()
                record("owned_service_stopped")
    if result["status"] != "passed_resident_micro_update":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
