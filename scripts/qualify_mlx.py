#!/usr/bin/env python3
"""Run resident Studio numerical/recovery gates without external services.

Run with the frozen Studio Python and source bundle, e.g.:
  MLX_ENABLE_TF32=0 PYTHONPATH=src python scripts/qualify_mlx.py \
      --checkpoint-root /tmp/mlx-qualification-checkpoints \
      --output docs/mlx-qualification/studio.json --context-gate

This acquires the same GPU lease as training. Checkpoints and results are
immutable qualification artifacts; no existing model or environment is changed.
The 32K gate is a synthetic exact-token full-prompt update, not a live episode.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

os.environ.setdefault("MLX_ENABLE_TF32", "0")


@contextmanager
def gpu_session_lifetime(get_session):
    from shellsmith.jobs.lease import GpuLease

    with GpuLease():
        try:
            yield
        finally:
            session = get_session()
            if session is not None:
                session.close()


async def qualify(args):
    import mlx.core as mx
    import numpy as np
    from mlx.utils import tree_flatten

    from shellsmith.mlx import create_session
    from shellsmith.mlx.checkpoint import verify_checkpoint
    from shellsmith.mlx.identity import digest
    from shellsmith.train.backend import TokenBatch, TokenSequence
    from shellsmith.train.config import TrainConfig

    output = Path(args.output).expanduser().absolute()
    checkpoint_root = Path(args.checkpoint_root).expanduser().absolute()
    if output.exists() or checkpoint_root.exists():
        raise FileExistsError("Choose fresh output and checkpoint-root paths for each qualification run.")
    output.parent.mkdir(parents=True, exist_ok=True)
    document = {"format": "shellsmith-mlx-qualification-v1", "status": "running",
                "host": platform.node(), "started_at_unix": time.time(), "phases": [],
                "numerical_environment": {"MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")},
                "memory_limit_gib": 128, "context_gate_requested": args.context_gate,
                "soak_updates_requested": args.soak_updates}

    def persist():
        temporary = output.with_suffix(output.suffix + ".pending")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    def event(name, started, **fields):
        row = {"phase": name, "seconds": time.monotonic() - started,
               "peak_memory_bytes": mx.get_peak_memory(), "active_memory_bytes": mx.get_active_memory(), **fields}
        document["phases"].append(row)
        persist()
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)
        if row["peak_memory_bytes"] >= 128 * 1024 ** 3:
            raise AssertionError("Measured Metal allocator peak exceeded the 128 GiB qualification gate.")

    def tensor_hashes(tensors):
        result = {}
        for name, value in sorted(tensors.items()):
            mx.eval(value)
            raw = value.view(mx.uint16) if value.dtype == mx.bfloat16 else value
            array = np.ascontiguousarray(np.asarray(raw))
            result[name] = hashlib.sha256(memoryview(array).cast("B")).hexdigest()
        return result

    def frozen_hashes(session):
        return tensor_hashes({name: value for name, value in tree_flatten(session.model.parameters())
                              if not name.endswith((".a", ".b"))})

    def sequence(record):
        mask = [1.] * len(record.sampled_tokens)
        if record.forced_close_token_span is not None:
            start, stop = record.forced_close_token_span
            mask[start:stop] = [0.] * (stop - start)
        return TokenSequence(tuple(record.prompt.to_ints()), tuple(record.sampled_tokens),
                             tuple(record.sampled_logprobs), tuple(0.01 * value for value in mask),
                             tuple(mask), record.request_id)

    async def boundary(session, name):
        step = session.optimizer.steps
        sampler = await session.save_checkpoint(kind="sampler", name=name + "-sampler")
        position = {"step": step, "next_world_offset": step, "settled_world_offsets": [],
                    "sampler_path": sampler, "backend": "mlx", "policy_identity": session.policy_identity,
                    "metrics_head": {"qualification": name}}
        state = await session.save_checkpoint(kind="state", name=name + "-state", metadata=position)
        assert await session.publish_policy(revision=step) == sampler
        return {"path": state, **position}

    config = TrainConfig()
    config = replace(config, compute=replace(config.compute, backend="mlx"),
                     mlx=replace(config.mlx, model_path=args.model_path))
    session = None
    persist()
    try:
        with gpu_session_lifetime(lambda: session):
            started = time.monotonic()
            session = await create_session(config, checkpoint_root=checkpoint_root)
            document["policy_identity"] = session.policy_identity
            event("resident_load", started, adapter_factors=len(session.adapter.state()),
                  adapter_parameters=sum(value.size for value in session.adapter.state().values()))
            started = time.monotonic()
            frozen_before = frozen_hashes(session)
            event("frozen_base_initial_hash", started, tensors=len(frozen_before), sha256=digest(frozen_before))

            payload = {"messages": [{"role": "user", "content": "Reply with only OK."}],
                       "max_tokens": 64, "reasoning_effort": "none"}
            started = time.monotonic()
            record, assistant = session.sampler.sample(payload, request_id="qualify-plain", session_id="qualification")
            event("plain_generation", started, sampled_tokens=record.sampled_tokens,
                  behavior_logprobs=record.sampled_logprobs, assistant=assistant,
                  finish_reason=record.finish_reason, prompt_tokens=record.prompt_tokens)
            assert record.finish_reason == "stop", "Plain native generation did not reach EOS within its explicit cap."

            started = time.monotonic()
            thought, thought_assistant = session.sampler.sample(
                {**payload, "reasoning_effort": "xhigh", "max_tokens": 256},
                request_id="qualify-thinking", session_id="qualification")
            event("thinking_generation", started, sampled_tokens=thought.sampled_tokens,
                  behavior_logprobs=thought.sampled_logprobs, assistant=thought_assistant,
                  forced_close_token_span=thought.forced_close_token_span, protocol=thought.protocol_row,
                  diagnostic_generation_attempts=thought.diagnostic_generation_attempts)
            assert thought.protocol_row["thinking_mode"] == "thinking"

            started = time.monotonic()
            forced_cap = len(session.renderer.forced_close_ids) + 2
            forced_thought, forced_assistant = session.sampler.sample(
                {"messages": [{"role": "user", "content": "Compute 19 times 23. Think through the calculation."}],
                 "reasoning_effort": "xhigh", "max_tokens": forced_cap},
                request_id="qualify-forced-thinking", session_id="qualification")
            assert forced_thought.forced_close_token_span is not None
            assert len(forced_thought.sampled_tokens) <= forced_cap
            event("forced_thinking_boundary_generation", started, sampled_tokens=forced_thought.sampled_tokens,
                  behavior_logprobs=forced_thought.sampled_logprobs, assistant=forced_assistant,
                  explicit_total_cap=forced_cap, forced_close_token_span=forced_thought.forced_close_token_span,
                  protocol=forced_thought.protocol_row, diagnostic_generation_attempts=forced_thought.diagnostic_generation_attempts)

            tool_payload = {"messages": [{"role": "user", "content": "Call echo with text equal to hello. Use the tool now."}],
                            "reasoning_effort": "none", "max_tokens": 256,
                            "tools": [{"type": "function", "function": {"name": "echo", "description": "Echo text.",
                                       "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                                                      "required": ["text"], "additionalProperties": False}}}]}
            started = time.monotonic()
            tool_record, tool_assistant = session.sampler.sample(tool_payload, request_id="qualify-tool", session_id="qualification")
            event("native_tool_generation", started, sampled_tokens=tool_record.sampled_tokens,
                  behavior_logprobs=tool_record.sampled_logprobs, assistant=tool_assistant,
                  finish_reason=tool_record.finish_reason, protocol=tool_record.protocol_row)
            assert tool_assistant.get("tool_calls"), "Native tool smoke did not produce a tool call."

            started = time.monotonic()
            recovery = await boundary(session, "origin")
            event("full_checkpoint_origin", started, state_path=recovery["path"],
                  components=verify_checkpoint(recovery["path"])["files"])
            batch = TokenBatch((sequence(record),))
            # Draw from an identical reset prompt on both sides of process recovery.
            prompt = tuple(record.prompt.to_ints())
            expected_tokens, expected_probabilities, _, _ = session.draw_pass(
                session.prepare_prompt(prompt), maximum=8, stop_ids=set())
            started = time.monotonic()
            metrics = await session.update(batch, learning_rate=config.train.learning_rate, clip_low=0., clip_high=4.)
            expected_optimizer, expected_metadata = session.optimizer.state()
            expected_adapter_hashes = tensor_hashes(session.adapter.state())
            expected_optimizer_hashes = tensor_hashes(expected_optimizer)
            expected_rng = session.rng.tolist()
            expected_optimizer = None
            event("first_update", started, metrics=metrics, adapter_sha256=digest(expected_adapter_hashes),
                  optimizer_sha256=digest(expected_optimizer_hashes))
            await boundary(session, "first")
            session.close()
            session = None
            gc_started = time.monotonic()
            import gc
            gc.collect()
            mx.clear_cache()
            event("resident_close", gc_started)
            started = time.monotonic()
            session = await create_session(config, checkpoint_root=checkpoint_root, checkpoint=recovery)
            actual_tokens, actual_probabilities, _, _ = session.draw_pass(
                session.prepare_prompt(prompt), maximum=8, stop_ids=set())
            assert actual_tokens == expected_tokens and actual_probabilities == expected_probabilities
            event("resident_recovery_rng", started, exact_token_and_behavior_probability_match=True,
                  sampled_tokens=actual_tokens, behavior_logprobs=actual_probabilities)
            started = time.monotonic()
            repeated = await session.update(batch, learning_rate=config.train.learning_rate, clip_low=0., clip_high=4.)
            actual_optimizer, actual_metadata = session.optimizer.state()
            assert actual_metadata == expected_metadata
            assert tensor_hashes(session.adapter.state()) == expected_adapter_hashes
            assert tensor_hashes(actual_optimizer) == expected_optimizer_hashes
            assert session.rng.tolist() == expected_rng
            actual_optimizer = None
            event("resident_recovery_identical_next_update", started, bitwise_bf16_adapter_match=True,
                  bitwise_fp32_masters_and_moments_match=True, exact_rng_match=True, metrics=repeated)
            await boundary(session, "recovered-first")

            if args.context_gate:
                from shellsmith.policy.sampler import PlayerRequestError
                started = time.monotonic()
                overflow_payload = {"messages": [{"role": "user", "content": " item" * 30700}],
                                    "reasoning_effort": "xhigh", "max_tokens": 8192}
                overflow_prompt, _, _ = session.renderer.render(overflow_payload, thinking=True)
                rng_before = session.rng.tolist()
                try:
                    session.sampler.sample(overflow_payload, request_id="qualify-native-overflow", session_id="qualification")
                except PlayerRequestError as error:
                    assert error.status == 422 and "context_length_exceeded" in str(error)
                else:
                    raise AssertionError("Exact rendered overflow did not reject before generation.")
                assert session.rng.tolist() == rng_before
                event("native_context_budget_rejected_before_generation", started,
                      prompt_tokens=overflow_prompt.length, requested_completion_tokens=8192,
                      available=config.mlx.context_window - overflow_prompt.length, exact_rng_unchanged=True)

                started = time.monotonic()
                repeats = 32750
                capped_payload = {"messages": [{"role": "user", "content": "Reply with only OK." + " item" * repeats}],
                                  "reasoning_effort": "none", "max_tokens": 2}
                capped_prompt, _, _ = session.renderer.render(capped_payload, thinking=False)
                repeats += 32766 - capped_prompt.length
                capped_payload["messages"][0]["content"] = "Reply with only OK." + " item" * repeats
                capped_prompt, _, _ = session.renderer.render(capped_payload, thinking=False)
                assert capped_prompt.length == 32766
                capped_record, capped_assistant = session.sampler.sample(capped_payload,
                    request_id="qualify-native-explicit-cap", session_id="qualification")
                assert len(capped_record.sampled_tokens) <= 2
                event("native_explicit_two_token_cap_at32766", started, prompt_tokens=capped_record.prompt_tokens,
                      explicit_completion_cap=2, sampled_tokens=capped_record.sampled_tokens,
                      behavior_logprobs=capped_record.sampled_logprobs, assistant=capped_assistant,
                      prompt_sha256=digest(capped_record.prompt.to_ints()))
                started = time.monotonic()
                # Preserve the native rendered assistant controls and fill only
                # the user-body middle with an explicit repeated content token.
                filler = session.renderer.tokenizer.encode(" item", add_special_tokens=False)[0]
                target = config.mlx.context_window - args.completion_tokens
                native_ids = record.prompt.to_ints()
                long_prompt = tuple(native_ids[:8] + [filler] * (target - len(native_ids)) + native_ids[8:])
                assert len(long_prompt) == target
                # A deliberately teacher-forced numerical/memory fixture scores
                # the exact conditional selected-token probabilities. Actual
                # stochastic generation is qualified separately above.
                from shellsmith.mlx.replay import Replay
                corpus = session.renderer.tokenizer.encode("Qualification replay preserves complete context and gradients. ",
                                                            add_special_tokens=False)
                tokens = tuple(corpus[index % len(corpus)] for index in range(args.completion_tokens - 1)) + (next(iter(session.renderer.eos_ids)),)
                session.model.eval()
                def report(progress):
                    event("context_32k_" + progress["phase"], started,
                          segment=progress["segment"], segments=progress["segments"])
                current, _, _, _ = Replay(session.lm, progress=report).run(long_prompt, tokens)
                probabilities = tuple(current.tolist())
                document["context_sequence"] = {"kind": "synthetic_teacher_forced", "prompt_token_ids": long_prompt,
                    "completion_token_ids": tokens, "behavior_logprobs": probabilities}
                event("context_32k_teacherforced_likelihood", started, prompt_tokens=target,
                      completion_tokens=len(tokens), vocabulary_head_tiles=sum(
                          (stop - max(start, target - 1) + 127) // 128
                          for start in range(0, target + len(tokens) - 1, 2048)
                          if (stop := min(start + 2048, target + len(tokens) - 1)) > max(start, target - 1)),
                      source="synthetic_teacher_forced", token_sha256=digest(tokens), behavior_sha256=digest(probabilities))
                advantages = tuple(0.01 if index % 2 == 0 else -0.01 for index in range(len(tokens)))
                long_sequence = TokenSequence(long_prompt, tokens, probabilities, advantages,
                                              (1.,) * len(tokens), "qualify-32k")
                started = time.monotonic()
                session.replay_progress = report
                try:
                    metrics = await session.update(TokenBatch((long_sequence,)), learning_rate=config.train.learning_rate,
                                                   clip_low=0., clip_high=4.)
                finally:
                    session.replay_progress = None
                event("context_32k_fullprompt_update", started, prompt_tokens=target,
                      completion_tokens=len(tokens), context_tokens=target + len(tokens),
                      source="synthetic_teacher_forced", metrics=metrics)
                await boundary(session, "context-32k")

            for index in range(args.soak_updates):
                started = time.monotonic()
                current, _ = session.sampler.sample(payload, request_id=f"qualify-soak-{index}", session_id="qualification")
                metrics = await session.update(TokenBatch((sequence(current),)), learning_rate=config.train.learning_rate,
                                               clip_low=0., clip_high=4.)
                await boundary(session, f"soak-{index}")
                event("generation_update_soak", started, iteration=index, completion_tokens=len(current.sampled_tokens), metrics=metrics)
            started = time.monotonic()
            frozen_after = frozen_hashes(session)
            assert frozen_after == frozen_before
            event("frozen_base_final_hash", started, all_frozen_tensors_bitwise_unchanged=True,
                  tensors=len(frozen_after), sha256=digest(frozen_after))
            document["status"] = "passed"
    except BaseException as error:
        document["status"] = "failed"
        document["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        if session is not None:
            session.close()
        document["finished_at_unix"] = time.time()
        document["peak_memory_bytes"] = mx.get_peak_memory()
        persist()


async def qualify_fresh_process_recovery(args):
    """Verify the prior run's next update after a genuinely fresh Python start."""
    import mlx.core as mx
    from shellsmith.mlx import create_session
    from shellsmith.mlx.checkpoint import verify_checkpoint
    from shellsmith.train.backend import TokenBatch, TokenSequence
    from shellsmith.train.config import TrainConfig

    source = json.loads(Path(args.proof_source).read_text())
    phases = {row["phase"]: row for row in source["phases"]}
    origin = Path(phases["full_checkpoint_origin"]["state_path"])
    expected_state = origin.parent / "first-state"
    origin_document = verify_checkpoint(origin)
    expected_document = verify_checkpoint(expected_state)
    payload = {"messages": [{"role": "user", "content": "Reply with only OK."}], "reasoning_effort": "none"}
    config = TrainConfig()
    config = replace(config, compute=replace(config.compute, backend="mlx"), mlx=replace(config.mlx, model_path=args.model_path))
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = {"format": "shellsmith-mlx-fresh-process-recovery-v1", "status": "running",
              "origin_state": str(origin), "expected_next_state": str(expected_state),
              "python_pid": os.getpid(), "source": str(Path(args.proof_source).absolute())}

    def write():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")

    def equal(actual, expected):
        if set(actual) != set(expected):
            return False
        return all(value.dtype == expected[name].dtype and value.shape == expected[name].shape and
                   bool(mx.array_equal(value, expected[name]).item()) for name, value in actual.items())

    session = None
    started = time.monotonic()
    try:
        with gpu_session_lifetime(lambda: session):
            session = await create_session(config, checkpoint_root=Path(args.checkpoint_root),
                checkpoint={"path": str(origin), **origin_document["trainer_position"]})
            prompt, _, _ = session.renderer.render(payload, thinking=False)
            wanted = phases.get("resident_recovery_rng", phases.get("process_recovery_rng"))
            tokens, probabilities, _, _ = session.draw_pass(session.prepare_prompt(prompt.to_ints()), maximum=8, stop_ids=set())
            assert tokens == wanted["sampled_tokens"] and probabilities == wanted["behavior_logprobs"]
            generated = phases["plain_generation"]
            n = len(generated["sampled_tokens"])
            batch = TokenBatch((TokenSequence(tuple(prompt.to_ints()), tuple(generated["sampled_tokens"]),
                    tuple(generated["behavior_logprobs"]), (0.01,) * n, (1.,) * n, "qualify-plain"),))
            metrics = await session.update(batch, learning_rate=config.train.learning_rate, clip_low=0., clip_high=4.)
            actual_optimizer, metadata = session.optimizer.state()
            assert metadata == expected_document["optimizer"]
            assert equal(session.adapter.state(), mx.load(str(expected_state / "adapters.safetensors")))
            assert equal(actual_optimizer, mx.load(str(expected_state / "optimizer.safetensors")))
            assert equal({"key": session.rng}, mx.load(str(expected_state / "rng.safetensors")))
            result.update(status="passed", exact_sample_ids_and_behavior_probabilities=True,
                          bitwise_bf16_adapters=True, bitwise_fp32_masters_moments=True, exact_rng=True, metrics=metrics)
    except BaseException as error:
        result.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if session is not None:
            session.close()
        result.update(seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory())
        write()
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(Path.home() / "models/Qwen3.8-27B-8bit"))
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-gate", action="store_true")
    parser.add_argument("--completion-tokens", type=int, default=8192,
                        help="Synthetic 32K gate target count (default: 8192 targets +24576 prompt)")
    parser.add_argument("--soak-updates", type=int, default=2)
    parser.add_argument("--proof-source", help="Prior qualification JSON: run only the fresh-process checkpoint recovery proof")
    args = parser.parse_args()
    if args.soak_updates < 0:
        parser.error("--soak-updates must be nonnegative")
    if not 1 <= args.completion_tokens <= 8192:
        parser.error("--completion-tokens must lie in 1..8192")
    if os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error("MLX_ENABLE_TF32 must be 0 before importing the pinned numerical runtime")
    asyncio.run(qualify_fresh_process_recovery(args) if args.proof_source else qualify(args))


if __name__ == "__main__":
    main()
