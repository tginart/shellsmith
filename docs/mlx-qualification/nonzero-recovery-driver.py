"""Two fresh Python invocations prove restoration of populated Adam moments.

Use the frozen qualification PYTHONPATH and Studio Python. Run --phase baseline
then --phase fresh with the same --reference/--control/--checkpoint-root paths.
No production backend source or original model/environment is changed.
"""

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("baseline", "fresh"), required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    if os.environ["MLX_ENABLE_TF32"] != "0":
        parser.error("MLX_ENABLE_TF32 must be0 before loading MLX")
    asyncio.run(run(args))


async def run(args):
    import mlx.core as mx
    from shellsmith.jobs.lease import GpuLease
    from shellsmith.mlx import create_session
    from shellsmith.mlx.checkpoint import verify_checkpoint
    from shellsmith.train.backend import TokenBatch, TokenSequence
    from shellsmith.train.config import TrainConfig

    def equal(actual, wanted):
        return set(actual) == set(wanted) and all(value.dtype == wanted[name].dtype and
            value.shape == wanted[name].shape and bool(mx.array_equal(value, wanted[name]).item())
            for name, value in actual.items())

    reference = json.loads(args.reference.read_text())
    rows = {row["phase"]: row for row in reference["phases"]}
    original = Path(rows["full_checkpoint_origin"]["state_path"]).parent / "first-state"
    original_doc = verify_checkpoint(original)
    assert original_doc["policy_version"] == 1
    config = TrainConfig()
    config = replace(config, compute=replace(config.compute, backend="mlx"))
    started, session = time.monotonic(), None
    result = {"format": "shellsmith-mlx-nonzero-recovery-v1", "phase": args.phase,
              "origin_checkpoint_step": 1, "next_update_step": 2, "python_pid": os.getpid(),
              "policy_identity_sha256": reference["policy_identity"]["identity_sha256"]}
    with GpuLease():
        try:
            session = await create_session(config, checkpoint_root=args.checkpoint_root,
                checkpoint={"path": str(original), **original_doc["trainer_position"]})
            assert session.policy_identity == reference["policy_identity"]
            original_tensors, original_metadata = session.optimizer.state()
            assert original_metadata == original_doc["optimizer"]
            assert equal(original_tensors, mx.load(str(original / "optimizer.safetensors")))
            assert any(float(mx.sum(mx.abs(value)).item()) > 0 for name, value in original_tensors.items()
                       if name.startswith("first."))
            original_tensors = None
            result["populated_optimizer_roundtrip_exact"] = True
            prompt, _, _ = session.renderer.render({"messages": [{"role": "user", "content": "Reply with only OK."}]}, thinking=False)
            tokens, probabilities, _, cursor = session.draw_pass(session.prepare_prompt(prompt.to_ints()), maximum=8, stop_ids=set())
            del cursor
            generated = rows["plain_generation"]
            n = len(generated["sampled_tokens"])
            batch = TokenBatch((TokenSequence(tuple(prompt.to_ints()), tuple(generated["sampled_tokens"]),
                tuple(generated["behavior_logprobs"]), (0.01,) * n, (1.,) * n, "qualify-nonzero"),))
            metrics = await session.update(batch, learning_rate=config.train.learning_rate, clip_low=0., clip_high=4.)
            result.update(sampled_tokens=tokens, behavior_logprobs=probabilities, metrics=metrics)
            if args.phase == "baseline":
                if args.control.exists():
                    raise FileExistsError(args.control)
                sampler = await session.save_checkpoint(kind="sampler", name="nonzero-next-sampler")
                metadata = {"step": 2, "sampler_path": sampler, "backend": "mlx", "policy_identity": session.policy_identity}
                expected = await session.save_checkpoint(kind="state", name="nonzero-next-state", metadata=metadata)
                await session.publish_policy(revision=2)
                result.update(status="baseline_saved", expected_next_state=expected)
                args.control.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            else:
                control = json.loads(args.control.read_text())
                assert control["python_pid"] != os.getpid()
                assert tokens == control["sampled_tokens"] and probabilities == control["behavior_logprobs"]
                expected = Path(control["expected_next_state"])
                document = verify_checkpoint(expected, expected_identity=session.policy_identity)
                actual_tensors, actual_metadata = session.optimizer.state()
                assert actual_metadata == document["optimizer"]
                assert equal(actual_tensors, mx.load(str(expected / "optimizer.safetensors")))
                assert equal(session.adapter.state(), mx.load(str(expected / "adapters.safetensors")))
                assert equal({"key": session.rng}, mx.load(str(expected / "rng.safetensors")))
                result.update(status="passed", distinct_python_process=True,
                    bitwise_bf16_adapters=True, bitwise_fp32_masters_moments=True, exact_rng=True,
                    exact_sample_ids_and_behavior_probabilities=True, baseline_python_pid=control["python_pid"])
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        finally:
            if session is not None:
                session.close()
    result.update(seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory())
    target = args.control if args.phase == "baseline" else args.output
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
