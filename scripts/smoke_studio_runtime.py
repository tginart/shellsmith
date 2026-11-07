#!/usr/bin/env python3
"""Bounded real-27B RPC, cache, scheduling and fresh-process recovery probe.

Run under run_bounded_studio.py. Child services inherit its owned process group.
No GM or agent episodes are launched here.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from studio_compute.client import Client
from studio_compute.storage import atomic_json


def restore_control(client, checkpoint, call):
    """Release the isolation probe's live acquisition before restoring state."""
    call(client, "suspend", {}, "control-suspend")
    return call(client, "restore", {"checkpoint": checkpoint}, "control-restore")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--activation-budget-gib", type=float,
                        help="Optional saved-activation cap for an explicit interleave probe.")
    args = parser.parse_args()
    root = Path(args.output_root);root.mkdir(exist_ok=False)
    report = {"status": "running", "checks": {}, "operations": [], "services": []}
    process = None;log = None
    a = Client(f"http://127.0.0.1:{args.port}", "smoke-a", timeout=120)
    b = Client(a.endpoint, "smoke-b", timeout=120)

    def record():atomic_json(root / "report.json", report)

    def stop():
        nonlocal process, log
        if process and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:process.kill();process.wait();raise
        if log:log.close()
        process = None;log = None

    def start(cap="auto"):
        nonlocal process, log
        index = len(report["services"])
        log = (root / f"service-{index}.log").open("xb")
        activation_options = ([] if args.activation_budget_gib is None else
                              ["--activation-budget-gib", str(args.activation_budget_gib)])
        process = subprocess.Popen([sys.executable, "-u", "-m", "studio_compute.cli", "serve",
            "--model", args.model, "--state-root", str(root / "state"), "--port", str(args.port),
            "--headroom-gib", "16", "--prefix-cache-gib", cap, *activation_options], stdout=log, stderr=subprocess.STDOUT)
        report["services"].append({"pid": process.pid, "cache_cap": cap})
        until = time.monotonic() + 180
        while time.monotonic() < until:
            if process.poll() is not None:raise RuntimeError("Service exited during startup.")
            try:
                capabilities = a.capabilities()
                report["services"][-1]["capabilities"] = capabilities
                record();return capabilities
            except OSError:time.sleep(.25)
        raise TimeoutError("Service startup exceeded 180 seconds.")

    def call(client, kind, body, name):
        started = time.monotonic()
        value = client.call(kind, body, request_id=name)
        report["operations"].append({"id": name, "kind": kind, "wall_seconds": time.monotonic() - started,
            "result": value})
        record();return value

    try:
        capabilities = start()
        atomic_json(root / "ready.json", {"ready": True, "pid": process.pid})
        assert capabilities["runtime"]["prefix_cache_mode"] == "available_memory"
        assert capabilities["host_headroom_bytes"] == 16 * 1024**3
        for client in (a, b):call(client, "create", {"seed": 17 if client is a else 23}, "create-" + client.client_id)
        p = call(a, "publish", {"revision": 0}, "publish-zero")["policy"]
        before = call(a, "inspect", {"hash_state": True, "hash_base": True}, "before")
        other = call(b, "inspect", {"hash_state": True}, "other-before")
        assert before["base_allocation"] == other["base_allocation"]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        unit = tokenizer.encode(" item", add_special_tokens=False)
        prompt = (unit * 512)[:512]
        sample = {"policy": p, "prompt_token_ids": prompt, "max_tokens": 8, "seed": 31, "cache_precision": "bf16"}
        cold = call(a, "sample", {**sample, "use_prefix_cache": False}, "cold")
        call(a, "sample", sample, "store-prefix")
        warm = call(a, "sample", sample, "warm")
        assert warm["prefix_cache_tokens"] == len(prompt)
        assert warm["sampled_tokens"] == cold["sampled_tokens"]
        error = max(abs(x-y) for x,y in zip(warm["behavior_logprobs"], cold["behavior_logprobs"]))
        assert error < 1e-5
        report["checks"]["prefix_reuse"] = {"passed": True, "logprob_error": error,
            "cold_actual_prefill_tokens": len(prompt), "warm_actual_prefill_tokens": 0}
        score = {"policy": p, "prompt_token_ids": prompt + cold["sampled_tokens"], "target_token_ids": [unit[0]] * 4}
        extended = call(a, "logprobs", score, "extended-warm")
        extended_cold = call(a, "logprobs", {**score, "use_prefix_cache": False}, "extended-cold")
        error = max(abs(x-y) for x,y in zip(extended["behavior_logprobs"], extended_cold["behavior_logprobs"]))
        assert error < .15  # BF16 segmentation/recurrent rounding; record actual error.
        report["checks"]["prefix_extension"] = {"passed": True, "bf16_logprob_error": error,
            "actual_prefill_tokens": len(score["prompt_token_ids"]) - extended["prefix_cache_tokens"]}
        # Mixed prompts and output limits share request-local decode rows.
        parallel = {**sample, "prompt_token_ids": (unit * 1024)[:1024], "use_prefix_cache": False}
        requests = [{**parallel, "seed": 40 + index,
                     "prompt_token_ids": (unit * (1024 + index * 128))[:1024 + index * 128],
                     "max_tokens": 12 + index} for index in range(8)]
        submitted = time.monotonic()
        for index, request in enumerate(requests):a.submit("sample", request, request_id=f"eight-{index}")
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda index:call(a,"sample",requests[index],f"eight-{index}"),range(8)))
        sizes = [x.get("batch_size", 1) for x in results]
        assert max(sizes) > 1 and max(sizes) <= capabilities["max_inference_batch"]
        report["checks"]["eight_requests"] = {"passed": True, "batch_sizes": sizes,
            "wall_seconds": time.monotonic()-submitted,
            "actual_prefill_tokens": sum(len(request["prompt_token_ids"]) for request in requests)}
        a.submit("sample", {**sample, "prompt_token_ids": prompt[:32],
                            "max_tokens": 128}, request_id="cancel-active")
        until = time.monotonic() + 15
        while a.operation("cancel-active")["status"] == "queued":
            if time.monotonic() > until:
                raise TimeoutError("Cancellation probe was not admitted.")
            time.sleep(.01)
        a.cancel("cancel-active")
        canceled = a.operation("cancel-active", wait=15)
        assert canceled["status"] == "interrupted", canceled
        report["checks"]["active_cancellation"] = {"passed": True,
            "terminal_status": canceled["status"]}
        training_prompt = tokenizer.encode("Question: Why should software be tested?\nAnswer:", add_special_tokens=False)
        background = tokenizer.encode("Background notes for a software quality review. ", add_special_tokens=False)
        training_prompt = (background * 256)[:256-len(training_prompt)] + training_prompt
        sequences = []
        for index, (text, advantage) in enumerate([
            (" Tests catch bugs early and improve reliability.", .1),
            (" Reviews alone cannot establish correct program behavior.", -.1),
        ]):
            targets = tokenizer.encode(text, add_special_tokens=False)[:8]
            scored = call(a,"logprobs",{"policy":p,"prompt_token_ids":training_prompt,
                "target_token_ids":targets},f"training-score-{index}")
            sequences.append({"prompt_token_ids":training_prompt,"sampled_tokens":targets,
                "behavior_logprobs":scored["behavior_logprobs"],"advantages":[advantage]*len(targets)})
        backward = {"loss": "cispo", "sequences": sequences}
        a.submit("forward_backward", backward, request_id="tiny-backward")
        # Another client's inference must finish while replay still has units.
        inference = call(b, "sample", {**sample,"prompt_token_ids":prompt[:32],"max_tokens":1}, "interleaved")
        status = a.operation("tiny-backward")["status"]
        assert status in ("running", "queued"), status
        forward = call(a, "forward_backward", backward, "tiny-backward")
        assert math.isfinite(forward["forward_metrics"]["loss"])
        report["checks"]["fairness"] = {"passed": True, "backward_status_at_sample_completion":status}
        call(a,"optim_step",{"learning_rate":5e-5,"backward_operation":"tiny-backward"},"tiny-optimizer")
        after = call(a,"inspect",{"hash_state":True,"hash_base":True},"after")
        assert after["frozen_base_hash"] == before["frozen_base_hash"]
        isolated = call(b,"inspect",{"hash_state":True},"other-after")
        assert isolated["state_hashes"] == other["state_hashes"]
        published = call(a,"publish",{"revision":1},"publish-one")
        saved = call(a,"save",{},"save-one")
        restore_control(b, saved["checkpoint"], call)
        next_sample = {"policy":published["policy"],"prompt_token_ids":prompt[:64],"max_tokens":8}
        expected = call(b,"sample",next_sample,"expected-next")
        stop();start()
        restored = call(a,"inspect",{"hash_state":True,"hash_base":True},"restored")
        assert restored["state_hashes"] == after["state_hashes"]
        assert restored["frozen_base_hash"] == before["frozen_base_hash"]
        actual = call(a,"sample",next_sample,"actual-next")
        assert actual["sampled_tokens"] == expected["sampled_tokens"]
        assert actual["behavior_logprobs"] == expected["behavior_logprobs"]
        assert after["state_hashes"]["adapter"] != before["state_hashes"]["adapter"]
        report["checks"]["training_recovery"] = {"passed":True,
            "prompt_tokens":sum(len(x["prompt_token_ids"]) for x in sequences),
            "completion_tokens":sum(len(x["sampled_tokens"]) for x in sequences),
            "checkpoint":saved["checkpoint"],"publication":published,"exact_state_and_next_sample":True}
        # Bound real retained caches without filling the Studio allocator pool.
        stop();start("0.5")
        for index in range(6):
            last = call(a,"sample",{**next_sample,"prompt_token_ids":[unit[0]+index]*512,"seed":31},f"eviction-{index}")
        cache = last["execution_metrics"]["prefix_cache"]
        assert cache["bytes"] <= cache["limit_bytes"] and cache["evictions"] > 0, cache
        report["checks"]["bounded_eviction"] = {"passed":True,"cache":cache}
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop();record()
        print(json.dumps({"status":report["status"],"checks":report["checks"]}),flush=True)


if __name__ == "__main__":main()
