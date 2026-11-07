#!/usr/bin/env python3
"""Direct 262,144-token service qualification, with fresh-process recovery.

This records capacity/training/recovery only. It neither grows context through
episodes nor claims agent-behavior qualification. No 32K/128K online stages.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.error import URLError

from studio_compute.client import Client
from studio_compute.storage import atomic_json
from studio_compute.tokenizer import materialize

WINDOW, PROMPT, TARGET = 262144, 253952, 8192


def host():
    rss = subprocess.check_output(["ps", "-axo", "pid=,rss=,comm="], text=True)
    return {
        "time": time.time(),
        "swap": subprocess.check_output(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True
        ).strip(),
        "processes": rss.splitlines(),
    }


def exact_ids(tokenizer, length, *, opening=None, closing=None):
    prefix = tokenizer.encode(
        opening or "<|im_start|>user\nRead this document carefully.\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        closing
        or "\nSummarize briefly.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    unit = tokenizer.encode(" item", add_special_tokens=False)
    if len(unit) != 1 or length < len(prefix) + len(suffix):
        raise ValueError(
            "Exact fixture requires a one-token filler and room for controls."
        )
    result = prefix + unit * (length - len(prefix) - len(suffix)) + suffix
    assert len(result) == length
    return result


def fact_ids(tokenizer):
    prefix = tokenizer.encode(
        "<|im_start|>system\nRead the entire document. Later facts replace earlier facts with the same key. Answer only the requested JSON."
        "<|im_end|>\n<|im_start|>user\nALPHA = copper731\n",
        add_special_tokens=False,
    )
    middle = tokenizer.encode(
        "\nMID = violet219\nOMEGA = amber555\n", add_special_tokens=False
    )
    suffix = tokenizer.encode(
        "\nOMEGA = jade887\nEnd of document. Return current values for ALPHA, MID, OMEGA as a JSON object."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )
    item = tokenizer.encode(" item", add_special_tokens=False)
    assert len(item) == 1
    remaining = WINDOW - 512 - len(prefix) - len(middle) - len(suffix)
    ids = (
        prefix
        + item * (remaining // 2)
        + middle
        + item * (remaining - remaining // 2)
        + suffix
    )
    assert len(ids) == WINDOW - 512
    return ids, {
        "beginning_token": 0,
        "middle_token": len(prefix) + remaining // 2,
        "end_token": len(ids) - len(suffix),
        "facts": {"ALPHA": "copper731", "MID": "violet219", "OMEGA": "jade887"},
        "superseded": {"OMEGA": "amber555"},
    }


def qualify(args):
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    state_root = root / "service-state"
    report = {
        "scope": "capacity_training_recovery",
        "status": "running",
        "context_window": WINDOW,
        "required_prompt_tokens": PROMPT,
        "required_completion_targets": TARGET,
        "events": [],
        "processes": [],
        "online_32k_128k": "skipped",
        "agent_behavior_qualification": False,
        "junior_run": "not_started",
    }
    output = root / "report.json"
    process = None
    client = Client(f"http://127.0.0.1:{args.port}", "capacity", timeout=48 * 3600)

    def event(name, **values):
        row = {"phase": name, "time": time.time(), **values}
        report["events"].append(row)
        atomic_json(output, report)
        print(json.dumps(row), flush=True)

    def start():
        nonlocal process
        log = (root / f"service-{len(report['processes']) + 1}.log").open("wb")
        command = [
            sys.executable,
            "-u",
            "-m",
            "studio_compute.cli",
            "serve",
            "--model",
            args.model,
            "--state-root",
            str(state_root),
            "--port",
            str(args.port),
            "--headroom-gib",
            str(args.headroom_gib),
            "--checkpoint-layers",
            "8",
        ]
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        log.close()
        report["processes"].append(
            {"pid": process.pid, "started": time.time(), "command": command}
        )
        atomic_json(output, report)
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Service startup failed; see service log (exit {process.returncode})."
                )
            try:
                capability = client.capabilities()
                if capability["context_window"] != WINDOW:
                    raise AssertionError(
                        "Service did not admit the exact target context."
                    )
                event("service_ready", pid=process.pid, capabilities=capability)
                return capability
            except (URLError, ConnectionError, TimeoutError):
                time.sleep(1)
        raise TimeoutError("Service did not become ready within 15 minutes.")

    def stop():
        nonlocal process
        if process and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            process.wait(timeout=120)
            report["processes"][-1].update(
                stopped=time.time(), exit_code=process.returncode
            )
            event("service_stopped", pid=process.pid, exit_code=process.returncode)

    def call(kind, body, id, *, rpc=client):
        started = time.monotonic()
        rpc.submit(kind, body, request_id=id)
        last_phase = None
        last_print = 0
        while True:
            op = rpc.operation(id, wait=30)
            progress = op.get("progress") or {}
            phase = progress.get("phase")
            if phase != last_phase or time.monotonic() - last_print >= 60:
                print(
                    json.dumps(
                        {"operation": id, "status": op["status"], "progress": progress}
                    ),
                    flush=True,
                )
                last_phase, last_print = phase, time.monotonic()
            if op["status"] not in ("queued", "running"):
                if op["status"] != "succeeded":
                    raise RuntimeError(op)
                event(
                    id,
                    kind=kind,
                    seconds=time.monotonic() - started,
                    result=op["result"],
                )
                return op["result"]

    def files(reference):
        folder = state_root / "checkpoints" / reference.split(":")[1]
        doc = json.loads((folder / "manifest.json").read_text())
        return doc["files"], doc["state"]

    try:
        report["host_before"] = host()
        atomic_json(output, report)
        start()
        call("create", {"seed": 31}, "create")
        policy = call("publish", {"revision": 0}, "base-publication")["policy"]
        bundle = client.tokenizer_bundle()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            materialize(bundle), local_files_only=True, trust_remote_code=False
        )
        report["tokenizer_hash"] = bundle["hash"]
        frozen = call("inspect", {"hash_base": True}, "frozen-before")[
            "frozen_base_hash"
        ]
        # The overflow request must fail before a receipt or RNG mutation exists.
        near = exact_ids(tokenizer, WINDOW - 2)
        state_before = call("inspect", {}, "before-overflow")
        try:
            client.submit(
                "sample",
                {"policy": policy, "prompt_token_ids": near, "max_tokens": 3},
                request_id="overflow",
            )
        except ValueError as error:
            event(
                "overflow_rejected",
                prompt_tokens=len(near),
                completion_budget=3,
                error=str(error),
            )
        else:
            raise AssertionError("Overflow was admitted.")
        state_after = call("inspect", {}, "after-overflow")
        assert state_before["rng"] == state_after["rng"]
        near_result = call(
            "sample",
            {
                "policy": policy,
                "prompt_token_ids": near,
                "max_tokens": 2,
                "stop_token_ids": [],
                "cache_precision": "bf16",
            },
            "exact-admission-near-limit",
        )
        assert near_result["prompt_tokens"] + near_result["completion_tokens"] == WINDOW
        facts, positions = fact_ids(tokenizer)
        result = call(
            "sample",
            {
                "policy": policy,
                "prompt_token_ids": facts,
                "max_tokens": 512,
                "stop_token_ids": tokenizer.encode(
                    "<|im_end|>", add_special_tokens=False
                ),
                "cache_precision": "bf16",
            },
            "retrieval",
        )
        text = tokenizer.decode(result["sampled_tokens"], skip_special_tokens=True)
        try:
            # Accept an optional Markdown fence, but require the current values
            # under their actual keys, including the superseded OMEGA fact.
            answer = json.loads(text[text.index("{") : text.rindex("}") + 1])
        except (ValueError, json.JSONDecodeError):
            answer = None
        retrieval = answer == positions["facts"]
        event(
            "retrieval_checked",
            prompt_tokens=len(facts),
            text=text,
            parsed_answer=answer,
            fixture=positions,
            passed=retrieval,
        )
        # Complete training/recovery even if behavior of this retrieval fixture
        # fails, keeping capacity evidence separate and withholding junior.
        prompt = exact_ids(tokenizer, PROMPT)
        target_unit = tokenizer.encode(
            " The result is finite and the immutable base stays frozen.\n",
            add_special_tokens=False,
        )
        targets = (target_unit * ((TARGET + len(target_unit) - 1) // len(target_unit)))[
            :TARGET
        ]
        targets[-1] = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
        sequence = {
            "prompt_token_ids": prompt,
            "sampled_tokens": targets,
            "behavior_logprobs": [-5.0] * TARGET,
            "advantages": [1.0 / TARGET] * TARGET,
            "loss_mask": [1.0] * TARGET,
        }
        report["input_sha256"] = hashlib.sha256(
            json.dumps(sequence, sort_keys=True).encode()
        ).hexdigest()
        atomic_json(root / "full-context-input.json", sequence)
        checkpoint = None
        for update in (1, 2):
            forward = call(
                "forward_backward",
                {"sequences": [sequence], "loss": "cispo", "reduction": "sum"},
                f"full-{update}-backward",
            )
            assert (
                forward["forward_metrics"]["prompt_tokens"] == PROMPT
                and forward["forward_metrics"]["completion_tokens"] == TARGET
            )
            result = call(
                "optim_step",
                {
                    "learning_rate": 5e-5,
                    "backward_operation": f"full-{update}-backward",
                },
                f"full-{update}-optimizer",
            )
            assert result["optim_metrics"]["optimizer_steps"] == update
            publication = call(
                "publish", {"revision": update}, f"full-{update}-publication"
            )
            assert publication["checkpoint"] == result["checkpoint"]
            checkpoint = result["checkpoint"]
        after = call(
            "inspect", {"hash_base": True, "hash_state": True}, "frozen-after-full"
        )
        assert after["frozen_base_hash"] == frozen
        expected_files, expected_state = files(checkpoint)
        # A separate client forks update 2. Its default RNG and next optimizer
        # update provide controls without advancing the primary client's state.
        fork = Client(client.endpoint, "recovery-control", timeout=client.timeout)
        call("restore", {"checkpoint": checkpoint}, "control-restore", rpc=fork)
        fork_policy = call("publish", {"revision": 2}, "control-publication", rpc=fork)[
            "policy"
        ]
        short = exact_ids(tokenizer, 256)
        control_sample = call(
            "sample",
            {"policy": fork_policy, "prompt_token_ids": short, "max_tokens": 8},
            "control-next-sample",
            rpc=fork,
        )
        bounded = {
            "prompt_token_ids": short,
            "sampled_tokens": control_sample["sampled_tokens"],
            "behavior_logprobs": control_sample["behavior_logprobs"],
            "advantages": [0.1] * 8,
            "loss_mask": [1.0] * 8,
        }
        call(
            "forward_backward",
            {"sequences": [bounded], "loss": "cispo"},
            "control-next-backward",
            rpc=fork,
        )
        control = call(
            "optim_step",
            {"learning_rate": 5e-5, "backward_operation": "control-next-backward"},
            "control-next-optimizer",
            rpc=fork,
        )
        control_hashes = call(
            "inspect", {"hash_state": True}, "control-state", rpc=fork
        )["state_hashes"]
        control_files, _ = files(control["checkpoint"])
        old_pid = process.pid
        report["stop_after_update"] = 2
        report["checkpoint_at_stop"] = checkpoint
        stop()
        report["host_at_stop"] = host()
        atomic_json(output, report)
        start()
        assert process.pid != old_pid
        assert (
            client.operation("full-2-optimizer")["result"]["checkpoint"] == checkpoint
        )
        restored = call("save", {"kind": "state"}, "restored-state")
        restored_files, restored_state = files(restored["checkpoint"])
        restored_hashes = call(
            "inspect", {"hash_state": True}, "restored-state-hashes"
        )["state_hashes"]
        assert restored_hashes == after["state_hashes"]
        assert (
            restored_state["revision"] == 2
            and restored_state["rng_key"] == expected_state["rng_key"]
        )
        policy = call("publish", {"revision": 2}, "restored-policy")["policy"]
        reproduced = call(
            "sample",
            {"policy": policy, "prompt_token_ids": short, "max_tokens": 8},
            "reproduced-next-sample",
        )
        assert reproduced["sampled_tokens"] == control_sample["sampled_tokens"]
        assert reproduced["behavior_logprobs"] == control_sample["behavior_logprobs"]
        event(
            "fresh_process_recovery",
            old_pid=old_pid,
            new_pid=process.pid,
            adapter_optimizer_rng_exact=True,
            next_sample_exact=True,
        )
        for update in range(3, 13):
            call(
                "forward_backward",
                {"sequences": [bounded], "loss": "cispo"},
                f"bounded-{update}-backward",
            )
            result = call(
                "optim_step",
                {
                    "learning_rate": 5e-5,
                    "backward_operation": f"bounded-{update}-backward",
                },
                f"bounded-{update}-optimizer",
            )
            assert result["optim_metrics"]["optimizer_steps"] == update
            if update == 3:
                reproduced_files, _ = files(result["checkpoint"])
                reproduced_hashes = call(
                    "inspect", {"hash_state": True}, "reproduced-state-hashes"
                )["state_hashes"]
                assert reproduced_hashes == control_hashes
                event(
                    "next_bounded_update_reproduced",
                    files=reproduced_files,
                    control_files=control_files,
                    state_hashes=reproduced_hashes,
                    exact=True,
                )
            policy = call(
                "publish", {"revision": update}, f"bounded-{update}-publication"
            )["policy"]
        final = call("inspect", {"hash_base": True}, "frozen-after-12")
        assert final["frozen_base_hash"] == frozen and final["optimizer_steps"] == 12
        report.update(
            status="passed" if retrieval else "failed_retrieval",
            capacity_training_recovery_passed=True,
            retrieval_passed=retrieval,
            updates_completed=12,
            final_policy=policy,
            frozen_base_unchanged=True,
        )
        event("qualification_finished", status=report["status"])
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        event("qualification_failed", error=report["error"])
        raise
    finally:
        stop()
        report["host_after"] = host()
        atomic_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--headroom-gib", type=float, default=16)
    qualify(parser.parse_args())
