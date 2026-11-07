"""
Purpose: Measure the real memory of one recorded Studio forward_backward, safely.
Input: A saved forward_backward body (for example perf4 update 2), or reconstructed
    update sequences, and the Flash model.
Output: memory.jsonl with active and peak memory by replay phase, and result.json.
Method:
    1. Take the GPU lease and build the engine with the production service options.
    2. Start a guard thread. It ends the process when MLX active memory or the
       host memory level crosses a limit, before the OS can stall.
    3. Run the recorded forward_backward and log memory at every phase change
       and at every 1 GiB step.
    4. Record the wall time of the layer forwards, the group VJPs, and the head
       tiles. The rest is gradient sums, the objective, and host overhead.
Note: The script never opens a service state root. It uses a fresh client.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import threading
import time
from pathlib import Path

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"
GIB = 1 << 30


def main() -> None:
    args = _parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX)  # Wait in line for the shared lease.
    _write_identity(args.output / "process.json")
    body = _load_body(args.batch, args.indices)
    engine = _load_engine(args)
    import mlx.core as mx

    log = (args.output / "memory.jsonl").open("a", buffering=1)
    _start_guard(mx, args, log)
    admission = _record_admission(engine, args.output / "admission.json")
    phase_seconds = _time_replay_phases()
    body = {**body, "operation_id": f"probe-{time.time()}"}
    mx.reset_peak_memory()
    started = time.monotonic()
    result = _drain_logged(engine.work("probe", "forward_backward", body), mx, log,
                           started)
    _write_json(args.output / "result.json", {
        "seconds": time.monotonic() - started,
        "peak_gib": mx.get_peak_memory() / GIB,
        "measured_additional_gib": mx.get_peak_memory() / GIB - admission["persistent_gib"],
        "admission": admission,
        "phase_seconds": phase_seconds,
        "indices": args.indices,
        "loss": result["forward_metrics"]["loss"],
        "stats": {key: value for key, value in result["groups"][0].items()
                  if not isinstance(value, list)},
    })


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True,
                        help="JSON with the recorded forward_backward body.")
    parser.add_argument("--model", type=Path, required=True, help="Flash model path.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--guard-gib", type=float, default=216.0,
                        help="End the process above this MLX active memory.")
    parser.add_argument("--min-memory-level", type=int, default=8,
                        help="End the process below this kern.memorystatus_level.")
    parser.add_argument("--headroom-gib", type=float, default=32.0,
                        help="Service host headroom; sets the allocator budget.")
    parser.add_argument("--checkpoint-layers", type=int, default=None,
                        help="Fixed checkpoint interval; the default is auto.")
    parser.add_argument("--segment-tokens", type=int, default=2048,
                        help="Replay segment length (the service uses 2048).")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Replay only these sequence indices; the default is all.")
    return parser.parse_args()


def _load_body(path: Path, indices: list[int] | None) -> dict:
    """Read a journal body, or wrap reconstructed sequences in the CISPO update body."""
    payload = json.loads(path.read_text())
    if "body" in payload:
        body = payload["body"]
    else:
        body = {"sequences": payload["sequences"], "loss": "cispo",
                "reduction": "sum", "clip_low": 0.0, "clip_high": 4.0}
    if indices is None:
        return body
    return {**body, "sequences": [body["sequences"][index] for index in indices]}


def _time_replay_phases() -> dict:
    """
    Purpose: Add the wall time of the replay phases to one dict.
    Input: None.
    Output: A dict that the replay fills with seconds per phase and call counts.
    Note: The replay profile hook times each layer forward (with its eval) and
        each group VJP. A generator wrapper times the head tiles. The *_build
        keys time only the lazy graph construction on the host, while the GPU
        waits.
    """
    from studio_compute.numerics.replay import Replay

    totals, scope = {}, {}

    class Recorder:
        def record(self, kind, index, is_linear, seconds):
            totals[kind] = totals.get(kind, 0.0) + seconds
            totals[kind + "_calls"] = totals.get(kind + "_calls", 0) + 1

    initialize = Replay.__init__

    def profiled_init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        self.profile = Recorder()

    tree_heads = Replay._tree_heads

    def timed_heads(self, *args):
        tiles = tree_heads(self, *args)
        while True:
            started = time.perf_counter()
            try:
                record = next(tiles)
            except StopIteration:
                return
            finally:
                totals["head"] = totals.get("head", 0.0) + time.perf_counter() - started
            yield record

    Replay.__init__ = profiled_init
    Replay._tree_heads = timed_heads
    Replay._group_vjp = _timed_method(Replay._group_vjp, totals, "vjp_build", scope)
    Replay._layer = _timed_method(Replay._layer, totals, "forward_build", scope)
    return totals


def _timed_method(method, totals: dict, key: str, scope: dict):
    """Wrap one replay method; add its outermost wall time to totals[key]."""

    def timed(self, *args, **kwargs):
        if scope.get("active"):
            return method(self, *args, **kwargs)
        scope["active"] = True
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            scope["active"] = False
            totals[key] = totals.get(key, 0.0) + time.perf_counter() - started

    return timed


def _load_engine(args: argparse.Namespace):
    """Build the engine with the perf4 service options and one fresh client."""
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from studio_compute.flash_model import QwenFlashModel
    from studio_compute.runtime import RuntimeEngine

    engine = RuntimeEngine(
        str(args.model), model_factory=QwenFlashModel, execution_profile="bf16-fast",
        headroom_gib=args.headroom_gib, activation_budget_gib=16,
        inference_concurrency=8, prefill_tokens=1024,
        segment_tokens=args.segment_tokens,
        head_rows=128, replay_layout="segment_trie", training_moe_chunk=32,
        prefill_moe_chunk=32, checkpoint_layers=args.checkpoint_layers,
    )
    engine.create("probe", {
        "seed": 20261019, "lora_rank": 16, "lora_scale": 2.0, "learning_rate": 5e-5,
        "optimizer": {"beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8},
    })
    return engine


def _start_guard(mx, args: argparse.Namespace, log) -> None:
    """End the process before memory can stall the host."""

    def watch() -> None:
        last_level_check = 0.0
        while True:
            active_gib = mx.get_active_memory() / GIB
            level = None
            now = time.monotonic()
            if now - last_level_check > 1.0:
                last_level_check = now
                level = int(subprocess.check_output(
                    ["sysctl", "-n", "kern.memorystatus_level"], text=True))
            if active_gib > args.guard_gib or (
                    level is not None and level < args.min_memory_level):
                log.write(json.dumps({"guard_stop": True, "active_gib": active_gib,
                                      "memory_level": level}) + "\n")
                log.flush()
                os._exit(3)
            time.sleep(0.05)

    threading.Thread(target=watch, daemon=True).start()


def _record_admission(engine, path: Path) -> dict:
    """Keep the interval, sum level cap, and peak that trie admission chooses."""
    from studio_compute.activation_budget import SAFETY_BYTES

    record = {}
    original = engine._tree_admission

    def wrapped(sequences, future_gradient_bytes):
        persistent = max(engine.mx.get_active_memory(),
                         engine.persistent_bytes() + sum(engine.reserved.values()))
        interval, peak, budgets, levels = original(sequences, future_gradient_bytes)
        record.update(interval=interval, gradient_sum_levels=levels,
                      peak_gib=peak / GIB, persistent_gib=persistent / GIB,
                      estimated_additional_gib=(peak - SAFETY_BYTES) / GIB,
                      gradient_gib=future_gradient_bytes / GIB)
        _write_json(path, record)
        return interval, peak, budgets, levels

    engine._tree_admission = wrapped
    return record


def _drain_logged(work, mx, log, started: float):
    """Drain the operation and log memory at phase changes and 1 GiB steps."""
    last_phase, last_gib = None, 0.0
    while True:
        try:
            progress = next(work)
        except StopIteration as stop:
            return stop.value
        active_gib = mx.get_active_memory() / GIB
        phase = progress.get("phase")
        if phase != last_phase or abs(active_gib - last_gib) >= 1.0:
            log.write(json.dumps({
                "t": round(time.monotonic() - started, 1), "phase": phase,
                "segment": progress.get("segment"), "layer": progress.get("layer"),
                "active_gib": round(active_gib, 2),
                "peak_gib": round(mx.get_peak_memory() / GIB, 2),
            }) + "\n")
            last_phase, last_gib = phase, active_gib


def _write_identity(path: Path) -> None:
    pid = os.getpid()
    started = subprocess.check_output(["ps", "-o", "lstart=,command=", "-p", str(pid)],
                                      text=True).strip()
    _write_json(path, {"pid": pid, "ps_start_and_command": started,
                       "lease": str(LEASE)})


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


if __name__ == "__main__":
    main()
