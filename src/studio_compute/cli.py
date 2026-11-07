"""Standalone daemon entry point; no Shellsmith imports."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import threading
import tomllib

SERVICE_DEFAULTS = {
    "model": None,
    "model_profile": "qwen27",
    "execution_profile": "bf16-fast",
    "replay_layout": "prompt_groups",
    "training_moe_chunk": 0,
    "prefill_moe_chunk": 0,
    "state_root": "~/.local/state/studio-compute",
    "host": "127.0.0.1",
    "port": 8766,
    "headroom_gib": 16.0,
    "allocator_gib": None,
    "checkpoint_layers": None,
    "activation_budget_gib": None,
    "segment_tokens": 2048,
    "head_rows": 128,
    "prefill_tokens": 2048,
    "inference_concurrency": 8,
    "metal_cache_gib": 2.0,
    "prefix_cache_gib": None,
    "gpu_lease_root": "~/.local/state/shellsmith/gpu-lease",
    "checkpoint_retention": 0,
    "recovery_state": "",
    "checkpoint_reserve_gib": 16.0,
}


def cache_limit(value):
    if value == "auto":
        return None
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use auto or a nonnegative GiB cap.") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Prefix cache cap must be finite and nonnegative.")
    return number



def checkpoint_limit(value):
    if value == "auto":
        return None
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use auto or a positive layer interval.") from error
    if number<1:
        raise argparse.ArgumentTypeError("Checkpoint interval must be positive.")
    return number

def parse_args(argv=None):
    """Read standalone service TOML; explicit CLI flags override its settings."""
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config_path = probe.parse_known_args(argv)[0].config
    defaults = dict(SERVICE_DEFAULTS)
    if config_path:
        raw = tomllib.loads(config_path.read_text())
        if set(raw) != {"service"} or not isinstance(raw["service"], dict):
            raise ValueError("Studio service config requires one [service] table.")
        unknown = set(raw["service"]) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown service settings: {sorted(unknown)}")
        for name, value in raw["service"].items():
            expected = defaults[name]
            if name == "checkpoint_layers":
                if value == "auto":
                    value = None
                elif type(value) is not int or value<1:
                    raise ValueError("[service].checkpoint_layers must be auto or a positive integer.")
            elif name in {"allocator_gib", "prefix_cache_gib", "activation_budget_gib"} and value == "auto":
                value = None
            elif name in {"allocator_gib", "prefix_cache_gib", "activation_budget_gib"} or type(expected) is float:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"[service].{name} must be finite numeric or auto where supported.")
            elif name == "model":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("[service].model must be a nonempty path.")
            elif type(value) is not type(expected):
                raise ValueError(f"[service].{name} has an invalid type.")
            defaults[name] = value
    parser = argparse.ArgumentParser(prog="studio-compute")
    commands = parser.add_subparsers(dest="command", required=True)
    metrics = commands.add_parser("metrics", help="Read live local inference throughput and concurrency")
    metrics.add_argument("--endpoint", default="http://127.0.0.1:8766")
    serve = commands.add_parser("serve")
    serve.add_argument("--config", type=Path)
    serve.add_argument("--model", required=not defaults["model"])
    serve.add_argument(
        "--model-profile", choices=("qwen27", "flash-experimental"),
        help="Select the admitted dense profile or the experimental Flash LoRA profile.",
    )
    serve.add_argument(
        "--execution-profile", choices=("legacy", "bf16-fast"),
        help="Select legacy precision or the BF16 arithmetic implementation.",
    )
    serve.add_argument(
        "--replay-layout", choices=("prompt_groups", "segment_trie"),
        help="Share replay segments among identical prompts, or along any identical token prefix.",
    )
    serve.add_argument(
        "--training-moe-chunk", type=int,
        help="Group routed-expert training rows into tiles of this height; 0 keeps per-row products.",
    )
    serve.add_argument(
        "--prefill-moe-chunk", type=int,
        help="Group routed-expert prompt-prefill rows into tiles of this height; 0 keeps sorted rows.",
    )
    serve.add_argument("--state-root", default="~/.local/state/studio-compute")
    serve.add_argument("--checkpoint-retention", type=int, default=0,
                       help="Keep this many recoverable revisions per client; zero retains all.")
    serve.add_argument("--recovery-state", default="",
                       help="Protect the latest full checkpoint in this trainer state JSON file.")
    serve.add_argument("--checkpoint-reserve-gib", type=float, default=16.0,
                       help="Require this free disk reserve after the next checkpoint write.")
    from .http import loopback_host
    serve.add_argument("--host", type=loopback_host, default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8766)
    serve.add_argument("--headroom-gib", type=float, default=16)
    serve.add_argument("--allocator-gib", type=float)
    serve.add_argument("--checkpoint-layers",type=checkpoint_limit,metavar="auto|N",
                       help="Use the smallest admitted BF16 interval, or an explicit positive interval.")
    serve.add_argument("--activation-budget-gib",type=cache_limit,metavar="auto|N",
                       help="Cap optional saved activations after mandatory state, scratch, and safety reserves.")
    serve.add_argument("--segment-tokens", type=int, default=2048)
    serve.add_argument("--head-rows", type=int, default=128)
    serve.add_argument("--prefill-tokens", type=int, default=2048)
    serve.add_argument(
        "--inference-concurrency", type=int,
        help="Maximum active inference requests, independent of episode workers; excess requests queue.",
    )
    serve.add_argument("--metal-cache-gib", type=float, default=2)
    serve.add_argument(
        "--prefix-cache-gib", type=cache_limit, default=None, metavar="auto|GiB",
        help="Use available allocator memory by default; evict for active work. 0 disables retention.",
    )
    serve.add_argument(
        "--gpu-lease-root", default="~/.local/state/shellsmith/gpu-lease"
    )
    serve.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.command == "metrics":
        return args
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535].")
    if args.checkpoint_retention < 0 or args.checkpoint_retention == 1:
        parser.error("checkpoint_retention must be zero or at least two.")
    if not math.isfinite(args.checkpoint_reserve_gib) or args.checkpoint_reserve_gib < 0:
        parser.error("checkpoint_reserve_gib must be finite and nonnegative.")
    if args.checkpoint_layers is None and args.execution_profile == "legacy":
        args.checkpoint_layers = 8
    for name in ("segment_tokens", "head_rows", "prefill_tokens", "inference_concurrency"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive.")
    if args.execution_profile not in ("legacy", "bf16-fast"):
        parser.error("execution-profile must be legacy or bf16-fast.")
    if args.model_profile not in ("qwen27", "flash-experimental"):
        parser.error("model-profile must be qwen27 or flash-experimental.")
    if args.model_profile == "flash-experimental" and args.execution_profile != "bf16-fast":
        parser.error("The experimental Flash profile requires bf16-fast.")
    if args.prefill_tokens > 2048:
        parser.error("prefill_tokens must be <= 2048 for the current scratch admission estimate.")
    for name in ("headroom_gib", "metal_cache_gib", "allocator_gib", "prefix_cache_gib", "activation_budget_gib"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0 or (name == "allocator_gib" and value == 0)):
            parser.error(f"{name} has an invalid memory limit.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "metrics":
        from .client import Client
        print(json.dumps(Client(args.endpoint).metrics(), indent=2, allow_nan=False))
        return
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    root = Path(args.gpu_lease_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "owner.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another process owns the Studio GPU lease.")
        from .runtime import RuntimeEngine
        from .http import server
        from .service import Service

        model_options = {}
        if args.model_profile == "flash-experimental":
            from .flash_model import QwenFlashModel
            model_options["model_factory"] = QwenFlashModel

        service = Service(
            Path(args.state_root).expanduser(),
            lambda: RuntimeEngine(
                args.model,
                execution_profile=args.execution_profile,
                headroom_gib=args.headroom_gib,
                allocator_gib=args.allocator_gib,
                checkpoint_layers=args.checkpoint_layers,
                activation_budget_gib=args.activation_budget_gib,
                replay_layout=args.replay_layout,
                training_moe_chunk=args.training_moe_chunk,
                prefill_moe_chunk=args.prefill_moe_chunk,
                segment_tokens=args.segment_tokens,
                head_rows=args.head_rows,
                prefill_tokens=args.prefill_tokens,
                inference_concurrency=args.inference_concurrency,
                metal_cache_gib=args.metal_cache_gib,
                prefix_cache_gib=args.prefix_cache_gib,
                **model_options,
            ),
            checkpoint_retention=args.checkpoint_retention,
            recovery_state=args.recovery_state,
            checkpoint_reserve_gib=args.checkpoint_reserve_gib,
        )
        http = server(service, (args.host, args.port))

        def stop(*_):
            threading.Thread(target=http.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            http.serve_forever()
        finally:
            http.server_close()
            service.close()


if __name__ == "__main__":
    main()
