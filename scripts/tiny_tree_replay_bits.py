"""
Purpose: Hash the bits of a tiny Flash trie replay, to compare two source snapshots.
Input: The snapshot root, for its tests/test_flash_lora.py tiny model. Optional
    --record or --replay file for the layer VJP outputs.
Output: One line with the loss, a logprob hash, and a gradient bit hash.
Method:
    1. Build the tiny Flash model and a trie of 6 branches over 4-token segments.
       The node and tree sums then keep several partials.
    2. --record saves every layer VJP output in call order.
    3. --replay returns the saved outputs instead of new VJPs. The gradients then
       depend only on the accumulation code, so two sources must give equal bits.
Note: Layer VJPs are not bitwise reproducible from run to run on the GPU.
    Run the script with the PYTHONPATH of each snapshot.
"""

import argparse
import fcntl
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

LEASE = Path.home() / ".local/state/shellsmith/gpu-lease/owner.lock"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Snapshot root.")
    parser.add_argument("--record", type=Path, help="Save the layer VJP outputs here.")
    parser.add_argument("--replay", type=Path, help="Use these saved layer VJP outputs.")
    args = parser.parse_args()
    lease = LEASE.open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(args.root / "tests"))
    import mlx.core as mx
    from test_flash_lora import branch, consume, tiny_flash
    from studio_compute.numerics.flash_ops import FlashOps
    from studio_compute.numerics.replay import Replay

    lm, _ = tiny_flash(indexer_budget=8)
    prefix = list(range(1, 11))
    items = [
        branch(lm, prefix, [11 + i for i in range(9)]),
        branch(lm, prefix + [11, 12], [20 + i for i in range(13)]),
        branch(lm, prefix + [11, 12], [50 + i for i in range(7)]),
        branch(lm, prefix, [60 + i % 30 for i in range(21)]),
        branch(lm, prefix + [11, 12, 13, 14, 15, 16], [70 + i for i in range(8)]),
        branch(lm, list(range(30, 39)), [40 + i for i in range(15)]),
    ]
    for item in items:
        item["token_ids"] = item["input_ids"][0].tolist()
    replay = Replay(SimpleNamespace(language_model=lm), segment_tokens=4,
                    checkpoint_layers=1, execution_profile="bf16-fast",
                    model_ops=FlashOps())
    calls = _patch_layer_vjp(replay, record=args.record, saved=args.replay)
    probabilities, loss, gradients, stats = consume(replay.run_tree(items))
    if args.record is not None:
        mx.save_safetensors(str(args.record), calls)
    mx.eval(loss, gradients, probabilities)
    gradient_hash = hashlib.sha256()
    for name in sorted(gradients):
        gradient_hash.update(name.encode())
        gradient_hash.update(bytes(memoryview(gradients[name].view(mx.uint16))))
    logprob_hash = hashlib.sha256(b"".join(
        bytes(memoryview(value.astype(mx.float32))) for value in probabilities))
    print(args.root.name, "nodes", stats["tree_nodes"], "loss", float(loss.item()),
          "logprobs", logprob_hash.hexdigest()[:16], "gradients",
          gradient_hash.hexdigest()[:16], flush=True)


def _patch_layer_vjp(replay, *, record: Path | None, saved: Path | None) -> dict:
    """Record the layer VJP outputs in call order, or return saved ones."""
    import mlx.core as mx

    calls, counter, original = {}, [0], replay._layer_vjp
    stored = None if saved is None else mx.load(str(saved))

    def layer_vjp(*args, **kwargs):
        index = counter[0]
        counter[0] += 1
        names, back = original(*args, **kwargs) if stored is None else (
            _saved_names(stored, index), None)
        if stored is not None:
            back = [stored[f"{index}.{position}"] for position in
                    range(int(stored[f"{index}.count"].item()))]
            return names, back
        mx.eval(back)
        for position, value in enumerate(back):
            calls[f"{index}.{position}"] = value
        calls[f"{index}.count"] = mx.array(len(back))
        calls[f"{index}.names"] = mx.array([ord(c) for c in "\n".join(names)] or [0])
        return names, back

    if record is not None or saved is not None:
        replay._layer_vjp = layer_vjp
    return calls


def _saved_names(stored: dict, index: int) -> list:
    text = "".join(chr(code) for code in stored[f"{index}.names"].tolist() if code)
    return text.split("\n") if text else []


if __name__ == "__main__":
    main()
