"""
Purpose: Compare preserved gradient and Adam-update diagnostics without a GPU.
Input: Matched legacy and BF16 safetensors files from disposable benchmark runs.
Output: FP64 norm, direction, and per-tensor differences. Agreement is diagnostic.
Note: Independent derivatives and real-model memorization remain the correctness gates.
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import struct
from pathlib import Path
from typing import Any


class TensorFile:
    """
    Purpose: Read one tensor at a time without MLX or model allocation.
    Input: An existing safetensors file. Output: Exact BF16/F32 values as FP64 arrays.
    """

    def __init__(self, path: Path) -> None:
        self.file = path.open("rb")
        self.data = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        length = struct.unpack("<Q", self.data[:8])[0]
        self.header = json.loads(self.data[8 : 8 + length])
        self.offset = 8 + length

    def names(self) -> set[str]:
        return set(self.header) - {"__metadata__"}

    def tensor(self, name: str) -> Any:
        import numpy as np

        row = self.header[name]
        start, end = row["data_offsets"]
        count = math.prod(row["shape"])
        if row["dtype"] == "BF16":
            assert end - start == count * 2
            value = np.frombuffer(
                self.data, dtype="<u2", count=count, offset=self.offset + start
            )
            result = (value.astype(np.uint32) << 16).view("<f4").astype(np.float64)
        elif row["dtype"] == "F32":
            assert end - start == count * 4
            result = np.frombuffer(
                self.data, dtype="<f4", count=count, offset=self.offset + start
            ).astype(np.float64)
        else:
            raise ValueError("Unsupported diagnostic dtype: " + row["dtype"])
        assert np.isfinite(result).all(), name
        return result.reshape(row["shape"])

    def close(self) -> None:
        self.data.close()
        self.file.close()


def compare(
    reference: TensorFile, candidate: TensorFile, *, update_delta: bool = False
) -> dict:
    """
    Purpose: Reduce diagnostic errors with FP64 arrays and no accelerator work.
    Input: Matched tensor inventories. Output: Overall and per-tensor measurements.
    """
    import numpy as np

    names = reference.names()
    if update_delta:
        names = {name for name in names if name.startswith("master.")}
        assert names == {
            name for name in candidate.names() if name.startswith("master.")
        }
    else:
        assert names == candidate.names(), "Gradient inventories differ."
    assert names, "The diagnostic tensor inventory is empty."
    totals = [0.0, 0.0, 0.0, 0.0]
    rows = []
    for name in sorted(names):
        a = reference.tensor(name)
        b = candidate.tensor(name)
        assert a.shape == b.shape, name
        if update_delta:
            initial = "initial." + name.removeprefix("master.")
            assert np.array_equal(reference.tensor(initial), candidate.tensor(initial))
            a = a - reference.tensor(initial)
            b = b - candidate.tensor(initial)
        a, b = a.reshape(-1), b.reshape(-1)
        difference = b - a
        aa, bb, dot, error = [
            float(v)
            for v in (
                np.dot(a, a),
                np.dot(b, b),
                np.dot(a, b),
                np.dot(difference, difference),
            )
        ]
        for index, value in enumerate((aa, bb, dot, error)):
            totals[index] += value
        rows.append(
            {
                "name": name,
                "reference_l2": math.sqrt(aa),
                "candidate_l2": math.sqrt(bb),
                "relative_l2": math.sqrt(error / max(aa, 1e-60)),
                "cosine": dot / math.sqrt(aa * bb) if aa and bb else None,
                "both_zero": aa == bb == 0,
            }
        )
    aa, bb, dot, error = totals
    return {
        "tensor_count": len(rows),
        "both_zero_count": sum(row["both_zero"] for row in rows),
        "reference_l2": math.sqrt(aa),
        "candidate_l2": math.sqrt(bb),
        "relative_l2": math.sqrt(error / max(aa, 1e-60)),
        "cosine": dot / math.sqrt(aa * bb) if aa and bb else None,
        "scope": "In-memory master-weight delta"
        if update_delta
        else "Accumulated adapter gradients",
        "per_tensor": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        type=Path,
        required=True,
        help="Legacy gradient or diagnostic-Adam file.",
    )
    parser.add_argument(
        "--bf16",
        type=Path,
        required=True,
        help="Matched BF16 file with the same initial policy and data.",
    )
    parser.add_argument(
        "--update-delta",
        action="store_true",
        help="Compare master minus initial values in diagnostic-Adam files.",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fresh diagnostic JSON path."
    )
    args = parser.parse_args()
    assert not args.output.exists()
    reference, candidate = TensorFile(args.legacy), TensorFile(args.bf16)
    try:
        result = compare(reference, candidate, update_delta=args.update_delta)
        result["correctness_gate"] = False
        result["interpretation"] = (
            "Agreement with legacy is diagnostic; the old update is not a numerical oracle."
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        reference.close()
        candidate.close()
