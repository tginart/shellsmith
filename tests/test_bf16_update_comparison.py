"""Verify exact BF16 diagnostic decoding and FP64 update comparisons without MLX."""

import importlib.util
import json
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")


def test_bf16_gradients_decode_negative_values_and_match_norms(tmp_path):
    module = _module()
    a = np.array([[1, -2], [0, 0.5]], dtype=np.float32)
    b = np.array([[1.0078125, -2.015625], [0, 0.5]], dtype=np.float32)
    _write(tmp_path / "a", {"gradient": a}, bf16=False)
    _write(tmp_path / "b", {"gradient": b}, bf16=True)
    left, right = module.TensorFile(tmp_path / "a"), module.TensorFile(tmp_path / "b")
    try:
        np.testing.assert_array_equal(right.tensor("gradient"), b)
        report = module.compare(left, right)
        assert report["relative_l2"] == pytest.approx(
            np.linalg.norm(b.astype(float) - a) / np.linalg.norm(a.astype(float))
        )
        assert report["tensor_count"] == 1
    finally:
        left.close()
        right.close()


def test_master_delta_comparison_uses_matching_initial_values(tmp_path):
    module = _module()
    values = {
        "initial.factor": np.array([[1.0, -2.0]], np.float32),
        "master.factor": np.array([[1.5, -1.5]], np.float32),
    }
    _write(tmp_path / "a", values, bf16=False)
    _write(tmp_path / "b", values, bf16=True)
    left, right = module.TensorFile(tmp_path / "a"), module.TensorFile(tmp_path / "b")
    try:
        result = module.compare(left, right, update_delta=True)
        assert result["relative_l2"] == 0 and result["cosine"] == pytest.approx(1)
        assert result["reference_l2"] == pytest.approx(2**-0.5)
    finally:
        left.close()
        right.close()


def _write(path, values, *, bf16):
    header, parts, offset = {}, [], 0
    for name, value in values.items():
        data = (
            (value.view(np.uint32) >> 16).astype("<u2").tobytes()
            if bf16
            else value.astype("<f4").tobytes()
        )
        header[name] = {
            "dtype": "BF16" if bf16 else "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        parts.append(data)
    raw = json.dumps(header).encode()
    raw += b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(parts))


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_bf16_updates.py"
    spec = importlib.util.spec_from_file_location("compare_bf16_updates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
