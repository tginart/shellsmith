"""Keep disjoint layer totals separate from nested VJP build/evaluation timings."""

import importlib.util
from pathlib import Path


def test_report_does_not_double_count_nested_backward_times():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/summarize_bf16_qualification.py"
    )
    spec = importlib.util.spec_from_file_location("summarize_bf16_qualification", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    row = {
        "execution_receipt": {"status": "passed"},
        "profile": {
            "forward:gdn": {"seconds": 2},
            "forward:attention": {"seconds": 3},
            "vjp_total:gdn": {"seconds": 7},
            "vjp_total:attention": {"seconds": 11},
            "vjp_build:gdn": {"seconds": 3},
            "vjp_eval:gdn": {"seconds": 4},
        },
        "whole_update_seconds": 30,
        "first_update_ready_including_benchmark_preparation_seconds": 40,
        "forward_metrics": {
            "sequences": 1,
            "completed_tokens": 100,
            "total_tokens": 100,
            "supervised_tokens": 10,
            "loss_kind": "cispo",
            "phase_seconds": {"head_backward": 2},
        },
        "warmup": {"seconds_including_first_call_compilation": 1},
        "update_peak_active_memory_bytes": 1000,
        "process_memory_at_update_boundary": {
            "lifetime_max_physical_footprint_bytes": 1100
        },
        "execution_metrics": {"swap_growth_bytes": 0},
        "gradient_hash": "same",
        "updated_adapter_hash": "same",
        "runtime_identity": {"kernel_build_id": "k"},
        "initial_adapter_hash": "a",
        "packed_sequences_sha256": "s",
    }
    result = module._reduce([row, row, row], 3)
    assert result["median_layer_forward_seconds"] == 5
    assert result["median_layer_backward_seconds"] == 18
    assert result["median_head_backward_seconds"] == 2
    assert result["median_total_backward_seconds"] == 20
    assert result["median_first_ready_seconds"] == 40
    assert (
        result["repeated_gradients_exact"] and result["repeated_updated_adapters_exact"]
    )
