import sqlite3

import pytest

from agentic_experiments.runners.analyze_work_audit_cuda_kernels import analyze, forward_kernels


def test_nsight_kernels_join_by_range_thread_and_correlation():
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
            (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
            (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
            (start INTEGER, end INTEGER, globalPid INTEGER, copyKind INTEGER);
        INSERT INTO NVTX_EVENTS VALUES
            (100, 3500, 'agentic_kv:worker.forward_batch_generation:TpModelWorker.forward_batch_generation call_id=f1', 16777217);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (1000, 2000, 16777217, 7), (3000, 4000, 16777217, 8),
            (1000, 2000, 33554433, 9);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (2500, 2750, 16777216, 7), (4100, 4600, 16777216, 8),
            (2500, 8000, 33554432, 9);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
            (2800, 3900, 16777216, 1), (3000, 4000, 33554432, 1);
    """)
    row = forward_kernels(db)["f1"]
    assert row["kernel_count"] == 2
    assert row["kernel_duration_sum_ms"] == 0.001
    assert row["kernel_gap_inside_span_ms"] == 0.001
    assert row["htod_during_kernel_gaps_ms"] == 0.001


def test_nsight_analysis_requires_every_forward_to_have_kernels():
    summary = {
        "run_id": "r", "pairs": [{"pair": 1, "early_case_id": "a", "post_short_case_id": "b"}],
        "cases": [
            {"case_id": "a", "short_decode": {"decode_batches":
                [{"model_forward_call_ids": ["f1"]}]}},
            {"case_id": "b", "short_decode": {"decode_batches":
                [{"model_forward_call_ids": ["f2"]}]}},
        ],
    }
    sample = {"kernel_count": 1, "kernel_duration_sum_ms": 1.0,
              "kernel_active_union_ms": 1.0, "kernel_span_ms": 2.0,
              "kernel_gap_inside_span_ms": 1.0, "cuda_runtime_api_ms": 0.2,
              "nvtx_forward_wall_ms": 3.0, "htod_during_kernel_span_ms": 0.1,
              "htod_during_kernel_gaps_ms": 0.1}
    with pytest.raises(ValueError, match="Missing CUDA kernels"):
        analyze(summary, {"f1": sample})
    result = analyze(summary, {"f1": sample, "f2": sample})
    assert result["status"] == "validated"
    assert result["pairs"][0]["early_minus_post_short_ms"]["kernel_duration_sum_ms"] == 0
