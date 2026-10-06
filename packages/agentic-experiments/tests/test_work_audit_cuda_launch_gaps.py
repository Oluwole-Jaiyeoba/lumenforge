import sqlite3

import pytest

from agentic_experiments.runners.analyze_work_audit_cuda_launch_gaps import analyze


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, globalTid INTEGER);
        CREATE TABLE StringIds (id INTEGER, value TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
            (start INTEGER, end INTEGER, globalTid INTEGER, correlationId INTEGER, nameId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
            (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_SYNCHRONIZATION
            (start INTEGER, end INTEGER, globalPid INTEGER, correlationId INTEGER, syncType INTEGER);
        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel'), (2, 'cudaStreamSynchronize');
        INSERT INTO NVTX_EVENTS VALUES
            (0, 9000000, 'agentic_kv:worker.forward_batch_generation:x call_id=early', 16777217),
            (10000000, 19000000, 'agentic_kv:worker.forward_batch_generation:x call_id=post', 16777217);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (1000000, 2000000, 16777217, 1, 1),
            (5000000, 6000000, 16777217, 2, 1),
            (6000000, 6500000, 16777217, 50, 2),
            (11000000, 12000000, 16777217, 3, 1),
            (12500000, 13500000, 16777217, 4, 1),
            (5000000, 6000000, 33554433, 5, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES
            (2000000, 3000000, 16777216, 1),
            (7000000, 8000000, 16777216, 2),
            (12000000, 13000000, 16777216, 3),
            (15000000, 16000000, 16777216, 4),
            (7000000, 8000000, 33554432, 5);
        INSERT INTO CUPTI_ACTIVITY_KIND_SYNCHRONIZATION VALUES
            (6000000, 6250000, 16777216, 2, 2);
    """)
    return db


def _summary() -> dict:
    return {
        "run_id": "sample",
        "pairs": [{"pair": 1, "early_case_id": "a", "post_short_case_id": "b"}],
        "cases": [
            {"case_id": "a", "short_decode": {"decode_batches":
                [{"model_forward_call_ids": ["early"]}]}},
            {"case_id": "b", "short_decode": {"decode_batches":
                [{"model_forward_call_ids": ["post"]}]}},
        ],
    }


def test_launch_gap_decomposition_and_sync_api_are_call_linked():
    result = analyze(_db(), _summary(), 1)
    assert result["early"]["kernel_count"] == 2
    assert result["early"]["gap_ms"] == 4
    assert result["early"]["cpu_before_launch_ms"] == 2
    assert result["early"]["launch_api_ms"] == 1
    assert result["early"]["after_launch_api_ms"] == 1
    assert result["early"]["stream_wait_event_gpu_ms"] == 0.25
    assert result["early"]["blocking_sync_api_ms"] == 0.5
    assert result["early"]["forward_calls"][0]["sync_api"] == [
        {"name": "cudaStreamSynchronize", "count": 1, "duration_ms": 0.5}]
    assert result["post_short"]["cpu_before_launch_ms"] == 0
    assert result["post_short"]["launch_api_ms"] == 0.5
    assert result["post_short"]["after_launch_api_ms"] == 1.5
    assert result["early_minus_post_short_ms"]["cpu_before_launch_ms"] == 2


def test_missing_cuda_linkage_fails_loudly():
    db = _db()
    db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId IN (1, 2)")
    with pytest.raises(ValueError, match="No CUDA kernels"):
        analyze(db, _summary(), 1)


def test_kernel_report_cross_check_rejects_partial_capture():
    db = _db()
    db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE correlationId = 2")
    reference = {"pairs": [{"pair": 1, "early": {"kernel_count": 2,
                      "kernel_gap_inside_span_ms": 4}, "post_short": {
                      "kernel_count": 2, "kernel_gap_inside_span_ms": 2}}]}
    with pytest.raises(ValueError, match="disagree with kernel attribution"):
        analyze(db, _summary(), 1, reference)
