import sqlite3

import pytest

from agentic_experiments.runners import analyze_work_audit_decode_submission as submission


def test_submission_attribution_sums_kernel_and_host_gap_time(monkeypatch):
    db = sqlite3.connect(":memory:")
    for table in ("NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME",
                  "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION",
                  "StringIds"):
        db.execute(f"CREATE TABLE {table} (marker INTEGER)")
    db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_RUNTIME")
    db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
               "(correlationId INTEGER, globalTid INTEGER, start INTEGER)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
               "(correlationId INTEGER, globalPid INTEGER, start INTEGER, end INTEGER)")
    for correlation, launch, kernel_start, kernel_end in (
        (1, 10, 20, 100), (2, 110, 120, 200),
        (3, 310, 320, 400), (4, 410, 420, 500),
    ):
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, 256, ?)",
                   (correlation, launch * 1_000_000))
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, 256, ?, ?)",
                   (correlation, kernel_start * 1_000_000, kernel_end * 1_000_000))
    monkeypatch.setattr(submission, "_target_forwards", lambda *args: {"a", "b"})
    monkeypatch.setattr(submission, "_forward_ranges", lambda *args: {
        "a": (0, 250_000_000, 256), "b": (300_000_000, 550_000_000, 256),
    })
    monkeypatch.setattr(submission, "_call_gaps", lambda *args: {
        "kernel_count": 2, "gap_ms": 0.01, "cpu_before_launch_ms": 0.006,
        "launch_api_ms": 0.002, "after_launch_api_ms": 0.002,
        "stream_wait_event_gpu_ms": 0, "blocking_sync_api_ms": 0,
        "sync_api": [], "largest_gaps": [],
    })
    result = submission.analyze(db, {
        "run_id": "paired", "target": {"request_id": "target",
                                      "chunk_times_ns": [1], "request_end_ns": 550},
    }, [])
    assert result["status"] == "verified"
    assert result["forward_count"] == 2
    assert result["kernel_count"] == 4
    assert result["between_forward_gap_ms"] == pytest.approx(50)
    assert result["kernel_execution_ms"] == pytest.approx(320)
    assert result["cpu_before_launch_ms"] == pytest.approx(0.012)


def test_submission_refuses_missing_cuda_tables():
    with pytest.raises(ValueError, match="Incomplete CUDA launch trace"):
        submission.analyze(sqlite3.connect(":memory:"), {}, [])
