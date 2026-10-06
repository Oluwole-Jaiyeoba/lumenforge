import sqlite3

import pytest

from agentic_experiments.runners.analyze_work_audit_physical_overlap import analyze


def _sample():
    db = sqlite3.connect(":memory:")
    db.executescript("""
        CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME
            (start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL
            (start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER);
        CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY
            (start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,
             copyKind INTEGER,bytes INTEGER);
        INSERT INTO NVTX_EVENTS VALUES
            (100,200,16777217,'agentic_kv:worker.forward_batch_generation:x call_id=f1'),
            (300,400,16777217,'agentic_kv:worker.forward_batch_generation:x call_id=f2'),
            (150,350,16777218,'agentic_kv:async_load load_id=l1');
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES
            (160,170,16777218,500);
        INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES
            (20000000,30000000,16777216,500,1,1000000);
    """)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)",
                   [(100 + i % 50, 101 + i % 50, 16777217, i)
                    for i in range(1, 51)] +
                   [(300 + i % 50, 301 + i % 50, 16777217, i)
                    for i in range(51, 101)])
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?)",
                   [(10000000 + i * 100000, 10050000 + i * 100000, 16777216, i)
                    for i in range(1, 101)])
    trace = [
        {"event": "scheduler.run_batch.start", "call_id": "b1", "ts_ns": 2000,
         "kv_context": {"batch": {"requests": [{"agent_request_id": "target"}]}}},
        {"event": "worker.forward_batch_generation.start", "call_id": "f1", "ts_ns": 2100},
        {"event": "worker.forward_batch_generation.end", "call_id": "f1", "ts_ns": 2200},
        {"event": "scheduler.run_batch.end", "call_id": "b1", "ts_ns": 3000},
        {"event": "scheduler.run_batch.start", "call_id": "b2", "ts_ns": 4000,
         "kv_context": {"batch": {"requests": [{"agent_request_id": "target"}]}}},
        {"event": "worker.forward_batch_generation.start", "call_id": "f2", "ts_ns": 4100},
        {"event": "worker.forward_batch_generation.end", "call_id": "f2", "ts_ns": 4200},
        {"event": "scheduler.run_batch.end", "call_id": "b2", "ts_ns": 5000},
    ]
    summary = {"run_id": "test", "planned_overlap": 1,
               "target": {"request_id": "target", "chunk_times_ns": [1000],
                          "request_end_ns": 10000},
               "donors": [{"load_id": "l1"}]}
    return db, summary, trace


def test_physical_copy_is_linked_by_load_id_and_cuda_correlation():
    db, summary, trace = _sample()
    result = analyze(db, summary, trace)
    assert result["status"] == "verified"
    assert result["target_kernel_count"] == 100
    assert result["physical_overlap_load_count"] == 1
    assert result["donors"][0]["copy_bytes"] == 1_000_000
    assert result["physical_overlap_ms"] > 0


def test_missing_copy_fails_closed():
    db, summary, trace = _sample()
    db.execute("DELETE FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    with pytest.raises(ValueError, match="lost physical H-to-D copies"):
        analyze(db, summary, trace)


def test_missing_cuda_activity_tables_are_not_zero_overlap():
    db, summary, trace = _sample()
    db.execute("DROP TABLE CUPTI_ACTIVITY_KIND_KERNEL")
    with pytest.raises(ValueError, match="captured no complete CUDA activity"):
        analyze(db, summary, trace)
