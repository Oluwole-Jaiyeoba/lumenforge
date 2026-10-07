import gzip
import json

import pytest

from agentic_experiments.runners.analyze_work_audit_storage import summarize


def _arm(root, seed, arm, native_hits, control_hits):
    path = root / f"seed{seed}_{arm}"
    path.mkdir()
    case = {
        "arm": arm,
        "due_to_first_token_ms": {"on_demand": 300, "host_stage": 120, "full_prepare": 80}[arm],
        "tool_end_to_first_token_ms": 100,
        "replay": {"ttft_ms": 100, "total_latency_ms": 400},
        "stage_completed_before_due": arm != "on_demand",
        "storage_eviction": {"evicted_tokens": 2048},
        "storage_stage": {"completed": {"storage_loaded_tokens": control_hits}} if control_hits else None,
    }
    (path / "case_results.json").write_text(json.dumps(case), encoding="utf-8")
    with gzip.open(path / "backend_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "hiradix.storage_hit_tokens.end", "result": native_hits,
                                 "kv_context": {"request_id": "session-replay"}}) + "\n")
        handle.write(json.dumps({"event": "hiradix.match_prefix.end",
                                 "kv_context": {"agent_request_id": "session-replay"},
                                 "result": [{"index_count": 2100}]}) + "\n")


def test_paired_storage_summary_requires_real_hits(tmp_path):
    _arm(tmp_path, 1, "on_demand", 2048, 0)
    _arm(tmp_path, 1, "host_stage", 0, 2048)
    _arm(tmp_path, 1, "full_prepare", 0, 2048)
    summary = summarize(tmp_path)
    assert summary["paired"] == [{"seed": 1, "host_stage_delta_ms": -180,
                                   "full_prepare_delta_ms": -220}]


def test_unproven_on_demand_hit_fails(tmp_path):
    _arm(tmp_path, 1, "on_demand", 0, 0)
    _arm(tmp_path, 1, "host_stage", 0, 2048)
    _arm(tmp_path, 1, "full_prepare", 0, 2048)
    with pytest.raises(ValueError, match="no proven native L3 hit"):
        summarize(tmp_path)


def test_peer_overlap_requires_intersecting_request_and_preparation(tmp_path):
    _arm(tmp_path, 1, "on_demand", 2048, 0)
    _arm(tmp_path, 1, "host_stage", 0, 2048)
    _arm(tmp_path, 1, "full_prepare", 0, 2048)
    for arm in ("on_demand", "host_stage", "full_prepare"):
        path = tmp_path / f"seed1_{arm}" / "case_results.json"
        case = json.loads(path.read_text(encoding="utf-8"))
        if arm != "on_demand":
            case["storage_stage"] = {
                "accepted": {"control_request_ns": 100},
                "completed": {"storage_loaded_tokens": 2048},
                "completed_observed_ns": 300,
            }
        case["peers"] = [
            {"request_start_ns": 150, "request_end_ns": 400, "ttft_ms": 25,
             "total_latency_ms": 80},
            {"request_start_ns": 400, "request_end_ns": 500, "ttft_ms": 26,
             "total_latency_ms": 82},
        ]
        path.write_text(json.dumps(case), encoding="utf-8")
    rows = summarize(tmp_path)["rows"]
    assert [row["peers_overlapping_preparation"] for row in rows if row["arm"] != "on_demand"] == [1, 1]


def test_native_storage_timing_requires_ordered_correlated_evidence(tmp_path):
    _arm(tmp_path, 1, "on_demand", 2048, 0)
    for arm in ("host_stage", "full_prepare"):
        _arm(tmp_path, 1, arm, 0, 2048)
        path = tmp_path / f"seed1_{arm}"
        case = json.loads((path / "case_results.json").read_text(encoding="utf-8"))
        case["storage_stage"] = {
            "accepted": {"storage_request_id": arm, "storage_requested_ns": 100_000_000},
            "completed": {"storage_loaded_tokens": 2048, "storage_data_ready_ns": 130_000_000,
                          "storage_host_committed_ns": 180_000_000},
            "completed_observed_ns": 200_000_000,
        }
        (path / "case_results.json").write_text(json.dumps(case), encoding="utf-8")
        with gzip.open(path / "backend_trace.jsonl.gz", "at", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "storage_prefetch.data_ready", "ts_ns": 130_000_000,
                                     "storage_request_id": arm, "completed_tokens": 2048}) + "\n")
    rows = summarize(tmp_path, require_native_timing=True)["rows"]
    assert [row["storage_data_ready_ms"] for row in rows if row["arm"] != "on_demand"] == [30.0, 30.0]
    assert [row["storage_commit_after_ready_ms"] for row in rows if row["arm"] != "on_demand"] == [50.0, 50.0]

    path = tmp_path / "seed1_host_stage" / "case_results.json"
    case = json.loads(path.read_text(encoding="utf-8"))
    case["storage_stage"]["completed"]["storage_data_ready_ns"] = 210_000_000
    path.write_text(json.dumps(case), encoding="utf-8")
    with pytest.raises(ValueError, match="ordered native storage timestamps missing"):
        summarize(tmp_path, require_native_timing=True)
