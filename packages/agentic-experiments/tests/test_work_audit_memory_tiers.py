import gzip
import json

from agentic_experiments.runners.analyze_work_audit_memory_tiers import analyze, trace_evidence
from agentic_experiments.runners.run_work_audit_memory_tiers import percentile, tool_result


def _arm(root, mode, *, host=0, storage=0, workflow=1000):
    folder = root / "arms" / "seed1" / f"burst_{mode}"
    folder.mkdir(parents=True)
    request_id = f"tier-burst-{mode}-s00-turn01"
    case = {
        "seed": 1, "pattern": "burst", "mode": mode, "started_ns": 100,
        "workflow_duration_ms": workflow,
        "config": {"turns": 1},
        "metrics": {
            "mean_due_to_first_token_ms": workflow / 10,
            "p95_due_to_first_token_ms": workflow / 10,
            "total_due_to_first_token_ms": workflow / 10,
            "mean_ttft_ms": workflow / 20,
            "p95_ttft_ms": workflow / 20,
            "total_ttft_ms": workflow / 20,
            "total_submission_delay_ms": 0,
            "output_tokens": 16,
        },
        "per_session_completion_ms": {f"tier-burst-{mode}-s00": workflow},
        "sessions": [{"session_id": f"tier-burst-{mode}-s00", "turns": [
            {"turn": 0, "request_id": f"tier-burst-{mode}-s00-turn00"},
            {"turn": 1, "request_id": request_id, "prompt_tokens": 4096},
        ]}],
        "rounds": [{"inspections": [{
            "gpu_tokens": 4096 - host - storage, "host_tokens": host,
            "storage_candidate_tokens": storage,
            "inspection_started_ns": 1, "inspection_finished_ns": 2,
        }]}],
    }
    (folder / "case_results.json").write_text(json.dumps(case), encoding="utf-8")
    rid = f"rid-{mode}"
    rows = [{
        "event": "hiradix.match_prefix.end",
        "kv_context": {"agent_request_id": request_id, "request": {"rid": rid}},
        "result": [{"index_count": 4096 - host - storage}, {}, {}, host],
    }]
    if storage:
        rows.append({"event": "hiradix.storage_hit_tokens.end", "result": storage,
                     "kv_context": {"request_id": rid}})
    with gzip.open(folder / "backend_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return folder


def test_small_helpers():
    assert percentile([3, 1, 2, 4], .95) == 4
    assert "call 2" in tool_result("s1", 2, 3)


def test_trace_evidence_links_storage_to_agent_request(tmp_path):
    folder = _arm(tmp_path, "storage", storage=2048)
    matches, storage = trace_evidence(folder / "backend_trace.jsonl.gz")
    assert matches["tier-burst-storage-s00-turn01"]["gpu_tokens"] == 2048
    assert storage == {"tier-burst-storage-s00-turn01": 2048}


def test_three_tiers_require_native_exposure_and_compare_to_resident(tmp_path):
    _arm(tmp_path, "resident", workflow=1000)
    _arm(tmp_path, "host", host=2048, workflow=1200)
    _arm(tmp_path, "storage", storage=2048, workflow=1500)
    summary = analyze(tmp_path, [1], ["burst"], ["resident", "host", "storage"])
    assert summary["status"] == "complete"
    comparisons = {row["mode"]: row for row in summary["comparisons"]}
    assert comparisons["host"]["workload_change_pct"] == 20
    assert comparisons["storage"]["workload_change_pct"] == 50


def test_missing_lower_tier_exposure_is_not_a_performance_claim(tmp_path):
    _arm(tmp_path, "resident")
    _arm(tmp_path, "host")
    _arm(tmp_path, "storage")
    summary = analyze(tmp_path, [1], ["burst"], ["resident", "host", "storage"])
    assert summary["status"] == "insufficient_exposure"
    assert len(summary["exposure_warnings"]) == 2
