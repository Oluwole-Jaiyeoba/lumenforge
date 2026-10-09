import gzip
import json

from agentic_experiments.runners.analyze_work_audit_capacity_safe_tiers import analyze


def _arm(root, mode, *, workflow=1000, violate=False):
    folder = root / "arms" / "seed1" / f"burst_{mode}"
    folder.mkdir(parents=True)
    request_id = f"capacity-seed1-{mode}-turn01"
    source = mode if mode != "resident" else "gpu"
    before = {"gpu_tokens": 4096 if source == "gpu" else 0,
              "host_tokens": 4096 if source == "host" else 0,
              "storage_candidate_tokens": 4096 if source == "storage" else 0}
    metrics = {
        "mean_due_to_first_token_ms": workflow / 10,
        "p95_due_to_first_token_ms": workflow / 10,
        "total_due_to_first_token_ms": workflow / 10,
        "mean_ttft_ms": workflow / 20, "p95_ttft_ms": workflow / 20,
        "total_ttft_ms": workflow / 20, "total_submission_delay_ms": 0,
        "mean_slot_wait_ms": 10, "p95_slot_wait_ms": 10, "total_slot_wait_ms": 10,
        "mean_kv_prepare_ms": 20 if mode != "resident" else 0,
        "p95_kv_prepare_ms": 20 if mode != "resident" else 0,
        "total_kv_prepare_ms": 20 if mode != "resident" else 0,
        "output_tokens": 16,
    }
    case = {
        "seed": 1, "pattern": "burst", "mode": mode, "started_ns": 100,
        "workflow_duration_ms": workflow, "config": {"turns": 1}, "metrics": metrics,
        "per_session_completion_ms": {"s": workflow},
        "capacity": {"max_active_allowed": 2, "active_token_limit": 12288,
                     "max_active_observed": 2, "max_active_tokens_observed": 10000,
                     "violations": [{"bad": True}] if violate else []},
        "sessions": [{"session_id": "s", "turns": [
            {"turn": 0, "request_id": "prime"},
            {"turn": 1, "request_id": request_id, "prompt_tokens": 4096,
             "preparation": {"source_tier": source, "before": before}},
        ]}],
    }
    (folder / "case_results.json").write_text(json.dumps(case), encoding="utf-8")
    row = {"event": "hiradix.match_prefix.end",
           "kv_context": {"agent_request_id": request_id, "request": {"rid": request_id}},
           "result": [{"index_count": 4096}, {}, {}, 0]}
    with gzip.open(folder / "backend_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_capacity_safe_tiers_prove_source_and_active_fit(tmp_path):
    _arm(tmp_path, "resident", workflow=1000)
    _arm(tmp_path, "host", workflow=1200)
    _arm(tmp_path, "storage", workflow=1500)
    summary = analyze(tmp_path, [1], ["burst"], ["resident", "host", "storage"])
    assert summary["status"] == "complete"
    arms = {arm["mode"]: arm for arm in summary["arms"]}
    assert arms["host"]["native_host_hit_replays"] == 1
    assert arms["storage"]["native_storage_hit_replays"] == 1
    assert all(arm["native_gpu_ready_replays"] == 1 for arm in arms.values())
    assert {row["mode"]: row["workload_change_pct"] for row in summary["comparisons"]} == {
        "host": 20.0, "storage": 50.0,
    }


def test_capacity_violation_blocks_result(tmp_path):
    _arm(tmp_path, "resident")
    _arm(tmp_path, "host", violate=True)
    _arm(tmp_path, "storage")
    summary = analyze(tmp_path, [1], ["burst"], ["resident", "host", "storage"])
    assert summary["status"] == "blocked"
    assert any("active-capacity invariant failed" in issue for issue in summary["issues"])
