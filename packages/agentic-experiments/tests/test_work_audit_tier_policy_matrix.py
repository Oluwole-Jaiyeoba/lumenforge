import gzip
import json

from agentic_experiments.runners.analyze_work_audit_tier_policy_matrix import analyze


def _arm(root, mode, policy, *, workflow, mismatch_capacity=False):
    folder = root / "arms" / "seed1" / f"burst_{mode}_{policy}"
    folder.mkdir(parents=True)
    request_id = f"paired-s00-{mode}-{policy}-turn01"
    source = "gpu" if mode == "resident" else mode
    backend = {
        "tier_mode": mode,
        "gpu_token_limit": 40960 if mode == "resident" else 12288,
        "host_cache_gb": 3 if mode == "resident" else 2 if mode == "host" else 1,
        "storage_enabled": mode == "storage",
        "storage_backend": "file" if mode == "storage" else None,
        "cuda_graph": True,
        "overlap_schedule": True,
        "trace_profile": "kv_lifecycle_lean",
    }
    if mismatch_capacity and policy == "capacity_safe":
        backend["gpu_token_limit"] += 1
    metrics = {
        "mean_due_to_first_token_ms": workflow / 10,
        "p95_due_to_first_token_ms": workflow / 10,
        "total_due_to_first_token_ms": workflow / 10,
        "mean_ttft_ms": workflow / 20,
        "p95_ttft_ms": workflow / 20,
        "total_ttft_ms": workflow / 20,
        "total_submission_delay_ms": 0,
        "output_tokens": 16,
        "mean_slot_wait_ms": 0,
        "mean_kv_prepare_ms": 0,
    }
    turn = {"turn": 1, "request_id": request_id, "prompt_tokens": 4096}
    if policy == "capacity_safe":
        before = {
            "gpu_tokens": 4096 if source == "gpu" else 0,
            "host_tokens": 4096 if source == "host" else 0,
            "storage_candidate_tokens": 4096 if source == "storage" else 0,
        }
        turn["preparation"] = {"source_tier": source, "before": before}
    case = {
        "seed": 1, "pattern": "burst", "mode": mode, "policy": policy,
        "started_ns": 100, "workflow_duration_ms": workflow,
        "config": {"turns": 1}, "metrics": metrics,
        "per_session_completion_ms": {"s": workflow},
        "workload_fingerprint": "same-workload", "workload_contract": {"same": True},
        "backend_contract": backend,
        "policy_contract": {"policy": policy},
        "sessions": [{"session_id": "s", "turns": [
            {"turn": 0, "request_id": "prime"}, turn,
        ]}],
        "rounds": [{"inspections": []}],
    }
    if policy == "capacity_safe":
        case["capacity"] = {
            "max_active_allowed": 6 if mode == "resident" else 2,
            "active_token_limit": backend["gpu_token_limit"],
            "max_active_observed": 1,
            "max_active_tokens_observed": 4096,
            "violations": [],
        }
    (folder / "case_results.json").write_text(json.dumps(case), encoding="utf-8")

    rid = f"rid-{mode}-{policy}"
    gpu = 4096 if policy == "capacity_safe" or mode == "resident" else 2048
    host = 2048 if policy == "native_sglang" and mode == "host" else 0
    rows = [{
        "event": "hiradix.match_prefix.end",
        "kv_context": {"agent_request_id": request_id, "request": {"rid": rid}},
        "result": [{"index_count": gpu}, {}, {}, host],
    }]
    if policy == "native_sglang" and mode == "storage":
        rows.append({"event": "hiradix.storage_hit_tokens.end", "result": 2048,
                     "kv_context": {"request_id": rid}})
    with gzip.open(folder / "backend_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _matrix(root, *, mismatch_capacity=False):
    for index, mode in enumerate(("resident", "host", "storage")):
        _arm(root, mode, "native_sglang", workflow=1000 + index * 200,
             mismatch_capacity=mismatch_capacity)
        _arm(root, mode, "capacity_safe", workflow=900 + index * 150,
             mismatch_capacity=mismatch_capacity)


def test_matrix_proves_all_tiers_and_pairwise_capacity(tmp_path):
    _matrix(tmp_path)
    summary = analyze(tmp_path, [1], ["burst"])
    assert summary["status"] == "complete"
    assert len(summary["policy_comparisons"]) == 3
    assert len(summary["tier_comparisons"]) == 4
    assert all(row["workflow_change_pct"] < 0 for row in summary["policy_comparisons"])


def test_pairwise_capacity_mismatch_blocks_result(tmp_path):
    _matrix(tmp_path, mismatch_capacity=True)
    summary = analyze(tmp_path, [1], ["burst"])
    assert summary["status"] == "blocked"
    assert sum("capacities differ" in issue for issue in summary["issues"]) == 3
