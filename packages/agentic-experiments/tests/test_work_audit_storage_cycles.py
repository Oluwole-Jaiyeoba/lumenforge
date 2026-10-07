import gzip
import json

from agentic_experiments.runners.analyze_work_audit_storage_cycles import analyze, native_hits


def _case(root, seed, arm, *, due_ms, workflow_ms, hit_tokens):
    folder = root / f"seed{seed}_{arm}"
    folder.mkdir()
    request_id = f"s{seed}-turn01"
    case = {
        "arm": arm, "seed": seed, "started_ns": 100,
        "session_count": 1, "workflow_duration_ms": workflow_ms,
        "sessions": [{"completion_ms": workflow_ms}],
        "natural_storage_candidate_waits": 1,
        "inspections_before_due": 1,
        "stage_before_due_count": int(arm == "host_stage"),
        "turns": [{"turn": 1, "request_id": request_id,
                   "due_to_first_token_ms": due_ms, "ttft_ms": due_ms,
                   "prompt_tokens": 3000,
                   "preparation": {"stage": {"completed": {"storage_loaded_tokens": 512}}}
                   if arm == "host_stage" else None}],
    }
    (folder / "case_results.json").write_text(json.dumps(case), encoding="utf-8")
    with gzip.open(folder / "backend_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        for event in (
            {"event": "hiradix.storage_hit_tokens.end", "result": hit_tokens,
             "kv_context": {"request_id": "backend-rid"}},
            {"event": "hiradix.match_prefix.end",
             "kv_context": {"request": {"rid": "backend-rid"},
                            "agent_request_id": request_id}},
        ):
            handle.write(json.dumps(event) + "\n")
    return folder


def test_native_hit_joins_backend_rid_to_replay(tmp_path):
    folder = _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000,
                   hit_tokens=2048)
    assert native_hits(folder / "backend_trace.jsonl.gz") == {"s1-turn01": 2048}


def test_paired_storage_cycles_keep_workflow_and_stage_cost(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    _case(tmp_path, 1, "host_stage", due_ms=200, workflow_ms=9500, hit_tokens=0)
    summary = analyze(tmp_path)
    assert summary["paired_comparisons"] == [{
        "seed": 1, "mode": "host_stage", "workflow_delta_ms": 500,
        "median_due_to_first_token_delta_ms": -200,
        "median_replay_ttft_delta_ms": -200,
    }]
    staged = next(row for row in summary["arms"] if row["arm"] == "host_stage")
    assert staged["stage_loaded_tokens"] == 512
    assert staged["stage_before_due_count"] == 1


def test_failed_arm_blocks_three_seed_conclusion(tmp_path):
    _case(tmp_path, 1, "on_demand", due_ms=400, workflow_ms=9000, hit_tokens=2048)
    _case(tmp_path, 1, "host_stage", due_ms=200, workflow_ms=9500, hit_tokens=0)
    failure = tmp_path / "seed2_host_stage"
    failure.mkdir()
    (failure / "case_failure.json").write_text(json.dumps({"error": "HTTP 500"}), encoding="utf-8")
    (failure / "server.log").write_text("AssertionError: cache tree\n", encoding="utf-8")
    summary = analyze(tmp_path, expected_seeds=[1, 2, 3],
                      expected_arms=["on_demand", "host_stage"])
    assert summary["status"] == "blocked"
    assert len(summary["paired_comparisons"]) == 1
    assert summary["failed_arms"][0]["backend_assertion"] == "cache tree"
    assert "seed3_host_stage" in summary["missing_arm_ids"]
