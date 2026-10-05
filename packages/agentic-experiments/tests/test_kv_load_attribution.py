import json

from agentic_experiments.runners.analyze_kv_load_attribution import _load_phases, compare


def test_load_phases_keep_cpu_and_cuda_timings_separate(tmp_path):
    events = [
        {"event": "agentic_kv.prepare_prefix.result", "load_id": "load-1", "loaded_tokens": 512,
         "control_queued_ns": 1_000_000, "control_dequeued_ns": 3_000_000,
         "command_started_ns": 4_000_000, "load_back_returned_ns": 10_000_000,
         "ready_to_load_started_ns": 11_000_000, "ready_to_load_returned_ns": 13_000_000},
        {"event": "agentic_kv.prepare_prefix.load_status", "load_id": "load-1",
         "status": "finished", "cuda_elapsed_ms": 8.5},
    ]
    (tmp_path / "backend_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in events) + "\n", encoding="utf-8")
    assert _load_phases(tmp_path) == [{
        "load_id": "load-1", "loaded_tokens": 512, "control_queue_ms": 2.0,
        "scheduler_preparation_ms": 1.0, "load_back_call_ms": 6.0,
        "before_ready_call_ms": 1.0, "ready_to_load_call_ms": 2.0,
        "cuda_elapsed_ms": 8.5,
    }]


def test_three_arms_separate_checks_from_loads(tmp_path):
    times = {"baseline": (10, 20, 40, 60),
             "check_only": (12, 23, 43, 65),
             "controller": (14, 29, 55, 78)}
    for mode, (received, queued, matched, first) in times.items():
        arm_dir = tmp_path / f"seed1_{mode}"
        arm_dir.mkdir()
        replay = {"request_id": "s-replay-1", "request_start_ns": 0,
                  "first_token_ns": first * 1_000_000, "request_end_ns": (first + 10) * 1_000_000,
                  "stream_chunks": 4}
        summary = {"seed": 1, "mode": mode, "run_id": mode, "frontend_priority": "none",
                   "forced_eviction": False, "workload": {"sessions": 1}, "replay_count": 1,
                   "replays": [replay], "decisions": [], "controller_load_attempts": 0,
                   "controller_plan_checks": 0, "native_load_events": 1,
                   "total_replay_ttft_ms": first, "workflow_makespan_ms": first + 10}
        (arm_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        events = [
            {"event": "kv_telemetry.request_stage", "phase": "start", "request_id": "s-replay-1",
             "stage_order": order, "ts_ns": ms * 1_000_000}
            for order, ms in ((10, received), (30, queued), (70, matched))
        ]
        (arm_dir / "backend_trace.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    result = compare(tmp_path, "test")
    seed = result["seeds"][0]
    assert seed["check_cost"]["submit_to_receive"]["mean_added_ms"] == 2
    assert seed["load_association"]["queue_to_cache_lookup"]["mean_added_ms"] == 6
    assert seed["load_association"]["cache_lookup_to_first_token"]["mean_added_ms"] == 1
    assert seed["load_association"]["substantive_decode"]["matched_replays"] == 1
    assert seed["load_association"]["substantive_decode"]["mean_added_ms"] == 0


def test_missing_stage_is_unmeasured_not_zero(tmp_path):
    for mode in ("baseline", "check_only", "controller"):
        arm_dir = tmp_path / f"seed1_{mode}"
        arm_dir.mkdir()
        (arm_dir / "summary.json").write_text(json.dumps({
            "seed": 1, "mode": mode, "run_id": mode, "frontend_priority": "none",
            "forced_eviction": False, "workload": {}, "replay_count": 1,
            "replays": [{"request_id": "r", "request_start_ns": 1,
                         "first_token_ns": 2, "request_end_ns": 3}],
            "decisions": [], "controller_load_attempts": 0, "controller_plan_checks": 0,
            "native_load_events": 0, "total_replay_ttft_ms": 1, "workflow_makespan_ms": 2,
        }), encoding="utf-8")
        (arm_dir / "backend_trace.jsonl").write_text("", encoding="utf-8")
    result = compare(tmp_path, "test")
    missing = result["seeds"][0]["load_association"]["queue_to_cache_lookup"]
    assert missing["matched_replays"] == 0
    assert missing["mean_added_ms"] is None
