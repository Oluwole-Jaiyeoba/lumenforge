import copy

import pytest

from agentic_experiments.runners.analyze_busy_kv_audit import compare
from agentic_experiments.runners.run_busy_kv_audit import wait_ms


def arm(mode: str, *, natural_loads: int = 1, prepared: int = 1) -> dict:
    return {
        "schema": "agentic_work_audit.busy_arm.v1", "seed": 1, "mode": mode,
        "run_id": f"seed1_{mode}", "workload": {"session_count": 1, "tool_waits_per_session": 1},
        "frontend_priority": "none", "forced_eviction": False,
        "replay_count": 1, "workflow_makespan_ms": 1000 if mode == "baseline" else 900,
        "total_replay_ttft_ms": 300 if mode == "baseline" else 200,
        "total_return_to_first_token_ms": 320 if mode == "baseline" else 220,
        "return_to_first_token": {"median_ms": 320 if mode == "baseline" else 220,
                                  "p95_ms": 320 if mode == "baseline" else 220, "max_ms": 320},
        "native_load_events": natural_loads,
        "controller_plan_checks": prepared,
        "controller_load_attempts": prepared,
        "controller_loads_finished_before_tool_return": prepared,
        "controller_loads_finished_after_tool_return": 0,
        "decisions": [{"plan_status": "would_load_back"}] if prepared else [],
        "replays": [{"request_id": "s1-replay-1", "session_id": "s1",
                     "tool_return_to_first_token_ms": 320 if mode == "baseline" else 220}],
    }


def test_comparison_requires_matching_work_and_keeps_session_harm() -> None:
    result = compare([arm("baseline"), arm("controller")], "busy")
    assert result["status"] == "validated"
    assert result["pairs"][0]["workflow_saved_ms"] == 100
    assert result["pairs"][0]["sessions_helped"] == 1
    changed = arm("controller")
    changed["workload"]["session_count"] = 2
    with pytest.raises(ValueError, match="different workload"):
        compare([arm("baseline"), changed], "busy")


def test_comparison_marks_missing_natural_opportunity_inconclusive() -> None:
    result = compare([arm("baseline", natural_loads=0), arm("controller", prepared=0)], "busy")
    assert result["status"] == "inconclusive"
    assert result["pairs"][0]["evidence_gate"] == "limited"


def test_seed_recipe_is_arm_independent() -> None:
    class Args:
        seed = 2
        wait_min_ms = 800
        wait_max_ms = 3500

    args = Args()
    values = [wait_ms(args, index, step) for index in range(12) for step in range(3)]
    assert values == [wait_ms(copy.copy(args), index, step)
                      for index in range(12) for step in range(3)]
    assert min(values) >= args.wait_min_ms and max(values) <= args.wait_max_ms
