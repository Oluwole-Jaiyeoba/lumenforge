import json

import pytest

from agentic_controller.coordinated_swap import PairRotation, SessionPipelinePolicy
from agentic_experiments.runners.run_work_audit_coordinated_swap import initial_prompt, metrics
from agentic_experiments.runners.run_kv_movement_interference import make_prompt
from agentic_experiments.runners.analyze_work_audit_coordinated_swap import analyze, first_matches, measured_controls


def test_rotation_has_two_pairs_not_a_shared_context():
    schedule = PairRotation()
    assert [schedule.pair(i) for i in range(20)] == [0] * 10 + [1] * 10
    assert schedule.tool_due(10) == 11
    assert schedule.boundary(10, 10.5) == 11
    assert schedule.boundary(10, 12) == 12
    with pytest.raises(ValueError):
        PairRotation(21)


def test_session_pipeline_spreads_arrivals_and_evicts_farthest_safe_session():
    policy = SessionPipelinePolicy(prefetch_lead_ms=350, headroom_sessions=2, max_inflight=8)
    assert policy.initial_due_ns(1_000_000_000, 10) == 1_500_000_000
    assert policy.prefetch_at_ns(2_000_000_000) == 1_650_000_000
    assert policy.next_tool_due_ns(2_000_000_000) == 3_000_000_000
    states = [
        {"id": "soon", "due_ns": 3, "resident": True},
        {"id": "late", "due_ns": 9, "resident": True},
        {"id": "active", "due_ns": 12, "resident": True, "active": True},
        {"id": "host", "due_ns": 20, "resident": False},
    ]
    assert policy.victim(states, incoming_due_ns=2)["id"] == "late"
    assert policy.victim(states, incoming_due_ns=10) is None


def test_metrics_include_wait_before_submission():
    row = dict(request_end_ns=2_000_000_000, due_to_first_token_ms=500,
               ttft_ms=100, submission_delay_ms=400, completion_tokens=32)
    value = metrics([row], 1_000_000_000)
    assert value["workload_ms"] == 1000
    assert value["total_due_to_first_token_ms"] == 500
    assert value["total_ttft_ms"] == 100


def test_first_match_not_post_completion_match(tmp_path):
    trace = tmp_path / "trace.jsonl"
    rows = [{"event": "hiradix.match_prefix.end", "kv_context": {"request": {"agent_request_id": "a"}},
             "result": [{"index_count": count}, {}, {}, 0]} for count in (64, 8192)]
    trace.write_text("\n".join(map(json.dumps, rows)))
    assert first_matches(trace)["a"]["gpu_tokens"] == 64


def test_missing_arms_cannot_pass(tmp_path):
    result = analyze(tmp_path, [1], ["coordinated", "resident", "independent"])
    assert result["status"] == "blocked"
    assert len(result["issues"]) == 3


def test_linear_prompt_builder_preserves_existing_prompt():
    assert initial_prompt(2, 128) == make_prompt("private-session-02", 128)


def test_control_totals_exclude_setup_and_do_not_double_count_cuda_polls():
    def control(action, sent, duration=1, **result):
        return dict(action=action, sent_ns=sent * 1_000_000,
                    returned_ns=(sent + duration) * 1_000_000, result=result)
    case = {"started_ns": 10_000_000, "ended_ns": 30_000_000, "controls": [
        control("prepare", 1, load_id="setup"),
        control("prepare_group", 11, load_id="measured"),
        control("load_status", 13, load_id="measured", cuda_elapsed_ms=2),
        control("load_status", 15, load_id="measured", cuda_elapsed_ms=2),
        control("prefix_residency", 31),
    ]}
    result = measured_controls(case)
    assert result["control_count_by_action"] == {"prepare_group": 1, "load_status": 2}
    assert result["native_cuda_interval_count"] == 1
    assert result["native_cuda_interval_total_ms"] == 2
    assert result["control_wall_ms_by_action"] == {"prepare_group": 1, "load_status": 2}
