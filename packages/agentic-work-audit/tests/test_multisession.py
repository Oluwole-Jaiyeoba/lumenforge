from agentic_work_audit.events import AuditEvent
from agentic_work_audit.multisession import (
    analyze_multisession, compare_controller_windows, compare_multisession_pairs,
    compare_multisession_windows,
)


def event(kind, tick, label="", request="", **evidence):
    return AuditEvent(kind=kind, ts_ns=tick * 1_000_000, source="test",
                      session_id=f"run-{label}" if label else "",
                      request_id=f"run-{label}-{request}" if request else "", evidence=evidence)


def sample():
    return [
        event("runtime_hooks", 1, backend_version="0.5.10.post1", adapter="v0510"),
        event("initial_sent", 1, "long", "initial"),
        event("initial_sent", 1, "short", "initial"),
        event("initial_sent", 1, "ends", "initial"),
        event("initial_finished", 2, "long", "initial"),
        event("initial_finished", 2, "short", "initial"),
        event("initial_finished", 2, "ends", "initial"),
        event("tool_start", 3, "long", "initial", expected_ms=10, importance="equal"),
        event("tool_start", 3, "short", "initial", expected_ms=5, importance="equal"),
        event("capacity_policy", 4, "long", "initial", active_prefix_budget=2,
              action="explicit_evict_long_prefix"),
        event("device_evict_proof", 5, "long", "initial"),
        event("host_resident_proof", 6, "long", "initial"),
        event("session_end", 7, "ends", "initial"),
        event("ended_prefix_evict_proof", 7, "ends", "initial", evicted_tokens=512),
        event("tool_end", 8, "short", "initial"),
        event("replay_sent", 9, "short", "replay"),
        event("cache_match", 10, "short", "replay", cached_prefix_tokens=512),
        event("replay_first_token", 11, "short", "replay"),
        event("replay_finished", 12, "short", "replay"),
        event("tool_end", 13, "long", "initial"),
        event("load_requested", 14, "long", "initial"),
        event("replay_sent", 15, "long", "replay"),
        event("load_accepted", 16, "long", "initial"),
        event("layer_copy", 17, "long", "initial"),
        event("load_complete", 18, "long", "initial"),
        event("cache_match", 19, "long", "replay", cached_prefix_tokens=256),
        event("replay_first_token", 20, "long", "replay"),
        event("replay_finished", 21, "long", "replay"),
        event("replay_2_sent", 22, "long", "replay-2"),
        event("cache_match", 23, "long", "replay-2", cached_prefix_tokens=512),
        event("replay_2_first_token", 24, "long", "replay-2"),
        event("replay_2_finished", 25, "long", "replay-2"),
    ]


def test_multisession_marks_opportunity_without_claiming_avoidable_work():
    result = analyze_multisession(sample(), "run", expected_runtime=("0.5.10.post1", "v0510"))
    assert result["status"] == "validated"
    assert result["setup"]["tool_waits_overlap"]
    assert result["sessions"]["long"]["load_requested_after_tool_return"]
    assert result["sessions"]["long"]["native_load_accepted_before_replay"] is False
    assert result["sessions"]["long"]["load_acceptance_after_tool_ms"] == 3
    assert result["plausibly_mistimed"]
    assert result["avoidable_work"].startswith("unknown")


def test_multisession_requires_second_replay_linkage():
    result = analyze_multisession([row for row in sample() if row.kind != "cache_match" or
                                   not row.request_id.endswith("replay-2")], "run",
                                  expected_runtime=("0.5.10.post1", "v0510"))
    assert result["status"] == "failed"
    assert any("second-replay reuse" in failure for failure in result["failures"])


def test_multisession_rejects_frontend_importance():
    rows = [event(row.kind, row.ts_ns // 1_000_000,
                  row.session_id.removeprefix("run-") if row.session_id else "",
                  row.request_id.removeprefix(row.session_id + "-") if row.request_id else "",
                  **({**row.evidence, "importance": "high"} if row.kind == "tool_start" else row.evidence))
            for row in sample()]
    result = analyze_multisession(rows, "run", expected_runtime=("0.5.10.post1", "v0510"))
    assert result["status"] == "failed"
    assert any("importance" in failure for failure in result["failures"])


def test_early_case_requires_load_completion_before_tool_return_and_released_slot():
    rows = [row for row in sample() if row.kind not in (
        "load_requested", "load_accepted", "layer_copy", "load_complete")]
    rows += [event("load_requested", 9, "long", "initial"),
             event("load_accepted", 10, "long", "initial"),
             event("layer_copy", 11, "long", "initial"),
             event("load_complete", 12, "long", "initial")]
    result = analyze_multisession(rows, "run", expected_runtime=("0.5.10.post1", "v0510"),
                                  condition="early", require_end_eviction=True)
    assert result["status"] == "validated"
    assert result["sessions"]["long"]["load_requested_after_tool_return"] is False
    assert result["load_overlapped_short_request"]
    without_release = [row for row in rows if row.kind != "ended_prefix_evict_proof"]
    rejected = analyze_multisession(without_release, "run", expected_runtime=("0.5.10.post1", "v0510"),
                                    condition="early", require_end_eviction=True)
    assert rejected["status"] == "failed"
    assert any("released" in failure for failure in rejected["failures"])


def test_pair_comparison_reports_other_session_cost_separately():
    late = analyze_multisession(sample(), "run", expected_runtime=("0.5.10.post1", "v0510"),
                                require_end_eviction=True)
    early = {**late, "load_timing": "early", "run_id": "early",
             "load_overlapped_short_request": True,
             "sessions": {"short": {**late["sessions"]["short"], "completion_after_tool_ms": 10},
                          "long": {**late["sessions"]["long"], "first_token_after_tool_ms": 4}},
             "workflow_makespan_ms": late["workflow_makespan_ms"] - 3}
    late["pair"] = early["pair"] = 1
    result = compare_multisession_pairs([late, early], "study")
    assert result["status"] == "validated"
    assert result["pairs"][0]["long_due_to_token_saved_ms"] == 3
    assert result["pairs"][0]["short_due_to_finish_change_ms"] == 6
    assert result["pairs"][0]["workflow_makespan_saved_ms"] == 3


def test_pair_comparison_rejects_cold_start_drift():
    late = analyze_multisession(sample(), "run", expected_runtime=("0.5.10.post1", "v0510"),
                                require_end_eviction=True)
    early = {**late, "load_timing": "early", "run_id": "early", "initial_phase_ms": 4000,
             "load_overlapped_short_request": True}
    late["pair"] = early["pair"] = 1
    result = compare_multisession_pairs([late, early], "study")
    assert result["status"] == "failed"
    assert result["comparable_pairs"] == 0
    assert any("initial phases differ" in reason for reason in result["pairs"][0]["reasons"])


def test_post_short_requires_load_after_short_finish_and_before_long_return():
    rows = [row for row in sample() if row.kind not in (
        "load_requested", "load_accepted", "layer_copy", "load_complete")]
    rows += [event("load_requested", 12.1, "long", "initial"),
             event("load_accepted", 12.2, "long", "initial"),
             event("layer_copy", 12.3, "long", "initial"),
             event("load_complete", 12.4, "long", "initial")]
    good = analyze_multisession(rows, "run", expected_runtime=("0.5.10.post1", "v0510"),
                                condition="post_short", require_end_eviction=True)
    assert good["status"] == "validated"
    assert not good["load_overlapped_short_request"]
    early_request = [event("load_requested", 11.9, "long", "initial") if row.kind ==
                     "load_requested" else row for row in rows]
    bad = analyze_multisession(early_request, "run", expected_runtime=("0.5.10.post1", "v0510"),
                               condition="post_short", require_end_eviction=True)
    assert bad["status"] == "failed"
    assert any("after the short replay finished" in reason for reason in bad["failures"])


def test_three_mode_comparison_keeps_raw_outcomes_and_gates_post_short():
    late = analyze_multisession(sample(), "run", expected_runtime=("0.5.10.post1", "v0510"),
                                require_end_eviction=True)
    early = {**late, "load_timing": "early", "run_id": "early",
             "load_overlapped_short_request": True,
             "sessions": {"short": {**late["sessions"]["short"], "completion_after_tool_ms": 10},
                          "long": {**late["sessions"]["long"], "first_token_after_tool_ms": 4}},
             "workflow_makespan_ms": late["workflow_makespan_ms"] - 3}
    post = {**late, "load_timing": "post_short", "run_id": "post",
            "sessions": {"short": {**late["sessions"]["short"], "completion_after_tool_ms": 5},
                         "long": {**late["sessions"]["long"], "first_token_after_tool_ms": 5}},
            "workflow_makespan_ms": late["workflow_makespan_ms"] - 2}
    for case in (late, early, post):
        case["pair"] = 1
    result = compare_multisession_windows([late, early, post], "study")
    assert result["status"] == "validated"
    pair = result["pairs"][0]
    assert pair["post_vs_late_long_saved_ms"] == 2
    assert pair["post_vs_early_short_saved_ms"] == 5
    assert pair["post_vs_late_workflow_saved_ms"] == 2
    assert pair["post_short_short_due_to_finish_ms"] == 5
    post["status"] = "failed"
    assert compare_multisession_windows([late, early, post], "study")["status"] == "failed"
    post["status"] = "validated"
    duplicate = compare_multisession_windows([late, early, post, post], "study")
    assert duplicate["status"] == "failed"
    assert any("duplicate post-short" in reason for reason in duplicate["failures"])


def test_controller_case_requires_observation_decision_and_confirmed_window():
    rows = [row for row in sample() if row.kind not in (
        "load_requested", "load_accepted", "layer_copy", "load_complete")]
    rows += [event("controller_prepare_decision", 8, "long", "initial", action="hold",
                   reason="blocking_replay_not_finished"),
             event("controller_prepare_decision", 12.05, "long", "initial", action="load",
                   reason="quiet_window_before_tool_return"),
             event("load_requested", 12.1, "long", "initial"),
             event("load_accepted", 12.2, "long", "initial"),
             event("layer_copy", 12.3, "long", "initial"),
             event("load_complete", 12.4, "long", "initial"),
             event("controller_prepare_outcome", 12.5, "long", "initial", outcome="good_window")]
    result = analyze_multisession(rows, "run", expected_runtime=("0.5.10.post1", "v0510"),
                                  condition="controller_window", require_end_eviction=True)
    assert result["status"] == "validated"
    assert [item["action"] for item in result["controller_decisions"]] == ["hold", "load"]
    missing = [row for row in rows if row.kind != "controller_prepare_decision"]
    assert analyze_multisession(missing, "run", expected_runtime=("0.5.10.post1", "v0510"),
                                condition="controller_window")["status"] == "failed"


def test_four_mode_comparison_requires_controller_case():
    late = analyze_multisession(sample(), "run", expected_runtime=("0.5.10.post1", "v0510"),
                                require_end_eviction=True)
    late["pair"] = 1
    early = {**late, "load_timing": "early", "run_id": "early",
             "load_overlapped_short_request": True}
    post = {**late, "load_timing": "post_short", "run_id": "post"}
    controller = {**late, "load_timing": "controller_window", "run_id": "controller",
                  "controller_decisions": [{"action": "hold"}, {"action": "load"}]}
    result = compare_controller_windows([late, early, post, controller], "study")
    assert result["status"] == "validated"
    assert result["pairs"][0]["controller_action"] == "load"
    assert compare_controller_windows([late, early, post], "study")["status"] == "failed"
