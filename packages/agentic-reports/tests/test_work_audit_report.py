import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentic_reports.builders.build_work_audit_report import _run_finding, main, render


def test_busy_audit_report_keeps_pair_numbers_and_limits_visible():
    arm = {"workflow_makespan_ms": 1000, "total_replay_ttft_ms": 300,
           "total_return_to_first_token_ms": 320,
           "return_to_first_token": {"p95_ms": 320}, "native_load_events": 1}
    controlled = {**arm, "workflow_makespan_ms": 900, "total_replay_ttft_ms": 200,
                  "controller_plan_checks": 3,
                  "controller_load_attempts": 1,
                  "controller_loads_finished_before_tool_return": 1,
                  "controller_loads_finished_after_tool_return": 0}
    summary = {"schema": "agentic_work_audit.busy_comparison.v1", "run_id": "busy",
               "status": "validated", "seed_count": 1, "median_workflow_saved_ms": 100,
               "median_total_replay_ttft_saved_ms": 100,
               "_manifest": {**_manifest(), "workload": {
                   "research_question_id": "RQ8", "session_count": 12,
                   "tool_waits_per_session": 3, "prefix_tokens": 8192,
                   "replay_tokens": 64, "wait_range_ms": [800, 3500],
                   "seeds": [1], "hicache_size_gb": 8, "mem_fraction_static": .8}},
               "pairs": [{"seed": 1, "baseline": arm, "controller": controlled,
                          "workflow_saved_ms": 100, "total_replay_ttft_saved_ms": 100,
                          "eligible_host_checks": 1, "sessions_helped": 10,
                          "sessions_harmed": 2, "evidence_gate": "comparable",
                          "evidence_reasons": []}]}
    page = render([(Path("runs/busy/summary.json"), summary)])
    assert "Busy workload · controller KV timing" in page
    assert "Seed 1 · Ordinary replay" in page
    assert "Seed 1 · Controller-timed KV" in page
    assert "10 sessions helped, 2 harmed" in page
    assert "infra/container/run_work_audit_busy.sh" in page
    assert "not an isolated hardware-bandwidth measurement" in page


def test_attribution_report_exposes_three_arms_and_stage_coverage():
    stage = {key: {"mean_added_ms": 5, "median_added_ms": 5,
                   "matched_replays": 1, "total_replays": 1}
             for key in ("submit_to_receive", "receive_to_queue", "queue_to_cache_lookup",
                         "receive_to_cache_lookup", "substantive_decode",
                         "cache_lookup_to_first_token", "first_token_to_finish")}
    arm = {"total_replay_ttft_ms": 100, "workflow_makespan_ms": 200,
           "controller_plan_checks": 1, "native_load_events": 2,
           "load_windows": {"load_attempts": 1, "confirmed_control_windows": 1,
                            "windows_with_other_replay_before_first_token": 1,
                            "windows_with_other_replay_after_first_token": 0}}
    summary = {"schema": "agentic_work_audit.kv_load_attribution.v1", "run_id": "attr",
               "status": "complete", "seed_count": 1, "interpretation_limit": "Not physical copy time.",
               "_manifest": {**_manifest(), "workload": {
                   "research_question_id": "RQ8", "session_count": 12,
                   "tool_waits_per_session": 3, "modes": ["baseline", "check_only", "controller"],
                   "seeds": [1]}},
               "seeds": [{"seed": 1, "arms": {mode: arm for mode in
                          ("baseline", "check_only", "controller")},
                          "check_cost": stage, "load_association": stage}]}
    page = render([(Path("runs/attr/summary.json"), summary)])
    assert "Busy workload · KV-load attribution" in page
    assert "Checks only" in page and "Checks + loads" in page
    assert "Queue → first cache lookup" in page
    assert "1/1" in page
    assert "Not physical copy time" in page


def _manifest(order=("warm", "host")):
    return {
        "hardware_profile": "nvidia_a10g_24gb",
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "backend_version": "0.5.10.post1",
        "backend_runtime_contract": {"adapter": "v0510"},
        "workload": {
            "cases": list(order), "frontend_priority": "none", "tool_wait_ms": 500,
            "replays_per_case": 2, "pairs": 1, "warmup_pairs": 0,
            "exact_trace_indices": 256,
        },
    }


def test_report_uses_trace_time_and_saved_evidence(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    for name, second in (("first", 1), ("second", 2)):
        run = runs / name
        run.mkdir(parents=True)
        (run / "summary.json").write_text(json.dumps({
            "schema": "agentic_work_audit.validation.v1", "run_id": name,
            "status": "validated",
            "cases": [
                {"case_type": "warm_control", "replay_ttft_ms": 10},
                {"case_type": "host_backed", "replay_ttft_ms": 11},
            ],
        }), encoding="utf-8")
        (run / "run_manifest.json").write_text(json.dumps(_manifest()), encoding="utf-8")
        ns = int(datetime(2026, 10, 2, 14, 30, second, tzinfo=timezone.utc).timestamp() * 1e9)
        (run / "harness_events.jsonl").write_text(
            json.dumps({"kind": "initial_sent", "ts_ns": ns}) + "\n", encoding="utf-8",
        )
        (run / "block_audit.json").write_text(json.dumps({
            "status": "validated", "cases": [{"case_type": "host_backed",
                                               "planned_pre_replay_loaded_tokens": 4096}],
        }), encoding="utf-8")
    out = tmp_path / "index.html"
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"milestones": [{
        "id": "RQ1", "short_question": "Does early KV preparation help?",
        "evidence_date_utc": "2026-10-02",
        "question": "Does early KV preparation help?",
        "answer": "Yes in this controlled test.",
        "unknown": "Whether other sessions are delayed.",
        "evidence_run_ids": ["first", "second"],
        "related_run_ids": ["first", "second"],
    }]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["report", "--results-dir", str(runs),
                                      "--progress-file", str(progress), "--out", str(out)])
    main()
    page = out.read_text(encoding="utf-8")
    assert "Given what the harness knew at the time" in page
    assert "Five audit ledgers" in page
    assert all(name in page for name in (
        "Host backups", "GPU evictions", "Session resumes", "HBM occupancy", "GPU time",
    ))
    assert "not yet graded" in page
    assert "RQ8 tests a busy workload with natural cache pressure" in page
    assert "<th>Date</th><th>Time (UTC)</th>" in page
    assert page.index("second</small>") < page.index("first</small>")
    assert "14:30:02" in page and "14:30:01" in page
    assert page.count("class='detail-toggle'") == 2
    assert "aria-controls='detail-first'" in page
    assert "id='detail-first' hidden><td colspan='9'>" in page
    assert "body{width:100%;margin:0" in page
    assert ".results-table{table-layout:fixed}" in page
    assert "row.hidden = !row.hidden" in page
    assert 'href="runs/first/run_manifest.json"' in page
    assert 'href="runs/second/block_audit.json"' in page
    assert "Reconstructed command" in page
    assert "frontend priority: none" in page
    assert "Research progress" in page
    assert "Does early KV preparation help?" in page
    assert "Yes in this controlled test." in page
    assert 'href="#run-first"' in page and 'href="#run-second"' in page
    assert "Question tested." in page
    assert "<th>Research question</th>" in page
    assert "<th>Main result</th><th>Finding</th><th>Evidence gate</th>" in page
    assert page.count('href="#rq-RQ1"') == 2
    assert 'id=\'rq-RQ1\'' in page
    assert page.count("this was not a policy-speed comparison") == 2


def test_timing_details_keep_pair_metrics_separate():
    summary = {
        "schema": "agentic_work_audit.timing.v1", "run_id": "timing", "status": "validated",
        "_manifest": _manifest(("early", "late")), "_trace_profile": "kv_lifecycle_lean",
        "cases": [
            {"pair": 1, "condition": "early", "first_token_after_due_ms": 82,
             "replay_ttft_ms": 81, "submission_after_due_ms": 1},
            {"pair": 1, "condition": "late", "first_token_after_due_ms": 256,
             "replay_ttft_ms": 83, "submission_after_due_ms": 173},
        ],
        "pairs": [{"pair": 1, "comparable": True,
                   "late_minus_early_first_token_after_due_ms": 174}],
    }
    page = render([(Path("runs/timing/summary.json"), summary)])
    assert "Early vs late" in page
    assert "Late +174.0 ms to first token" in page
    assert "82.0 ms" in page and "256.0 ms" in page
    assert "<th scope='row'>Early load</th>" in page
    assert "<th scope='row'>Late load</th>" in page
    assert "Replay TTFT" in page and "81.0 ms" in page and "83.0 ms" in page
    assert "Before replay submission" in page and "1.0 ms" in page and "173.0 ms" in page
    assert "Trial 1" in page
    assert "kv_lifecycle_lean" in page
    assert "replay TTFT starts only after submission" in page


def test_missing_start_uses_labeled_completion_time():
    manifest = _manifest()
    manifest["created_at_ms"] = 1790951400000
    page = render([(Path("runs/legacy/summary.json"), {
        "run_id": "legacy", "status": "validated", "_manifest": manifest,
    })])
    assert "Manifest completion time (UTC); start unavailable" in page
    assert "<th>Time (UTC)</th>" in page
    assert "Original shell invocation was not saved" in page


def test_missing_all_timestamps_does_not_invent_one():
    page = render([(Path("runs/unknown/summary.json"), {"run_id": "unknown"})])
    assert "No timestamp in saved evidence" in page
    assert "not recorded" in page


def test_future_manifest_shows_effective_request_settings():
    manifest = _manifest()
    manifest["workload"].update({
        "prompt_words_target": 4090, "max_output_tokens": 16,
        "minimum_host_tokens": 512, "eviction_rounds": 4,
        "hicache_size_gb": 8, "mem_fraction_static": 0.70,
    })
    page = render([(Path("runs/future/summary.json"), {
        "run_id": "future", "status": "validated", "_manifest": manifest,
    })])
    assert "Prompt target: 4090 words" in page
    assert "output cap: 16 tokens" in page
    assert "WORK_AUDIT_EVICTION_ROUNDS=4" in page
    assert "MEM_FRACTION_STATIC=0.7" in page


def test_lifecycle_details_separate_replay_timing_from_host_load_evidence():
    summary = {
        "run_id": "host", "status": "validated",
        "cases": [
            {"case_type": "warm_control", "replay_ttft_ms": 81,
             "second_replay_ttft_ms": 82},
            {"case_type": "host_backed", "replay_ttft_ms": 85,
             "second_replay_ttft_ms": 84},
        ],
        "_block_audit": {"cases": [{"case_type": "host_backed",
                                     "planned_pre_replay_loaded_tokens": 4096,
                                     "replay_time_loaded_tokens": 0}]},
    }
    page = render([(Path("runs/host/summary.json"), summary)])
    assert "<th scope='row'>Warm control</th>" in page
    assert "<th scope='row'>Host-backed</th>" in page
    assert "Replay 1 TTFT" in page and "Replay 2 TTFT" in page
    assert "Planned load before replay: 4096 tokens" in page
    assert "load during replay: 0 tokens" in page
    assert "loaded GPU slots in the replay's matched prefix: not recorded" in page


def test_progress_keeps_milestone_order_and_marks_unarchived_evidence():
    milestones = [
        {"id": "RQ2", "short_question": "New question", "question": "New question",
         "answer": "New answer", "unknown": "Next test", "evidence_run_ids": ["missing"]},
        {"id": "RQ1", "short_question": "Older question", "question": "Older question",
         "answer": "Older answer", "unknown": "Old limit", "evidence_run_ids": ["saved"]},
    ]
    page = render([(Path("runs/saved/summary.json"), {"run_id": "saved"})], milestones)
    assert page.index("New question") < page.index("Older question")
    assert "missing (not archived here)" in page
    assert 'href="#run-saved"' in page
    assert "Question tested." in page
    assert 'href="#rq-RQ1"' in page


def test_manifest_question_id_links_without_archived_run_mapping():
    manifest = _manifest()
    manifest["workload"]["research_question_id"] = "RQ1"
    milestone = {"id": "RQ1", "short_question": "Can we trace the KV lifecycle?",
                 "question": "Can host residency and replay be linked?", "answer": "Yes.",
                 "unknown": "Kernel use.", "evidence_run_ids": []}
    page = render([(Path("runs/future/summary.json"), {
        "run_id": "future", "_manifest": manifest,
    })], [milestone])
    assert 'href="#rq-RQ1"' in page
    assert "Can we trace the KV lifecycle?" in page
    assert "Can host residency and replay be linked?" in page


def test_conflicting_question_assignments_fail_loudly():
    manifest = _manifest()
    manifest["workload"]["research_question_id"] = "RQ2"
    milestone = {"id": "RQ1", "short_question": "Lifecycle", "question": "Lifecycle?",
                 "evidence_run_ids": ["saved"]}
    with pytest.raises(ValueError, match="conflicting research question IDs"):
        render([(Path("runs/saved/summary.json"), {
            "run_id": "saved", "_manifest": manifest,
        })], [milestone])


def test_nonblocking_timing_run_exposes_comparability_and_cache_reuse():
    summary = {
        "schema": "agentic_work_audit.timing.v1", "run_id": "triple", "status": "validated",
        "cases": [
            {"pair": 1, "condition": "early", "first_token_after_due_ms": 80},
            {"pair": 1, "condition": "late", "first_token_after_due_ms": 250},
            {"pair": 1, "condition": "late_nonblocking", "first_token_after_due_ms": 95,
             "submission_after_due_ms": 1, "load_accepted_before_replay": False,
             "replay_cache_matches": 0},
        ],
        "pairs": [{"pair": 1, "comparable": True,
                   "late_minus_early_first_token_after_due_ms": 170,
                   "nonblocking_comparable": False,
                   "blocking_minus_nonblocking_first_token_after_due_ms": None}],
    }
    page = render([(Path("runs/triple/summary.json"), summary)])
    assert "Nonblocking late load: 0/1 comparable pair(s)" in page
    assert "95.0 ms" in page
    assert "<th scope='row'>Late, nonblocking</th>" in page
    assert "without waiting for the response" in page
    assert "withheld" in page


def test_multisession_report_keeps_observation_separate_from_avoidability():
    manifest = _manifest(("long", "short", "ends"))
    manifest["workload"].update({"short_wait_ms": 900, "long_wait_ms": 2500,
                                 "prompt_words_target": 4090})
    summary = {
        "schema": "agentic_work_audit.multisession.v1", "run_id": "three",
        "status": "validated", "_manifest": manifest,
        "sessions": {
            "short": {"first_token_after_tool_ms": 90, "cached_prefix_tokens": 2048},
            "long": {"first_token_after_tool_ms": 230, "cached_prefix_tokens": 1024,
                     "second_replay_cached_prefix_tokens": 3072},
        },
        "observations": ["Tool waits overlapped"],
        "plausibly_mistimed": ["Load requested after tool return"],
        "avoidable_work": "unknown: no same-capacity counterfactual",
    }
    page = render([(Path("runs/three/summary.json"), summary)])
    assert "Concurrent timeline" in page
    assert "3 concurrent sessions" in page
    assert "short first token 90.0 ms; long first token 230.0 ms" in page
    assert "<th scope='row'>Short</th>" in page
    assert "<th scope='row'>Long</th>" in page
    assert "Long-session load event" in page
    assert "explicit control command" in page
    assert "Load requested after tool return" in page
    assert "unknown: no same-capacity counterfactual" in page
    assert "WORK_AUDIT_STUDY=multisession" in page


def test_concurrent_comparison_shows_both_session_effects_and_reproduction():
    manifest = _manifest(("late_nonblocking", "early"))
    manifest["workload"].update({"pairs": 2, "short_wait_ms": 900,
                                 "long_wait_ms": 2500, "early_at_ms": 1200,
                                 "prompt_words_target": 4090})
    summary = {
        "schema": "agentic_work_audit.multisession_comparison.v1",
        "run_id": "compare", "status": "validated", "_manifest": manifest,
        "comparable_pairs": 2, "median_long_due_to_token_saved_ms": 150,
        "pairs": [{"pair": 1, "comparable": True, "long_due_to_token_saved_ms": 140,
                   "early_long_due_to_token_ms": 90, "late_long_due_to_token_ms": 230,
                   "early_short_due_to_finish_ms": 900, "late_short_due_to_finish_ms": 892,
                   "short_due_to_finish_change_ms": 8, "workflow_makespan_saved_ms": 50}],
        "cases": [{"pair": 1, "load_timing": "early", "workflow_makespan_ms": 7900},
                  {"pair": 1, "load_timing": "late_nonblocking", "workflow_makespan_ms": 7950}],
    }
    page = render([(Path("runs/compare/summary.json"), summary)])
    assert "Concurrent early vs late" in page
    assert "long replay 150.0 ms faster" in page
    assert "Short return to finish" in page
    assert "<th scope='row'>Late, nonblocking</th>" in page
    assert "<th scope='row'>Early</th>" in page
    assert "7.90 s" in page and "7.95 s" in page
    assert "WORK_AUDIT_STUDY=multisession_compare" in page
    assert "WORK_AUDIT_EARLY_AT_MS=1200" in page


def test_three_window_report_shows_raw_session_and_workflow_metrics():
    manifest = _manifest(("late_nonblocking", "early", "post_short"))
    manifest["workload"].update({"pairs": 2, "short_wait_ms": 900,
                                 "long_wait_ms": 2500, "early_at_ms": 1200})
    summary = {
        "schema": "agentic_work_audit.multisession_window.v1", "run_id": "windows",
        "status": "validated", "_manifest": manifest, "comparable_pairs": 2,
        "median_post_vs_late_long_saved_ms": 160,
        "median_post_vs_early_short_saved_ms": 80,
        "pairs": [{"pair": 1, "comparable": True,
                   "late_long_due_to_token_ms": 250, "early_long_due_to_token_ms": 50,
                   "post_short_long_due_to_token_ms": 90,
                   "late_short_due_to_finish_ms": 100, "early_short_due_to_finish_ms": 180,
                   "post_short_short_due_to_finish_ms": 100,
                   "late_workflow_makespan_ms": 4000,
                   "early_workflow_makespan_ms": 3900,
                   "post_short_workflow_makespan_ms": 3950},
                  {"pair": 2, "comparable": True,
                   "late_long_due_to_token_ms": 600, "early_long_due_to_token_ms": 51,
                   "post_short_long_due_to_token_ms": 91,
                   "late_short_due_to_finish_ms": 101, "early_short_due_to_finish_ms": 181,
                   "post_short_short_due_to_finish_ms": 102,
                   "late_workflow_makespan_ms": 4200,
                   "early_workflow_makespan_ms": 3910,
                   "post_short_workflow_makespan_ms": 3960}],
    }
    page = render([(Path("runs/windows/summary.json"), summary)])
    assert "Three concurrent load windows" in page
    assert "2 measured trials" in page
    assert "post-short long replay 160.0 ms faster vs late" in page
    assert "Short return to finish" in page and "Whole workflow" in page
    assert "<th scope='row'>After short finishes</th>" in page
    assert "Trial 1:</strong> 250.0 ms" in page
    assert "Trial 2:</strong> 600.0 ms" in page
    assert "Trial 1:</strong> 4.00 s" in page
    assert "Trial 2:</strong> 4.20 s" in page
    assert "90.0 ms" in page
    assert "observing that replay finish" in page
    assert "WORK_AUDIT_STUDY=multisession_window" in page


def test_findings_reflect_supported_outcomes_and_withheld_comparisons():
    window = {
        "schema": "agentic_work_audit.multisession_window.v1", "status": "validated",
        "pairs": [{"comparable": True, "post_vs_late_long_saved_ms": 188,
                   "post_vs_early_short_saved_ms": 204,
                   "post_vs_late_workflow_saved_ms": 208}],
    }
    assert _run_finding(window) == (
        "Post-short loading improved long replay and workflow time versus late loading, "
        "while sparing the short session versus early loading."
    )
    assert "mixed effects" in _run_finding({
        **window, "pairs": [{**window["pairs"][0], "post_vs_early_short_saved_ms": -2}],
    })
    assert "conclusion is withheld" in _run_finding({
        **window, "pairs": [{"comparable": False}],
    })
    assert "no performance conclusion" in _run_finding({**window, "status": "failed"})

    comparison = {
        "schema": "agentic_work_audit.multisession_comparison.v1", "status": "validated",
        "pairs": [{"comparable": True, "long_due_to_token_saved_ms": 201,
                   "short_due_to_finish_change_ms": 178,
                   "workflow_makespan_saved_ms": 212}],
    }
    assert "but delayed the short session" in _run_finding(comparison)

    timing = {
        "schema": "agentic_work_audit.timing.v1", "status": "validated",
        "pairs": [{"comparable": True,
                   "late_minus_early_first_token_after_due_ms": 170}],
    }
    assert "first token sooner" in _run_finding(timing)
    nonblocking = {**timing, "cases": [{"condition": "late_nonblocking"}],
                   "pairs": [{**timing["pairs"][0], "nonblocking_comparable": False}]}
    assert "strict nonblocking comparison was withheld" in _run_finding(nonblocking)
    nonblocking["pairs"][0]["pair"] = 1
    nonblocking["cases"] = [
        {"pair": 1, "condition": "late_nonblocking", "submission_after_due_ms": 0.5},
        {"pair": 1, "condition": "late", "submission_after_due_ms": 170},
    ]
    assert "shortened the client gap" in _run_finding(nonblocking)


def test_controller_window_report_shows_decisions_and_reproduction():
    manifest = _manifest(("late_nonblocking", "early", "post_short", "controller_window"))
    manifest["workload"].update({"pairs": 1, "short_wait_ms": 900,
                                 "long_wait_ms": 2500, "early_at_ms": 1200,
                                 "estimated_load_ms": 250, "load_margin_ms": 150})
    pair = {"pair": 1, "comparable": True, "controller_action": "load",
            "late_long_due_to_token_ms": 280, "early_long_due_to_token_ms": 80,
            "post_short_long_due_to_token_ms": 90, "controller_long_due_to_token_ms": 95,
            "late_short_due_to_finish_ms": 820, "early_short_due_to_finish_ms": 1030,
            "post_short_short_due_to_finish_ms": 830, "controller_short_due_to_finish_ms": 825,
            "late_workflow_makespan_ms": 8200, "early_workflow_makespan_ms": 7900,
            "post_short_workflow_makespan_ms": 7910, "controller_workflow_makespan_ms": 7920,
            "controller_vs_late_long_saved_ms": 185,
            "controller_vs_late_workflow_saved_ms": 280}
    summary = {"schema": "agentic_work_audit.controller_window.v1", "run_id": "policy",
               "status": "validated", "_manifest": manifest, "comparable_pairs": 1,
               "pairs": [pair], "median_controller_vs_late_long_saved_ms": 185,
               "median_controller_vs_late_workflow_saved_ms": 280}
    page = render([(Path("runs/policy/summary.json"), summary)])
    assert "Controller-chosen load window" in page
    assert "Controller decision" in page
    assert "Trial 1: load" in page
    assert "WORK_AUDIT_STUDY=multisession_controller" in page
    assert "WORK_AUDIT_ESTIMATED_LOAD_MS=250" in page

def test_nonperformance_runs_do_not_claim_a_speedup():
    assert "not a policy-speed comparison" in _run_finding({
        "schema": "agentic_work_audit.validation.v1", "status": "validated",
        "cases": [{"case_type": "host_backed"}],
    })
    assert "finding is withheld" in _run_finding({
        "schema": "agentic_work_audit.validation.v1", "status": "validated",
    })
    assert "linked overlapping tool waits" in _run_finding({
        "schema": "agentic_work_audit.multisession.v1", "status": "validated",
        "sessions": {"short": {"replay_ttft_ms": 80}, "long": {"replay_ttft_ms": 200}},
    })
    assert "no cross-session finding" in _run_finding({
        "schema": "agentic_work_audit.multisession.v1", "status": "validated",
    })
