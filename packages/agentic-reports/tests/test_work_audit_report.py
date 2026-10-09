import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentic_reports.builders.build_work_audit_report import (
    _markdown_index_outcome, _pair_rq11_controls, _result_parts, _run_finding, main, render,
    render_markdown,
)


def test_memory_tier_report_proves_native_tier_hits_and_full_system_metrics():
    base = {
        "seed": 1, "pattern": "burst", "started_ns": 1791485498893546751,
        "session_count": 6, "turns_per_session": 10, "replay_count": 60,
        "p95_due_to_first_token_ms": 120, "mean_ttft_ms": 90,
        "p95_ttft_ms": 115, "total_due_to_first_token_ms": 6000,
        "total_ttft_ms": 5400, "total_submission_delay_ms": 8,
        "native_match_count": 60, "native_gpu_hit_replays": 60,
        "native_gpu_hit_tokens": 200000, "native_host_hit_replays": 0,
        "native_host_hit_tokens": 0, "native_storage_hit_replays": 0,
        "native_storage_hit_tokens": 0,
    }
    arms = []
    for mode, workload, delay in (("resident", 20000, 100), ("host", 24000, 300),
                                   ("storage", 30000, 600)):
        arm = dict(base, mode=mode, workflow_duration_ms=workload,
                   mean_due_to_first_token_ms=delay)
        if mode == "host":
            arm.update(native_gpu_hit_replays=30, native_gpu_hit_tokens=100000,
                       native_host_hit_replays=30, native_host_hit_tokens=100000)
        if mode == "storage":
            arm.update(native_gpu_hit_replays=20, native_gpu_hit_tokens=70000,
                       native_host_hit_replays=20, native_host_hit_tokens=70000,
                       native_storage_hit_replays=20, native_storage_hit_tokens=60000)
        arms.append(arm)
    comparisons = [
        {"seed": 1, "pattern": "burst", "mode": mode,
         "workload_delta_ms": delta, "workload_change_pct": pct,
         "mean_due_to_first_token_delta_ms": replay, "p95_due_to_first_token_delta_ms": replay,
         "mean_ttft_delta_ms": replay}
        for mode, delta, pct, replay in (("host", 4000, 20, 200), ("storage", 10000, 50, 500))
    ]
    summary = {
        "schema": "agentic_work_audit.memory_tiers.summary.v1", "run_id": "tiers-1",
        "status": "complete", "started_ns": base["started_ns"], "arms": arms,
        "comparisons": comparisons, "issues": [], "exposure_warnings": [],
        "limitations": ["Synthetic equal-priority sessions."],
        "_manifest": {"model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                      "hardware_profile": "nvidia_a10g_24gb", "backend_version": "0.5.10.post1",
                      "workload": {"research_question_id": "RQ23", "seeds": [1],
                                   "return_patterns": ["burst"],
                                   "modes": ["resident", "host", "storage"],
                                   "session_count": 6, "turns_per_session": 10,
                                   "initial_tokens": 4096, "tool_result_words": 16,
                                   "decode_tokens": 16, "tool_wait_ms": 1000,
                                   "burst_window_ms": 75, "spread_window_ms": 1000,
                                   "resident_gpu_tokens": 40960, "restricted_gpu_tokens": 8192,
                                   "host_cache_gb": 2, "storage_host_cache_gb": 1,
                                   "max_inflight": 6, "cuda_graph": True,
                                   "overlap_schedule": True, "trace_profile": "kv_lifecycle_lean"}},
    }
    path = Path("docs/reports/work_audit/tiers-1/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    for document in (page, markdown):
        assert "GPU / CPU / storage KV gap" in document
        assert "Proven GPU / CPU / storage source requests" in document
        assert "20 / 20 / 20" in document
        assert "run_work_audit_memory_tiers.sh" in document
    assert "20.00 s → 24.00 s → 30.00 s" in markdown.replace("&nbsp;", " ")


def test_capacity_safe_tier_report_preserves_admission_limits_and_reproduction():
    arms = []
    for mode, workflow, source in (
        ("resident", 30000, "gpu"), ("host", 34000, "host"),
        ("storage", 60000, "storage"),
    ):
        hits = {tier: 60 if tier == source else 0 for tier in ("gpu", "host", "storage")}
        arms.append({
            "seed": 1, "pattern": "burst", "mode": mode,
            "workflow_duration_ms": workflow, "mean_due_to_first_token_ms": workflow / 30,
            "p95_due_to_first_token_ms": workflow / 20, "mean_ttft_ms": 150,
            "p95_ttft_ms": 180, "total_due_to_first_token_ms": workflow * 2,
            "total_ttft_ms": 9000, "total_submission_delay_ms": workflow,
            "native_gpu_hit_replays": hits["gpu"], "native_gpu_hit_tokens": hits["gpu"] * 4096,
            "native_host_hit_replays": hits["host"], "native_host_hit_tokens": hits["host"] * 4096,
            "native_storage_hit_replays": hits["storage"],
            "native_storage_hit_tokens": hits["storage"] * 4096,
            "max_active_observed": 2, "max_active_allowed": 2,
            "max_active_tokens_observed": 9342, "active_token_limit": 12288,
            "mean_slot_wait_ms": 100, "mean_kv_prepare_ms": 80,
        })
    comparisons = [
        {"seed": 1, "pattern": "burst", "mode": mode,
         "workload_delta_ms": delta, "workload_change_pct": pct,
         "mean_due_to_first_token_delta_ms": delay,
         "p95_due_to_first_token_delta_ms": delay, "mean_ttft_delta_ms": 0}
        for mode, delta, pct, delay in (("host", 4000, 13.3, 300),
                                        ("storage", 30000, 100, 1500))
    ]
    summary = {
        "schema": "agentic_work_audit.memory_tiers.summary.v1",
        "variant": "capacity_safe_active_set", "run_id": "capacity-safe",
        "status": "complete", "arms": arms, "comparisons": comparisons,
        "issues": [], "exposure_warnings": [], "limitations": [],
        "_manifest": {"model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                      "hardware_profile": "nvidia_a10g_24gb", "backend_version": "0.5.10.post1",
                      "workload": {"research_question_id": "RQ24", "seeds": [1],
                                   "return_patterns": ["burst"],
                                   "modes": ["resident", "host", "storage"],
                                   "session_count": 6, "turns_per_session": 10,
                                   "initial_tokens": 4096, "decode_tokens": 16,
                                   "tool_wait_ms": 1000, "burst_window_ms": 75,
                                   "spread_window_ms": 1000, "resident_gpu_tokens": 40960,
                                   "restricted_gpu_tokens": 12288, "host_cache_gb": 2,
                                   "storage_host_cache_gb": 1, "max_active": 2,
                                   "active_token_limit": 12288, "cuda_graph": True,
                                   "overlap_schedule": True,
                                   "trace_profile": "kv_lifecycle_lean"}},
    }
    path = Path("docs/reports/work_audit/capacity-safe/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    for document in (page, markdown):
        assert "2 / 2" in document
        assert "9,342 / 12,288 tokens" in document
        assert "Waiting sessions were restored completely before admission" in document
        assert "run_work_audit_capacity_safe_tiers.sh" in document
        assert "CAPACITY_TIER_MAX_ACTIVE=2" in document


def test_storage_audit_report_shows_native_hits_and_reproduction():
    rows = [
        {"seed": 1, "arm": arm, "due_to_first_token_ms": delay,
         "replay_ttft_ms": delay, "workflow_duration_ms": 21000 + delay,
         "native_replay_storage_hit_tokens": 2100 if arm == "on_demand" else 0,
         "control_storage_hit_tokens": 0 if arm == "on_demand" else 2100,
         "stage_completed_before_due": arm != "on_demand"}
        for arm, delay in (("on_demand", 4000), ("host_stage", 200), ("full_prepare", 100))
    ]
    summary = {"schema": "agentic_work_audit.storage_replay.v1", "run_id": "storage-1",
               "status": "complete", "rows": rows, "paired": [{"seed": 1}],
               "median_host_stage_delta_ms": -3800, "median_full_prepare_delta_ms": -3900,
               "_manifest": {"model": "Qwen/Qwen2.5-1.5B-Instruct",
                             "hardware_profile": "nvidia_standard", "backend_version": "0.5.10.post1",
                             "created_at_ms": 1791320000000,
                             "workload": {"research_question_id": "RQ15", "seeds": [1],
                                          "prompt_tokens": 2048, "tool_wait_ms": 20000}}}
    page = render([(Path("runs/storage-1/summary.json"), summary)])
    markdown = render_markdown([(Path("runs/storage-1/summary.json"), summary)])
    assert "Storage-tier KV timing" in page
    assert "L3 tokens at replay" in page
    assert "4000 → 200 → 100 ms" in markdown.replace("&nbsp;", " ")
    assert "run_work_audit_storage.sh" in markdown
    assert "seed1_on_demand" in markdown


def test_two_arm_storage_ladder_report_keeps_peer_and_workflow_changes():
    rows = [
        {"seed": 1, "arm": arm, "due_to_first_token_ms": delay,
         "replay_ttft_ms": delay, "workflow_duration_ms": workflow,
         "peer_count": 1, "peer_ttft_median_ms": peer_ttft,
         "peer_completion_median_ms": peer_finish,
         "peers_overlapping_preparation": int(arm == "host_stage"),
         "native_replay_storage_hit_tokens": 2048 if arm == "on_demand" else 0,
         "control_storage_hit_tokens": 0 if arm == "on_demand" else 2048,
         "stage_completed_before_due": arm == "host_stage"}
        for arm, delay, workflow, peer_ttft, peer_finish in (
            ("on_demand", 320, 6700, 180, 500),
            ("host_stage", 160, 6800, 220, 530),
        )
    ]
    summary = {"schema": "agentic_work_audit.storage_replay.v1", "run_id": "ladder-2",
               "status": "complete", "rows": rows,
               "paired": [{"seed": 1, "host_stage_delta_ms": -160,
                           "host_stage_workflow_delta_ms": 100,
                           "peer_ttft_deltas_ms": [40], "peer_completion_deltas_ms": [30]}],
               "median_host_stage_delta_ms": -160, "median_full_prepare_delta_ms": None,
               "_manifest": {"hardware_profile": "nvidia_a10g_24gb",
                             "backend_version": "0.5.10.post1", "model": "qwen",
                             "workload": {"research_question_id": "RQ18", "seeds": [1],
                                          "arms": ["on_demand", "host_stage"],
                                          "prompt_tokens": 2048, "tool_wait_ms": 5000,
                                          "peer_count": 1, "cuda_graph": True,
                                          "overlap_schedule": True}}}
    path = Path("docs/reports/work_audit/ladder-2/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Safe storage-stage session ladder" in page
    assert "Each peer" in page and "TTFT change" in page
    assert "On demand → host stage" in markdown
    assert "Whole workload: 6.70 → 6.80 s" in markdown.replace("&nbsp;", " ")
    assert "WORK_AUDIT_STORAGE_CUDA_GRAPH=1" in markdown


def test_storage_control_placebo_report_separates_poll_wait_and_peer_phases():
    rows = []
    for arm, peer_ttft in (("on_demand", 100), ("control_only", 125), ("host_stage", 160)):
        rows.append({"seed": 1, "arm": arm, "due_to_first_token_ms": 300,
                     "replay_ttft_ms": 150, "workflow_duration_ms": 6000,
                     "peer_count": 1, "peer_ttft_median_ms": peer_ttft,
                     "peer_completion_median_ms": 2000, "peers_overlapping_preparation": 1,
                     "native_replay_storage_hit_tokens": 2048 if arm != "host_stage" else 0,
                     "control_storage_hit_tokens": 2048 if arm == "host_stage" else 0,
                     "stage_completed_before_due": arm == "host_stage",
                     "peer_phase_times": [{"request_to_lookup_ms": 20,
                                           "lookup_to_first_token_ms": peer_ttft - 20}],
                     "storage_ready_to_poll_ms": 188 if arm == "host_stage" else None,
                     "storage_poll_queue_ms": 700 if arm == "host_stage" else None,
                     "storage_poll_execution_ms": 6 if arm == "host_stage" else None})
    summary = {"schema": "agentic_work_audit.storage_replay.v1", "run_id": "placebo-1",
               "status": "complete", "rows": rows, "paired": [{
                   "seed": 1, "host_stage_delta_ms": 0,
                   "control_only_peer_ttft_deltas_ms": [25],
                   "host_stage_peer_ttft_deltas_ms": [60],
                   "control_only_peer_completion_deltas_ms": [20],
                   "host_stage_peer_completion_deltas_ms": [50]}],
               "median_host_stage_delta_ms": 0, "median_full_prepare_delta_ms": None,
               "_manifest": {"workload": {"research_question_id": "RQ19", "peer_count": 1,
                                          "arms": ["on_demand", "control_only", "host_stage"]}}}
    path = Path("docs/reports/work_audit/placebo-1/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Storage staging and peer-delay control" in page
    assert "Poll queued" in page
    assert "Peer request → cache lookup" in markdown
    assert "Control-only peer TTFT changes" in page
    assert "WORK_AUDIT_STORAGE_ARMS='on_demand control_only host_stage'" in markdown


def test_storage_cycles_report_shows_paired_workflow_and_replay_cost():
    arms = [{
        "seed": 1, "arm": mode, "session_count": 8, "replay_count": 48,
        "workflow_duration_ms": workflow, "due_to_first_token_median_ms": delay,
        "due_to_first_token_p95_ms": delay + 100, "replay_ttft_median_ms": delay,
        "natural_storage_candidate_waits": 10, "stage_before_due_count": staged,
        "stage_loaded_tokens": staged * 512, "native_replay_storage_hit_count": 2,
        "native_replay_storage_hit_tokens": 2048, "preparation_errors": [],
    } for mode, workflow, delay, staged in (
        ("on_demand", 40000, 500, 0), ("host_stage", 41000, 300, 7))]
    summary = {
        "schema": "agentic_work_audit.storage_cycles.summary.v1", "arms": arms,
        "paired_comparisons": [{"seed": 1, "mode": "host_stage", "workflow_delta_ms": 1000,
                                "median_due_to_first_token_delta_ms": -200,
                                "median_replay_ttft_delta_ms": -200}],
        "_manifest": {"run_id": "storage_cycles_rq17_20261006",
                      "hardware_profile": "nvidia_a10g_24gb", "backend_version": "0.5.10.post1",
                      "workload": {"seeds": [1], "arms": ["on_demand", "host_stage"],
                                   "session_count": 8, "turns_per_session": 6}},
    }
    path = Path("docs/reports/work_audit/rq17/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Repeated storage-resume timing" in page
    assert "Whole-workload change" in page
    assert "Full GPU preparation is excluded" in page
    assert "Whole workload: 40.00 → 41.00 s" in markdown.replace("&nbsp;", " ")
    assert "run_work_audit_storage_cycles.sh" in markdown


def test_selective_storage_report_shows_policy_and_each_session():
    arms = [{"seed": 1, "arm": mode, "session_count": 2, "replay_count": 12,
             "workflow_duration_ms": duration, "due_to_first_token_median_ms": delay,
             "due_to_first_token_p95_ms": delay + 100, "replay_ttft_median_ms": delay,
             "natural_storage_candidate_waits": 4, "stage_before_due_count": staged,
             "stage_loaded_tokens": staged * 512, "native_replay_storage_hit_count": 2,
             "native_replay_storage_hit_tokens": 1024, "preparation_errors": []}
            for mode, duration, delay, staged in (
                ("on_demand", 20000, 500, 0), ("selective_stage", 19000, 300, 2))]
    summary = {"schema": "agentic_work_audit.storage_cycles.summary.v1", "status": "complete",
               "arms": arms, "paired_comparisons": [{"seed": 1, "mode": "selective_stage",
                   "workflow_delta_ms": -1000, "median_due_to_first_token_delta_ms": -200,
                   "median_replay_ttft_delta_ms": -200,
                   "session_completion_delta_by_id_ms": {"session-0": -1000, "session-1": 100}}],
               "_manifest": {"run_id": "rq20_selective_pair", "model": "model",
                   "workload": {"research_question_id": "RQ20", "seeds": [1],
                       "arms": ["on_demand", "selective_stage"], "session_count": 2,
                       "turns_per_session": 6, "inspect_fraction": 0.2,
                       "min_stage_slack_ms": 1200}}}
    path = Path("docs/reports/work_audit/rq20_selective_pair/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Selective storage staging across repeated sessions" in page
    assert "Finish-time change" in page
    assert "WORK_AUDIT_CYCLES_INSPECT_FRACTION" in markdown
    assert "session-1" in markdown


def test_selective_storage_mixed_seeds_are_not_averaged_into_a_win():
    arms = [{"seed": seed, "arm": arm, "session_count": 8, "replay_count": 48,
             "workflow_duration_ms": duration, "due_to_first_token_median_ms": delay,
             "due_to_first_token_p95_ms": delay + 100,
             "replay_ttft_median_ms": delay, "natural_storage_candidate_waits": 4,
             "stage_before_due_count": int(arm == "selective_stage"),
             "stage_loaded_tokens": 512, "native_replay_storage_hit_count": 2,
             "native_replay_storage_hit_tokens": 1024, "preparation_errors": []}
            for seed, arm, duration, delay in (
                (1, "on_demand", 28000, 1000), (1, "selective_stage", 27000, 1100),
                (2, "on_demand", 29000, 900), (2, "selective_stage", 30000, 1000))]
    summary = {"schema": "agentic_work_audit.storage_cycles.summary.v1", "status": "complete",
               "arms": arms, "paired_comparisons": [
                   {"seed": 1, "mode": "selective_stage", "workflow_delta_ms": -1000,
                    "median_due_to_first_token_delta_ms": 100, "median_replay_ttft_delta_ms": 100},
                   {"seed": 2, "mode": "selective_stage", "workflow_delta_ms": 1000,
                    "median_due_to_first_token_delta_ms": 100, "median_replay_ttft_delta_ms": 100}],
               "_manifest": {"run_id": "rq20_selective_pair", "workload": {
                   "research_question_id": "RQ20", "seeds": [1, 2],
                   "arms": ["on_demand", "selective_stage"], "session_count": 8,
                   "turns_per_session": 6, "host_cache_gb": 1.0}}}
    markdown = render_markdown([(Path("docs/reports/work_audit/rq20/summary.json"), summary)])
    assert "Mixed result across 2 paired seeds" in markdown
    assert "Whole workload: seed 1 28.00 → 27.00 s; seed 2 29.00 → 30.00 s" in markdown.replace("&nbsp;", " ")
    assert "WORK_AUDIT_CYCLES_HOST_GB='1'" in markdown


def test_storage_cycles_blocked_run_does_not_claim_validated_win():
    arms = [{"seed": 1, "arm": mode, "session_count": 8, "replay_count": 48,
             "workflow_duration_ms": workflow, "due_to_first_token_median_ms": delay,
             "due_to_first_token_p95_ms": delay, "replay_ttft_median_ms": delay,
             "natural_storage_candidate_waits": 10, "stage_before_due_count": staged,
             "stage_loaded_tokens": 1000, "native_replay_storage_hit_count": 2,
             "native_replay_storage_hit_tokens": 2000, "preparation_errors": []}
            for mode, workflow, delay, staged in (
                ("on_demand", 40000, 500, 0), ("host_stage", 41000, 300, 7))]
    summary = {"schema": "agentic_work_audit.storage_cycles.summary.v1", "status": "blocked",
               "arms": arms, "paired_comparisons": [{"seed": 1, "mode": "host_stage",
                   "workflow_delta_ms": 1000, "median_due_to_first_token_delta_ms": -200,
                   "median_replay_ttft_delta_ms": -200}],
               "failed_arms": [{"arm_id": "seed2_host_stage", "backend_assertion": "cache tree",
                                "client_error": "HTTP 500"}],
               "missing_arm_ids": ["seed2_host_stage", "seed2_on_demand"]}
    path = Path("docs/reports/work_audit/rq17/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Blocked after one completed pair" in page
    assert "one-pair observation is not a validated performance conclusion" in markdown
    assert "seed2_host_stage server failure" in markdown


def test_guarded_storage_cycles_blocked_run_names_preparation_error():
    arms = [{"seed": 1, "arm": arm, "session_count": 8, "replay_count": 48,
             "workflow_duration_ms": 29000, "due_to_first_token_median_ms": 500,
             "due_to_first_token_p95_ms": 700, "replay_ttft_median_ms": 500,
             "natural_storage_candidate_waits": 11, "stage_before_due_count": int(arm == "host_stage") * 2,
             "stage_loaded_tokens": 1000, "native_replay_storage_hit_count": 2,
             "native_replay_storage_hit_tokens": 2000,
             "preparation_skips": ["host_capacity_insufficient"] if arm == "host_stage" else [],
             "preparation_errors": ["native_prefetch_not_admitted"] if arm == "host_stage" else []}
            for arm in ("on_demand", "host_stage")]
    summary = {"schema": "agentic_work_audit.storage_cycles.summary.v1", "status": "blocked",
               "arms": arms, "paired_comparisons": [{"seed": 1, "mode": "host_stage",
                   "workflow_delta_ms": 394, "median_due_to_first_token_delta_ms": -11,
                   "median_replay_ttft_delta_ms": -10}], "failed_arms": [], "missing_arm_ids": [],
               "_manifest": {"run_id": "rq18_guarded_cycles_20261007",
                             "workload": {"seeds": [1], "arms": ["on_demand", "host_stage"],
                                          "session_count": 8, "turns_per_session": 6,
                                          "cuda_graph": True, "overlap_schedule": True}}}
    page = render([(Path("docs/reports/work_audit/rq18_guarded_cycles/summary.json"), summary)])
    markdown = render_markdown([(Path("docs/reports/work_audit/rq18_guarded_cycles/summary.json"), summary)])
    assert "Native prefetch was declined once" in page
    assert "cache-tree assertion" not in page
    assert "WORK_AUDIT_CYCLES_CUDA_GRAPH=&#x27;1&#x27;" in page
    assert "Stage skips" in page
    assert "Stage skip reasons" in markdown
    assert "host_capacity_insufficient (1)" in markdown
    assert "failed partial-prefetch behavior" not in markdown
    assert "will not reproduce the earlier unclassified refusal exactly" in markdown


def test_storage_report_calls_out_peer_cost_when_preparation_overlaps():
    rows = [
        {"seed": 1, "arm": arm, "due_to_first_token_ms": delay,
         "replay_ttft_ms": delay, "peer_count": 2,
         "peer_ttft_median_ms": peer_ttft,
         "peers_overlapping_preparation": overlap,
         "native_replay_storage_hit_tokens": 2048 if arm == "on_demand" else 0,
         "control_storage_hit_tokens": 0 if arm == "on_demand" else 2048,
         "stage_completed_before_due": arm != "on_demand"}
        for arm, delay, peer_ttft, overlap in (
            ("on_demand", 330, 154, 0),
            ("host_stage", 165, 196, 2),
            ("full_prepare", 65, 203, 2),
        )
    ]
    summary = {"schema": "agentic_work_audit.storage_replay.v1", "run_id": "storage-peer",
               "status": "complete", "rows": rows, "paired": [{"seed": 1}],
               "_manifest": {"workload": {"seeds": [1], "peer_count": 2}}}
    assert "This is not a proven win-win" in render_markdown(
        [(Path("runs/storage-peer/summary.json"), summary)]
    )


def test_storage_report_shows_native_read_and_commit_separately():
    rows = [
        {"seed": 1, "arm": arm, "due_to_first_token_ms": delay,
         "replay_ttft_ms": delay, "peer_count": 2, "peer_ttft_median_ms": 100,
         "storage_data_ready_ms": 42 if arm != "on_demand" else None,
         "storage_commit_after_ready_ms": 61 if arm != "on_demand" else None,
         "native_replay_storage_hit_tokens": 2048 if arm == "on_demand" else 0,
         "control_storage_hit_tokens": 0 if arm == "on_demand" else 2048,
         "stage_completed_before_due": arm != "on_demand"}
        for arm, delay in (("on_demand", 330), ("host_stage", 165), ("full_prepare", 65))
    ]
    summary = {"schema": "agentic_work_audit.storage_replay.v1", "run_id": "storage-native",
               "status": "complete", "rows": rows, "paired": [{"seed": 1}],
               "_manifest": {"workload": {"seeds": [1], "peer_count": 2}}}
    markdown = render_markdown([(Path("runs/storage-native/summary.json"), summary)])
    assert "Native L3 → host ready (ms)" in markdown
    assert "Ready → status observed (ms)" in markdown
    assert "Native data readiness took a median 42 ms" in markdown
    assert "status-poll/host-commit interval took 61 ms" in markdown
    assert "42" in markdown and "61" in markdown


def test_tool_cycles_report_keeps_latency_and_load_evidence_separate():
    summary = {
        "schema": "agentic_work_audit.tool_cycles.v1",
        "active_workflow_makespan_ms": 22000,
        "measurements": {"active_replay_count": 1, "active_ttft_median_ms": 90,
                         "active_ttft_p95_ms": 90, "kv_load_back_operations": 0,
                         "active_replays_with_kv_load_back": 0},
        "turns": [{"kind": "active", "session_id": "s1", "turn": 1,
                   "prompt_tokens": 120, "matched_prefix_tokens": 80,
                   "first_token_after_tool_ms": 90, "lookup_to_batch_ms": 2,
                   "completion_after_tool_ms": 400}],
    }
    headline, detail = _result_parts(summary)
    finding = _run_finding(summary)
    assert "first token median 90" in headline
    assert "KV load-backs 0" in headline
    assert "stage" not in headline.lower()
    assert "Stage timing identifies where time was spent" in finding
    assert "scheduler-method boundary" in detail


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
           "load_phases": [{"load_id": "load-1", "loaded_tokens": 512,
                            "control_queue_ms": 2, "scheduler_preparation_ms": 1,
                            "load_back_call_ms": 6, "ready_to_load_call_ms": 2,
                            "cuda_elapsed_ms": 8.5}],
           "load_windows": {"load_attempts": 1, "confirmed_control_windows": 1,
                            "windows_with_other_replay_before_first_token": 1,
                            "windows_with_other_replay_after_first_token": 0}}
    summary = {"schema": "agentic_work_audit.kv_load_attribution.v1", "run_id": "attr",
               "status": "complete", "seed_count": 1, "interpretation_limit": "Not physical copy time.",
               "_manifest": {**_manifest(), "workload": {
                   "research_question_id": "RQ8", "session_count": 12,
                   "tool_waits_per_session": 3, "modes": ["baseline", "check_only", "controller"],
                   "seeds": [1], "hicache_io_backend": "kernel", "load_execution": "worker"}},
               "seeds": [{"seed": 1, "arms": {mode: arm for mode in
                          ("baseline", "check_only", "controller")},
                          "check_cost": stage, "load_association": stage}]}
    page = render([(Path("runs/attr/summary.json"), summary)])
    assert "Busy workload · KV-load attribution" in page
    assert "Checks only" in page and "Checks + loads" in page
    assert "Queue → first cache lookup" in page
    assert "1/1" in page
    assert "Completed early-load phases" in page
    assert "8.5 ms" in page
    assert "KV I/O backend kernel" in page
    assert "load execution worker" in page
    assert "WORK_AUDIT_LOAD_EXECUTION" in page
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


def test_decode_overlap_report_keeps_stage_attribution_and_runner():
    manifest = _manifest(("early", "post_short"))
    manifest["workload"].update({"research_question_id": "RQ10", "session_count": 3,
                                  "short_wait_ms": 900, "long_wait_ms": 10000,
                                  "early_at_ms": 1200, "load_execution": "worker",
                                  "prompt_words_target": 4090, "max_output_tokens": 16})
    cases = []
    for mode, finish, forward, overlap in (("early", 970, 420, 280),
                                           ("post_short", 815, 235, 0)):
        cases.append({"pair": 1, "condition": mode,
                      "audit": {"sessions": {"short": {"first_token_after_tool_ms": 305,
                                                         "completion_after_tool_ms": finish}}},
                      "short_decode": {"decode_batches": [{"duration_ms": forward + 5,
                                                               "model_forward_ms": forward,
                                                               "non_forward_ms": 5}],
                                       "inter_batch_gaps_ms": [200]},
                      "load_overlap": {"short_decode_overlap_ms": overlap}})
    summary = {"schema": "agentic_work_audit.decode_overlap.v1", "run_id": "overlap",
               "status": "validated", "_manifest": manifest,
               "_trace_profile": "kv_decode_overlap", "cases": cases,
               "pairs": [{"pair": 1, "comparable": True, "reasons": [],
                          "early_short_first_token_ms": 305,
                          "post_short_first_token_ms": 305,
                          "early_short_finish_ms": 970, "post_short_finish_ms": 815,
                          "early_long_first_token_ms": 120,
                          "post_short_long_first_token_ms": 120,
                          "early_workflow_ms": 15000,
                          "post_short_workflow_ms": 15000}]}
    path = Path("runs/overlap/summary.json")
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Decode overlap attribution" in page
    assert "Model forward" in page and "420.0 ms" in page
    assert "WORK_AUDIT_STUDY=multisession_overlap" in page
    assert "WORK_AUDIT_STUDY=multisession_overlap" in markdown
    assert "WORK_AUDIT_FORWARD_TRACE=1" in markdown
    assert "815" in markdown and "970" in markdown
    summary["_cuda_kernel_subset"] = {"status": "validated_subset", "pairs": [{
        "pair": 1,
        "early_minus_post_short_ms": {"kernel_duration_sum_ms": 0.3,
                                      "kernel_gap_inside_span_ms": 172.4},
        "early": {"kernel_count": 4872, "kernel_duration_sum_ms": 470.0,
                  "kernel_gap_inside_span_ms": 190.0,
                  "htod_during_kernel_span_ms": 0.0},
        "post_short": {"kernel_count": 4872, "kernel_duration_sum_ms": 469.7,
                       "kernel_gap_inside_span_ms": 17.6,
                       "htod_during_kernel_span_ms": 0.0},
    }]}
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "GPU kernel check (captured pair only)" in page
    assert "Between-kernel gaps" in markdown
    assert "WORK_AUDIT_NSYS_ENABLE=1" in markdown
    assert "partial CUDA capture" in markdown
    summary["_cuda_launch_gaps"] = {
        "early_minus_post_short_ms": {"cpu_before_launch_ms": 167.4},
        "early": {"gap_ms": 190.0, "cpu_before_launch_ms": 178.3,
                  "launch_api_ms": 7.5, "after_launch_api_ms": 4.2,
                  "stream_wait_event_gpu_ms": 1.0, "blocking_sync_api_ms": 0.0},
        "post_short": {"gap_ms": 17.6, "cpu_before_launch_ms": 10.9,
                       "launch_api_ms": 1.5, "after_launch_api_ms": 5.2,
                       "stream_wait_event_gpu_ms": 0.9, "blocking_sync_api_ms": 0.0},
    }
    page = render([(path, summary)])
    markdown = render_markdown([(path, summary)])
    assert "Before CPU launch" in page
    assert "167.4 ms occurred before the CPU began the next CUDA launch" in page
    assert "Before CPU launch (ms)" in markdown
    assert "Recorded stream-wait event activity" in markdown
    assert "CUDA launch gaps" in markdown


def test_overlap_dose_report_distinguishes_physical_copy_from_worker_window():
    manifest = _manifest()
    manifest["workload"] = {"research_question_id": "RQ11", "session_count": 6,
                            "donor_count": 4, "planned_overlap": 2, "seed": 1,
                            "decode_tokens": 96, "prompt_words_target": 4090,
                            "target_wait_ms": 900, "donor_wait_ms": 10000,
                            "forward_trace_enabled": 1, "nsys_enabled": 1}
    donor = {"load_id": "load1", "loaded_tokens": 4096, "cuda_elapsed_ms": 12,
             "worker_window_overlap_ms": 120}
    summary = {"schema": "agentic_work_audit.overlap_dose.v1", "run_id": "dose",
               "status": "worker_window_only", "session_count": 6, "donor_count": 4,
               "planned_overlap": 2, "seed": 1, "_manifest": manifest,
               "realized_worker_window_overlap_count": 2,
               "target": {"first_token_after_tool_ms": 100,
                          "completion_after_tool_ms": 900},
               "active_requests": [{"completion_after_tool_ms": 900},
                                   {"completion_after_tool_ms": 810}],
               "workflow_makespan_ms": 12000, "donors": [donor] * 4,
               "_physical_overlap": {"status": "verified", "physical_overlap_load_count": 1,
                                     "physical_overlap_ms": 8,
                                     "concurrent_kernel_copy_ms": 0.25,
                                     "donors": [{"load_id": "load1",
                                                 "copy_during_target_decode_ms": 8,
                                                 "copy_concurrent_with_target_kernels_ms": 1}]}}
    page = render([(Path("runs/dose/summary.json"), summary)])
    markdown = render_markdown([(Path("runs/dose/summary.json"), summary)])
    assert "1/4 physical copies overlapped" in page
    assert "Physical copy in decode" in page
    assert "WORK_AUDIT_PLANNED_OVERLAP" in page
    assert "WORK_AUDIT_PLANNED_OVERLAP='2'" in markdown
    assert "Physical H-to-D overlaps" in markdown
    assert "Copy concurrent with decode kernels (ms)" in markdown
    assert "[Physical copy overlap](runs/dose/nsys/physical_overlap.json)" in markdown
    manifest["workload"].update({"cuda_graph_requested": True,
                                 "overlap_schedule_requested": False})
    page = render([(Path("runs/dose/summary.json"), summary)])
    assert "CUDA graphs on; overlap scheduling off" in page
    assert "WORK_AUDIT_CUDA_GRAPH=&#x27;1&#x27;" in page
    assert "WORK_AUDIT_OVERLAP_SCHEDULE=&#x27;0&#x27;" in page


def test_profiled_overlap_reports_launch_gap_without_claiming_clean_latency():
    summary = {"schema": "agentic_work_audit.overlap_dose.v1", "run_id": "profiled",
               "status": "worker_window_only", "donor_count": 4, "planned_overlap": 4,
               "_physical_overlap": {"status": "verified", "physical_overlap_load_count": 4},
               "_decode_submission": {"cpu_before_launch_ms": 550,
                                      "kernel_execution_ms": 3192},
               "_dose_control": {"decode_submission": {"cpu_before_launch_ms": 68,
                                                       "kernel_execution_ms": 3191}}}
    finding = _run_finding(summary)
    assert "482 ms" in finding
    assert "+1.0 ms" in finding
    assert "unprofiled speedup" in finding
    assert _markdown_index_outcome(summary)[1] == "Profiled mechanism; no latency claim"


def test_overlap_dose_pairs_only_same_workload_and_seed():
    workload = {"active_prompt_words": 512, "donor_prompt_words": 4090,
                "target_wait_ms": 900, "donor_wait_ms": 40000,
                "decode_tokens": 96, "nsys_enabled": 0}
    base = {"schema": "agentic_work_audit.overlap_dose.v1", "run_id": "base",
            "session_count": 12, "seed": 1, "planned_overlap": 0,
            "_manifest": {"workload": workload},
            "target": {"completion_after_tool_ms": 4000},
            "active_requests": [{}, {"completion_after_tool_ms": 4100}],
            "donors": [{"loaded_tokens": 4096}]}
    high = {**base, "run_id": "high", "planned_overlap": 4,
            "target": {"completion_after_tool_ms": 4800},
            "active_requests": [{}, {"completion_after_tool_ms": 4900}]}
    other_seed = {**high, "run_id": "other", "seed": 2}
    other_bytes = {**high, "run_id": "other_bytes",
                   "donors": [{"loaded_tokens": 2048}]}
    other_pair = {**high, "run_id": "other_pair",
                  "_manifest": {"workload": {**workload, "pair_id": "different"}}}
    rows = [(Path(f"runs/{item['run_id']}/summary.json"), item)
            for item in (base, high, other_seed, other_bytes, other_pair)]
    _pair_rq11_controls(rows)
    assert high["_dose_control"]["target_ms"] == 4000
    assert "800 ms later" in _run_finding(high)
    assert "_dose_control" not in other_seed
    assert "_dose_control" not in other_bytes
    assert "_dose_control" not in other_pair


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
        "significance": "research_pivot",
        "pivot_title": "Early KV preparation",
        "manager_takeaway": "Preparing KV early reduced replay delay in the controlled case.",
        "why_it_matters": "It established that tool-return timing can expose a useful preparation window.",
        "evidence_date_utc": "2026-10-02",
        "question": "Does early KV preparation help?",
        "answer": "Yes in this controlled test.",
        "unknown": "Whether other sessions are delayed.",
        "evidence_run_ids": ["first", "second"],
        "related_run_ids": ["first", "second"],
    }]}), encoding="utf-8")
    markdown_out = tmp_path / "index.md"
    monkeypatch.setattr(sys, "argv", ["report", "--results-dir", str(runs),
                                      "--progress-file", str(progress), "--out", str(out),
                                      "--markdown-out", str(markdown_out)])
    main()
    page = out.read_text(encoding="utf-8")
    assert "Given what the harness knew at the time" in page
    assert "Five audit ledgers" in page
    assert all(name in page for name in (
        "Host backups", "GPU evictions", "Session resumes", "HBM occupancy", "GPU time",
    ))
    assert "not yet graded" in page
    assert "RQ8 tests busy cache pressure, and RQ9 tests off-scheduler loading" in page
    assert "<th>Date</th><th>Time (Central)</th>" in page
    assert page.index("second</small>") < page.index("first</small>")
    assert "9:30:02 AM" in page and "9:30:01 AM" in page
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
    assert "Key Research Pivots" in page
    assert "Early KV preparation" in page
    assert "Main finding" in page and "Why it matters" in page
    assert "Next direction" not in page
    assert page.count("Research pivot") >= 3
    assert "pivot-run" in page
    assert "Does early KV preparation help?" in page
    assert "Yes in this controlled test." in page
    assert 'href="#run-first"' in page and 'href="#run-second"' in page
    assert "Question tested." in page
    assert "<th>Research question</th>" in page
    assert "<th>Main result</th><th>Finding</th><th>Evidence gate</th>" in page
    assert page.count('href="#rq-RQ1"') == 2
    assert 'id=\'rq-RQ1\'' in page
    assert page.count("this was not a policy-speed comparison") == 2
    markdown = markdown_out.read_text(encoding="utf-8")
    assert "# KV Lifecycle Audit" in markdown
    assert "## Key Research Pivots" in markdown
    assert "| Research pivot | Main finding | Why it matters |" in markdown
    assert "Next direction" not in markdown
    assert "<kbd>Research pivot</kbd>" in markdown
    assert "Does early KV preparation help?" in markdown
    assert markdown.index("[Lifecycle validation](#run-second)") < markdown.index(
        "[Lifecycle validation](#run-first)")
    assert markdown.count("<details>") == 2
    assert "| Case | Replay TTFT (ms)" in markdown
    assert "**Setup.**" in markdown and "**Limits**" not in markdown
    assert "```bash" in markdown
    assert "[Run manifest](runs/first/run_manifest.json)" in markdown
    assert "[Harness timeline](runs/second/harness_events.jsonl)" in markdown


def test_markdown_rq9_index_shows_absolute_early_and_late_results():
    manifest = _manifest()
    manifest["workload"]["research_question_id"] = "RQ9"
    summary = {
        "schema": "agentic_work_audit.multisession_comparison.v1",
        "run_id": "worker", "status": "validated", "_manifest": manifest,
        "_started_ns": int(datetime(2026, 10, 5, 18, 28, 34, tzinfo=timezone.utc).timestamp() * 1e9),
        "cases": [
            {"pair": 1, "load_timing": "early", "sessions": {
                "long": {"first_token_after_tool_ms": 89.89},
                "short": {"completion_after_tool_ms": 1006.444}},
             "workflow_makespan_ms": 7900.198},
            {"pair": 1, "load_timing": "late_nonblocking", "sessions": {
                "long": {"first_token_after_tool_ms": 1131.173},
                "short": {"completion_after_tool_ms": 808.527}},
             "workflow_makespan_ms": 8953.478},
        ],
        "pairs": [{"pair": 1, "comparable": True}],
    }
    page = render_markdown([(Path("runs/worker/summary.json"), summary)])
    index = page.split("## Experiment details", 1)[0].replace("&nbsp;", " ")
    assert "| Central date / time |" in index
    assert "| Replay / long session | Other session | Whole workflow | Plain-English finding |" in index
    assert "Oct 5, 2026, 1:28:34 p.m. CDT" in index
    assert "Late loading → early loading (1 pair)" in index
    assert "1,131 → 90 ms (1,041 ms faster)" in index
    assert "809 → 1,006 ms (198 ms later)" in index
    assert "8,953 → 7,900 ms (1,053 ms sooner)" in index
    assert "short session finished later" in index
    assert "1041.3 ms faster" not in index
    assert "1:28:34 p.m. CDT" in page.split("## Experiment details", 1)[1]


def test_markdown_uses_cst_for_winter_and_no_arrow_for_validation_only():
    summary = {"schema": "agentic_work_audit.validation.v1", "run_id": "winter",
               "status": "validated", "_manifest": _manifest(),
               "_started_ns": int(datetime(2026, 1, 5, 18, 0, tzinfo=timezone.utc).timestamp() * 1e9),
               "cases": [{"case_type": "host_backed", "replay_ttft_ms": 85.1}]}
    index = render_markdown([(Path("runs/winter/summary.json"), summary)]).split(
        "## Experiment details", 1)[0].replace("&nbsp;", " ")
    assert "Jan 5, 2026, 12:00:00 p.m. CST" in index
    assert "Observation only; no policy comparison" in index
    assert "Host-backed replay TTFT: 85.1 ms" in index
    row = next(line for line in index.splitlines() if line.startswith("| Jan 5, 2026"))
    assert " → " not in "|".join(row.split("|")[5:9])


def test_markdown_does_not_call_mixed_other_session_trials_a_consistent_win():
    manifest = _manifest()
    manifest["workload"]["research_question_id"] = "RQ7"
    cases = []
    for pair, late_short, controlled_short in ((1, 800, 780), (2, 800, 830)):
        cases.extend((
            {"pair": pair, "load_timing": "late_nonblocking", "sessions": {
                "long": {"first_token_after_tool_ms": 300},
                "short": {"completion_after_tool_ms": late_short}},
             "workflow_makespan_ms": 9000},
            {"pair": pair, "load_timing": "controller_window", "sessions": {
                "long": {"first_token_after_tool_ms": 90},
                "short": {"completion_after_tool_ms": controlled_short}},
             "workflow_makespan_ms": 8000},
        ))
    summary = {"schema": "agentic_work_audit.controller_window.v1", "run_id": "mixed",
               "status": "validated", "_manifest": manifest, "cases": cases,
               "pairs": [{"pair": 1, "comparable": True}, {"pair": 2, "comparable": True}]}
    index = render_markdown([(Path("runs/mixed/summary.json"), summary)]).split(
        "## Experiment details", 1)[0].replace("&nbsp;", " ")
    assert "Late loading → controller-timed loading (per-mode median, 2 pairs)" in index
    assert "short-session effect varied" in index
    assert "workflow sooner" in index


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
    assert "<th>Time (Central)</th>" in page
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
         "answer": "New answer", "hypothesis": "Host launches may be delayed.",
         "unknown": "Next test", "evidence_run_ids": ["missing"]},
        {"id": "RQ1", "short_question": "Older question", "question": "Older question",
         "answer": "Older answer", "unknown": "Old limit", "evidence_run_ids": ["saved"]},
    ]
    page = render([(Path("runs/saved/summary.json"), {"run_id": "saved"})], milestones)
    assert page.index("New question") < page.index("Older question")
    assert "missing (not archived here)" in page
    assert 'href="#run-saved"' in page
    assert "Question tested." in page
    assert 'href="#rq-RQ1"' in page
    assert "Working hypothesis.</strong> Host launches may be delayed." in page
    markdown = render_markdown([(Path("runs/saved/summary.json"), {"run_id": "saved"})], milestones)
    assert "**Working hypothesis.** Host launches may be delayed." in markdown


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
                                 "prompt_words_target": 4090,
                                 "load_execution": "worker", "research_question_id": "RQ9"})
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
    assert "AGENTIC_KV_PREPARE_LOAD_WORKER=1" in page
    assert "WORK_AUDIT_RESEARCH_QUESTION_ID=RQ9" in page


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
