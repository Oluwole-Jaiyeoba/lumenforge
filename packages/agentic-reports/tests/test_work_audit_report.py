import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from agentic_reports.builders.build_work_audit_report import main, render


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
            "run_id": name, "status": "validated",
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
        "evidence_date_utc": "2026-10-02",
        "question": "Does early KV preparation help?",
        "answer": "Yes in this controlled test.",
        "unknown": "Whether other sessions are delayed.",
        "evidence_run_ids": ["first", "second"],
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
    assert "competing-session cost unmeasured" in page
    assert "<th>Date</th><th>Time (UTC)</th>" in page
    assert page.index("second</small>") < page.index("first</small>")
    assert "14:30:02" in page and "14:30:01" in page
    assert page.count("<summary>View</summary>") == 2
    assert 'href="runs/first/run_manifest.json"' in page
    assert 'href="runs/second/block_audit.json"' in page
    assert "Reconstructed command" in page
    assert "frontend priority: none" in page
    assert "Research progress" in page
    assert "Does early KV preparation help?" in page
    assert "Yes in this controlled test." in page
    assert 'href="#run-first"' in page and 'href="#run-second"' in page
    assert "Question tested." in page


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
    assert "81.0 ms / 83.0 ms" in page
    assert "1.0 ms / 173.0 ms" in page
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


def test_progress_keeps_milestone_order_and_marks_unarchived_evidence():
    milestones = [
        {"question": "New question", "answer": "New answer", "unknown": "Next test",
         "evidence_run_ids": ["missing"]},
        {"question": "Older question", "answer": "Older answer", "unknown": "Old limit",
         "evidence_run_ids": ["saved"]},
    ]
    page = render([(Path("runs/saved/summary.json"), {"run_id": "saved"})], milestones)
    assert page.index("New question") < page.index("Older question")
    assert "missing (not archived here)" in page
    assert 'href="#run-saved"' in page
    assert "Question tested." in page


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
    assert "Accepted before replay" in page
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
    assert "90.0 ms / 230.0 ms" in page
    assert "explicit control command" in page
    assert "Load requested after tool return" in page
    assert "unknown: no same-capacity counterfactual" in page
    assert "WORK_AUDIT_STUDY=multisession" in page
