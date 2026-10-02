import json
import sys
from pathlib import Path

from agentic_reports.builders.build_work_audit_report import main, render


def test_report_links_saved_validation_summary(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "sample"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "run_id": "sample",
        "status": "validated",
        "opportunity_ledgers": {"backup_reuse": "unknown: no validated block identity ledger"},
        "cases": [
            {"case_type": "warm_control", "replay_ttft_ms": 10},
            {"case_type": "host_backed", "replay_ttft_ms": 11,
             "host_resident_tokens": 2048, "native_loaded_tokens": 2048,
             "native_layer_copies": 28},
        ],
    }), encoding="utf-8")
    (run / "block_audit.json").write_text(json.dumps({
        "status": "validated", "logical_block_records": 2, "logical_loaded_records": 1,
        "interpretation": "One load, not 28 loads.",
        "cases": [{"case_type": "host_backed", "semantic_load_transitions": 1,
                   "layer_copy_observations": 28,
                   "same_loaded_block_used_by_replay": "not_proven"}],
    }), encoding="utf-8")
    (run / "instrumentation_analysis.json").write_text(json.dumps({
        "cases": [{"case_type": "host_backed", "loaded_slots_matched_by_replay": 42}],
    }), encoding="utf-8")
    (run / "instrumentation_audit.json").write_text(json.dumps({
        "gate": {"profile": "kv_lifecycle_lean"},
    }), encoding="utf-8")
    out = tmp_path / "index.html"
    monkeypatch.setattr(sys, "argv", ["report", "--results-dir", str(run.parent), "--out", str(out)])
    main()
    html = out.read_text(encoding="utf-8")
    assert "sample" in html
    assert "sequentially" in html
    assert "href='runs/sample/summary.json'" in html
    assert "href='runs/sample/run_manifest.json'" in html
    assert "href='runs/sample/block_audit.json'" in html
    assert "One load, not 28 loads." in html
    assert "not proven" in html
    assert "exact loaded-block consumption by replay is not proven" in html
    assert "42" in html
    assert "<th>Trace profile</th>" in html
    assert "kv_lifecycle_lean" in html
    assert "href='runs/sample/instrumentation_analysis.json'" in html
    assert "no validated block identity ledger" not in html


def test_report_separates_both_replays_and_load_periods():
    html = render([(Path("runs/two/summary.json"), {
        "run_id": "two", "status": "validated", "require_second_replay": True,
        "cases": [
            {"case_type": "warm_control", "replay_ttft_ms": 220, "second_replay_ttft_ms": 223},
            {"case_type": "host_backed", "replay_ttft_ms": 574, "second_replay_ttft_ms": 218},
        ],
        "_manifest": {"workload": {"cases": ["host", "warm"]}},
        "_block_audit": {"status": "validated", "cases": [{
            "case_type": "host_backed", "planned_pre_replay_loads": 1,
            "planned_pre_replay_loaded_tokens": 4096, "replay_time_loads": 1,
            "replay_time_loaded_tokens": 41,
        }]},
        "_has_compressed_trace": True,
    })])
    assert "host, warm" in html
    assert "574" in html and "218" in html
    assert "4096" in html and "41" in html
    assert "backend_trace.jsonl.gz" in html


def test_report_keeps_timing_pairs_separate_from_lifecycle_validation():
    html = render([(Path("runs/timing/summary.json"), {
        "schema": "agentic_work_audit.timing.v1", "run_id": "timing", "status": "validated",
        "cases": [
            {"pair": 1, "condition": "early", "loaded_tokens": 2048,
             "completion_observed_before_due": True, "first_token_after_due_ms": 45,
             "replay_ttft_ms": 25, "loaded_slots_matched_by_replay": 1024,
             "loaded_slots_match_status": "supported"},
            {"pair": 1, "condition": "late", "loaded_tokens": 2048,
             "completion_observed_before_due": False, "first_token_after_due_ms": 125,
             "replay_ttft_ms": 95, "loaded_slots_matched_by_replay": 0,
             "loaded_slots_match_status": "unknown"},
        ],
        "pairs": [{"pair": 1, "late_minus_early_first_token_after_due_ms": 80,
                   "late_minus_early_task_latency_ms": 80}],
        "_trace_profile": "kv_lifecycle_lean",
    })])
    assert "Early vs. Late KV Preparation" in html
    assert "Late minus early (ms)" in html
    assert "kv_lifecycle_lean" in html
    assert "Lifecycle Validation Runs" in html
    assert "unknown: 0" in html
