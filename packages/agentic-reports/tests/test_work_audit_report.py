import json
import sys

from agentic_reports.builders.build_work_audit_report import main


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
    assert "no validated block identity ledger" not in html
