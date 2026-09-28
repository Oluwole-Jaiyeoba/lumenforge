from __future__ import annotations

import json

from agentic_reports.builders.build_hardware_kv_movement_report import main


def _probe(condition: str, value: float) -> dict:
    return {
        "run_id": f"run_{condition}",
        "condition": condition,
        "hardware_profile": "test_gpu",
        "backend_version": "0.5.10.post1",
        "model": "test-model",
        "workload_id": "test_workload",
        "seed": 7,
        "instrumentation_profile": "lightweight_backend_trace",
        "samples": [{"sample_id": "trial_000", "metrics_ms": {"replay_ttft_ms": value, "replay_lateness_ms": value - 5}}],
        "probe_metadata": {
            "trial_details": [
                {
                    "interference_load": (
                        {"loaded_tokens": 20, "status": "queued"} if condition == "interference" else None
                    )
                }
            ]
        },
    }


def test_builds_paired_report(tmp_path, monkeypatch) -> None:
    control = tmp_path / "control.json"
    interference = tmp_path / "interference.json"
    control.write_text(json.dumps(_probe("control", 10.0)), encoding="utf-8")
    interference.write_text(json.dumps(_probe("interference", 14.0)), encoding="utf-8")
    control_trace = tmp_path / "control.jsonl"
    interference_trace = tmp_path / "interference.jsonl"
    control_trace.write_text('{"event":"trace.install.summary"}\n', encoding="utf-8")
    interference_trace.write_text(
        '{"event":"trace.install.summary"}\n{"event":"hostpool.load_to_device_per_layer.end"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.html"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--control", str(control),
            "--interference", str(interference),
            "--control-backend-trace", str(control_trace),
            "--interference-backend-trace", str(interference_trace),
            "--out", str(out),
            "--summary-out", str(summary),
        ],
    )

    main()

    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["comparison"]["median_paired_change_ms"] == 4.0
    assert result["lateness_comparison"]["median_paired_change_ms"] == 4.0
    assert result["software_visible_h2d"]["interference"] == 1
    assert result["interference_loads"]["verified_loads"] == 1
    assert "KV Movement Interference" in out.read_text(encoding="utf-8")
