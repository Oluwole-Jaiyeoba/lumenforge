from __future__ import annotations

import json

from agentic_reports.builders.build_device_memory_attribution_report import main


def _run(condition: str) -> dict:
    collision = condition == "host_reload_collision"
    return {
        "schema_version": "device_memory_attribution.v1",
        "run_id": f"run_{condition}", "condition": condition, "sample_set_id": "shared",
        "model": "test-model", "trials": [{
            "sample_id": "trial_000", "valid": True, "native_overlap": collision,
            "cuda_load_duration_ms": 12.5 if collision else None,
            "loaded_tokens": 1024 if collision else None,
            "decode": {"ttft_ms": 8.0, "total_latency_ms": 110.0 if collision else 100.0},
        }],
    }


def test_builds_native_reload_attribution_report(tmp_path, monkeypatch) -> None:
    paths = {}
    for condition in ("target_only", "device_resident_control", "host_reload_collision"):
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(_run(condition)), encoding="utf-8")
        paths[condition] = path
    out, summary = tmp_path / "report.html", tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [
        "report", "--target-only", str(paths["target_only"]), "--device-resident", str(paths["device_resident_control"]),
        "--host-reload", str(paths["host_reload_collision"]), "--out", str(out), "--summary-out", str(summary),
    ])
    main()
    result = json.loads(summary.read_text(encoding="utf-8"))
    assert result["paired_decode_change_vs_target_only"]["host_reload_collision"]["median_delta_ms"] == 10.0
    assert "Native KV Reload Attribution" in out.read_text(encoding="utf-8")
