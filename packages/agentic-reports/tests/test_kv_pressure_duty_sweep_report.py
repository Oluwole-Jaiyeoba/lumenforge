from __future__ import annotations

import json

from agentic_reports.builders.build_kv_pressure_duty_sweep_report import main


def _run(condition: str, load_count: int) -> dict:
    return {
        "condition": condition,
        "load_count": load_count,
        "sample_set_id": "shared",
        "model": "test-model",
        "trials": [
            {
                "sample_id": "trial_000",
                "decode": {"ttft_ms": 9.0, "total_latency_ms": 100.0 + load_count},
                "pressure_proof": {
                    "valid": True,
                    "completed_loads": load_count,
                    "recycle_proofs": max(0, load_count - 1),
                    "total_loaded_tokens": load_count * 100,
                    "total_cuda_load_ms": load_count * 2.0,
                    "cuda_load_share_of_decode_pct": load_count * 2.0,
                    "pressure_envelope_ms": load_count * 3.0,
                    "pressure_envelope_share_of_decode_pct": load_count * 3.0,
                },
            }
        ],
    }


def test_builds_four_condition_pressure_report(tmp_path, monkeypatch) -> None:
    paths = {}
    for condition, loads in (("decode_control", 0), ("pressure_low", 2), ("pressure_medium", 4), ("pressure_high", 8)):
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(_run(condition, loads)), encoding="utf-8")
        paths[condition] = path
    out = tmp_path / "report.html"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report", "--control", str(paths["decode_control"]), "--low", str(paths["pressure_low"]),
            "--medium", str(paths["pressure_medium"]), "--high", str(paths["pressure_high"]),
            "--out", str(out), "--summary-out", str(summary),
        ],
    )
    main()
    result = json.loads(summary.read_text(encoding="utf-8"))
    high = next(row for row in result["conditions"] if row["condition"] == "pressure_high")
    assert high["completed_loads"] == 8.0
    assert result["paired_decode_changes"]["pressure_high"]["median_decode_change_ms"] == 8.0
    assert "KV Pressure Duty Sweep" in out.read_text(encoding="utf-8")
