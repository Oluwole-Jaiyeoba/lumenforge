from __future__ import annotations

import json

from agentic_reports.builders.build_sustained_kv_pressure_report import main


def _run(condition: str) -> dict:
    load_count = {"decode_control": 0, "single_reload": 1, "sustained_reload": 3}[condition]
    loads = [
        {
            "donor_index": index,
            "start_ns": 20_000_000 + index * 10_000_000,
            "finish_ns": 25_000_000 + index * 10_000_000,
            "cuda_duration_ms": 4.0,
            "loaded_tokens": 4096,
        }
        for index in range(load_count)
    ]
    return {
        "schema_version": "sustained_kv_pressure.v1",
        "run_id": f"run_{condition}",
        "condition": condition,
        "sample_set_id": "shared_samples",
        "hardware_profile": "test_gpu",
        "backend_version": "0.5.10.post1",
        "model": "test-model",
        "workload_id": "sustained_kv_pressure_v1",
        "seed": 7,
        "trials": [
            {
                "sample_id": "trial_000",
                "decode": {
                    "request_start_ns": 0,
                    "request_end_ns": 100_000_000,
                    "ttft_ms": 8.0,
                    "total_latency_ms": 100.0 + load_count,
                    "chunk_times_ns": [0, 10_000_000, 30_000_000, 50_000_000, 90_000_000],
                },
                "loads": loads,
                "pressure_proof": {
                    "valid": True,
                    "completed_loads": load_count,
                    "total_loaded_tokens": 4096 * load_count,
                    "total_cuda_load_ms": 4.0 * load_count,
                    "pressure_envelope_ms": 5.0 + max(0, load_count - 1) * 10.0 if load_count else 0.0,
                    "pressure_share_of_decode_pct": float(load_count),
                    "maximum_inter_load_gap_ms": 5.0 if load_count > 1 else None,
                    "observed_load_duty_cycle_pct": 80.0 if load_count else None,
                },
                "regions": {
                    "before": {"interval_count": 1, "median_interval_ms": 10.0, "p95_interval_ms": 10.0, "max_interval_ms": 10.0, "visible_updates_per_second": 100.0},
                    "during": {"interval_count": load_count, "median_interval_ms": 20.0, "p95_interval_ms": 20.0, "max_interval_ms": 20.0, "visible_updates_per_second": 50.0},
                    "between": {"interval_count": 0, "median_interval_ms": None, "p95_interval_ms": None, "max_interval_ms": None, "visible_updates_per_second": None},
                    "after": {"interval_count": 1, "median_interval_ms": 40.0, "p95_interval_ms": 40.0, "max_interval_ms": 40.0, "visible_updates_per_second": 25.0},
                },
            }
        ],
    }


def test_builds_sustained_pressure_report(tmp_path, monkeypatch) -> None:
    paths = {}
    for condition in ("decode_control", "single_reload", "sustained_reload"):
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(_run(condition)), encoding="utf-8")
        paths[condition] = path
    output = tmp_path / "report.html"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--control", str(paths["decode_control"]),
            "--single", str(paths["single_reload"]),
            "--sustained", str(paths["sustained_reload"]),
            "--out", str(output),
            "--summary-out", str(summary),
        ],
    )

    main()

    result = json.loads(summary.read_text(encoding="utf-8"))
    sustained = next(row for row in result["conditions"] if row["condition"] == "sustained_reload")
    assert sustained["median_completed_loads"] == 3.0
    assert result["paired_decode_deltas"]["sustained_reload"]["median_decode_change_ms"] == 3.0
    assert "Sustained KV Pressure During Decode" in output.read_text(encoding="utf-8")
