from __future__ import annotations

import json

from agentic_reports.builders.build_sustained_decode_kv_overlap_report import main


def _run(condition: str, *, valid: bool = True) -> dict:
    return {
        "schema_version": "sustained_decode_kv_overlap.v1",
        "run_id": f"run_{condition}",
        "condition": condition,
        "sample_set_id": "shared_samples",
        "hardware_profile": "test_gpu",
        "backend_version": "0.5.10.post1",
        "model": "test-model",
        "workload_id": "sustained_decode_kv_overlap_v1",
        "seed": 7,
        "trials": [
            {
                "sample_id": "trial_000",
                "valid_overlap": valid,
                "overlap_duration_ms": 12.0 if condition == "direct_overlap_reload" else None,
                "decode": {
                    "ttft_ms": 8.0,
                    "total_latency_ms": 60.0,
                    "chunk_times_ns": [0, 10_000_000, 30_000_000, 42_000_000],
                },
                "regions": {
                    "before": {"interval_count": 1, "median_itl_ms": 10.0, "p95_itl_ms": 10.0},
                    "during": {"interval_count": 1, "median_itl_ms": 20.0, "p95_itl_ms": 20.0},
                    "after": {"interval_count": 1, "median_itl_ms": 12.0, "p95_itl_ms": 12.0},
                },
                "load_result": {"loaded_tokens": 8192} if condition != "decode_control" else None,
            }
        ],
    }


def test_builds_overlap_report(tmp_path, monkeypatch) -> None:
    paths = {}
    for condition in ("decode_control", "non_overlap_reload", "direct_overlap_reload"):
        path = tmp_path / f"{condition}.json"
        path.write_text(json.dumps(_run(condition)), encoding="utf-8")
        paths[condition] = path
    out = tmp_path / "report.html"
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "report",
            "--control", str(paths["decode_control"]),
            "--non-overlap", str(paths["non_overlap_reload"]),
            "--direct-overlap", str(paths["direct_overlap_reload"]),
            "--out", str(out),
            "--summary-out", str(summary),
        ],
    )

    main()

    result = json.loads(summary.read_text(encoding="utf-8"))
    direct = next(row for row in result["conditions"] if row["condition"] == "direct_overlap_reload")
    assert direct["valid_overlap_trials"] == 1
    assert direct["regions"]["during"]["median_itl_ms"] == 20.0
    assert "Sustained Decode Versus KV Reload" in out.read_text(encoding="utf-8")
