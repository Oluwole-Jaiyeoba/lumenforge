from __future__ import annotations

import json

from agentic_hardware_probes import summarize_timeline


def test_classifies_copy_compute_overlap(tmp_path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "gpu_memcpy", "name": "Memcpy HtoD (Pinned -> Device)", "ts": 10, "dur": 20},
                    {"cat": "kernel", "name": "decode", "ts": 25, "dur": 20},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline(trace)

    assert summary["actual_gpu_collision"] is True
    assert summary["h2d_kernel_overlap_us"] == 5.0


def test_classifies_serialized_copy_and_compute(tmp_path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {"cat": "gpu_memcpy", "name": "Memcpy HtoD (Pinned -> Device)", "ts": 10, "dur": 10},
                    {"cat": "kernel", "name": "decode", "ts": 25, "dur": 10},
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_timeline(trace)

    assert summary["actual_gpu_collision"] is False
    assert summary["classification"] == "serialized_no_gpu_overlap"
