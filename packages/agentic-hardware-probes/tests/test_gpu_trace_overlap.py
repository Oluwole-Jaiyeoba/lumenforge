import gzip
import json
import pytest

from agentic_hardware_probes.gpu_trace_overlap import analyze


def test_analyze_reports_only_physical_gpu_overlap(tmp_path):
    backend = tmp_path / "backend.json"
    worker = tmp_path / "worker.json"
    backend.write_text(json.dumps({"baseTimeNanoseconds": 1234, "traceEvents": [
        {"ph": "X", "cat": "kernel", "ts": 0, "dur": 5},
        {"ph": "X", "cat": "kernel", "ts": 12, "dur": 6},
        {"ph": "X", "cat": "kernel", "ts": 30, "dur": 5},
    ]}, indent=2))
    worker.write_text(json.dumps({"baseTimeNanoseconds": 1234, "traceEvents": [
        {"ph": "X", "cat": "gpu_memcpy", "ts": 10, "dur": 4},
        {"ph": "X", "cat": "gpu_memcpy", "ts": 16, "dur": 4},
    ]}, indent=2))

    witness = tmp_path / "witness.json.gz"
    result = analyze(backend, worker, witness)

    assert result["copy_events"] == 2
    assert result["backend_kernel_events"] == 3
    assert result["backend_kernels_in_copy_window"] == 1
    assert result["backend_kernels_overlapping_copies"] == 1
    assert result["summed_kernel_copy_overlap_ms"] == 0.004
    assert result["physical_overlap_observed"] is True
    assert result["trace_base_time_ns"] == 1234
    with gzip.open(witness, "rt", encoding="utf-8") as source:
        assert len(json.load(source)["backend_kernel_intervals_us"]) == 1


def test_analyze_rejects_different_trace_clocks(tmp_path):
    backend = tmp_path / "backend.json"
    worker = tmp_path / "worker.json"
    backend.write_text('{"baseTimeNanoseconds": 1}')
    worker.write_text('{"baseTimeNanoseconds": 2}')

    with pytest.raises(ValueError, match="clock base"):
        analyze(backend, worker)
