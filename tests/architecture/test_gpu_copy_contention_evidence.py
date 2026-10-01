import gzip
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/reports/hardware/gpu_copy_contention_20261001_154937"


def test_published_timing_matches_trial_rows():
    trials = json.loads((EVIDENCE / "paired_results.json").read_text())["trials"]
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    for mode in ("idle", "copy"):
        rows = [row for row in trials if row["mode"] == mode]
        assert len(rows) == summary["conditions"][mode]["trials"]
        after_first = statistics.median(
            row["decode"]["total_latency_ms"] - row["decode"]["ttft_ms"]
            for row in rows
        )
        window_gap = statistics.median(row["window_mean_chunk_gap_ms"] for row in rows)
        assert abs(after_first - summary["conditions"][mode]["median_decode_after_first_token_ms"]) < 0.001
        assert abs(window_gap - summary["conditions"][mode]["median_window_chunk_gap_ms"]) < 0.001


def test_published_overlap_has_gpu_event_witness():
    with gzip.open(EVIDENCE / "witness.json.gz", "rt", encoding="utf-8") as source:
        witness = json.load(source)
    overlap = json.loads((EVIDENCE / "overlap.json").read_text())
    assert witness["trace_base_time_ns"] == overlap["trace_base_time_ns"]
    assert len(witness["copy_gpu_intervals_us"]) == overlap["copy_events"]
    assert len(witness["backend_kernel_intervals_us"]) == overlap["backend_kernels_in_copy_window"]
