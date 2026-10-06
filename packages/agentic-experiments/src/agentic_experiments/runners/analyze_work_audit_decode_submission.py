"""Attribute a profiled RQ11/RQ12 target decode to GPU work and launch gaps."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from .analyze_work_audit_cuda_launch_gaps import _call_gaps, _forward_ranges
from .analyze_work_audit_physical_overlap import _target_forwards


def analyze(db: sqlite3.Connection, summary: dict[str, Any],
            trace: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION", "StringIds"}
    available = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if missing := required - available:
        raise ValueError(f"Incomplete CUDA launch trace: {sorted(missing)}")

    target = summary["target"]
    call_ids = _target_forwards(trace, target["request_id"], target["chunk_times_ns"][0],
                                target["request_end_ns"])
    ranges = _forward_ranges(db, call_ids)
    calls = []
    for call_id, (start, end, tid) in sorted(ranges.items(), key=lambda item: item[1][0]):
        launch = _call_gaps(db, start, end, tid)
        kernels = db.execute(
            "SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId "
            "AND (k.globalPid >> 24)=(r.globalTid >> 24) "
            "WHERE r.globalTid=? AND r.start BETWEEN ? AND ?",
            (tid, start, end),
        ).fetchall()
        if len(kernels) != launch["kernel_count"]:
            raise ValueError(f"Kernel linkage changed for forward {call_id}")
        calls.append({"call_id": call_id, "start_ns": start, "end_ns": end,
                      "forward_wall_ms": round((end - start) / 1e6, 3),
                      "kernel_execution_ms": round(sum(b - a for a, b in kernels) / 1e6, 3),
                      **launch})

    between_forward_ms = round(sum(max(0, right["start_ns"] - left["end_ns"])
                                   for left, right in zip(calls, calls[1:])) / 1e6, 3)
    total_keys = ("forward_wall_ms", "kernel_execution_ms", "gap_ms",
                  "cpu_before_launch_ms", "launch_api_ms", "after_launch_api_ms",
                  "stream_wait_event_gpu_ms", "blocking_sync_api_ms")
    totals = {key: round(sum(call[key] for call in calls), 3) for key in total_keys}
    return {
        "schema": "agentic_work_audit.decode_submission.v1",
        "status": "verified",
        "run_id": summary["run_id"],
        "target_request_id": target["request_id"],
        "forward_count": len(calls),
        "kernel_count": sum(call["kernel_count"] for call in calls),
        "between_forward_gap_ms": between_forward_ms,
        **totals,
        "forward_calls": calls,
        "limitations": [
            "Nsight timing is mechanism evidence, not an unprofiled latency estimate.",
            "CPU-before-launch is a timeline category, not proof of why the host delayed.",
            "Between-forward gaps can include scheduler and other request work.",
            "Summed kernel execution can exceed wall time when kernels run concurrently.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with args.trace.open(encoding="utf-8") as handle:
        trace = [json.loads(line) for line in handle if line.strip()]
    with sqlite3.connect(args.sqlite) as db:
        result = analyze(db, summary, trace)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("status", "forward_count", "kernel_count", "kernel_execution_ms",
                       "cpu_before_launch_ms", "between_forward_gap_ms")}, indent=2))


if __name__ == "__main__":
    main()
