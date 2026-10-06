"""Verify RQ11 donor H-to-D copies against a target replay's CUDA decode kernels."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


CALL_ID = re.compile(r"\bcall_id=([^\s]+)")
LOAD_ID = re.compile(r"\bload_id=([^\s]+)")


def _merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _overlap_ns(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    total = 0
    i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _target_forwards(trace: list[dict[str, Any]], request_id: str,
                     first_ns: int, finish_ns: int) -> set[str]:
    batch_starts = {}
    forward_starts = {}
    target_batches = []
    forwards = []
    for row in trace:
        event, call_id = row.get("event"), str(row.get("call_id"))
        if event == "scheduler.run_batch.start":
            batch_starts[call_id] = row
        elif event == "scheduler.run_batch.end":
            start = batch_starts.pop(call_id, None)
            if start and first_ns <= row["ts_ns"] <= finish_ns:
                batch = (start.get("kv_context") or {}).get("batch") or {}
                requests = batch.get("requests") or (start.get("kv_context") or {}).get("requests") or []
                if any(request_id in (request.get("agent_request_id"), request.get("request_id"))
                       for request in requests):
                    target_batches.append((start["ts_ns"], row["ts_ns"]))
        elif event == "worker.forward_batch_generation.start":
            forward_starts[call_id] = row
        elif event == "worker.forward_batch_generation.end":
            start = forward_starts.pop(call_id, None)
            if start:
                forwards.append((start["ts_ns"], row["ts_ns"], call_id))
    matched = {call_id for a, b, call_id in forwards
               if any(left <= a and b <= right for left, right in target_batches)}
    if len(matched) < 2:
        raise ValueError(f"Only {len(matched)} target decode forwards linked to request {request_id}")
    return matched


def analyze(db: sqlite3.Connection, summary: dict[str, Any],
            trace: list[dict[str, Any]]) -> dict[str, Any]:
    required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_RUNTIME",
                "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY"}
    available = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if missing := required - available:
        raise ValueError("Nsight captured no complete CUDA activity for physical-overlap proof; "
                         f"missing tables: {', '.join(sorted(missing))}. "
                         "NVTX worker windows alone cannot establish physical copy overlap.")
    target = summary["target"]
    call_ids = _target_forwards(trace, target["request_id"], target["chunk_times_ns"][0],
                                target["request_end_ns"])
    forward_ranges = {}
    load_ranges = {}
    for start, end, tid, label in db.execute(
        "SELECT start,end,globalTid,text FROM NVTX_EVENTS WHERE end IS NOT NULL "
        "AND (text LIKE 'agentic_kv:worker.forward_batch_generation:% call_id=%' "
        "OR text LIKE 'agentic_kv:async_load load_id=%')"
    ):
        call = CALL_ID.search(label)
        load = LOAD_ID.search(label)
        if call and call.group(1) in call_ids:
            forward_ranges[call.group(1)] = (start, end, tid)
        if load:
            if load.group(1) in load_ranges:
                raise ValueError(f"Duplicate load NVTX range: {load.group(1)}")
            load_ranges[load.group(1)] = (start, end, tid)
    if call_ids - forward_ranges.keys():
        raise ValueError(f"Nsight lost {len(call_ids - forward_ranges.keys())} target forwards")
    kernels = []
    for start, end, tid in forward_ranges.values():
        kernels.extend(db.execute(
            "SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId "
            "AND (k.globalPid >> 24)=(r.globalTid >> 24) "
            "WHERE r.globalTid=? AND r.start BETWEEN ? AND ?", (tid, start, end)
        ).fetchall())
    if len(kernels) < 100:
        raise ValueError(f"Nsight lost target CUDA kernels: only {len(kernels)} linked")
    active = _merged([(int(a), int(b)) for a, b in kernels])
    envelope = [(active[0][0], active[-1][1])]
    donors = []
    for donor in summary["donors"]:
        load_id = donor["load_id"]
        if load_id not in load_ranges:
            raise ValueError(f"Nsight lost donor load NVTX range: {load_id}")
        start, end, tid = load_ranges[load_id]
        copies = db.execute(
            "SELECT c.start,c.end,c.bytes FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_MEMCPY c ON c.correlationId=r.correlationId "
            "AND (c.globalPid >> 24)=(r.globalTid >> 24) "
            "WHERE r.globalTid=? AND r.start BETWEEN ? AND ? AND c.copyKind=1",
            (tid, start, end)
        ).fetchall()
        if not copies:
            raise ValueError(f"Nsight lost physical H-to-D copies for load {load_id}")
        intervals = _merged([(int(a), int(b)) for a, b, _ in copies])
        donors.append({
            "load_id": load_id, "copy_count": len(copies),
            "copy_bytes": sum(int(size) for _, _, size in copies),
            "copy_active_ms": round(sum(b-a for a, b in intervals) / 1e6, 3),
            "copy_during_target_decode_ms": round(_overlap_ns(intervals, envelope) / 1e6, 3),
            "copy_concurrent_with_target_kernels_ms": round(_overlap_ns(intervals, active) / 1e6, 3),
        })
    return {
        "schema": "agentic_work_audit.physical_overlap.v1", "status": "verified",
        "run_id": summary["run_id"], "target_forward_count": len(call_ids),
        "target_kernel_count": len(kernels),
        "planned_overlap": summary["planned_overlap"],
        "physical_overlap_load_count": sum(row["copy_during_target_decode_ms"] > 0 for row in donors),
        "physical_overlap_ms": round(sum(row["copy_during_target_decode_ms"] for row in donors), 3),
        "concurrent_kernel_copy_ms": round(sum(row["copy_concurrent_with_target_kernels_ms"]
                                                for row in donors), 3),
        "donors": donors,
        "limitations": ["Decode envelope includes gaps between the target's CUDA kernels.",
                        "Summed overlap across loads can double-count simultaneous copies.",
                        "Nsight timing is mechanism evidence, not an unprofiled slowdown estimate."],
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
    print(json.dumps({key: result[key] for key in ("status", "planned_overlap",
                      "physical_overlap_load_count", "physical_overlap_ms",
                      "concurrent_kernel_copy_ms")}, indent=2))


if __name__ == "__main__":
    main()
