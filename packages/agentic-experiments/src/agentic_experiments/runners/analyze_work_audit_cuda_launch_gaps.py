"""Locate CPU submission and GPU start delays inside captured decode forwards."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


CALL_ID = re.compile(r"\bcall_id=([^\s]+)")
NS_PER_MS = 1_000_000


def _ms(ns: int) -> float:
    return round(ns / NS_PER_MS, 3)


def _forward_ranges(db: sqlite3.Connection, wanted: set[str]) -> dict[str, tuple[int, int, int]]:
    rows = db.execute(
        "SELECT start, end, globalTid, text FROM NVTX_EVENTS "
        "WHERE text LIKE 'agentic_kv:worker.forward_batch_generation:% call_id=%' "
        "AND end IS NOT NULL"
    )
    result = {}
    for start, end, tid, label in rows:
        match = CALL_ID.search(label)
        if match and match.group(1) in wanted:
            if match.group(1) in result:
                raise ValueError(f"Duplicate model-forward NVTX range: {match.group(1)}")
            result[match.group(1)] = (start, end, tid)
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f"Missing model-forward NVTX ranges: {sorted(missing)[:4]}")
    return result


def _call_gaps(db: sqlite3.Connection, start: int, end: int, tid: int) -> dict[str, Any]:
    rows = db.execute(
        "SELECT k.start, k.end, r.start, r.end, s.value "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId "
        "AND (k.globalPid >> 24) = (r.globalTid >> 24) "
        "JOIN StringIds s ON s.id = r.nameId "
        "WHERE r.globalTid = ? AND r.start >= ? AND r.start <= ? "
        "ORDER BY k.start, k.end", (tid, start, end)
    ).fetchall()
    if not rows:
        raise ValueError("No CUDA kernels linked to a model-forward call")
    if any("Launch" not in row[4] for row in rows):
        raise ValueError("A linked kernel has no CUDA launch API record")
    unique = {(row[0], row[1], row[2], row[3]) for row in rows}
    if len(unique) != len(rows):
        raise ValueError("Ambiguous duplicate CUDA launch-to-kernel linkage")

    totals = {key: 0 for key in (
        "gap_ns", "cpu_before_launch_ns", "launch_api_ns", "after_launch_api_ns"
    )}
    largest: list[dict[str, Any]] = []
    covered_end = rows[0][1]
    for kernel_start, kernel_end, launch_start, launch_end, name in rows[1:]:
        if kernel_start > covered_end:
            if not (launch_start <= kernel_start and launch_start <= launch_end):
                raise ValueError("CUDA launch API began after its kernel started")
            gap_start = covered_end
            gap_end = kernel_start
            parts = {
                "gap_ns": gap_end - gap_start,
                "cpu_before_launch_ns": max(0, launch_start - gap_start),
                "launch_api_ns": max(0, min(launch_end, gap_end) - max(launch_start, gap_start)),
                "after_launch_api_ns": max(0, gap_end - max(launch_end, gap_start)),
            }
            if sum(parts[key] for key in parts if key != "gap_ns") != parts["gap_ns"]:
                raise ValueError("CUDA gap decomposition did not account for the full gap")
            for key, value in parts.items():
                totals[key] += value
            largest.append({
                "gap_ms": _ms(parts["gap_ns"]),
                "cpu_before_launch_ms": _ms(parts["cpu_before_launch_ns"]),
                "launch_api_ms": _ms(parts["launch_api_ns"]),
                "after_launch_api_ms": _ms(parts["after_launch_api_ns"]),
                "next_launch_api": name,
            })
        covered_end = max(covered_end, kernel_end)

    sync_rows = db.execute(
        "SELECT s.value, COUNT(*), COALESCE(SUM(r.end-r.start),0) "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId "
        "WHERE r.globalTid = ? AND r.start >= ? AND r.start <= ? "
        "AND (s.value LIKE '%Synchronize%' OR s.value LIKE '%WaitEvent%' "
        "OR s.value LIKE '%Query%') GROUP BY s.value", (tid, start, end)
    ).fetchall()
    stream_wait_ns = db.execute(
        "SELECT COALESCE(SUM(x.end-x.start),0) "
        "FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION x "
        "JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = x.correlationId "
        "AND (x.globalPid >> 24) = (r.globalTid >> 24) "
        "WHERE r.globalTid = ? AND r.start >= ? AND r.start <= ? "
        "AND x.syncType = 2", (tid, start, end)
    ).fetchone()[0]
    return {
        "kernel_count": len(rows),
        "gap_ms": _ms(totals["gap_ns"]),
        "cpu_before_launch_ms": _ms(totals["cpu_before_launch_ns"]),
        "launch_api_ms": _ms(totals["launch_api_ns"]),
        "after_launch_api_ms": _ms(totals["after_launch_api_ns"]),
        "stream_wait_event_gpu_ms": _ms(stream_wait_ns),
        "blocking_sync_api_ms": _ms(sum(duration for name, _, duration in sync_rows
                                         if "Synchronize" in name)),
        "sync_api": [{"name": name, "count": count, "duration_ms": _ms(duration)}
                     for name, count, duration in sync_rows],
        "largest_gaps": sorted(largest, key=lambda row: row["gap_ms"], reverse=True)[:3],
    }


def analyze(db: sqlite3.Connection, summary: dict[str, Any], pair_number: int,
            expected: dict[str, Any] | None = None) -> dict[str, Any]:
    pairs = [pair for pair in summary["pairs"] if pair["pair"] == pair_number]
    if len(pairs) != 1:
        raise ValueError(f"Pair {pair_number} not found")
    pair = pairs[0]
    cases = {case["case_id"]: case for case in summary["cases"]}
    call_ids = {
        case_id: [call_id for batch in cases[case_id]["short_decode"]["decode_batches"]
                  for call_id in batch.get("model_forward_call_ids", [])]
        for case_id in (pair["early_case_id"], pair["post_short_case_id"])
    }
    if not all(call_ids.values()):
        raise ValueError("A short replay has no linked model-forward calls")
    ranges = _forward_ranges(db, set(call_ids[pair["early_case_id"]] +
                               call_ids[pair["post_short_case_id"]]))
    result = {}
    for mode, case_id in (("early", pair["early_case_id"]),
                          ("post_short", pair["post_short_case_id"])):
        calls = []
        for call_id in call_ids[case_id]:
            calls.append({"call_id": call_id, **_call_gaps(db, *ranges[call_id])})
        totals = {
            key: round(sum(row[key] for row in calls), 3)
            for key in ("gap_ms", "cpu_before_launch_ms", "launch_api_ms",
                        "after_launch_api_ms", "stream_wait_event_gpu_ms",
                        "blocking_sync_api_ms")
        }
        result[mode] = {"case_id": case_id, "forward_calls": calls,
                        "forward_call_count": len(calls),
                        "kernel_count": sum(row["kernel_count"] for row in calls), **totals}
        if expected is not None:
            expected_pairs = [row for row in expected["pairs"] if row["pair"] == pair_number]
            if len(expected_pairs) != 1:
                raise ValueError("Kernel attribution is missing the requested pair")
            reference = expected_pairs[0][mode]
            if (result[mode]["kernel_count"] != reference["kernel_count"] or
                    abs(result[mode]["gap_ms"] - reference["kernel_gap_inside_span_ms"]) > 0.01):
                raise ValueError(f"CUDA launch gaps disagree with kernel attribution in {mode}")
    return {
        "schema": "agentic_work_audit.cuda_launch_gaps.v1",
        "status": "validated_subset",
        "run_id": summary["run_id"],
        "pair": pair_number,
        **result,
        "early_minus_post_short_ms": {
            key: round(result["early"][key] - result["post_short"][key], 3)
            for key in ("gap_ms", "cpu_before_launch_ms", "launch_api_ms",
                        "after_launch_api_ms", "stream_wait_event_gpu_ms",
                        "blocking_sync_api_ms")
        },
        "limitations": [
            "Only the captured first pair is analyzed; later CUDA activity was missing.",
            "CPU-before-launch is a timeline classification, not proof of why the host delayed.",
            "After-launch gaps can include dependencies or stream waits; this trace does not prove their cause.",
            "Nsight profiling perturbs timing; use unprofiled runs for latency estimates.",
            "These gaps are inside model-forward calls only, not between forwards or request submission.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=1)
    parser.add_argument("--kernel-attribution", type=Path,
                        help="Cross-check kernel count and gap time against the saved kernel report")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    with sqlite3.connect(args.sqlite) as db:
        expected = (json.loads(args.kernel_attribution.read_text(encoding="utf-8"))
                    if args.kernel_attribution else None)
        result = analyze(db, summary, args.pair, expected)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": result["run_id"],
                      "early_minus_post_short_ms": result["early_minus_post_short_ms"]}, indent=2))


if __name__ == "__main__":
    main()
