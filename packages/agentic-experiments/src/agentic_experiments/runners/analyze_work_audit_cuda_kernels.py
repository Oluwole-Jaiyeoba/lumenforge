"""Join Nsight CUDA kernels to request-linked work-audit forward calls."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


CALL_ID = re.compile(r"\bcall_id=([^\s]+)")


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    merged = 0
    left, right = intervals[0]
    for start, end in intervals[1:]:
        if start > right:
            merged += right - left
            left, right = start, end
        else:
            right = max(right, end)
    return merged + right - left


def _merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def _intersection_ns(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    total = 0
    for a, b in left:
        for c, d in right:
            total += max(0, min(b, d) - max(a, c))
    return total


def forward_kernels(db: sqlite3.Connection, wanted: set[str] | None = None) -> dict[str, dict[str, Any]]:
    ranges = db.execute(
        "SELECT start, end, text, globalTid FROM NVTX_EVENTS "
        "WHERE text LIKE 'agentic_kv:worker.forward_batch_generation:% call_id=%' "
        "AND end IS NOT NULL"
    ).fetchall()
    if not ranges:
        raise ValueError("Nsight trace has no call-linked model-forward NVTX ranges")
    result: dict[str, dict[str, Any]] = {}
    for start, end, label, tid in ranges:
        match = CALL_ID.search(label)
        if not match:
            continue
        if wanted is not None and match.group(1) not in wanted:
            continue
        kernels = db.execute(
            "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
            "JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId "
            "AND (k.globalPid >> 24) = (r.globalTid >> 24) "
            "WHERE r.globalTid = ? AND r.start >= ? AND r.start <= ? "
            "ORDER BY k.start", (tid, start, end)
        ).fetchall()
        runtime = db.execute(
            "SELECT COALESCE(SUM(end-start),0) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE globalTid = ? AND start >= ? AND start <= ?", (tid, start, end)
        ).fetchone()[0]
        intervals = [(int(a), int(b)) for a, b in kernels]
        span_start = intervals[0][0] if intervals else start
        span_end = intervals[-1][1] if intervals else end
        htod = db.execute(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY "
            "WHERE (globalPid >> 24) = (? >> 24) AND copyKind = 1 "
            "AND start < ? AND end > ? ORDER BY start", (tid, span_end, span_start)
        ).fetchall()
        clipped_htod = _merged([(max(int(a), span_start), min(int(b), span_end))
                                for a, b in htod])
        active_kernels = _merged(intervals)
        gaps = [(a[1], b[0]) for a, b in zip(active_kernels, active_kernels[1:])]
        active = _union_ns(intervals)
        span = intervals[-1][1] - intervals[0][0] if intervals else 0
        result[match.group(1)] = {
            "kernel_count": len(intervals),
            "kernel_duration_sum_ms": round(sum(b - a for a, b in intervals) / 1e6, 3),
            "kernel_active_union_ms": round(active / 1e6, 3),
            "kernel_span_ms": round(span / 1e6, 3),
            "kernel_gap_inside_span_ms": round((span - active) / 1e6, 3),
            "cuda_runtime_api_ms": round(runtime / 1e6, 3),
            "nvtx_forward_wall_ms": round((end - start) / 1e6, 3),
            "htod_during_kernel_span_ms": round(_union_ns(clipped_htod) / 1e6, 3),
            "htod_during_kernel_gaps_ms": round(_intersection_ns(gaps, clipped_htod) / 1e6, 3),
        }
    return result


def analyze(summary: dict[str, Any], calls: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {}
    for case in summary["cases"]:
        batches = case["short_decode"]["decode_batches"]
        call_ids = [call_id for batch in batches for call_id in
                    batch.get("model_forward_call_ids", [])]
        matched = [calls[call_id] for call_id in call_ids if call_id in calls]
        if len(matched) != len(call_ids) or not matched or any(
                row["kernel_count"] == 0 for row in matched):
            raise ValueError(f"Missing CUDA kernels for short replay in {case['case_id']}: "
                             f"{len(matched)}/{len(call_ids)} forward calls matched")
        metrics = {
            "forward_call_count": len(matched),
            "kernel_count": sum(row["kernel_count"] for row in matched),
        }
        for key in ("kernel_duration_sum_ms", "kernel_active_union_ms", "kernel_span_ms",
                    "kernel_gap_inside_span_ms", "cuda_runtime_api_ms", "nvtx_forward_wall_ms",
                    "htod_during_kernel_span_ms", "htod_during_kernel_gaps_ms"):
            metrics[key] = round(sum(row[key] for row in matched), 3)
        metrics["forward_calls"] = [dict(call_id=call_id, **calls[call_id]) for call_id in call_ids]
        by_id[case["case_id"]] = metrics
        case["cuda_kernels"] = metrics
    pairs = []
    for pair in summary["pairs"]:
        early = by_id[pair["early_case_id"]]
        post = by_id[pair["post_short_case_id"]]
        deltas = {key: round(early[key] - post[key], 3) for key in
                  ("kernel_duration_sum_ms", "kernel_active_union_ms", "kernel_span_ms",
                   "kernel_gap_inside_span_ms", "cuda_runtime_api_ms", "nvtx_forward_wall_ms",
                   "htod_during_kernel_span_ms", "htod_during_kernel_gaps_ms")}
        pairs.append({"pair": pair["pair"], "early": early, "post_short": post,
                      "early_minus_post_short_ms": deltas})
    return {
        "schema": "agentic_work_audit.cuda_kernel_attribution.v1",
        "status": "validated", "run_id": summary["run_id"], "pairs": pairs,
        "limitations": [
            "This is a profiled mechanism run; use the separate unprofiled run for latency claims.",
            "Summed kernel durations can overlap; active union avoids double-counting within a forward call.",
            "Kernel gaps may reflect CPU launch, dependencies, or synchronization; they are not proof of HBM contention.",
            "CUDA launches are assigned by thread and NVTX range, not by GPU completion time.",
            "Host-to-device copies inside a forward's GPU kernel span are not identified as a specific session's KV copy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", type=int,
                        help="Analyze only this pair when a later part of the CUDA trace was lost")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if args.pair is not None:
        selected = [pair for pair in summary["pairs"] if pair["pair"] == args.pair]
        if len(selected) != 1:
            raise SystemExit(f"Pair {args.pair} not found")
        case_ids = {selected[0]["early_case_id"], selected[0]["post_short_case_id"]}
        summary["pairs"] = selected
        summary["cases"] = [case for case in summary["cases"] if case["case_id"] in case_ids]
    wanted = {call_id for case in summary["cases"] for batch in
              case["short_decode"]["decode_batches"] for call_id in
              batch.get("model_forward_call_ids", [])}
    with sqlite3.connect(args.sqlite) as db:
        attribution = analyze(summary, forward_kernels(db, wanted))
    if args.pair is not None:
        attribution["status"] = "validated_subset"
        attribution["scope"] = (f"pair {args.pair} only; later CUDA trace coverage is incomplete "
                                "and the full run did not pass the attribution gate")
        attribution["limitations"].append("This subset alone is not an order-balanced experiment.")
        args.out.write_text(json.dumps(attribution, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": attribution["status"], "pairs": [item["early_minus_post_short_ms"]
                          for item in attribution["pairs"]]}, indent=2))
        return
    summary["nsys_kernel_attribution"] = {
        "status": attribution["status"], "path": str(args.out),
        "pairs": [{"pair": item["pair"], "early_minus_post_short_ms": item["early_minus_post_short_ms"]}
                  for item in attribution["pairs"]],
    }
    args.out.write_text(json.dumps(attribution, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["nsys_kernel_attribution"], indent=2))


if __name__ == "__main__":
    main()
