#!/usr/bin/env python3
"""Compare no-check, check-only, and load arms by replay stage."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

from agentic_instrumentation.catalog import REQUEST_STAGE_ORDERS

STAGE_BY_ORDER = {order: stage for stage, order in REQUEST_STAGE_ORDERS.items()}
SEGMENTS = (
    ("submit_to_receive", "request_start_ns", "backend_receive"),
    ("receive_to_queue", "backend_receive", "queue_enter"),
    ("queue_to_cache_lookup", "queue_enter", "cache_lookup"),
    ("receive_to_cache_lookup", "backend_receive", "cache_lookup"),
    ("cache_lookup_to_first_token", "cache_lookup", "first_token_ns"),
    ("first_token_to_finish", "first_token_ns", "request_end_ns"),
)
MODES = ("baseline", "check_only", "controller")


def _trace_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _timeline(arm_dir: Path, replays: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    trace = arm_dir / "backend_trace.jsonl"
    if not trace.exists():
        trace = trace.with_suffix(".jsonl.gz")
    if not trace.exists():
        raise FileNotFoundError(f"missing backend trace for {arm_dir}")
    timeline = {row["request_id"]: {
        "request_start_ns": row["request_start_ns"],
        "first_token_ns": row["first_token_ns"],
        "request_end_ns": row["request_end_ns"],
    } for row in replays}
    for event in _trace_rows(trace):
        if event.get("event") != "kv_telemetry.request_stage" or event.get("phase") != "start":
            continue
        request_id, stage = event.get("request_id"), STAGE_BY_ORDER.get(event.get("stage_order"))
        if request_id in timeline and stage and isinstance(event.get("ts_ns"), int):
            previous = timeline[request_id].get(stage)
            timeline[request_id][stage] = min(previous, event["ts_ns"]) if previous else event["ts_ns"]
    return timeline


def _stage_summary(timeline: dict[str, dict[str, int]]) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for name, start, end in SEGMENTS:
        values = [(row[end] - row[start]) / 1_000_000 for row in timeline.values()
                  if start in row and end in row and row[end] >= row[start]]
        result[name] = {
            "covered_replays": len(values), "total_replays": len(timeline),
            "mean_ms": round(statistics.mean(values), 3) if values else None,
            "median_ms": round(statistics.median(values), 3) if values else None,
            "sum_ms": round(sum(values), 3) if values else None,
        }
    return result


def _paired_deltas(left: dict[str, dict[str, int]], right: dict[str, dict[str, int]]) -> dict[str, Any]:
    if left.keys() != right.keys():
        raise ValueError("unmatched replay identities")
    result = {}
    for name, start, end in SEGMENTS:
        deltas = []
        for request_id, row in left.items():
            other = right[request_id]
            if all(key in item for item in (row, other) for key in (start, end)):
                if row[end] < row[start] or other[end] < other[start]:
                    continue
                deltas.append(((other[end] - other[start]) - (row[end] - row[start])) / 1_000_000)
        result[name] = {
            "matched_replays": len(deltas), "total_replays": len(left),
            "mean_added_ms": round(statistics.mean(deltas), 3) if deltas else None,
            "median_added_ms": round(statistics.median(deltas), 3) if deltas else None,
            "sum_added_ms": round(sum(deltas), 3) if deltas else None,
        }
    return result


def _substantive_decode(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {row["request_id"]: row for row in left["replays"] if row.get("stream_chunks", 0) > 2}
    right_rows = {row["request_id"]: row for row in right["replays"] if row.get("stream_chunks", 0) > 2}
    matched = sorted(left_rows.keys() & right_rows.keys())
    deltas = [((right_rows[key]["request_end_ns"] - right_rows[key]["first_token_ns"]) -
               (left_rows[key]["request_end_ns"] - left_rows[key]["first_token_ns"])) / 1_000_000
              for key in matched]
    return {
        "matched_replays": len(deltas), "total_replays": len(left["replays"]),
        "mean_added_ms": round(statistics.mean(deltas), 3) if deltas else None,
        "median_added_ms": round(statistics.median(deltas), 3) if deltas else None,
        "sum_added_ms": round(sum(deltas), 3) if deltas else None,
        "excluded_short_responses": len(left["replays"]) - len(matched),
    }


def _load_windows(arm: dict[str, Any], timeline: dict[str, dict[str, int]]) -> dict[str, int]:
    candidates = [row for row in arm["decisions"] if row.get("decision") == "load"]
    observed = [row for row in candidates if isinstance(row.get("load_request_ns"), int)
                and isinstance(row.get("load_finished_observed_ns"), int)]
    intervals = [(row["load_request_ns"], row["load_finished_observed_ns"]) for row in observed]
    return {
        "load_decisions": len(candidates), "load_attempts": arm["controller_load_attempts"],
        "confirmed_control_windows": len(intervals),
        "windows_with_other_replay_before_first_token": sum(any(
            request_id != f"{row['session_id']}-replay-{row['step']}"
            and replay["request_start_ns"] < end and replay["first_token_ns"] > start
            for request_id, replay in timeline.items()) for row, (start, end) in zip(observed, intervals)),
        "windows_with_other_replay_after_first_token": sum(any(
            request_id != f"{row['session_id']}-replay-{row['step']}"
            and replay["first_token_ns"] < end and replay["request_end_ns"] > start
            for request_id, replay in timeline.items()) for row, (start, end) in zip(observed, intervals)),
    }


def compare(arms_dir: Path, run_id: str) -> dict[str, Any]:
    by_seed: dict[int, dict[str, tuple[dict[str, Any], dict[str, dict[str, int]]]]] = {}
    for path in sorted(arms_dir.glob("*/summary.json")):
        arm = json.loads(path.read_text(encoding="utf-8"))
        mode = arm["mode"]
        if mode not in MODES or arm["frontend_priority"] != "none" or arm["forced_eviction"]:
            raise ValueError(f"invalid attribution arm {path}")
        seed = int(arm["seed"])
        if mode in by_seed.setdefault(seed, {}):
            raise ValueError(f"duplicate seed {seed} mode {mode}")
        by_seed[seed][mode] = arm, _timeline(path.parent, arm["replays"])
    if not by_seed:
        raise ValueError("no attribution arms")
    seeds = []
    for seed, modes in sorted(by_seed.items()):
        if set(modes) != set(MODES):
            raise ValueError(f"seed {seed} requires {MODES}")
        if len({json.dumps(modes[mode][0]["workload"], sort_keys=True) for mode in MODES}) != 1:
            raise ValueError(f"seed {seed} workload mismatch")
        if len({frozenset(modes[mode][1]) for mode in MODES}) != 1:
            raise ValueError(f"seed {seed} replay mismatch")
        summaries = {}
        for mode in MODES:
            arm, timeline = modes[mode]
            summaries[mode] = {
                "run_id": arm["run_id"], "replay_count": arm["replay_count"],
                "total_replay_ttft_ms": arm["total_replay_ttft_ms"],
                "workflow_makespan_ms": arm["workflow_makespan_ms"],
                "native_load_events": arm["native_load_events"],
                "controller_plan_checks": arm["controller_plan_checks"],
                "stage_timing": _stage_summary(timeline),
                "load_windows": _load_windows(arm, timeline),
            }
        check_cost = _paired_deltas(modes["baseline"][1], modes["check_only"][1])
        load_association = _paired_deltas(modes["check_only"][1], modes["controller"][1])
        check_cost["substantive_decode"] = _substantive_decode(modes["baseline"][0], modes["check_only"][0])
        load_association["substantive_decode"] = _substantive_decode(modes["check_only"][0], modes["controller"][0])
        seeds.append({
            "seed": seed, "arms": summaries,
            "check_cost": check_cost, "load_association": load_association,
        })
    return {
        "schema": "agentic_work_audit.kv_load_attribution.v1", "run_id": run_id,
        "status": "complete", "seed_count": len(seeds), "seeds": seeds,
        "interpretation_limit": (
            "A load control-to-confirmation window is not the physical CUDA copy interval. "
            "Stage differences isolate check-only from check-plus-load policy, but concurrent "
            "batch trajectories can diverge; they do not prove HBM bandwidth contention. "
            "Missing request-linked stages are reported as missing coverage, never zero delay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.arms_dir, args.run_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compared {result['seed_count']} three-arm seeds")


if __name__ == "__main__":
    main()
