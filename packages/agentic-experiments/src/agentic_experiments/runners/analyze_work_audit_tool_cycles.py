"""Join replay tool returns to cache and first-batch timestamps."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from agentic_backends.sglang.tool_cycle_v0510 import normalize_tool_cycle_stage


def _duration_ms(start_ns: int, end_ns: int) -> float:
    return round((end_ns - start_ns) / 1e6, 3)


def analyze(summary: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    by_request: dict[str, dict[str, Any]] = {row["request_id"]: {} for row in summary["turns"]}
    layer_copy_count = 0
    for event in trace:
        stage = normalize_tool_cycle_stage(event)
        if not stage:
            continue
        kind = stage["kind"]
        if kind == "layer_copy_start":
            layer_copy_count += 1
        for request_id in stage.get("request_ids") or []:
            if request_id not in by_request:
                continue
            stages = by_request[request_id]
            if kind in ("lookup_start", "lookup_end"):
                stages.setdefault(f"{kind}_ns", stage["ts_ns"])
            elif kind == "matched_prefix":
                stages.setdefault("matched_prefix_tokens", stage["matched_prefix_tokens"])
            elif kind == "load_start":
                stages = by_request[request_id]
                stages["kv_load_back_count"] = stages.get("kv_load_back_count", 0) + 1
                stages.setdefault("kv_load_first_start_ns", stage["ts_ns"])
            elif kind == "load_end":
                stages["kv_load_back_call_ms"] = round(
                    stages.get("kv_load_back_call_ms", 0) + stage["duration_ms"], 3)
                stages["kv_load_last_end_ns"] = stage["ts_ns"]
            elif kind in ("batch_start", "batch_end"):
                key = "first_batch_start_ns" if kind == "batch_start" else "first_batch_end_ns"
                stages.setdefault(key, stage["ts_ns"])
    required = ("lookup_start_ns", "lookup_end_ns", "first_batch_start_ns", "first_batch_end_ns",
                "matched_prefix_tokens")
    for row in summary["turns"]:
        stages = by_request[row["request_id"]]
        missing = [key for key in required if key not in stages]
        if missing:
            raise ValueError(f"{row['request_id']}: missing stage evidence: {', '.join(missing)}")
        timeline = [row["tool_return_ns"], row["request_start_ns"], stages["lookup_start_ns"],
                    stages["lookup_end_ns"], stages["first_batch_start_ns"],
                    stages["first_batch_end_ns"], row["first_token_ns"]]
        if timeline != sorted(timeline):
            raise ValueError(f"{row['request_id']}: stage timestamps are out of order: {timeline}")
        row.update(stages)
        row.setdefault("kv_load_back_count", 0)
        row.setdefault("kv_load_back_call_ms", 0)
        row["request_to_lookup_ms"] = _duration_ms(row["request_start_ns"], stages["lookup_start_ns"])
        row["lookup_ms"] = _duration_ms(stages["lookup_start_ns"], stages["lookup_end_ns"])
        row["lookup_to_batch_ms"] = _duration_ms(stages["lookup_end_ns"], stages["first_batch_start_ns"])
        row["first_batch_method_ms"] = _duration_ms(stages["first_batch_start_ns"],
                                                     stages["first_batch_end_ns"])
        row["batch_end_to_client_ms"] = _duration_ms(stages["first_batch_end_ns"], row["first_token_ns"])
        row["uncached_prompt_tokens"] = max(0, row["prompt_tokens"] - stages["matched_prefix_tokens"])
        if row["kv_load_back_count"]:
            start = stages.get("kv_load_first_start_ns")
            end = stages.get("kv_load_last_end_ns")
            if isinstance(start, int) and start >= stages["lookup_end_ns"]:
                row["lookup_to_load_ms"] = _duration_ms(stages["lookup_end_ns"], start)
            if isinstance(end, int) and end <= stages["first_batch_start_ns"]:
                row["load_end_to_batch_ms"] = _duration_ms(end, stages["first_batch_start_ns"])
    active = [row for row in summary["turns"] if row["kind"] == "active"]
    if len(active) != summary["active_count"] * summary["turn_count"]:
        raise ValueError("Active replay count does not match the planned workload")
    loaded = [row for row in active if row["kv_load_back_count"]]
    unloaded = [row for row in active if not row["kv_load_back_count"]]

    def p95(key: str) -> float:
        values = sorted(row[key] for row in active)
        return round(values[math.ceil(len(values) * 0.95) - 1], 3)

    summary["evidence_status"] = "complete_stage_join"
    summary["status"] = "observed"
    summary["limitations"] = [
        "The first-batch end timestamp is a scheduler-method boundary, not proof that GPU work completed.",
        "These are deterministic synthetic tool results, not autonomous tool or model decisions.",
        "Donor traffic is not proof of KV movement; host-to-device copy counts are reported separately.",
    ]
    summary["measurements"] = {
        "active_replay_count": len(active),
        "active_ttft_sum_ms": round(sum(row["ttft_ms"] for row in active), 3),
        "active_ttft_median_ms": round(statistics.median(row["ttft_ms"] for row in active), 3),
        "active_ttft_p95_ms": p95("ttft_ms"),
        "active_lookup_to_batch_median_ms": round(statistics.median(row["lookup_to_batch_ms"] for row in active), 3),
        "active_lookup_to_batch_p95_ms": p95("lookup_to_batch_ms"),
        "active_loaded_lookup_to_batch_median_ms": (round(statistics.median(
            row["lookup_to_batch_ms"] for row in loaded), 3) if loaded else None),
        "active_unloaded_lookup_to_batch_median_ms": (round(statistics.median(
            row["lookup_to_batch_ms"] for row in unloaded), 3) if unloaded else None),
        "active_first_token_after_tool_sum_ms": round(sum(row["first_token_after_tool_ms"] for row in active), 3),
        "active_prompt_tokens_first_last": [min(row["prompt_tokens"] for row in active if row["turn"] == 1),
                                            max(row["prompt_tokens"] for row in active if row["turn"] == summary["turn_count"])],
        "host_to_device_copy_calls": layer_copy_count,
        "kv_load_back_operations": sum(stage.get("kv_load_back_count", 0)
                                       for stage in by_request.values()),
        "active_replays_with_kv_load_back": sum(row["kv_load_back_count"] > 0 for row in active),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = analyze(summary, trace)
    args.summary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["measurements"], indent=2))


if __name__ == "__main__":
    main()
