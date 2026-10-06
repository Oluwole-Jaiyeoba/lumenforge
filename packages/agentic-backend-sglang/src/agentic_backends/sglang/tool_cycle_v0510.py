"""Map pinned SGLang trace rows to neutral tool-cycle stage observations."""

from __future__ import annotations

from typing import Any


def normalize_tool_cycle_stage(row: dict[str, Any]) -> dict[str, Any] | None:
    name = row.get("event")
    context = row.get("kv_context") or {}
    request = context.get("request") or {}
    base = {"ts_ns": row.get("ts_ns"), "request_ids": []}
    if name == "kv_telemetry.request_stage" and row.get("stage") == "cache_match_prefix":
        if row.get("phase") not in ("start", "end"):
            return None
        return {**base, "kind": f"lookup_{row['phase']}",
                "request_ids": [row.get("request_id")]}
    if name == "hiradix.match_prefix.end":
        result = row.get("result") or []
        if not result or not isinstance(result[0], dict):
            return None
        return {**base, "kind": "matched_prefix",
                "request_ids": [request.get("agent_request_id")],
                "matched_prefix_tokens": result[0].get("index_count")}
    if name in ("hiradix.load_back.start", "hiradix.load_back.end"):
        result = {**base, "kind": "load_start" if name.endswith(".start") else "load_end",
                  "request_ids": [context.get("agent_request_id")]}
        if name.endswith(".end"):
            result["duration_ms"] = row.get("duration_ms") or 0
        return result
    if name in ("scheduler.run_batch.start", "scheduler.run_batch.end"):
        batch = context.get("batch") or {}
        return {**base, "kind": "batch_start" if name.endswith(".start") else "batch_end",
                "request_ids": [item.get("agent_request_id") for item in batch.get("requests") or []]}
    if name == "hostpool.load_to_device_per_layer.start":
        return {**base, "kind": "layer_copy_start"}
    return None
