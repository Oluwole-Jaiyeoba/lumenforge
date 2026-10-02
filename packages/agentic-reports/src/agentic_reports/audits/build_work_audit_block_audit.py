#!/usr/bin/env python3
"""Reuse the existing KV ledger for a conservative work-audit lifecycle view."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from agentic_backends.sglang.audit_v0510 import lifecycle_role, normalize_lifecycle_evidence
from agentic_instrumentation import assess_loaded_match, validate_events
from agentic_work_audit import AuditEvent, read_events

from agentic_reports.block_ledger import (
    KVEventType,
    NormalizedKVEvent,
    block_ledger_rows,
    build_block_ledger,
    normalize_sglang_trace_events as normalize_trace_events,
)


SEMANTIC_ROLES = {
    "semantic_write", "semantic_evict_gpu", "semantic_evict_host", "semantic_load",
}


def _first(events: list[AuditEvent], kind: str) -> AuditEvent | None:
    return next((event for event in events if event.kind == kind), None)


def _period(time_ns: int | None, timeline: dict[str, AuditEvent | None]) -> str:
    if time_ns is None:
        return "unknown"
    tool_start = timeline["tool_start"]
    tool_end = timeline["tool_end"]
    replay = timeline["replay_sent"]
    replay_end = timeline["replay_finished"]
    if not all((tool_start, tool_end, replay)):
        return "unknown"
    if time_ns < tool_start.ts_ns:
        return "initial_request"
    if time_ns < tool_end.ts_ns:
        return "tool_wait"
    if time_ns < replay.ts_ns:
        return "ready_before_replay"
    if replay_end and time_ns <= replay_end.ts_ns:
        return "replay"
    return "after_replay"


def analyze_block_audit(
    trace_rows: list[dict[str, Any]], harness_events: list[AuditEvent], validation: dict[str, Any],
) -> dict[str, Any]:
    runtime = validation.get("runtime") or {}
    failures: list[str] = []
    if validation.get("status") != "validated":
        failures.append("the preceding work-audit validation did not pass")
    if (runtime.get("backend_version"), runtime.get("adapter")) != ("0.5.10.post1", "v0510"):
        failures.append("the lifecycle role map requires the pinned v0510 backend")

    normalized = normalize_trace_events(trace_rows, version="0.5.10.post1")
    evidence = [item for row in trace_rows if (item := normalize_lifecycle_evidence(row)) is not None]
    semantic = [event for event in normalized if lifecycle_role(event.source_event) in SEMANTIC_ROLES]
    layer_copies = [event for event in normalized if lifecycle_role(event.source_event) == "layer_copy"]
    ledger_rows = block_ledger_rows(build_block_ledger(semantic))
    role_counts = Counter(lifecycle_role(event.source_event) for event in normalized)

    cases: list[dict[str, Any]] = []
    for original in validation.get("cases") or []:
        session = str(original["session_id"])
        timeline_rows = [event for event in harness_events if event.session_id == session]
        timeline = {kind: _first(timeline_rows, kind) for kind in
                    ("tool_start", "tool_end", "replay_sent", "replay_first_token",
                     "replay_finished", "host_resident_proof")}
        replay = timeline["replay_sent"]
        first_token = timeline["replay_first_token"]
        replay_id = replay.request_id if replay else ""
        loads = [event for event in semantic if event.session_id == session
                 and event.event_type == KVEventType.LOAD_GPU]
        copies = [event for event in layer_copies if event.session_id == session]
        replay_matches = [event for event in normalized if event.session_id == session
                          and event.event_type == KVEventType.MATCH_PREFIX
                          and replay_id
                          and replay_id in (event.request_id, event.agent_request_id)
                          and replay and first_token and event.time_ns is not None
                          and replay.ts_ns <= event.time_ns <= first_token.ts_ns
                          and event.token_count > 0]
        selected_node = str(timeline["host_resident_proof"].evidence.get("node_id") or "") \
            if timeline["host_resident_proof"] else ""
        linked_copies: list[NormalizedKVEvent] = []
        for copy in copies:
            candidates = [load for load in loads
                          if load.host_index_signature and load.device_index_signature
                          and load.host_index_signature == copy.host_index_signature
                          and load.device_index_signature == copy.device_index_signature]
            if len(candidates) == 1:
                linked_copies.append(copy)
        case_failures: list[str] = []
        session_evidence = [event for event in evidence if event.session_id == session]
        if original["case_type"] == "host_backed":
            coverage = validate_events("kv_lifecycle", session_evidence)
            if not coverage["valid"]:
                case_failures.append(
                    "missing required lifecycle evidence: "
                    + ", ".join([*coverage["missing"], *coverage["invalid_fields"]])
                )
        matched_evidence = [event for event in session_evidence if event.signal_id == "kv.prefix_match"
                            and event.request_id == replay_id and replay and first_token
                            and replay.ts_ns <= event.time_ns <= first_token.ts_ns]
        slot_proof = assess_loaded_match(
            (event for event in session_evidence if event.signal_id == "kv.load_gpu"),
            matched_evidence,
            (event for event in session_evidence if event.signal_id == "kv.evict_gpu"),
        )
        if original["case_type"] == "host_backed":
            if len(loads) != 1:
                case_failures.append("expected one session-linked semantic load transition")
            else:
                load = loads[0]
                if not selected_node or load.node_id != selected_node:
                    case_failures.append("selected host node did not match the loaded node")
                if load.token_count != int(original.get("native_loaded_tokens") or 0):
                    case_failures.append("semantic load token count differs from native load completion")
                if _period(load.time_ns, timeline) != "ready_before_replay":
                    case_failures.append("semantic load did not finish between tool return and replay")
            if not copies or len(linked_copies) != len(copies):
                case_failures.append("some per-layer copies could not be linked by exact index signatures")
        failures.extend(f"{session}: {issue}" for issue in case_failures)
        cases.append({
            "session_id": session,
            "case_type": original["case_type"],
            "status": "validated" if not case_failures else "failed",
            "semantic_load_transitions": len(loads),
            "nested_load_observations": sum(1 for event in normalized if event.session_id == session
                                            and lifecycle_role(event.source_event) == "nested_load"),
            "layer_copy_observations": len(copies),
            "exactly_linked_layer_copies": len(linked_copies),
            "selected_host_node_id": selected_node or None,
            "loaded_node_ids": sorted({event.node_id for event in loads if event.node_id}),
            "semantic_loaded_tokens": sum(event.token_count for event in loads),
            "load_periods": [_period(event.time_ns, timeline) for event in loads],
            "replay_prefix_match_observations": len(replay_matches),
            "max_replay_matched_tokens": max((event.token_count for event in replay_matches), default=0),
            "same_loaded_block_used_by_replay": "not_proven" if loads else "not_applicable",
            **slot_proof,
            "failures": case_failures,
        })

    return {
        "schema": "agentic_work_audit.block_lifecycle.v1",
        "run_id": validation.get("run_id"),
        "status": "validated" if not failures else "failed",
        "failures": failures,
        "backend_version": runtime.get("backend_version"),
        "adapter": runtime.get("adapter"),
        "event_roles": dict(sorted(role_counts.items())),
        "logical_block_records": len(ledger_rows),
        "logical_loaded_records": sum(int(row.get("load_gpu_events") or 0) > 0 for row in ledger_rows),
        "cases": cases,
        "blocks": ledger_rows,
        "interpretation": (
            "Semantic cache transitions feed the existing logical-block ledger. Nested load calls and "
            "per-layer copies support one load; they are not additional logical loads. Verified GPU-slot "
            "overlap with a replay prefix match is stronger than a prefix-length correlation, but it "
            "does not prove those exact slots were consumed by model kernels."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.trace.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    validation = json.loads(args.summary.read_text(encoding="utf-8"))
    result = analyze_block_audit(rows, read_events(args.harness), validation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": result["failures"],
                      "event_roles": result["event_roles"], "cases": result["cases"]}, indent=2))
    if result["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
