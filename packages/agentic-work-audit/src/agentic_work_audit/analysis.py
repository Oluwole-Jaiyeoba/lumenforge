"""Conservative analysis: distinguish observed timing from inferred opportunity."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .events import AuditEvent


def _first(rows: list[AuditEvent], kind: str) -> AuditEvent | None:
    return next((row for row in rows if row.kind == kind), None)


def analyze_validation(
    events: Iterable[AuditEvent], *, expected_sessions: tuple[str, ...],
    expected_runtime: tuple[str, str] | None = None,
) -> dict[str, Any]:
    rows = sorted(events, key=lambda row: row.ts_ns)
    kinds = Counter(row.kind for row in rows)
    runtime = _first(rows, "runtime_hooks")
    failures: list[str] = []
    if runtime is None:
        failures.append("backend hook installation proof is missing")
    else:
        detail = runtime.evidence
        if expected_runtime and (detail.get("backend_version"), detail.get("adapter")) != expected_runtime:
            failures.append("backend runtime differs from the selected reference")
        if detail.get("missing_required_hooks"):
            failures.append("required backend hooks are missing")
        capabilities = set(detail.get("audit_capabilities") or [])
        for required in ("cache_match", "native_load", "host_copy"):
            if required not in capabilities:
                failures.append(f"required audit capability is absent: {required}")
    if kinds["hook_error"]:
        failures.append(f"{kinds['hook_error']} backend hook errors were recorded")

    by_session: dict[str, list[AuditEvent]] = defaultdict(list)
    for row in rows:
        if row.session_id:
            by_session[row.session_id].append(row)

    cases: list[dict[str, Any]] = []
    for session_id in expected_sessions:
        case_rows = by_session[session_id]
        tool_start = _first(case_rows, "tool_start")
        tool_end = _first(case_rows, "tool_end")
        replay = _first(case_rows, "replay_sent")
        first_token = _first(case_rows, "replay_first_token")
        host = _first(case_rows, "host_resident_proof")
        evict = _first(case_rows, "device_evict_proof")
        load = _first(case_rows, "load_complete")
        matches = [row for row in case_rows if row.kind == "cache_match" and replay
                   and row.request_id == replay.request_id
                   and int(row.evidence.get("cached_prefix_tokens") or 0) > 0]
        copies = [row for row in case_rows if row.kind == "layer_copy"]
        case_failures: list[str] = []
        if not all((tool_start, tool_end, replay, first_token)):
            case_failures.append("tool/replay timeline is incomplete")
        elif not (tool_start.ts_ns < tool_end.ts_ns <= replay.ts_ns <= first_token.ts_ns):
            case_failures.append("tool/replay timestamps are not ordered")
        case_type = "host_backed" if session_id.endswith("-host") else "warm_control"
        if case_type == "host_backed":
            if not evict or not host or not load:
                case_failures.append("host-backed case lacks eviction, host-residency, or completed-load proof")
            elif not (evict.ts_ns <= host.ts_ns <= load.ts_ns):
                case_failures.append("host-backed evidence is out of order")
            if not copies:
                case_failures.append("host-backed case has no identity-linked native layer copy")
        elif not matches:
            case_failures.append("warm control has no cache-match observation")

        # Presence in host memory at submission does not prove that the replay
        # stalled on it. Require both an identity-linked load and time overlap.
        replay_stalled_on_load: bool | None = None
        if replay and first_token and load and load.request_id == replay.request_id:
            replay_stalled_on_load = replay.ts_ns <= load.ts_ns <= first_token.ts_ns

        cases.append({
            "session_id": session_id,
            "case_type": case_type,
            "status": "validated" if not case_failures else "failed",
            "failures": case_failures,
            "tool_wait_ms": round((tool_end.ts_ns - tool_start.ts_ns) / 1e6, 3) if tool_start and tool_end else None,
            "replay_ttft_ms": round((first_token.ts_ns - replay.ts_ns) / 1e6, 3) if replay and first_token else None,
            "host_resident_tokens": int(host.evidence.get("host_tokens") or 0) if host else 0,
            "native_loaded_tokens": int(load.evidence.get("loaded_tokens") or 0) if load else 0,
            "native_layer_copies": len(copies),
            "exposed_prepare_gap_ms": round((load.ts_ns - tool_end.ts_ns) / 1e6, 3)
            if tool_end and load and tool_end.ts_ns <= load.ts_ns <= replay.ts_ns else None,
            "replay_stalled_on_load": replay_stalled_on_load,
            "cache_match_observations": len(matches),
            "max_cached_prefix_tokens": max((int(row.evidence.get("cached_prefix_tokens") or 0)
                                             for row in matches), default=0),
        })
        failures.extend(f"{session_id}: {item}" for item in case_failures)

    return {
        "schema": "agentic_work_audit.validation.v1",
        "status": "validated" if not failures else "failed",
        "failures": failures,
        "runtime": runtime.evidence if runtime else None,
        "event_counts": dict(sorted(kinds.items())),
        "cases": cases,
        "opportunity_ledgers": {
            "backup_reuse": "unknown: no validated block identity ledger",
            "avoidable_eviction": "unknown: no policy-constrained counterfactual",
            "hbm_occupancy": "unknown: no validated block residency timeline",
            "idle_stageable_work": "unknown: no validated GPU-idle and actionable-work join",
        },
        "interpretation": "This validates event linkage, not an agentic performance gain or a percentage of avoidable work.",
    }
