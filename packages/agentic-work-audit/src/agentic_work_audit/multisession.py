"""Evidence gate for a small, concurrent, equal-importance cache timeline."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .events import AuditEvent


def analyze_multisession(
    events: Iterable[AuditEvent], run_id: str, *, expected_runtime: tuple[str, str]
) -> dict[str, Any]:
    rows = sorted(events, key=lambda row: row.ts_ns)
    sessions = {label: f"{run_id}-{label}" for label in ("long", "short", "ends")}
    groups = {label: [row for row in rows if row.session_id == session]
              for label, session in sessions.items()}

    def first(label: str, kind: str) -> AuditEvent | None:
        return next((row for row in groups[label] if row.kind == kind), None)

    failures: list[str] = []
    hooks = next((row for row in rows if row.kind == "runtime_hooks"), None)
    if not hooks or (hooks.evidence.get("backend_version"), hooks.evidence.get("adapter")) != expected_runtime:
        failures.append("pinned backend hook proof missing or mismatched")
    if any(row.kind == "hook_error" for row in rows):
        failures.append("backend hook error recorded")
    capacity = first("long", "capacity_policy")
    eviction = first("long", "device_evict_proof")
    host = first("long", "host_resident_proof")
    load = first("long", "load_complete")
    if not all((capacity, eviction, host, load)):
        failures.append("capacity-policy eviction, host residency, or native load proof missing")
    elif not (capacity.ts_ns <= eviction.ts_ns <= host.ts_ns <= load.ts_ns):
        failures.append("capacity-policy and native load timestamps out of order")
    if capacity and (capacity.evidence.get("active_prefix_budget") != 2 or
                     capacity.evidence.get("action") != "explicit_evict_long_prefix"):
        failures.append("unexpected capacity-policy action or budget")
    if not any(row.kind == "layer_copy" for row in groups["long"]):
        failures.append("identity-linked native layer copy missing")

    results: dict[str, dict[str, Any]] = {}
    waits: dict[str, tuple[int, int]] = {}
    for label in ("short", "long"):
        initial_sent, initial_finished = first(label, "initial_sent"), first(label, "initial_finished")
        start, end = first(label, "tool_start"), first(label, "tool_end")
        replay, token = first(label, "replay_sent"), first(label, "replay_first_token")
        if not initial_sent or not initial_finished or not start or not (
            initial_sent.ts_ns <= initial_finished.ts_ns <= start.ts_ns
        ):
            failures.append(f"{label}: initial request was not completed before tool wait")
        if start and start.evidence.get("importance") != "equal":
            failures.append(f"{label}: frontend importance was not equal")
        if not all((start, end, replay, token)) or not (
            start.ts_ns < end.ts_ns <= replay.ts_ns <= token.ts_ns
        ):
            failures.append(f"{label}: tool and replay timeline incomplete or unordered")
            continue
        waits[label] = (start.ts_ns, end.ts_ns)
        matches = [row for row in groups[label] if row.kind == "cache_match"
                   and row.request_id == replay.request_id
                   and int(row.evidence.get("cached_prefix_tokens") or 0) > 0]
        if not matches:
            failures.append(f"{label}: request-linked replay cache match missing")
        results[label] = {
            "session_id": sessions[label],
            "expected_tool_wait_ms": start.evidence.get("expected_ms"),
            "observed_tool_wait_ms": round((end.ts_ns - start.ts_ns) / 1e6, 3),
            "submission_after_tool_ms": round((replay.ts_ns - end.ts_ns) / 1e6, 3),
            "first_token_after_tool_ms": round((token.ts_ns - end.ts_ns) / 1e6, 3),
            "replay_ttft_ms": round((token.ts_ns - replay.ts_ns) / 1e6, 3),
            "cached_prefix_tokens": max((int(row.evidence.get("cached_prefix_tokens") or 0)
                                         for row in matches), default=0),
        }
    end_event = first("ends", "session_end")
    if not end_event or not first("ends", "initial_finished") or not first("ends", "initial_sent"):
        failures.append("ending session has no initial completion or end event")
    if any(row.kind.startswith("replay") for row in groups["ends"]):
        failures.append("ending session unexpectedly replayed")
    if len(waits) == 2 and not (
        waits["long"][0] < waits["short"][1] and waits["short"][0] < waits["long"][1]
    ):
        failures.append("tool waits did not overlap")
    if len(waits) == 2 and waits["short"][1] >= waits["long"][1]:
        failures.append("short tool call did not return first")

    second = first("long", "replay_2_sent")
    second_token = first("long", "replay_2_first_token")
    second_matches = [row for row in groups["long"] if second and row.kind == "cache_match"
                      and row.request_id == second.request_id
                      and int(row.evidence.get("cached_prefix_tokens") or 0) > 0]
    if not second or not second_token or not second_matches:
        failures.append("long session has no request-linked second-replay reuse proof")
    elif "long" in results:
        results["long"]["second_replay_cached_prefix_tokens"] = max(
            int(row.evidence.get("cached_prefix_tokens") or 0) for row in second_matches
        )

    long_end = first("long", "tool_end")
    requested = first("long", "load_requested")
    accepted = first("long", "load_accepted")
    late_request = bool(host and long_end and requested and
                        host.ts_ns < long_end.ts_ns <= requested.ts_ns)
    results.setdefault("long", {})["load_requested_after_tool_return"] = late_request
    for field, row in (("load_request_after_tool_ms", requested),
                       ("load_acceptance_after_tool_ms", accepted),
                       ("load_completion_observed_after_tool_ms", load)):
        results["long"][field] = (
            round((row.ts_ns - long_end.ts_ns) / 1e6, 3) if row and long_end else None
        )
    results["long"]["native_load_accepted_before_replay"] = bool(
        accepted and first("long", "replay_sent") and
        accepted.ts_ns <= first("long", "replay_sent").ts_ns
    )
    backend_prefill_starts = sum(row.kind == "cache_match" and row.source == "backend_v0510"
                                 for row in rows)
    return {
        "schema": "agentic_work_audit.multisession.v1",
        "run_id": run_id,
        "status": "validated" if not failures else "failed",
        "failures": failures,
        "setup": {
            "session_count": 3, "equal_importance": True,
            "capacity_policy": "explicit two-prefix budget; not natural memory-pressure eviction",
            "tool_waits_overlap": len(waits) == 2 and not any(
                item == "tool waits did not overlap" for item in failures
            ),
            "ended_session_id": sessions["ends"],
        },
        "sessions": results,
        "event_counts": dict(Counter(row.kind for row in rows)),
        "backend_cache_match_observations": backend_prefill_starts,
        "observations": [
            "Long session was explicitly evicted to host and later underwent a native load"
            if host and load else "Host eviction/load lineage incomplete",
            "The short and long tool waits overlapped" if len(waits) == 2 else
            "Concurrent tool wait proof incomplete",
            "One session ended without replay" if end_event else "Session end not proved",
        ],
        "plausibly_mistimed": [
            "Host-backed load was not requested until the long tool call returned; "
            "an earlier request was possible in this trace, but its effect under the same load is unmeasured"
        ] if late_request else [],
        "avoidable_work": "unknown: no same-capacity, same-load counterfactual was run",
        "limitations": [
            "Explicit test-budget eviction is not evidence of organic backend capacity pressure.",
            "The three initial sessions are candidates; only the long session has direct eviction and host-residency proof.",
            "A cache match is not proof that model kernels consumed those exact KV slots.",
            "Backend cache-match events indicate request activity, not measured GPU occupancy.",
        ],
    }
