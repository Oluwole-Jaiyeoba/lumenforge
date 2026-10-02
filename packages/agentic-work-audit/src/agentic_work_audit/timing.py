"""Backend-neutral timing analysis for paired early/late KV preparation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from .events import AuditEvent


def _first(rows: list[AuditEvent], kind: str, request: str = "") -> AuditEvent | None:
    return next((row for row in rows if row.kind == kind and (not request or row.request_id == request)), None)


def _ms(start: AuditEvent | None, end: AuditEvent | None) -> float | None:
    return round((end.ts_ns - start.ts_ns) / 1_000_000, 3) if start and end else None


def analyze_timing(events: Iterable[AuditEvent], specs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(events, key=lambda row: row.ts_ns)
    runtime = _first(rows, "runtime_hooks")
    failures: list[str] = []
    if not runtime or (runtime.evidence.get("backend_version"), runtime.evidence.get("adapter")) != (
        "0.5.10.post1", "v0510"
    ):
        failures.append("pinned backend hook installation proof is missing")
    elif runtime.evidence.get("missing_required_hooks"):
        failures.append("required backend hooks are missing")
    if any(row.kind == "hook_error" for row in rows):
        failures.append("backend hook error occurred")

    by_session: dict[str, list[AuditEvent]] = defaultdict(list)
    for row in rows:
        if row.session_id:
            by_session[row.session_id].append(row)
    cases: list[dict[str, Any]] = []
    for spec in specs:
        session = str(spec["session_id"])
        condition = str(spec["condition"])
        case_rows = by_session[session]
        timeline = {kind: _first(case_rows, kind) for kind in (
            "initial_sent", "device_evict_proof", "host_resident_proof", "tool_start", "tool_end",
            "load_requested", "load_accepted", "replay_sent", "replay_first_token", "replay_finished",
            "tool_2_start", "tool_2_end", "replay_2_sent", "replay_2_first_token", "replay_2_finished",
        )}
        accepted = timeline["load_accepted"]
        load_id = accepted.evidence.get("load_id") if accepted else None
        completed = next((row for row in case_rows if row.kind == "load_complete"
                          and row.evidence.get("load_id") == load_id), None) if load_id else None
        replay = timeline["replay_sent"]
        first_token = timeline["replay_first_token"]
        replay_2 = timeline["replay_2_sent"]
        first_token_2 = timeline["replay_2_first_token"]
        matches = [row for row in case_rows if row.kind == "cache_match" and replay and first_token
                   and row.request_id == replay.request_id
                   and replay.ts_ns <= row.ts_ns <= first_token.ts_ns
                   and int(row.evidence.get("cached_prefix_tokens") or 0) > 0]
        second_matches = [row for row in case_rows if row.kind == "cache_match" and replay_2 and first_token_2
                          and row.request_id == replay_2.request_id
                          and replay_2.ts_ns <= row.ts_ns <= first_token_2.ts_ns
                          and int(row.evidence.get("cached_prefix_tokens") or 0) > 0]
        case_failures: list[str] = []
        if condition not in {"early", "late"}:
            case_failures.append("unknown timing condition")
        if not all(timeline.values()) or not completed:
            case_failures.append("required timeline or native CUDA completion observation is missing")
        else:
            t = timeline
            if not (t["initial_sent"].ts_ns < t["device_evict_proof"].ts_ns
                    <= t["host_resident_proof"].ts_ns < t["tool_start"].ts_ns
                    < t["tool_end"].ts_ns <= t["replay_sent"].ts_ns
                    <= t["replay_first_token"].ts_ns <= t["replay_finished"].ts_ns
                    <= t["tool_2_start"].ts_ns < t["tool_2_end"].ts_ns
                    <= t["replay_2_sent"].ts_ns <= t["replay_2_first_token"].ts_ns
                    <= t["replay_2_finished"].ts_ns):
                case_failures.append("tool, eviction, or replay events are out of order")
            if condition == "early" and not (t["tool_start"].ts_ns <= t["load_requested"].ts_ns
                                                  < t["tool_end"].ts_ns):
                case_failures.append("early load was not requested during the tool wait")
            if condition == "late" and t["load_requested"].ts_ns < t["tool_end"].ts_ns:
                case_failures.append("late load was requested before the tool finished")
            if t["load_requested"].ts_ns > t["load_accepted"].ts_ns or (
                t["load_accepted"].ts_ns > t["replay_sent"].ts_ns
            ):
                case_failures.append("load acceptance was not before replay submission")
            if completed.ts_ns < t["load_requested"].ts_ns:
                case_failures.append("native load completion predates its request")
        if not matches or not second_matches:
            case_failures.append("one or both replays lack a request-linked prefix match")
        if not any(row.kind == "layer_copy" for row in case_rows):
            case_failures.append("native per-layer copy evidence is missing")
        if accepted and completed and int(accepted.evidence.get("loaded_tokens") or 0) != int(
            completed.evidence.get("loaded_tokens") or 0
        ):
            case_failures.append("accepted and completed native load token counts differ")
        failures.extend(f"{session}: {reason}" for reason in case_failures)
        tool_end = timeline["tool_end"]
        cases.append({
            "session_id": session, "pair": spec["pair"], "condition": condition,
            "status": "validated" if not case_failures else "failed", "failures": case_failures,
            "prompt_words": spec.get("prompt_words"),
            "initial_latency_ms": (spec.get("initial") or {}).get("total_latency_ms"),
            "host_tokens": timeline["host_resident_proof"].evidence.get("host_tokens")
            if timeline["host_resident_proof"] else None,
            "loaded_tokens": completed.evidence.get("loaded_tokens") if completed else None,
            "native_cuda_load_ms": completed.evidence.get("cuda_elapsed_ms") if completed else None,
            "load_control_duration_ms": (spec.get("load_acceptance") or {}).get("control_duration_ms"),
            "load_id": load_id,
            "tool_wait_ms": _ms(timeline["tool_start"], tool_end),
            "load_request_from_wait_start_ms": _ms(timeline["tool_start"], timeline["load_requested"]),
            "load_request_from_due_ms": _ms(tool_end, timeline["load_requested"]),
            "completion_observed_before_due": completed.ts_ns <= tool_end.ts_ns
            if completed and tool_end else None,
            "completion_observed_from_due_ms": _ms(tool_end, completed),
            "submission_after_due_ms": _ms(tool_end, replay),
            "first_token_after_due_ms": _ms(tool_end, first_token),
            "replay_ttft_ms": _ms(replay, first_token),
            "replay_completion_after_due_ms": _ms(tool_end, timeline["replay_finished"]),
            "second_replay_ttft_ms": _ms(replay_2, first_token_2),
            "task_latency_ms": _ms(timeline["initial_sent"], timeline["replay_2_finished"]),
            "post_tool_start_duration_ms": _ms(timeline["tool_start"], timeline["replay_2_finished"]),
            "replay_cache_matches": len(matches), "second_replay_cache_matches": len(second_matches),
            "timing_label": "candidate_late_load" if condition == "late" else "prepared_during_wait",
        })

    pairs: list[dict[str, Any]] = []
    for pair in sorted({case["pair"] for case in cases}):
        selected = {case["condition"]: case for case in cases if case["pair"] == pair}
        early, late = selected.get("early"), selected.get("late")
        reasons: list[str] = []
        if not early or not late:
            reasons.append("one timing condition is missing")
            failures.append(f"pair {pair}: one timing condition is missing")
        else:
            if early["status"] != "validated" or late["status"] != "validated":
                reasons.append("one case failed its evidence gate")
            if early["loaded_tokens"] != late["loaded_tokens"] or early["host_tokens"] != late["host_tokens"]:
                reasons.append("host residency or loaded-token counts differ")
            if early["prompt_words"] != late["prompt_words"]:
                reasons.append("prompt word counts differ")
            a, b = early["initial_latency_ms"], late["initial_latency_ms"]
            if a is None or b is None or abs(a - b) > max(1000, 0.3 * min(a, b)):
                reasons.append("initial request latency drift suggests warmup or backend-state bias")
        comparable = not reasons
        task_reasons = list(reasons)
        if early and late:
            a, b = early["second_replay_ttft_ms"], late["second_replay_ttft_ms"]
            if a is None or b is None or abs(a - b) > max(100, 0.3 * min(a, b)):
                task_reasons.append("second-replay TTFT drift makes full-task timing incomparable")
        task_comparable = not task_reasons
        pairs.append({
            "pair": pair, "comparable": comparable, "comparability_reasons": reasons,
            "task_comparable": task_comparable, "task_comparability_reasons": task_reasons,
            "late_minus_early_first_token_after_due_ms": round(
                late["first_token_after_due_ms"] - early["first_token_after_due_ms"], 3
            ) if comparable else None,
            "late_minus_early_submission_after_due_ms": round(
                late["submission_after_due_ms"] - early["submission_after_due_ms"], 3
            ) if comparable else None,
            "late_minus_early_replay_ttft_ms": round(
                late["replay_ttft_ms"] - early["replay_ttft_ms"], 3
            ) if comparable else None,
            "late_minus_early_task_latency_ms": round(
                late["task_latency_ms"] - early["task_latency_ms"], 3
            ) if task_comparable else None,
            "late_minus_early_post_tool_duration_ms": round(
                late["post_tool_start_duration_ms"] - early["post_tool_start_duration_ms"], 3
            ) if task_comparable else None,
        })
    return {
        "schema": "agentic_work_audit.timing.v1", "status": "validated" if not failures else "failed",
        "failures": failures, "event_counts": dict(sorted(Counter(row.kind for row in rows).items())),
        "cases": cases, "pairs": pairs,
        "interpretation": (
            "Late preparation is a candidate timing opportunity, not automatically a harmful decision. "
            "Native CUDA completion is observed when polled, so a post-due observation is an upper bound "
            "on completion time. Paired differences are exploratory until overhead, repeated order, "
            "token shape, and competing work are controlled."
        ),
    }
