"""Evidence gate for a small, concurrent, equal-importance cache timeline."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable

from .events import AuditEvent


def analyze_multisession(
    events: Iterable[AuditEvent], run_id: str, *, expected_runtime: tuple[str, str],
    condition: str = "late_nonblocking", require_end_eviction: bool = False,
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
        finished = first(label, "replay_finished")
        if not initial_sent or not initial_finished or not start or not (
            initial_sent.ts_ns <= initial_finished.ts_ns <= start.ts_ns
        ):
            failures.append(f"{label}: initial request was not completed before tool wait")
        if start and start.evidence.get("importance") != "equal":
            failures.append(f"{label}: frontend importance was not equal")
        if not all((start, end, replay, token, finished)) or not (
            start.ts_ns < end.ts_ns <= replay.ts_ns <= token.ts_ns <= finished.ts_ns
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
            "completion_after_tool_ms": round((finished.ts_ns - end.ts_ns) / 1e6, 3),
            "replay_total_ms": round((finished.ts_ns - replay.ts_ns) / 1e6, 3),
            "replay_ttft_ms": round((token.ts_ns - replay.ts_ns) / 1e6, 3),
            "cached_prefix_tokens": max((int(row.evidence.get("cached_prefix_tokens") or 0)
                                         for row in matches), default=0),
        }
    end_event = first("ends", "session_end")
    if not end_event or not first("ends", "initial_finished") or not first("ends", "initial_sent"):
        failures.append("ending session has no initial completion or end event")
    if any(row.kind.startswith("replay") for row in groups["ends"]):
        failures.append("ending session unexpectedly replayed")
    ended_eviction = first("ends", "ended_prefix_evict_proof")
    if require_end_eviction and (not ended_eviction or not end_event or
                                 ended_eviction.ts_ns < end_event.ts_ns or
                                 int(ended_eviction.evidence.get("evicted_tokens") or 0) <= 0):
        failures.append("ending session's device prefix was not proved released")
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
    short_replay = first("short", "replay_sent")
    short_finished = first("short", "replay_finished")
    load_overlapped_short_request = bool(requested and load and short_replay and short_finished and
                                         requested.ts_ns < short_finished.ts_ns and
                                         load.ts_ns > short_replay.ts_ns)
    controller_decisions = [row for row in groups["long"] if row.kind == "controller_prepare_decision"]
    controller_outcome = first("long", "controller_prepare_outcome")
    controller_action = controller_decisions[-1].evidence.get("action") if controller_decisions else None
    if condition == "controller_window":
        initial_hold_valid = (
            len(controller_decisions) == 1 or
            (len(controller_decisions) == 2 and short_finished and
             controller_decisions[0].ts_ns <= short_finished.ts_ns and
             controller_decisions[0].evidence.get("action") == "hold" and
             controller_decisions[0].evidence.get("reason") == "blocking_replay_not_finished")
        )
        if (not initial_hold_valid or controller_action not in ("load", "defer") or
                not controller_outcome or not short_finished or not long_end or
                controller_decisions[-1].ts_ns < short_finished.ts_ns or
                controller_decisions[-1].ts_ns >= long_end.ts_ns):
            failures.append("controller observation, decision, or outcome proof missing")
        elif controller_action == "load" and controller_outcome.evidence.get("outcome") != "good_window":
            failures.append("controller load did not finish within the tool-return window")
        elif controller_action == "defer" and controller_outcome.evidence.get("outcome") != "deferred_to_tool_return":
            failures.append("controller defer outcome missing")
        if controller_action == "load" and (not requested or
                                             requested.ts_ns < controller_decisions[-1].ts_ns):
            failures.append("controller load did not follow its decision")
    if condition in ("early", "post_short") or (condition == "controller_window" and controller_action == "load"):
        if not all((host, ended_eviction, requested, accepted, load, long_end)) or not (
            host.ts_ns < ended_eviction.ts_ns <= requested.ts_ns <= accepted.ts_ns <=
            load.ts_ns < long_end.ts_ns
        ):
            failures.append(f"{condition} native load did not finish between slot release and tool return")
        if condition in ("post_short", "controller_window") and (not short_finished or not requested or
                                          short_finished.ts_ns > requested.ts_ns or
                                          load_overlapped_short_request):
            failures.append("post-short/controller load did not start after the short replay finished")
    elif condition == "late_nonblocking" or (condition == "controller_window" and controller_action == "defer"):
        if not all((host, requested, long_end)) or not (
            host.ts_ns < long_end.ts_ns <= requested.ts_ns
        ):
            failures.append("late native load was not requested after tool return")
    else:
        failures.append(f"unsupported load timing: {condition}")
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
    initial_times = [row.ts_ns for label in sessions for row in groups[label]
                     if row.kind == "initial_sent"]
    initial_finish_times = [row.ts_ns for label in sessions for row in groups[label]
                            if row.kind == "initial_finished"]
    finish_times = [row.ts_ns for label in ("short", "long") for row in groups[label]
                    if row.kind in ("replay_finished", "replay_2_finished")]
    workflow_makespan_ms = (round((max(finish_times) - min(initial_times)) / 1e6, 3)
                            if initial_times and finish_times else None)
    initial_phase_ms = (round((max(initial_finish_times) - min(initial_times)) / 1e6, 3)
                        if initial_times and initial_finish_times else None)
    backend_prefill_starts = sum(row.kind == "cache_match" and row.source == "backend_v0510"
                                 for row in rows)
    return {
        "schema": "agentic_work_audit.multisession.v1",
        "run_id": run_id,
        "load_timing": condition,
        "controller_decisions": [row.evidence for row in controller_decisions] if condition == "controller_window" else [],
        "controller_outcome": controller_outcome.evidence if condition == "controller_window" and controller_outcome else None,
        "status": "validated" if not failures else "failed",
        "failures": failures,
        "setup": {
            "session_count": 3, "equal_importance": True,
            "capacity_policy": "explicit two-prefix logical budget; not physical occupancy proof",
            "tool_waits_overlap": len(waits) == 2 and not any(
                item == "tool waits did not overlap" for item in failures
            ),
            "ended_session_id": sessions["ends"],
        },
        "sessions": results,
        "initial_phase_ms": initial_phase_ms,
        "workflow_makespan_ms": workflow_makespan_ms,
        "load_overlapped_short_request": load_overlapped_short_request,
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
            "The two-prefix budget is enforced by explicit actions, not measured physical GPU occupancy.",
            "The three initial sessions are candidates; only the long session has direct eviction and host-residency proof.",
            "A cache match is not proof that model kernels consumed those exact KV slots.",
            "Backend cache-match events indicate request activity, not measured GPU occupancy.",
        ],
    }


def compare_multisession_pairs(cases: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Compare policy outcomes only when both same-shape cases pass evidence gates."""
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    failures: list[str] = []
    for case in cases:
        pair = case["pair"]
        condition = case["load_timing"]
        if condition in by_pair.setdefault(pair, {}):
            failures.append(f"pair {pair}: duplicate {condition} case")
        by_pair[pair][condition] = case
    pairs: list[dict[str, Any]] = []
    for number, modes in sorted(by_pair.items()):
        early, late = modes.get("early"), modes.get("late_nonblocking")
        reasons = []
        if not early or not late:
            reasons.append("missing early or late case")
        elif early["status"] != "validated" or late["status"] != "validated":
            reasons.append("one or both case evidence gates failed")
        else:
            if not early["load_overlapped_short_request"]:
                reasons.append("early native load did not overlap the short request lifetime")
            initial_delta = abs(early["initial_phase_ms"] - late["initial_phase_ms"])
            if initial_delta > max(1000, 0.2 * min(early["initial_phase_ms"], late["initial_phase_ms"])):
                reasons.append(f"initial phases differ by {initial_delta:.1f} ms")
            for label in ("short", "long"):
                wait_delta = abs(early["sessions"][label]["observed_tool_wait_ms"] -
                                 late["sessions"][label]["observed_tool_wait_ms"])
                if wait_delta > 100:
                    reasons.append(f"{label} tool waits differ by {wait_delta:.1f} ms")
        pair_result: dict[str, Any] = {
            "pair": number, "comparable": not reasons, "reasons": reasons,
            "early_case_id": early.get("run_id") if early else None,
            "late_case_id": late.get("run_id") if late else None,
        }
        if reasons:
            failures.append(f"pair {number}: {', '.join(reasons)}")
        if not reasons and early and late:
            es, ls = early["sessions"], late["sessions"]
            pair_result.update({
                "long_due_to_token_saved_ms": round(
                    ls["long"]["first_token_after_tool_ms"] -
                    es["long"]["first_token_after_tool_ms"], 3),
                "short_due_to_token_change_ms": round(
                    es["short"]["first_token_after_tool_ms"] -
                    ls["short"]["first_token_after_tool_ms"], 3),
                "short_due_to_finish_change_ms": round(
                    es["short"]["completion_after_tool_ms"] -
                    ls["short"]["completion_after_tool_ms"], 3),
                "workflow_makespan_saved_ms": round(
                    late["workflow_makespan_ms"] - early["workflow_makespan_ms"], 3),
                "early_long_due_to_token_ms": es["long"]["first_token_after_tool_ms"],
                "late_long_due_to_token_ms": ls["long"]["first_token_after_tool_ms"],
                "early_short_due_to_finish_ms": es["short"]["completion_after_tool_ms"],
                "late_short_due_to_finish_ms": ls["short"]["completion_after_tool_ms"],
                "early_initial_phase_ms": early["initial_phase_ms"],
                "late_initial_phase_ms": late["initial_phase_ms"],
            })
        pairs.append(pair_result)
    comparable = [pair for pair in pairs if pair["comparable"]]
    if not comparable:
        failures.append("no comparable pairs")
    return {
        "schema": "agentic_work_audit.multisession_comparison.v1",
        "run_id": run_id,
        "status": "validated" if not failures and len(comparable) == len(pairs) else "failed",
        "failures": failures,
        "pairs": pairs,
        "comparable_pairs": len(comparable),
        "median_long_due_to_token_saved_ms": (
            round(median(pair["long_due_to_token_saved_ms"] for pair in comparable), 3)
            if comparable else None),
        "median_short_due_to_finish_change_ms": (
            round(median(pair["short_due_to_finish_change_ms"] for pair in comparable), 3)
            if comparable else None),
        "median_workflow_makespan_saved_ms": (
            round(median(pair["workflow_makespan_saved_ms"] for pair in comparable), 3)
            if comparable else None),
        "interpretation": (
            "Positive long/workflow savings favor early loading; positive short change is harm. "
            "These are matched synthetic policy outcomes, not proof of naturally avoidable work."
        ),
        "limitations": [
            "The two-prefix cap is a test policy enforced by explicit evictions, not measured physical occupancy.",
            "Session-specific prompts have equal shape but are not byte-identical.",
            "The comparison is concurrent but synthetic and does not establish a production effect.",
        ],
    }


def compare_multisession_windows(cases: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Gate three timing policies together and expose each session's raw outcome."""
    baseline = compare_multisession_pairs(
        [case for case in cases if case["load_timing"] in ("early", "late_nonblocking")], run_id
    )
    failures = list(baseline["failures"])
    post_by_pair: dict[int, dict[str, Any]] = {}
    for case in cases:
        if case["load_timing"] == "post_short":
            if case["pair"] in post_by_pair:
                failures.append(f"pair {case['pair']}: duplicate post-short case")
            post_by_pair[case["pair"]] = case
    pairs: list[dict[str, Any]] = []
    for pair in baseline["pairs"]:
        number = pair["pair"]
        post = post_by_pair.get(number)
        early = next((case for case in cases if case["pair"] == number and
                      case["load_timing"] == "early"), None)
        late = next((case for case in cases if case["pair"] == number and
                     case["load_timing"] == "late_nonblocking"), None)
        reasons = list(pair["reasons"])
        if not post:
            reasons.append("missing post-short case")
        elif post["status"] != "validated":
            reasons.append("post-short evidence gate failed")
        elif not reasons and early and late:
            for reference in (early, late):
                initial_delta = abs(post["initial_phase_ms"] - reference["initial_phase_ms"])
                if initial_delta > max(1000, 0.2 * min(post["initial_phase_ms"], reference["initial_phase_ms"])):
                    reasons.append(f"post-short initial phase differs from {reference['load_timing']} "
                                   f"by {initial_delta:.1f} ms")
                for label in ("short", "long"):
                    wait_delta = abs(post["sessions"][label]["observed_tool_wait_ms"] -
                                     reference["sessions"][label]["observed_tool_wait_ms"])
                    if wait_delta > 100:
                        reasons.append(f"post-short {label} wait differs from "
                                       f"{reference['load_timing']} by {wait_delta:.1f} ms")
        result: dict[str, Any] = {
            "pair": number, "comparable": not reasons, "reasons": reasons,
            "late_case_id": late["run_id"] if late else None,
            "early_case_id": early["run_id"] if early else None,
            "post_short_case_id": post["run_id"] if post else None,
        }
        if reasons:
            failures.append(f"pair {number}: {', '.join(reasons)}")
        elif post and early and late:
            for mode, case in (("late", late), ("early", early), ("post_short", post)):
                result[f"{mode}_long_due_to_token_ms"] = case["sessions"]["long"]["first_token_after_tool_ms"]
                result[f"{mode}_short_due_to_finish_ms"] = case["sessions"]["short"]["completion_after_tool_ms"]
                result[f"{mode}_workflow_makespan_ms"] = case["workflow_makespan_ms"]
                result[f"{mode}_initial_phase_ms"] = case["initial_phase_ms"]
            result["post_vs_late_long_saved_ms"] = round(
                late["sessions"]["long"]["first_token_after_tool_ms"] -
                post["sessions"]["long"]["first_token_after_tool_ms"], 3)
            result["post_vs_early_short_saved_ms"] = round(
                early["sessions"]["short"]["completion_after_tool_ms"] -
                post["sessions"]["short"]["completion_after_tool_ms"], 3)
            result["post_vs_late_workflow_saved_ms"] = round(
                late["workflow_makespan_ms"] - post["workflow_makespan_ms"], 3)
        pairs.append(result)
    extra = sorted(set(post_by_pair) - {pair["pair"] for pair in pairs})
    if extra:
        failures.append(f"unexpected post-short pairs: {extra}")
    comparable = [pair for pair in pairs if pair["comparable"]]
    if not comparable:
        failures.append("no comparable three-mode pairs")
    return {
        "schema": "agentic_work_audit.multisession_window.v1",
        "run_id": run_id,
        "status": "validated" if not failures and len(comparable) == len(pairs) else "failed",
        "failures": failures,
        "pairs": pairs,
        "comparable_pairs": len(comparable),
        "median_post_vs_late_long_saved_ms": (
            round(median(pair["post_vs_late_long_saved_ms"] for pair in comparable), 3)
            if comparable else None),
        "median_post_vs_early_short_saved_ms": (
            round(median(pair["post_vs_early_short_saved_ms"] for pair in comparable), 3)
            if comparable else None),
        "median_post_vs_late_workflow_saved_ms": (
            round(median(pair["post_vs_late_workflow_saved_ms"] for pair in comparable), 3)
            if comparable else None),
        "limitations": [
            "The after-short policy uses an observed client completion event, not a production prediction.",
            "The two-prefix cap is enforced by explicit evictions, not measured physical occupancy.",
            "Session-specific prompts have equal shape but are not byte-identical.",
            "Request-level overlap does not establish GPU kernel or HBM bandwidth contention.",
        ],
    }


def compare_controller_windows(cases: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    """Extend the matched timing study with an evidence-gated controller choice."""
    result = compare_multisession_windows(
        [case for case in cases if case["load_timing"] != "controller_window"], run_id
    )
    failures = list(result["failures"])
    controllers: dict[int, dict[str, Any]] = {}
    for case in cases:
        if case["load_timing"] != "controller_window":
            continue
        if case["pair"] in controllers:
            failures.append(f"pair {case['pair']}: duplicate controller case")
        controllers[case["pair"]] = case
    for pair in result["pairs"]:
        number = pair["pair"]
        controller = controllers.get(number)
        references = [case for case in cases if case["pair"] == number and
                      case["load_timing"] != "controller_window"]
        reasons = list(pair["reasons"])
        if not controller or controller["status"] != "validated":
            reasons.append("missing or unvalidated controller case")
        elif not reasons:
            for reference in references:
                initial_delta = abs(controller["initial_phase_ms"] - reference["initial_phase_ms"])
                if initial_delta > max(1000, 0.2 * min(controller["initial_phase_ms"],
                                                        reference["initial_phase_ms"])):
                    reasons.append(f"controller initial phase differs from {reference['load_timing']}")
                for label in ("short", "long"):
                    wait_delta = abs(controller["sessions"][label]["observed_tool_wait_ms"] -
                                     reference["sessions"][label]["observed_tool_wait_ms"])
                    if wait_delta > 100:
                        reasons.append(f"controller {label} wait differs from {reference['load_timing']}")
        pair["reasons"] = reasons
        pair["comparable"] = not reasons
        pair["controller_case_id"] = controller["run_id"] if controller else None
        if reasons:
            failures.append(f"pair {number}: {', '.join(reasons)}")
        elif controller:
            pair["controller_action"] = controller["controller_decisions"][-1]["action"]
            pair["controller_long_due_to_token_ms"] = controller["sessions"]["long"]["first_token_after_tool_ms"]
            pair["controller_short_due_to_finish_ms"] = controller["sessions"]["short"]["completion_after_tool_ms"]
            pair["controller_workflow_makespan_ms"] = controller["workflow_makespan_ms"]
            pair["controller_vs_late_long_saved_ms"] = round(
                pair["late_long_due_to_token_ms"] - pair["controller_long_due_to_token_ms"], 3)
            pair["controller_vs_late_workflow_saved_ms"] = round(
                pair["late_workflow_makespan_ms"] - pair["controller_workflow_makespan_ms"], 3)
    extra = sorted(set(controllers) - {pair["pair"] for pair in result["pairs"]})
    if extra:
        failures.append(f"unexpected controller pairs: {extra}")
    comparable = [pair for pair in result["pairs"] if pair["comparable"]]
    if not comparable:
        failures.append("no comparable four-mode pairs")
    result.update({
        "schema": "agentic_work_audit.controller_window.v1",
        "status": "validated" if not failures else "failed",
        "failures": failures,
        "comparable_pairs": len(comparable),
        "median_controller_vs_late_long_saved_ms": round(median(
            pair["controller_vs_late_long_saved_ms"] for pair in comparable), 3) if comparable else None,
        "median_controller_vs_late_workflow_saved_ms": round(median(
            pair["controller_vs_late_workflow_saved_ms"] for pair in comparable), 3) if comparable else None,
        "limitations": [
            "Controller used a fixed predeclared load-time estimate, not exact future runtime.",
            *result["limitations"],
        ],
    })
    return result
