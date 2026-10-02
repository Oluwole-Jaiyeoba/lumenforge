"""Build the compact, chronological KV lifecycle audit from saved evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

CACHE_SIZE_FIELD = "hicache_size_gb"


def _esc(value: object) -> str:
    return html.escape(str(value if value is not None else "not recorded"), quote=True)


def _ms(value: object) -> str:
    return f"{value:.1f} ms" if isinstance(value, (int, float)) else "not recorded"


def _direction(value: object, positive: str, negative: str) -> str:
    if not isinstance(value, (int, float)):
        return "not recorded"
    return f"{abs(value):.1f} ms {positive if value >= 0 else negative}"


def _first_request_ns(path: Path) -> int | None:
    for filename in ("harness_events.jsonl", "normalized_events.jsonl"):
        events = path.with_name(filename)
        if not events.exists():
            continue
        with events.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("kind") == "initial_sent" and isinstance(event.get("ts_ns"), int):
                    return event["ts_ns"]
    return None


def _time(summary: dict) -> tuple[int, str, str, str]:
    started_ns = summary.get("_started_ns")
    if isinstance(started_ns, int):
        instant = datetime.fromtimestamp(started_ns / 1_000_000_000, timezone.utc)
        return started_ns, instant.strftime("%Y-%m-%d"), instant.strftime("%H:%M:%S"), "First request (UTC)"
    completed_ms = (summary.get("_manifest") or {}).get("created_at_ms")
    if isinstance(completed_ms, int):
        instant = datetime.fromtimestamp(completed_ms / 1000, timezone.utc)
        return completed_ms * 1_000_000, instant.strftime("%Y-%m-%d"), instant.strftime("%H:%M:%S"), "Manifest completion time (UTC); start unavailable"
    return -1, "not recorded", "not recorded", "No timestamp in saved evidence"


def _links(path: Path, summary: dict) -> str:
    labels = (
        ("summary.json", "Summary JSON"), ("run_manifest.json", "Run manifest"),
        ("instrumentation_audit.json", "Trace gate"), ("block_audit.json", "Block audit"),
        ("instrumentation_analysis.json", "Slot analysis"),
        ("normalized_events.jsonl", "Timeline"), ("harness_events.jsonl", "Harness events"),
        ("backend_trace.jsonl.gz", "Raw trace"),
    )
    files = summary.get("_files") or {"summary.json"}
    return " · ".join(f'<a href="{_esc(path.with_name(name).as_posix())}">{label}</a>'
                      for name, label in labels if name in files)


def _setup(summary: dict, timing: bool) -> tuple[str, str]:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    order = " → ".join(map(str, workload.get("cases") or [])) or "not recorded"
    profile = summary.get("_trace_profile") or next(
        (name for name in manifest.get("enabled_instrumentation") or [] if name.startswith("kv_lifecycle")),
        None,
    )
    if summary.get("schema") in ("agentic_work_audit.multisession_comparison.v1",
                                 "agentic_work_audit.multisession_window.v1"):
        window = summary.get("schema") == "agentic_work_audit.multisession_window.v1"
        brief = (f"3 equal-importance sessions · {_esc(workload.get('pairs'))} measured pair(s) · "
                 f"{_esc(workload.get('short_wait_ms'))} / {_esc(workload.get('long_wait_ms'))} ms waits")
        detail = (
            f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
            f"{_esc(manifest.get('model'))}; SGLang {_esc(manifest.get('backend_version'))}; "
            f"trace {_esc(profile)}. Each case started three equal-importance sessions; "
            "two waited for tools and one ended. The long prefix was explicitly evicted to host. "
            "The ended session's prefix was released in every mode before load-back, "
            "enforcing a logical two-prefix budget. Only load timing changed: late control "
            "submitted at tool return without blocking replay, or early control submitted "
            f"at {_esc(workload.get('early_at_ms'))} ms during the long tool wait. "
            + ("The third mode requested load immediately after observing the short replay finish, "
               "before long tool return. " if window else "") +
            f"{_esc(workload.get('warmup_pairs'))} warmup pair(s) were excluded. "
            "Condition order reversed on alternate trials. "
            f"Prompt target: {_esc(workload.get('prompt_words_target'))} words; "
            f"output cap: {_esc(workload.get('max_output_tokens'))} tokens; "
            f"host cache: {_esc(workload.get(CACHE_SIZE_FIELD))} GB; "
            f"GPU memory fraction: {_esc(workload.get('mem_fraction_static'))}. "
            "The logical cap is not a measurement of physical GPU occupancy."
        )
        return brief, detail
    if summary.get("schema") == "agentic_work_audit.multisession.v1":
        brief = (f"3 concurrent sessions · {_esc(workload.get('short_wait_ms'))} / "
                 f"{_esc(workload.get('long_wait_ms'))} ms tool waits")
        detail = (
            f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
            f"{_esc(manifest.get('model'))}; backend {_esc(manifest.get('backend_version'))}; "
            f"trace {_esc(profile)}. Three equal-importance sessions began together. "
            "Two tool waits overlapped; the third session ended without replay. "
            "An explicit control command evicted the long-wait session's GPU prefix under "
            "a synthetic two-prefix budget, then the client proved host residency. "
            "Its load was requested when the tool returned without gating replay on the control response. "
            f"Prompt target: {_esc(workload.get('prompt_words_target'))} words; "
            f"output cap: {_esc(workload.get('max_output_tokens'))} tokens; "
            f"host cache: {_esc(workload.get(CACHE_SIZE_FIELD))} GB; "
            f"GPU memory fraction: {_esc(workload.get('mem_fraction_static'))}. "
            "No frontend task had higher semantic priority."
        )
        return brief, detail
    brief = (f"{_esc(order)} · {_esc(workload.get('pairs'))} measured pair(s)" if timing else
             f"{_esc(order)} · {_esc(workload.get('replays_per_case'))} replay(s)/case")
    brief += f" · {_esc(workload.get('tool_wait_ms'))} ms waits"
    detail = (
        f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
        f"{_esc(manifest.get('model'))}; SGLang {_esc(manifest.get('backend_version'))} "
        f"with {_esc((manifest.get('backend_runtime_contract') or {}).get('adapter'))} adapter. "
        f"Case order: {_esc(order)}; tool wait: {_esc(workload.get('tool_wait_ms'))} ms; "
        f"replays/case: {_esc(workload.get('replays_per_case'))}; measured pairs: {_esc(workload.get('pairs'))}; "
        f"warmup pairs: {_esc(workload.get('warmup_pairs'))}; trace: {_esc(profile)}; "
        f"exact-index limit: {_esc(workload.get('exact_trace_indices'))}; "
        f"frontend priority: {_esc(workload.get('frontend_priority'))}. "
        "Cases ran sequentially with no intentionally competing filler requests. "
        "The synthetic client explicitly evicted the GPU prefix, proved a host copy, then requested a native load."
    )
    if "prompt_words_target" in workload or "max_output_tokens" in workload:
        detail += (f" Prompt target: {_esc(workload.get('prompt_words_target'))} words; "
                   f"output cap: {_esc(workload.get('max_output_tokens'))} tokens; "
                   f"minimum host prefix: {_esc(workload.get('minimum_host_tokens'))} tokens; "
                   f"eviction attempts: {_esc(workload.get('eviction_rounds'))}; "
                   f"host cache: {_esc(workload.get(CACHE_SIZE_FIELD))} GB; "
                   f"GPU memory fraction: {_esc(workload.get('mem_fraction_static'))}.")
    return brief, detail


def _reproduction(summary: dict, timing: bool) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    cases = workload.get("cases") or []
    if summary.get("schema") in ("agentic_work_audit.multisession_comparison.v1",
                                 "agentic_work_audit.multisession_window.v1"):
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_STUDY": ("multisession_window" if summary.get("schema") ==
                                 "agentic_work_audit.multisession_window.v1" else "multisession_compare"),
            "WORK_AUDIT_TRACE_PROFILE": summary.get("_trace_profile"),
            "WORK_AUDIT_CASE_ORDER": "-".join(map(str, cases)),
            "WORK_AUDIT_PAIRS": workload.get("pairs"),
            "WORK_AUDIT_WARMUP_PAIRS": workload.get("warmup_pairs"),
            "WORK_AUDIT_SHORT_WAIT_MS": workload.get("short_wait_ms"),
            "WORK_AUDIT_LONG_WAIT_MS": workload.get("long_wait_ms"),
            "WORK_AUDIT_EARLY_AT_MS": workload.get("early_at_ms"),
            "WORK_AUDIT_PROMPT_WORDS": workload.get("prompt_words_target"),
            "WORK_AUDIT_MAX_OUTPUT_TOKENS": workload.get("max_output_tokens"),
            "WORK_AUDIT_MINIMUM_HOST_TOKENS": workload.get("minimum_host_tokens"),
            "WORK_AUDIT_EVICTION_ROUNDS": workload.get("eviction_rounds"),
            "HICACHE_SIZE_GB": workload.get(CACHE_SIZE_FIELD),
            "MEM_FRACTION_STATIC": workload.get("mem_fraction_static"),
        }
        prefix = " ".join(f"{key}={value}" for key, value in settings.items() if value is not None)
        command = f"{prefix} bash infra/container/run_work_audit_validation.sh {manifest.get('model') or '<model>'}"
        return ("<p><strong>Reconstructed command.</strong> Configure the image and model cache on the target host.</p>"
                f"<pre><code>{_esc(command)}</code></pre>")
    if summary.get("schema") == "agentic_work_audit.multisession.v1":
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_STUDY": "multisession",
            "WORK_AUDIT_TRACE_PROFILE": summary.get("_trace_profile"),
            "WORK_AUDIT_SHORT_WAIT_MS": workload.get("short_wait_ms"),
            "WORK_AUDIT_LONG_WAIT_MS": workload.get("long_wait_ms"),
            "WORK_AUDIT_PROMPT_WORDS": workload.get("prompt_words_target"),
            "WORK_AUDIT_MAX_OUTPUT_TOKENS": workload.get("max_output_tokens"),
            "WORK_AUDIT_MINIMUM_HOST_TOKENS": workload.get("minimum_host_tokens"),
            "WORK_AUDIT_EVICTION_ROUNDS": workload.get("eviction_rounds"),
            "HICACHE_SIZE_GB": workload.get(CACHE_SIZE_FIELD),
            "MEM_FRACTION_STATIC": workload.get("mem_fraction_static"),
        }
        prefix = " ".join(f"{key}={value}" for key, value in settings.items() if value is not None)
        command = f"{prefix} bash infra/container/run_work_audit_validation.sh {manifest.get('model') or '<model>'}"
        return ("<p><strong>Reconstructed command.</strong> Configure the image and model cache on the target host.</p>"
                f"<pre><code>{_esc(command)}</code></pre>")
    if len(cases) not in (2, 3):
        return "<p>Original invocation was not saved; see run manifest for recorded settings.</p>"
    settings = {
        "WORK_AUDIT_RUN_ID": summary.get("run_id"),
        "WORK_AUDIT_STUDY": "timing" if timing else "validation",
        "WORK_AUDIT_TRACE_PROFILE": summary.get("_trace_profile"),
        "WORK_AUDIT_CASE_ORDER": "-".join(map(str, cases)),
        "WORK_AUDIT_SECOND_REPLAY": "1" if not timing and workload.get("replays_per_case") == 2 else None,
        "WORK_AUDIT_PAIRS": workload.get("pairs") if timing else None,
        "WORK_AUDIT_WARMUP_PAIRS": workload.get("warmup_pairs") if timing else None,
        "WORK_AUDIT_WAIT_MS": workload.get("tool_wait_ms"),
        "WORK_AUDIT_PROMPT_WORDS": workload.get("prompt_words_target"),
        "WORK_AUDIT_MAX_OUTPUT_TOKENS": workload.get("max_output_tokens"),
        "WORK_AUDIT_MINIMUM_HOST_TOKENS": workload.get("minimum_host_tokens"),
        "WORK_AUDIT_EVICTION_ROUNDS": workload.get("eviction_rounds"),
        "HICACHE_SIZE_GB": workload.get(CACHE_SIZE_FIELD),
        "MEM_FRACTION_STATIC": workload.get("mem_fraction_static"),
        "WORK_AUDIT_EXACT_INDICES": workload.get("exact_trace_indices") if timing else None,
        "WORK_AUDIT_REQUIRE_SLOT_PROOF": "1" if timing and workload.get("slot_proof_required") else None,
    }
    prefix = " ".join(f"{key}={value}" for key, value in settings.items() if value is not None)
    command = f"{prefix} bash infra/container/run_work_audit_validation.sh {manifest.get('model') or '<model>'}"
    return ("<p><strong>Reconstructed command.</strong> Original shell invocation was not saved; "
            "configure the image and model cache on the target host.</p>"
            f"<pre><code>{_esc(command)}</code></pre>")


def _timing_result(summary: dict) -> tuple[str, str]:
    pairs = summary.get("pairs") or []
    nonblocking = any(case.get("condition") == "late_nonblocking" for case in summary.get("cases") or [])
    comparable = [p for p in pairs if p.get("comparable") and
                  isinstance(p.get("late_minus_early_first_token_after_due_ms"), (int, float))]
    deltas = [_ms(p["late_minus_early_first_token_after_due_ms"]) for p in comparable]
    if nonblocking:
        matched = sum(bool(pair.get("nonblocking_comparable")) for pair in pairs)
        headline = f"Nonblocking late load: {matched}/{len(pairs)} comparable pair(s)"
    else:
        headline = ("Late +" + " / +".join(deltas) + " to first token") if deltas else "Comparison unavailable"
    cases = {(c.get("pair"), c.get("condition")): c for c in summary.get("cases") or []}
    rows = []
    for pair in pairs:
        number = pair.get("pair")
        early, late = cases.get((number, "early"), {}), cases.get((number, "late"), {})
        concurrent = cases.get((number, "late_nonblocking"), {})
        delta = _ms(pair.get("late_minus_early_first_token_after_due_ms")) if pair.get("comparable") else "withheld"
        extra = (
            f"<td>{_ms(concurrent.get('first_token_after_due_ms'))}</td>"
            f"<td>{_ms(concurrent.get('submission_after_due_ms'))}</td>"
            f"<td>{_esc(concurrent.get('load_accepted_before_replay'))}</td>"
            f"<td>{_esc(concurrent.get('replay_cache_matches'))}</td>"
            f"<td>{_ms(pair.get('blocking_minus_nonblocking_first_token_after_due_ms')) if pair.get('nonblocking_comparable') else 'withheld'}</td>"
        ) if nonblocking else ""
        rows.append(
            f"<tr><td>{_esc(number)}</td><td>{_ms(early.get('first_token_after_due_ms'))}</td>"
            f"<td>{_ms(late.get('first_token_after_due_ms'))}</td><td>{delta}</td>"
            f"<td>{_ms(early.get('replay_ttft_ms'))} / {_ms(late.get('replay_ttft_ms'))}</td>"
            f"<td>{_ms(early.get('submission_after_due_ms'))} / {_ms(late.get('submission_after_due_ms'))}</td>"
            f"{extra}</tr>"
        )
    nonblocking_note = (
        " Nonblocking late preparation issues the load control call and submits replay without waiting for "
        "the response; it is comparable only when native acceptance precedes replay and cached-prefix reuse is observed."
    ) if nonblocking else ""
    extra_headers = (
        "<th>Nonblocking due→token</th><th>Nonblocking submit gap</th>"
        "<th>Accepted before replay</th><th>Prefix matches</th><th>Blocking − nonblocking</th>"
    ) if nonblocking else ""
    detail = (
        "<p><strong>What was measured.</strong> Early preparation requested the native load during the tool wait; "
        "late preparation requested it after the wait. Tool completion to first token includes the submission gap; "
        "replay TTFT starts only after submission." + nonblocking_note + "</p>"
        "<div class='detail-scroll'><table class='pair-table'><thead><tr><th>Pair</th><th>Early due→token</th>"
        "<th>Late due→token</th><th>Late − early</th><th>Replay TTFT, E / L</th>"
        "<th>Submit gap, E / L</th>" + extra_headers + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<p><strong>Limit.</strong> CUDA completion is observed when polled, not at the exact finish instant. "
        "Exact-index and sampled runs have different tracing costs and must not be pooled. "
        "Unknown slot lineage is not zero reuse. This sequential probe is not a production speed result.</p>"
    )
    return headline, detail


def _lifecycle_result(summary: dict) -> tuple[str, str]:
    cases = summary.get("cases") or []
    warm = next((c for c in cases if c.get("case_type") == "warm_control"), {})
    host = next((c for c in cases if c.get("case_type") == "host_backed"), {})
    block = summary.get("_block_audit") or {}
    host_block = next((c for c in block.get("cases") or [] if c.get("case_type") == "host_backed"), {})
    analysis = summary.get("_instrumentation_analysis") or {}
    host_slots = next((c for c in analysis.get("cases") or [] if c.get("case_type") == "host_backed"), {})
    headline = (f"Host load validated · warm / host replay "
                f"{_ms(warm.get('replay_ttft_ms'))} / {_ms(host.get('replay_ttft_ms'))}")
    detail = (
        "<p><strong>What was measured.</strong> Warm-prefix control versus a separate host-backed session. "
        "The host-backed session explicitly evicted its GPU copy, proved host residency, and requested load-back. "
        "These runs validate lifecycle evidence, not which policy is faster.</p>"
        f"<p>Replay 1 TTFT, warm / host: {_ms(warm.get('replay_ttft_ms'))} / {_ms(host.get('replay_ttft_ms'))}. "
        f"Replay 2 TTFT: {_ms(warm.get('second_replay_ttft_ms'))} / {_ms(host.get('second_replay_ttft_ms'))}. "
        f"Planned pre-replay load: {_esc(host_block.get('planned_pre_replay_loaded_tokens'))} tokens; "
        f"replay-time load: {_esc(host_block.get('replay_time_loaded_tokens'))} tokens. "
        f"Loaded GPU slots in replay match: {_esc(host_slots.get('loaded_slots_matched_by_replay'))}. "
        "A cache match does not prove that a model kernel consumed those exact slots.</p>"
    )
    return headline, detail


def _multisession_result(summary: dict) -> tuple[str, str]:
    sessions = summary.get("sessions") or {}
    short, long = sessions.get("short") or {}, sessions.get("long") or {}
    headline = (f"Overlapping waits · short / long due→token "
                f"{_ms(short.get('first_token_after_tool_ms'))} / "
                f"{_ms(long.get('first_token_after_tool_ms'))}")
    observations = "".join(f"<li>{_esc(item)}</li>" for item in summary.get("observations") or [])
    opportunities = "".join(f"<li>{_esc(item)}</li>" for item in summary.get("plausibly_mistimed") or [])
    detail = (
        "<p><strong>Observed.</strong> The short-wait replay and long-wait replay came from "
        "different sessions. The long session's GPU prefix was explicitly evicted and "
        "proved host-resident before its tool returned.</p>"
        f"<ul>{observations}</ul>"
        f"<p>Short tool return→submission: {_ms(short.get('submission_after_tool_ms'))}; "
        f"replay TTFT: {_ms(short.get('replay_ttft_ms'))}; cached prefix: "
        f"{_esc(short.get('cached_prefix_tokens'))} tokens.</p>"
        f"<p>Long tool return→submission: {_ms(long.get('submission_after_tool_ms'))}; "
        f"replay TTFT: {_ms(long.get('replay_ttft_ms'))}; cached prefix: "
        f"{_esc(long.get('cached_prefix_tokens'))} tokens; second replay cached prefix: "
        f"{_esc(long.get('second_replay_cached_prefix_tokens'))} tokens.</p>"
        f"<p>Long session's load request / acceptance / observed completion after tool return: "
        f"{_ms(long.get('load_request_after_tool_ms'))} / "
        f"{_ms(long.get('load_acceptance_after_tool_ms'))} / "
        f"{_ms(long.get('load_completion_observed_after_tool_ms'))}. "
        "These timestamps locate work; they do not by themselves assign a causal delay.</p>"
        f"<p><strong>Plausibly mistimed.</strong></p><ul>{opportunities or '<li>None established.</li>'}</ul>"
        f"<p><strong>Avoidable work.</strong> {_esc(summary.get('avoidable_work'))}</p>"
    )
    return headline, detail


def _multisession_comparison_result(summary: dict) -> tuple[str, str]:
    count = summary.get("comparable_pairs") or 0
    total = len(summary.get("pairs") or [])
    headline = (f"{count}/{total} matched pairs · long replay "
                f"{_direction(summary.get('median_long_due_to_token_saved_ms'), 'faster', 'slower')}; "
                f"short completion {_direction(summary.get('median_short_due_to_finish_change_ms'), 'slower', 'faster')}; "
                f"workflow {_direction(summary.get('median_workflow_makespan_saved_ms'), 'faster', 'slower')}")
    rows = []
    for pair in summary.get("pairs") or []:
        rows.append(
            f"<tr><td>{_esc(pair.get('pair'))}</td>"
            f"<td>{_ms(pair.get('early_long_due_to_token_ms'))}</td>"
            f"<td>{_ms(pair.get('late_long_due_to_token_ms'))}</td>"
            f"<td>{_ms(pair.get('long_due_to_token_saved_ms'))}</td>"
            f"<td>{_ms(pair.get('short_due_to_finish_change_ms'))}</td>"
            f"<td>{_ms(pair.get('workflow_makespan_saved_ms'))}</td>"
            f"<td>{_esc('; '.join(pair.get('reasons') or []) or 'passed')}</td></tr>"
        )
    detail = (
        "<p><strong>Measured tradeoff.</strong> Positive long-replay and workflow savings favor "
        "early loading. Positive short-session change means early loading slowed that session. "
        "The two load schedules use equal frontend importance and the same logical cache budget.</p>"
        "<div class='detail-scroll'><table class='pair-table'><thead><tr>"
        "<th>Pair</th><th>Long early due→token</th><th>Long late due→token</th>"
        "<th>Long saving</th><th>Short completion change</th><th>Workflow saving</th>"
        "<th>Evidence gate</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<p><strong>Limit.</strong> This is a synthetic matched policy comparison with explicit "
        "evictions. It does not prove natural capacity pressure or production avoidability.</p>"
    )
    return headline, detail


def _multisession_window_result(summary: dict) -> tuple[str, str]:
    pairs = summary.get("pairs") or []
    headline = (f"{summary.get('comparable_pairs', 0)}/{len(pairs)} matched triplets · "
                f"post-short long replay {_direction(summary.get('median_post_vs_late_long_saved_ms'), 'faster', 'slower')} "
                "vs late; short completion "
                f"{_direction(summary.get('median_post_vs_early_short_saved_ms'), 'faster', 'slower')} vs early")
    rows = []
    for pair in pairs:
        if pair.get("comparable"):
            metrics = "".join(
                f"<td>{_ms(pair.get(f'{mode}_{metric}'))}</td>"
                for metric in ("long_due_to_token_ms", "short_due_to_finish_ms", "workflow_makespan_ms")
                for mode in ("late", "early", "post_short")
            )
        else:
            metrics = "<td>withheld</td>" * 9
        rows.append(f"<tr><td>{_esc(pair.get('pair'))}</td>{metrics}"
                    f"<td>{_esc('; '.join(pair.get('reasons') or []) or 'passed')}</td></tr>")
    detail = (
        "<p><strong>Three load schedules.</strong> Late requests the native load at long tool return; "
        "early requests it during the short session's replay; post-short requests it only after "
        "observing that replay finish. Values are milliseconds. Lower is better in all three measures. "
        "Due→token starts when the long tool returns; due→finish starts when the short tool returns. "
        "Workflow is measured from the first initial request through the last replay completion.</p>"
        "<div class='detail-scroll'><table class='pair-table'><thead><tr><th rowspan='2'>Triplet</th>"
        "<th colspan='3'>Long due→token</th><th colspan='3'>Short due→finish</th>"
        "<th colspan='3'>Whole workflow</th><th rowspan='2'>Evidence gate</th></tr><tr>"
        + "<th>Late</th><th>Early</th><th>Post-short</th>" * 3 +
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
        "<p><strong>Limit.</strong> Post-short uses an observed client completion event; it does not "
        "show that a controller can predict that time. Capacity eviction was explicit, and "
        "request overlap does not establish kernel or HBM contention.</p>"
    )
    return headline, detail


def _question_index(milestones: list[dict]) -> tuple[dict[str, dict], dict[str, str]]:
    by_id: dict[str, dict] = {}
    by_run: dict[str, str] = {}
    for milestone in milestones:
        question_id = milestone.get("id")
        if not question_id or not milestone.get("short_question") or question_id in by_id:
            raise ValueError("Research progress needs unique IDs and short questions")
        by_id[question_id] = milestone
        related = milestone.get("related_run_ids") or milestone.get("evidence_run_ids") or []
        if not set(milestone.get("evidence_run_ids") or []).issubset(related):
            raise ValueError(f"Supporting runs must belong to {question_id}")
        for run_id in related:
            if run_id in by_run:
                raise ValueError(f"Run {run_id} belongs to more than one research question")
            by_run[run_id] = question_id
    return by_id, by_run


def _progress_html(milestones: list[dict], run_ids: set[str]) -> str:
    if not milestones:
        return ""
    rows = []
    for milestone in milestones:
        evidence = []
        for run in milestone.get("evidence_run_ids") or []:
            if run in run_ids:
                evidence.append(f'<a href="#run-{_esc(quote(run, safe=""))}">{_esc(run)}</a>')
            else:
                evidence.append(f"{_esc(run)} (not archived here)")
        question_id = str(milestone.get("id") or "")
        evidence_date = milestone.get("evidence_date_utc")
        evidence_label = (f"Evidence through {_esc(evidence_date)} UTC" if evidence_date and
                          evidence_date != "pending" else "Evidence pending")
        rows.append(
            f"<tr id='rq-{_esc(quote(question_id, safe=''))}'><td data-label='Question'>"
            f"<small>{_esc(question_id)} · {evidence_label}</small>"
            f"{_esc(milestone.get('question'))}</td>"
            f"<td data-label='Answer supported by evidence'>{_esc(milestone.get('answer'))}"
            f"<small>Supporting runs: {' · '.join(evidence) if evidence else 'not recorded'}</small></td>"
            f"<td data-label='Still unknown'>{_esc(milestone.get('unknown'))}</td></tr>"
        )
    return (
        '<section class="progress"><h2>Research progress</h2>'
        '<div class="table-scroll"><table class="progress-table"><thead><tr>'
        '<th>Question</th><th>Answer supported by evidence</th><th>Still unknown</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></section>'
    )


def render(summaries: list[tuple[Path, dict]], milestones: list[dict] | None = None) -> str:
    ordered = sorted(summaries, key=lambda item: (_time(item[1])[0], item[0].parent.name), reverse=True)
    run_ids = {str(summary.get("run_id") or path.parent.name) for path, summary in ordered}
    questions, run_questions = _question_index(milestones or [])
    rows: list[str] = []
    for path, summary in ordered:
        _, date, time, source = _time(summary)
        timing = summary.get("schema") == "agentic_work_audit.timing.v1"
        multisession = summary.get("schema") == "agentic_work_audit.multisession.v1"
        comparison = summary.get("schema") == "agentic_work_audit.multisession_comparison.v1"
        window = summary.get("schema") == "agentic_work_audit.multisession_window.v1"
        run = str(summary.get("run_id") or path.parent.name)
        kind = ("Three concurrent load windows" if window else
                "Concurrent early vs late" if comparison else "Concurrent timeline" if multisession
                else "Early vs late" if timing else "Lifecycle validation")
        manifest_question_id = ((summary.get("_manifest") or {}).get("workload") or {}).get("research_question_id")
        archived_question_id = run_questions.get(run)
        if manifest_question_id and archived_question_id and manifest_question_id != archived_question_id:
            raise ValueError(f"Run {run} has conflicting research question IDs")
        question_id = manifest_question_id or archived_question_id
        milestone = questions.get(question_id)
        fallback_question = (
            "Can loading after the short replay finishes preserve the long replay benefit without "
            "delaying the short session?" if window else
            "Under the same logical cache budget, does early preparation help the returning "
            "session without delaying another session?" if comparison else
            "When differently timed but equally important agent sessions overlap, which cache events "
            "happen before or after each replay becomes ready?" if multisession else
            "Does loading host KV during the tool wait reduce tool-return-to-first-token time "
            "versus loading after the wait?" if timing else
            "Can host residency, native load-back, and replay be linked to the same session?"
        )
        question = milestone["question"] if milestone else fallback_question
        question_cell = (
            f'<a href="#rq-{_esc(quote(str(question_id), safe=""))}">'
            f'<strong>{_esc(question_id)}</strong><span>{_esc(milestone["short_question"])}</span></a>'
            if milestone else f'<span class="unmapped">Unmapped research question'
            f'{" (" + _esc(question_id) + ")" if question_id else ""}</span>'
        )
        setup, method = _setup(summary, timing)
        result, findings = (_multisession_window_result(summary) if window else
                            _multisession_comparison_result(summary) if comparison else
                            _multisession_result(summary) if multisession else
                            _timing_result(summary) if timing else _lifecycle_result(summary))
        status = str(summary.get("status") or "unknown")
        limits = "".join(f"<li>{_esc(item)}</li>" for item in
                         [*(summary.get("failures") or []), *(summary.get("limitations") or [])])
        rows.append(
            f"<tr id='run-{_esc(quote(run, safe=''))}'><td title='{_esc(source)}'>{_esc(date)}</td>"
            f"<td title='{_esc(source)}'>{_esc(time)}</td>"
            f"<td><strong>{kind}</strong><small>{_esc(run)}</small></td>"
            f"<td class='question-cell'>{question_cell}</td><td>{setup}</td>"
            f"<td>{result}</td><td><span class='status {_esc(status)}'>{_esc(status)}</span></td>"
            f"<td><details><summary>View</summary><div class='detail'>"
            f"<p><strong>Question tested.</strong> {_esc(question)}</p>"
            f"<p>{method}</p>{findings}{_reproduction(summary, timing)}"
            f"{'<p><strong>Run-specific limits.</strong></p><ul>' + limits + '</ul>' if limits else ''}"
            f"<p><strong>Evidence.</strong> {_links(path, summary)}</p>"
            "</div></details></td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='8'>No saved run summaries are archived yet.</td></tr>")
    progress_html = _progress_html(milestones or [], run_ids)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>KV Lifecycle Audit</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182733;background:#f7f9f8}
body{max-width:1500px;margin:auto;padding:28px 24px 64px;line-height:1.45}
h1{font-size:1.7rem;margin:0 0 8px;letter-spacing:0}p{color:#3d5260}
.purpose{border-top:4px solid #d36650;padding:10px 0 16px;margin:16px 0 18px}
.purpose h2{font-size:1.05rem;margin:8px 0 6px;color:#234150}.purpose p{max-width:105ch;margin:5px 0 10px}
.scope{list-style:none;margin:8px 0 0;padding:0;max-width:1100px}
.scope li{display:grid;grid-template-columns:175px minmax(0,1fr);gap:16px;padding:7px 0;border-bottom:1px solid #d7e2e6;color:#3d5260}
.scope strong{color:#182733}
.scope li:nth-child(3){border-left:3px solid #198e7d;padding-left:10px;background:#edf7f2}
.progress{margin:0 0 24px}.progress h2{font-size:1.1rem;margin:0 0 8px;color:#234150}
.progress-table{min-width:850px}.progress-table td{white-space:normal!important;min-width:0!important;max-width:520px}
.progress-table td:first-child{width:32%}.progress-table td:nth-child(2){width:35%}
.progress-table th:nth-child(2){background:#dcefe8;color:#185b4f}
.progress-table th:nth-child(3){background:#f4eace;color:#6c541e}
.progress-table td:nth-child(2){background:#f2faf6;border-left:3px solid #37a48a}
.progress-table td:nth-child(3){background:#fffaf0}
.progress-table small{margin:0 0 4px}.progress-table td:nth-child(2) small{margin-top:8px}
.progress-table a{overflow-wrap:anywhere}
.intro{max-width:90ch;margin:0 0 20px}.table-scroll{overflow-x:auto;border:1px solid #d7e2e6;background:#fff}
table{border-collapse:collapse;width:100%;min-width:1320px}th,td{padding:12px 14px;text-align:left;vertical-align:top;border-bottom:1px solid #e5ecef}
th{background:#e4f0ef;color:#204a4a;font-size:.88rem;white-space:nowrap}td:nth-child(1),td:nth-child(2){white-space:nowrap;font-variant-numeric:tabular-nums}
tbody tr:hover{background:#f8fbfa}
td:nth-child(3){min-width:165px}td:nth-child(5){min-width:210px}td:nth-child(6){min-width:260px}
.question-cell{min-width:225px;max-width:285px;white-space:normal}
.question-cell a{display:block;text-decoration:none}.question-cell a:hover{text-decoration:underline}
.question-cell strong{display:block;color:#17685e}.question-cell span{display:block;margin-top:3px}
.unmapped{color:#8a551d}
small{display:block;color:#5a6c77;overflow-wrap:anywhere;font-size:.8rem;margin-top:3px}
.status{display:inline-block;font-weight:650}.validated{color:#126746}.failed{color:#b63839}
details{min-width:56px}summary{cursor:pointer;color:#086780;font-weight:650;list-style:none}summary::-webkit-details-marker{display:none}
.detail{min-width:430px;max-width:690px;padding:8px 0}.detail p{margin:10px 0}.detail-scroll{overflow-x:auto}
.pair-table{min-width:650px;font-size:.88rem}.pair-table th,.pair-table td{padding:7px 9px}
pre{overflow-x:auto;background:#edf4f5;padding:10px;white-space:pre-wrap;overflow-wrap:anywhere}
a{color:#086780}a:hover{text-decoration:underline}
@media(max-width:760px){
body{padding:16px 10px 40px}.detail{min-width:300px}.scope li{grid-template-columns:1fr;gap:2px}
.progress-table{min-width:0}.progress-table thead{display:none}.progress-table tr{display:block;border-bottom:1px solid #d7e2e6}
.progress-table td{display:block;width:auto!important;max-width:none!important;border-bottom:0;padding:10px 12px}
.progress-table td::before{content:attr(data-label);display:block;margin-bottom:5px;color:#234150;font-size:.82rem;font-weight:700}
}
</style></head><body><h1>KV Lifecycle Audit</h1>
<section class="purpose" aria-labelledby="research-question"><h2 id="research-question">Research question</h2>
<p><strong>Given what the harness knew at the time, was the GPU or memory system doing the wrong work at the wrong time?</strong></p>
<p>Join tool waits and session lifetimes to observed cache, transfer, request, and batch activity. Distinguish useful work, reasonable insurance, mistimed work, and potentially wasted work. Test whether a feasible alternative improves the whole system before claiming avoidable harm or a hardware opportunity. This is an audit of timing and placement, not just HBM bandwidth interference.</p>
<h2>Five audit ledgers</h2><ul class="scope">
<li><strong>Host backups</strong><span>Host residency and load-back observed; useful versus insurance versus wasted backup not yet graded.</span></li>
<li><strong>GPU evictions</strong><span>Forced eviction observed; capacity necessity and avoidability not yet graded.</span></li>
<li><strong>Session resumes</strong><span>Controlled early, late, and after-short timing measured; natural capacity effects remain unmeasured.</span></li>
<li><strong>HBM occupancy</strong><span>Useful, idle, and dead block-seconds not yet measured.</span></li>
<li><strong>GPU time</strong><span>Useful compute, recompute, and idle-with-stageable-work not yet measured.</span></li>
</ul></section>""" + progress_html + """
<p class="intro">One row per saved experiment, newest first. The research-question link opens the corresponding answer above. Date and time are UTC from the first recorded request; a completion-time fallback is labeled on hover. Timing runs compare early and late host-KV preparation; concurrent runs also show the other session's cost. Lifecycle runs check that cache events can be linked to a replay; their TTFTs are not a policy win.</p>
<div class="table-scroll"><table><thead><tr><th>Date</th><th>Time (UTC)</th><th>Experiment</th><th>Research question</th><th>Setup</th><th>Main result</th><th>Evidence gate</th><th>Details</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    summaries = []
    for path in args.results_dir.glob("*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        for name, key in (("block_audit.json", "_block_audit"),
                          ("instrumentation_analysis.json", "_instrumentation_analysis"),
                          ("run_manifest.json", "_manifest")):
            evidence = path.with_name(name)
            if evidence.exists():
                summary[key] = json.loads(evidence.read_text(encoding="utf-8"))
        audit = path.with_name("instrumentation_audit.json")
        if audit.exists():
            summary["_trace_profile"] = (json.loads(audit.read_text(encoding="utf-8")).get("gate") or {}).get("profile")
        summary["_started_ns"] = _first_request_ns(path)
        summary["_files"] = {evidence.name for evidence in path.parent.iterdir() if evidence.is_file()}
        summaries.append((Path(os.path.relpath(path, args.out.parent)), summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    milestones = []
    if args.progress_file:
        milestones = json.loads(args.progress_file.read_text(encoding="utf-8"))["milestones"]
    args.out.write_text(render(summaries, milestones), encoding="utf-8")
    print(f"Wrote {args.out} with {len(summaries)} saved run(s)")


if __name__ == "__main__":
    main()
