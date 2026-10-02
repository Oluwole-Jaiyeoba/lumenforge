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


def _seconds(value: object) -> str:
    return f"{value / 1000:.2f} s" if isinstance(value, (int, float)) else "not recorded"


def _direction(value: object, positive: str, negative: str) -> str:
    if not isinstance(value, (int, float)):
        return "not recorded"
    return f"{abs(value):.1f} ms {positive if value >= 0 else negative}"


def _trial_values(pairs: list[dict], value_for_pair, format_value=_ms) -> str:
    if not pairs:
        return "not recorded"
    return "".join(
        f'<span class="trial-value"><strong>Trial {_esc(pair.get("pair"))}:</strong> '
        f'{format_value(value_for_pair(pair))}</span>' for pair in pairs
    )


def _mode_table(headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    head = "".join(f"<th scope='col'>{_esc(header)}</th>" for header in headers)
    body = "".join(
        "<tr><th scope='row'>" + _esc(row[0]) + "</th>" +
        "".join(f"<td>{cell}</td>" for cell in row[1:]) + "</tr>"
        for row in rows
    )
    return ("<div class='detail-scroll'><table class='mode-table'><thead><tr>" + head +
            "</tr></thead><tbody>" + body + "</tbody></table></div>")


def _pair_gates(pairs: list[dict], *, nonblocking: bool = False) -> str:
    notes = []
    for pair in pairs:
        number = _esc(pair.get("pair"))
        reasons = pair.get("reasons") or pair.get("comparability_reasons") or []
        status = "comparable" if pair.get("comparable") else "not comparable"
        note = f"Trial {number}: {status}"
        if reasons:
            note += f" ({_esc('; '.join(reasons))})"
        if nonblocking and pair.get("nonblocking_comparable") is False:
            extra = pair.get("nonblocking_comparability_reasons") or []
            note += "; nonblocking comparison withheld"
            if extra:
                note += f" ({_esc('; '.join(extra))})"
        if pair.get("task_comparable") is False:
            extra = pair.get("task_comparability_reasons") or []
            note += "; full-task comparison withheld"
            if extra:
                note += f" ({_esc('; '.join(extra))})"
        notes.append(note)
    return "<p><strong>Evidence gate.</strong> " + "; ".join(notes or ["not recorded"]) + ".</p>"


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
                                 "agentic_work_audit.multisession_window.v1",
                                 "agentic_work_audit.controller_window.v1"):
        controller = summary.get("schema") == "agentic_work_audit.controller_window.v1"
        window = controller or summary.get("schema") == "agentic_work_audit.multisession_window.v1"
        unit = "trial" if window else "pair"
        measured_count = workload.get("pairs")
        warmup_count = workload.get("warmup_pairs")
        brief = (f"3 equal-importance sessions · {_esc(measured_count)} measured "
                 f"{unit if measured_count == 1 else unit + 's'} · "
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
            (f"The fourth mode let the controller decide using a {_esc(workload.get('estimated_load_ms'))} ms "
             f"load estimate and {_esc(workload.get('load_margin_ms'))} ms margin. " if controller else "") +
            f"{_esc(warmup_count)} warmup {unit if warmup_count == 1 else unit + 's'} were excluded. "
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
                                 "agentic_work_audit.multisession_window.v1",
                                 "agentic_work_audit.controller_window.v1"):
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_STUDY": ("multisession_controller" if summary.get("schema") ==
                                 "agentic_work_audit.controller_window.v1" else
                                 "multisession_window" if summary.get("schema") ==
                                 "agentic_work_audit.multisession_window.v1" else "multisession_compare"),
            "WORK_AUDIT_TRACE_PROFILE": summary.get("_trace_profile"),
            "WORK_AUDIT_CASE_ORDER": "-".join(map(str, cases)),
            "WORK_AUDIT_PAIRS": workload.get("pairs"),
            "WORK_AUDIT_WARMUP_PAIRS": workload.get("warmup_pairs"),
            "WORK_AUDIT_SHORT_WAIT_MS": workload.get("short_wait_ms"),
            "WORK_AUDIT_LONG_WAIT_MS": workload.get("long_wait_ms"),
            "WORK_AUDIT_EARLY_AT_MS": workload.get("early_at_ms"),
            "WORK_AUDIT_ESTIMATED_LOAD_MS": workload.get("estimated_load_ms"),
            "WORK_AUDIT_LOAD_MARGIN_MS": workload.get("load_margin_ms"),
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
    modes = [("Early load", "early"), ("Late load", "late")]
    if nonblocking:
        modes.append(("Late, nonblocking", "late_nonblocking"))
    rows = [
        (label, *(
            _trial_values(pairs, lambda pair, field=field, mode=mode:
                          cases.get((pair.get("pair"), mode), {}).get(field))
            for field in ("first_token_after_due_ms", "submission_after_due_ms", "replay_ttft_ms")
        )) for label, mode in modes
    ]
    nonblocking_note = (
        " Nonblocking late preparation issues the load control call and submits replay without waiting for "
        "the response; it is comparable only when native acceptance precedes replay and cached-prefix reuse is observed."
    ) if nonblocking else ""
    detail = (
        "<p><strong>What was measured.</strong> Early preparation requested the native load during the tool wait; "
        "late preparation requested it after the wait. Tool completion to first token includes the submission gap; "
        "replay TTFT starts only after submission. Each trial repeats the listed modes with fresh sessions; "
        "warmups are excluded. Lower is better in every column." + nonblocking_note + "</p>" +
        _mode_table(("Load timing", "Tool return to first token", "Before replay submission", "Replay TTFT"), rows) +
        _pair_gates(pairs, nonblocking=nonblocking) +
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
    headline = (f"Host load validated · warm replay {_ms(warm.get('replay_ttft_ms'))}; "
                f"host-backed replay {_ms(host.get('replay_ttft_ms'))}")
    detail = (
        "<p><strong>What was measured.</strong> Warm-prefix control versus a separate host-backed session. "
        "The host-backed session explicitly evicted its GPU copy, proved host residency, and requested load-back. "
        "These runs validate lifecycle evidence, not which policy is faster.</p>" +
        _mode_table(("Session", "Replay 1 TTFT", "Replay 2 TTFT"), [
            ("Warm control", _ms(warm.get("replay_ttft_ms")), _ms(warm.get("second_replay_ttft_ms"))),
            ("Host-backed", _ms(host.get("replay_ttft_ms")), _ms(host.get("second_replay_ttft_ms"))),
        ]) +
        f"<p><strong>Host-load evidence.</strong> Planned load before replay: "
        f"{_esc(host_block.get('planned_pre_replay_loaded_tokens'))} tokens; "
        f"load during replay: {_esc(host_block.get('replay_time_loaded_tokens'))} tokens; "
        f"loaded GPU slots in the replay's matched prefix: "
        f"{_esc(host_slots.get('loaded_slots_matched_by_replay'))}. "
        "Missing evidence is shown as not recorded, not zero. A cache match does not prove "
        "that a model kernel consumed those exact slots.</p>"
    )
    return headline, detail


def _multisession_result(summary: dict) -> tuple[str, str]:
    sessions = summary.get("sessions") or {}
    short, long = sessions.get("short") or {}, sessions.get("long") or {}
    headline = (f"Overlapping waits · short first token {_ms(short.get('first_token_after_tool_ms'))}; "
                f"long first token {_ms(long.get('first_token_after_tool_ms'))} after tool return")
    observations = "".join(f"<li>{_esc(item)}</li>" for item in summary.get("observations") or [])
    opportunities = "".join(f"<li>{_esc(item)}</li>" for item in summary.get("plausibly_mistimed") or [])
    detail = (
        "<p><strong>Observed.</strong> The short-wait replay and long-wait replay came from "
        "different sessions. The long session's GPU prefix was explicitly evicted and "
        "proved host-resident before its tool returned.</p>"
        f"<ul>{observations}</ul>" +
        _mode_table(("Session", "Observed tool wait", "Return to submission",
                     "Return to first token", "Replay TTFT"), [
            ("Short", _ms(short.get("observed_tool_wait_ms")),
             _ms(short.get("submission_after_tool_ms")),
             _ms(short.get("first_token_after_tool_ms")), _ms(short.get("replay_ttft_ms"))),
            ("Long", _ms(long.get("observed_tool_wait_ms")),
             _ms(long.get("submission_after_tool_ms")),
             _ms(long.get("first_token_after_tool_ms")), _ms(long.get("replay_ttft_ms"))),
        ]) +
        f"<p><strong>Prefix matches.</strong> Short replay: {_esc(short.get('cached_prefix_tokens'))} tokens; "
        f"long replay: {_esc(long.get('cached_prefix_tokens'))} tokens; long second replay: "
        f"{_esc(long.get('second_replay_cached_prefix_tokens'))} tokens.</p>" +
        _mode_table(("Long-session load event", "Time relative to tool return"), [
            ("Load requested", _ms(long.get("load_request_after_tool_ms"))),
            ("Load accepted", _ms(long.get("load_acceptance_after_tool_ms"))),
            ("Completion observed", _ms(long.get("load_completion_observed_after_tool_ms"))),
        ]) +
        "<p>Positive load-event times are after tool return. These timestamps locate work; "
        "they do not by themselves assign a causal delay.</p>"
        f"<p><strong>Plausibly mistimed.</strong></p><ul>{opportunities or '<li>None established.</li>'}</ul>"
        f"<p><strong>Avoidable work.</strong> {_esc(summary.get('avoidable_work'))}</p>"
    )
    return headline, detail


def _multisession_comparison_result(summary: dict) -> tuple[str, str]:
    pairs = summary.get("pairs") or []
    count = summary.get("comparable_pairs") or 0
    total = len(pairs)
    headline = (f"{count}/{total} matched pairs · long replay "
                f"{_direction(summary.get('median_long_due_to_token_saved_ms'), 'faster', 'slower')}; "
                f"short completion {_direction(summary.get('median_short_due_to_finish_change_ms'), 'slower', 'faster')}; "
                f"workflow {_direction(summary.get('median_workflow_makespan_saved_ms'), 'faster', 'slower')}")
    cases = {(case.get("pair"), case.get("load_timing")): case for case in summary.get("cases") or []}
    rows = []
    for label, mode in (("Late, nonblocking", "late"), ("Early", "early")):
        source_mode = "late_nonblocking" if mode == "late" else mode
        rows.append((
            label,
            _trial_values(pairs, lambda pair, mode=mode:
                          pair.get(f"{mode}_long_due_to_token_ms")),
            _trial_values(pairs, lambda pair, mode=mode:
                          pair.get(f"{mode}_short_due_to_finish_ms")),
            _trial_values(pairs, lambda pair, source_mode=source_mode:
                          cases.get((pair.get("pair"), source_mode), {}).get("workflow_makespan_ms"),
                          _seconds),
        ))
    detail = (
        "<p><strong>Measured tradeoff.</strong> Each trial repeats both load schedules with "
        "fresh, equally important sessions; warmups are excluded. Long timing starts when the long "
        "tool returns; short timing starts when the short tool returns. Workflow runs from the first "
        "initial request through the last replay completion. Lower is better in every column.</p>" +
        _mode_table(("Load timing", "Long return to first token", "Short return to finish",
                     "Whole workflow"), rows) +
        _pair_gates(pairs) +
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
    for label, mode in (("Late load", "late"), ("Early load", "early"),
                        ("After short finishes", "post_short")):
        rows.append((
            label,
            _trial_values(pairs, lambda pair, mode=mode:
                          pair.get(f"{mode}_long_due_to_token_ms")),
            _trial_values(pairs, lambda pair, mode=mode:
                          pair.get(f"{mode}_short_due_to_finish_ms")),
            _trial_values(pairs, lambda pair, mode=mode:
                          pair.get(f"{mode}_workflow_makespan_ms"), _seconds),
        ))
    detail = (
        "<p><strong>Three load schedules.</strong> Late requests the native load at long tool return; "
        "early requests it during the short session's replay; post-short requests it only after "
        "observing that replay finish. Each trial repeats all three modes with fresh sessions and "
        "alternating order; the warmup triplet is excluded. Long timing starts when the long tool "
        "returns; short timing starts when the short tool returns. Workflow runs from the first "
        "initial request through the last replay completion. Lower is better in every column.</p>" +
        _mode_table(("Load timing", "Long return to first token", "Short return to finish",
                     "Whole workflow"), rows) +
        _pair_gates(pairs) +
        "<p><strong>Limit.</strong> Post-short uses an observed client completion event; it does not "
        "show that a controller can predict that time. Capacity eviction was explicit, and "
        "request overlap does not establish kernel or HBM contention.</p>"
    )
    return headline, detail


def _controller_window_result(summary: dict) -> tuple[str, str]:
    pairs = summary.get("pairs") or []
    headline = (f"{summary.get('comparable_pairs', 0)}/{len(pairs)} matched four-mode trials · "
                f"controller long replay {_direction(summary.get('median_controller_vs_late_long_saved_ms'), 'faster', 'slower')} "
                "vs late; workflow "
                f"{_direction(summary.get('median_controller_vs_late_workflow_saved_ms'), 'faster', 'slower')} vs late")
    rows = []
    for label, mode in (("Late load", "late"), ("Early load", "early"),
                        ("Scripted after-short", "post_short"), ("Controller decision", "controller")):
        rows.append((
            label,
            _trial_values(pairs, lambda pair, mode=mode: pair.get(f"{mode}_long_due_to_token_ms")),
            _trial_values(pairs, lambda pair, mode=mode: pair.get(f"{mode}_short_due_to_finish_ms")),
            _trial_values(pairs, lambda pair, mode=mode:
                          pair.get(f"{mode}_workflow_makespan_ms"), _seconds),
        ))
    decisions = " · ".join(
        f"Trial {_esc(pair.get('pair'))}: {_esc(pair.get('controller_action', 'unavailable'))}"
        for pair in pairs
    )
    detail = (
        "<p><strong>Four matched schedules.</strong> The controller observed short-replay completion, "
        "checked host residency and slot release, and compared remaining wait against a declared "
        "load-time estimate plus margin. Its decisions are recorded in the timeline. "
        "Lower is better in all timing columns; warmups are excluded.</p>" +
        _mode_table(("Load timing", "Long return to first token", "Short return to finish",
                     "Whole workflow"), rows) +
        f"<p><strong>Controller action.</strong> {decisions}</p>" +
        _pair_gates(pairs) +
        "<p><strong>Limit.</strong> The load estimate is predeclared, not learned from production; "
        "the cache budget is synthetic and capacity eviction is explicit.</p>"
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


def _run_finding(summary: dict) -> str:
    status = summary.get("status")
    if status == "failed":
        return "Evidence checks failed; no performance conclusion is supported."
    if status != "validated":
        return "Evidence status is unavailable; no finding is claimed."

    schema = summary.get("schema")
    pairs = summary.get("pairs") or []
    if schema == "agentic_work_audit.controller_window.v1":
        if not pairs or not all(pair.get("comparable") for pair in pairs):
            return "The controller comparison was not fully validated; its conclusion is withheld."
        if all(isinstance(pair.get("controller_vs_late_long_saved_ms"), (int, float)) and
               pair["controller_vs_late_long_saved_ms"] > 0 and
               isinstance(pair.get("controller_vs_late_workflow_saved_ms"), (int, float)) and
               pair["controller_vs_late_workflow_saved_ms"] > 0 for pair in pairs):
            return "The controller improved long replay and workflow time versus late loading in every matched trial."
        return "The controller made observed decisions, but its timing benefit varied across trials."
    if schema in ("agentic_work_audit.multisession_window.v1",
                  "agentic_work_audit.multisession_comparison.v1"):
        if not pairs or not all(pair.get("comparable") for pair in pairs):
            return "The matched comparison was not fully validated; its conclusion is withheld."
        if schema == "agentic_work_audit.multisession_window.v1":
            if all(all(isinstance(pair.get(key), (int, float)) and pair[key] > 0 for key in (
                "post_vs_late_long_saved_ms", "post_vs_early_short_saved_ms",
                "post_vs_late_workflow_saved_ms")) for pair in pairs):
                return ("Post-short loading improved long replay and workflow time versus late loading, "
                        "while sparing the short session versus early loading.")
            return "The three load schedules had mixed effects across the measured trials."
        if all(all(isinstance(pair.get(key), (int, float)) and pair[key] > 0 for key in (
            "long_due_to_token_saved_ms", "short_due_to_finish_change_ms",
            "workflow_makespan_saved_ms")) for pair in pairs):
            return "Early loading sped the long replay and whole workflow, but delayed the short session."
        return "Early and late loading had mixed effects across the measured sessions."

    if schema == "agentic_work_audit.timing.v1":
        if any(case.get("condition") == "late_nonblocking" for case in summary.get("cases") or []):
            if pairs and all(pair.get("nonblocking_comparable") is False for pair in pairs):
                cases = {(case.get("pair"), case.get("condition")): case
                         for case in summary.get("cases") or []}
                gaps = [
                    (cases.get((pair.get("pair"), "late_nonblocking"), {}).get("submission_after_due_ms"),
                     cases.get((pair.get("pair"), "late"), {}).get("submission_after_due_ms"))
                    for pair in pairs
                ]
                if gaps and all(isinstance(new, (int, float)) and
                                isinstance(old, (int, float)) and new < old
                                for new, old in gaps):
                    return ("Nonblocking submission shortened the client gap, but the strict "
                            "nonblocking comparison was withheld.")
                return "The strict nonblocking comparison was withheld despite observed timings."
            if pairs and any(pair.get("nonblocking_comparable") is False for pair in pairs):
                return "Some nonblocking comparisons were withheld; see the per-trial evidence."
            return "Nonblocking loading was measured; see the matched per-mode timings."
        comparable = [pair for pair in pairs if pair.get("comparable")]
        if comparable and all(isinstance(pair.get("late_minus_early_first_token_after_due_ms"),
                                      (int, float)) and
                              pair["late_minus_early_first_token_after_due_ms"] > 0
                              for pair in comparable):
            return "Loading during the tool wait brought the first token sooner in every matched trial."
        return "The early-versus-late first-token comparison did not show a consistent gain."

    if schema == "agentic_work_audit.multisession.v1":
        sessions = summary.get("sessions") or {}
        if not (sessions.get("short") and sessions.get("long")):
            return "The saved timeline lacks both replay sessions; no cross-session finding is claimed."
        return "The trace linked overlapping tool waits, host-KV movement, and replays across separate sessions."
    if schema == "agentic_work_audit.validation.v1":
        if not any(case.get("case_type") == "host_backed" for case in summary.get("cases") or []):
            return "No host-backed case is recorded; a KV load-back finding is withheld."
        return "Host-backed KV movement and replay were linked; this was not a policy-speed comparison."
    return "No run-specific interpretation is available for this evidence format."


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
        controller = summary.get("schema") == "agentic_work_audit.controller_window.v1"
        run = str(summary.get("run_id") or path.parent.name)
        kind = ("Controller-chosen load window" if controller else "Three concurrent load windows" if window else
                "Concurrent early vs late" if comparison else "Concurrent timeline" if multisession
                else "Early vs late" if timing else "Lifecycle validation")
        manifest_question_id = ((summary.get("_manifest") or {}).get("workload") or {}).get("research_question_id")
        archived_question_id = run_questions.get(run)
        if manifest_question_id and archived_question_id and manifest_question_id != archived_question_id:
            raise ValueError(f"Run {run} has conflicting research question IDs")
        question_id = manifest_question_id or archived_question_id
        milestone = questions.get(question_id)
        fallback_question = (
            "Can the controller choose a safe KV load window from observed events?" if controller else
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
        result, findings = (_controller_window_result(summary) if controller else
                            _multisession_window_result(summary) if window else
                            _multisession_comparison_result(summary) if comparison else
                            _multisession_result(summary) if multisession else
                            _timing_result(summary) if timing else _lifecycle_result(summary))
        status = str(summary.get("status") or "unknown")
        finding = _run_finding(summary)
        limits = "".join(f"<li>{_esc(item)}</li>" for item in
                         [*(summary.get("failures") or []), *(summary.get("limitations") or [])])
        detail_id = f"detail-{_esc(quote(run, safe=''))}"
        rows.append(
            f"<tr class='run-row' id='run-{_esc(quote(run, safe=''))}'>"
            f"<td data-label='Date' title='{_esc(source)}'>{_esc(date)}</td>"
            f"<td data-label='Time (UTC)' title='{_esc(source)}'>{_esc(time)}</td>"
            f"<td data-label='Experiment'><strong>{kind}</strong><small>{_esc(run)}</small></td>"
            f"<td data-label='Research question' class='question-cell'>{question_cell}</td>"
            f"<td data-label='Setup'>{setup}</td>"
            f"<td data-label='Main result'>{result}</td>"
            f"<td data-label='Finding' class='finding-cell'>{_esc(finding)}</td>"
            f"<td data-label='Evidence gate'><span class='status {_esc(status)}'>{_esc(status)}</span></td>"
            f"<td data-label='Details'><button class='detail-toggle' type='button' "
            f"aria-expanded='false' aria-controls='{detail_id}'>View</button></td></tr>"
            f"<tr class='detail-row' id='{detail_id}' hidden><td colspan='9'><div class='detail'>"
            f"<p><strong>Question tested.</strong> {_esc(question)}</p>"
            f"<p>{method}</p>{findings}{_reproduction(summary, timing)}"
            f"{'<p><strong>Run-specific limits.</strong></p><ul>' + limits + '</ul>' if limits else ''}"
            f"<p><strong>Evidence.</strong> {_links(path, summary)}</p>"
            "</div></td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='9'>No saved run summaries are archived yet.</td></tr>")
    progress_html = _progress_html(milestones or [], run_ids)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>KV Lifecycle Audit</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182733;background:#f7f9f8;box-sizing:border-box}
*,*::before,*::after{box-sizing:inherit}
body{width:100%;margin:0;padding:24px clamp(12px,1.3vw,24px) 64px;line-height:1.45}
h1{font-size:1.7rem;margin:0 0 8px;letter-spacing:0}p{color:#3d5260}
.purpose{border-top:4px solid #d36650;padding:10px 0 16px;margin:16px 0 18px}
.purpose h2{font-size:1.05rem;margin:8px 0 6px;color:#234150}.purpose p{max-width:105ch;margin:5px 0 10px}
.scope{list-style:none;margin:8px 0 0;padding:0;max-width:1100px}
.scope li{display:grid;grid-template-columns:175px minmax(0,1fr);gap:16px;padding:7px 0;border-bottom:1px solid #d7e2e6;color:#3d5260}
.scope strong{color:#182733}
.scope li:nth-child(3){border-left:3px solid #198e7d;padding-left:10px;background:#edf7f2}
.progress{margin:0 0 24px}.progress h2{font-size:1.1rem;margin:0 0 8px;color:#234150}
.progress-table{table-layout:fixed}.progress-table td{white-space:normal!important;min-width:0!important;max-width:520px}
.progress-table td:first-child{width:32%}.progress-table td:nth-child(2){width:35%}
.progress-table th:nth-child(2){background:#dcefe8;color:#185b4f}
.progress-table th:nth-child(3){background:#f4eace;color:#6c541e}
.progress-table td:nth-child(2){background:#f2faf6;border-left:3px solid #37a48a}
.progress-table td:nth-child(3){background:#fffaf0}
.progress-table small{margin:0 0 4px}.progress-table td:nth-child(2) small{margin-top:8px}
.progress-table a{overflow-wrap:anywhere}
.intro{max-width:90ch;margin:0 0 20px}.table-scroll{width:100%;border:1px solid #d7e2e6;background:#fff}
table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;vertical-align:top;border-bottom:1px solid #e5ecef}
th{background:#e4f0ef;color:#204a4a;font-size:.88rem}td:nth-child(1),td:nth-child(2){white-space:nowrap;font-variant-numeric:tabular-nums}
.results-table{table-layout:fixed}.results-table td{overflow-wrap:anywhere}
tbody tr:hover{background:#f8fbfa}
.finding-cell{white-space:normal;background:#f1f8f5;border-left:3px solid #46a58b;color:#24544b;font-weight:600}
.question-cell{white-space:normal}
.question-cell a{display:block;text-decoration:none}.question-cell a:hover{text-decoration:underline}
.question-cell strong{display:block;color:#17685e}.question-cell span{display:block;margin-top:3px}
.unmapped{color:#8a551d}
small{display:block;color:#5a6c77;overflow-wrap:anywhere;font-size:.8rem;margin-top:3px}
.status{display:inline-block;font-weight:650}.validated{color:#126746}.failed{color:#b63839}
.detail-toggle{border:0;background:none;padding:0;color:#086780;font:inherit;font-weight:650;cursor:pointer;text-decoration:underline}
.detail-toggle:focus-visible{outline:2px solid #086780;outline-offset:3px}
.detail-row[hidden]{display:none!important}.detail-row>td{padding:16px 20px;background:#f5f9f8}
.detail{width:100%;min-width:0}.detail>p,.detail>ul{max-width:110ch}.detail p{margin:10px 0}.detail-scroll{overflow-x:auto}
.pair-table{min-width:650px;font-size:.88rem}.pair-table th,.pair-table td{padding:7px 9px}
.mode-table{min-width:620px;font-size:.88rem;table-layout:fixed}
.mode-table th,.mode-table td{padding:9px 10px;white-space:normal;overflow-wrap:break-word}
.mode-table th:first-child{width:23%}.mode-table tbody th{background:#f0f7f5;color:#224b45}
.trial-value{display:block;white-space:nowrap;font-variant-numeric:tabular-nums}
.trial-value + .trial-value{margin-top:4px}.trial-value strong{display:inline-block;min-width:50px;margin-right:6px;color:#526777;font-size:.77rem}
pre{overflow-x:auto;background:#edf4f5;padding:10px;white-space:pre-wrap;overflow-wrap:anywhere}
a{color:#086780}a:hover{text-decoration:underline}
@media(max-width:1300px){
.results-table,.results-table tbody{display:block}.results-table thead,.results-table colgroup{display:none}
.results-table .run-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-bottom:2px solid #cbdcde}
.results-table .run-row td{display:block;min-width:0;max-width:none;white-space:normal;padding:10px 12px}
.results-table .run-row td::before{content:attr(data-label);display:block;margin-bottom:4px;color:#526777;font-size:.78rem;font-weight:700}
.results-table .detail-row{display:block}.results-table .detail-row td{display:block;width:100%}
}
@media(max-width:960px){
.progress-table thead{display:none}.progress-table tr{display:block;border-bottom:1px solid #d7e2e6}
.progress-table td{display:block;width:auto!important;max-width:none!important;border-bottom:0;padding:10px 12px}
.progress-table td::before{content:attr(data-label);display:block;margin-bottom:5px;color:#234150;font-size:.82rem;font-weight:700}
}
@media(max-width:760px){
body{padding:16px 10px 40px}.scope li{grid-template-columns:1fr;gap:2px}
.results-table .run-row{grid-template-columns:repeat(2,minmax(0,1fr))}
}
@media(max-width:540px){.results-table .run-row{grid-template-columns:1fr}}
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
<p class="intro">One row per saved experiment, newest first. Main result shows the measurements; Finding states the run-specific deduction. The research-question link opens the broader answer above. Date and time are UTC from the first recorded request; a completion-time fallback is labeled on hover. Lifecycle timing is not a policy win.</p>
<div class="table-scroll"><table class="results-table"><colgroup><col style="width:8%"><col style="width:7%"><col style="width:11%"><col style="width:14%"><col style="width:14%"><col style="width:17%"><col style="width:15%"><col style="width:8%"><col style="width:6%"></colgroup><thead><tr><th>Date</th><th>Time (UTC)</th><th>Experiment</th><th>Research question</th><th>Setup</th><th>Main result</th><th>Finding</th><th>Evidence gate</th><th>Details</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
<script>
document.querySelectorAll('.detail-toggle').forEach((button) => {
  button.addEventListener('click', () => {
    const row = document.getElementById(button.getAttribute('aria-controls'));
    if (!row) return;
    row.hidden = !row.hidden;
    button.setAttribute('aria-expanded', String(!row.hidden));
    button.textContent = row.hidden ? 'View' : 'Hide';
  });
});
</script></body></html>"""


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
