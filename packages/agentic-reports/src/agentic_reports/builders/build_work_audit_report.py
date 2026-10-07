"""Build the compact, chronological KV lifecycle audit from saved evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
from statistics import median
from urllib.parse import quote
from zoneinfo import ZoneInfo

CACHE_SIZE_FIELD = "hicache_size_gb"


def _esc(value: object) -> str:
    return html.escape(str(value if value is not None else "not recorded"), quote=True)


def _ms(value: object) -> str:
    return f"{value:.1f} ms" if isinstance(value, (int, float)) else "not recorded"


def _tool_cycle_user_metrics(summary: dict) -> tuple[float, float, float]:
    values = sorted(row["ttft_ms"] for row in summary["turns"] if row["kind"] == "active")
    return median(values), values[math.ceil(0.95 * len(values)) - 1], sum(values)


def _seconds(value: object) -> str:
    return f"{value / 1000:.2f} s" if isinstance(value, (int, float)) else "not recorded"


def _forward_trace_enabled(summary: dict) -> bool:
    if "model_forward_required" in summary:
        return bool(summary["model_forward_required"])
    return any(batch.get("model_forward_ms") is not None
               for case in summary.get("cases") or []
               for batch in (case.get("short_decode") or {}).get("decode_batches") or [])


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
    width = max(620, len(headers) * 145)
    return (f"<div class='detail-scroll'><table class='mode-table' style='min-width:{width}px'><thead><tr>" + head +
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
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
            return summary.get("started_ns")
        if summary.get("schema") == "agentic_work_audit.tool_cycles.v1":
            return summary.get("started_ns")
        if summary.get("schema") == "agentic_work_audit.overlap_dose.v1":
            target_start = (summary.get("target") or {}).get("request_start_ns")
            if isinstance(target_start, int):
                return target_start
            events = path.with_name("harness_events.jsonl")
            if events.exists():
                with events.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row.get("event") == "rq11.staged" and isinstance(row.get("ts_ns"), int):
                            return row["ts_ns"]
        if summary.get("schema") in ("agentic_work_audit.busy_comparison.v1",
                                     "agentic_work_audit.kv_load_attribution.v1"):
            starts = []
            for events in path.parent.glob("arms/*/harness_events.jsonl"):
                with events.open(encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line)
                        if row.get("kind") == "initial_finished" and isinstance(row.get("request_start_ns"), int):
                            starts.append(row["request_start_ns"])
            return min(starts) if starts else None
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
        source = ("Target replay or staging (UTC)" if summary.get("schema") ==
                  "agentic_work_audit.overlap_dose.v1" else "First request (UTC)")
        return started_ns, instant.strftime("%Y-%m-%d"), instant.strftime("%H:%M:%S"), source
    completed_ms = (summary.get("_manifest") or {}).get("created_at_ms")
    if isinstance(completed_ms, int):
        instant = datetime.fromtimestamp(completed_ms / 1000, timezone.utc)
        return completed_ms * 1_000_000, instant.strftime("%Y-%m-%d"), instant.strftime("%H:%M:%S"), "Manifest completion time (UTC); start unavailable"
    return -1, "not recorded", "not recorded", "No timestamp in saved evidence"


def _links(path: Path, summary: dict) -> str:
    if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
        base = path.parent.as_posix()
        links = [f'<a href="{_esc(path.as_posix())}">Summary JSON</a>',
                 f'<a href="{_esc(base)}/run_manifest.json">Run manifest</a>']
        for row in summary["arms"]:
            arm = f"seed{row['seed']}_{row['arm']}"
            links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/case_results.json">'
                         f'{_esc(arm)} per-turn timings</a>')
            links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/backend_trace.jsonl.gz">'
                         f'{_esc(arm)} raw trace</a>')
        for failure in summary.get("failed_arms") or []:
            arm = failure["arm_id"]
            links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/server.log">'
                         f'{_esc(arm)} backend failure</a>')
            links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/backend_trace.jsonl.gz">'
                         f'{_esc(arm)} partial trace</a>')
        for related in summary.get("related_run_ids") or []:
            links.append(f'<a href="{_esc(path.parent.parent.as_posix())}/{_esc(related)}'
                         f'/arms/seed1_full_prepare/server.log">Full-prepare pilot failure</a>')
        return " · ".join(links)
    if summary.get("schema") == "agentic_work_audit.storage_replay.v1":
        base = path.parent.as_posix()
        links = [f'<a href="{_esc(path.as_posix())}">Summary JSON</a>',
                 f'<a href="{_esc(base)}/run_manifest.json">Run manifest</a>']
        for row in summary["rows"]:
            arm = f"seed{row['seed']}_{row['arm']}"
            links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/case_results.json">'
                         f'{_esc(arm)} raw timings</a>')
            links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/backend_trace.jsonl.gz">'
                         f'{_esc(arm)} backend trace</a>')
        return " · ".join(links)
    if summary.get("schema") in ("agentic_work_audit.busy_comparison.v1",
                                 "agentic_work_audit.kv_load_attribution.v1"):
        base = path.parent.as_posix()
        links = [f'<a href="{_esc(path.as_posix())}">Summary JSON</a>',
                 f'<a href="{_esc(base)}/run_manifest.json">Run manifest</a>']
        seeds = summary.get("pairs") or summary.get("seeds") or []
        modes = (("baseline", "check_only", "controller") if summary.get("seeds")
                 else ("baseline", "controller"))
        for seed in seeds:
            for mode in modes:
                arm = f"seed{seed['seed']}_{mode}"
                for name, label in (("summary.json", "metrics"),
                                    ("harness_events.jsonl", "timeline"),
                                    ("instrumentation_audit.json", "hook gate"),
                                    ("backend_trace.jsonl.gz", "raw trace")):
                    links.append(f'<a href="{_esc(base)}/arms/{_esc(arm)}/{name}">'
                                 f'{_esc(arm)} {label}</a>')
        return " · ".join(links)
    labels = (
        ("summary.json", "Summary JSON"), ("run_manifest.json", "Run manifest"),
        ("instrumentation_audit.json", "Trace gate"), ("block_audit.json", "Block audit"),
        ("instrumentation_analysis.json", "Slot analysis"),
        ("normalized_events.jsonl", "Timeline"), ("harness_events.jsonl", "Harness events"),
        ("backend_trace.jsonl.gz", "Raw trace"), ("backend_trace.jsonl", "Raw trace"),
    )
    files = summary.get("_files") or {"summary.json"}
    links = [f'<a href="{_esc(path.with_name(name).as_posix())}">{label}</a>'
             for name, label in labels if name in files]
    if (path.parent / "runtime" / "backend_features.json").exists():
        links.append(f'<a href="{_esc(path.parent.as_posix())}/runtime/backend_features.json">Backend features</a>')
    if summary.get("_cuda_kernel_subset"):
        base = path.parent.as_posix()
        links.append(f'<a href="{_esc(base)}/nsys/kernel_attribution_pair01.json">Captured GPU kernels</a>')
        if summary.get("_cuda_launch_gaps"):
            links.append(f'<a href="{_esc(base)}/nsys/launch_gap_attribution_pair01.json">CUDA launch gaps</a>')
        links.append(f'<a href="{_esc(base)}/nsys/backend.sqlite.gz">Nsight SQLite trace</a>')
    if summary.get("_physical_overlap"):
        base = path.parent.as_posix()
        links.append(f'<a href="{_esc(base)}/nsys/physical_overlap.json">Physical copy overlap</a>')
    if summary.get("_decode_submission"):
        base = path.parent.as_posix()
        links.append(f'<a href="{_esc(base)}/nsys/decode_submission.json">Decode launch timing</a>')
    if (path.parent / "nsys" / "backend.nsys-rep").exists():
        links.append(f'<a href="{_esc(path.parent.as_posix())}/nsys/backend.nsys-rep">Nsight capture</a>')
    if summary.get("_profile_status"):
        base = path.parent.as_posix()
        links.append(f'<a href="{_esc(base)}/nsys/profile_status.json">Profiler status</a>')
    return " · ".join(links)


def _setup(summary: dict, timing: bool) -> tuple[str, str]:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
        historical_unguarded = str(manifest.get("run_id") or "").startswith("storage_cycles_rq17_")
        earlier_rate_limit_gap = any(
            "native_prefetch_not_admitted" in error
            for arm in summary["arms"] for error in arm.get("preparation_errors") or []
        )
        brief = (f"{_esc(workload.get('session_count'))} equal-priority sessions · "
                 f"{_esc(workload.get('turns_per_session'))} tool returns each · natural file-cache pressure")
        detail = (f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
                  f"{_esc(manifest.get('model'))}; pinned SGLang {_esc(manifest.get('backend_version'))}. "
                  f"Each session began near {_esc(workload.get('initial_tokens'))} prompt tokens and gained "
                  f"{_esc(workload.get('tool_result_words'))} tool-result words per turn. "
                  f"Tool waits varied from {_esc(workload.get('tool_wait_base_ms'))} to "
                  f"{_esc((workload.get('tool_wait_base_ms') or 0) + (workload.get('tool_wait_spread_ms') or 0))} ms "
                  "by a fixed seed. Fresh backend and file-cache path per arm; "
                  f"GPU KV cap {_esc(workload.get('gpu_kv_token_cap'))} tokens, host cache "
                  f"{_esc(workload.get('host_cache_gb'))} GiB, "
                  f"{_esc(workload.get('page_size_tokens'))}-token pages. "
                  "No explicit eviction or frontend priority. The host-stage arm observes natural "
                  "residency during each wait; replay is submitted at its due time even if "
                  "staging misses it. "
                  + ("The historical blocked run used an unguarded partial-suffix prefetch; "
                     "the current adapter refuses prefetch when it would require host eviction. "
                     if historical_unguarded else
                     "The run checked suffix anchor and host capacity, but did not yet classify "
                     "a native rate-limit refusal; the current adapter checks that limit too. "
                     if earlier_rate_limit_gap else
                     "The adapter validates the suffix anchor, host capacity, and native rate limit "
                     "before native prefetch. ")
                  + (f"CUDA graphs {'on' if workload.get('cuda_graph') else 'off'}; "
                     f"overlap scheduling {'on' if workload.get('overlap_schedule') else 'off'}. "
                     if 'cuda_graph' in workload else "")
                  + ("A separate full-GPU-prepare pilot hit a scheduler assertion and is excluded."
                     if historical_unguarded else "Full GPU preparation was not tested in this run."))
        return brief, detail
    if summary.get("schema") == "agentic_work_audit.storage_replay.v1":
        peers = int(workload.get("peer_count") or 0)
        arms_note = ("The control-only arm made three residency checks without transferring KV; "
                     "the host-stage arm fetched KV from L3 to L2 before tool return. "
                     if "control_only" in (workload.get("arms") or []) else
                     "The early arms staged L3→L2 or L3→L2→L1 before tool return. ")
        brief = (f"1 returning + {peers} peer session(s) · {workload.get('prompt_tokens')} prompt tokens · "
                 f"{workload.get('tool_wait_ms')} ms tool wait · file-backed L3")
        detail = (f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
                  f"{_esc(manifest.get('model'))}; pinned backend "
                  f"{_esc(manifest.get('backend_version'))}. Fresh backend and storage path per arm; "
                  f"write-through storage, {_esc(workload.get('page_size_tokens'))}-token pages, "
                  "equal frontend priority, lean KV trace. "
                  "Each arm populated a prefix, evicted it from GPU and host, then replayed it "
                  "after the same synthetic tool wait. A positive native L3 hit was required. "
                  + arms_note
                  + f"There were {peers} peer session(s), started "
                  f"{_esc(workload.get('peer_start_ms', 1000))} ms into the tool wait."
                  + (f" CUDA graphs {'on' if workload.get('cuda_graph') else 'off'}; "
                     f"overlap scheduling {'on' if workload.get('overlap_schedule') else 'off'}."
                     if 'cuda_graph' in workload else "")
                  + (" Matching replay output hashes were required across arms."
                     if workload.get('verify_output') else ""))
        return brief, detail
    if summary.get("schema") == "agentic_work_audit.tool_cycles.v1":
        brief = (f"{_esc(summary.get('active_count'))} active + "
                 f"{_esc(summary.get('donor_count'))} donor sessions · "
                 f"{_esc(summary.get('turn_count'))} tool returns each")
        detail = (f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
                  f"{_esc(manifest.get('model'))}; backend {_esc(manifest.get('backend_version'))}; "
                  f"seed {_esc(summary.get('seed'))}. Repeated, deterministic synthetic tool outputs "
                  f"grew each session's history from {_esc(workload.get('initial_tokens'))} initial words "
                  f"by {_esc(workload.get('tool_result_words'))} words per turn. "
                  f"Tool wait {_esc(workload.get('wait_ms'))} ms plus seeded jitter; "
                  f"output cap {_esc(workload.get('decode_tokens'))} tokens. "
                  f"CUDA graphs {_esc('on' if workload.get('cuda_graph_requested') else 'off')}; "
                  f"overlap scheduling {_esc('on' if workload.get('overlap_schedule_requested') else 'off')}. "
                  "No frontend importance ranks. Donor traffic does not by itself prove KV movement."
                  + (" Backend tracing was disabled; stage and load-back evidence is unavailable."
                     if summary.get("status") == "trace_off_control" else ""))
        return brief, detail
    if summary.get("schema") == "agentic_work_audit.overlap_dose.v1":
        brief = (f"{_esc(summary.get('session_count'))} equal-priority sessions · "
                 f"{_esc(summary.get('donor_count'))} host-resident donor prefixes · "
                 f"{_esc(summary.get('planned_overlap'))} planned load overlaps")
        detail = (f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
                  f"{_esc(manifest.get('model'))}; backend {_esc(manifest.get('backend_version'))}; "
                  f"seed {_esc(summary.get('seed'))}. One target and "
                  f"{_esc(summary.get('session_count') - summary.get('donor_count') - 1)} "
                  "other active decoders resumed after a tool wait. The donor prefixes were "
                  "explicitly host-resident; the same number of native worker loads ran in every dose, "
                  "with their timing shifted around target decode. "
                  f"Target output cap {_esc(workload.get('decode_tokens'))} tokens; "
                  f"tool waits {_esc(workload.get('target_wait_ms'))} / "
                  f"{_esc(workload.get('donor_wait_ms'))} ms; active/donor prompts "
                  f"{_esc(workload.get('active_prompt_words', workload.get('prompt_words_target')))} / "
                  f"{_esc(workload.get('donor_prompt_words', workload.get('prompt_words_target')))} words. "
                  + (f"CUDA graphs {_esc('on' if workload['cuda_graph_requested'] else 'off')}; "
                     f"overlap scheduling {_esc('on' if workload['overlap_schedule_requested'] else 'off')}. "
                     if 'cuda_graph_requested' in workload and 'overlap_schedule_requested' in workload
                     else "") +
                  "Fresh backend for each dose; no frontend importance ranks.")
        return brief, detail
    if summary.get("schema") in ("agentic_work_audit.busy_comparison.v1",
                                 "agentic_work_audit.kv_load_attribution.v1"):
        seed_count = summary.get("seed_count")
        brief = (f"{_esc(workload.get('session_count'))} sessions × "
                 f"{_esc(workload.get('tool_waits_per_session'))} tool waits; "
                 f"{_esc(seed_count)} paired {'seed' if seed_count == 1 else 'seeds'}; natural capacity pressure")
        detail = (f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
                  f"{_esc(manifest.get('model'))}; backend {_esc(manifest.get('backend_version'))}. "
                  f"Fresh backend per arm, order reversed by seed. Prefix target "
                  f"{_esc(workload.get('prefix_tokens'))} tokens, replay cap "
                  f"{_esc(workload.get('replay_tokens'))} tokens, tool waits "
                  f"{_esc(workload.get('wait_range_ms'))} ms, host cache "
                  f"{_esc(workload.get(CACHE_SIZE_FIELD))} GB, KV I/O backend "
                  f"{_esc(workload.get('hicache_io_backend', 'direct'))}, load execution "
                  f"{_esc(workload.get('load_execution', 'scheduler'))}, GPU memory fraction "
                  f"{_esc(workload.get('mem_fraction_static'))}. "
                  "Controller requires a host-resident prefix and at least "
                  f"{_esc(workload.get('estimated_load_ms'))} + {_esc(workload.get('load_margin_ms'))} "
                  "ms before expected tool return. No frontend importance ranks or forced eviction; "
                  + ("focused ingress + KV trace." if summary.get("seeds") else "lean KV trace."))
        return brief, detail
    order = " → ".join(map(str, workload.get("cases") or [])) or "not recorded"
    profile = summary.get("_trace_profile") or next(
        (name for name in manifest.get("enabled_instrumentation") or []
         if name.startswith("kv_lifecycle") or name == "kv_decode_overlap"),
        None,
    )
    if summary.get("schema") == "agentic_work_audit.decode_overlap.v1":
        count = workload.get("pairs")
        brief = (f"3 equal-importance sessions · {_esc(count)} measured "
                 f"{'pair' if count == 1 else 'pairs'} · early vs after-short worker load")
        detail = (
            f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
            f"{_esc(manifest.get('model'))}; backend {_esc(manifest.get('backend_version'))}; "
            f"trace {_esc(profile)}; model-forward trace "
            f"{'on' if _forward_trace_enabled(summary) else 'off'}. "
            "Each case began three equal-importance sessions: "
            "a short tool wait, a long tool wait, and one session that ended. The long prefix "
            "was explicitly evicted to host. Its worker load began either during short decode "
            "or after short completion; both loads had to finish before long tool return. "
            f"Tool waits: {_esc(workload.get('short_wait_ms'))} / "
            f"{_esc(workload.get('long_wait_ms'))} ms; early load at "
            f"{_esc(workload.get('early_at_ms'))} ms. "
            f"{_esc(workload.get('warmup_pairs'))} warmup pairs excluded; order reversed by pair. "
            f"Prompt target {_esc(workload.get('prompt_words_target'))} words, output cap "
            f"{_esc(workload.get('max_output_tokens'))} tokens, host cache "
            f"{_esc(workload.get(CACHE_SIZE_FIELD))} GB, GPU memory fraction "
            f"{_esc(workload.get('mem_fraction_static'))}."
        )
        return brief, detail
    if summary.get("schema") in ("agentic_work_audit.multisession_comparison.v1",
                                 "agentic_work_audit.multisession_window.v1",
                                 "agentic_work_audit.controller_window.v1",
                                 "agentic_work_audit.decode_overlap.v1"):
        controller = summary.get("schema") == "agentic_work_audit.controller_window.v1"
        window = controller or summary.get("schema") == "agentic_work_audit.multisession_window.v1"
        unit = "trial" if window else "pair"
        measured_count = workload.get("pairs")
        warmup_count = workload.get("warmup_pairs")
        brief = (f"3 equal-importance sessions · {_esc(measured_count)} measured "
                 f"{unit if measured_count == 1 else unit + 's'} · "
                 f"{_esc(workload.get('short_wait_ms'))} / {_esc(workload.get('long_wait_ms'))} ms waits")
        if workload.get("load_execution"):
            brief += f" · {_esc(workload['load_execution'])} KV load"
        detail = (
            f"<strong>How it ran.</strong> {_esc(manifest.get('hardware_profile'))}; "
            f"{_esc(manifest.get('model'))}; SGLang {_esc(manifest.get('backend_version'))}; "
            f"trace {_esc(profile)}; load execution {_esc(workload.get('load_execution', 'scheduler'))}. "
            "Each case started three equal-importance sessions; "
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
    pair_count = workload.get("pairs")
    replay_count = workload.get("replays_per_case")
    brief = (f"Case order: {_esc(order)} · {_esc(pair_count)} measured "
             f"{'pair' if pair_count == 1 else 'pairs'}" if timing else
             f"Case order: {_esc(order)} · {_esc(replay_count)} replays/case")
    wait_ms = workload.get("tool_wait_ms")
    brief += f" · {wait_ms} ms waits" if wait_ms is not None else " · wait not recorded"
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
    if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
        settings = {
            "WORK_AUDIT_CYCLES_SEEDS": " ".join(map(str, workload.get("seeds") or [])),
            "WORK_AUDIT_CYCLES_ARMS": " ".join(workload.get("arms") or []),
            "WORK_AUDIT_CYCLES_SESSIONS": workload.get("session_count"),
            "WORK_AUDIT_CYCLES_TURNS": workload.get("turns_per_session"),
            "WORK_AUDIT_CYCLES_INITIAL_TOKENS": workload.get("initial_tokens"),
            "WORK_AUDIT_CYCLES_TOOL_WORDS": workload.get("tool_result_words"),
            "WORK_AUDIT_CYCLES_DECODE_TOKENS": workload.get("decode_tokens"),
            "WORK_AUDIT_CYCLES_WAIT_MS": workload.get("tool_wait_base_ms"),
            "WORK_AUDIT_CYCLES_WAIT_SPREAD_MS": workload.get("tool_wait_spread_ms"),
            "WORK_AUDIT_CYCLES_STAGGER_MS": workload.get("session_stagger_ms"),
            "WORK_AUDIT_CYCLES_GPU_TOKENS": workload.get("gpu_kv_token_cap"),
            "WORK_AUDIT_CYCLES_HOST_GB": workload.get("host_cache_gb"),
            "WORK_AUDIT_CYCLES_CUDA_GRAPH": int(workload["cuda_graph"]) if "cuda_graph" in workload else None,
            "WORK_AUDIT_CYCLES_OVERLAP_SCHEDULE": int(workload["overlap_schedule"]) if "overlap_schedule" in workload else None,
        }
        command = " ".join(f"{key}='{value}'" for key, value in settings.items() if value is not None)
        note = ("<p><strong>Safety note.</strong> The historical unguarded partial-prefetch "
                "behavior is unavailable; this command uses a capacity-guarded suffix path "
                "and is not an exact replay of the failed prototype.</p>"
                if str(manifest.get("run_id") or "").startswith("storage_cycles_rq17_") else "")
        if any("native_prefetch_not_admitted" in error for arm in summary["arms"]
               for error in arm.get("preparation_errors") or []):
            note += ("<p><strong>Reproduction note.</strong> The current adapter checks native "
                     "rate limiting before prefetch; this command will not reproduce the earlier "
                     "unclassified refusal exactly.</p>")
        return (note + "<p><strong>How to run the safe variant.</strong> Choose a new run ID:</p><pre><code>"
                + _esc(command + " WORK_AUDIT_CYCLES_RUN_ID='new_unique_id' "
                       "bash infra/container/run_work_audit_storage_cycles.sh") + "</code></pre>")
    if summary.get("schema") == "agentic_work_audit.storage_replay.v1":
        storage_extra = ""
        if workload.get("research_question_id") in {"RQ18", "RQ19"}:
            storage_extra = (f"WORK_AUDIT_STORAGE_ARMS='{ ' '.join(workload.get('arms') or []) }' "
                             f"WORK_AUDIT_STORAGE_CUDA_GRAPH={int(workload.get('cuda_graph', False))} "
                             f"WORK_AUDIT_STORAGE_OVERLAP_SCHEDULE={int(workload.get('overlap_schedule', False))} "
                             + ("WORK_AUDIT_STORAGE_VERIFY_OUTPUT=1 " if workload.get("verify_output") else ""))
        command = (f"WORK_AUDIT_STORAGE_SEEDS='{ ' '.join(map(str, workload.get('seeds') or [])) }' "
                   f"WORK_AUDIT_STORAGE_RESEARCH_QUESTION_ID={workload.get('research_question_id', 'RQ15')} "
                   f"WORK_AUDIT_STORAGE_PROMPT_ID={workload.get('prompt_id', '')} "
                   f"WORK_AUDIT_STORAGE_MODEL={workload.get('model', manifest.get('model', ''))} "
                   f"WORK_AUDIT_STORAGE_WAIT_MS={workload.get('tool_wait_ms')} "
                   f"WORK_AUDIT_STORAGE_PROMPT_TOKENS={workload.get('prompt_tokens')} "
                   f"WORK_AUDIT_STORAGE_PAGE_SIZE={workload.get('page_size_tokens')} "
                   f"WORK_AUDIT_STORAGE_HOST_GB={workload.get('host_cache_gb', 14)} "
                   f"WORK_AUDIT_STORAGE_MEM_FRACTION={workload.get('gpu_mem_fraction', 0.7)} "
                   f"WORK_AUDIT_STORAGE_PEERS={workload.get('peer_count', 0)} "
                   f"WORK_AUDIT_STORAGE_PEER_START_MS={workload.get('peer_start_ms', 1000)} "
                   f"WORK_AUDIT_STORAGE_PEER_PROMPT_TOKENS={workload.get('peer_prompt_tokens', 1024)} "
                   f"WORK_AUDIT_STORAGE_PEER_MAX_TOKENS={workload.get('peer_max_tokens', 96)} "
                   f"{storage_extra}bash infra/container/run_work_audit_storage.sh")
        return "<p><strong>How to reproduce.</strong></p><pre><code>" + _esc(command) + "</code></pre>"
    if summary.get("schema") == "agentic_work_audit.tool_cycles.v1":
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_STUDY": "tool_cycles",
            "WORK_AUDIT_SEED": workload.get("seed"),
            "WORK_AUDIT_TOOL_CYCLE_ACTIVE_COUNT": workload.get("active_count"),
            "WORK_AUDIT_DONOR_COUNT": workload.get("donor_count"),
            "WORK_AUDIT_TOOL_CYCLE_TURNS": workload.get("turn_count"),
            "WORK_AUDIT_TOOL_CYCLE_INITIAL_TOKENS": workload.get("initial_tokens"),
            "WORK_AUDIT_TOOL_CYCLE_DONOR_INITIAL_TOKENS": workload.get("donor_initial_tokens"),
            "WORK_AUDIT_TOOL_CYCLE_RESULT_WORDS": workload.get("tool_result_words"),
            "WORK_AUDIT_TOOL_CYCLE_WAIT_MS": workload.get("wait_ms"),
            "WORK_AUDIT_DECODE_TOKENS": workload.get("decode_tokens"),
            "WORK_AUDIT_CUDA_GRAPH": int(bool(workload.get("cuda_graph_requested"))),
            "WORK_AUDIT_OVERLAP_SCHEDULE": int(bool(workload.get("overlap_schedule_requested"))),
            "WORK_AUDIT_TRACE_ENABLE": int(bool(workload.get("trace_enabled", True))),
            CACHE_SIZE_FIELD.upper(): workload.get(CACHE_SIZE_FIELD),
            "MEM_FRACTION_STATIC": workload.get("mem_fraction_static"),
        }
        prefix = " ".join(f"{key}='{value}'" for key, value in settings.items() if value is not None)
        return ("<p><strong>How to reproduce.</strong></p><pre><code>"
                + _esc(prefix + " bash infra/container/run_work_audit_validation.sh "
                       + str(manifest.get("model") or "Qwen/Qwen2.5-Coder-7B-Instruct"))
                + "</code></pre>")
    if summary.get("schema") == "agentic_work_audit.overlap_dose.v1":
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_STUDY": "overlap_dose",
            "WORK_AUDIT_RESEARCH_QUESTION_ID": workload.get("research_question_id"),
            "WORK_AUDIT_PAIR_ID": workload.get("pair_id"),
            "WORK_AUDIT_SESSION_COUNT": workload.get("session_count"),
            "WORK_AUDIT_DONOR_COUNT": workload.get("donor_count"),
            "WORK_AUDIT_PLANNED_OVERLAP": workload.get("planned_overlap"),
            "WORK_AUDIT_SEED": workload.get("seed"),
            "WORK_AUDIT_DECODE_TOKENS": workload.get("decode_tokens"),
            "WORK_AUDIT_ACTIVE_PROMPT_WORDS": workload.get("active_prompt_words", workload.get("prompt_words_target")),
            "WORK_AUDIT_DONOR_PROMPT_WORDS": workload.get("donor_prompt_words", workload.get("prompt_words_target")),
            "WORK_AUDIT_SHORT_WAIT_MS": workload.get("target_wait_ms"),
            "WORK_AUDIT_DONOR_WAIT_MS": workload.get("donor_wait_ms"),
            "WORK_AUDIT_FORWARD_TRACE": workload.get("forward_trace_enabled"),
            "WORK_AUDIT_NSYS_ENABLE": workload.get("nsys_enabled"),
            "WORK_AUDIT_CUDA_GRAPH": (int(workload["cuda_graph_requested"])
                                      if "cuda_graph_requested" in workload else None),
            "WORK_AUDIT_OVERLAP_SCHEDULE": (int(workload["overlap_schedule_requested"])
                                            if "overlap_schedule_requested" in workload else None),
            "AGENTIC_KV_PREPARE_LOAD_WORKER": "1",
            CACHE_SIZE_FIELD.upper(): workload.get(CACHE_SIZE_FIELD),
            "MEM_FRACTION_STATIC": workload.get("mem_fraction_static"),
        }
        prefix = " ".join(f"{key}='{value}'" for key, value in settings.items() if value is not None)
        command = f"{prefix} bash infra/container/run_work_audit_validation.sh {manifest.get('model') or '<model>'}"
        return ("<p><strong>Runner.</strong> <code>infra/container/run_work_audit_validation.sh</code> "
                "starts the pinned backend; <code>agentic_experiments.runners.run_work_audit_overlap_sweep</code> "
                "drives equal-priority sessions. Set the container image and model cache on the target host.</p>"
                f"<pre><code>{_esc(command)}</code></pre>")
    if summary.get("schema") in ("agentic_work_audit.busy_comparison.v1",
                                 "agentic_work_audit.kv_load_attribution.v1"):
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_RESEARCH_QUESTION_ID": workload.get("research_question_id"),
            "WORK_AUDIT_SEEDS": " ".join(map(str, workload.get("seeds") or [])),
            "WORK_AUDIT_MODES": " ".join(workload.get("modes") or ["baseline", "controller"]),
            "WORK_AUDIT_SESSION_COUNT": workload.get("session_count"),
            "WORK_AUDIT_TOOL_WAITS": workload.get("tool_waits_per_session"),
            "WORK_AUDIT_PREFIX_TOKENS": workload.get("prefix_tokens"),
            "WORK_AUDIT_REPLAY_TOKENS": workload.get("replay_tokens"),
            "WORK_AUDIT_WAIT_MIN_MS": (workload.get("wait_range_ms") or [None, None])[0],
            "WORK_AUDIT_WAIT_MAX_MS": (workload.get("wait_range_ms") or [None, None])[1],
            "WORK_AUDIT_ESTIMATED_LOAD_MS": workload.get("estimated_load_ms"),
            "WORK_AUDIT_MARGIN_MS": workload.get("load_margin_ms"),
            "WORK_AUDIT_MINIMUM_HOST_TOKENS": workload.get("minimum_host_tokens"),
            "WORK_AUDIT_LOAD_EXECUTION": workload.get("load_execution"),
            CACHE_SIZE_FIELD.upper(): workload.get(CACHE_SIZE_FIELD),
            "MEM_FRACTION_STATIC": workload.get("mem_fraction_static"),
        }
        prefix = " ".join(f"{key}='{value}'" for key, value in settings.items() if value is not None)
        command = f"{prefix} bash infra/container/run_work_audit_busy.sh {manifest.get('model') or '<model>'}"
        return ("<p><strong>Runner.</strong> "
                "<code>infra/container/run_work_audit_busy.sh</code> orchestrates fresh backends; "
                "<code>agentic_experiments.runners.run_busy_kv_audit</code> drives each arm. "
                "Set the container image and model cache on the target host.</p>"
                f"<pre><code>{_esc(command)}</code></pre>")
    cases = workload.get("cases") or []
    if summary.get("schema") in ("agentic_work_audit.multisession_comparison.v1",
                                 "agentic_work_audit.multisession_window.v1",
                                 "agentic_work_audit.controller_window.v1",
                                 "agentic_work_audit.decode_overlap.v1"):
        settings = {
            "WORK_AUDIT_RUN_ID": summary.get("run_id"),
            "WORK_AUDIT_RESEARCH_QUESTION_ID": workload.get("research_question_id"),
            "WORK_AUDIT_STUDY": ("multisession_controller" if summary.get("schema") ==
                                 "agentic_work_audit.controller_window.v1" else
                                 "multisession_overlap" if summary.get("schema") ==
                                 "agentic_work_audit.decode_overlap.v1" else
                                 "multisession_window" if summary.get("schema") ==
                                 "agentic_work_audit.multisession_window.v1" else "multisession_compare"),
            "WORK_AUDIT_TRACE_PROFILE": summary.get("_trace_profile"),
            "WORK_AUDIT_FORWARD_TRACE": ("1" if _forward_trace_enabled(summary) else "0")
                                            if summary.get("schema") == "agentic_work_audit.decode_overlap.v1" else None,
            "WORK_AUDIT_NSYS_ENABLE": "1" if summary.get("_cuda_kernel_subset") else None,
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
            "AGENTIC_KV_PREPARE_LOAD_WORKER": ("1" if workload.get("load_execution") == "worker" else "0"
                                               if "load_execution" in workload else None),
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
            "WORK_AUDIT_RESEARCH_QUESTION_ID": workload.get("research_question_id"),
            "WORK_AUDIT_STUDY": "multisession",
            "WORK_AUDIT_TRACE_PROFILE": summary.get("_trace_profile"),
            "WORK_AUDIT_SHORT_WAIT_MS": workload.get("short_wait_ms"),
            "WORK_AUDIT_LONG_WAIT_MS": workload.get("long_wait_ms"),
            "WORK_AUDIT_PROMPT_WORDS": workload.get("prompt_words_target"),
            "WORK_AUDIT_MAX_OUTPUT_TOKENS": workload.get("max_output_tokens"),
            "WORK_AUDIT_MINIMUM_HOST_TOKENS": workload.get("minimum_host_tokens"),
            "WORK_AUDIT_EVICTION_ROUNDS": workload.get("eviction_rounds"),
            "AGENTIC_KV_PREPARE_LOAD_WORKER": ("1" if workload.get("load_execution") == "worker" else "0"
                                               if "load_execution" in workload else None),
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
        "WORK_AUDIT_RESEARCH_QUESTION_ID": workload.get("research_question_id"),
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
        "AGENTIC_KV_PREPARE_LOAD_WORKER": ("1" if workload.get("load_execution") == "worker" else "0"
                                           if "load_execution" in workload else None),
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


def _decode_overlap_result(summary: dict) -> tuple[str, str]:
    pairs = [pair for pair in summary.get("pairs") or [] if pair.get("comparable")]
    finish_delta = [pair["early_short_finish_ms"] - pair["post_short_finish_ms"]
                    for pair in pairs]
    first_delta = [pair["early_short_first_token_ms"] - pair["post_short_first_token_ms"]
                   for pair in pairs]
    headline = (
        f"{len(pairs)}/{len(summary.get('pairs') or [])} matched pairs · "
        f"short finish {_direction(median(finish_delta) if finish_delta else None, 'later', 'earlier')} "
        "with early load; first token "
        f"{_direction(median(first_delta) if first_delta else None, 'later', 'earlier')}"
    )
    rows = []
    for case in summary.get("cases") or []:
        short = case.get("short_decode") or {}
        batches = short.get("decode_batches") or []
        def total(key: str) -> float | None:
            values = [batch[key] for batch in batches if isinstance(batch.get(key), (int, float))]
            return round(sum(values), 1) if values else None
        rows.append((
            f"Trial {case.get('pair')} · {case.get('condition')}",
            _ms((case.get("audit") or {}).get("sessions", {}).get("short", {}).get("first_token_after_tool_ms")),
            _ms((case.get("audit") or {}).get("sessions", {}).get("short", {}).get("completion_after_tool_ms")),
            _ms(total("duration_ms")), _ms(total("model_forward_ms")),
            _ms(total("non_forward_ms")),
            _ms(sum(short.get("inter_batch_gaps_ms") or [])),
            _ms((case.get("load_overlap") or {}).get("short_decode_overlap_ms")),
        ))
    detail = (
        "<p><strong>What the times mean.</strong> All short timings begin at short tool return. "
        "Batch time is measured inside the backend's run_batch call after first token; model-forward "
        "time is the nested forward_batch_generation call. Gap time is between those batches. "
        "The overlap column is worker start-to-commit intersected with short decode, an upper "
        "bound on actual GPU-copy overlap. Both modes load before long tool return.</p>" +
        _mode_table(("Trial / load timing", "Short first token", "Short finish", "Batch total",
                     "Model forward", "Other batch time", "Between batches", "Load overlap"), rows) +
        _pair_gates(summary.get("pairs") or []) +
        "<p><strong>Limit.</strong> Model-forward wall time includes GPU work and synchronization. "
        "This identifies the affected software stage but does not isolate HBM bandwidth or "
        "prove a hardware offload benefit.</p>"
    )
    cuda = summary.get("_cuda_kernel_subset")
    if cuda:
        delta = cuda["pairs"][0]["early_minus_post_short_ms"]
        headline = (f"Captured pair: kernel execution {_direction(delta['kernel_duration_sum_ms'], 'longer', 'shorter')}; "
                    f"between-kernel gaps {_direction(delta['kernel_gap_inside_span_ms'], 'longer', 'shorter')}. "
                    "Later CUDA capture incomplete.")
        kernel_rows = []
        for pair in cuda.get("pairs") or []:
            for mode, label in (("post_short", "After short response"), ("early", "Early worker load")):
                arm = pair[mode]
                kernel_rows.append((f"Pair {pair['pair']} · {label}", arm["kernel_count"],
                                    _ms(arm["kernel_duration_sum_ms"]),
                                    _ms(arm["kernel_gap_inside_span_ms"]),
                                    _ms(arm["htod_during_kernel_span_ms"])))
        detail += (
            "<p><strong>GPU kernel check (captured pair only).</strong> Nsight lost CUDA activity "
            "for a later case, so the full profiler run did not pass the gate. These forward "
            "calls have complete kernel linkage; this is mechanism evidence, not a new clean "
            "latency comparison.</p>" +
            _mode_table(("Captured case", "Kernels", "Kernel execution", "Between-kernel gaps",
                         "H-to-D during kernel spans"), kernel_rows) +
            "<p><strong>Limit.</strong> Kernel duration and gaps alone do not identify "
            "CPU launch delay versus GPU synchronization. No H-to-D copy overlapped these kernel "
            "spans, which argues against direct copy-bandwidth contention in the captured pair.</p>"
        )
        launch = summary.get("_cuda_launch_gaps")
        if launch:
            split = launch["early_minus_post_short_ms"]
            headline += (f" Of the added gaps, {split['cpu_before_launch_ms']:.1f} ms occurred "
                         "before the CPU began the next CUDA launch.")
            launch_rows = []
            for mode, label in (("post_short", "After short response"),
                                ("early", "Early worker load")):
                arm = launch[mode]
                launch_rows.append((label, _ms(arm["gap_ms"]),
                                    _ms(arm["cpu_before_launch_ms"]),
                                    _ms(arm["launch_api_ms"]),
                                    _ms(arm["after_launch_api_ms"])))
            detail += (
                "<p><strong>CUDA launch check (same captured pair).</strong> Each short-forward "
                "kernel gap is split at the CPU launch call for the next kernel. The extra "
                "delay is mainly before that call, especially in the first three forwards.</p>" +
                _mode_table(("Captured case", "Total gaps", "Before CPU launch",
                             "During launch API", "After launch API"), launch_rows) +
                "<p>Recorded stream-wait event activity: "
                f"{launch['post_short']['stream_wait_event_gpu_ms']:.3f} ms after-short, "
                f"{launch['early']['stream_wait_event_gpu_ms']:.3f} ms early; "
                f"blocking CUDA synchronization API time: "
                f"{launch['post_short']['blocking_sync_api_ms']:.3f} ms after-short, "
                f"{launch['early']['blocking_sync_api_ms']:.3f} ms early. "
                "These sync times can overlap the gap categories and are not additive.</p>" +
                "<p><strong>Limit.</strong> Before-launch time can include host scheduling, "
                "model-forward CPU work, or other software waits; it does not prove which one. "
                "These are gaps between this request's kernels, not a measure of global GPU idle time.</p>"
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


def _busy_result(summary: dict) -> tuple[str, str]:
    pairs = summary.get("pairs") or []
    headline = (f"{summary.get('seed_count', 0)} paired seeds · "
                f"workflow {_direction(summary.get('median_workflow_saved_ms'), 'faster', 'slower')}; "
                f"total replay TTFT {_direction(summary.get('median_total_replay_ttft_saved_ms'), 'lower', 'higher')}")
    rows = []
    for pair in pairs:
        for label, arm in (("Ordinary replay", pair["baseline"]),
                           ("Controller-timed KV", pair["controller"])):
            rows.append((f"Seed {pair['seed']} · {label}",
                         _seconds(arm["workflow_makespan_ms"]),
                         _seconds(arm["total_replay_ttft_ms"]),
                         _seconds(arm["total_return_to_first_token_ms"]),
                         _ms(arm["return_to_first_token"]["p95_ms"]),
                         str(arm["native_load_events"])))
    decisions = "".join(
        f"<li>Seed {_esc(pair['seed'])}: {_esc(pair['controller']['controller_plan_checks'])} "
        f"control checks, {_esc(pair['eligible_host_checks'])} found host-resident KV; "
        f"{_esc(pair['controller']['controller_load_attempts'])} load attempts; "
        f"{_esc(pair['controller']['controller_loads_finished_before_tool_return'])} finished before return; "
        f"{_esc(pair['sessions_helped'])} sessions helped, {_esc(pair['sessions_harmed'])} harmed; "
        f"gate {_esc(pair['evidence_gate'])}"
        f"{': ' + _esc('; '.join(pair['evidence_reasons'])) if pair['evidence_reasons'] else ''}.</li>"
        for pair in pairs
    )
    detail = (
        "<p><strong>What was measured.</strong> The same equal-importance prompts, waits, and replay "
        "limits were run with a fresh backend process for each arm. Baseline uses ordinary cache "
        "handling; the controller checks host residency during each tool wait and requests a native "
        "load only when its declared time window is sufficient. Positive saved time means the "
        "controller arm was faster. Totals sum all replays, so they are not wall-clock seconds.</p>" +
        _mode_table(("Arm", "Whole workflow", "Summed replay TTFT", "Summed return to first token",
                     "P95 return to first token", "Native loads"), rows) +
        "<p><strong>Controller decisions and per-session impact.</strong></p><ul>" + decisions + "</ul>" +
        "<p><strong>Limit.</strong> Native load counts do not prove all bytes were used. "
        "The extra control checks also consume backend time and are not isolated from early-copy cost. "
        "Concurrency can change batching between arms, so this is a system-level comparison, "
        "not an isolated hardware-bandwidth measurement.</p>"
    )
    return headline, detail


def _attribution_result(summary: dict) -> tuple[str, str]:
    attempts = sum(seed["arms"]["controller"]["load_windows"]["load_attempts"]
                   for seed in summary.get("seeds") or [])
    extra_ttft = [seed["arms"]["controller"]["total_replay_ttft_ms"] -
                  seed["arms"]["check_only"]["total_replay_ttft_ms"]
                  for seed in summary.get("seeds") or []]
    changes = " / ".join(f"+{value / 1000:.1f} s" if value >= 0 else f"{value / 1000:.1f} s"
                         for value in extra_ttft)
    headline = (f"{attempts} early-load attempts · load arm {changes} summed replay TTFT "
                "vs checks-only")
    sections = []
    stage_labels = (
        ("submit_to_receive", "Submission → backend receive"),
        ("receive_to_queue", "Receive → queue"),
        ("queue_to_cache_lookup", "Queue → first cache lookup"),
        ("receive_to_cache_lookup", "Receive → first cache lookup"),
        ("cache_lookup_to_first_token", "Cache lookup → first token"),
        ("first_token_to_finish", "First token → completion"),
        ("substantive_decode", "First token → completion, multi-chunk replies"),
    )
    for seed in summary.get("seeds") or []:
        arms = seed["arms"]
        mode_rows = []
        for mode, label in (("baseline", "No checks"), ("check_only", "Checks only"),
                            ("controller", "Checks + loads")):
            arm = arms[mode]
            mode_rows.append((label, _seconds(arm["total_replay_ttft_ms"]),
                              _seconds(arm["workflow_makespan_ms"]),
                              str(arm["controller_plan_checks"]),
                              str(arm["load_windows"]["load_attempts"]),
                              str(arm["native_load_events"])))
        stage_rows = []
        for key, label in stage_labels:
            checks = seed["check_cost"][key]
            loads = seed["load_association"][key]
            coverage = f"{loads['matched_replays']}/{loads['total_replays']}"
            stage_rows.append((label,
                               f"{_ms(checks['mean_added_ms'])} ({_ms(checks['median_added_ms'])})",
                               f"{_ms(loads['mean_added_ms'])} ({_ms(loads['median_added_ms'])})",
                               coverage))
        windows = arms["controller"]["load_windows"]
        load_phases = arms["controller"].get("load_phases") or []
        phase_rows = [(
            str(load.get("load_id")), _esc(load.get("loaded_tokens")),
            _ms(load.get("control_queue_ms")), _ms(load.get("scheduler_preparation_ms")),
            _ms(load.get("load_back_call_ms")), _ms(load.get("ready_to_load_call_ms")),
            _ms(load.get("cuda_elapsed_ms")),
        ) for load in load_phases]
        phase_detail = (
            "<p><strong>Completed early-load phases.</strong> Control queue and calls are wall time; "
            "the CUDA event is device elapsed time and may overlap those calls. Do not sum them.</p>" +
            _mode_table(("Load ID", "Tokens", "Control queue", "Scheduler prep", "Load call",
                         "Ready call", "CUDA event"), phase_rows)
        ) if phase_rows else ""
        sections.append(
            f"<h4>Seed {_esc(seed['seed'])}</h4>" +
            _mode_table(("Arm", "Summed replay TTFT", "Workflow", "Checks", "Load attempts",
                         "Native loads"), mode_rows) +
            "<p><strong>Added milliseconds per replay, measured against the preceding arm.</strong> "
            "Values are mean (median); positive means slower. Missing stage coverage is shown, "
            "not counted as zero.</p>" +
            _mode_table(("Stage", "Checks vs no checks", "Loads vs checks only", "Matched replays"),
                        stage_rows) +
            f"<p><strong>Load windows:</strong> {_esc(windows['confirmed_control_windows'])} "
            f"confirmed control-to-completion windows; {_esc(windows['windows_with_other_replay_before_first_token'])} "
            "overlapped another replay before first token; "
            f"{_esc(windows['windows_with_other_replay_after_first_token'])} overlapped another "
            "replay after first token.</p>" + phase_detail
        )
    detail = (
        "<p><strong>What was measured.</strong> All arms use equal-priority sessions and fresh "
        "backends. Check-only makes the same plan calls and a plan-only placebo call when "
        "a load would be requested, but never loads KV. The last arm requests real loads. "
        "Stage times are wall-clock "
        "observations, not exclusive GPU-kernel time; summed TTFT is not elapsed experiment time.</p>" +
        "".join(sections) +
        f"<p><strong>Evidence size.</strong> Only {attempts} early-load attempts across these seeds; "
        "do not generalize a per-copy cost from the full-workload difference.</p>" +
        f"<p><strong>Limit.</strong> {_esc(summary.get('interpretation_limit'))}</p>"
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
        hypothesis = milestone.get("hypothesis")
        hypothesis_html = (f"<p><strong>Working hypothesis.</strong> {_esc(hypothesis)}</p>"
                           if hypothesis else "")
        rows.append(
            f"<tr id='rq-{_esc(quote(question_id, safe=''))}'><td data-label='Question'>"
            f"<small>{_esc(question_id)} · {evidence_label}</small>"
            f"{_esc(milestone.get('question'))}</td>"
            f"<td data-label='Answer supported by evidence'>{_esc(milestone.get('answer'))}"
            f"<small>Supporting runs: {' · '.join(evidence) if evidence else 'not recorded'}</small></td>"
            f"<td data-label='Still unknown'>{hypothesis_html}"
            f"{_esc(milestone.get('unknown'))}</td></tr>"
        )
    return (
        '<section class="progress"><h2>Research progress</h2>'
        '<div class="table-scroll"><table class="progress-table"><thead><tr>'
        '<th>Question</th><th>Answer supported by evidence</th><th>Still unknown</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></section>'
    )


def _run_finding(summary: dict) -> str:
    status = summary.get("status")
    if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
        pairs = summary.get("paired_comparisons") or []
        baseline = [arm for arm in summary["arms"] if arm["arm"] == "on_demand"]
        staged = [arm for arm in summary["arms"] if arm["arm"] == "host_stage"]
        if summary.get("status") == "blocked":
            rate_limit_gap = any("native_prefetch_not_admitted" in error
                                 for arm in summary["arms"]
                                 for error in arm.get("preparation_errors") or [])
            failure = ("Native prefetch was declined once; an explicit rate-limit check was added afterward. "
                       if rate_limit_gap else
                       "A later host-stage arm hit a pinned SGLang cache-tree assertion. "
                       if any(row.get("backend_assertion") for row in summary.get("failed_arms") or []) else
                       "The staged arm recorded a preparation error or a required arm failed. ")
            if pairs:
                pair = pairs[0]
                return ("Blocked after one completed pair: host staging changed median replay "
                        f"delay by {pair['median_due_to_first_token_delta_ms']:+.1f} ms and "
                        f"whole-workload time by {pair['workflow_delta_ms']:+.1f} ms. "
                        + failure +
                        "This one-pair observation is not a validated performance conclusion.")
            return "Blocked: " + failure.strip() + " No validated paired comparison completed."
        if summary.get("status") == "insufficient_exposure":
            return "Paired modes finished, but no storage stage completed before a tool deadline; no staging effect is established."
        if not pairs:
            return "Calibration run only; no paired mode comparison."
        workflow = median(pair["workflow_delta_ms"] for pair in pairs)
        first_token = median(pair["median_due_to_first_token_delta_ms"] for pair in pairs)
        return (f"Across {len(pairs)} paired seeds, host staging changed median replay delay "
                f"by {first_token:+.0f} ms and whole-workload duration by {workflow:+.0f} ms "
                "(negative is faster). "
                f"On-demand replay had {sum(arm['native_replay_storage_hit_count'] for arm in baseline)} "
                "native storage hits; staging completed before due time on "
                f"{sum(arm['stage_before_due_count'] for arm in staged)} waits. "
                "These are associations in a small synthetic, file-backed workload, not an "
                "isolated physical-SSD or hardware speedup.")
    if summary.get("schema") == "agentic_work_audit.storage_replay.v1":
        workload = (summary.get("_manifest") or {}).get("workload") or {}
        if workload.get("research_question_id") == "RQ19":
            rows = {arm: [row for row in summary["rows"] if row["arm"] == arm]
                    for arm in ("on_demand", "control_only", "host_stage")}
            peer = {arm: median(row["peer_ttft_median_ms"] for row in arm_rows)
                    for arm, arm_rows in rows.items()}
            return ("Three-way control: peer median first-token time was "
                    f"{peer['on_demand']:.0f} ms without early activity, "
                    f"{peer['control_only']:.0f} ms with control probes only, and "
                    f"{peer['host_stage']:.0f} ms with actual KV staging. "
                    "This small synthetic comparison locates an effect; it does not prove a hardware cause.")
        if workload.get("research_question_id") == "RQ18":
            baseline = median(row["due_to_first_token_ms"] for row in summary["rows"]
                              if row["arm"] == "on_demand")
            staged = median(row["due_to_first_token_ms"] for row in summary["rows"]
                            if row["arm"] == "host_stage")
            peer_count = int(workload.get("peer_count") or 0)
            peer_note = ("No competing session was present; this only validates the load path."
                         if not peer_count else
                         "Peer timing and whole-workload changes are recorded separately; "
                         "a single pair is not enough to claim a win-win.")
            return (f"Native storage-only KV and replay reuse were verified. First-token "
                    f"delay changed from {baseline:.0f} to {staged:.0f} ms with host staging "
                    f"across {len(summary['paired'])} paired seed(s). {peer_note}")
        base, host, full = (
            median(row["due_to_first_token_ms"] for row in summary["rows"] if row["arm"] == arm)
            for arm in ("on_demand", "host_stage", "full_prepare")
        )
        peer_rows = [row for row in summary["rows"] if row.get("peer_count")]
        if peer_rows:
            peer_base, peer_full = (
                median(row["peer_ttft_median_ms"] for row in peer_rows if row["arm"] == arm)
                for arm in ("on_demand", "full_prepare")
            )
            if any(row.get("peers_overlapping_preparation") for row in peer_rows):
                limit = (f"Peers overlapped preparation; their median TTFT changed "
                         f"from {peer_base:.0f} to {peer_full:.0f} ms. This is not a proven win-win.")
            else:
                limit = ("Peers began after preparation finished, so this run does not measure "
                         "contention during the transfer.")
        else:
            limit = "No peer-session or whole-system benefit is established by this single-session run."
        native_times = [row["storage_data_ready_ms"] for row in summary["rows"]
                        if row.get("storage_data_ready_ms") is not None]
        poll_times = [row["storage_commit_after_ready_ms"] for row in summary["rows"]
                      if row.get("storage_commit_after_ready_ms") is not None]
        native_note = (f" Native data readiness took a median {median(native_times):.0f} ms; "
                       f"the later status-poll/host-commit interval took {median(poll_times):.0f} ms."
                       if native_times and poll_times else "")
        return (f"With storage-only KV proven, first token after tool due was "
                f"{base:.0f} ms on demand, {host:.0f} ms after host staging, "
                f"and {full:.0f} ms after full preparation "
                f"(median across {len(summary['paired'])} paired seed(s)). " + limit + native_note)
    if summary.get("schema") == "agentic_work_audit.tool_cycles.v1":
        if summary.get("status") == "trace_off_control":
            ttft_median, ttft_p95, _ = _tool_cycle_user_metrics(summary)
            return (f"Trace-off control: first-token delay {_ms(ttft_median)} median, "
                    f"{_ms(ttft_p95)} at p95 across "
                    f"{summary['active_count'] * summary['turn_count']} active replays. "
                    "Backend stage and KV-load evidence was deliberately not captured.")
        measurements = summary.get("measurements") or {}
        return (f"Across {measurements.get('active_replay_count')} active replays, "
                f"first-token delay was {measurements.get('active_ttft_median_ms')} ms median "
                f"and {measurements.get('active_ttft_p95_ms')} ms at p95. "
                f"{measurements.get('active_replays_with_kv_load_back')} active replays "
                "had a recorded KV load-back. Stage timing identifies where time was spent, "
                "not why the backend waited.")
    if summary.get("schema") == "agentic_work_audit.overlap_dose.v1":
        if summary.get("status") == "excluded":
            return f"Excluded diagnostic: {summary.get('failure_reason', 'run did not complete')}"
        physical = summary.get("_physical_overlap")
        profile = summary.get("_profile_status")
        submission = summary.get("_decode_submission")
        control = summary.get("_dose_control")
        control_submission = (control or {}).get("decode_submission")
        if physical and submission and control_submission:
            cpu_change = (submission["cpu_before_launch_ms"] -
                          control_submission["cpu_before_launch_ms"])
            kernel_change = (submission["kernel_execution_ms"] -
                             control_submission["kernel_execution_ms"])
            return (f"{physical['physical_overlap_load_count']} physical KV loads overlapped decode. "
                    f"Versus the paired profiled control, CPU-before-launch gaps grew "
                    f"{cpu_change:.0f} ms while summed kernel execution changed "
                    f"{kernel_change:+.1f} ms. This locates delay in launch cadence; "
                    "it does not identify why the host waited or measure an unprofiled speedup.")
        if physical and submission and summary.get("planned_overlap") == 0:
            return ("Profiled zero-overlap control: no KV copies ran during target decode. "
                    f"It captured {submission['forward_count']} forwards and "
                    f"{submission['kernel_count']} linked kernels for the paired launch comparison.")
        if physical and physical.get("status") == "verified":
            return (f"{physical['physical_overlap_load_count']} of "
                    f"{summary['donor_count']} native loads physically overlapped the target "
                    "decode window. This profiled run establishes overlap, not an unprofiled "
                    "performance effect or hardware cause.")
        if profile and profile.get("status") == "capture_incomplete":
            return ("Nsight recorded the worker ranges but no CUDA kernels or copies. "
                    "Physical overlap is unverified; this profiled run is excluded from "
                    "the clean timing comparison.")
        control = summary.get("_dose_control")
        if control:
            target_change = summary["target"]["completion_after_tool_ms"] - control["target_ms"]
            peer_change = median(row["completion_after_tool_ms"] for row in
                                 summary["active_requests"][1:]) - control["peer_median_ms"]
            return (f"Compared with its matched zero-overlap control, the target finished "
                    f"{target_change:.0f} ms later and peer decoders finished a median "
                    f"{peer_change:.0f} ms later. Worker windows were observed; physical "
                    "copy overlap and the precise cause remain unverified.")
        if summary.get("planned_overlap") == 0:
            return ("Zero-overlap control: the same four donor loads ran only after target "
                    "decode. This is the reference for other doses with the same seed and workload.")
        return ("Worker-window overlap was measured, but physical CUDA-copy overlap was not "
                "verified. This run alone cannot establish a dose-response slowdown.")
    if summary.get("schema") == "agentic_work_audit.kv_load_attribution.v1" and status == "complete":
        return ("The actual-load arm had higher replay TTFT in both seeds, while substantive "
                "post-first-token generation was not slower. Check-only effects varied by seed; "
                "the comparison does not prove copy-engine or HBM contention.")
    if status == "failed":
        return "Evidence checks failed; no performance conclusion is supported."
    if summary.get("schema") == "agentic_work_audit.busy_comparison.v1" and status == "inconclusive":
        return "Paired timings were collected, but the natural reload or on-time preparation gate was not met."
    if status != "validated":
        return "Evidence status is unavailable; no finding is claimed."

    schema = summary.get("schema")
    pairs = summary.get("pairs") or []
    if schema == "agentic_work_audit.decode_overlap.v1":
        if summary.get("_cuda_kernel_subset"):
            return ("In the captured pair, GPU kernels ran for about the same time; pauses "
                    "between them grew. No H-to-D copy overlapped those kernel spans. Later "
                    "profiler data was incomplete, so hardware attribution remains provisional.")
        cases = {(case.get("pair"), case.get("condition")): case
                 for case in summary.get("cases") or []}
        if not pairs or not all(pair.get("comparable") for pair in pairs):
            return "Decode-overlap comparison failed its evidence gate; no timing finding is claimed."
        finish_later = all(pair["early_short_finish_ms"] > pair["post_short_finish_ms"]
                           for pair in pairs)
        forward_available = all(any(batch.get("model_forward_ms") is not None for batch in
                                    cases[(pair["pair"], mode)]["short_decode"]["decode_batches"])
                                for pair in pairs for mode in ("early", "post_short"))
        forward_later = all(
            sum(batch.get("model_forward_ms") or 0 for batch in
                cases[(pair["pair"], "early")]["short_decode"]["decode_batches"]) >
            sum(batch.get("model_forward_ms") or 0 for batch in
                cases[(pair["pair"], "post_short")]["short_decode"]["decode_batches"])
            for pair in pairs)
        if finish_later and forward_available and forward_later:
            return ("Early worker loading delayed the short response in every pair; the added "
                    "batch time was in model forward, not queue gaps. HBM contention is not established.")
        if finish_later:
            return ("Early worker loading delayed the short response in every pair. "
                    "The trace places the added time inside backend batches, not queue gaps.")
        return "The load overlapped short decode, but its effect on short completion varied across pairs."
    if schema == "agentic_work_audit.busy_comparison.v1":
        if all(pair["workflow_saved_ms"] > 0 and pair["total_replay_ttft_saved_ms"] > 0
               for pair in pairs):
            return "Controller-timed KV preparation improved workflow and total replay TTFT in every paired seed."
        if all(pair["workflow_saved_ms"] < 0 and pair["total_replay_ttft_saved_ms"] < 0
               for pair in pairs):
            return "Controller-timed KV preparation worsened workflow and total replay TTFT in every paired seed."
        return "The controller changed system timing, but the benefits and costs differed across paired seeds."
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


def _kind(summary: dict) -> str:
    schema = summary.get("schema")
    names = {
        "agentic_work_audit.kv_load_attribution.v1": "Busy workload · KV-load attribution",
        "agentic_work_audit.busy_comparison.v1": "Busy workload · controller KV timing",
        "agentic_work_audit.controller_window.v1": "Controller-chosen load window",
        "agentic_work_audit.multisession_window.v1": "Three concurrent load windows",
        "agentic_work_audit.multisession_comparison.v1": "Concurrent early vs late",
        "agentic_work_audit.decode_overlap.v1": "Decode overlap attribution",
        "agentic_work_audit.overlap_dose.v1": "KV-load overlap pressure",
        "agentic_work_audit.tool_cycles.v1": "Repeated tool-return startup",
        "agentic_work_audit.multisession.v1": "Concurrent timeline",
        "agentic_work_audit.timing.v1": "Early vs late",
        "agentic_work_audit.validation.v1": "Lifecycle validation",
        "agentic_work_audit.storage_replay.v1": "Storage-tier KV timing",
        "agentic_work_audit.storage_cycles.summary.v1": "Repeated storage-resume timing",
    }
    kind = names.get(schema, "Audit experiment")
    if schema == "agentic_work_audit.tool_cycles.v1" and summary.get("status") == "trace_off_control":
        return "Repeated tool returns · trace-off control"
    workload = (summary.get("_manifest") or {}).get("workload") or {}
    if workload.get("research_question_id") == "RQ13":
        return "Backend scheduling and KV overlap"
    if workload.get("research_question_id") == "RQ18":
        return "Safe storage-stage session ladder"
    if workload.get("research_question_id") == "RQ19":
        return "Storage staging and peer-delay control"
    if workload.get("research_question_id") == "RQ12":
        return "Decode slowdown attribution"
    if workload.get("research_question_id") == "RQ9":
        kind += f" · {workload.get('load_execution') or 'scheduler'} load"
    return kind


def _result_parts(summary: dict) -> tuple[str, str]:
    schema = summary.get("schema")
    if schema == "agentic_work_audit.storage_cycles.summary.v1":
        rows = [(f"Seed {arm['seed']} · {arm['arm']}",
                 _ms(arm["workflow_duration_ms"]),
                 _ms(arm["due_to_first_token_median_ms"]),
                 _ms(arm["due_to_first_token_p95_ms"]),
                 _ms(arm["replay_ttft_median_ms"]),
                 arm["natural_storage_candidate_waits"], arm["stage_before_due_count"],
                 len(arm.get("preparation_skips") or []), arm["stage_loaded_tokens"],
                 arm["native_replay_storage_hit_count"],
                 arm["native_replay_storage_hit_tokens"],
                 len(arm["preparation_errors"])) for arm in summary["arms"]]
        detail = _mode_table(("Seed / mode", "Whole workload", "Due → first token median",
                              "Due → first token p95", "Replay TTFT median",
                              "Storage candidates", "Stage ready by due", "Stage skips", "Tokens staged from L3",
                              "Native L3 replay hits", "L3 replay tokens", "Stage errors"), rows)
        pairs = summary.get("paired_comparisons") or []
        pair_rows = [(pair["seed"], _ms(pair["workflow_delta_ms"]),
                      _ms(pair["median_due_to_first_token_delta_ms"]),
                      _ms(pair["median_replay_ttft_delta_ms"])) for pair in pairs]
        detail += _mode_table(("Seed", "Whole-workload change", "Replay-delay change",
                               "Replay-TTFT change"), pair_rows)
        detail += ("<p>Negative changes favor host staging. Per-session completion and all "
                   "per-turn timings are retained in each arm's raw case results. ")
        skip_reasons = [reason for arm in summary["arms"]
                        for reason in arm.get("preparation_skips") or []]
        if skip_reasons:
            detail += "Skipped prefetches: " + _esc(", ".join(
                f"{reason} ({skip_reasons.count(reason)})" for reason in sorted(set(skip_reasons)))) + ". "
        if str((summary.get("_manifest") or {}).get("run_id") or "").startswith("storage_cycles_rq17_"):
            detail += "Full GPU preparation is excluded after a live scheduler assertion.</p>"
        else:
            detail += "Full GPU preparation was not compared in this run.</p>"
        if summary.get("status") == "blocked":
            failed = "; ".join(f"{row['arm_id']}: {row['backend_assertion'] or row['client_error']}"
                               for row in summary.get("failed_arms") or [])
            detail += f"<p><strong>Blocked:</strong> {_esc(failed)}. Missing arms: "
            detail += _esc(", ".join(summary.get("missing_arm_ids") or [])) + ".</p>"
        return (f"{summary.get('status', 'unknown')} · {len(pairs)} paired seed(s) · "
                f"{summary['arms'][0]['session_count']} sessions · "
                f"{summary['arms'][0]['replay_count']} replays per arm", detail)
    if schema == "agentic_work_audit.storage_replay.v1":
        rows = [(f"Seed {row['seed']} · {row['arm']}",
                 _ms(row['due_to_first_token_ms']), _ms(row['replay_ttft_ms']),
                 _ms(row.get('peer_ttft_median_ms')),
                 row.get('peers_overlapping_preparation'),
                 _ms(row.get('storage_data_ready_ms')),
                 _ms(row.get('storage_commit_after_ready_ms')),
                 row['native_replay_storage_hit_tokens'], row['control_storage_hit_tokens'],
                 row.get('replay_matched_prefix_tokens'),
                 "yes" if row['stage_completed_before_due'] else "not applicable")
                for row in summary['rows']]
        detail = _mode_table(("Arm", "Due → first token", "Replay TTFT", "Peer TTFT median",
                              "Peers overlapping preparation",
                              "Native L3 → host ready", "Ready → status observed",
                              "L3 tokens at replay", "L3 tokens during wait",
                              "Matched replay prefix",
                              "Staging met due time"), rows)
        if any(row["arm"] == "control_only" for row in summary["rows"]):
            pair_rows = [(pair["seed"],
                          ", ".join(_ms(value) for value in pair["control_only_peer_ttft_deltas_ms"]),
                          ", ".join(_ms(value) for value in pair["host_stage_peer_ttft_deltas_ms"]),
                          ", ".join(_ms(value) for value in pair["control_only_peer_completion_deltas_ms"]),
                          ", ".join(_ms(value) for value in pair["host_stage_peer_completion_deltas_ms"]))
                         for pair in summary["paired"]]
            detail += _mode_table(("Seed", "Control-only peer TTFT changes", "Staged peer TTFT changes",
                                   "Control-only peer completion changes", "Staged peer completion changes"),
                                  pair_rows)
            phase_rows = [(row["seed"], row["arm"],
                           _ms(median(peer["request_to_lookup_ms"] for peer in row["peer_phase_times"]
                                      if peer["request_to_lookup_ms"] is not None)),
                           _ms(median(peer["lookup_to_first_token_ms"] for peer in row["peer_phase_times"]
                                      if peer["lookup_to_first_token_ms"] is not None)))
                          for row in summary["rows"] if row.get("peer_phase_times")
                          and all(peer["request_to_lookup_ms"] is not None
                                  and peer["lookup_to_first_token_ms"] is not None
                                  for peer in row["peer_phase_times"])]
            if phase_rows:
                detail += _mode_table(("Seed", "Arm", "Peer request → cache lookup median",
                                       "Peer cache lookup → first token median"), phase_rows)
            poll_rows = [(row["seed"], _ms(row["storage_ready_to_poll_ms"]),
                          _ms(row["storage_poll_queue_ms"]),
                          _ms(row["storage_poll_execution_ms"]))
                         for row in summary["rows"] if row.get("storage_ready_to_poll_ms") is not None]
            if poll_rows:
                detail += _mode_table(("Seed", "Data ready → poll queued", "Poll queued → dequeued",
                                       "Poll execution"), poll_rows)
            return (f"{len(summary['paired'])} paired seed(s) · baseline, control probes, KV staging · "
                    "native L3 replay hits and staged reuse verified",
                    detail + "<p>Changes are relative to on-demand baseline. The status-commit timestamp "
                             "is observed when a control poll executes, not necessarily when data first "
                             "became usable.</p>")
        if "median_full_prepare_delta_ms" in summary and summary["median_full_prepare_delta_ms"] is None:
            pair_rows = [(pair["seed"], _ms(pair["host_stage_delta_ms"]),
                          _ms(pair.get("host_stage_workflow_delta_ms")),
                          ", ".join(_ms(value) for value in pair["peer_ttft_deltas_ms"]) or "no peers",
                          ", ".join(_ms(value) for value in pair["peer_completion_deltas_ms"]) or "no peers")
                         for pair in summary["paired"]]
            detail += _mode_table(("Seed", "Replay-delay change", "Whole-workload change",
                                   "Each peer's TTFT change", "Each peer's completion change"), pair_rows)
            return (f"{len(summary['paired'])} paired seed(s) · native L3 hit and replay reuse verified · "
                    f"host-stage replay change {summary['median_host_stage_delta_ms']:+.0f} ms",
                    detail + "<p>Negative changes favor staging. This is a controlled, "
                             "file-backed diagnostic, not a production result.</p>")
        return (f"{len(summary['paired'])} paired seed(s) · native L3 hit verified · "
                f"host-stage median change {summary['median_host_stage_delta_ms']:+.0f} ms · "
                f"full-prepare median change {summary['median_full_prepare_delta_ms']:+.0f} ms",
                detail + "<p>Negative change means faster than on-demand. This is a small "
                         "controlled timing check, not a production result.</p>")
    if schema == "agentic_work_audit.tool_cycles.v1":
        if summary.get("status") == "trace_off_control":
            ttft_median, ttft_p95, _ = _tool_cycle_user_metrics(summary)
            rows = [(f"{row['session_id']} turn {row['turn']}", row['prompt_tokens'],
                     _ms(row['first_token_after_tool_ms']),
                     _ms(row['completion_after_tool_ms']))
                    for row in summary['turns'] if row['kind'] == 'active']
            return (f"Trace off · first token median {_ms(ttft_median)}, p95 {_ms(ttft_p95)} · "
                    f"active workflow {_ms(summary['active_workflow_makespan_ms'])}",
                    _mode_table(("Replay", "Prompt tokens", "Tool return to first token",
                                 "Tool return to finish"), rows)
                    + "<p>No backend trace was captured; cache residency and load-back "
                      "cannot be inferred from this control.</p>")
        measure = summary["measurements"]
        headline = (f"{measure['active_replay_count']} replays · "
                    f"first token median {_ms(measure['active_ttft_median_ms'])}, "
                    f"p95 {_ms(measure['active_ttft_p95_ms'])} · "
                    f"KV load-backs {measure['kv_load_back_operations']} · "
                    f"active workflow {_ms(summary['active_workflow_makespan_ms'])}")
        rows = [(f"{row['session_id']} turn {row['turn']}", row['prompt_tokens'],
                 row['matched_prefix_tokens'], row.get('kv_load_back_count', 0),
                 _ms(row['first_token_after_tool_ms']),
                 _ms(row['lookup_to_batch_ms']), _ms(row.get('kv_load_back_call_ms')),
                 _ms(row.get('load_end_to_batch_ms')),
                 _ms(row['completion_after_tool_ms']))
                for row in summary['turns'] if row['kind'] == 'active']
        detail = (_mode_table(("Replay", "Prompt tokens", "Cached prefix", "KV load-backs",
                               "Tool return to first token", "Lookup to first batch", "Load call",
                               "Load end to batch",
                               "Tool return to finish"), rows)
                  + "<p>The first-batch marker is a scheduler-method boundary, not a GPU-completion "
                    "timestamp. Donor traffic alone does not prove KV transfers.</p>")
        return headline, detail
    if schema == "agentic_work_audit.overlap_dose.v1":
        if summary.get("status") == "excluded":
            reason = _esc(summary.get("failure_reason", "run did not complete"))
            return f"Excluded: {reason}", f"<p>No timing comparison: {reason}</p>"
        target = summary["target"]
        physical = summary.get("_physical_overlap")
        submission = summary.get("_decode_submission")
        count = (physical["physical_overlap_load_count"] if physical else
                 summary["realized_worker_window_overlap_count"])
        kind = "physical copies" if physical else "worker windows (proxy)"
        if submission:
            headline = (f"{count}/{summary['donor_count']} {kind} overlapped · "
                        f"CPU-before-launch {_ms(submission['cpu_before_launch_ms'])} · "
                        f"kernel execution {_ms(submission['kernel_execution_ms'])}")
        else:
            headline = (f"{count}/{summary['donor_count']} {kind} overlapped · "
                        f"target first token {_ms(target.get('first_token_after_tool_ms'))} · "
                        f"target finish {_ms(target.get('completion_after_tool_ms'))}")
        rows = []
        for index, donor in enumerate(summary["donors"]):
            copy = next((row for row in (physical or {}).get("donors", [])
                         if row["load_id"] == donor["load_id"]), {})
            rows.append((f"Donor {index + 1}", donor.get("loaded_tokens"),
                         _ms(donor.get("cuda_elapsed_ms")),
                         _ms(donor.get("worker_window_overlap_ms")),
                         _ms(copy.get("copy_during_target_decode_ms")),
                         _ms(copy.get("copy_concurrent_with_target_kernels_ms"))))
        detail = (_mode_table(("Native load", "KV tokens", "CUDA event time",
                               "Worker-window overlap", "Physical copy in decode",
                               "Copy concurrent with kernels"), rows) +
                  f"<p>Target first token after tool return: "
                  f"{_ms(target.get('first_token_after_tool_ms'))}; "
                  f"target completion: {_ms(target.get('completion_after_tool_ms'))}; "
                  f"whole workload: {_ms(summary.get('workflow_makespan_ms'))}. "
                  f"Verified physical copy overlap: "
                  f"{_ms(physical.get('physical_overlap_ms')) if physical else 'not captured'}. "
                  "Worker windows are upper bounds, not proof of device-copy overlap. "
                  "Nsight profiled timing should not be compared with unprofiled latency.</p>")
        if submission:
            control_submission = ((summary.get("_dose_control") or {}).get("decode_submission") or {})
            detail += _mode_table(
                ("Decode measurement", "This run", "Profiled control, if paired"),
                (("Linked forwards", submission["forward_count"],
                  control_submission.get("forward_count", "not paired")),
                 ("Linked kernels", submission["kernel_count"],
                  control_submission.get("kernel_count", "not paired")),
                 ("Kernel execution", _ms(submission["kernel_execution_ms"]),
                  _ms(control_submission.get("kernel_execution_ms"))),
                 ("CPU before launch", _ms(submission["cpu_before_launch_ms"]),
                  _ms(control_submission.get("cpu_before_launch_ms"))),
                 ("Between forwards", _ms(submission["between_forward_gap_ms"]),
                  _ms(control_submission.get("between_forward_gap_ms")))))
        return headline, detail
    if schema == "agentic_work_audit.decode_overlap.v1":
        return _decode_overlap_result(summary)
    if schema == "agentic_work_audit.kv_load_attribution.v1":
        return _attribution_result(summary)
    if schema == "agentic_work_audit.busy_comparison.v1":
        return _busy_result(summary)
    if schema == "agentic_work_audit.controller_window.v1":
        return _controller_window_result(summary)
    if schema == "agentic_work_audit.multisession_window.v1":
        return _multisession_window_result(summary)
    if schema == "agentic_work_audit.multisession_comparison.v1":
        return _multisession_comparison_result(summary)
    if schema == "agentic_work_audit.multisession.v1":
        return _multisession_result(summary)
    if schema == "agentic_work_audit.timing.v1":
        return _timing_result(summary)
    return _lifecycle_result(summary)


def render(summaries: list[tuple[Path, dict]], milestones: list[dict] | None = None) -> str:
    ordered = sorted(summaries, key=lambda item: (_time(item[1])[0], item[0].parent.name), reverse=True)
    run_ids = {str(summary.get("run_id") or path.parent.name) for path, summary in ordered}
    questions, run_questions = _question_index(milestones or [])
    rows: list[str] = []
    for path, summary in ordered:
        started_ns, date, time, source = _time(summary)
        if started_ns >= 0:
            local = datetime.fromtimestamp(started_ns / 1_000_000_000,
                                           ZoneInfo("America/Chicago"))
            date, time = local.strftime("%Y-%m-%d"), local.strftime("%I:%M:%S %p").lstrip("0")
        timing = summary.get("schema") == "agentic_work_audit.timing.v1"
        multisession = summary.get("schema") == "agentic_work_audit.multisession.v1"
        comparison = summary.get("schema") == "agentic_work_audit.multisession_comparison.v1"
        window = summary.get("schema") == "agentic_work_audit.multisession_window.v1"
        controller = summary.get("schema") == "agentic_work_audit.controller_window.v1"
        busy = summary.get("schema") == "agentic_work_audit.busy_comparison.v1"
        attribution = summary.get("schema") == "agentic_work_audit.kv_load_attribution.v1"
        run = str(summary.get("run_id") or path.parent.name)
        kind = _kind(summary)
        manifest_question_id = ((summary.get("_manifest") or {}).get("workload") or {}).get("research_question_id")
        archived_question_id = run_questions.get(run)
        if manifest_question_id and archived_question_id and manifest_question_id != archived_question_id:
            raise ValueError(f"Run {run} has conflicting research question IDs")
        question_id = manifest_question_id or archived_question_id
        milestone = questions.get(question_id)
        fallback_question = (
            "When early worker KV loading overlaps another session's decode, which part of its "
            "response path slows?" if summary.get("schema") == "agentic_work_audit.decode_overlap.v1" else
            "Which replay stages change when the controller checks KV residency and then requests early loads?"
            if attribution else
            "In a busy, naturally evicting system, does using each session's tool-return estimate "
            "to time KV preparation improve the whole workload without harming other sessions?" if busy else
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
        result, findings = _result_parts(summary)
        status = ("physical copy verified" if summary.get("_decode_submission") and
                  (summary.get("_physical_overlap") or {}).get("status") == "verified" else
                  "partial CUDA capture" if summary.get("_cuda_kernel_subset") else
                  str(summary.get("status") or "unknown"))
        finding = _run_finding(summary)
        limits = "".join(f"<li>{_esc(item)}</li>" for item in
                         [*(summary.get("failures") or []), *(summary.get("limitations") or [])])
        detail_id = f"detail-{_esc(quote(run, safe=''))}"
        rows.append(
            f"<tr class='run-row' id='run-{_esc(quote(run, safe=''))}'>"
            f"<td data-label='Date' title='{_esc(source)}'>{_esc(date)}</td>"
            f"<td data-label='Time (Central)' title='{_esc(source)}'>{_esc(time)}</td>"
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
.status{display:inline-block;font-weight:650}.validated{color:#126746}.failed{color:#b63839}.inconclusive{color:#9a621b}
.detail-toggle{border:0;background:none;padding:0;color:#086780;font:inherit;font-weight:650;cursor:pointer;text-decoration:underline}
.detail-toggle:focus-visible{outline:2px solid #086780;outline-offset:3px}
.detail-row[hidden]{display:none!important}.detail-row>td{padding:16px 20px;background:#f5f9f8;white-space:normal}
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
<li><strong>Session resumes</strong><span>Early, late, and controller timing measured; RQ8 tests busy cache pressure, and RQ9 tests off-scheduler loading.</span></li>
<li><strong>HBM occupancy</strong><span>Useful, idle, and dead block-seconds not yet measured.</span></li>
<li><strong>GPU time</strong><span>Useful compute, recompute, and idle-with-stageable-work not yet measured.</span></li>
</ul></section>""" + progress_html + """
<p class="intro">One row per saved experiment, newest first. Main result shows the measurements; Finding states the run-specific deduction. The research-question link opens the broader answer above. Date and time are UTC from the first recorded request; a completion-time fallback is labeled on hover. Lifecycle timing is not a policy win.</p>
<div class="table-scroll"><table class="results-table"><colgroup><col style="width:8%"><col style="width:7%"><col style="width:11%"><col style="width:14%"><col style="width:14%"><col style="width:17%"><col style="width:15%"><col style="width:8%"><col style="width:6%"></colgroup><thead><tr><th>Date</th><th>Time (Central)</th><th>Experiment</th><th>Research question</th><th>Setup</th><th>Main result</th><th>Finding</th><th>Evidence gate</th><th>Details</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
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


class _ReportText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.command_parts: list[str] = []
        self.in_pre = False

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            self.in_pre = True
        elif tag in ("br", "p", "li", "tr"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "pre":
            self.in_pre = False
        elif tag in ("p", "li", "tr"):
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
        if self.in_pre:
            self.command_parts.append(data)


def _report_text(fragment: str) -> str:
    parser = _ReportText()
    parser.feed(fragment)
    return " ".join("".join(parser.parts).split())


def _reproduction_command(summary: dict) -> str | None:
    parser = _ReportText()
    parser.feed(_reproduction(summary, summary.get("schema") == "agentic_work_audit.timing.v1"))
    return "".join(parser.command_parts).strip() or None


def _md_cell(value: object) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _md_nowrap(value: object) -> str:
    return _md_cell(value).replace(" ", "&nbsp;")


def _md_table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    if not rows:
        return "No comparable measurements recorded."
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_md_cell(cell) for cell in row) + " |" for row in rows]
    return "\n".join((head, rule, *body))


def _markdown_metrics(summary: dict) -> str:
    schema = summary.get("schema")
    if schema == "agentic_work_audit.storage_cycles.summary.v1":
        rows = [(arm["seed"], arm["arm"], arm["session_count"], arm["replay_count"],
                 arm["workflow_duration_ms"], arm["due_to_first_token_median_ms"],
                 arm["due_to_first_token_p95_ms"], arm["replay_ttft_median_ms"],
                 arm["natural_storage_candidate_waits"], arm["stage_before_due_count"],
                 len(arm.get("preparation_skips") or []), arm["stage_loaded_tokens"],
                 arm["native_replay_storage_hit_count"],
                 arm["native_replay_storage_hit_tokens"], len(arm["preparation_errors"]))
                for arm in summary["arms"]]
        return _md_table(("Seed", "Mode", "Sessions", "Replays", "Whole workload (ms)",
                          "Due → first token median (ms)", "Due → first token p95 (ms)",
                          "Replay TTFT median (ms)", "Storage candidates", "Stage ready by due",
                          "Stage skips", "Tokens staged from L3", "Native L3 replay hits", "L3 replay tokens",
                          "Stage errors"), rows)
    if schema == "agentic_work_audit.storage_replay.v1":
        rows = [(row['seed'], row['arm'], row['due_to_first_token_ms'], row['replay_ttft_ms'],
                 row.get('peer_ttft_median_ms'), row.get('peer_completion_median_ms'),
                 row.get('peers_overlapping_preparation'),
                 row.get('storage_data_ready_ms'), row.get('storage_commit_after_ready_ms'),
                 row.get('workflow_duration_ms'), row['native_replay_storage_hit_tokens'],
                 row['control_storage_hit_tokens'], row.get('replay_matched_prefix_tokens'),
                 "not applicable" if row['arm'] in {'on_demand', 'control_only'} else
                 row['stage_completed_before_due'])
                for row in summary['rows']]
        table = _md_table(("Seed", "Arm", "Due → first token (ms)", "Replay TTFT (ms)",
                          "Peer TTFT median (ms)", "Peer completion median (ms)",
                          "Peers overlapping preparation",
                          "Native L3 → host ready (ms)", "Ready → status observed (ms)",
                          "Whole workflow (ms)", "L3 tokens at replay", "L3 tokens in wait",
                          "Matched prefix tokens",
                          "Stage ready by due"), rows)
        if any(row['arm'] == 'control_only' for row in summary['rows']):
            phase_rows = [(row['seed'], row['arm'],
                           median(peer['request_to_lookup_ms'] for peer in row['peer_phase_times']),
                           median(peer['lookup_to_first_token_ms'] for peer in row['peer_phase_times']))
                          for row in summary['rows'] if row.get('peer_phase_times')
                          and all(peer['request_to_lookup_ms'] is not None
                                  and peer['lookup_to_first_token_ms'] is not None
                                  for peer in row['peer_phase_times'])]
            if phase_rows:
                table += "\n\n" + _md_table(("Seed", "Arm", "Peer request → cache lookup median (ms)",
                                                 "Peer cache lookup → first token median (ms)"), phase_rows)
            poll_rows = [(row['seed'], row['storage_ready_to_poll_ms'],
                          row['storage_poll_queue_ms'], row['storage_poll_execution_ms'])
                         for row in summary['rows'] if row.get('storage_ready_to_poll_ms') is not None]
            if poll_rows:
                table += "\n\n" + _md_table(("Seed", "Data ready → poll queued (ms)",
                                                 "Poll queued → dequeued (ms)", "Poll execution (ms)"), poll_rows)
        return table
    if schema == "agentic_work_audit.tool_cycles.v1":
        if summary.get("status") == "trace_off_control":
            rows = [(f"{row['session_id']} · {row['turn']}", row['prompt_tokens'],
                     row['first_token_after_tool_ms'], row['completion_after_tool_ms'])
                    for row in summary['turns'] if row['kind'] == 'active']
            return (_md_table(("Session · turn", "Prompt tokens", "Tool return → first token (ms)",
                               "Tool return → finish (ms)"), rows)
                    + "\n\nTrace disabled: no backend stages or KV-load evidence was captured.")
        rows = [(f"{row['session_id']} · {row['turn']}", row['prompt_tokens'],
                 row['matched_prefix_tokens'], row.get('kv_load_back_count', 0),
                 row['first_token_after_tool_ms'],
                 row['lookup_to_batch_ms'], row.get('kv_load_back_call_ms'),
                 row.get('load_end_to_batch_ms'), row['completion_after_tool_ms'])
                for row in summary['turns'] if row['kind'] == 'active']
        return (_md_table(("Session · turn", "Prompt tokens", "Cached prefix tokens", "KV load-backs",
                           "Tool return → first token (ms)", "Lookup → batch (ms)",
                           "Load call (ms)", "Load end → batch (ms)",
                           "Tool return → finish (ms)"), rows)
                + "\n\nThe batch boundary is a scheduler-method timestamp, not measured GPU completion. "
                  "Donor traffic is not proof of KV movement.")
    if schema == "agentic_work_audit.decode_overlap.v1":
        rows = []
        for case in summary.get("cases") or []:
            short = case.get("short_decode") or {}
            batches = short.get("decode_batches") or []
            def total(key: str) -> float | None:
                values = [batch[key] for batch in batches if isinstance(batch.get(key), (int, float))]
                return round(sum(values), 1) if values else None
            rows.append((f"Trial {case.get('pair')} · {case.get('condition')}",
                         (case.get("audit") or {}).get("sessions", {}).get("short", {}).get("first_token_after_tool_ms"),
                         (case.get("audit") or {}).get("sessions", {}).get("short", {}).get("completion_after_tool_ms"),
                         total("duration_ms"), total("model_forward_ms"),
                         total("non_forward_ms"),
                         round(sum(short.get("inter_batch_gaps_ms") or []), 1),
                         (case.get("load_overlap") or {}).get("short_decode_overlap_ms")))
        table = _md_table(("Trial / mode", "Short first token (ms)", "Short finish (ms)",
                           "Batch time (ms)", "Model forward (ms)", "Other batch time (ms)",
                           "Between batches (ms)", "Load overlap (ms)"), rows)
        cuda = summary.get("_cuda_kernel_subset")
        if cuda:
            kernel_rows = []
            for pair in cuda.get("pairs") or []:
                for mode in ("post_short", "early"):
                    arm = pair[mode]
                    kernel_rows.append((f"Pair {pair['pair']} · {mode}", arm["kernel_count"],
                                        arm["kernel_duration_sum_ms"],
                                        arm["kernel_gap_inside_span_ms"],
                                        arm["htod_during_kernel_span_ms"]))
            table += ("\n\n**Nsight GPU check (captured pair only).** The full profiler run lost "
                      "later CUDA data; these rows have complete kernel linkage. Profiled time is "
                      "mechanism evidence, not the clean performance estimate.\n\n" +
                      _md_table(("Captured case", "Kernels", "Kernel execution (ms)",
                                 "Between-kernel gaps (ms)", "H-to-D overlap (ms)"), kernel_rows))
            launch = summary.get("_cuda_launch_gaps")
            if launch:
                launch_rows = []
                for mode in ("post_short", "early"):
                    arm = launch[mode]
                    launch_rows.append((mode, arm["gap_ms"], arm["cpu_before_launch_ms"],
                                        arm["launch_api_ms"], arm["after_launch_api_ms"]))
                table += ("\n\n**CUDA launch check (same captured pair).** Most added gap "
                          "time passed before the CPU started the next launch. This does not "
                          "identify why the host waited or measure global GPU idle time.\n\n" +
                          _md_table(("Captured case", "Total gaps (ms)", "Before CPU launch (ms)",
                                     "During launch API (ms)", "After launch API (ms)"),
                                    launch_rows))
                table += ("\n\nRecorded stream-wait event activity was "
                          f"{launch['post_short']['stream_wait_event_gpu_ms']:.3f} ms after-short "
                          f"and {launch['early']['stream_wait_event_gpu_ms']:.3f} ms early; "
                          "blocking CUDA synchronization API time was "
                          f"{launch['post_short']['blocking_sync_api_ms']:.3f} ms and "
                          f"{launch['early']['blocking_sync_api_ms']:.3f} ms. "
                          "These are not additive to the gap categories.")
        return table
    if schema == "agentic_work_audit.overlap_dose.v1":
        if summary.get("status") == "excluded":
            return _md_table(("Measurement", "Value"),
                             (("Sessions", summary.get("session_count")),
                              ("Planned overlapping loads", summary.get("planned_overlap")),
                              ("Failure", summary.get("failure_reason")),
                              ("Comparable timing", "unavailable")))
        physical = summary.get("_physical_overlap")
        profile = summary.get("_profile_status")
        submission = summary.get("_decode_submission")
        control = summary.get("_dose_control")
        target = summary["target"]
        rows = [("Sessions", summary["session_count"]),
                ("Planned overlapping loads", summary["planned_overlap"]),
                ("Worker-window overlaps (proxy)", summary["realized_worker_window_overlap_count"]),
                ("Physical H-to-D overlaps", physical.get("physical_overlap_load_count")
                 if physical else "not captured"),
                ("Physical copy overlap (ms)", physical.get("physical_overlap_ms")
                 if physical else "not captured"),
                ("Copy concurrent with decode kernels (ms)",
                 physical.get("concurrent_kernel_copy_ms") if physical else "not captured"),
                ("Profiler status", "CUDA capture verified" if submission else
                 profile.get("status") if profile else "not requested"),
                ("Target first token after tool return (ms)", target.get("first_token_after_tool_ms")),
                ("Target finish after tool return (ms)", target.get("completion_after_tool_ms")),
                ("Whole workload (ms)", summary.get("workflow_makespan_ms"))]
        if submission:
            rows.extend((("Linked decode forwards", submission["forward_count"]),
                         ("Linked decode kernels", submission["kernel_count"]),
                         ("Summed kernel execution (ms)", submission["kernel_execution_ms"]),
                         ("Inside-forward CPU-before-launch gaps (ms)",
                          submission["cpu_before_launch_ms"]),
                         ("Between-forward gaps (ms)", submission["between_forward_gap_ms"])))
        control_submission = (control or {}).get("decode_submission")
        if submission and control_submission:
            rows.extend((("Kernel execution change vs profiled control (ms)",
                          round(submission["kernel_execution_ms"] -
                                control_submission["kernel_execution_ms"], 3)),
                         ("CPU-before-launch change vs profiled control (ms)",
                          round(submission["cpu_before_launch_ms"] -
                                control_submission["cpu_before_launch_ms"], 3))))
        if control:
            rows.extend((("Matched control target finish (ms)", round(control["target_ms"], 3)),
                         ("Target finish change vs control (ms)",
                          round(target["completion_after_tool_ms"] - control["target_ms"], 3)),
                         ("Peer median finish change vs control (ms)",
                          round(median(row["completion_after_tool_ms"] for row in
                                       summary["active_requests"][1:]) - control["peer_median_ms"], 3))))
        active_rows = [("Target" if index == 0 else f"Peer {index}",
                        row.get("completion_tokens"), row.get("first_token_after_tool_ms"),
                        row.get("completion_after_tool_ms"))
                       for index, row in enumerate(summary["active_requests"])]
        donor_rows = []
        for index, donor in enumerate(summary["donors"], start=1):
            copy = next((item for item in (physical or {}).get("donors", [])
                         if item["load_id"] == donor["load_id"]), {})
            donor_rows.append((index, donor.get("loaded_tokens"), donor.get("cuda_elapsed_ms"),
                               donor.get("worker_window_overlap_ms"),
                               copy.get("copy_during_target_decode_ms", "not captured"),
                               copy.get("copy_concurrent_with_target_kernels_ms", "not captured"),
                               (summary.get("donor_replay_ttft_ms") or [None] * len(summary["donors"]))[index - 1]))
        return (_md_table(("Measurement", "Value"), rows) + "\n\n" +
                _md_table(("Active replay", "Output tokens", "First token after tool ms",
                           "Finish after tool ms"), active_rows) + "\n\n" +
                _md_table(("Donor", "KV tokens", "CUDA event ms", "Worker window ms",
                           "Physical copy in decode ms", "Copy with target kernels ms",
                           "Donor replay TTFT ms"), donor_rows) +
                "\n\nWorker windows are timing proxies, not proof of physical copy overlap. "
                "Profiled timing must not be used as an unprofiled slowdown estimate.")
    if schema == "agentic_work_audit.kv_load_attribution.v1":
        rows = []
        for seed in summary.get("seeds") or []:
            for mode, arm in (seed.get("arms") or {}).items():
                rows.append((f"Seed {seed.get('seed')} · {mode}", arm.get("replay_count"),
                             arm.get("total_replay_ttft_ms"), arm.get("workflow_makespan_ms"),
                             len(arm.get("load_phases") or [])))
        return _md_table(("Arm", "Replays", "Total replay TTFT (ms)",
                          "Workflow (ms)", "Recorded load phases"), rows)
    if schema == "agentic_work_audit.busy_comparison.v1":
        rows = []
        for pair in summary.get("pairs") or []:
            for mode in ("baseline", "controller"):
                arm = pair.get(mode) or {}
                rows.append((f"Seed {pair.get('seed')} · {mode}", pair.get("replay_count"),
                             arm.get("total_replay_ttft_ms"), arm.get("workflow_makespan_ms"),
                             arm.get("native_load_events")))
        return _md_table(("Arm", "Replays", "Total replay TTFT (ms)",
                          "Workflow (ms)", "Native loads"), rows)
    if schema in ("agentic_work_audit.multisession_comparison.v1",
                  "agentic_work_audit.multisession_window.v1",
                  "agentic_work_audit.controller_window.v1"):
        rows = []
        for case in summary.get("cases") or []:
            if case.get("warmup"):
                continue
            sessions = case.get("sessions") or {}
            long = sessions.get("long") or {}
            short = sessions.get("short") or {}
            rows.append((f"Trial {case.get('pair')} · {case.get('load_timing')}",
                         long.get("first_token_after_tool_ms"),
                         short.get("completion_after_tool_ms"),
                         case.get("workflow_makespan_ms"),
                         long.get("load_completion_observed_after_tool_ms")))
        return _md_table(("Trial / mode", "Long first token after tool (ms)",
                          "Short finish after tool (ms)", "Workflow (ms)",
                          "Load complete relative to tool return (ms)"), rows)
    if schema == "agentic_work_audit.multisession.v1":
        rows = [(label, session.get("observed_tool_wait_ms"),
                 session.get("first_token_after_tool_ms"), session.get("replay_ttft_ms"),
                 session.get("cached_prefix_tokens"))
                for label, session in (summary.get("sessions") or {}).items()]
        return _md_table(("Session", "Tool wait (ms)", "First token after tool (ms)",
                          "Replay TTFT (ms)", "Cached prefix tokens"), rows)
    if schema == "agentic_work_audit.timing.v1":
        rows = [(f"Trial {case.get('pair')} · {case.get('condition')}",
                 case.get("submission_after_due_ms"), case.get("first_token_after_due_ms"),
                 case.get("replay_ttft_ms"), case.get("task_latency_ms"))
                for case in summary.get("cases") or [] if case.get("status") == "validated"]
        return _md_table(("Trial / mode", "Submit after due (ms)",
                          "First token after due (ms)", "Replay TTFT (ms)",
                          "Task duration (ms)"), rows)
    rows = [(case.get("case_type"), case.get("replay_ttft_ms"),
             case.get("host_resident_tokens"), case.get("native_loaded_tokens"),
             case.get("max_cached_prefix_tokens"))
            for case in summary.get("cases") or []]
    return _md_table(("Case", "Replay TTFT (ms)", "Host tokens", "Loaded tokens",
                      "Largest matched prefix (tokens)"), rows)


def _central_stamp(summary: dict) -> tuple[str, str]:
    timestamp_ns, _date, _time_utc, source = _time(summary)
    if timestamp_ns < 0:
        return "Not recorded", source
    instant = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, ZoneInfo("America/Chicago"))
    date = f"{instant.strftime('%b')} {instant.day}, {instant.year}"
    clock = instant.strftime("%I:%M:%S").lstrip("0")
    period = "a.m." if instant.hour < 12 else "p.m."
    return f"{date}, {clock} {period} {instant.tzname()}", source.replace(" (UTC)", "")


def _arrow(before: object, after: object, *, seconds: bool = False,
           lower: str = "faster", higher: str = "slower") -> str:
    if not isinstance(before, (int, float)) or not isinstance(after, (int, float)):
        return "Not measured"
    scale = 1000 if seconds else 1
    precision = 1 if seconds else 0
    unit = "s" if seconds else "ms"
    change = "unchanged" if abs(after - before) < 0.5 else lower if after < before else higher
    if change == "unchanged":
        return f"{before / scale:,.{precision}f} → {after / scale:,.{precision}f} {unit} (unchanged)"
    return (f"{before / scale:,.{precision}f} → {after / scale:,.{precision}f} {unit} "
            f"({abs(after - before) / scale:,.{precision}f} {unit} {change})")


def _median_field(rows: list[dict], field: str) -> float | None:
    values = [row.get(field) for row in rows]
    return median(values) if values and all(isinstance(value, (int, float)) for value in values) else None


def _paired_index_outcome(summary: dict) -> tuple[str, str, str, str, str, str]:
    schema = summary.get("schema")
    if schema == "agentic_work_audit.controller_window.v1":
        target_mode, target_label = "controller_window", "controller-timed loading"
    elif schema == "agentic_work_audit.multisession_window.v1":
        target_mode, target_label = "post_short", "post-short loading"
    else:
        target_mode, target_label = "early", "early loading"
    valid_pairs = {pair.get("pair") for pair in summary.get("pairs") or []
                   if pair.get("comparable") and pair.get("pair") is not None}
    cases = {(case.get("pair"), case.get("load_timing")): case
             for case in summary.get("cases") or [] if not case.get("warmup")}
    selected = [(cases[(pair, "late_nonblocking")], cases[(pair, target_mode)])
                for pair in sorted(valid_pairs) if (pair, "late_nonblocking") in cases and
                (pair, target_mode) in cases]
    comparison = f"Late loading → {target_label}"
    if not selected:
        return comparison, "Comparison withheld", "Not measured", "Not measured", \
            "The saved evidence did not validate a matched speed comparison.", "No matched pair"
    count = len(selected)
    comparison += f" (per-mode median, {count} pairs)" if count > 1 else " (1 pair)"
    def values(session: str, field: str) -> tuple[float | None, float | None]:
        return (_median_field([((before.get("sessions") or {}).get(session) or {})
                              for before, _after in selected], field),
                _median_field([((after.get("sessions") or {}).get(session) or {})
                              for _before, after in selected], field))
    long_before, long_after = values("long", "first_token_after_tool_ms")
    short_before, short_after = values("short", "completion_after_tool_ms")
    workflow_before = _median_field([before for before, _after in selected], "workflow_makespan_ms")
    workflow_after = _median_field([after for _before, after in selected], "workflow_makespan_ms")
    def trend(before_values: list[object], after_values: list[object],
              improved: str, worsened: str, varied: str) -> str:
        if not all(isinstance(old, (int, float)) and isinstance(new, (int, float))
                   for old, new in zip(before_values, after_values)):
            return "effect not measured"
        changes = [new - old for old, new in zip(before_values, after_values)]
        if all(change < 0 for change in changes):
            return improved
        if all(change > 0 for change in changes):
            return worsened
        return varied
    long_trend = trend(
        [((before.get("sessions") or {}).get("long") or {}).get("first_token_after_tool_ms")
         for before, _after in selected],
        [((after.get("sessions") or {}).get("long") or {}).get("first_token_after_tool_ms")
         for _before, after in selected],
        "replay sooner", "replay later", "replay effect varied")
    short_trend = trend(
        [((before.get("sessions") or {}).get("short") or {}).get("completion_after_tool_ms")
         for before, _after in selected],
        [((after.get("sessions") or {}).get("short") or {}).get("completion_after_tool_ms")
         for _before, after in selected],
        "short session finished sooner", "short session finished later", "short-session effect varied")
    workflow_trend = trend([before.get("workflow_makespan_ms") for before, _after in selected],
                           [after.get("workflow_makespan_ms") for _before, after in selected],
                           "workflow sooner", "workflow later",
                           "workflow effect varied")
    scope = "1 pair" if count == 1 else f"{count} pairs"
    finding = f"{long_trend.capitalize()}; {short_trend}; {workflow_trend} ({scope})."
    return (comparison, _arrow(long_before, long_after),
            _arrow(short_before, short_after, lower="earlier", higher="later"),
            _arrow(workflow_before, workflow_after, lower="sooner", higher="later"),
            finding, f"{summary.get('status', 'unknown')}; {count} {'pair' if count == 1 else 'pairs'}")


def _busy_index_outcome(summary: dict) -> tuple[str, str, str, str, str, str]:
    attribution = summary.get("schema") == "agentic_work_audit.kv_load_attribution.v1"
    rows = []
    for item in (summary.get("seeds") if attribution else summary.get("pairs")) or []:
        arms = item.get("arms") if attribution else item
        before = (arms or {}).get("check_only" if attribution else "baseline") or {}
        after = (arms or {}).get("controller") or {}
        if before and after:
            rows.append((before, after, item))
    comparison = "Checks only → checks + early loads" if attribution else "Ordinary replay → controller-timed loads"
    if not rows:
        return comparison, "Not measured", "Not isolated", "Not measured", \
            "No paired workload result was saved.", "No paired seed"
    before_ttft = _median_field([before for before, _after, _item in rows], "total_replay_ttft_ms")
    after_ttft = _median_field([after for _before, after, _item in rows], "total_replay_ttft_ms")
    before_workflow = _median_field([before for before, _after, _item in rows], "workflow_makespan_ms")
    after_workflow = _median_field([after for _before, after, _item in rows], "workflow_makespan_ms")
    replay_count = rows[0][2].get("replay_count")
    if replay_count is None:
        replay_count = rows[0][1].get("replay_count")
    comparison += f" (per-arm median, {len(rows)} seeds)" if len(rows) > 1 else " (1 seed)"
    if replay_count:
        comparison += f"; {replay_count} replays/seed"
    replay_worse = all(isinstance(before.get("total_replay_ttft_ms"), (int, float)) and
                       isinstance(after.get("total_replay_ttft_ms"), (int, float)) and
                       after["total_replay_ttft_ms"] > before["total_replay_ttft_ms"]
                       for before, after, _item in rows)
    workflow_worse = all(isinstance(before.get("workflow_makespan_ms"), (int, float)) and
                         isinstance(after.get("workflow_makespan_ms"), (int, float)) and
                         after["workflow_makespan_ms"] > before["workflow_makespan_ms"]
                         for before, after, _item in rows)
    if replay_worse and workflow_worse:
        finding = "Combined replay first-token time and total workload time both increased in this sample."
    elif replay_worse and all(isinstance(before.get("workflow_makespan_ms"), (int, float)) and
                              isinstance(after.get("workflow_makespan_ms"), (int, float)) and
                              after["workflow_makespan_ms"] < before["workflow_makespan_ms"]
                              for before, after, _item in rows):
        finding = "Combined replay first-token time increased, although the workload finished sooner."
    elif before_ttft is not None and after_ttft is not None:
        finding = "Replay and workflow effects differed; see the per-seed measurements."
    else:
        finding = "The saved run does not contain a comparable replay total."
    if attribution:
        other = "Other-session effect not isolated"
    else:
        helped = [item.get("sessions_helped") for _before, _after, item in rows]
        harmed = [item.get("sessions_harmed") for _before, _after, item in rows]
        other = (f"Per seed: {helped[0]} helped; {harmed[0]} harmed"
                 if len(set(helped)) == 1 and len(set(harmed)) == 1 and
                 helped[0] is not None and harmed[0] is not None else "Per-session effect varies; see details")
    return (comparison, _arrow(before_ttft, after_ttft, seconds=True, lower="lower", higher="higher"),
            other, _arrow(before_workflow, after_workflow, seconds=True, lower="sooner", higher="later"),
            finding, f"{summary.get('status', 'unknown')}; {len(rows)} {'seeds' if len(rows) > 1 else 'seed'}")


def _timing_index_outcome(summary: dict) -> tuple[str, str, str, str, str, str]:
    cases = {(case.get("pair"), case.get("condition")): case for case in summary.get("cases") or []}
    pairs = [pair for pair in summary.get("pairs") or [] if pair.get("comparable")]
    if any(case.get("condition") == "late_nonblocking" for case in summary.get("cases") or []):
        return ("Late blocking → late nonblocking (comparison withheld)",
                "No validated replay-speed delta", "No other session", "Full-task comparison withheld",
                "Submission was faster, but the measured first-token delay remained; the strict comparison was withheld.",
                summary.get("status", "unknown"))
    selected = [(cases[(pair.get("pair"), "late")], cases[(pair.get("pair"), "early")])
                for pair in pairs if (pair.get("pair"), "late") in cases and
                (pair.get("pair"), "early") in cases]
    if not selected:
        return "Late → early loading", "Comparison withheld", "No other session", \
            "Not comparable", "No matched replay comparison was validated.", summary.get("status", "unknown")
    before = _median_field([old for old, _new in selected], "first_token_after_due_ms")
    after = _median_field([new for _old, new in selected], "first_token_after_due_ms")
    count = len(selected)
    consistent_benefit = all(isinstance(old.get("first_token_after_due_ms"), (int, float)) and
                             isinstance(new.get("first_token_after_due_ms"), (int, float)) and
                             new["first_token_after_due_ms"] < old["first_token_after_due_ms"]
                             for old, new in selected)
    return (f"Late → early loading (per-mode median, {count} pairs)" if count > 1 else "Late → early loading (1 pair)",
            _arrow(before, after), "No other session", "Full-task effect not established",
            "Loading during the tool wait brought the first token sooner in the matched replays."
            if consistent_benefit else
            "The matched replays did not show a consistent first-token benefit.",
            f"{summary.get('status', 'unknown')}; {count} {'pairs' if count > 1 else 'pair'}")


def _markdown_index_outcome(summary: dict) -> tuple[str, str, str, str, str, str]:
    schema = summary.get("schema")
    if schema == "agentic_work_audit.storage_cycles.summary.v1":
        pairs = summary.get("paired_comparisons") or []
        if not pairs:
            return ("Calibration only", "No paired comparison", "No peer comparison",
                    "Not established", _run_finding(summary), "pilot")
        baseline = [arm for arm in summary["arms"] if arm["arm"] == "on_demand"]
        staged = [arm for arm in summary["arms"] if arm["arm"] == "host_stage"]
        base_delay = median(arm["due_to_first_token_median_ms"] for arm in baseline)
        stage_delay = median(arm["due_to_first_token_median_ms"] for arm in staged)
        base_work = median(arm["workflow_duration_ms"] for arm in baseline)
        stage_work = median(arm["workflow_duration_ms"] for arm in staged)
        return (("Blocked: on demand → host staging" if summary.get("status") == "blocked"
                 else "No effective staging: on demand → host staging"
                 if summary.get("status") == "insufficient_exposure"
                 else "On demand → host staging"),
                f"Replay delay: {base_delay:.0f} → {stage_delay:.0f} ms",
                f"Whole workload: {base_work / 1000:.2f} → {stage_work / 1000:.2f} s",
                ("Blocked by backend assertion; incomplete paired study"
                 if any(row.get("backend_assertion") for row in summary.get("failed_arms") or []) else
                 "Blocked by preparation error or incomplete arm")
                if summary.get("status") == "blocked" else
                "Synthetic file-backed storage; full-GPU preparation not tested",
                _run_finding(summary), "blocked" if summary.get("status") == "blocked"
                else "native L3 hits verified")
    if schema == "agentic_work_audit.storage_replay.v1":
        if any(row['arm'] == 'control_only' for row in summary['rows']):
            by_arm = {arm: [row for row in summary['rows'] if row['arm'] == arm]
                      for arm in ('on_demand', 'control_only', 'host_stage')}
            replay = [median(row['due_to_first_token_ms'] for row in by_arm[arm])
                      for arm in by_arm]
            peer = [median(row['peer_completion_median_ms'] for row in by_arm[arm])
                    for arm in by_arm]
            workflow = [median(row['workflow_duration_ms'] for row in by_arm[arm])
                        for arm in by_arm]
            return ('No early action → control checks → KV staging',
                    'First token: ' + ' → '.join(f'{value:.0f}' for value in replay) + ' ms',
                    'Peer finish: ' + ' → '.join(f'{value:.0f}' for value in peer) + ' ms',
                    'Whole workload: ' + ' → '.join(f'{value / 1000:.2f}' for value in workflow) + ' s',
                    'Early staging helped the returning session but delayed peers; control checks alone '
                    'explain only part of the peer delay.',
                    'Native L3 hits and replay reuse verified')
        if "median_full_prepare_delta_ms" in summary and summary["median_full_prepare_delta_ms"] is None:
            base = median(row["due_to_first_token_ms"] for row in summary["rows"]
                          if row["arm"] == "on_demand")
            staged = median(row["due_to_first_token_ms"] for row in summary["rows"]
                            if row["arm"] == "host_stage")
            base_work = median(row["workflow_duration_ms"] for row in summary["rows"]
                               if row["arm"] == "on_demand")
            stage_work = median(row["workflow_duration_ms"] for row in summary["rows"]
                                if row["arm"] == "host_stage")
            peer_count = max((row.get("peer_count") or 0 for row in summary["rows"]), default=0)
            peer = "No peer session"
            if peer_count:
                base_peer = median(row["peer_ttft_median_ms"] for row in summary["rows"]
                                   if row["arm"] == "on_demand")
                stage_peer = median(row["peer_ttft_median_ms"] for row in summary["rows"]
                                    if row["arm"] == "host_stage")
                peer = f"Peer TTFT: {base_peer:.0f} → {stage_peer:.0f} ms"
            return ("On demand → host stage", f"First token: {base:.0f} → {staged:.0f} ms",
                    peer, f"Whole workload: {base_work / 1000:.2f} → {stage_work / 1000:.2f} s",
                    _run_finding(summary), "L3 hit and replay reuse verified")
        arms = {
            arm: {
                key: median(row[key] for row in summary['rows'] if row['arm'] == arm)
                for key in ('due_to_first_token_ms', 'peer_ttft_median_ms')
                if all(row.get(key) is not None for row in summary['rows'] if row['arm'] == arm)
            }
            for arm in ('on_demand', 'host_stage', 'full_prepare')
        }
        base = arms['on_demand']['due_to_first_token_ms']
        full = arms['full_prepare']['due_to_first_token_ms']
        peer_count = max((row.get('peer_count') or 0 for row in summary['rows']), default=0)
        peer = (f"Peer TTFT: {arms['on_demand']['peer_ttft_median_ms']:.0f} → "
                f"{arms['host_stage']['peer_ttft_median_ms']:.0f} → "
                f"{arms['full_prepare']['peer_ttft_median_ms']:.0f} ms") if peer_count else "No peer session"
        return ("On demand → host stage → full prepare",
                f"First token: {base:.0f} → {arms['host_stage']['due_to_first_token_ms']:.0f} "
                f"→ {full:.0f} ms", peer,
                "Small synthetic workload", _run_finding(summary), "L3 hit verified")
    if schema == "agentic_work_audit.tool_cycles.v1":
        if summary.get("status") == "trace_off_control":
            ttft_median, _, _ = _tool_cycle_user_metrics(summary)
            return (f"Trace-off control; {summary['turn_count']} tool returns per session; "
                    f"{summary['donor_count']} donor sessions",
                    f"Median first token: {_ms(ttft_median)}", "Backend stages: not captured",
                    f"Active workflow: {summary['active_workflow_makespan_ms'] / 1000:.1f} s",
                    _run_finding(summary), "trace_disabled")
        measure = summary['measurements']
        return (f"{summary['turn_count']} tool returns per session; "
                f"{summary['donor_count']} donor sessions",
                f"Median first token: {_ms(measure['active_ttft_median_ms'])}",
                f"Lookup → batch: {_ms(measure['active_lookup_to_batch_median_ms'])}",
                f"Active workflow: {summary['active_workflow_makespan_ms'] / 1000:.1f} s",
                _run_finding(summary), summary.get('evidence_status', 'unknown'))
    if schema == "agentic_work_audit.overlap_dose.v1":
        if summary.get("status") == "excluded":
            return ("Excluded diagnostic", "No comparable replay", "No comparable peer",
                    "Not measured", _run_finding(summary), "excluded")
        physical = summary.get("_physical_overlap")
        profile = summary.get("_profile_status")
        submission = summary.get("_decode_submission")
        control = summary.get("_dose_control")
        count = (physical["physical_overlap_load_count"] if physical else
                 summary["realized_worker_window_overlap_count"])
        label = "verified copies" if physical else "worker-window proxies"
        if submission:
            return (f"0 → {count} {label}" if control else
                    f"Observed {count} {label}; no profiled control",
                    "Profiled mechanism; no latency claim",
                    "See target launch timing in details",
                    "Not used for speed comparison",
                    _run_finding(summary), "physical copy verified")
        others = [row.get("completion_after_tool_ms") for row in summary["active_requests"][1:]]
        others = [value for value in others if isinstance(value, (int, float))]
        target_ms = summary["target"].get("completion_after_tool_ms")
        peer_ms = median(others) if others else None
        target_text = (f"{_ms(control['target_ms'])} → {_ms(target_ms)}" if control else
                       f"Target finish: {_ms(target_ms)}")
        peer_text = (f"{_ms(control['peer_median_ms'])} → {_ms(peer_ms)}" if control and peer_ms else
                     f"Other active decoders: median {_ms(peer_ms) if peer_ms else 'not measured'}")
        return (f"0 → {count} {label}" if control else
                f"Planned {summary['planned_overlap']} → observed {count} {label}; single dose",
                target_text, peer_text,
                f"{summary['workflow_makespan_ms'] / 1000:.1f} s",
                _run_finding(summary), "physical copy verified" if physical else
                "profiler incomplete" if profile else "worker-window proxy only")
    if schema == "agentic_work_audit.decode_overlap.v1":
        if summary.get("_cuda_kernel_subset"):
            return ("GPU mechanism check; profiled timing not used", "See clean RQ10 run",
                    "See captured kernel table", "No profiled workflow claim",
                    _run_finding(summary), "validated timing; partial CUDA capture")
        pairs = [pair for pair in summary.get("pairs") or [] if pair.get("comparable")]
        def med(key: str) -> float | None:
            values = [pair[key] for pair in pairs if isinstance(pair.get(key), (int, float))]
            return median(values) if values else None
        return ("After-short → early worker load, both before tool return",
                _arrow(med("post_short_long_first_token_ms"), med("early_long_first_token_ms")),
                _arrow(med("post_short_finish_ms"), med("early_short_finish_ms")),
                _arrow(med("post_short_workflow_ms"), med("early_workflow_ms"), seconds=True),
                _run_finding(summary),
                "validated timing; partial CUDA capture" if summary.get("_cuda_kernel_subset")
                else summary.get("status", "unknown"))
    if schema in ("agentic_work_audit.multisession_comparison.v1",
                  "agentic_work_audit.multisession_window.v1",
                  "agentic_work_audit.controller_window.v1"):
        return _paired_index_outcome(summary)
    if schema in ("agentic_work_audit.kv_load_attribution.v1",
                  "agentic_work_audit.busy_comparison.v1"):
        return _busy_index_outcome(summary)
    if schema == "agentic_work_audit.timing.v1":
        return _timing_index_outcome(summary)
    if schema == "agentic_work_audit.multisession.v1":
        sessions = summary.get("sessions") or {}
        long = (sessions.get("long") or {}).get("first_token_after_tool_ms")
        short = (sessions.get("short") or {}).get("first_token_after_tool_ms")
        return ("Observation only; no policy comparison",
                f"Long first token: {_ms(long)}", f"Short first token: {_ms(short)}",
                "Not measured", "Linked tool waits, KV movement, and replay across sessions; no speed win tested.",
                summary.get("status", "unknown"))
    host = next((case for case in summary.get("cases") or [] if case.get("case_type") == "host_backed"), {})
    return ("Observation only; no policy comparison",
            f"Host-backed replay TTFT: {_ms(host.get('replay_ttft_ms'))}", "No other session",
            "Not measured", "Linked host-backed KV movement to replay; no speed win tested.",
            summary.get("status", "unknown"))


def render_markdown(summaries: list[tuple[Path, dict]], milestones: list[dict] | None = None) -> str:
    ordered = sorted(summaries, key=lambda item: (_time(item[1])[0], item[0].parent.name), reverse=True)
    questions, run_questions = _question_index(milestones or [])
    lines = [
        "# KV Lifecycle Audit", "",
        "**Research question:** Given what the harness knew, was KV moved, kept, or rebuilt at the wrong time?",
        "",
        "This audit compares observed cache work with replay timing and whole-workload outcomes. "
        "A faster control call is not automatically a faster agent task. All times below are from "
        "saved runs; experimental and hypothetical claims are kept separate.", "",
        "## Research progress", "",
    ]
    for milestone in milestones or []:
        lines.extend((f"### {milestone['id']}: {milestone['short_question']}", "",
                      f"**Question.** {milestone['question']}", "",
                      f"**What the evidence says.** {milestone['answer']}", ""))
        if milestone.get("hypothesis"):
            lines.extend((f"**Working hypothesis.** {milestone['hypothesis']}", ""))
        lines.extend((f"**Not yet proved.** {milestone['unknown']}", ""))
    lines.extend(("## Experiment index", "",
                  "Newest first. Each arrow goes from the named control to the changed case in the "
                  "**Compared** column; lower times are better. Three-session rows show the long replay's "
                  "first token and the short session's finish after tool return. Busy rows show summed replay "
                  "TTFT across all replays. Workflow is total elapsed time. Rows without a validated "
                  "comparison have no arrow. For multi-trial runs, arrows compare each mode's median "
                  "time, which can differ from the median trial-by-trial improvement. Select an "
                  "experiment for exact trial values and limits.", "",
                  "| Central date / time | Experiment | Question | Setup | Compared | Replay / long session | Other session | Whole workflow | Plain-English finding | Evidence |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"))
    for path, summary in ordered:
        stamp, _source = _central_stamp(summary)
        run = str(summary.get("run_id") or path.parent.name)
        workload = (summary.get("_manifest") or {}).get("workload") or {}
        question_id = workload.get("research_question_id") or run_questions.get(run)
        if workload.get("research_question_id") and run_questions.get(run) and question_id != run_questions[run]:
            raise ValueError(f"Run {run} has conflicting research question IDs")
        setup, _detail = _setup(summary, summary.get("schema") == "agentic_work_audit.timing.v1")
        comparison, replay, other, workflow, finding, gate = _markdown_index_outcome(summary)
        cells = (stamp, f"[{_kind(summary)}](#run-{run})", question_id,
                 _report_text(setup), comparison, replay, other, workflow, finding, gate)
        lines.append("| " + " | ".join(_md_nowrap(cell) if index in (0, 3, 5, 6, 7, 8) else _md_cell(cell)
                                         for index, cell in enumerate(cells)) + " |")
    lines.extend(("", "## Experiment details", ""))
    for path, summary in ordered:
        run = str(summary.get("run_id") or path.parent.name)
        stamp, source = _central_stamp(summary)
        workload = (summary.get("_manifest") or {}).get("workload") or {}
        question_id = workload.get("research_question_id") or run_questions.get(run)
        question = (questions.get(question_id) or {}).get("question", "Question not recorded")
        _brief, setup = _setup(summary, summary.get("schema") == "agentic_work_audit.timing.v1")
        setup_text = _report_text(setup).removeprefix("How it ran. ")
        limits = list(summary.get("failures") or []) + list(summary.get("limitations") or [])
        if summary.get("interpretation_limit"):
            limits.append(summary["interpretation_limit"])
        if summary.get("_cuda_kernel_subset"):
            limits.append("Nsight lost CUDA activity for a later case; only pair 1 has complete GPU kernel attribution.")
        base = path.parent.as_posix()
        files = summary.get("_files") or {"summary.json"}
        evidence = [f"[Summary]({path.as_posix()})"]
        for filename, label in (("run_manifest.json", "Run manifest"),
                                ("instrumentation_audit.json", "Hook gate"),
                                ("harness_events.jsonl", "Harness timeline"),
                                ("backend_trace.jsonl.gz", "Raw trace"),
                                ("backend_trace.jsonl", "Raw trace")):
            if filename in files:
                evidence.append(f"[{label}]({base}/{filename})")
        if summary.get("schema") == "agentic_work_audit.storage_replay.v1":
            for row in summary["rows"]:
                arm = f"seed{row['seed']}_{row['arm']}"
                evidence.append(f"[{arm} timings]({base}/arms/{arm}/case_results.json)")
                evidence.append(f"[{arm} trace]({base}/arms/{arm}/backend_trace.jsonl.gz)")
        if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
            for row in summary["arms"]:
                arm = f"seed{row['seed']}_{row['arm']}"
                evidence.append(f"[{arm} per-turn timings]({base}/arms/{arm}/case_results.json)")
                evidence.append(f"[{arm} trace]({base}/arms/{arm}/backend_trace.jsonl.gz)")
            for failure in summary.get("failed_arms") or []:
                arm = failure["arm_id"]
                evidence.append(f"[{arm} server failure]({base}/arms/{arm}/server.log)")
                evidence.append(f"[{arm} partial trace]({base}/arms/{arm}/backend_trace.jsonl.gz)")
            for related in summary.get("related_run_ids") or []:
                evidence.append(f"[Full-prepare pilot failure]({path.parent.parent.as_posix()}/"
                                f"{related}/arms/seed1_full_prepare/server.log)")
        if (path.parent / "runtime" / "backend_features.json").exists():
            evidence.append(f"[Backend features]({base}/runtime/backend_features.json)")
        if summary.get("_cuda_kernel_subset"):
            evidence.append(f"[Captured GPU kernels]({base}/nsys/kernel_attribution_pair01.json)")
            if summary.get("_cuda_launch_gaps"):
                evidence.append(f"[CUDA launch gaps]({base}/nsys/launch_gap_attribution_pair01.json)")
            evidence.append(f"[Nsight SQLite trace]({base}/nsys/backend.sqlite.gz)")
        if summary.get("_physical_overlap"):
            evidence.append(f"[Physical copy overlap]({base}/nsys/physical_overlap.json)")
        if summary.get("_decode_submission"):
            evidence.append(f"[Decode launch timing]({base}/nsys/decode_submission.json)")
        if (path.parent / "nsys" / "backend.nsys-rep").exists():
            evidence.append(f"[Nsight capture]({base}/nsys/backend.nsys-rep)")
        if (path.parent / "nsys" / "backend.sqlite.gz").exists():
            evidence.append(f"[Nsight SQLite trace]({base}/nsys/backend.sqlite.gz)")
        if summary.get("_profile_status"):
            evidence.append(f"[Profiler status]({base}/nsys/profile_status.json)")
        command = _reproduction_command(summary)
        lines.extend((f'<a id="run-{html.escape(run, quote=True)}"></a>',
                      "<details>",
                      f"<summary><strong>{html.escape(stamp)} · "
                      f"{html.escape(_kind(summary))}</strong> · {html.escape(run)}</summary>",
                      "", f"**Question ({question_id or 'unmapped'}).** {question}", "",
                      f"**Finding.** {_run_finding(summary)}", "",
                      f"**Setup.** {setup_text}", "",
                      "**Key measurements**", "", _markdown_metrics(summary), "",
                      f"**Evidence gate.** {'physical copy and decode launches verified' if summary.get('_decode_submission') and (summary.get('_physical_overlap') or {}).get('status') == 'verified' else 'validated timing; partial CUDA capture' if summary.get('_cuda_kernel_subset') else summary.get('status', 'unknown')}. Timestamp: {source}; displayed in Central Time.", ""))
        if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1":
            skip_reasons = [reason for arm in summary["arms"]
                            for reason in arm.get("preparation_skips") or []]
            if skip_reasons:
                lines.extend(("**Stage skip reasons.** " + ", ".join(
                    f"{reason} ({skip_reasons.count(reason)})" for reason in sorted(set(skip_reasons))), ""))
        if limits:
            lines.extend(("**Limits**", "", *(f"- {limit}" for limit in limits[:3]), ""))
        if command:
            blocked_variant = (summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1"
                               and str((summary.get("_manifest") or {}).get("run_id") or "")
                               .startswith("storage_cycles_rq17_"))
            if blocked_variant:
                lines.extend(("**Safety note.** The failed partial-prefetch behavior is intentionally "
                              "unavailable in current code. This command runs the guarded safe variant, "
                              "not an exact replay of the failed prototype.", ""))
            if summary.get("schema") == "agentic_work_audit.storage_cycles.summary.v1" and any(
                "native_prefetch_not_admitted" in error for arm in summary["arms"]
                for error in arm.get("preparation_errors") or []
            ):
                lines.extend(("**Reproduction note.** The current adapter checks native rate limiting "
                              "before prefetch; this command will not reproduce the earlier unclassified "
                              "refusal exactly.", ""))
            lines.extend(("**Run guarded variant** (set the container image and model cache for the target host):"
                          if blocked_variant else
                          "**Reproduce** (set the container image and model cache for the target host):",
                          "", "```bash", command, "```", ""))
        lines.extend(("**Evidence:** " + " · ".join(evidence), "", "</details>", ""))
    return "\n".join(lines).rstrip() + "\n"


def _pair_rq11_controls(summaries: list[tuple[Path, dict]]) -> None:
    def key(summary: dict) -> tuple | None:
        if (summary.get("schema") != "agentic_work_audit.overlap_dose.v1" or
                summary.get("status") == "excluded"):
            return None
        workload = (summary.get("_manifest") or {}).get("workload") or {}
        return (summary.get("session_count"), summary.get("seed"),
                workload.get("active_prompt_words", workload.get("prompt_words_target")),
                workload.get("donor_prompt_words", workload.get("prompt_words_target")),
                workload.get("target_wait_ms"), workload.get("donor_wait_ms"),
                workload.get("decode_tokens"), workload.get("nsys_enabled"),
                workload.get("research_question_id"), workload.get("pair_id"))

    controls = {key(summary): summary for _, summary in summaries
                if key(summary) is not None and summary.get("planned_overlap") == 0}
    for _, summary in summaries:
        group = key(summary)
        if group is None or summary.get("planned_overlap") == 0 or group not in controls:
            continue
        control = controls[group]
        if [row.get("loaded_tokens") for row in summary["donors"]] != [
                row.get("loaded_tokens") for row in control["donors"]]:
            continue
        peer_times = [row["completion_after_tool_ms"] for row in control["active_requests"][1:]]
        if peer_times:
            summary["_dose_control"] = {
                "run_id": control["run_id"],
                "target_ms": control["target"]["completion_after_tool_ms"],
                "peer_median_ms": median(peer_times),
                "decode_submission": control.get("_decode_submission"),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path)
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
        cuda = path.parent / "nsys" / "kernel_attribution_pair01.json"
        if cuda.exists():
            summary["_cuda_kernel_subset"] = json.loads(cuda.read_text(encoding="utf-8"))
        launch = path.parent / "nsys" / "launch_gap_attribution_pair01.json"
        if launch.exists():
            summary["_cuda_launch_gaps"] = json.loads(launch.read_text(encoding="utf-8"))
        physical = path.parent / "nsys" / "physical_overlap.json"
        if physical.exists():
            summary["_physical_overlap"] = json.loads(physical.read_text(encoding="utf-8"))
        submission = path.parent / "nsys" / "decode_submission.json"
        if submission.exists():
            summary["_decode_submission"] = json.loads(submission.read_text(encoding="utf-8"))
        profile = path.parent / "nsys" / "profile_status.json"
        if profile.exists():
            summary["_profile_status"] = json.loads(profile.read_text(encoding="utf-8"))
        summary["_started_ns"] = _first_request_ns(path)
        summary["_files"] = {evidence.name for evidence in path.parent.iterdir() if evidence.is_file()}
        summaries.append((Path(os.path.relpath(path, args.out.parent)), summary))
    _pair_rq11_controls(summaries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    milestones = []
    if args.progress_file:
        milestones = json.loads(args.progress_file.read_text(encoding="utf-8"))["milestones"]
    args.out.write_text(render(summaries, milestones), encoding="utf-8")
    print(f"Wrote {args.out} with {len(summaries)} saved run(s)")
    if args.markdown_out:
        markdown_summaries = [
            (Path(os.path.relpath(args.results_dir / path.parent.name / "summary.json",
                                  args.markdown_out.parent)), summary)
            for path, summary in summaries
        ]
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(markdown_summaries, milestones), encoding="utf-8")
        print(f"Wrote {args.markdown_out} with {len(summaries)} saved run(s)")


if __name__ == "__main__":
    main()
