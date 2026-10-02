"""Build the compact, chronological KV lifecycle audit from saved evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path


def _esc(value: object) -> str:
    return html.escape(str(value if value is not None else "not recorded"), quote=True)


def _ms(value: object) -> str:
    return f"{value:.1f} ms" if isinstance(value, (int, float)) else "not recorded"


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
                   f"host cache: {_esc(workload.get('hicache_size_gb'))} GB; "
                   f"GPU memory fraction: {_esc(workload.get('mem_fraction_static'))}.")
    return brief, detail


def _reproduction(summary: dict, timing: bool) -> str:
    manifest = summary.get("_manifest") or {}
    workload = manifest.get("workload") or {}
    cases = workload.get("cases") or []
    if len(cases) != 2:
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
        "HICACHE_SIZE_GB": workload.get("hicache_size_gb"),
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
    comparable = [p for p in pairs if p.get("comparable") and
                  isinstance(p.get("late_minus_early_first_token_after_due_ms"), (int, float))]
    deltas = [_ms(p["late_minus_early_first_token_after_due_ms"]) for p in comparable]
    headline = ("Late +" + " / +".join(deltas) + " to first token") if deltas else "Comparison unavailable"
    cases = {(c.get("pair"), c.get("condition")): c for c in summary.get("cases") or []}
    rows = []
    for pair in pairs:
        number = pair.get("pair")
        early, late = cases.get((number, "early"), {}), cases.get((number, "late"), {})
        delta = _ms(pair.get("late_minus_early_first_token_after_due_ms")) if pair.get("comparable") else "withheld"
        rows.append(
            f"<tr><td>{_esc(number)}</td><td>{_ms(early.get('first_token_after_due_ms'))}</td>"
            f"<td>{_ms(late.get('first_token_after_due_ms'))}</td><td>{delta}</td>"
            f"<td>{_ms(early.get('replay_ttft_ms'))} / {_ms(late.get('replay_ttft_ms'))}</td>"
            f"<td>{_ms(early.get('submission_after_due_ms'))} / {_ms(late.get('submission_after_due_ms'))}</td></tr>"
        )
    detail = (
        "<p><strong>What was measured.</strong> Early preparation requested the native load during the tool wait; "
        "late preparation requested it after the wait. Tool completion to first token includes the submission gap; "
        "replay TTFT starts only after submission.</p>"
        "<div class='detail-scroll'><table class='pair-table'><thead><tr><th>Pair</th><th>Early due→token</th>"
        "<th>Late due→token</th><th>Late − early</th><th>Replay TTFT, E / L</th>"
        "<th>Submit gap, E / L</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
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


def render(summaries: list[tuple[Path, dict]]) -> str:
    ordered = sorted(summaries, key=lambda item: (_time(item[1])[0], item[0].parent.name), reverse=True)
    rows: list[str] = []
    for path, summary in ordered:
        _, date, time, source = _time(summary)
        timing = summary.get("schema") == "agentic_work_audit.timing.v1"
        run = str(summary.get("run_id") or path.parent.name)
        kind = "Early vs late" if timing else "Lifecycle validation"
        setup, method = _setup(summary, timing)
        result, findings = _timing_result(summary) if timing else _lifecycle_result(summary)
        status = str(summary.get("status") or "unknown")
        limits = "".join(f"<li>{_esc(item)}</li>" for item in
                         [*(summary.get("failures") or []), *(summary.get("limitations") or [])])
        rows.append(
            f"<tr><td title='{_esc(source)}'>{_esc(date)}</td><td title='{_esc(source)}'>{_esc(time)}</td>"
            f"<td><strong>{kind}</strong><small>{_esc(run)}</small></td><td>{setup}</td>"
            f"<td>{result}</td><td><span class='status {_esc(status)}'>{_esc(status)}</span></td>"
            f"<td><details><summary>View</summary><div class='detail'>"
            f"<p>{method}</p>{findings}{_reproduction(summary, timing)}"
            f"{'<p><strong>Run-specific limits.</strong></p><ul>' + limits + '</ul>' if limits else ''}"
            f"<p><strong>Evidence.</strong> {_links(path, summary)}</p>"
            "</div></details></td></tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='7'>No saved run summaries are archived yet.</td></tr>")
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>KV Lifecycle Audit</title>
<style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182733;background:#f5f8f9}
body{max-width:1500px;margin:auto;padding:28px 24px 64px;line-height:1.45}
h1{font-size:1.7rem;margin:0 0 8px;letter-spacing:0}p{color:#3d5260}
.intro{max-width:90ch;margin:0 0 20px}.table-scroll{overflow-x:auto;border:1px solid #d7e2e6;background:#fff}
table{border-collapse:collapse;width:100%;min-width:1100px}th,td{padding:12px 14px;text-align:left;vertical-align:top;border-bottom:1px solid #e5ecef}
th{background:#e4f0ef;color:#204a4a;font-size:.88rem;white-space:nowrap}td:nth-child(1),td:nth-child(2){white-space:nowrap;font-variant-numeric:tabular-nums}
td:nth-child(3){min-width:165px}td:nth-child(4){min-width:210px}td:nth-child(5){min-width:260px}
small{display:block;color:#5a6c77;overflow-wrap:anywhere;font-size:.8rem;margin-top:3px}
.status{display:inline-block;font-weight:650}.validated{color:#126746}.failed{color:#b63839}
details{min-width:56px}summary{cursor:pointer;color:#086780;font-weight:650;list-style:none}summary::-webkit-details-marker{display:none}
.detail{min-width:430px;max-width:690px;padding:8px 0}.detail p{margin:10px 0}.detail-scroll{overflow-x:auto}
.pair-table{min-width:650px;font-size:.88rem}.pair-table th,.pair-table td{padding:7px 9px}
pre{overflow-x:auto;background:#edf4f5;padding:10px;white-space:pre-wrap;overflow-wrap:anywhere}
a{color:#086780}a:hover{text-decoration:underline}
@media(max-width:760px){body{padding:16px 10px 40px}.detail{min-width:300px}}
</style></head><body><h1>KV Lifecycle Audit</h1>
<p class="intro">One row per saved experiment, newest first. Date and time are UTC from the first recorded request; a completion-time fallback is labeled on hover. Timing runs compare early and late host-KV preparation. Lifecycle runs check that cache events can be linked to a replay; their TTFTs are not a policy win.</p>
<div class="table-scroll"><table><thead><tr><th>Date</th><th>Time (UTC)</th><th>Experiment</th><th>Setup</th><th>Main result</th><th>Evidence gate</th><th>Details</th></tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    args.out.write_text(render(summaries), encoding="utf-8")
    print(f"Wrote {args.out} with {len(summaries)} saved run(s)")


if __name__ == "__main__":
    main()
