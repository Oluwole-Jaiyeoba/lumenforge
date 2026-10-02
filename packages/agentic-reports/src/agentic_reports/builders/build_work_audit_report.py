"""Build the fourth-lane evidence index from immutable validation summaries."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


def _esc(value: object) -> str:
    return html.escape(str(value if value is not None else "n/a"))


def render(summaries: list[tuple[Path, dict]]) -> str:
    rows: list[str] = []
    timing_rows: list[str] = []
    details: list[str] = []
    for path, summary in sorted(summaries, key=lambda item: item[0].parent.name, reverse=True):
        run = str(summary.get("run_id") or path.parent.name)
        status = str(summary.get("status") or "unknown")
        if summary.get("schema") == "agentic_work_audit.timing.v1":
            cases = summary.get("cases") or []
            order = ", ".join(case.get("condition", "?") for case in cases[:2])
            settings = (summary.get("_manifest") or {}).get("workload") or {}
            by_pair = {(case.get("pair"), case.get("condition")): case for case in cases}
            for pair in summary.get("pairs") or []:
                number = pair.get("pair")
                early = by_pair.get((number, "early"), {})
                late = by_pair.get((number, "late"), {})
                timing_rows.append(
                    "<tr>"
                    f"<td>{_esc(run)}</td><td>{_esc(number)}</td><td>{_esc(order)}</td>"
                    f"<td>{_esc(settings.get('exact_trace_indices'))}</td>"
                    f"<td class='{_esc(status)}'>{_esc(status)}</td>"
                    f"<td>{_esc(pair.get('comparable'))}</td>"
                    f"<td>{_esc(early.get('initial_latency_ms'))} / "
                    f"{_esc(late.get('initial_latency_ms'))}</td>"
                    f"<td>{_esc(early.get('loaded_tokens'))} / {_esc(late.get('loaded_tokens'))}</td>"
                    f"<td>{_esc(early.get('native_cuda_load_ms'))} / "
                    f"{_esc(late.get('native_cuda_load_ms'))}</td>"
                    f"<td>{_esc(early.get('load_control_duration_ms'))} / "
                    f"{_esc(late.get('load_control_duration_ms'))}</td>"
                    f"<td>{_esc(early.get('load_request_from_due_ms'))} / "
                    f"{_esc(late.get('load_request_from_due_ms'))}</td>"
                    f"<td>{_esc(early.get('completion_observed_from_due_ms'))} / "
                    f"{_esc(late.get('completion_observed_from_due_ms'))}</td>"
                    f"<td>{_esc(early.get('completion_observed_before_due'))}</td>"
                    f"<td>{_esc(late.get('completion_observed_before_due'))}</td>"
                    f"<td>{_esc(early.get('submission_after_due_ms'))} / "
                    f"{_esc(late.get('submission_after_due_ms'))}</td>"
                    f"<td>{_esc(early.get('first_token_after_due_ms'))}</td>"
                    f"<td>{_esc(late.get('first_token_after_due_ms'))}</td>"
                    f"<td>{_esc(pair.get('late_minus_early_first_token_after_due_ms'))}</td>"
                    f"<td>{_esc(early.get('replay_ttft_ms'))} / {_esc(late.get('replay_ttft_ms'))}</td>"
                    f"<td>{_esc(pair.get('late_minus_early_replay_ttft_ms'))}</td>"
                    f"<td>{_esc(early.get('second_replay_ttft_ms'))} / "
                    f"{_esc(late.get('second_replay_ttft_ms'))}</td>"
                    f"<td>{_esc(pair.get('task_comparable'))}</td>"
                    f"<td>{_esc(pair.get('late_minus_early_post_tool_duration_ms'))}</td>"
                    f"<td>{_esc(pair.get('late_minus_early_task_latency_ms'))}</td>"
                    f"<td>{_esc(early.get('loaded_slots_match_status'))}: "
                    f"{_esc(early.get('loaded_slots_matched_by_replay'))} / "
                    f"{_esc(late.get('loaded_slots_match_status'))}: "
                    f"{_esc(late.get('loaded_slots_matched_by_replay'))}</td>"
                    f"<td><a href='{_esc(path.as_posix())}'>JSON</a></td></tr>"
                )
            details.append(
                f"<details><summary>{_esc(run)}: timing limits</summary>"
                f"<p>{_esc(summary.get('interpretation'))}</p>"
                f"<p>Trace profile: {_esc(summary.get('_trace_profile') or 'not recorded')}; "
                f"case order: {_esc(order)}; exact slot proof required: "
                f"{_esc(summary.get('slot_lineage_required'))}. "
                "A larger late-minus-early delay is a paired observation, "
                "not yet proof of a system-wide scheduling benefit.</p>"
                f"<p><a href='{_esc(path.with_name('instrumentation_audit.json').as_posix())}'>Trace gate</a>"
                f" · <a href='{_esc(path.with_name('normalized_events.jsonl').as_posix())}'>Timeline events</a></p>"
                f"<p>Failures: {_esc('; '.join(summary.get('failures') or []) or 'none')}; "
                f"limitations: {_esc('; '.join(summary.get('limitations') or []) or 'none')}; "
                f"pair-level task limits: {_esc('; '.join(reason for pair in summary.get('pairs') or [] for reason in pair.get('task_comparability_reasons') or []) or 'none')}</p></details>"
            )
            continue
        block_audit = summary.get("_block_audit") or {}
        instrumentation_analysis = summary.get("_instrumentation_analysis") or {}
        block_status = str(block_audit.get("status") or "pending")
        cases = summary.get("cases") or []
        warm = next((case for case in cases if case.get("case_type") == "warm_control"), {})
        host = next((case for case in cases if case.get("case_type") == "host_backed"), {})
        block_host = next((case for case in block_audit.get("cases") or []
                           if case.get("case_type") == "host_backed"), {})
        slot_host = next((case for case in instrumentation_analysis.get("cases") or []
                          if case.get("case_type") == "host_backed"), block_host)
        manifest = summary.get("_manifest") or {}
        order = ", ".join((manifest.get("workload") or {}).get("cases") or []) or "n/a"
        profile = summary.get("_trace_profile") or "not recorded"
        link = path.as_posix()
        rows.append(
            "<tr>"
            f"<td>{_esc(run)}</td><td class='{_esc(status)}'>{_esc(status)}</td>"
            f"<td class='{_esc(block_status)}'>{_esc(block_status)}</td>"
            f"<td>{_esc(order)}</td><td>{_esc(profile)}</td>"
            f"<td>{_esc(warm.get('replay_ttft_ms'))}</td><td>{_esc(host.get('replay_ttft_ms'))}</td>"
            f"<td>{_esc(warm.get('second_replay_ttft_ms'))}</td>"
            f"<td>{_esc(host.get('second_replay_ttft_ms'))}</td>"
            f"<td>{_esc(block_host.get('planned_pre_replay_loaded_tokens'))}</td>"
            f"<td>{_esc(block_host.get('replay_time_loaded_tokens'))}</td>"
            f"<td>{_esc(slot_host.get('loaded_slots_matched_by_replay'))}</td>"
            f"<td><a href='{_esc(link)}'>JSON</a></td></tr>"
        )
        failures = summary.get("failures") or []
        limits = dict(summary.get("opportunity_ledgers") or {})
        if block_audit and "backup_reuse" in limits:
            limits["backup_reuse"] = "unknown: exact loaded-block consumption by replay is not proven"
        block_link = path.with_name("block_audit.json").as_posix()
        block_evidence = (
            f"<p><a href='{_esc(block_link)}'>Block lifecycle audit</a>: "
            f"{_esc(block_audit.get('logical_block_records'))} logical records; "
            f"{_esc(block_audit.get('logical_loaded_records'))} loaded record(s). "
            f"{_esc(block_audit.get('interpretation'))}</p>"
        ) if block_audit else "<p>Block lifecycle audit pending.</p>"
        if instrumentation_analysis:
            analysis_link = path.with_name("instrumentation_analysis.json").as_posix()
            block_evidence += (
                f"<p><a href='{_esc(analysis_link)}'>Shared-instrumentation analysis</a>: "
                f"{_esc(slot_host.get('loaded_slots_matched_by_replay'))} loaded GPU slots "
                "also appeared in replay's cache match; exact model consumption remains unproven.</p>"
            )
        raw_link = (
            f" · <a href='{_esc(path.with_name('backend_trace.jsonl.gz').as_posix())}'>Compressed raw trace</a>"
            if summary.get("_has_compressed_trace") else ""
        )
        details.append(
            f"<details><summary>{_esc(run)}: evidence and limits</summary>"
            f"<p>{_esc(summary.get('interpretation'))}</p>"
            f"{block_evidence}"
            f"<p><a href='{_esc(path.with_name('run_manifest.json').as_posix())}'>Run manifest</a>"
            f" · <a href='{_esc(path.with_name('normalized_events.jsonl').as_posix())}'>Normalized events</a>"
            f"{raw_link}</p>"
            f"<p>Planned pre-replay loads: {_esc(block_host.get('planned_pre_replay_loads'))}; "
            f"replay-time loads: {_esc(block_host.get('replay_time_loads'))}; "
            f"second-replay matched loaded slots: {_esc(block_host.get('second_replay_loaded_slots_match_status'))}. "
            "A replay-time load is observed work, not by itself proof that it caused a stall.</p>"
            f"<p>Gate failures: {_esc('; '.join(failures) if failures else 'none')}</p>"
            "<ul>" + "".join(f"<li>{_esc(name)}: {_esc(value)}</li>" for name, value in limits.items()) + "</ul>"
            "</details>"
        )
    if not rows:
        rows = ["<tr><td colspan='13'>Live validation pending. Synthetic fixtures are not research results.</td></tr>"]
    timing_html = ""
    if timing_rows:
        timing_html = (
            '<h2>Early vs. Late KV Preparation</h2><p>Each case begins with a proved host-backed prefix. '
            'Early requests a native load during the tool wait; late requests it after the tool finishes. '
            'The primary delay is from tool completion to first token, including any submission gap. '
            'Completion is timestamped when CUDA status is observed, not at the exact GPU finish instant. '
            'A zero matched-slot count labeled unknown means exact lineage was not proved, not that no slots were reused.</p>'
            '<div class="table-scroll"><table><thead><tr>'
            '<th>Run</th><th>Pair</th><th>Order</th><th>Index limit</th><th>Gate</th><th>Comparable</th>'
            '<th>Initial latency E / L (ms)</th><th>Load tokens E / L</th>'
            '<th>CUDA load E / L (ms)</th><th>Control call E / L (ms)</th>'
            '<th>Load request vs. due E / L (ms)</th><th>Completion observed vs. due E / L (ms)</th>'
            '<th>Early load confirmed by due</th><th>Late load confirmed by due</th>'
            '<th>Submit gap E / L (ms)</th>'
            '<th>Early due-to-token (ms)</th><th>Late due-to-token (ms)</th>'
            '<th>Late minus early (ms)</th><th>Replay TTFT E / L (ms)</th><th>TTFT delta (ms)</th>'
            '<th>Replay 2 TTFT E / L (ms)</th><th>Task comparable</th>'
            '<th>Post-tool duration delta (ms)</th><th>Full task delta (ms)</th>'
            '<th>Matched GPU slots E / L</th><th>Evidence</th>'
            '</tr></thead><tbody>' + "".join(timing_rows) + '</tbody></table></div>'
        )
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KV Lifecycle Audit</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182332;background:#f5f8fb}
body{max-width:1500px;margin:0 auto;padding:32px 24px 64px;line-height:1.45}
h1{font-size:1.8rem;margin:0 0 8px}h2{font-size:1.2rem;margin-top:28px}
p{max-width:82ch;color:#425268}a{color:#086d88}a:hover{text-decoration:underline}
.table-scroll{overflow-x:auto;background:white;border:1px solid #dce6ed;border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:1530px}th,td{padding:11px 13px;text-align:left;border-bottom:1px solid #e4ebef;white-space:nowrap}
th{background:#e6f2f4;color:#1b4550;font-size:.88rem}tr:last-child td{border-bottom:0}
.validated{color:#14664d;font-weight:700}.failed{color:#ac4138;font-weight:700}
details{background:#fff;border-top:1px solid #dce6ed;padding:12px 15px}summary{cursor:pointer;font-weight:600}
code{font-size:.95em}
</style></head><body><h1>KV Lifecycle Audit</h1>
<p>This lane checks when cache and GPU work happens across tool waits and replays. Lifecycle runs validate event linkage; timing runs compare early and late preparation on the pinned backend. Neither establishes production waste or a system-wide controller benefit. Cases run sequentially, with order reversed in separate runs. Trace profiles and exact-index capture have different overhead; do not compare their absolute TTFTs as a workload result. Per-layer copies are not additional logical loads.</p>
""" + timing_html + """<h2>Lifecycle Validation Runs</h2><div class="table-scroll"><table><thead><tr>
<th>Run</th><th>Trace gate</th><th>Block gate</th><th>Case order</th><th>Trace profile</th>
<th>Warm replay 1 TTFT (ms)</th><th>Host replay 1 TTFT (ms)</th>
<th>Warm replay 2 TTFT (ms)</th><th>Host replay 2 TTFT (ms)</th>
<th>Planned load (tokens)</th><th>Replay-time load (tokens)</th>
<th>Loaded slots in replay 1 match</th><th>Evidence</th>
</tr></thead><tbody>""" + "".join(rows) + """</tbody></table></div><h2>Interpretation</h2>""" + "".join(details) + """
<p>A host-backed prefix and a completed load are observations. An avoidable stall, unused backup, or memory-bandwidth bottleneck requires further evidence and is left unknown here.</p>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for path in args.results_dir.glob("*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        block_path = path.with_name("block_audit.json")
        if block_path.exists():
            summary["_block_audit"] = json.loads(block_path.read_text(encoding="utf-8"))
        analysis_path = path.with_name("instrumentation_analysis.json")
        if analysis_path.exists():
            summary["_instrumentation_analysis"] = json.loads(analysis_path.read_text(encoding="utf-8"))
        manifest_path = path.with_name("run_manifest.json")
        if manifest_path.exists():
            summary["_manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        audit_path = path.with_name("instrumentation_audit.json")
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            summary["_trace_profile"] = (audit.get("gate") or {}).get("profile")
        summary["_has_compressed_trace"] = path.with_name("backend_trace.jsonl.gz").exists()
        summaries.append((Path(os.path.relpath(path, args.out.parent)), summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(summaries), encoding="utf-8")
    print(f"Wrote {args.out} with {len(summaries)} validation run(s)")


if __name__ == "__main__":
    main()
