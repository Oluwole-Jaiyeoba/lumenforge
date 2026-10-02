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
    details: list[str] = []
    for path, summary in sorted(summaries, key=lambda item: item[0].parent.name, reverse=True):
        run = str(summary.get("run_id") or path.parent.name)
        status = str(summary.get("status") or "unknown")
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
        link = path.as_posix()
        rows.append(
            "<tr>"
            f"<td>{_esc(run)}</td><td class='{_esc(status)}'>{_esc(status)}</td>"
            f"<td class='{_esc(block_status)}'>{_esc(block_status)}</td>"
            f"<td>{_esc(warm.get('replay_ttft_ms'))}</td><td>{_esc(host.get('replay_ttft_ms'))}</td>"
            f"<td>{_esc(host.get('host_resident_tokens'))}</td>"
            f"<td>{_esc(host.get('native_loaded_tokens'))}</td>"
            f"<td>{_esc(block_host.get('semantic_load_transitions'))}</td>"
            f"<td>{_esc(block_host.get('layer_copy_observations', host.get('native_layer_copies')))}</td>"
            f"<td>{_esc(slot_host.get('loaded_slots_matched_by_replay'))}</td>"
            f"<td>{_esc(str(block_host.get('same_loaded_block_used_by_replay') or 'unknown').replace('_', ' '))}</td>"
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
        details.append(
            f"<details><summary>{_esc(run)}: evidence and limits</summary>"
            f"<p>{_esc(summary.get('interpretation'))}</p>"
            f"{block_evidence}"
            f"<p><a href='{_esc(path.with_name('run_manifest.json').as_posix())}'>Run manifest</a>"
            f" · <a href='{_esc(path.with_name('normalized_events.jsonl').as_posix())}'>Normalized events</a></p>"
            f"<p>Gate failures: {_esc('; '.join(failures) if failures else 'none')}</p>"
            "<ul>" + "".join(f"<li>{_esc(name)}: {_esc(value)}</li>" for name, value in limits.items()) + "</ul>"
            "</details>"
        )
    if not rows:
        rows = ["<tr><td colspan='12'>Live validation pending. Synthetic fixtures are not research results.</td></tr>"]
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agentic Work Audit</title><style>
:root{font-family:system-ui,-apple-system,sans-serif;color:#182332;background:#f5f8fb}
body{max-width:1500px;margin:0 auto;padding:32px 24px 64px;line-height:1.45}
h1{font-size:1.8rem;margin:0 0 8px}h2{font-size:1.2rem;margin-top:28px}
p{max-width:82ch;color:#425268}a{color:#086d88}a:hover{text-decoration:underline}
.table-scroll{overflow-x:auto;background:white;border:1px solid #dce6ed;border-radius:6px}
table{border-collapse:collapse;width:100%;min-width:1390px}th,td{padding:11px 13px;text-align:left;border-bottom:1px solid #e4ebef;white-space:nowrap}
th{background:#e6f2f4;color:#1b4550;font-size:.88rem}tr:last-child td{border-bottom:0}
.validated{color:#14664d;font-weight:700}.failed{color:#ac4138;font-weight:700}
details{background:#fff;border-top:1px solid #dce6ed;padding:12px 15px}summary{cursor:pointer;font-weight:600}
code{font-size:.95em}
</style></head><body><h1>Agentic Work Audit</h1>
<p>This lane checks when cache and GPU work happens across tool waits and replays. The first runs validate event linkage on the pinned backend; they do not estimate production waste or prove a controller benefit. Warm and host cases run sequentially, so their TTFT values are not a controlled speed comparison. Per-layer copies are not additional logical loads.</p>
<h2>Live Validation Runs</h2><div class="table-scroll"><table><thead><tr>
<th>Run</th><th>Trace gate</th><th>Block gate</th><th>Warm replay TTFT (ms)</th><th>Host replay TTFT (ms)</th>
<th>Selected host node (tokens)</th><th>Native loaded tokens</th><th>Logical loads</th>
<th>Layer-copy observations</th><th>Loaded slots in replay match</th><th>Exact model consumption</th><th>Evidence</th>
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
        summaries.append((Path(os.path.relpath(path, args.out.parent)), summary))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(summaries), encoding="utf-8")
    print(f"Wrote {args.out} with {len(summaries)} validation run(s)")


if __name__ == "__main__":
    main()
