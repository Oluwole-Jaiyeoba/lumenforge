#!/usr/bin/env python3
"""Render the observation-only natural multi-agent KV-pressure report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):,.1f} ms"


def build(summary: dict[str, Any]) -> str:
    workload = summary["workload"]
    bucket_rows = "".join(
        f"<tr><td>{html.escape(name.replace('_', ' '))}</td><td>{count}</td></tr>"
        for name, count in summary["pressure_buckets"].items()
    )
    observations = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['session_id']))}</td><td>{row['tool_wait_step']}</td><td>{row['tool_wait_ms']} ms</td>"
        f"<td>{fmt_ms(row['ttft_ms'])}</td><td>{fmt_ms(row.get('replay_duration_ms', row.get('total_decode_ms')))}</td>"
        f"<td>{row['natural_reload_count']}</td><td>{row['cross_session_reload_count']}</td>"
        f"<td>{html.escape(row['pressure_bucket'].replace('_', ' '))}</td>"
        f"<td>{html.escape(', '.join(row['cross_session_reload_sessions']) or 'none')}</td></tr>"
        for row in summary["observations"]
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Natural Multi-Agent KV Pressure</title><style>
:root {{ font-family:Arial,sans-serif; color:#182331; background:#f6f8fa; }} body {{ max-width:1300px; margin:32px auto; padding:0 20px 56px; line-height:1.45; }}
section {{ margin:22px 0; padding:18px; border:1px solid #d8e0ea; background:#fff; overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; }} th,td {{ text-align:left; vertical-align:top; padding:9px; border-bottom:1px solid #d8e0ea; white-space:nowrap; }} th {{ background:#edf4f7; }} .note {{ color:#526273; }}
</style></head><body><h1>Natural Multi-Agent KV Pressure</h1>
<p class=\"note\">Observation-only workload. All sessions have equal frontend semantics. No prepared-prefix control request or synthetic reload injection is used.</p>
<section><h2>Workload</h2><table><tbody><tr><th>Sessions</th><td>{workload['session_count']}</td></tr><tr><th>Tool waits per session</th><td>{workload['tool_waits_per_session']}</td></tr><tr><th>Session prefix</th><td>{workload['session_prefix_tokens']} tokens</td></tr><tr><th>Replay decode</th><td>{workload['replay_tokens']} tokens</td></tr><tr><th>Tool wait range</th><td>{workload['tool_wait_range_ms'][0]}-{workload['tool_wait_range_ms'][1]} ms</td></tr></tbody></table></section>
<section><h2>Natural Reload Coverage</h2><p>{summary['native_reload_events']} native SGLang load-back event(s) were observed across {summary['replay_count']} replay decodes.</p><table><thead><tr><th>Observed pressure bucket</th><th>Replay decodes</th></tr></thead><tbody>{bucket_rows}</tbody></table></section>
<section><h2>Replay Observations</h2><table><thead><tr><th>Session</th><th>Tool step</th><th>Tool wait</th><th>TTFT</th><th>Replay duration</th><th>All reloads during decode</th><th>Other-session reloads</th><th>Reference bucket</th><th>Other reloading sessions</th></tr></thead><tbody>{observations}</tbody></table></section>
<section><h2>Interpretation Boundary</h2><p>{html.escape(summary['interpretation'])}</p><p>This maps natural overlap counts to the controlled low/medium/high reference buckets. It does not claim that count alone equals the controlled CUDA-copy share; a later profiler pass is needed for that physical attribution.</p></section>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary.get("schema_version") != "natural_multi_agent_kv_pressure.v1":
        raise SystemExit("unsupported summary schema")
    args.out.write_text(build(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
