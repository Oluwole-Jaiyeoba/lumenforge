#!/usr/bin/env python3
"""Aggregate repeated natural multi-agent KV-pressure observations."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.1f} ms"


def group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ttft = [float(row["ttft_ms"]) for row in rows]
    duration = [float(row["replay_duration_ms"]) for row in rows]
    return {
        "replays": len(rows),
        "median_ttft_ms": round(float(median(ttft)), 3) if ttft else None,
        "p95_ttft_ms": round(float(percentile(ttft, 0.95)), 3),
        "median_replay_duration_ms": round(float(median(duration)), 3) if duration else None,
        "p95_replay_duration_ms": round(float(percentile(duration, 0.95)), 3),
    }


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    by_sessions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    native_events: dict[int, int] = defaultdict(int)
    trials: dict[int, set[str]] = defaultdict(set)
    for summary in summaries:
        session_count = int(summary["workload"]["session_count"])
        by_sessions[session_count].extend(summary["observations"])
        native_events[session_count] += int(summary["native_reload_events"])
        trials[session_count].add(str(summary["run_id"]))

    levels: list[dict[str, Any]] = []
    for session_count in sorted(by_sessions):
        rows = by_sessions[session_count]
        cross_overlap = [row for row in rows if int(row["cross_session_reload_count"]) > 0]
        no_cross_overlap = [row for row in rows if int(row["cross_session_reload_count"]) == 0]
        levels.append(
            {
                "session_count": session_count,
                "trials": len(trials[session_count]),
                "native_reload_events": native_events[session_count],
                "all_replays": group_stats(rows),
                "cross_session_overlap": group_stats(cross_overlap),
                "no_cross_session_overlap": group_stats(no_cross_overlap),
                "cross_session_overlap_pct": round(100 * len(cross_overlap) / len(rows), 1) if rows else 0.0,
            }
        )
    return {
        "schema_version": "natural_kv_pressure_comparison.v1",
        "levels": levels,
        "interpretation": (
            "This is an observational comparison. A cross-session overlap is a native host-to-GPU "
            "load-back after another replay's first token; it is not by itself a causal latency estimate."
        ),
    }


def build(report: dict[str, Any]) -> str:
    coverage_rows = "".join(
        "<tr>"
        f"<td>{level['session_count']}</td><td>{level['trials']}</td><td>{level['all_replays']['replays']}</td>"
        f"<td>{level['native_reload_events']}</td><td>{level['cross_session_overlap']['replays']}</td>"
        f"<td>{level['cross_session_overlap_pct']:.1f}%</td></tr>"
        for level in report["levels"]
    )
    latency_rows = "".join(
        "<tr>"
        f"<td>{level['session_count']}</td><td>{html.escape(group_name)}</td><td>{group['replays']}</td>"
        f"<td>{fmt(group['median_ttft_ms'])}</td><td>{fmt(group['p95_ttft_ms'])}</td>"
        f"<td>{fmt(group['median_replay_duration_ms'])}</td><td>{fmt(group['p95_replay_duration_ms'])}</td></tr>"
        for level in report["levels"]
        for group_name, group in (("No cross-session overlap", level["no_cross_session_overlap"]), ("Cross-session overlap", level["cross_session_overlap"]))
    )
    return f"""<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Natural KV Pressure Performance Comparison</title><style>
:root{{font-family:Arial,sans-serif;color:#182331;background:#f6f8fa}}body{{max-width:1200px;margin:32px auto;padding:0 20px 56px;line-height:1.45}}section{{margin:22px 0;padding:18px;border:1px solid #d8e0ea;background:#fff;overflow-x:auto}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;vertical-align:top;padding:9px;border-bottom:1px solid #d8e0ea;white-space:nowrap}}th{{background:#edf4f7}}.note{{color:#526273}}
</style></head><body><h1>Natural KV Pressure Performance Comparison</h1>
<p class=\"note\">All requests have equal frontend semantics. No prepared-prefix command or synthetic reload injection was used.</p>
<section><h2>Coverage</h2><table><thead><tr><th>Sessions</th><th>Trials</th><th>Replays</th><th>Native reloads</th><th>Cross-session overlaps</th><th>Overlap rate</th></tr></thead><tbody>{coverage_rows}</tbody></table></section>
<section><h2>Replay Performance by Observed Overlap</h2><table><thead><tr><th>Sessions</th><th>Observed group</th><th>Replays</th><th>Median TTFT</th><th>P95 TTFT</th><th>Median replay duration</th><th>P95 replay duration</th></tr></thead><tbody>{latency_rows}</tbody></table></section>
<section><h2>Interpretation Boundary</h2><p>{html.escape(report['interpretation'])}</p></section></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.cases_root.glob("*/natural_multi_agent_kv_pressure_summary.json"))
    if not paths:
        raise SystemExit(f"no completed natural-pressure summaries under {args.cases_root}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    report = aggregate(summaries)
    args.summary_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.out.write_text(build(report), encoding="utf-8")


if __name__ == "__main__":
    main()
