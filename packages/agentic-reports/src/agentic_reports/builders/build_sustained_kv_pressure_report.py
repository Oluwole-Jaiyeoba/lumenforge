#!/usr/bin/env python3
"""Build the evidence report for sustained KV pressure during decode."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


CONDITION_LABELS = {
    "decode_control": "Decode control",
    "single_reload": "Single KV reload",
    "sustained_reload": "Sustained KV reload",
}
REGION_LABELS = {
    "before": "Before reload",
    "during": "Intersecting reload",
    "between": "Between reloads",
    "after": "After reload",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numbers(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median(values: Iterable[Any]) -> float | None:
    rows = numbers(values)
    return round(statistics.median(rows), 3) if rows else None


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    trials = list(run.get("trials") or [])
    condition = str(run.get("condition") or "")
    regions: dict[str, dict[str, Any]] = {}
    for name in REGION_LABELS:
        rows = [trial.get("regions", {}).get(name, {}) for trial in trials]
        regions[name] = {
            "median_interval_ms": median(row.get("median_interval_ms") for row in rows),
            "median_p95_interval_ms": median(row.get("p95_interval_ms") for row in rows),
            "median_max_interval_ms": median(row.get("max_interval_ms") for row in rows),
            "median_visible_updates_per_second": median(
                row.get("visible_updates_per_second") for row in rows
            ),
            "interval_count": sum(int(row.get("interval_count") or 0) for row in rows),
        }
    return {
        "condition": condition,
        "label": CONDITION_LABELS.get(condition, condition),
        "trials": len(trials),
        "valid_trials": sum(bool(trial.get("pressure_proof", {}).get("valid")) for trial in trials),
        "median_ttft_ms": median(trial.get("decode", {}).get("ttft_ms") for trial in trials),
        "median_total_decode_ms": median(
            trial.get("decode", {}).get("total_latency_ms") for trial in trials
        ),
        "median_completed_loads": median(
            trial.get("pressure_proof", {}).get("completed_loads") for trial in trials
        ),
        "median_total_loaded_tokens": median(
            trial.get("pressure_proof", {}).get("total_loaded_tokens") for trial in trials
        ),
        "median_total_cuda_load_ms": median(
            trial.get("pressure_proof", {}).get("total_cuda_load_ms") for trial in trials
        ),
        "median_pressure_envelope_ms": median(
            trial.get("pressure_proof", {}).get("pressure_envelope_ms") for trial in trials
        ),
        "median_pressure_share_of_decode_pct": median(
            trial.get("pressure_proof", {}).get("pressure_share_of_decode_pct") for trial in trials
        ),
        "median_maximum_inter_load_gap_ms": median(
            trial.get("pressure_proof", {}).get("maximum_inter_load_gap_ms") for trial in trials
        ),
        "median_observed_load_duty_cycle_pct": median(
            trial.get("pressure_proof", {}).get("observed_load_duty_cycle_pct") for trial in trials
        ),
        "regions": regions,
    }


def paired_decode_deltas(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    controls = {
        trial["sample_id"]: float(trial["decode"]["total_latency_ms"])
        for trial in runs["decode_control"].get("trials", [])
    }
    result: dict[str, Any] = {}
    for condition in ("single_reload", "sustained_reload"):
        rows = []
        for trial in runs[condition].get("trials", []):
            sample_id = trial.get("sample_id")
            if sample_id in controls:
                rows.append(float(trial["decode"]["total_latency_ms"]) - controls[sample_id])
        result[condition] = {
            "paired_samples": len(rows),
            "median_decode_change_ms": median(rows),
            "mean_decode_change_ms": round(statistics.mean(rows), 3) if rows else None,
            "changes_ms": [round(value, 3) for value in rows],
        }
    return result


def fmt(value: Any, suffix: str = " ms") -> str:
    return "n/a" if value is None else f"{float(value):,.3f}{suffix}"


def timeline(condition: str, trial: dict[str, Any], *, zoom: bool = False) -> str:
    decode = trial["decode"]
    request_start = int(decode["request_start_ns"])
    request_end = int(decode["request_end_ns"])
    loads = trial.get("loads") or []
    if zoom and loads:
        view_start = max(request_start, min(int(load["start_ns"]) for load in loads) - 500_000_000)
        view_end = min(request_end, max(int(load["finish_ns"]) for load in loads) + 500_000_000)
    else:
        view_start, view_end = request_start, request_end
    span = max(1, view_end - view_start)
    bars = []
    for load in loads:
        start = max(view_start, int(load["start_ns"]))
        finish = min(view_end, int(load["finish_ns"]))
        if finish <= start:
            continue
        left = 100.0 * (start - view_start) / span
        width = max(0.25, 100.0 * (finish - start) / span)
        bars.append(
            f'<span class="load" style="left:{left:.4f}%;width:{width:.4f}%" '
            f'title="Load {int(load["donor_index"]) + 1}: '
            f'{fmt(load.get("cuda_duration_ms"))}"></span>'
        )
    label = "pressure-window zoom" if zoom else "complete decode"
    duration_ms = span / 1_000_000
    return (
        f'<div class="timeline-row"><div><strong>{html.escape(CONDITION_LABELS[condition])}</strong> '
        f'<span>{label}; {duration_ms:,.1f} ms</span></div>'
        f'<div class="timeline"><span class="decode"></span>{"".join(bars)}</div></div>'
    )


def build_html(summary: dict[str, Any], runs: dict[str, dict[str, Any]]) -> str:
    summaries = summary["conditions"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{row['valid_trials']}/{row['trials']}</td>"
        f"<td>{fmt(row['median_ttft_ms'])}</td>"
        f"<td>{fmt(row['median_total_decode_ms'])}</td>"
        f"<td>{fmt(row['median_completed_loads'], '')}</td>"
        f"<td>{fmt(row['median_total_loaded_tokens'], ' tokens')}</td>"
        f"<td>{fmt(row['median_total_cuda_load_ms'])}</td>"
        f"<td>{fmt(row['median_pressure_envelope_ms'])}</td>"
        f"<td>{fmt(row['median_pressure_share_of_decode_pct'], '%')}</td>"
        "</tr>"
        for row in summaries
    )
    region_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td><td>{html.escape(REGION_LABELS[region])}</td>"
        f"<td>{fmt(row['regions'][region]['median_interval_ms'])}</td>"
        f"<td>{fmt(row['regions'][region]['median_p95_interval_ms'])}</td>"
        f"<td>{fmt(row['regions'][region]['median_max_interval_ms'])}</td>"
        f"<td>{fmt(row['regions'][region]['median_visible_updates_per_second'], ' updates/s')}</td>"
        f"<td>{row['regions'][region]['interval_count']}</td></tr>"
        for row in summaries
        for region in REGION_LABELS
        if row["regions"][region]["interval_count"]
    )
    delta_rows = "".join(
        "<tr>"
        f"<td>{html.escape(CONDITION_LABELS[condition])}</td>"
        f"<td>{value['paired_samples']}</td>"
        f"<td>{fmt(value['median_decode_change_ms'])}</td>"
        f"<td>{fmt(value['mean_decode_change_ms'])}</td></tr>"
        for condition, value in summary["paired_decode_deltas"].items()
    )
    timeline_rows = "".join(
        timeline(condition, runs[condition]["trials"][0])
        for condition in CONDITION_LABELS
    )
    zoom_rows = "".join(
        timeline(condition, runs[condition]["trials"][0], zoom=True)
        for condition in ("single_reload", "sustained_reload")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sustained KV Pressure During Decode</title><style>
:root {{ font-family: Arial, sans-serif; color: #172033; background: #f6f8fb; }}
body {{ max-width: 1280px; margin: 36px auto; padding: 0 20px 60px; line-height: 1.45; }}
h1 {{ margin-bottom: .15rem; }} .sub,.timeline-row span {{ color: #52627a; }}
section {{ margin: 26px 0; padding: 20px; border: 1px solid #d8e0ea; background: white; border-radius: 6px; overflow-x:auto; }}
table {{ border-collapse: collapse; width: 100%; }} th,td {{ text-align:left; padding:9px; border-bottom:1px solid #d8e0ea; white-space:nowrap; }}
th {{ background:#eef3f8; }} .timeline-row {{ margin:16px 0; }} .timeline-row span {{ margin-left:8px; font-size:13px; }}
.timeline {{ height:28px; position:relative; margin-top:6px; background:#e8eef4; border:1px solid #becbd6; overflow:hidden; }}
.timeline .decode {{ position:absolute; inset:8px 0; background:#4f6074; margin:0; }}
.timeline .load {{ position:absolute; top:2px; height:22px; background:#df4a36; min-width:2px; margin:0; z-index:2; }}
.warn {{ border-left:3px solid #b66a18; padding-left:12px; }}
</style></head><body>
<h1>Sustained KV Pressure During Decode</h1>
<p class="sub">Equal frontend semantics. Red windows are native host-to-device KV reloads while the gray track is target decode.</p>
<section><h2>Condition Summary</h2><table><thead><tr><th>Condition</th><th>Proof</th><th>TTFT</th><th>Total decode</th><th>Loads</th><th>KV loaded</th><th>Total CUDA load</th><th>Pressure envelope</th><th>Decode overlap</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Paired Total-Decode Change Versus Control</h2><table><thead><tr><th>Condition</th><th>Pairs</th><th>Median change</th><th>Mean change</th></tr></thead><tbody>{delta_rows}</tbody></table></section>
<section><h2>First-Trial Timeline</h2>{timeline_rows}<h3>Pressure Window Zoom</h3>{zoom_rows}</section>
<section><h2>Visible Output Timing</h2><table><thead><tr><th>Condition</th><th>Region</th><th>Median interval</th><th>p95 interval</th><th>Maximum interval</th><th>Update rate</th><th>Intervals</th></tr></thead><tbody>{region_rows}</tbody></table></section>
<section><h2>Proof Boundary</h2><p>Each non-control trial must prove that every expected native load completed with positive CUDA duration inside the active decode. The sustained case also enforces a minimum total load duration and maximum gap between reloads.</p><p class="warn">Client-visible stream updates are not assumed to equal individual generated tokens. This lightweight report proves temporal overlap. Physical memory-bandwidth attribution still requires a separate profiler pass whose timings must not be pooled with these runs.</p></section>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--sustained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    rows = [load_json(args.control), load_json(args.single), load_json(args.sustained)]
    runs = {str(row.get("condition")): row for row in rows}
    if set(runs) != set(CONDITION_LABELS):
        raise SystemExit(f"Expected {sorted(CONDITION_LABELS)}, got {sorted(runs)}")
    if len({row.get("sample_set_id") for row in rows}) != 1:
        raise SystemExit("Conditions do not share one sample set.")
    if len({row.get("model") for row in rows}) != 1:
        raise SystemExit("Conditions do not share one model.")
    trial_counts = {len(row.get("trials") or []) for row in rows}
    if len(trial_counts) != 1 or not next(iter(trial_counts), 0):
        raise SystemExit("Conditions do not contain the same nonzero number of trials.")
    for condition, run in runs.items():
        failures = [
            trial.get("sample_id")
            for trial in run.get("trials", [])
            if not trial.get("pressure_proof", {}).get("valid")
        ]
        if failures:
            raise SystemExit(f"{condition} has failed proof gates: {failures}")

    summary = {
        "schema_version": "sustained_kv_pressure_report.v1",
        "sample_set_id": rows[0].get("sample_set_id"),
        "model": rows[0].get("model"),
        "conditions": [summarize(runs[name]) for name in CONDITION_LABELS],
        "paired_decode_deltas": paired_decode_deltas(runs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(summary, runs), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
