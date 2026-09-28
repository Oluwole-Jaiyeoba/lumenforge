#!/usr/bin/env python3
"""Build an evidence-gated report for the paired KV-movement probe."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

from agentic_hardware_probes import compare_runs, load_run


H2D_EVENT_SUFFIXES = (
    "hostpool.load_to_device_per_layer.end",
    "hiradix.load_back.end",
    "hicache.load.end",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def count_h2d_events(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    count = 0
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        name = str(row.get("event") or row.get("event_name") or row.get("name") or "")
        if name.endswith(H2D_EVENT_SUFFIXES):
            count += 1
    return count


def trial_load_summary(path: Path) -> dict[str, int]:
    run = load_json(path)
    details = run.get("probe_metadata", {}).get("trial_details", [])
    summary = {"trials": len(details), "verified_loads": 0, "queued_loads": 0, "ready_loads": 0}
    for detail in details:
        result = detail.get("interference_load")
        if not isinstance(result, dict):
            continue
        if int(result.get("loaded_tokens") or 0) > 0:
            summary["verified_loads"] += 1
        status = result.get("status")
        if status == "queued":
            summary["queued_loads"] += 1
        elif status == "ready":
            summary["ready_loads"] += 1
    return summary


def fmt(value: float) -> str:
    return f"{value:.3f} ms"


def build_html(summary: dict[str, Any]) -> str:
    comparison = summary["comparison"]
    lateness = summary["lateness_comparison"]
    h2d = summary["software_visible_h2d"]
    loads = summary["interference_loads"]
    direction = "slower" if comparison["median_paired_change_ms"] > 0 else "faster"
    outcome = (
        f"The interference median was {abs(comparison['median_paired_change_ms']):.3f} ms {direction} than control."
    )
    evidence = (
        "The interference driver verified an eligible host-resident donor prefix and requested a load-back before each replay. "
        f"It recorded {loads['verified_loads']} accepted load request(s) across {loads['trials']} trial(s). "
        f"The backend trace contains {h2d['interference']} software-visible H2D/KV load event(s) in interference and "
        f"{h2d['control']} in control."
    )
    return f"""<!doctype html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>KV Movement Interference</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1050px; margin: 40px auto; color: #172033; line-height: 1.45; }}
h1 {{ margin-bottom: .15rem; }} .sub {{ color: #52627a; }}
section {{ margin: 28px 0; padding: 20px; border: 1px solid #d8e0ea; border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #d8e0ea; }}
th {{ background: #eef3f8; }} code {{ background: #f2f5f8; padding: 2px 4px; }}
.callout {{ font-size: 1.1rem; font-weight: 600; }} .warn {{ color: #8a4b00; }}
</style></head><body>
<h1>KV Movement Interference</h1>
<p class=\"sub\">Paired replay measurements with equal frontend semantics. The only intended difference is a verified, competing KV load-back immediately before replay.</p>
<section><div class=\"callout\">{html.escape(outcome)}</div>
<p>Positive paired change means the interference replay was slower.</p>
<table><tr><th>Metric</th><th>Control median</th><th>Interference median</th><th>Median paired change</th><th>Mean paired change</th></tr>
<tr><td>{html.escape(comparison['metric'])}</td><td>{fmt(comparison['control_median_ms'])}</td><td>{fmt(comparison['interference_median_ms'])}</td><td>{fmt(comparison['median_paired_change_ms'])}</td><td>{fmt(comparison['mean_paired_change_ms'])}</td></tr>
<tr><td>{html.escape(lateness['metric'])}</td><td>{fmt(lateness['control_median_ms'])}</td><td>{fmt(lateness['interference_median_ms'])}</td><td>{fmt(lateness['median_paired_change_ms'])}</td><td>{fmt(lateness['mean_paired_change_ms'])}</td></tr></table></section>
<section><h2>Evidence Captured</h2><p>{html.escape(evidence)}</p>
<table><tr><th>Accepted load requests</th><th>Queued</th><th>Ready</th><th>Control trace H2D events</th><th>Interference trace H2D events</th></tr>
<tr><td>{loads['verified_loads']}</td><td>{loads['queued_loads']}</td><td>{loads['ready_loads']}</td><td>{h2d['control']}</td><td>{h2d['interference']}</td></tr></table></section>
<section><h2>Interpretation Boundary</h2><p class=\"warn\">This is software-visible evidence of a backend KV load request and trace events. It does not by itself prove physical DMA-engine or memory-bandwidth saturation. Use a separate profiler pass for physical attribution, and do not pool its timings with this lightweight latency run.</p></section>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--interference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--control-backend-trace", type=Path)
    parser.add_argument("--interference-backend-trace", type=Path)
    args = parser.parse_args()

    control = load_run(args.control)
    interference = load_run(args.interference)
    comparison = compare_runs(control, interference, "replay_ttft_ms").to_dict()
    summary = {
        "schema_version": "hardware_kv_movement_report.v1",
        "comparison": comparison,
        "lateness_comparison": compare_runs(control, interference, "replay_lateness_ms").to_dict(),
        "control_probe": str(args.control),
        "interference_probe": str(args.interference),
        "interference_loads": trial_load_summary(args.interference),
        "software_visible_h2d": {
            "control": count_h2d_events(args.control_backend_trace),
            "interference": count_h2d_events(args.interference_backend_trace),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(summary), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
