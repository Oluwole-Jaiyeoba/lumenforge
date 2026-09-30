#!/usr/bin/env python3
"""Build a narrowly scoped report for native KV load-back attribution."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


LABELS = {
    "target_only": "Target only",
    "device_resident_control": "Device-resident control",
    "host_reload_collision": "Host reload during decode",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: Iterable[Any]) -> float | None:
    rows = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.median(rows), 3) if rows else None


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    trials = list(run.get("trials") or [])
    return {
        "condition": run["condition"], "label": LABELS[run["condition"]], "trials": len(trials),
        "valid_trials": sum(bool(row.get("valid")) for row in trials),
        "median_ttft_ms": median(row.get("decode", {}).get("ttft_ms") for row in trials),
        "median_total_decode_ms": median(row.get("decode", {}).get("total_latency_ms") for row in trials),
        "median_native_load_ms": median(row.get("cuda_load_duration_ms") for row in trials),
        "median_loaded_tokens": median(row.get("loaded_tokens") for row in trials),
        "native_overlap_trials": sum(bool(row.get("native_overlap")) for row in trials),
    }


def delta(control: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    baseline = {row.get("sample_id"): row.get("decode", {}).get("total_latency_ms") for row in control.get("trials") or []}
    values = [float(row["decode"]["total_latency_ms"]) - float(baseline[row.get("sample_id")])
              for row in other.get("trials") or []
              if row.get("sample_id") in baseline and isinstance(row.get("decode", {}).get("total_latency_ms"), (int, float))
              and isinstance(baseline[row.get("sample_id")], (int, float))]
    return {"pairs": len(values), "median_delta_ms": median(values), "mean_delta_ms": round(statistics.mean(values), 3) if values else None}


def fmt(value: Any, suffix: str = " ms") -> str:
    return "n/a" if value is None else f"{float(value):,.3f}{suffix}"


def build_html(summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td><td>{row['valid_trials']}/{row['trials']}</td>"
        f"<td>{fmt(row['median_ttft_ms'])}</td><td>{fmt(row['median_total_decode_ms'])}</td>"
        f"<td>{row['native_overlap_trials']}</td><td>{fmt(row['median_native_load_ms'])}</td>"
        f"<td>{fmt(row['median_loaded_tokens'], ' tokens')}</td></tr>"
        for row in summary["conditions"]
    )
    delta_rows = "".join(
        "<tr>"
        f"<td>{html.escape(LABELS[name])}</td><td>{row['pairs']}</td>"
        f"<td>{fmt(row['median_delta_ms'])}</td><td>{fmt(row['mean_delta_ms'])}</td></tr>"
        for name, row in summary["paired_decode_change_vs_target_only"].items()
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Native KV Reload Attribution</title><style>
:root {{ font-family: Arial,sans-serif; color:#152238; background:#f4f7fb; }} body {{ max-width:1180px;margin:36px auto;padding:0 20px 56px;line-height:1.45; }}
h1 {{ margin-bottom:.15rem; }} .sub {{ color:#52627a; }} section {{ background:#fff;border:1px solid #d7e1ed;border-radius:6px;padding:20px;margin:22px 0;overflow-x:auto; }}
table {{ width:100%;border-collapse:collapse; }} th,td {{ text-align:left;padding:10px;border-bottom:1px solid #dbe4ee;white-space:nowrap; }} th {{ background:#e9f2f6; }} .note {{ border-left:4px solid #168a88;padding-left:12px; }} .limit {{ color:#8a4b00; }}
</style></head><body><h1>Native KV Reload Attribution</h1>
<p class="sub">Equal frontend treatment. This timing lane asks whether SGLang's own host-to-device KV load-back overlaps and slows an already active target decode.</p>
<section><h2>Measured Conditions</h2><table><thead><tr><th>Condition</th><th>Proof</th><th>TTFT</th><th>Total decode</th><th>Native overlaps</th><th>Native CUDA load</th><th>KV loaded during decode</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Paired Decode Change Versus Target Only</h2><table><thead><tr><th>Condition</th><th>Pairs</th><th>Median change</th><th>Mean change</th></tr></thead><tbody>{delta_rows}</tbody></table></section>
<section><h2>What Each Case Means</h2><p><strong>Target only:</strong> a host-resident donor exists, but no donor action occurs after decode begins.</p><p><strong>Device-resident control:</strong> the donor is loaded before target decode, then the same post-start coordinator checkpoint performs a no-load state check.</p><p><strong>Host reload during decode:</strong> the donor is host-resident when target decode starts; the run is accepted only if SGLang reports a positive native CUDA load that begins and ends inside target decode.</p></section>
<section><h2>Boundary</h2><p class="note">This proves a clean software timeline using SGLang-native load-back events. It does not by itself prove device-memory bandwidth saturation. Keep profiler measurements in a separate run and compare them against this timing lane rather than pooling them.</p><p class="limit">On the current standard NVIDIA runtime, describe this as device-memory attribution. Reserve claims specifically about HBM for a platform with HBM and a profiler pass.</p></section>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-only", type=Path, required=True)
    parser.add_argument("--device-resident", type=Path, required=True)
    parser.add_argument("--host-reload", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    runs = [load(args.target_only), load(args.device_resident), load(args.host_reload)]
    if {row.get("condition") for row in runs} != set(LABELS):
        raise SystemExit("Expected target_only, device_resident_control, and host_reload_collision.")
    if len({row.get("sample_set_id") for row in runs}) != 1 or len({row.get("model") for row in runs}) != 1:
        raise SystemExit("Conditions must use the same sample set and model.")
    ordered = {row["condition"]: row for row in runs}
    for name, run in ordered.items():
        failures = [row.get("sample_id") for row in run.get("trials") or [] if not row.get("valid")]
        if failures:
            raise SystemExit(f"{name} failed its attribution proof: {failures}")
    summary = {
        "schema_version": "device_memory_attribution_report.v1", "sample_set_id": runs[0]["sample_set_id"], "model": runs[0]["model"],
        "conditions": [summarize(ordered[name]) for name in LABELS],
        "paired_decode_change_vs_target_only": {
            "device_resident_control": delta(ordered["target_only"], ordered["device_resident_control"]),
            "host_reload_collision": delta(ordered["target_only"], ordered["host_reload_collision"]),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(summary), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
