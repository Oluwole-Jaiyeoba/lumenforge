#!/usr/bin/env python3
"""Build a compact report for a four-level KV-pressure duty sweep."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


CONDITION_LABELS = {
    "decode_control": "Control: no KV reload",
    "pressure_low": "Low KV pressure",
    "pressure_medium": "Medium KV pressure",
    "pressure_high": "High KV pressure",
}


def numbers(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median(values: Iterable[Any]) -> float | None:
    rows = numbers(values)
    return round(statistics.median(rows), 3) if rows else None


def fmt(value: Any, suffix: str = " ms") -> str:
    return "n/a" if value is None else f"{float(value):,.3f}{suffix}"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    trials = list(run.get("trials") or [])
    proof = [trial.get("pressure_proof") or {} for trial in trials]
    return {
        "condition": str(run.get("condition") or ""),
        "label": CONDITION_LABELS.get(str(run.get("condition") or ""), str(run.get("condition") or "")),
        "trials": len(trials),
        "valid_trials": sum(bool(row.get("valid")) for row in proof),
        "requested_loads": run.get("load_count"),
        "ttft_ms": median(trial.get("decode", {}).get("ttft_ms") for trial in trials),
        "total_decode_ms": median(trial.get("decode", {}).get("total_latency_ms") for trial in trials),
        "completed_loads": median(row.get("completed_loads") for row in proof),
        "recycle_proofs": median(row.get("recycle_proofs") for row in proof),
        "loaded_tokens": median(row.get("total_loaded_tokens") for row in proof),
        "cuda_load_ms": median(row.get("total_cuda_load_ms") for row in proof),
        "cuda_share_pct": median(row.get("cuda_load_share_of_decode_pct") for row in proof),
        "envelope_ms": median(row.get("pressure_envelope_ms") for row in proof),
        "envelope_share_pct": median(row.get("pressure_envelope_share_of_decode_pct") for row in proof),
    }


def paired_changes(runs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    control = {
        trial.get("sample_id"): float(trial["decode"]["total_latency_ms"])
        for trial in runs["decode_control"].get("trials", [])
    }
    result: dict[str, dict[str, Any]] = {}
    for condition in CONDITION_LABELS:
        if condition == "decode_control":
            continue
        values = [
            float(trial["decode"]["total_latency_ms"]) - control[trial.get("sample_id")]
            for trial in runs[condition].get("trials", [])
            if trial.get("sample_id") in control
        ]
        result[condition] = {
            "paired_samples": len(values),
            "median_decode_change_ms": median(values),
            "mean_decode_change_ms": round(statistics.mean(values), 3) if values else None,
            "changes_ms": [round(value, 3) for value in values],
        }
    return result


def build_html(summary: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td><td>{row['valid_trials']}/{row['trials']}</td>"
        f"<td>{row['requested_loads']}</td><td>{fmt(row['ttft_ms'])}</td>"
        f"<td>{fmt(row['total_decode_ms'])}</td><td>{fmt(row['completed_loads'], '')}</td>"
        f"<td>{fmt(row['recycle_proofs'], '')}</td><td>{fmt(row['loaded_tokens'], ' token slots')}</td>"
        f"<td>{fmt(row['cuda_load_ms'])}</td><td>{fmt(row['cuda_share_pct'], '%')}</td>"
        f"<td>{fmt(row['envelope_ms'])}</td><td>{fmt(row['envelope_share_pct'], '%')}</td></tr>"
        for row in summary["conditions"]
    )
    changes = "".join(
        "<tr>"
        f"<td>{html.escape(CONDITION_LABELS[condition])}</td><td>{value['paired_samples']}</td>"
        f"<td>{fmt(value['median_decode_change_ms'])}</td><td>{fmt(value['mean_decode_change_ms'])}</td></tr>"
        for condition, value in summary["paired_decode_changes"].items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KV Pressure Duty Sweep</title><style>
:root {{ font-family: Arial, sans-serif; color:#172033; background:#f6f8fb; }}
body {{ max-width:1400px; margin:36px auto; padding:0 20px 60px; line-height:1.45; }}
section {{ margin:24px 0; padding:20px; border:1px solid #d8e0ea; background:#fff; border-radius:6px; overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ text-align:left; padding:9px; border-bottom:1px solid #d8e0ea; white-space:nowrap; }} th {{ background:#eef3f8; }}
.sub {{ color:#52627a; }} .warn {{ border-left:3px solid #b66a18; padding-left:12px; }}
</style></head><body>
<h1>KV Pressure Duty Sweep</h1>
<p class="sub">All frontend requests have equal semantics. The only change is the number of native host-to-GPU KV reloads verified during the same decode.</p>
<section><h2>Measured Conditions</h2><table><thead><tr><th>Condition</th><th>Proof</th><th>Requested loads</th><th>TTFT</th><th>Total decode</th><th>Completed loads</th><th>Recycle proofs</th><th>KV loaded</th><th>CUDA load time</th><th>CUDA share of decode</th><th>Pressure envelope</th><th>Envelope share</th></tr></thead><tbody>{rows}</tbody></table></section>
<section><h2>Paired Decode Change Versus Control</h2><table><thead><tr><th>Condition</th><th>Pairs</th><th>Median change</th><th>Mean change</th></tr></thead><tbody>{changes}</tbody></table></section>
<section><h2>Interpretation Boundary</h2><p>Each pressure run must prove every requested native reload completed with positive CUDA duration during active decoding. A recycle proof confirms that a donor was released from GPU memory while retaining a host-backed copy eligible for another native reload.</p><p class="warn">The measured CUDA-load share is the authoritative pressure value. The labels low, medium, and high describe the requested reload budget, not an assumed amount of physical contention.</p></section>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "decode_control": args.control,
        "pressure_low": args.low,
        "pressure_medium": args.medium,
        "pressure_high": args.high,
    }
    runs = {condition: load(path) for condition, path in paths.items()}
    if any(run.get("condition") != condition for condition, run in runs.items()):
        raise SystemExit("A supplied probe does not match its expected condition.")
    if len({run.get("sample_set_id") for run in runs.values()}) != 1:
        raise SystemExit("Conditions do not share one sample set.")
    if len({run.get("model") for run in runs.values()}) != 1:
        raise SystemExit("Conditions do not share one model.")
    if len({len(run.get("trials") or []) for run in runs.values()}) != 1:
        raise SystemExit("Conditions do not contain the same number of trials.")
    for condition, run in runs.items():
        failures = [trial.get("sample_id") for trial in run.get("trials", []) if not (trial.get("pressure_proof") or {}).get("valid")]
        if failures:
            raise SystemExit(f"{condition} has failed proof gates: {failures}")
    summary = {
        "schema_version": "kv_pressure_duty_sweep_report.v1",
        "sample_set_id": next(iter(runs.values())).get("sample_set_id"),
        "model": next(iter(runs.values())).get("model"),
        "conditions": [summarize(runs[condition]) for condition in CONDITION_LABELS],
        "paired_decode_changes": paired_changes(runs),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(summary), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
