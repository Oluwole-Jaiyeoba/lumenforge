#!/usr/bin/env python3
"""Build the report for sustained decode versus native KV load-back."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


CONDITION_LABELS = {
    "decode_control": "Decode control",
    "non_overlap_reload": "Reload before decode",
    "direct_overlap_reload": "Reload during decode",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if isinstance(value, (int, float))]


def median(values: Iterable[Any]) -> float | None:
    rows = numeric(values)
    return round(statistics.median(rows), 3) if rows else None


def effective_load_start_ns(trial: dict[str, Any]) -> tuple[int | None, str]:
    if trial.get("active_status_observed") and trial.get("load_started_ns"):
        return int(trial["load_started_ns"]), "active_poll"
    status_rows = trial.get("load_status_history") or []
    final_status = status_rows[-1] if status_rows else {}
    load_result = trial.get("load_result") or {}
    for source, value in (
        ("server_command", final_status.get("command_started_ns")),
        ("server_command", load_result.get("command_started_ns")),
        ("client_request", load_result.get("control_request_started_ns")),
    ):
        if isinstance(value, (int, float)) and int(value) > 0:
            return int(value), source
    # Older sanity artifacts encoded the server-side command start as the last
    # component of load_id. Recovering it keeps the raw artifact immutable.
    load_id = str(load_result.get("load_id") or "")
    try:
        encoded = int(load_id.rsplit(":", 1)[-1])
    except (TypeError, ValueError):
        encoded = 0
    return (encoded, "load_id") if encoded > 0 else (None, "unavailable")


def recompute_regions(trial: dict[str, Any], start_ns: int | None) -> dict[str, dict[str, Any]]:
    finish_ns = trial.get("load_finished_ns")
    chunks = trial.get("decode", {}).get("chunk_times_ns") or []
    values: dict[str, list[float]] = {"before": [], "during": [], "after": []}
    for previous, current in zip(chunks, chunks[1:]):
        latency_ms = (current - previous) / 1_000_000
        if start_ns is None or current <= start_ns:
            values["before"].append(latency_ms)
        elif isinstance(finish_ns, (int, float)) and previous >= int(finish_ns):
            values["after"].append(latency_ms)
        else:
            values["during"].append(latency_ms)
    result = {}
    for name, rows in values.items():
        ordered = sorted(rows)
        p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1)) if ordered else 0
        result[name] = {
            "interval_count": len(rows),
            "median_itl_ms": round(statistics.median(rows), 3) if rows else None,
            "p95_itl_ms": round(ordered[p95_index], 3) if rows else None,
        }
    return result


def summarize(run: dict[str, Any]) -> dict[str, Any]:
    trials = list(run.get("trials") or [])
    starts = [effective_load_start_ns(trial) for trial in trials]
    trial_regions = [
        recompute_regions(trial, start)
        if trial.get("load_finished_ns") and start
        else trial.get("regions", {})
        for trial, (start, _source) in zip(trials, starts)
    ]
    all_intervals: list[float] = []
    for trial in trials:
        chunks = trial.get("decode", {}).get("chunk_times_ns") or []
        all_intervals.extend((current - previous) / 1_000_000 for previous, current in zip(chunks, chunks[1:]))

    regions: dict[str, dict[str, Any]] = {}
    for name in ("before", "during", "after"):
        rows = [regions.get(name, {}) for regions in trial_regions]
        regions[name] = {
            "median_itl_ms": median(row.get("median_itl_ms") for row in rows),
            "median_p95_itl_ms": median(row.get("p95_itl_ms") for row in rows),
            "interval_count": sum(int(row.get("interval_count") or 0) for row in rows),
        }

    condition = str(run.get("condition") or "")
    return {
        "condition": condition,
        "label": CONDITION_LABELS.get(condition, condition),
        "trials": len(trials),
        "valid_overlap_trials": sum(bool(trial.get("valid_overlap")) for trial in trials),
        "median_ttft_ms": median(trial.get("decode", {}).get("ttft_ms") for trial in trials),
        "median_total_decode_ms": median(trial.get("decode", {}).get("total_latency_ms") for trial in trials),
        "median_overall_itl_ms": median(all_intervals),
        "median_overlap_duration_ms": median(
            (int(trial.get("load_finished_ns")) - start) / 1_000_000
            for trial, (start, _source) in zip(trials, starts)
            if start and trial.get("load_finished_ns")
        ),
        "median_cuda_load_duration_ms": median(trial.get("cuda_load_duration_ms") for trial in trials),
        "load_start_sources": sorted({source for _start, source in starts}),
        "median_loaded_tokens": median(
            trial.get("load_result", {}).get("loaded_tokens")
            for trial in trials
            if isinstance(trial.get("load_result"), dict)
        ),
        "regions": regions,
    }


def fmt(value: Any, suffix: str = " ms") -> str:
    return "n/a" if value is None else f"{float(value):,.3f}{suffix}"


def build_html(summary: dict[str, Any]) -> str:
    rows = summary["conditions"]
    direct = next(row for row in rows if row["condition"] == "direct_overlap_reload")
    validity = (
        f"{direct['valid_overlap_trials']}/{direct['trials']} direct-overlap trials satisfied the ordering and token-interval gate."
    )
    body_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td>{row['trials']}</td>"
        f"<td>{fmt(row['median_ttft_ms'])}</td>"
        f"<td>{fmt(row['median_total_decode_ms'])}</td>"
        f"<td>{fmt(row['median_overall_itl_ms'])}</td>"
        f"<td>{fmt(row['regions']['during']['median_itl_ms'])}</td>"
        f"<td>{fmt(row['median_overlap_duration_ms'])}</td>"
        f"<td>{fmt(row['median_cuda_load_duration_ms'])}</td>"
        f"<td>{fmt(row['median_loaded_tokens'], ' tokens')}</td>"
        "</tr>"
        for row in rows
    )
    region_rows = "".join(
        "<tr>"
        f"<td>{html.escape(region.title())}</td>"
        f"<td>{fmt(direct['regions'][region]['median_itl_ms'])}</td>"
        f"<td>{fmt(direct['regions'][region]['median_p95_itl_ms'])}</td>"
        f"<td>{direct['regions'][region]['interval_count']}</td>"
        "</tr>"
        for region in ("before", "during", "after")
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Sustained Decode KV Overlap</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1180px; margin: 40px auto; color: #172033; line-height: 1.45; }}
h1 {{ margin-bottom: .15rem; }} .sub {{ color: #52627a; }}
section {{ margin: 28px 0; padding: 20px; border: 1px solid #d8e0ea; border-radius: 6px; }}
table {{ border-collapse: collapse; width: 100%; }} th, td {{ text-align: left; padding: 10px; border-bottom: 1px solid #d8e0ea; }}
th {{ background: #eef3f8; }} .callout {{ font-size: 1.08rem; font-weight: 600; }} .warn {{ color: #8a4b00; }}
</style></head><body>
<h1>Sustained Decode Versus KV Reload</h1>
<p class="sub">Equal frontend semantics. This isolates whether native host-to-device KV load-back slows an already-running decode.</p>
<section><div class="callout">{html.escape(validity)}</div>
<table><tr><th>Condition</th><th>Trials</th><th>TTFT</th><th>Total decode</th><th>Overall stream interval</th><th>Stream interval during reload</th><th>Observed load envelope</th><th>CUDA load duration</th><th>Loaded KV</th></tr>{body_rows}</table></section>
<section><h2>Direct-Overlap Decode Timing</h2>
<table><tr><th>Region</th><th>Median stream-update interval</th><th>Median trial p95</th><th>Intervals observed</th></tr>{region_rows}</table></section>
<section><h2>Validity Rule</h2><p>A direct-overlap trial is valid only when decode starts first, SGLang queues a native load, its CUDA start/finish events report positive physical duration, at least the configured number of stream intervals intersects the observed load envelope, the load finishes, and decode finishes last.</p></section>
<section><h2>Interpretation Boundary</h2><p class="warn">The lightweight run uses SGLang's CUDA start/finish events and client-visible stream updates. A stream update is not assumed to equal exactly one generated token. The run proves software-visible temporal overlap, but it does not measure memory-controller bandwidth. A separate profiler run is required for physical bandwidth attribution, and its timing must not be pooled with this report.</p></section>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--non-overlap", type=Path, required=True)
    parser.add_argument("--direct-overlap", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    runs = [load_json(args.control), load_json(args.non_overlap), load_json(args.direct_overlap)]
    sample_sets = {run.get("sample_set_id") for run in runs}
    models = {run.get("model") for run in runs}
    if len(sample_sets) != 1 or len(models) != 1:
        raise SystemExit("Conditions do not share the same sample set and model.")
    conditions = {run.get("condition") for run in runs}
    if conditions != set(CONDITION_LABELS):
        raise SystemExit(f"Expected conditions {sorted(CONDITION_LABELS)}, got {sorted(conditions)}")

    ordered = {run["condition"]: run for run in runs}
    summary = {
        "schema_version": "sustained_decode_kv_overlap_report.v1",
        "sample_set_id": next(iter(sample_sets)),
        "model": next(iter(models)),
        "conditions": [summarize(ordered[name]) for name in CONDITION_LABELS],
    }
    direct = next(row for row in summary["conditions"] if row["condition"] == "direct_overlap_reload")
    if direct["valid_overlap_trials"] != direct["trials"]:
        raise SystemExit("At least one direct-overlap trial failed the proof gate; refusing to build the report.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(summary), encoding="utf-8")
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
