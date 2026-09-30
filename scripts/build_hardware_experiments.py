#!/usr/bin/env python3
"""Render the hardware lane from its evidence-gated JSON registry."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_LABELS = {
    "planned": "Planned - not measured",
    "measured_pending_review": "Measured - review pending",
    "validated": "Validated result",
}
RESULT_FIELDS = ("run_id", "command", "report_path", "manifest_path", "evidence_path")


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def validate_registry(registry: dict[str, object], root: Path = ROOT) -> None:
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported hardware experiment registry version")
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("registry needs at least one experiment")
    ids: set[str] = set()
    required = (
        "id", "name", "status", "question", "platform", "backend", "control",
        "interference", "held_constant", "measurements", "evidence_rule", "results",
    )
    for item in experiments:
        if not isinstance(item, dict) or any(not item.get(key) for key in required if key != "results"):
            raise ValueError("experiment is missing a required field")
        experiment_id = item["id"]
        if not isinstance(experiment_id, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", experiment_id):
            raise ValueError(f"invalid experiment ID: {experiment_id}")
        if experiment_id in ids:
            raise ValueError(f"duplicate experiment ID: {experiment_id}")
        ids.add(experiment_id)
        status = item["status"]
        if status not in STATUS_LABELS:
            raise ValueError(f"{experiment_id}: unknown status {status}")
        if not isinstance(item["measurements"], list) or not item["measurements"]:
            raise ValueError(f"{experiment_id}: measurements must be a nonempty list")
        if not isinstance(item["results"], dict):
            raise ValueError(f"{experiment_id}: results must be an object")
        if status == "planned":
            if item["results"] or any(item.get(key) for key in RESULT_FIELDS):
                raise ValueError(f"{experiment_id}: planned case must not claim a run or results")
            continue
        if not item["results"] or any(not item.get(key) for key in RESULT_FIELDS):
            raise ValueError(f"{experiment_id}: measured case needs results, command, and evidence links")
        for key in ("report_path", "manifest_path", "evidence_path"):
            value = item[key]
            if not isinstance(value, str) or Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError(f"{experiment_id}: {key} must be a repository-relative path")
            path = root / value
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"{experiment_id}: missing {key}: {value}")
        manifest = json.loads((root / item["manifest_path"]).read_text(encoding="utf-8"))
        if manifest.get("schema_version") != "agentic_run_manifest.v2":
            raise ValueError(f"{experiment_id}: unsupported run manifest schema")
        if manifest.get("run_id") != item["run_id"]:
            raise ValueError(f"{experiment_id}: run manifest ID does not match registry")
        if manifest.get("completion_status") in (None, "", "created", "running"):
            raise ValueError(f"{experiment_id}: run manifest is not complete")
        try:
            evidence = json.loads((root / item["evidence_path"]).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{experiment_id}: evidence must be JSON for the numerical ledger") from exc
        if not isinstance(evidence, dict):
            raise ValueError(f"{experiment_id}: evidence must be a JSON object for the numerical ledger")

def metric_label(key: str) -> str:
    return key.replace("_", " ").replace("itl", "inter-token latency").replace("ttft", "TTFT")


def metric_unit(key: str) -> str:
    key = key.lower()
    if key.endswith("_ms") or "duration_ms" in key:
        return "ms"
    if key.endswith("_pct") or "share" in key or "duty_cycle" in key:
        return "%"
    if "tokens" in key:
        return "tokens"
    if "loads" in key or "events" in key or "replays" in key or "trials" in key or "count" in key:
        return "count"
    if "updates_per_second" in key:
        return "updates/s"
    return "value"


def metric_statistic(key: str) -> str:
    key = key.lower()
    for marker, label in (("p95", "p95"), ("median", "median"), ("mean", "mean"), ("max", "maximum")):
        if marker in key:
            return label
    if key.startswith("valid_"):
        return "accepted count"
    return "reported value"


def add_metric(
    rows: list[dict[str, object]],
    *,
    condition: str,
    metric: str,
    value: int | float,
    sample_count: int | float | None,
    provenance: str,
) -> None:
    rows.append(
        {
            "condition": condition,
            "metric": metric_label(metric),
            "value": value,
            "unit": metric_unit(metric),
            "statistic": metric_statistic(metric),
            "sample_count": sample_count,
            "provenance": provenance,
        }
    )


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def condition_measurements(summary: dict[str, object]) -> list[dict[str, object]]:
    """Extract every numeric condition field plus region metrics from known summaries."""

    rows: list[dict[str, object]] = []
    for condition in summary.get("conditions", []):
        if not isinstance(condition, dict):
            continue
        label = str(condition.get("label") or condition.get("condition") or "Condition")
        sample_count = condition.get("replays") or condition.get("trials")
        for key, value in condition.items():
            if is_number(value):
                add_metric(rows, condition=label, metric=key, value=value, sample_count=sample_count, provenance=f"conditions.{key}")
        regions = condition.get("regions")
        if isinstance(regions, dict):
            for region_name, region in regions.items():
                if not isinstance(region, dict):
                    continue
                region_count = region.get("interval_count")
                for key, value in region.items():
                    if is_number(value):
                        add_metric(
                            rows,
                            condition=f"{label} / {region_name}",
                            metric=key,
                            value=value,
                            sample_count=region_count if is_number(region_count) else sample_count,
                            provenance=f"conditions.regions.{region_name}.{key}",
                        )
    for paired_key in ("paired_decode_changes", "paired_decode_deltas"):
        paired = summary.get(paired_key)
        if not isinstance(paired, dict):
            continue
        for condition, values in paired.items():
            if not isinstance(values, dict):
                continue
            sample_count = values.get("paired_samples")
            for key, value in values.items():
                if is_number(value):
                    add_metric(rows, condition=str(condition), metric=key, value=value, sample_count=sample_count, provenance=f"{paired_key}.{condition}.{key}")
                elif key == "changes_ms" and isinstance(value, list):
                    for sample_index, sample in enumerate(value, start=1):
                        if is_number(sample):
                            add_metric(rows, condition=f"{condition} / paired sample {sample_index}", metric="decode_change_ms", value=sample, sample_count=1, provenance=f"{paired_key}.{condition}.changes_ms[{sample_index - 1}]")
    return rows


def natural_comparison_measurements(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level in summary.get("levels", []):
        if not isinstance(level, dict):
            continue
        sessions = level.get("session_count")
        prefix = f"{sessions} sessions"
        total_replays = level.get("all_replays", {}).get("replays") if isinstance(level.get("all_replays"), dict) else None
        for key in ("trials", "native_reload_events", "cross_session_overlap_pct"):
            value = level.get(key)
            if is_number(value):
                add_metric(rows, condition=prefix, metric=key, value=value, sample_count=total_replays if is_number(total_replays) else None, provenance=f"levels.{key}")
        for group_key, group_label in (
            ("all_replays", "all replays"),
            ("no_cross_session_overlap", "no cross-session overlap"),
            ("cross_session_overlap", "cross-session overlap"),
        ):
            group = level.get(group_key)
            if not isinstance(group, dict):
                continue
            sample_count = group.get("replays")
            for key, value in group.items():
                if is_number(value):
                    add_metric(rows, condition=f"{prefix} / {group_label}", metric=key, value=value, sample_count=sample_count if is_number(sample_count) else None, provenance=f"levels.{group_key}.{key}")
    return rows


def natural_observation_measurements(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    replay_count = summary.get("replay_count")
    for key in ("replay_count", "native_reload_events"):
        value = summary.get(key)
        if is_number(value):
            add_metric(rows, condition="Run total", metric=key, value=value, sample_count=replay_count if is_number(replay_count) else None, provenance=key)
    buckets = summary.get("pressure_buckets")
    if isinstance(buckets, dict):
        for key, value in buckets.items():
            if is_number(value):
                add_metric(rows, condition="Run total", metric=f"{key}_replays", value=value, sample_count=replay_count if is_number(replay_count) else None, provenance=f"pressure_buckets.{key}")
    for observation in summary.get("observations", []):
        if not isinstance(observation, dict):
            continue
        request_id = str(observation.get("request_id") or "replay")
        for key, value in observation.items():
            if is_number(value) and not key.endswith("_ns"):
                add_metric(rows, condition=request_id, metric=key, value=value, sample_count=1, provenance=f"observations.{key}")
    for load_index, native_load in enumerate(summary.get("native_loads", []), start=1):
        if not isinstance(native_load, dict):
            continue
        label = f"native reload {load_index}"
        for key, value in native_load.items():
            if is_number(value) and not key.endswith("_ns"):
                add_metric(rows, condition=label, metric=key, value=value, sample_count=1, provenance=f"native_loads[{load_index - 1}].{key}")
    return rows


def movement_measurements(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for section_name in ("comparison", "lateness_comparison", "interference_loads", "software_visible_h2d"):
        section = summary.get(section_name)
        if not isinstance(section, dict):
            continue
        sample_count = section.get("sample_count") or section.get("trials")
        for key, value in section.items():
            if is_number(value):
                add_metric(rows, condition=section_name.replace("_", " "), metric=key, value=value, sample_count=sample_count if is_number(sample_count) else None, provenance=f"{section_name}.{key}")
    return rows


def measurements_for(summary: dict[str, object]) -> list[dict[str, object]]:
    schema = str(summary.get("schema_version", ""))
    if schema == "natural_kv_pressure_comparison.v1":
        return natural_comparison_measurements(summary)
    if schema == "natural_multi_agent_kv_pressure.v1":
        return natural_observation_measurements(summary)
    if schema == "hardware_kv_movement_report.v1":
        return movement_measurements(summary)
    return condition_measurements(summary)


def run_date(run_id: str) -> str:
    match = re.search(r"_(\d{8})_\d{6}$", run_id)
    if not match:
        return "not recorded"
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def recorded_model(manifest: dict[str, object], evidence: dict[str, object]) -> str:
    model = manifest.get("model") or evidence.get("model")
    if isinstance(model, str) and model:
        return model
    if isinstance(model, dict):
        for key in ("name", "model", "model_id"):
            value = model.get(key)
            if isinstance(value, str) and value:
                return value
    return "not recorded"


def comparison_basis(experiment_id: str) -> str:
    if experiment_id in {"natural_multi_agent_kv_pressure", "natural_kv_pressure_performance_comparison"}:
        return "Natural workload: no overlap versus reload overlap"
    return "Controlled workload: control versus reload pressure"


def render_results_ledger(registry: dict[str, object], root: Path) -> str:
    body: list[str] = []
    for item in registry["experiments"]:
        if item["status"] == "planned":
            continue
        manifest = json.loads((root / str(item["manifest_path"])).read_text(encoding="utf-8"))
        evidence = json.loads((root / str(item["evidence_path"])).read_text(encoding="utf-8"))
        for row in measurements_for(evidence):
            source = (
                f'<a href="{escape(item["evidence_path"])}">JSON</a> · '
                f'<a href="{escape(item["report_path"])}">report</a> · '
                f'<code>{escape(row["provenance"])}</code>'
            )
            body.append(
                "<tr>"
                f"<td>{escape(run_date(str(item['run_id'])))}</td>"
                f"<td><code>{escape(item['run_id'])}</code></td>"
                f"<td>{escape(item['name'])}</td><td>{escape(row['condition'])}</td>"
                f"<td>{escape(row['metric'])}</td><td>{escape(row['value'])}</td>"
                f"<td>{escape(row['unit'])}</td><td>{escape(row['statistic'])}</td>"
                f"<td>{escape(row['sample_count'] if row['sample_count'] is not None else 'n/a')}</td>"
                f"<td>{escape(item['platform'])}</td><td>{escape(recorded_model(manifest, evidence))}</td>"
                f"<td>{escape(comparison_basis(str(item['id'])))}</td><td>{source}</td></tr>"
            )
    return f"""<section class="results-ledger">
<h2>Hardware Experiment Results Ledger</h2>
<p class="intro">One historical table for every completed hardware experiment. Each row is a numerical value from a run's evidence summary; values are not hand-selected. The source column links to the full evidence and run report.</p>
<div class="table-scroll"><table><thead><tr><th>Run date</th><th>Run ID</th><th>Experiment</th><th>Condition / subgroup</th><th>Metric</th><th>Value</th><th>Unit</th><th>Statistic</th><th>N</th><th>Platform</th><th>Model</th><th>Comparison basis</th><th>Source</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
</section>"""


def build(registry: dict[str, object], root: Path = ROOT) -> str:
    results_ledger = render_results_ledger(registry, root)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(registry['title'])}</title><style>
:root {{ font-family: system-ui, sans-serif; color: #182331; background: #f6f8fa; }}
body {{ margin: 0; line-height: 1.5; }} main {{ max-width: 1100px; margin: auto; padding: 32px 24px 72px; }}
h1 {{ font-size: 30px; margin: 0 0 8px; }} h2 {{ font-size: 21px; margin: 0 0 4px; }}
p {{ margin: 6px 0 12px; }} .intro {{ color: #526273; max-width: 850px; }} nav {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 22px 0 32px; }}
a {{ color: #08648b; }} .index, .table-scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }} th,td {{ text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid #dfe5eb; }}
.results-ledger {{ margin: 28px 0 34px; padding: 18px; background: #fff; border: 1px solid #dfe5eb; }} .results-ledger h2 {{ margin-bottom: 2px; }} .results-ledger table {{ font-size: 13px; }} .results-ledger th {{ white-space: nowrap; background: #edf4f7; }} .results-ledger td:nth-child(1), .results-ledger td:nth-child(6), .results-ledger td:nth-child(9) {{ font-variant-numeric: tabular-nums; white-space: nowrap; }} .results-ledger td:nth-child(2), .results-ledger td:nth-child(3), .results-ledger td:nth-child(10), .results-ledger td:nth-child(11), .results-ledger td:nth-child(12) {{ min-width: 170px; }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
@media (max-width: 640px) {{ main {{ padding: 22px 16px 48px; }} }}
</style></head><body><main>
<h1>{escape(registry['title'])}</h1>
<p class="intro">A separate research lane measuring GPU and memory bottlenecks before proposing hardware changes. Planned cases are not results. This page is generated from a structured registry.</p>
<nav aria-label="Project lanes"><a href="CONTROLLER_EXPERIMENTS.html">Controller experiments</a><a href="HINT_BENCHMARK_RUNBOOK.html">Hint benchmark runbook</a><a href="ARCHITECTURE.md">Architecture</a></nav>
{results_ledger}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/hardware_experiment_registry.json")
    parser.add_argument("--out", type=Path, default=ROOT / "HARDWARE_EXPERIMENTS.html")
    parser.add_argument("--check", action="store_true", help="Fail if the generated HTML is stale")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_registry(registry)
    rendered = build(registry, ROOT)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != rendered:
            parser.error(f"{args.out} is stale; rebuild it from the registry")
    else:
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
