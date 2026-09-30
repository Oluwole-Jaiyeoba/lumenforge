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
            raise ValueError(f"{experiment_id}: evidence must be JSON for the results ledger") from exc
        if not isinstance(evidence, dict):
            raise ValueError(f"{experiment_id}: evidence must be a JSON object for the results ledger")
        manager_summary = item.get("manager_summary")
        if not isinstance(manager_summary, list) or not manager_summary:
            raise ValueError(f"{experiment_id}: completed experiment needs a manager_summary")
        for row in manager_summary:
            if not isinstance(row, dict) or any(not row.get(key) for key in ("condition", "study_type", "collision_intensity", "outcome", "unit", "sample_count_path")):
                raise ValueError(f"{experiment_id}: invalid manager_summary row")
            has_comparison = bool(row.get("baseline_path") and row.get("collision_path"))
            has_observation = bool(row.get("observed_path"))
            if has_comparison == has_observation:
                raise ValueError(f"{experiment_id}: summary row needs either a comparison or an observation")
            for key in ("sample_count_path", "baseline_path", "collision_path", "observed_path"):
                path = row.get(key)
                if path and not isinstance(value_at_path(evidence, str(path)), (int, float)):
                    raise ValueError(f"{experiment_id}: {key} must resolve to a number")

def value_at_path(document: object, path: str) -> object:
    """Read a simple dotted evidence path, including list indexes such as conditions[1].ttft_ms."""

    value = document
    for part in path.split("."):
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?", part)
        if not match:
            raise ValueError(f"invalid evidence path: {path}")
        key, index = match.groups()
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"evidence path not found: {path}")
        value = value[key]
        if index is not None:
            if not isinstance(value, list) or int(index) >= len(value):
                raise ValueError(f"evidence path not found: {path}")
            value = value[int(index)]
    return value


def format_value(value: float | int, unit: str) -> str:
    if unit == "ms":
        return f"{value / 1000:.1f} s" if abs(value) >= 1000 else f"{value:.1f} ms"
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "count":
        return f"{value:,.0f}"
    return f"{value:.1f} {unit}"


def format_impact(difference: float, percent: float | None, unit: str) -> str:
    absolute = format_value(difference, unit)
    return absolute if percent is None else f"{absolute} ({percent:+.1f}%)"


def run_date(run_id: str) -> str:
    match = re.search(r"_(\d{8})_\d{6}$", run_id)
    if not match:
        return "not recorded"
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def run_sort_key(run_id: str) -> str:
    match = re.search(r"_(\d{8}_\d{6})$", run_id)
    return match.group(1) if match else ""


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


def render_results_ledger(registry: dict[str, object], root: Path) -> str:
    body: list[str] = []
    completed = sorted(
        (item for item in registry["experiments"] if item["status"] != "planned"),
        key=lambda item: run_sort_key(str(item["run_id"])),
        reverse=True,
    )
    for item in completed:
        manifest = json.loads((root / str(item["manifest_path"])).read_text(encoding="utf-8"))
        evidence = json.loads((root / str(item["evidence_path"])).read_text(encoding="utf-8"))
        for row in item["manager_summary"]:
            sample_count = value_at_path(evidence, row["sample_count_path"])
            baseline_path = row.get("baseline_path")
            collision_path = row.get("collision_path")
            observed_path = row.get("observed_path")
            if baseline_path and collision_path:
                baseline = value_at_path(evidence, baseline_path)
                collision = value_at_path(evidence, collision_path)
                difference = collision - baseline
                percent = difference / baseline * 100 if baseline else None
                baseline_text = format_value(baseline, row["unit"])
                collision_text = format_value(collision, row["unit"])
                impact_text = format_impact(difference, percent, row["unit"])
                impact_class = "impact-slower" if difference > 0 else "impact-faster" if difference < 0 else "impact-neutral"
                source_path = collision_path
            else:
                observed = value_at_path(evidence, observed_path)
                baseline_text = "n/a"
                collision_text = format_value(observed, row["unit"])
                impact_text = "n/a: occurrence-only run"
                impact_class = "impact-neutral"
                source_path = observed_path
            source = (
                f'<a href="{escape(item["evidence_path"])}">JSON</a> · '
                f'<a href="{escape(item["report_path"])}">report</a> · '
                f'<code>{escape(source_path)}</code>'
            )
            body.append(
                "<tr>"
                f"<td>{escape(run_date(str(item['run_id'])))}</td>"
                f"<td>{escape(item['name'])}</td><td>{escape(row['condition'])}</td>"
                f"<td>{escape(row['study_type'])}</td><td>{escape(row['collision_intensity'])}</td>"
                f"<td>{escape(row['outcome'])}</td><td>{escape(baseline_text)}</td>"
                f"<td>{escape(collision_text)}</td><td class=\"{impact_class}\">{escape(impact_text)}</td>"
                f"<td>{escape(format_value(sample_count, 'count'))}</td>"
                f"<td>{escape(item['platform'])}<br><span class=\"model\">{escape(recorded_model(manifest, evidence))}</span></td>"
                f"<td>{source}</td></tr>"
            )
    return f"""<section class="results-ledger">
<h2>Collision Impact Summary</h2>
<p class="intro">One row per meaningful collision condition. Controlled rows compare matched no-collision and collision runs. Observational rows show what occurred naturally and are not causal comparisons.</p>
<div class="table-scroll"><table><thead><tr><th>Run date</th><th>Experiment</th><th>Condition</th><th>Study type</th><th>Collision intensity</th><th>Main outcome</th><th>No collision</th><th>With collision</th><th>Impact</th><th>N</th><th>Runtime</th><th>Evidence</th></tr></thead><tbody>{''.join(body)}</tbody></table></div>
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
.results-ledger {{ margin: 28px 0 34px; padding: 20px; background: #fff; border: 1px solid #d7e2e8; box-shadow: 0 8px 20px rgba(29, 77, 94, .08); }} .results-ledger h2 {{ margin-bottom: 2px; color: #0d5167; }} .results-ledger table {{ font-size: 13px; }} .results-ledger th {{ white-space: nowrap; background: #dff3f0; color: #114957; }} .results-ledger tr:nth-child(even) {{ background: #f7fbfc; }} .results-ledger tr:hover {{ background: #fff8dc; }} .results-ledger td:nth-child(1), .results-ledger td:nth-child(7), .results-ledger td:nth-child(8), .results-ledger td:nth-child(9), .results-ledger td:nth-child(10) {{ font-variant-numeric: tabular-nums; white-space: nowrap; }} .results-ledger td:nth-child(2) {{ min-width: 170px; font-weight: 650; color: #243d4a; }} .results-ledger td:nth-child(5), .results-ledger td:nth-child(11) {{ min-width: 150px; }} .results-ledger .model {{ color: #58717c; }} .impact-slower {{ color: #a83b1d; font-weight: 700; background: #fff0e8; }} .impact-faster {{ color: #176b54; font-weight: 700; background: #e8f8ef; }} .impact-neutral {{ color: #566875; background: #eef3f5; }}
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
