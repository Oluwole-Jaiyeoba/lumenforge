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

    reference_results = registry.get("reference_results", [])
    if not isinstance(reference_results, list):
        raise ValueError("reference_results must be a list")
    required_reference_fields = (
        "experiment_id",
        "condition",
        "reloads",
        "copy_share",
        "total_decode",
        "change",
        "change_percent",
        "ttft",
        "trials",
        "valid",
    )
    for row in reference_results:
        if not isinstance(row, dict) or any(not row.get(field) for field in required_reference_fields):
            raise ValueError("reference result is missing a required field")
        if row["experiment_id"] not in ids:
            raise ValueError(f"reference result names an unknown experiment: {row['experiment_id']}")


def render_reference_results(rows: list[dict[str, object]], experiments: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    names = {str(item["id"]): str(item["name"]) for item in experiments}
    body = "".join(
        "<tr>"
        f'<td><a href="#{escape(row["experiment_id"])}">{escape(names[str(row["experiment_id"])])}</a></td>'
        f"<td>{escape(row['condition'])}</td><td>{escape(row['reloads'])}</td>"
        f"<td>{escape(row['copy_share'])}</td><td>{escape(row['total_decode'])}</td>"
        f"<td>{escape(row['change'])}</td><td>{escape(row['change_percent'])}</td>"
        f"<td>{escape(row['ttft'])}</td><td>{escape(row['trials'])}</td><td>{escape(row['valid'])}</td>"
        "</tr>"
        for row in rows
    )
    return f"""<section class=\"reference-results\">
<h2>Reference Results</h2>
<p class=\"intro\">Completed, evidence-gated measurements. Changes compare each pressure condition with its paired control.</p>
<div class=\"table-scroll\"><table><thead><tr><th>Experiment</th><th>Condition</th><th>Reloads</th><th>Copy share</th><th>Total decode</th><th>Change vs control</th><th>Change %</th><th>TTFT</th><th>Trials</th><th>Valid</th></tr></thead><tbody>{body}</tbody></table></div>
</section>"""


def render_experiment(item: dict[str, object]) -> str:
    status = str(item["status"])
    measurements = "".join(f"<li>{escape(value)}</li>" for value in item["measurements"])
    if status == "planned":
        result_html = "<p class=\"pending\">No GPU measurements yet. A command and result will appear only after the run is implemented and evidence is reviewed.</p>"
    else:
        rows = "".join(
            f"<tr><th>{escape(key.replace('_', ' '))}</th><td>{escape(value)}</td></tr>"
            for key, value in item["results"].items()
        )
        links = "".join(
            f'<a href="{escape(item[key])}">{label}</a>'
            for key, label in (
                ("report_path", "Report"),
                ("manifest_path", "Run manifest"),
                ("evidence_path", "Evidence"),
            )
        )
        result_html = (
            f'<p><strong>Run:</strong> <code>{escape(item["run_id"])}</code></p>'
            f'<pre><code>{escape(item["command"])}</code></pre>'
            f'<table><tbody>{rows}</tbody></table><p class="links">{links}</p>'
        )
    standard_fields = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(item[key])}</dd>"
        for key, label in (
            ("production_analogy", "Production analogy"),
            ("what_proves", "What this establishes"),
            ("limitations", "Limit"),
        )
        if item.get(key)
    )
    return f"""<section class="experiment" id="{escape(item['id'])}">
<div class="heading"><div><h2>{escape(item['name'])}</h2><p>{escape(item['question'])}</p></div><span class="status {escape(status)}">{escape(STATUS_LABELS[status])}</span></div>
<dl><dt>Platform</dt><dd>{escape(item['platform'])}</dd><dt>Backend</dt><dd>{escape(item['backend'])}</dd><dt>Control</dt><dd>{escape(item['control'])}</dd><dt>Interference</dt><dd>{escape(item['interference'])}</dd><dt>Held constant</dt><dd>{escape(item['held_constant'])}</dd>{standard_fields}</dl>
<h3>Measurements</h3><ul>{measurements}</ul><p class="evidence"><strong>Evidence rule:</strong> {escape(item['evidence_rule'])}</p>
<h3>Result</h3>{result_html}</section>"""


def build(registry: dict[str, object]) -> str:
    rows = "".join(
        f'<tr><td><a href="#{escape(item["id"])}">{escape(item["name"])}</a></td>'
        f'<td>{escape(item["question"])}</td><td>{escape(STATUS_LABELS[item["status"]])}</td></tr>'
        for item in registry["experiments"]
    )
    experiments = "\n".join(render_experiment(item) for item in registry["experiments"])
    reference_results = render_reference_results(registry.get("reference_results", []), registry["experiments"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(registry['title'])}</title><style>
:root {{ font-family: system-ui, sans-serif; color: #182331; background: #f6f8fa; }}
body {{ margin: 0; line-height: 1.5; }} main {{ max-width: 1100px; margin: auto; padding: 32px 24px 72px; }}
h1 {{ font-size: 30px; margin: 0 0 8px; }} h2 {{ font-size: 21px; margin: 0 0 4px; }} h3 {{ font-size: 15px; margin: 20px 0 4px; }}
p {{ margin: 6px 0 12px; }} .intro {{ color: #526273; max-width: 850px; }} nav {{ display: flex; flex-wrap: wrap; gap: 18px; margin: 22px 0 32px; }}
a {{ color: #08648b; }} .index, .table-scroll {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }} th,td {{ text-align: left; vertical-align: top; padding: 10px 12px; border-bottom: 1px solid #dfe5eb; }}
.index th:nth-child(1) {{ min-width: 180px; }} .index th:nth-child(2) {{ min-width: 300px; }} .index th:nth-child(3) {{ min-width: 170px; }}
.reference-results {{ margin: 28px 0 34px; padding: 18px; background: #fff; border: 1px solid #dfe5eb; }} .reference-results h2 {{ margin-bottom: 2px; }} .reference-results th {{ white-space: nowrap; background: #edf4f7; }} .reference-results td {{ white-space: nowrap; }}
.experiment {{ border-top: 2px solid #9ab4c4; margin-top: 36px; padding-top: 22px; }} .heading {{ display: flex; align-items: start; justify-content: space-between; gap: 16px; }}
.status {{ white-space: nowrap; font-size: 12px; font-weight: 650; padding: 5px 8px; border: 1px solid #b7c6d2; }} .validated {{ border-color: #478675; color: #14634d; }}
.measured_pending_review {{ border-color: #a67937; color: #79500d; }} dl {{ display: grid; grid-template-columns: 125px minmax(0, 1fr); margin: 18px 0; }}
dt,dd {{ margin: 0; padding: 9px 0; border-bottom: 1px solid #dfe5eb; }} dt {{ font-weight: 650; color: #465666; }} ul {{ margin: 8px 0 16px; padding-left: 22px; }}
.evidence {{ border-left: 3px solid #4d8998; padding-left: 12px; }} .pending {{ color: #5e6872; }} .links {{ display: flex; gap: 18px; flex-wrap: wrap; }}
pre {{ overflow-x: auto; background: #1b2933; color: #f3f6f8; padding: 14px; }} code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
@media (max-width: 640px) {{ main {{ padding: 22px 16px 48px; }} .heading {{ display: block; }} .status {{ display: inline-block; margin: 8px 0; }} dl {{ grid-template-columns: 100px minmax(0, 1fr); }} }}
</style></head><body><main>
<h1>{escape(registry['title'])}</h1>
<p class="intro">A separate research lane measuring GPU and memory bottlenecks before proposing hardware changes. Planned cases are not results. This page is generated from a structured registry.</p>
<nav aria-label="Project lanes"><a href="CONTROLLER_EXPERIMENTS.html">Controller experiments</a><a href="HINT_BENCHMARK_RUNBOOK.html">Hint benchmark runbook</a><a href="ARCHITECTURE.md">Architecture</a></nav>
{reference_results}
<div class="index"><table><thead><tr><th>Experiment</th><th>Question</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></div>
{experiments}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/hardware_experiment_registry.json")
    parser.add_argument("--out", type=Path, default=ROOT / "HARDWARE_EXPERIMENTS.html")
    parser.add_argument("--check", action="store_true", help="Fail if the generated HTML is stale")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    validate_registry(registry)
    rendered = build(registry)
    if args.check:
        if not args.out.is_file() or args.out.read_text(encoding="utf-8") != rendered:
            parser.error(f"{args.out} is stale; rebuild it from the registry")
    else:
        args.out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
