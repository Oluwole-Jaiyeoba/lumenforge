#!/usr/bin/env python3
"""Build the human-readable milestone registry from its JSON source."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


STATUS_LABELS = {
    "validated_reference": "Validated reference",
    "completed_pending_review": "Completed - review pending",
    "invalid_for_comparison": "Invalid for comparison",
    "planned": "Planned",
}


def text(value: object) -> str:
    return html.escape(str(value))


def link_or_text(value: str) -> str:
    if value.endswith(".html"):
        return f'<a href="{text(value)}">Open report</a>'
    return text(value)


def render_experiment(item: dict[str, object]) -> str:
    status = str(item["status"])
    parameters = item.get("parameters", {})
    results = item.get("results", {})
    parameter_rows = "".join(
        f"<tr><th>{text(key.replace('_', ' '))}</th><td>{text(value)}</td></tr>"
        for key, value in dict(parameters).items()
    )
    result_rows = "".join(
        f"<tr><th>{text(key.replace('_', ' '))}</th><td>{text(value)}</td></tr>"
        for key, value in dict(results).items()
    )
    spec = item.get("spec_path")
    spec_html = f'<p><strong>Reusable spec:</strong> <code>{text(spec)}</code></p>' if spec else ""
    return f"""
    <article class="experiment" id="{text(item['id'])}">
      <header>
        <div><p class="eyebrow">{text(item['id'])}</p><h2>{text(item['scenario'])}</h2></div>
        <span class="status {text(status)}">{text(STATUS_LABELS.get(status, status))}</span>
      </header>
      <p class="question">{text(item['question'])}</p>
      <div class="contract-grid">
        <section><h3>Controller scope</h3><p>{text(item['controller_capability'])}</p></section>
        <section><h3>Priority contract</h3><p>{text(item['priority_contract'])}</p></section>
        <section><h3>Compared modes</h3><p><code>{text(item['comparison'])}</code></p></section>
        <section><h3>Workload</h3><p>{text(item['workload'])}</p></section>
      </div>
      <details open><summary>Reproduction contract</summary>
        {spec_html}
        <table><tbody>{parameter_rows}</tbody></table>
        <p><strong>Run label:</strong> <code>{text(item['run_label'])}</code></p>
        <pre><code>{text(item['reproduction_command'])}</code></pre>
      </details>
      <details open><summary>Observed result</summary>
        <table><tbody>{result_rows}</tbody></table>
        <p><strong>Artifact:</strong> {link_or_text(str(item['report_path']))}</p>
      </details>
    </article>"""


def build(registry: dict[str, object]) -> str:
    rows = "".join(
        f"<tr><td><a href=\"#{text(item['id'])}\">{text(item['id'])}</a></td>"
        f"<td>{text(item['scenario'])}</td>"
        f"<td><span class=\"status {text(item['status'])}\">{text(STATUS_LABELS.get(str(item['status']), item['status']))}</span></td>"
        f"<td><code>{text(item['comparison'])}</code></td></tr>"
        for item in registry["experiments"]
    )
    cards = "\n".join(render_experiment(item) for item in registry["experiments"])
    rules = "".join(f"<li>{text(rule)}</li>" for rule in registry["comparison_safety_rules"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{text(registry['title'])}</title><style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #172033; background: #f5f7fb; }}
body {{ margin: 0; line-height: 1.5; }} main {{ max-width: 1240px; margin: 0 auto; padding: 42px 24px 72px; }}
h1,h2,h3,p {{ margin-top: 0; }} h1 {{ font-size: 32px; margin-bottom: 8px; }} h2 {{ font-size: 22px; margin-bottom: 4px; }} h3 {{ font-size: 15px; margin-bottom: 6px; }}
.lede {{ color: #52627b; max-width: 880px; }} .notice, .experiment, .registry {{ background: #fff; border: 1px solid #d8e0ec; border-radius: 8px; padding: 20px; margin-top: 20px; }}
.notice {{ border-left: 4px solid #16856b; }} .registry {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }} th,td {{ padding: 10px; border-bottom: 1px solid #e5eaf2; text-align: left; vertical-align: top; }} th {{ width: 28%; color: #4c5c74; font-weight: 600; }}
.registry th {{ white-space: nowrap; }} .experiment header {{ display: flex; gap: 16px; justify-content: space-between; align-items: start; }} .eyebrow {{ color: #62738d; font: 600 12px ui-monospace, monospace; margin-bottom: 5px; }}
.question {{ font-size: 17px; }} .contract-grid {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }} .contract-grid section {{ background:#f7f9fd; border:1px solid #e4eaf3; padding:14px; border-radius:6px; }}
.contract-grid p {{ margin-bottom:0; }} .status {{ display:inline-block; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:700; white-space:nowrap; }} .validated_reference {{ background:#d9f4e7; color:#0b6545; }} .completed_pending_review {{ background:#fff1c8; color:#795400; }} .invalid_for_comparison {{ background:#fee0e0; color:#9c2633; }} .planned {{ background:#e5ebf5; color:#3e516d; }}
details {{ margin-top: 15px; border-top: 1px solid #e5eaf2; padding-top: 14px; }} summary {{ cursor:pointer; font-weight:700; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#172033; color:#eef4ff; padding:14px; border-radius:6px; }} code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:.92em; }} a {{ color:#075f9d; }}
@media (max-width: 720px) {{ main {{ padding:28px 14px; }} .contract-grid {{ grid-template-columns:1fr; }} .experiment header {{ display:block; }} .status {{ margin-top:8px; }} }}
</style></head><body><main>
<h1>{text(registry['title'])}</h1><p class="lede">A reproducible record of controller milestones. The JSON registry is the machine-readable source of truth; this page is generated from it.</p>
<section class="notice"><h2>Comparison Safety</h2><ul>{rules}</ul></section>
<section class="registry"><h2>Milestone Index</h2><table><thead><tr><th>ID</th><th>Scenario</th><th>Status</th><th>Comparison</th></tr></thead><tbody>{rows}</tbody></table></section>
{cards}
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("configs/experiment_registry.json"))
    parser.add_argument("--out", type=Path, default=Path("MILESTONE_EXPERIMENTS.html"))
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    args.out.write_text(build(registry), encoding="utf-8")


if __name__ == "__main__":
    main()
