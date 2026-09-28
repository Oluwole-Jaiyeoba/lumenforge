"""The public hardware page must match a truthful, navigable registry."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from build_hardware_experiments import build, validate_registry  # noqa: E402


def registry() -> dict[str, object]:
    return json.loads((ROOT / "configs/hardware_experiment_registry.json").read_text(encoding="utf-8"))


def test_page_is_current_and_cases_are_planned() -> None:
    data = registry()
    validate_registry(data)
    assert all(item["status"] == "planned" for item in data["experiments"])
    assert (ROOT / "HARDWARE_EXPERIMENTS.html").read_text(encoding="utf-8") == build(data)


def test_top_level_navigation_and_handoff_are_wired() -> None:
    page = (ROOT / "HARDWARE_EXPERIMENTS.html").read_text(encoding="utf-8")
    assert 'href="CONTROLLER_EXPERIMENTS.html"' in page
    assert 'href="HINT_BENCHMARK_RUNBOOK.html"' in page
    for path in ("README.md", "HANDOFF.md", "ARCHITECTURE.md", "docs/index.md"):
        assert "HARDWARE_EXPERIMENTS.html" in (ROOT / path).read_text(encoding="utf-8")


def test_planned_case_cannot_claim_results() -> None:
    data = deepcopy(registry())
    data["experiments"][0]["results"] = {"replay_ttft_ms": "1"}
    with pytest.raises(ValueError, match="planned case"):
        validate_registry(data)


def test_validated_case_requires_real_evidence_files() -> None:
    data = deepcopy(registry())
    data["experiments"][0].update(
        status="validated",
        run_id="run_1",
        command="run-probe",
        results={"replay_ttft_ms": "1"},
        report_path="docs/reports/missing.html",
        manifest_path="docs/reports/missing.json",
        evidence_path="docs/reports/missing.csv",
    )
    with pytest.raises(ValueError, match="missing report_path"):
        validate_registry(data)


def test_validated_case_rejects_mismatched_run_manifest(tmp_path: Path) -> None:
    data = deepcopy(registry())
    for name, contents in (
        ("report.html", "<h1>Report</h1>"),
        ("manifest.json", '{"schema_version":"agentic_run_manifest.v2","run_id":"wrong","completion_status":"complete"}'),
        ("evidence.csv", "sample_id,value_ms\n1,10\n"),
    ):
        (tmp_path / name).write_text(contents, encoding="utf-8")
    data["experiments"][0].update(
        status="validated",
        run_id="run_1",
        command="run-probe",
        results={"replay_ttft_ms": "10"},
        report_path="report.html",
        manifest_path="manifest.json",
        evidence_path="evidence.csv",
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_registry(data, root=tmp_path)
