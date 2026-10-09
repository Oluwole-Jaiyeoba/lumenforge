"""Validate paired controller reruns and publish portable per-scenario evidence.

This is offline report analysis; it is deliberately outside experiment policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys

from agentic_backends.controller_audit import BACKEND_PRIORITY_FIELD, runtime_issues, runtime_pair_issues


SCHEMA = "agentic_work_audit.controller_pivot.summary.v1"


def json_rows(path: Path):
    if not path.exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def quantile95(values: list[float]) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]


def model_prompt_hash(row: dict) -> str:
    # The top-level hash includes the transport metadata, which differs by mode.
    return ((row.get("harness_controller_signal") or {}).get("cache") or {}).get("conversation_prefix_hash", "")


def fingerprint(starts: list[dict]) -> list[list]:
    return sorted([
        [str(row.get("session_id")), str(row.get("phase")), str(row.get("tool_wait_step")),
         str(model_prompt_hash(row)), str(row.get("max_tokens")), str(row.get("tool_wait_ms"))]
        for row in starts
    ])


def analyze_arm(root: Path, arm: dict, spec: dict) -> dict:
    cases = list((root / "raw/runs/controlled" / arm["label"]).glob("hatcher*"))
    if len(cases) != 1:
        raise ValueError(f"{arm['label']}: expected one case, found {len(cases)}")
    case = cases[0]
    report = root / "raw/reports" / arm["label"]
    trace = list(json_rows(case / "m27_trace.jsonl"))
    starts = [r for r in trace if r.get("event") == "m27.request.start" and r.get("harness") == "hatcher"]
    ends = [r for r in trace if r.get("event") == "m27.request.end" and r.get("harness") == "hatcher"]
    issues = []
    gate = json.loads((case / "live_sentinel_report.json").read_text())
    if gate.get("valid") is not True or gate.get("experiment_allowed") is not True:
        issues.append("Live instrumentation preflight did not pass")
    for label, rows in (("starts", starts), ("ends", ends)):
        if len(rows) != spec["expected_requests"]:
            issues.append(f"{len(rows)} request {label}, expected {spec['expected_requests']}")
    if len({r.get("request_id") or r.get("label") for r in starts}) != len(starts):
        issues.append("Duplicate request identity")
    for row in starts:
        signal = row.get("harness_controller_signal") or {}
        if (signal.get("phase") or {}).get("work_class") != "peer" or (signal.get("scheduling") or {}).get("urgency") != "normal":
            issues.append("Unequal frontend importance")
        if row.get("priority_intent") or row.get("harness_input_priority_signal"):
            issues.append("Frontend priority input present")
        if arm["mode"] in ("no_prefetch", "controller_proactive_kv_management") and row.get(BACKEND_PRIORITY_FIELD) not in (None, ""):
            issues.append("Unexpected queue/cache rank")
        if not model_prompt_hash(row) or row.get("max_tokens") is None:
            issues.append("Missing workload identity fields")
    raw = csv_rows(report / "global_kv_readiness_by_mode.csv")
    replays = [r for r in raw if r.get("phase") in ("replay", "pressure_filler") and r.get("harness") == "hatcher"]
    if len(replays) != spec["expected_replays"]:
        issues.append(f"{len(replays)} replay rows, expected {spec['expected_replays']}")
    timing_keys = ("ttft_ms", "first_token_lateness_ms", "due_to_request_start_ms")
    for row in replays:
        if any(row.get(key) in (None, "") for key in timing_keys) or row.get("error"):
            issues.append("Missing replay timing or failed replay")
    values = {key: [float(r[key]) for r in replays if r.get(key) not in (None, "")] for key in timing_keys}
    workloads = [r for r in trace if r.get("event") == "m27.workload_end"]
    if len(workloads) != 1 or not workloads[0].get("workload_duration_ms"):
        issues.append("Missing whole-workload clock")
    all_ttft = [float(r["ttft_ms"]) for r in ends if r.get("ttft_ms") not in (None, "") and r.get("first_token_observed")]
    if len(all_ttft) != spec["expected_requests"]:
        issues.append("Incomplete all-request TTFT coverage")
    if any(r.get("error") or r.get("status") != 200 for r in ends):
        issues.append("At least one model call failed")
    server_info = json.loads((case / "server_info.json").read_text())
    args = server_info.get("server_args", server_info)
    issues.extend(runtime_issues(args, queue_ranking=arm["mode"] == "controller_ready_time_gpu_backfill",
                                 retention_ranking=arm["mode"] == "controller_value_aware_eviction"))
    events = Counter(r.get("event", "") for r in trace)
    backend_events = Counter()
    backend_categories = Counter()
    workload_starts = [r for r in trace if r.get("event") == "m27.workload_start"]
    window_start = int(workload_starts[0]["ts_ns"]) if workload_starts else 0
    window_end = int(workloads[0]["ts_ns"]) if workloads else 0
    evictions, evicted_tokens = 0, 0
    for row in json_rows(case / "backend_trace.jsonl"):
        if not window_start <= int(row.get("ts_ns", 0)) <= window_end:
            continue
        backend_events[row.get("event", "")] += 1
        backend_categories[str(row.get("category", "")) + ":" + str(row.get("phase", ""))] += 1
        if row.get("method") == "evict_device" and row.get("event", "").endswith(".end"):
            indices = (row.get("kv_context") or {}).get("device_indices") or {}
            count = int(indices.get("index_count") or indices.get("numel") or 0)
            if count > 0:
                evictions += 1
                evicted_tokens += count
    prepared = {}
    for row in trace:
        if row.get("event") == "m27.targeted_kv_prefetch.prepare_prefix_control_result" and not row.get("plan_only"):
            result = row.get("result") or {}
            if result.get("load_id") and result.get("loaded_tokens", 0) > 0:
                prepared[result["load_id"]] = result
    ranked = sum(r.get(BACKEND_PRIORITY_FIELD) not in (None, "") for r in starts
                 if r.get("phase") in ("replay", "pressure_filler"))
    admission = Counter(r.get("decision") for r in trace
                        if r.get("event") == "m27.controller_ready_time_gpu_backfill.decision")
    exposure = {"ranked_replays": ranked, "accepted_prepare_loads": len(prepared),
                "device_eviction_calls": evictions, "device_evicted_token_slots": evicted_tokens,
                "admission_decisions": sum(admission.values()) if admission else None,
                "hold_decisions": admission.get("hold", 0) if admission else None}
    if arm["mode"] == "controller_ready_time_gpu_backfill" and ranked != spec["expected_replays"]:
        issues.append("Not every replay had a controller-derived queue rank")
    if arm["mode"] == "controller_proactive_kv_management" and not prepared:
        issues.append("No actual direct preparation load observed; cannot test preparation benefit")
    if arm["mode"] == "controller_value_aware_eviction" and (not evictions or ranked != spec["expected_replays"]):
        issues.append("Retention exposure missing: require ranked replays and actual GPU evictions")
    metrics = {
        "workload_ms": workloads[0].get("workload_duration_ms") if workloads else None,
        "total_all_ttft_ms": sum(all_ttft), "total_replay_ttft_ms": sum(values["ttft_ms"]),
        "total_replay_lateness_ms": sum(max(0, x) for x in values["first_token_lateness_ms"]),
        "median_replay_ttft_ms": median(values["ttft_ms"]) if values["ttft_ms"] else None,
        "median_replay_delay_ms": median(values["first_token_lateness_ms"]) if values["first_token_lateness_ms"] else None,
        "p95_replay_delay_ms": quantile95(values["first_token_lateness_ms"]) if values["first_token_lateness_ms"] else None,
        "total_presubmission_wait_ms": sum(values["due_to_request_start_ms"]),
        "replays": len(replays), "requests": len(starts),
    }
    return {"trial": arm["trial"], "mode": arm["mode"], "label": arm["label"], "metrics": metrics,
            "issues": sorted(set(issues)), "request_fingerprint": fingerprint(starts),
            "backend_args": args, "event_counts": dict(events), "backend_event_counts": dict(backend_events),
            "backend_category_counts": dict(backend_categories), "case_path": str(case.relative_to(root)),
            "exposure": exposure,
            "report_path": str(report.relative_to(root)),
            "good_admits": None, "bad_admits": None,
            "admission_classification_note": "These studies do not classify short-filler runtime predictions as good/bad admits. Raw controller decisions remain in the gateway trace."}


def pair_comparison(baseline: dict, treatment: dict) -> dict:
    issues = baseline["issues"] + treatment["issues"]
    if baseline["request_fingerprint"] != treatment["request_fingerprint"]:
        issues.append("Request/prompt/output/wait identity mismatch")
    issues.extend(runtime_pair_issues(baseline["backend_args"], treatment["backend_args"]))
    metrics = {}
    for key in ("workload_ms", "total_all_ttft_ms", "total_replay_ttft_ms", "total_replay_lateness_ms", "median_replay_delay_ms"):
        before, after = baseline["metrics"][key], treatment["metrics"][key]
        metrics[key] = {"baseline": before, "controller": after,
                        "change_pct": 100 * (after / before - 1) if before and after is not None and not issues else None}
    return {"trial": baseline["trial"], "comparable": not issues, "issues": sorted(set(issues)), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path, required=True)
    parser.add_argument("--progress-file", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    manifest = json.loads((root / "run_manifest.json").read_text())
    spec = manifest["spec"]
    for scenario in manifest["scenarios"]:
        records = [r for r in manifest["arms"] if r["scenario"] == scenario and r.get("completed_ns")]
        if len(records) != 4:
            print(f"Scenario {scenario}: waiting for all four arms")
            continue
        arms = [analyze_arm(root, arm, spec) for arm in records]
        pairs = []
        for trial in (1, 2):
            pair = [a for a in arms if a["trial"] == trial]
            baseline = next(a for a in pair if a["mode"] == "no_prefetch")
            treatment = next(a for a in pair if a["mode"] != "no_prefetch")
            pairs.append(pair_comparison(baseline, treatment))
        run_id = manifest["run_id"] + f"_s{scenario}"
        out = args.publish_dir / run_id
        out.mkdir(parents=True, exist_ok=True)
        for filename in ("experiment_spec.json", "source.tar.gz", "host_dependencies.txt", "container_dependencies.txt", "model_identity.json"):
            shutil.copy2(root / filename, out / filename)
        saved_manifest = {**manifest, "arms": records, "scenarios": [scenario]}
        (out / "run_manifest.json").write_text(json.dumps(saved_manifest, indent=2) + "\n")
        for arm in arms:
            for relative in (arm["case_path"], arm["report_path"], f"raw/runs/controlled/{arm['label']}/runtime"):
                shutil.copytree(root / relative, out / relative, dirs_exist_ok=True)
            shutil.copy2(root / f"{arm['label']}.log", out / f"{arm['label']}.log")
        try:
            analysis_revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            analysis_revision = None
        summary = {
            "schema": SCHEMA, "run_id": run_id, "scenario": scenario,
            "name": spec["scenarios"][scenario]["name"], "question": spec["scenarios"][scenario]["question"],
            "limits": spec["scenarios"][scenario]["limits"], "source_revision": manifest["source_revision"],
            "started_ns": min(r["started_ns"] for r in records), "arms": arms, "pairs": pairs,
            "status": "complete" if all(p["comparable"] for p in pairs) else "invalid_comparison",
            "measurement_boundary": manifest["measurement_boundary"], "matching": manifest["matching"],
            "analysis_revision": analysis_revision, "analysis_python_version": sys.version,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        shutil.copy2(Path(__file__), out / "analysis_source.py")
        hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(out.rglob("*")) if p.is_file() and p.name != "evidence_sha256.json"}
        (out / "evidence_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        if args.progress_file:
            from agentic_reports.builders.controller_pivot_report import finding
            progress = json.loads(args.progress_file.read_text())
            question_id = f"RQ{26 + int(scenario)}"
            existing = next((m for m in progress["milestones"] if m["id"] == question_id), None)
            if existing and existing.get("controller_scenario") != scenario:
                raise ValueError(f"Research question {question_id} is already owned by another study")
            milestone = {
                "id": question_id, "controller_scenario": scenario,
                "short_question": summary["question"], "pivot_title": f"Scenario {scenario}: {summary['name']}",
                "significance": "research_pivot" if summary["status"] == "complete" else "supporting_evidence",
                "manager_takeaway": finding(summary),
                "why_it_matters": "Tests one independent use of harness timing information, with equally important sessions. Measures the whole workload as well as replay response times.",
                "question": summary["question"], "answer": finding(summary),
                "unknown": summary["limits"] + " Two synthetic trials do not establish production-wide gains or a hardware bottleneck.",
                "evidence_date_utc": datetime.fromtimestamp(summary["started_ns"] / 1e9, timezone.utc).strftime("%Y-%m-%d"),
                "evidence_run_ids": [run_id],
                "related_run_ids": list(dict.fromkeys((existing or {}).get("related_run_ids", []) + [run_id])),
            }
            progress["milestones"] = [milestone] + [m for m in progress["milestones"] if m["id"] != question_id]
            args.progress_file.write_text(json.dumps(progress, indent=2) + "\n")
        print(f"Scenario {scenario}: {summary['status']} -> {out}")


if __name__ == "__main__":
    main()
