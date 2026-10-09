"""Validate and compare native SGLang with capacity-safe admission."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .analyze_work_audit_capacity_safe_tiers import summarize as summarize_capacity
from .analyze_work_audit_memory_tiers import summarize as summarize_native
from .analyze_work_audit_memory_tiers import trace_evidence


POLICIES = ("native_sglang", "capacity_safe")


def _measurement_boundary(case: dict[str, Any]) -> str:
    explicit = (case.get("workload_contract") or {}).get("measurement_boundary")
    if explicit:
        return str(explicit)
    if case.get("policy") == "capacity_safe" or case.get("measurement_started_ns") is not None:
        return "after_initial_prefix_population"
    return "includes_initial_prefix_population"


def _arm_summary(case: dict[str, Any], trace: Path) -> dict[str, Any]:
    matches, storage = trace_evidence(trace)
    policy = case.get("policy")
    if policy == "native_sglang":
        value = summarize_native(case, matches, storage)
    elif policy == "capacity_safe":
        value = summarize_capacity(case, matches)
    else:
        raise ValueError(f"unknown policy: {policy}")
    value.update({
        "policy": policy,
        "workload_fingerprint": case.get("workload_fingerprint"),
        "workload_contract": case.get("workload_contract"),
        "backend_contract": case.get("backend_contract"),
        "measurement_boundary": _measurement_boundary(case),
        "initial_setup_ms": case.get("initial_setup_ms"),
    })
    return value


def analyze(root: Path, expected_seeds: list[int], expected_patterns: list[str]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    issues: list[str] = []
    exposure: list[str] = []
    for case_path in sorted((root / "arms").glob("seed*/*/case_results.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        trace = case_path.parent / "backend_trace.jsonl.gz"
        if not trace.exists():
            trace = case_path.parent / "backend_trace.jsonl"
        try:
            arms.append(_arm_summary(case, trace))
        except Exception as exc:
            issues.append(f"{case_path.parent.relative_to(root)}: {type(exc).__name__}: {exc}")
    for failure in sorted((root / "arms").glob("seed*/*/case_failure.json")):
        value = json.loads(failure.read_text(encoding="utf-8"))
        issues.append(f"{failure.parent.relative_to(root)}: {value.get('error')}")

    observed = {(arm["seed"], arm["pattern"], arm["policy"]) for arm in arms}
    missing = [(seed, pattern, policy) for seed in expected_seeds
               for pattern in expected_patterns for policy in POLICIES
               if (seed, pattern, policy) not in observed]
    if missing:
        issues.append("missing arms: " + ", ".join(f"{s}/{p}/{m}" for s, p, m in missing))

    comparisons: list[dict[str, Any]] = []
    for seed in expected_seeds:
        for pattern in expected_patterns:
            pair = {arm["policy"]: arm for arm in arms
                    if arm["seed"] == seed and arm["pattern"] == pattern}
            native = pair.get("native_sglang")
            safe = pair.get("capacity_safe")
            if not native or not safe:
                continue
            label = f"{seed}/{pattern}"
            if native["workload_fingerprint"] != safe["workload_fingerprint"]:
                issues.append(f"{label}: workload fingerprints differ")
            if native["backend_contract"] != safe["backend_contract"]:
                issues.append(f"{label}: backend contracts differ")
            if native["measurement_boundary"] != safe["measurement_boundary"]:
                issues.append(
                    f"{label}: workload clocks differ: native={native['measurement_boundary']}, "
                    f"capacity_safe={safe['measurement_boundary']}"
                )
            if native["replay_count"] != safe["replay_count"]:
                issues.append(f"{label}: replay counts differ")
            if native["output_tokens"] != safe["output_tokens"]:
                issues.append(f"{label}: output-token work differs")
            if native["native_match_count"] != native["replay_count"]:
                issues.append(f"{label}: native arm lacks complete cache-match evidence")
            if native["native_host_hit_replays"] == 0:
                exposure.append(f"{label}: native arm had no host-KV replay")
            if safe["native_host_hit_replays"] == 0:
                exposure.append(f"{label}: capacity-safe arm had no CPU-to-GPU restoration")
            if safe["capacity_violations"]:
                issues.append(f"{label}: capacity-safe invariant failed")
            if safe["native_gpu_ready_replays"] != safe["replay_count"]:
                issues.append(f"{label}: capacity-safe admitted replay without GPU-ready KV")

            comparisons.append({
                "seed": seed,
                "pattern": pattern,
                "native_workflow_ms": native["workflow_duration_ms"],
                "capacity_safe_workflow_ms": safe["workflow_duration_ms"],
                "workflow_delta_ms": round(safe["workflow_duration_ms"] -
                                             native["workflow_duration_ms"], 3),
                "workflow_change_pct": round(100 * (safe["workflow_duration_ms"] /
                                                       native["workflow_duration_ms"] - 1), 3),
                "native_mean_due_to_first_token_ms": native["mean_due_to_first_token_ms"],
                "capacity_safe_mean_due_to_first_token_ms": safe["mean_due_to_first_token_ms"],
                "mean_due_to_first_token_delta_ms": round(
                    safe["mean_due_to_first_token_ms"] - native["mean_due_to_first_token_ms"], 3),
                "native_p95_due_to_first_token_ms": native["p95_due_to_first_token_ms"],
                "capacity_safe_p95_due_to_first_token_ms": safe["p95_due_to_first_token_ms"],
                "p95_due_to_first_token_delta_ms": round(
                    safe["p95_due_to_first_token_ms"] - native["p95_due_to_first_token_ms"], 3),
                "native_total_due_to_first_token_ms": native["total_due_to_first_token_ms"],
                "capacity_safe_total_due_to_first_token_ms": safe["total_due_to_first_token_ms"],
                "native_mean_ttft_ms": native["mean_ttft_ms"],
                "capacity_safe_mean_ttft_ms": safe["mean_ttft_ms"],
                "native_submission_delay_ms": native["total_submission_delay_ms"],
                "capacity_safe_submission_delay_ms": safe["total_submission_delay_ms"],
            })

    medians: list[dict[str, Any]] = []
    for pattern in expected_patterns:
        rows = [row for row in comparisons if row["pattern"] == pattern]
        if rows:
            medians.append({
                "pattern": pattern,
                "native_workflow_ms": statistics.median(row["native_workflow_ms"] for row in rows),
                "capacity_safe_workflow_ms": statistics.median(
                    row["capacity_safe_workflow_ms"] for row in rows),
                "workflow_change_pct": statistics.median(row["workflow_change_pct"] for row in rows),
                "native_mean_due_to_first_token_ms": statistics.median(
                    row["native_mean_due_to_first_token_ms"] for row in rows),
                "capacity_safe_mean_due_to_first_token_ms": statistics.median(
                    row["capacity_safe_mean_due_to_first_token_ms"] for row in rows),
                "mean_due_to_first_token_delta_ms": statistics.median(
                    row["mean_due_to_first_token_delta_ms"] for row in rows),
            })

    status = "blocked" if issues else "insufficient_exposure" if exposure else "complete"
    return {
        "schema": "agentic_work_audit.native_vs_capacity_safe.summary.v1",
        "run_id": root.name,
        "status": status,
        "started_ns": min((arm["started_ns"] for arm in arms), default=None),
        "arms": sorted(arms, key=lambda row: (row["seed"], row["pattern"], row["policy"])),
        "comparisons": comparisons,
        "pattern_medians": medians,
        "issues": issues,
        "exposure_warnings": exposure,
        "research_question": (
            "For an identical CPU-tier workload and backend capacity, does capacity-safe "
            "pre-admission outperform native SGLang queue and cache management?"
        ),
        "limitations": [
            "This compares complete policies, not admission logic in isolation.",
            "Capacity-safe explicitly restores and verifies KV before submission.",
            "Synthetic equal-priority coding sessions use fixed tool waits.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-patterns", nargs="+", required=True)
    args = parser.parse_args()
    summary = analyze(args.run_root, args.expected_seeds, args.expected_patterns)
    (args.run_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": summary["status"], "issues": summary["issues"],
        "exposure_warnings": summary["exposure_warnings"],
        "pattern_medians": summary["pattern_medians"],
    }, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
