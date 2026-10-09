"""Validate the paired native/capacity-safe comparison across three KV tiers."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from .analyze_work_audit_native_vs_capacity_safe import _arm_summary


POLICIES = ("native_sglang", "capacity_safe")
MODES = ("resident", "host", "storage")


def _label(arm: dict[str, Any]) -> str:
    return f"{arm['seed']}/{arm['pattern']}/{arm['mode']}/{arm['policy']}"


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
            arm = _arm_summary(case, trace)
            arm["mode"] = case["mode"]
            arm["policy_contract"] = case.get("policy_contract")
            arms.append(arm)
        except Exception as exc:
            issues.append(f"{case_path.parent.relative_to(root)}: {type(exc).__name__}: {exc}")
    for failure in sorted((root / "arms").glob("seed*/*/case_failure.json")):
        value = json.loads(failure.read_text(encoding="utf-8"))
        issues.append(f"{failure.parent.relative_to(root)}: {value.get('error')}")

    expected = {(seed, pattern, mode, policy) for seed in expected_seeds
                for pattern in expected_patterns for mode in MODES for policy in POLICIES}
    observed = {(arm["seed"], arm["pattern"], arm["mode"], arm["policy"]) for arm in arms}
    missing = sorted(expected - observed)
    if missing:
        issues.append("missing arms: " + ", ".join("/".join(map(str, row)) for row in missing))

    for arm in arms:
        label = _label(arm)
        if arm["native_match_count"] != arm["replay_count"]:
            issues.append(f"{label}: missing native replay cache matches")
        if arm["policy"] == "capacity_safe":
            if arm.get("capacity_violations"):
                issues.append(f"{label}: capacity-safe invariant failed")
            if arm.get("native_gpu_ready_replays") != arm["replay_count"]:
                issues.append(f"{label}: replay admitted without GPU-ready KV")
        lower_hits = arm["native_host_hit_replays"] + arm["native_storage_hit_replays"]
        if arm["mode"] == "resident" and lower_hits:
            issues.append(f"{label}: GPU-resident arm used a lower tier")
        elif arm["mode"] == "host":
            if arm["native_host_hit_replays"] == 0:
                exposure.append(f"{label}: no CPU-to-GPU recovery was proved")
            if arm["native_storage_hit_replays"]:
                issues.append(f"{label}: CPU-tier arm unexpectedly used storage")
        elif arm["mode"] == "storage" and arm["native_storage_hit_replays"] == 0:
            exposure.append(f"{label}: no storage-to-host recovery was proved")

    policy_comparisons: list[dict[str, Any]] = []
    tier_comparisons: list[dict[str, Any]] = []
    for seed in expected_seeds:
        for pattern in expected_patterns:
            group = [arm for arm in arms if arm["seed"] == seed and arm["pattern"] == pattern]
            fingerprints = {arm.get("workload_fingerprint") for arm in group}
            if len(fingerprints) > 1:
                issues.append(f"{seed}/{pattern}: workload fingerprints differ across the six arms")
            replay_counts = {arm["replay_count"] for arm in group}
            output_counts = {arm["output_tokens"] for arm in group}
            if len(replay_counts) > 1:
                issues.append(f"{seed}/{pattern}: replay counts differ")
            if len(output_counts) > 1:
                issues.append(f"{seed}/{pattern}: output-token work differs")

            for mode in MODES:
                pair = {arm["policy"]: arm for arm in group if arm["mode"] == mode}
                native = pair.get("native_sglang")
                safe = pair.get("capacity_safe")
                if not native or not safe:
                    continue
                if native.get("backend_contract") != safe.get("backend_contract"):
                    issues.append(f"{seed}/{pattern}/{mode}: native and capacity-safe capacities differ")
                if native.get("measurement_boundary") != safe.get("measurement_boundary"):
                    issues.append(
                        f"{seed}/{pattern}/{mode}: native and capacity-safe workload clocks differ"
                    )
                policy_comparisons.append({
                    "seed": seed, "pattern": pattern, "mode": mode,
                    "backend_contract": native.get("backend_contract"),
                    "native_workflow_ms": native["workflow_duration_ms"],
                    "capacity_safe_workflow_ms": safe["workflow_duration_ms"],
                    "workflow_delta_ms": round(safe["workflow_duration_ms"] - native["workflow_duration_ms"], 3),
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
                    "native_mean_ttft_ms": native["mean_ttft_ms"],
                    "capacity_safe_mean_ttft_ms": safe["mean_ttft_ms"],
                    "native_submission_delay_ms": native["total_submission_delay_ms"],
                    "capacity_safe_submission_delay_ms": safe["total_submission_delay_ms"],
                })

            for policy in POLICIES:
                by_mode = {arm["mode"]: arm for arm in group if arm["policy"] == policy}
                resident = by_mode.get("resident")
                if not resident:
                    continue
                for mode in ("host", "storage"):
                    candidate = by_mode.get(mode)
                    if not candidate:
                        continue
                    tier_comparisons.append({
                        "seed": seed, "pattern": pattern, "policy": policy, "mode": mode,
                        "workload_delta_ms": round(candidate["workflow_duration_ms"] -
                                                   resident["workflow_duration_ms"], 3),
                        "workload_change_pct": round(100 * (candidate["workflow_duration_ms"] /
                                                             resident["workflow_duration_ms"] - 1), 3),
                        "mean_due_to_first_token_delta_ms": round(
                            candidate["mean_due_to_first_token_ms"] -
                            resident["mean_due_to_first_token_ms"], 3),
                    })

    medians: list[dict[str, Any]] = []
    for pattern in expected_patterns:
        for mode in MODES:
            rows = [row for row in policy_comparisons
                    if row["pattern"] == pattern and row["mode"] == mode]
            if rows:
                medians.append({
                    "pattern": pattern, "mode": mode,
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
        "schema": "agentic_work_audit.tier_policy_matrix.summary.v1",
        "run_id": root.name,
        "status": status,
        "started_ns": min((arm["started_ns"] for arm in arms), default=None),
        "arms": sorted(arms, key=lambda row: (
            row["seed"], row["pattern"], row["mode"], row["policy"])),
        "policy_comparisons": policy_comparisons,
        "tier_comparisons": tier_comparisons,
        "pattern_tier_medians": medians,
        "issues": issues,
        "exposure_warnings": exposure,
        "research_question": (
            "For identical workloads and pair-matched memory capacities, how does capacity-safe "
            "admission compare with native SGLang in GPU-resident, CPU-tiered, and storage-tiered modes?"
        ),
        "apples_to_apples_contract": {
            "same_within_every_native_capacity_pair": [
                "model", "requests", "session count", "turn count", "prompt and output sizes",
                "tool waits", "return pattern", "random seed", "GPU KV capacity",
                "host KV capacity", "storage configuration", "CUDA graphs",
                "overlap scheduling", "frontend priority", "trace profile",
            ],
            "only_intentional_pair_difference": "admission and restore policy",
        },
        "limitations": [
            "This compares complete policies, not admission logic in isolation.",
            "Capacity-safe restores and verifies lower-tier KV before backend submission.",
            "File-backed storage may be served by the operating-system page cache.",
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
        "pattern_tier_medians": summary["pattern_tier_medians"],
    }, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
