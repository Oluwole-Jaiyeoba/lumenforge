"""Validate capacity-safe tier arms and summarize their full-system cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .analyze_work_audit_memory_tiers import trace_evidence


def summarize(case: dict[str, Any], matches: dict[str, dict[str, int]]) -> dict[str, Any]:
    turns = [row for session in case["sessions"] for row in session["turns"] if row["turn"] > 0]
    sources = [str((row.get("preparation") or {}).get("source_tier") or "unknown") for row in turns]
    matched = {row["request_id"]: matches.get(row["request_id"]) for row in turns}
    gpu_tokens = [int((value or {}).get("gpu_tokens") or 0) for value in matched.values()]
    source_counts = {tier: sources.count(tier) for tier in ("gpu", "host", "storage")}
    source_tokens = {tier: sum(
        int(((row.get("preparation") or {}).get("before") or {}).get(
            "storage_candidate_tokens" if tier == "storage" else f"{tier}_tokens"
        ) or 0)
        for row in turns if (row.get("preparation") or {}).get("source_tier") == tier
    ) for tier in ("gpu", "host", "storage")}
    capacity = case["capacity"]
    return {
        "seed": case["seed"], "pattern": case["pattern"], "mode": case["mode"],
        "started_ns": case["started_ns"], "session_count": len(case["sessions"]),
        "turns_per_session": case["config"]["turns"], "replay_count": len(turns),
        "workflow_duration_ms": case["workflow_duration_ms"], **case["metrics"],
        "per_session_completion_ms": case["per_session_completion_ms"],
        "native_match_count": sum(value is not None for value in matched.values()),
        "native_gpu_ready_replays": sum(tokens > 0 for tokens in gpu_tokens),
        "native_gpu_ready_tokens": sum(gpu_tokens),
        # The generic tier report consumes these fields. Here they describe the
        # native, pre-admission source proved by storage_status, not replay-time hits.
        "native_gpu_hit_replays": source_counts["gpu"],
        "native_gpu_hit_tokens": source_tokens["gpu"],
        "native_host_hit_replays": source_counts["host"],
        "native_host_hit_tokens": source_tokens["host"],
        "native_storage_hit_replays": source_counts["storage"],
        "native_storage_hit_tokens": source_tokens["storage"],
        "tier_source_evidence": "native_pre_admission_residency",
        "max_active_allowed": capacity["max_active_allowed"],
        "active_token_limit": capacity["active_token_limit"],
        "max_active_observed": capacity["max_active_observed"],
        "max_active_tokens_observed": capacity["max_active_tokens_observed"],
        "capacity_violations": capacity["violations"],
        "prompt_token_range": [min(row["prompt_tokens"] for row in turns),
                               max(row["prompt_tokens"] for row in turns)],
        "config": case["config"],
    }


def analyze(root: Path, expected_seeds: list[int], expected_patterns: list[str],
            expected_modes: list[str]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    issues: list[str] = []
    exposure: list[str] = []
    for case_path in sorted((root / "arms").glob("seed*/*/case_results.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        trace = case_path.parent / "backend_trace.jsonl.gz"
        if not trace.exists():
            trace = case_path.parent / "backend_trace.jsonl"
        matches, _storage = trace_evidence(trace)
        arms.append(summarize(case, matches))
    for failure in sorted((root / "arms").glob("seed*/*/case_failure.json")):
        value = json.loads(failure.read_text(encoding="utf-8"))
        issues.append(f"{failure.parent.relative_to(root / 'arms')}: {value.get('error')}")

    observed = {(arm["seed"], arm["pattern"], arm["mode"]) for arm in arms}
    missing = [(seed, pattern, mode) for seed in expected_seeds
               for pattern in expected_patterns for mode in expected_modes
               if (seed, pattern, mode) not in observed]
    if missing:
        issues.append("missing arms: " + ", ".join(f"{s}/{p}/{m}" for s, p, m in missing))

    for arm in arms:
        label = f"{arm['seed']}/{arm['pattern']}/{arm['mode']}"
        if arm["capacity_violations"]:
            issues.append(f"{label}: active-capacity invariant failed")
        if arm["max_active_observed"] > arm["max_active_allowed"]:
            issues.append(f"{label}: too many active sessions")
        if arm["max_active_tokens_observed"] > arm["active_token_limit"]:
            issues.append(f"{label}: active KV exceeded configured GPU-safe limit")
        if arm["native_match_count"] != arm["replay_count"]:
            issues.append(f"{label}: missing native replay cache matches")
        if arm["native_gpu_ready_replays"] != arm["replay_count"]:
            issues.append(f"{label}: one or more replays were admitted without GPU-ready KV")
        if arm["mode"] == "resident" and (
            arm["native_host_hit_replays"] or arm["native_storage_hit_replays"]
        ):
            issues.append(f"{label}: all-GPU reference used a lower-tier source")
        if arm["mode"] == "host" and arm["native_host_hit_replays"] == 0:
            exposure.append(f"{label}: no CPU-to-GPU restoration was proved")
        if arm["mode"] == "host" and arm["native_storage_hit_replays"]:
            issues.append(f"{label}: CPU-only tier unexpectedly used storage")
        if arm["mode"] == "storage" and arm["native_storage_hit_replays"] == 0:
            exposure.append(f"{label}: no storage-to-host restoration was proved")

    comparisons = []
    for seed in expected_seeds:
        for pattern in expected_patterns:
            modes = {arm["mode"]: arm for arm in arms
                     if arm["seed"] == seed and arm["pattern"] == pattern}
            resident = modes.get("resident")
            if not resident:
                continue
            for mode in ("host", "storage"):
                candidate = modes.get(mode)
                if not candidate:
                    continue
                comparisons.append({
                    "seed": seed, "pattern": pattern, "mode": mode,
                    "workload_delta_ms": round(candidate["workflow_duration_ms"] -
                                               resident["workflow_duration_ms"], 3),
                    "workload_change_pct": round(100 * (candidate["workflow_duration_ms"] /
                                                         resident["workflow_duration_ms"] - 1), 3),
                    "mean_due_to_first_token_delta_ms": round(
                        candidate["mean_due_to_first_token_ms"] - resident["mean_due_to_first_token_ms"], 3),
                    "p95_due_to_first_token_delta_ms": round(
                        candidate["p95_due_to_first_token_ms"] - resident["p95_due_to_first_token_ms"], 3),
                    "mean_ttft_delta_ms": round(candidate["mean_ttft_ms"] - resident["mean_ttft_ms"], 3),
                    "mean_slot_wait_delta_ms": round(
                        candidate["mean_slot_wait_ms"] - resident["mean_slot_wait_ms"], 3),
                    "mean_kv_prepare_delta_ms": round(
                        candidate["mean_kv_prepare_ms"] - resident["mean_kv_prepare_ms"], 3),
                })

    status = "blocked" if issues else "insufficient_exposure" if exposure else "complete"
    return {
        "schema": "agentic_work_audit.memory_tiers.summary.v1",
        "variant": "capacity_safe_active_set",
        "run_id": root.name, "status": status,
        "started_ns": min((arm["started_ns"] for arm in arms), default=None),
        "arms": sorted(arms, key=lambda row: (row["seed"], row["pattern"], row["mode"])),
        "comparisons": comparisons, "issues": issues, "exposure_warnings": exposure,
        "research_question": (
            "How large is the tiering penalty when every currently active session fits in GPU KV, "
            "while only waiting sessions are pushed to CPU or storage and restored after tool return?"
        ),
        "limitations": [
            "Synthetic equal-priority coding sessions with at most two active at once.",
            "Lower-tier restoration starts only after tool return; this is an unprepared worst case.",
            "File-backed storage can be served by the operating-system page cache.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--expected-patterns", nargs="+", required=True)
    parser.add_argument("--expected-modes", nargs="+", required=True)
    args = parser.parse_args()
    summary = analyze(args.run_root, args.expected_seeds, args.expected_patterns, args.expected_modes)
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "issues": summary["issues"],
                      "exposure_warnings": summary["exposure_warnings"],
                      "comparisons": summary["comparisons"]}, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
