#!/usr/bin/env python3
"""Join fresh-backend baseline/controller arms without hiding weak evidence."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def compare(arms: list[dict[str, Any]], run_id: str) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        seed = int(arm["seed"])
        mode = str(arm["mode"])
        if mode in grouped.setdefault(seed, {}):
            raise ValueError(f"duplicate {mode} arm for seed {seed}")
        grouped[seed][mode] = arm
    pairs = []
    for seed, modes in sorted(grouped.items()):
        if set(modes) != {"baseline", "controller"}:
            raise ValueError(f"seed {seed} does not contain exactly baseline and controller")
        base, controlled = modes["baseline"], modes["controller"]
        if base["workload"] != controlled["workload"]:
            raise ValueError(f"seed {seed} arms used different workload settings")
        if base["frontend_priority"] != "none" or controlled["frontend_priority"] != "none":
            raise ValueError("frontend priority must be absent in both arms")
        if base["forced_eviction"] or controlled["forced_eviction"]:
            raise ValueError("busy audit forbids forced eviction")
        left = {row["request_id"]: row for row in base["replays"]}
        right = {row["request_id"]: row for row in controlled["replays"]}
        if left.keys() != right.keys() or len(left) != base["replay_count"]:
            raise ValueError(f"seed {seed} has unmatched or duplicate replay IDs")
        session_deltas: dict[str, list[float]] = {}
        for request_id, baseline_replay in left.items():
            controller_replay = right[request_id]
            session = str(baseline_replay["session_id"])
            session_deltas.setdefault(session, []).append(
                round(baseline_replay["tool_return_to_first_token_ms"] -
                      controller_replay["tool_return_to_first_token_ms"], 3)
            )
        per_session = {session: round(sum(values), 3) for session, values in session_deltas.items()}
        decision_rows = controlled["decisions"]
        ready = [row for row in decision_rows if row["plan_status"] == "would_load_back"]
        completed = controlled["controller_loads_finished_before_tool_return"]
        evidence_reasons = []
        if base["native_load_events"] == 0:
            evidence_reasons.append("baseline had no native host reloads")
        if not ready:
            evidence_reasons.append("controller saw no eligible host-resident prefix")
        if not completed:
            evidence_reasons.append("no controller load was confirmed complete before tool return")
        pairs.append({
            "seed": seed, "baseline_run_id": base["run_id"],
            "controller_run_id": controlled["run_id"],
            "workload": base["workload"], "replay_count": len(left),
            "baseline": {key: base[key] for key in (
                "workflow_makespan_ms", "total_replay_ttft_ms", "total_return_to_first_token_ms",
                "return_to_first_token", "native_load_events")},
            "controller": {key: controlled[key] for key in (
                "workflow_makespan_ms", "total_replay_ttft_ms", "total_return_to_first_token_ms",
                "return_to_first_token", "native_load_events", "controller_plan_checks",
                "controller_load_attempts", "controller_loads_finished_before_tool_return",
                "controller_loads_finished_after_tool_return")},
            "workflow_saved_ms": round(base["workflow_makespan_ms"] - controlled["workflow_makespan_ms"], 3),
            "total_replay_ttft_saved_ms": round(base["total_replay_ttft_ms"] - controlled["total_replay_ttft_ms"], 3),
            "total_return_to_first_token_saved_ms": round(
                base["total_return_to_first_token_ms"] - controlled["total_return_to_first_token_ms"], 3),
            "per_session_return_to_first_token_saved_ms": per_session,
            "sessions_helped": sum(value > 0 for value in per_session.values()),
            "sessions_harmed": sum(value < 0 for value in per_session.values()),
            "sessions_unchanged": sum(value == 0 for value in per_session.values()),
            "eligible_host_checks": len(ready),
            "evidence_reasons": evidence_reasons,
            "evidence_gate": "comparable" if not evidence_reasons else "limited",
        })
    return {
        "schema": "agentic_work_audit.busy_comparison.v1", "run_id": run_id,
        "status": "validated" if all(pair["evidence_gate"] == "comparable" for pair in pairs) else "inconclusive",
        "pairs": pairs, "seed_count": len(pairs),
        "median_workflow_saved_ms": _median([pair["workflow_saved_ms"] for pair in pairs]),
        "median_total_replay_ttft_saved_ms": _median([pair["total_replay_ttft_saved_ms"] for pair in pairs]),
        "median_total_return_to_first_token_saved_ms": _median(
            [pair["total_return_to_first_token_saved_ms"] for pair in pairs]),
        "limitations": [
            "Synthetic coding-task prompts and tool waits, not measured production trajectories.",
            "Arms use fresh backends and identical recipes, but concurrent batch and eviction paths may diverge.",
            "A changed latency is not by itself proof of HBM contention or avoidable KV movement.",
            "Native load completion is observed by polling and may lag physical CUDA completion.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    arms = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.arms_dir.glob("*/summary.json"))]
    if not arms:
        raise SystemExit("no busy-audit arms found")
    summary = compare(arms, args.run_id)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compared {summary['seed_count']} paired seeds; status={summary['status']}")


if __name__ == "__main__":
    main()
