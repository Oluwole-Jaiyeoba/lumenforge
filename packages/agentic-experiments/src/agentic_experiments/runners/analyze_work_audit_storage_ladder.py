"""Summarize verified storage-stage pairs as concurrent sessions increase."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def analyze(runs_root: Path, run_id: str, counts: list[int], seeds: list[int]) -> dict[str, Any]:
    rungs = []
    for count in counts:
        path = runs_root / f"{run_id}_n{count}" / "summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or len(summary["paired"]) != len(seeds):
            raise ValueError(f"Incomplete paired rung: {path}")
        if sorted(pair["seed"] for pair in summary["paired"]) != sorted(seeds):
            raise ValueError(f"Seed mismatch at rung {count}: {path}")
        if any(row["peer_count"] != count - 1 for row in summary["rows"]):
            raise ValueError(f"Session-count mismatch at rung {count}: {path}")
        pairs = summary["paired"]
        rungs.append({
            "sessions": count, "summary_path": str(path), "paired_seeds": len(pairs),
            "median_replay_delay_delta_ms": statistics.median(
                pair["host_stage_delta_ms"] for pair in pairs),
            "median_workflow_delta_ms": statistics.median(
                pair["host_stage_workflow_delta_ms"] for pair in pairs),
            "peer_ttft_worsened_count": sum(delta > 0 for pair in pairs
                                             for delta in pair["peer_ttft_deltas_ms"]),
            "peer_completion_worsened_count": sum(delta > 0 for pair in pairs
                                                   for delta in pair["peer_completion_deltas_ms"]),
            "all_peers_unharmed_in_every_pair": all(pair["observed_no_peer_harm"] for pair in pairs),
            "all_workflows_improved": all(pair["observed_whole_workload_improvement"] for pair in pairs),
            "pairs": pairs,
        })
    return {
        "schema": "agentic_work_audit.storage_ladder.v1", "run_id": run_id,
        "status": "complete", "rungs": rungs,
        "first_observed_peer_tradeoff_sessions": next(
            (row["sessions"] for row in rungs if not row["all_peers_unharmed_in_every_pair"]), None),
        "interpretation_limit": "A positive peer delta in these small synthetic pairs is an observed tradeoff, "
                                "not a statistically established threshold or physical-SSD effect.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-counts", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.runs_root, args.run_id,
                     [int(value) for value in args.session_counts.split()],
                     [int(value) for value in args.seeds.split()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": args.run_id,
                      "first_observed_peer_tradeoff_sessions": result["first_observed_peer_tradeoff_sessions"]}))


if __name__ == "__main__":
    main()
