"""Compare scheduler and worker KV preparation under the same busy workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODES = ("baseline", "check_only", "controller")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _arm(run: Path, mode: str) -> dict[str, Any]:
    root = run / "arms" / f"seed1_{mode}"
    summary = _read(root / "summary.json")
    events = _events(root / "harness_events.jsonl")
    loads = [row for row in events if row.get("kind") == "controller_load"]
    outcomes = [row for row in events if row.get("kind") == "controller_load_outcome"]
    accepted = [row for row in loads if row.get("load_id") and int(row.get("loaded_tokens") or 0) > 0]
    return {
        "replay_count": summary["replay_count"],
        "total_replay_ttft_ms": summary["total_replay_ttft_ms"],
        "total_return_to_first_token_ms": summary["total_return_to_first_token_ms"],
        "workflow_makespan_ms": summary["workflow_makespan_ms"],
        "replays": summary["replays"],
        "load_attempts": len(loads),
        "accepted_loads": len(accepted),
        "load_sessions": sorted({row["session_id"] for row in accepted}),
        "load_submit_ms": [
            round((row["load_response_ns"] - row["load_request_ns"]) / 1e6, 3)
            for row in accepted
            if isinstance(row.get("load_response_ns"), int) and isinstance(row.get("load_request_ns"), int)
        ],
        "confirmed_loads": sum(1 for row in outcomes if not row.get("load_confirmation_error")),
        "load_errors": [row["load_confirmation_error"] for row in outcomes if row.get("load_confirmation_error")],
    }


def compare(scheduler_run: Path, worker_run: Path) -> dict[str, Any]:
    runs = {"scheduler": scheduler_run, "worker": worker_run}
    manifests = {name: _read(path / "run_manifest.json") for name, path in runs.items()}
    workloads = {name: dict(manifest["workload"]) for name, manifest in manifests.items()}
    for workload in workloads.values():
        workload.pop("load_execution", None)
    if workloads["scheduler"] != workloads["worker"]:
        raise ValueError("The two runs do not have the same workload parameters")
    if manifests["scheduler"]["hardware_profile"] != manifests["worker"]["hardware_profile"]:
        raise ValueError("Hardware profiles differ")
    if workloads["scheduler"]["seeds"] != [1] or workloads["scheduler"]["modes"] != list(MODES):
        raise ValueError("This comparison requires seed 1 and the baseline/check-only/controller arms")

    arms = {name: {mode: _arm(path, mode) for mode in MODES} for name, path in runs.items()}
    affected = sorted(set(arms["scheduler"]["controller"]["load_sessions"]) |
                      set(arms["worker"]["controller"]["load_sessions"]))
    result: dict[str, Any] = {
        "schema": "agentic_work_audit.async_kv_comparison.v1",
        "runs": {name: path.name for name, path in runs.items()},
        "hardware_profile": manifests["scheduler"]["hardware_profile"],
        "workload": workloads["scheduler"],
        "load_affected_sessions_union": affected,
        "arms": {},
        "limitations": (
            "Fresh backends and equal synthetic inputs, but one seed per implementation; "
            "batch trajectories and natural cache admissions can differ. Summed TTFT is "
            "not wall-clock duration. CUDA work may overlap other requests."
        ),
    }
    for name, by_mode in arms.items():
        check = by_mode["check_only"]
        data = {}
        for mode, arm in by_mode.items():
            other_replays = [row for row in arm["replays"] if row["session_id"] not in affected]
            data[mode] = {
                "replay_count": arm["replay_count"],
                "total_replay_ttft_ms": arm["total_replay_ttft_ms"],
                "total_return_to_first_token_ms": arm["total_return_to_first_token_ms"],
                "workflow_makespan_ms": arm["workflow_makespan_ms"],
                "other_session_replay_count": len(other_replays),
                "other_session_total_ttft_ms": round(sum(row["ttft_ms"] for row in other_replays), 3),
                "load_sessions": arm["load_sessions"],
                "load_attempts": arm["load_attempts"],
                "accepted_loads": arm["accepted_loads"],
                "load_submit_ms": arm["load_submit_ms"],
                "confirmed_loads": arm["confirmed_loads"],
                "load_errors": arm["load_errors"],
            }
        data["controller_minus_check_only"] = {
            "total_replay_ttft_ms": round(by_mode["controller"]["total_replay_ttft_ms"] - check["total_replay_ttft_ms"], 3),
            "workflow_makespan_ms": round(by_mode["controller"]["workflow_makespan_ms"] - check["workflow_makespan_ms"], 3),
            "other_session_total_ttft_ms": round(
                data["controller"]["other_session_total_ttft_ms"] -
                data["check_only"]["other_session_total_ttft_ms"], 3
            ),
        }
        result["arms"][name] = data
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-run", type=Path, required=True)
    parser.add_argument("--worker-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.scheduler_run, args.worker_run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Compared {result['runs']['scheduler']} with {result['runs']['worker']}")


if __name__ == "__main__":
    main()
