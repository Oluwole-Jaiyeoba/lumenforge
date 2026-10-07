"""Summarize repeated storage-resume runs without losing per-turn evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any


def native_hits(trace_path: Path) -> dict[str, int]:
    rid_to_request: dict[str, str] = {}
    hits_by_rid: dict[str, int] = {}
    with gzip.open(trace_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            event = row.get("event")
            details = row.get("kv_context") or {}
            if event == "hiradix.match_prefix.end":
                rid = (details.get("request") or {}).get("rid")
                request_id = details.get("agent_request_id")
                if rid and request_id:
                    rid_to_request[str(rid)] = str(request_id)
            elif event == "hiradix.storage_hit_tokens.end":
                rid = details.get("request_id")
                loaded = int(details.get("storage_loaded_tokens") or row.get("result") or 0)
                if rid and loaded > 0:
                    hits_by_rid[str(rid)] = max(loaded, hits_by_rid.get(str(rid), 0))
    return {rid_to_request[rid]: tokens for rid, tokens in hits_by_rid.items() if rid in rid_to_request}


def summarize_arm(case: dict[str, Any], hit_tokens: dict[str, int]) -> dict[str, Any]:
    turns = [row for row in case["turns"] if row["turn"] > 0]
    delays = [row["due_to_first_token_ms"] for row in turns]
    ttfts = [row["ttft_ms"] for row in turns]
    storage_hits = {row["request_id"]: hit_tokens[row["request_id"]]
                    for row in turns if row["request_id"] in hit_tokens}
    return {
        "arm": case["arm"], "seed": case["seed"],
        "started_ns": case["started_ns"],
        "session_count": case["session_count"], "replay_count": len(turns),
        "workflow_duration_ms": case["workflow_duration_ms"],
        "session_completion_ms": [row["completion_ms"] for row in case["sessions"]],
        "session_completion_by_id_ms": {row["session_id"]: row["completion_ms"] for row in case["sessions"]
                                        if "session_id" in row},
        "due_to_first_token_ms": delays,
        "due_to_first_token_median_ms": round(statistics.median(delays), 3),
        "due_to_first_token_p95_ms": round(sorted(delays)[min(len(delays)-1, int(len(delays)*0.95))], 3),
        "replay_ttft_ms": ttfts,
        "replay_ttft_median_ms": round(statistics.median(ttfts), 3),
        "natural_storage_candidate_waits": case["natural_storage_candidate_waits"],
        "safe_stage_eligible_waits": case.get("safe_stage_eligible_waits", 0),
        "inspection_before_due_count": case["inspections_before_due"],
        "native_replay_storage_hits": storage_hits,
        "native_replay_storage_hit_count": len(storage_hits),
        "native_replay_storage_hit_tokens": sum(storage_hits.values()),
        "stage_before_due_count": case["stage_before_due_count"],
        "stage_loaded_tokens": sum(int((row["preparation"]["stage"]["completed"] or {}).get(
            "storage_loaded_tokens") or 0) for row in turns
            if row["preparation"] and "stage" in row["preparation"]),
        "stage_attempt_count": sum(bool(row["preparation"]) for row in turns),
        "stage_admit_count": sum(bool((row["preparation"] or {}).get("stage")) for row in turns),
        "stage_after_due_count": sum(bool((row["preparation"] or {}).get("stage")) and
                                     not bool((row["preparation"] or {}).get("stage_before_due"))
                                     for row in turns),
        "preparation_errors": [row["preparation"]["error"] for row in turns
                               if row["preparation"] and "error" in row["preparation"]],
        "preparation_skips": [row["preparation"]["skip_reason"] for row in turns
                              if row["preparation"] and "skip_reason" in row["preparation"]],
        "prompt_token_range": [min(row["prompt_tokens"] for row in turns),
                               max(row["prompt_tokens"] for row in turns)],
    }


def analyze(arms_dir: Path, *, expected_seeds: list[int] | None = None,
            expected_arms: list[str] | None = None,
            related_run_ids: list[str] | None = None) -> dict[str, Any]:
    arms = []
    for case_path in sorted(arms_dir.glob("seed*/case_results.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        hits = native_hits(case_path.parent / "backend_trace.jsonl.gz")
        arms.append(summarize_arm(case, hits))
    if not arms:
        raise ValueError(f"No completed arm results under {arms_dir}")
    arms.sort(key=lambda arm: (arm["seed"], {"on_demand": 0, "host_stage": 1,
                                                  "selective_stage": 2}.get(arm["arm"], 9)))
    failed_arms = []
    for failure_path in sorted(arms_dir.glob("seed*/case_failure.json")):
        failure = json.loads(failure_path.read_text(encoding="utf-8"))
        server_log = failure_path.parent / "server.log"
        log_text = server_log.read_text(encoding="utf-8", errors="replace") if server_log.exists() else ""
        assertion = next((line.split("AssertionError:", 1)[1].strip()[:120]
                          for line in log_text.splitlines() if "AssertionError:" in line), None)
        failed_arms.append({"arm_id": failure_path.parent.name,
                            "client_error": failure.get("error"),
                            "backend_assertion": assertion})
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for arm in arms:
        by_seed.setdefault(arm["seed"], {})[arm["arm"]] = arm
    comparisons = []
    staged_replay_outcomes = []
    for seed, modes in sorted(by_seed.items()):
        baseline = modes.get("on_demand")
        if baseline is None:
            continue
        for mode_name in ("host_stage", "selective_stage"):
            mode = modes.get(mode_name)
            if mode is None:
                continue
            comparison = {
                "seed": seed, "mode": mode_name,
                "workflow_delta_ms": round(mode["workflow_duration_ms"] - baseline["workflow_duration_ms"], 3),
                "median_due_to_first_token_delta_ms": round(
                    mode["due_to_first_token_median_ms"] - baseline["due_to_first_token_median_ms"], 3),
                "median_replay_ttft_delta_ms": round(
                    mode["replay_ttft_median_ms"] - baseline["replay_ttft_median_ms"], 3),
            }
            if mode_name == "selective_stage":
                baseline_sessions = baseline["session_completion_by_id_ms"]
                comparison["session_completion_delta_by_id_ms"] = {
                    session_id: round(completion_ms - baseline_sessions[session_id], 3)
                    for session_id, completion_ms in mode["session_completion_by_id_ms"].items()
                    if session_id in baseline_sessions
                }
                comparison["p95_due_to_first_token_delta_ms"] = round(
                    mode["due_to_first_token_p95_ms"] - baseline["due_to_first_token_p95_ms"], 3)
                baseline_case = json.loads((arms_dir / f"seed{seed}_on_demand" /
                                            "case_results.json").read_text(encoding="utf-8"))
                staged_case = json.loads((arms_dir / f"seed{seed}_selective_stage" /
                                          "case_results.json").read_text(encoding="utf-8"))
                baseline_turns = {turn["request_id"]: turn for turn in baseline_case["turns"]}
                for turn in staged_case["turns"]:
                    preparation = turn.get("preparation") or {}
                    if not preparation.get("stage_before_due"):
                        continue
                    original = baseline_turns.get(turn["request_id"])
                    if original is None:
                        continue
                    staged_replay_outcomes.append({
                        "seed": seed, "request_id": turn["request_id"],
                        "baseline_due_to_first_token_ms": original["due_to_first_token_ms"],
                        "staged_due_to_first_token_ms": turn["due_to_first_token_ms"],
                        "delta_ms": round(turn["due_to_first_token_ms"] -
                                          original["due_to_first_token_ms"], 3),
                        "staged_storage_tokens": int((preparation["stage"]["completed"] or {}).get(
                            "storage_loaded_tokens") or 0),
                        "baseline_native_replay_storage_hit_tokens": baseline[
                            "native_replay_storage_hits"].get(turn["request_id"], 0),
                        "staged_native_replay_storage_hit_tokens": mode[
                            "native_replay_storage_hits"].get(turn["request_id"], 0),
                    })
            comparisons.append(comparison)
    observed_arm_ids = {f"seed{arm['seed']}_{arm['arm']}" for arm in arms}
    missing_arm_ids = [f"seed{seed}_{arm}" for seed in (expected_seeds or [])
                       for arm in (expected_arms or [])
                       if f"seed{seed}_{arm}" not in observed_arm_ids]
    staged_arms = [arm for arm in arms if arm["arm"] in {"host_stage", "selective_stage"}]
    has_effective_stage = any(arm["stage_before_due_count"] > 0 for arm in staged_arms)
    has_preparation_error = any(arm["preparation_errors"] for arm in staged_arms)
    selective_seed_exposure = {
        seed: {
            "baseline_native_storage_hits": modes["on_demand"]["native_replay_storage_hit_count"],
            "early_stages": modes["selective_stage"]["stage_before_due_count"],
            "valid": modes["on_demand"]["native_replay_storage_hit_count"] > 0
                     and modes["selective_stage"]["stage_before_due_count"] > 0,
        }
        for seed, modes in by_seed.items()
        if "on_demand" in modes and "selective_stage" in modes
    }
    selective_exposure_complete = all(row["valid"] for row in selective_seed_exposure.values())
    status = ("blocked" if failed_arms or missing_arm_ids or has_preparation_error else
              "complete" if comparisons and has_effective_stage and selective_exposure_complete else
              "insufficient_exposure" if comparisons else "calibration")
    return {"schema": "agentic_work_audit.storage_cycles.summary.v1", "status": status,
            "started_ns": min(arm["started_ns"] for arm in arms), "arms": arms,
            "paired_comparisons": comparisons, "failed_arms": failed_arms,
            "staged_replay_outcomes": staged_replay_outcomes,
            "selective_seed_exposure": selective_seed_exposure,
            "related_run_ids": related_run_ids or [],
            "missing_arm_ids": missing_arm_ids,
            "blocked_reason": ("A paired arm failed or a non-capacity storage preparation error occurred; "
                               "inspect the saved server log and per-turn control result.")
                              if status == "blocked" else None,
            "limitations": ["file-backed storage is not an independently benchmarked physical SSD",
                            "a missing-suffix residency observation is not proof of a physical SSD read",
                            "requests are synthetic equal-priority coding sessions"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="*", type=int)
    parser.add_argument("--expected-arms", nargs="*")
    parser.add_argument("--related-run-id", action="append", default=[])
    args = parser.parse_args()
    result = analyze(args.arms_dir, expected_seeds=args.expected_seeds,
                     expected_arms=args.expected_arms, related_run_ids=args.related_run_id)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "arms": len(result["arms"]),
                      "comparisons": len(result["paired_comparisons"])}))


if __name__ == "__main__":
    main()
