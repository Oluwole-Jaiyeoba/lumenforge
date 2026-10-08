"""Join three-tier workload timings to native SGLang cache evidence."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any, TextIO

from agentic_backends.coordinated_audit import replay_prefix_match


def _open_trace(path: Path) -> TextIO:
    return gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else path.open(encoding="utf-8")


def trace_evidence(path: Path) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    matches: dict[str, dict[str, int]] = {}
    rid_to_request: dict[str, str] = {}
    storage_by_rid: dict[str, int] = {}
    with _open_trace(path) as handle:
        for line in handle:
            row = json.loads(line)
            matched = replay_prefix_match(row)
            if matched and matched[0] not in matches:
                matches[matched[0]] = matched[1]
            event = row.get("event")
            details = row.get("kv_context") or {}
            if event == "hiradix.match_prefix.end":
                request = details.get("request") or {}
                rid = request.get("rid")
                request_id = details.get("agent_request_id") or request.get("agent_request_id")
                if rid and request_id:
                    rid_to_request[str(rid)] = str(request_id)
            elif event == "hiradix.storage_hit_tokens.end":
                rid = details.get("request_id")
                tokens = int(details.get("storage_loaded_tokens") or row.get("result") or 0)
                if rid and tokens > 0:
                    storage_by_rid[str(rid)] = max(tokens, storage_by_rid.get(str(rid), 0))
    storage = {rid_to_request[rid]: tokens for rid, tokens in storage_by_rid.items()
               if rid in rid_to_request}
    return matches, storage


def summarize(case: dict[str, Any], matches: dict[str, dict[str, int]],
              storage: dict[str, int]) -> dict[str, Any]:
    turns = [row for session in case["sessions"] for row in session["turns"] if row["turn"] > 0]
    replay_matches = {row["request_id"]: matches.get(row["request_id"]) for row in turns}
    native_host = {request_id: int(match.get("host_tokens") or 0)
                   for request_id, match in replay_matches.items() if match}
    native_gpu = {request_id: int(match.get("gpu_tokens") or 0)
                  for request_id, match in replay_matches.items() if match}
    inspections = [inspection for round_row in case["rounds"]
                   for inspection in round_row["inspections"]]
    return {
        "seed": case["seed"], "pattern": case["pattern"], "mode": case["mode"],
        "started_ns": case["started_ns"], "session_count": len(case["sessions"]),
        "turns_per_session": case["config"]["turns"], "replay_count": len(turns),
        "workflow_duration_ms": case["workflow_duration_ms"],
        **case["metrics"],
        "per_session_completion_ms": case["per_session_completion_ms"],
        "native_match_count": sum(match is not None for match in replay_matches.values()),
        "native_gpu_hit_replays": sum(tokens > 0 for tokens in native_gpu.values()),
        "native_gpu_hit_tokens": sum(native_gpu.values()),
        "native_host_hit_replays": sum(tokens > 0 for tokens in native_host.values()),
        "native_host_hit_tokens": sum(native_host.values()),
        "native_storage_hit_replays": sum(request_id in storage for request_id in replay_matches),
        "native_storage_hit_tokens": sum(storage.get(request_id, 0) for request_id in replay_matches),
        "inspection_gpu_tokens": sum(int(row.get("gpu_tokens") or 0) for row in inspections),
        "inspection_host_tokens": sum(int(row.get("host_tokens") or 0) for row in inspections),
        "inspection_storage_candidate_tokens": sum(int(row.get("storage_candidate_tokens") or 0)
                                                   for row in inspections),
        "inspection_control_wall_ms": sum((row["inspection_finished_ns"] - row["inspection_started_ns"]) / 1e6
                                          for row in inspections),
        "prompt_token_range": [min(row["prompt_tokens"] for row in turns),
                               max(row["prompt_tokens"] for row in turns)],
        "config": case["config"],
    }


def analyze(root: Path, expected_seeds: list[int], expected_patterns: list[str],
            expected_modes: list[str]) -> dict[str, Any]:
    arms: list[dict[str, Any]] = []
    failures: list[str] = []
    for case_path in sorted((root / "arms").glob("seed*/*/case_results.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        trace = case_path.parent / "backend_trace.jsonl.gz"
        if not trace.exists():
            trace = case_path.parent / "backend_trace.jsonl"
        matches, storage = trace_evidence(trace)
        arms.append(summarize(case, matches, storage))
    for failure in sorted((root / "arms").glob("seed*/*/case_failure.json")):
        value = json.loads(failure.read_text(encoding="utf-8"))
        failures.append(f"{failure.parent.relative_to(root / 'arms')}: {value.get('error')}")

    observed = {(arm["seed"], arm["pattern"], arm["mode"]) for arm in arms}
    missing = [(seed, pattern, mode) for seed in expected_seeds
               for pattern in expected_patterns for mode in expected_modes
               if (seed, pattern, mode) not in observed]
    issues = list(failures)
    if missing:
        issues.append("missing arms: " + ", ".join(f"{s}/{p}/{m}" for s, p, m in missing))
    exposure: list[str] = []
    for arm in arms:
        if arm["native_match_count"] != arm["replay_count"]:
            issues.append(f"{arm['seed']}/{arm['pattern']}/{arm['mode']}: missing native replay matches")
        if arm["mode"] == "resident" and arm["native_host_hit_replays"] + arm["native_storage_hit_replays"]:
            exposure.append(f"{arm['seed']}/{arm['pattern']}/resident unexpectedly used a lower tier")
        if arm["mode"] == "host" and arm["native_host_hit_replays"] == 0:
            exposure.append(f"{arm['seed']}/{arm['pattern']}/host had no native host-KV replay")
        if arm["mode"] == "host" and arm["native_storage_hit_replays"]:
            issues.append(f"{arm['seed']}/{arm['pattern']}/host unexpectedly used storage")
        if arm["mode"] == "storage" and arm["native_storage_hit_replays"] == 0:
            exposure.append(f"{arm['seed']}/{arm['pattern']}/storage had no native storage replay")

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
                })

    status = "blocked" if issues else "insufficient_exposure" if exposure else "complete"
    return {
        "schema": "agentic_work_audit.memory_tiers.summary.v1",
        "run_id": root.name,
        "status": status,
        "started_ns": min((arm["started_ns"] for arm in arms), default=None),
        "arms": sorted(arms, key=lambda row: (row["seed"], row["pattern"], row["mode"])),
        "comparisons": comparisons,
        "issues": issues,
        "exposure_warnings": exposure,
        "research_question": ("How wide is the performance gap between all-GPU KV, GPU+CPU tiering, "
                              "and GPU+CPU+storage tiering when equal tool calls return together "
                              "and no KV is prefetched?"),
        "limitations": [
            "Synthetic equal-priority coding sessions; no semantic request priority.",
            "File-backed L3 is normal system storage; the OS page cache is not flushed.",
            "This exposes an unprepared return burst, not the later tool-aware optimized policy.",
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
