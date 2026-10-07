"""Validate native L3 hits and summarize paired storage replay arms."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

from agentic_backends.sglang.evidence import storage_audit_event_kind


def _replay_evidence(trace: Path) -> tuple[int, int, dict[str, tuple[int, int]]]:
    hits = 0
    matched = 0
    native_completions: dict[str, tuple[int, int]] = {}
    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("event") == "storage_prefetch.data_ready":
                native_completions[str(row.get("storage_request_id"))] = (
                    int(row["ts_ns"]), int(row["completed_tokens"])
                )
            event_kind = storage_audit_event_kind(row)
            if event_kind == "storage_hit_tokens":
                try:
                    hits += int(row.get("result") or 0)
                except (TypeError, ValueError):
                    pass
            if event_kind == "prefix_match" and str(
                (row.get("kv_context") or {}).get("agent_request_id") or ""
            ).endswith("-replay"):
                result = row.get("result")
                if isinstance(result, list) and result and isinstance(result[0], dict):
                    matched = max(matched, int(result[0].get("index_count") or 0))
    return hits, matched, native_completions


def summarize(arms_dir: Path, *, require_native_timing: bool = False,
              expected_arms: tuple[str, ...] = ("on_demand", "host_stage", "full_prepare")) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case_path in sorted(arms_dir.glob("*/case_results.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        arm = str(case["arm"])
        native_hits, matched_prefix, native_completions = _replay_evidence(
            case_path.parent / "backend_trace.jsonl.gz"
        )
        control_hits = int(((case.get("storage_stage") or {}).get("completed") or {}).get("storage_loaded_tokens") or 0)
        if arm == "on_demand" and native_hits <= 0:
            raise ValueError(f"{case_path}: on-demand replay has no proven native L3 hit")
        if arm != "on_demand" and (control_hits <= 0 or native_hits > 0):
            raise ValueError(f"{case_path}: staging was not proved or replay fetched L3 again")
        if arm != "on_demand" and not case["stage_completed_before_due"]:
            raise ValueError(f"{case_path}: L3 staging missed the tool-return deadline")
        expected = native_hits if arm == "on_demand" else control_hits
        if matched_prefix < int(expected * 0.8):
            raise ValueError(f"{case_path}: replay did not reuse the restored prefix")
        stage = case.get("storage_stage") or {}
        prep_start = (stage.get("accepted") or {}).get("control_request_ns")
        prep_end = ((case.get("device_load") or {}).get("completed") or {}).get("observed_ns")
        prep_end = prep_end or stage.get("completed_observed_ns")
        accepted = stage.get("accepted") or {}
        completed = stage.get("completed") or {}
        requested_ns = accepted.get("storage_requested_ns")
        ready_ns = completed.get("storage_data_ready_ns")
        committed_ns = completed.get("storage_host_committed_ns")
        has_native_timing = (
            isinstance(requested_ns, int) and isinstance(ready_ns, int)
            and isinstance(committed_ns, int) and requested_ns <= ready_ns <= committed_ns
        )
        if arm != "on_demand" and require_native_timing and not has_native_timing:
            raise ValueError(f"{case_path}: ordered native storage timestamps missing")
        if arm != "on_demand" and has_native_timing:
            request_id = str(accepted.get("storage_request_id"))
            if native_completions.get(request_id) != (ready_ns, control_hits):
                raise ValueError(f"{case_path}: native data-ready trace does not match control result")
        peers_overlapping_preparation = sum(
            bool(prep_start and prep_end and peer.get("request_start_ns") and peer.get("request_end_ns")
                 and peer["request_start_ns"] < prep_end and peer["request_end_ns"] > prep_start)
            for peer in case.get("peers") or []
        )
        rows.append({
            "seed": int(case_path.parent.name.split("_")[0].removeprefix("seed")),
            "arm": arm,
            "due_to_first_token_ms": case["due_to_first_token_ms"],
            "tool_end_to_first_token_ms": case["tool_end_to_first_token_ms"],
            "replay_ttft_ms": case["replay"]["ttft_ms"],
            "replay_total_latency_ms": case["replay"]["total_latency_ms"],
            "output_verification_enabled": bool(case.get("output_verification_enabled")),
            "replay_output_sha256": case["replay"].get("output_sha256"),
            "replay_output_characters": case["replay"].get("output_characters"),
            "workflow_duration_ms": case.get("workflow_duration_ms"),
            "peer_count": len(case.get("peers") or []),
            "peers_overlapping_preparation": peers_overlapping_preparation,
            "storage_data_ready_ms": round((ready_ns - requested_ns) / 1e6, 3) if has_native_timing else None,
            "storage_commit_after_ready_ms": round((committed_ns - ready_ns) / 1e6, 3) if has_native_timing else None,
            "storage_confirmed_ms": (
                round((stage["completed_observed_ns"] - requested_ns) / 1e6, 3)
                if requested_ns else None
            ),
            "peer_ttft_median_ms": (
                statistics.median(peer["ttft_ms"] for peer in case["peers"])
                if case.get("peers") else None
            ),
            "peer_completion_median_ms": (
                statistics.median(peer["total_latency_ms"] for peer in case["peers"])
                if case.get("peers") else None
            ),
            "peer_ttft_ms": [peer["ttft_ms"] for peer in case.get("peers") or []],
            "peer_completion_ms": [peer["total_latency_ms"] for peer in case.get("peers") or []],
            "native_replay_storage_hit_tokens": native_hits,
            "replay_matched_prefix_tokens": matched_prefix,
            "control_storage_hit_tokens": control_hits,
            "stage_completed_before_due": case["stage_completed_before_due"],
            "removed_host_match_tokens": case["storage_eviction"].get(
                "removed_host_match_tokens",
                int(case["storage_eviction"].get("host_tokens") or 0)
                - int(case["storage_eviction"].get("host_tokens_after") or 0),
            ),
        })
    if not rows:
        raise ValueError(f"No completed cases found in {arms_dir}")
    by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    paired = []
    for seed, arms in sorted(by_seed.items()):
        if set(arms) != set(expected_arms):
            raise ValueError(f"Seed {seed} has incomplete arms: {sorted(arms)}")
        if any(row["output_verification_enabled"] for row in arms.values()):
            hashes = [row["replay_output_sha256"] for row in arms.values()]
            if (not all(row["output_verification_enabled"] and row["replay_output_characters"]
                        for row in arms.values()) or len(set(hashes)) != 1):
                raise ValueError(f"Seed {seed}: replay output verification failed")
        base = arms["on_demand"]["due_to_first_token_ms"]
        pair = {"seed": seed,
                "host_stage_delta_ms": arms["host_stage"]["due_to_first_token_ms"] - base}
        if all(arms[arm]["workflow_duration_ms"] is not None for arm in ("on_demand", "host_stage")):
            pair["host_stage_workflow_delta_ms"] = (arms["host_stage"]["workflow_duration_ms"]
                                                    - arms["on_demand"]["workflow_duration_ms"])
        if len(expected_arms) == 2:
            baseline_peers = arms["on_demand"]["peer_ttft_ms"]
            staged_peers = arms["host_stage"]["peer_ttft_ms"]
            baseline_completions = arms["on_demand"]["peer_completion_ms"]
            staged_completions = arms["host_stage"]["peer_completion_ms"]
            pair["peer_ttft_deltas_ms"] = [new - old for new, old in zip(staged_peers, baseline_peers)]
            pair["peer_completion_deltas_ms"] = [new - old for new, old in zip(staged_completions,
                                                                                baseline_completions)]
            pair["observed_no_peer_harm"] = all(delta <= 0 for delta in (
                pair["peer_ttft_deltas_ms"] + pair["peer_completion_deltas_ms"]
            ))
            pair["observed_whole_workload_improvement"] = (
                pair.get("host_stage_workflow_delta_ms") is not None
                and pair["host_stage_workflow_delta_ms"] <= 0
            )
        if "full_prepare" in expected_arms:
            pair["full_prepare_delta_ms"] = arms["full_prepare"]["due_to_first_token_ms"] - base
        paired.append(pair)
    peer_count = rows[0]["peer_count"]
    if any(row["peer_count"] != peer_count for row in rows):
        raise ValueError("Peer count differs between paired arms")
    scope = ("Single-session storage timing only. Peer-session cost and a busy-workload benefit are not measured."
             if peer_count == 0 else
             "Small concurrent-peer timing only; this does not establish a production-workload benefit.")
    return {"schema": "agentic_work_audit.storage_replay.v1", "run_id": arms_dir.parent.name,
            "status": "complete", "rows": rows, "paired": paired,
            "median_host_stage_delta_ms": statistics.median(p["host_stage_delta_ms"] for p in paired),
            "median_full_prepare_delta_ms": (statistics.median(p["full_prepare_delta_ms"] for p in paired)
                                             if "full_prepare" in expected_arms else None),
            "interpretation_limit": scope + " A file-backend L3 hit does not prove physical SSD I/O; "
                                    "the OS page cache may serve reads."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-native-timing", action="store_true")
    parser.add_argument("--arms", default="on_demand host_stage full_prepare")
    args = parser.parse_args()
    expected_arms = tuple(args.arms.split())
    if expected_arms not in (("on_demand", "host_stage"),
                             ("on_demand", "host_stage", "full_prepare")):
        parser.error("unsupported arm list")
    result = summarize(args.arms_dir, require_native_timing=args.require_native_timing,
                       expected_arms=expected_arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
