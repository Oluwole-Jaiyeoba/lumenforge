"""Validate native L3 hits and summarize paired storage replay arms."""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
from pathlib import Path
from typing import Any

from agentic_backends.sglang.evidence import storage_audit_event_kind


def _replay_evidence(trace: Path) -> tuple[int, int]:
    hits = 0
    matched = 0
    with gzip.open(trace, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
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
    return hits, matched


def summarize(arms_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case_path in sorted(arms_dir.glob("*/case_results.json")):
        case = json.loads(case_path.read_text(encoding="utf-8"))
        arm = str(case["arm"])
        native_hits, matched_prefix = _replay_evidence(case_path.parent / "backend_trace.jsonl.gz")
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
            "workflow_duration_ms": case.get("workflow_duration_ms"),
            "peer_count": len(case.get("peers") or []),
            "peers_overlapping_preparation": peers_overlapping_preparation,
            "peer_ttft_median_ms": (
                statistics.median(peer["ttft_ms"] for peer in case["peers"])
                if case.get("peers") else None
            ),
            "peer_completion_median_ms": (
                statistics.median(peer["total_latency_ms"] for peer in case["peers"])
                if case.get("peers") else None
            ),
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
        if set(arms) != {"on_demand", "host_stage", "full_prepare"}:
            raise ValueError(f"Seed {seed} has incomplete arms: {sorted(arms)}")
        base = arms["on_demand"]["due_to_first_token_ms"]
        paired.append({"seed": seed,
                       "host_stage_delta_ms": arms["host_stage"]["due_to_first_token_ms"] - base,
                       "full_prepare_delta_ms": arms["full_prepare"]["due_to_first_token_ms"] - base})
    peer_count = rows[0]["peer_count"]
    if any(row["peer_count"] != peer_count for row in rows):
        raise ValueError("Peer count differs between paired arms")
    scope = ("Single-session storage timing only. Peer-session cost and a busy-workload benefit are not measured."
             if peer_count == 0 else
             "Small concurrent-peer timing only; this does not establish a production-workload benefit.")
    return {"schema": "agentic_work_audit.storage_replay.v1", "run_id": arms_dir.parent.name,
            "status": "complete", "rows": rows, "paired": paired,
            "median_host_stage_delta_ms": statistics.median(p["host_stage_delta_ms"] for p in paired),
            "median_full_prepare_delta_ms": statistics.median(p["full_prepare_delta_ms"] for p in paired),
            "interpretation_limit": scope + " A file-backend L3 hit does not prove physical SSD I/O; "
                                    "the OS page cache may serve reads."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.arms_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
