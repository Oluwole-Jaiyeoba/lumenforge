#!/usr/bin/env python3
"""Join pinned backend evidence to the paired work-audit timing timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_backends.sglang.audit_v0510 import normalize_lifecycle_evidence, translate_trace
from agentic_instrumentation import assess_loaded_match
from agentic_work_audit import analyze_timing, read_events, write_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-slot-proof", action="store_true",
                        help="Fail if exact loaded GPU slots cannot be linked to first replay")
    args = parser.parse_args()
    specs = json.loads(args.case_results.read_text(encoding="utf-8"))
    events = read_events(args.harness) + translate_trace(args.trace)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_events(args.out_dir / "normalized_events.jsonl", sorted(events, key=lambda row: row.ts_ns))
    result = analyze_timing(events, specs)
    result["run_id"] = args.run_id
    result["slot_lineage_required"] = args.require_slot_proof
    result["limitations"] = []
    result["source_paths"] = {"harness": str(args.harness), "backend_trace": str(args.trace)}
    with args.trace.open(encoding="utf-8") as handle:
        evidence = [item for line in handle if line.strip()
                    if (item := normalize_lifecycle_evidence(json.loads(line))) is not None]
    for case in result["cases"]:
        session = case["session_id"]
        replay_id = f"{session}-replay"
        session_evidence = [item for item in evidence if item.session_id == session]
        proof = assess_loaded_match(
            (item for item in session_evidence if item.signal_id == "kv.load_gpu"),
            (item for item in session_evidence if item.signal_id == "kv.prefix_match"
             and item.request_id == replay_id),
            (item for item in evidence if item.signal_id == "kv.evict_gpu"),
        )
        case["semantic_loads"] = sum(item.signal_id == "kv.load_gpu" for item in session_evidence)
        case.update(proof)
        if not case["semantic_loads"]:
            case["status"] = "failed"
            case["failures"].append("native semantic load evidence is missing")
            result["failures"].append(f"{session}: native semantic load evidence is missing")
        if proof["loaded_slots_match_status"] != "supported":
            message = f"{session}: exact loaded-slot lineage was not proved"
            if args.require_slot_proof:
                case["status"] = "failed"
                case["failures"].append(message)
                result["failures"].append(message)
            else:
                result["limitations"].append(message)
    if result["failures"]:
        result["status"] = "failed"
        for pair in result["pairs"]:
            pair["comparable"] = False
            pair["comparability_reasons"].append("the cache-slot evidence gate failed")
            pair["task_comparable"] = False
            pair["task_comparability_reasons"].append("the cache-slot evidence gate failed")
            for key in ("late_minus_early_first_token_after_due_ms", "late_minus_early_replay_ttft_ms",
                        "late_minus_early_submission_after_due_ms", "late_minus_early_task_latency_ms",
                        "late_minus_early_post_tool_duration_ms"):
                pair[key] = None
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failures": result["failures"],
                      "pairs": result["pairs"]}, indent=2))
    if result["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
