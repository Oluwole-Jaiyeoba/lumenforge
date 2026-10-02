#!/usr/bin/env python3
"""Join concurrent harness and pinned-backend evidence into one audit timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_backends.sglang.audit_v0510 import translate_trace
from agentic_work_audit import read_events, write_events
from agentic_work_audit.multisession import (
    analyze_multisession, compare_controller_windows, compare_multisession_pairs,
    compare_multisession_windows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--case-results", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    events = sorted(read_events(args.harness) + translate_trace(args.trace), key=lambda row: row.ts_ns)
    write_events(args.out_dir / "normalized_events.jsonl", events)
    if args.case_results:
        raw_cases = json.loads(args.case_results.read_text(encoding="utf-8"))
        cases = []
        for case in raw_cases:
            case_id = case["case_id"]
            case_events = [row for row in events if row.session_id.startswith(f"{case_id}-") or
                           row.kind in ("runtime_hooks", "hook_error")]
            result = analyze_multisession(
                case_events, case_id, expected_runtime=("0.5.10.post1", "v0510"),
                condition=case["condition"], require_end_eviction=True,
            )
            result["pair"] = int(case["pair"])
            result["warmup"] = bool(case.get("warmup"))
            cases.append(result)
        measured = [case for case in cases if not case["warmup"]]
        warmups = [case for case in cases if case["warmup"]]
        compare = (compare_controller_windows if any(case["load_timing"] == "controller_window"
                                                     for case in measured) else
                   compare_multisession_windows if any(case["load_timing"] == "post_short"
                                                       for case in measured) else
                   compare_multisession_pairs)
        summary = compare(measured, args.run_id)
        summary["cases"] = measured
        summary["warmup_cases"] = warmups
        if any(case["status"] != "validated" for case in warmups):
            summary["status"] = "failed"
            summary["failures"].append("a warmup case failed its evidence gate")
    else:
        summary = analyze_multisession(events, args.run_id, expected_runtime=("0.5.10.post1", "v0510"))
    summary["source_paths"] = {"harness": str(args.harness), "backend_trace": str(args.trace)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "failures": summary["failures"],
                      "comparable_pairs": summary.get("comparable_pairs"),
                      "sessions": summary.get("sessions")}, indent=2))
    if summary["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
