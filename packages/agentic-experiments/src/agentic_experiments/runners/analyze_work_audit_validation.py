#!/usr/bin/env python3
"""Join host and v0510 backend evidence, then validate the tiny live audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agentic_backends.sglang.audit_v0510 import translate_trace
from agentic_work_audit import analyze_validation, read_events, write_events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-second-replay", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = read_events(args.harness) + translate_trace(args.trace)
    write_events(args.out_dir / "normalized_events.jsonl", sorted(events, key=lambda row: row.ts_ns))
    summary = analyze_validation(events, expected_sessions=(f"{args.run_id}-warm", f"{args.run_id}-host"),
                                 expected_runtime=("0.5.10.post1", "v0510"),
                                 require_second_replay=args.require_second_replay)
    summary["run_id"] = args.run_id
    summary["source_paths"] = {"harness": str(args.harness), "backend_trace": str(args.trace)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "failures": summary["failures"],
                      "cases": summary["cases"]}, indent=2))
    if summary["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
