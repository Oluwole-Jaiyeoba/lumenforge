#!/usr/bin/env python3
"""Validate the equal-importance contract from a completed Scenario 1 run."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MODES = {"no_prefetch", "controller_ready_time_gpu_backfill"}
REPLAY_PHASES = {"replay", "pressure_filler"}


def validate(run_root: Path, report_dir: Path, expected_replays: int) -> None:
    with (report_dir / "all_replay_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = {row["mode"]: row for row in csv.DictReader(handle)}
    if set(summary) != MODES:
        raise ValueError(f"Scenario 1 report has modes {sorted(summary)}, expected {sorted(MODES)}")
    for mode, row in summary.items():
        if int(row["replays"]) != expected_replays:
            raise ValueError(f"{mode}: {row['replays']} replay rows, expected {expected_replays}")
        if int(row["ttft_samples"]) != expected_replays or int(row["lateness_samples"]) != expected_replays:
            raise ValueError(f"{mode}: replay timing coverage is incomplete")

    observed: Counter[str] = Counter()
    ranked: Counter[str] = Counter()
    for path in run_root.glob("*/m27_trace.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                mode = row.get("mode")
                if row.get("event") != "m27.request.start" or mode not in MODES or row.get("harness") == "instrumentation":
                    continue
                signal = row.get("harness_controller_signal") or {}
                phase = signal.get("phase") or {}
                scheduling = signal.get("scheduling") or {}
                native = signal.get("native_signals") or {}
                request_id = row.get("request_id") or row.get("label")
                if phase.get("work_class") != "peer" or scheduling.get("urgency") != "normal":
                    raise ValueError(f"{mode} {request_id}: unequal harness work class or urgency")
                for field in ("priority_intent", "harness_input_priority_signal"):
                    if row.get(field) or native.get(field):
                        raise ValueError(f"{mode} {request_id}: frontend priority field {field} present")
                if row.get("phase") in REPLAY_PHASES:
                    observed[mode] += 1
                    priority = row.get("sglang_priority")
                    if mode == "no_prefetch" and priority not in (None, ""):
                        raise ValueError(f"{mode} {request_id}: baseline has backend priority {priority}")
                    if mode == "controller_ready_time_gpu_backfill":
                        if priority in (None, ""):
                            raise ValueError(f"{mode} {request_id}: missing derived queue rank")
                        ranked[mode] += 1

    for mode in MODES:
        if observed[mode] != expected_replays:
            raise ValueError(f"{mode}: {observed[mode]} gateway replay requests, expected {expected_replays}")
    if ranked["controller_ready_time_gpu_backfill"] != expected_replays:
        raise ValueError("RTG did not rank every replay")
    print(f"Scenario 1 equal-importance proof passed: {expected_replays} replays per mode; no frontend priority")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-replays", type=int, required=True)
    args = parser.parse_args()
    validate(args.run_root, args.report_dir, args.expected_replays)


if __name__ == "__main__":
    main()
