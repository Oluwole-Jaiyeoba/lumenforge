"""Join paired-swap measurements to native replay cache evidence."""

import argparse
import gzip
import json
from pathlib import Path

from agentic_backends.coordinated_audit import replay_prefix_match


def first_matches(path: Path) -> dict:
    matches = {}
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            match = replay_prefix_match(row)
            if match and match[0] not in matches:
                matches[match[0]] = match[1]
    return matches


def measured_controls(case: dict) -> dict:
    """Summarize host-observed controls, excluding setup and final inspection."""
    controls = [c for c in case["controls"] if case["started_ns"] <= c["sent_ns"] < case["ended_ns"]]
    wall, counts = {}, {}
    for control in controls:
        action = control["action"]
        wall[action] = wall.get(action, 0) + (control["returned_ns"] - control["sent_ns"]) / 1e6
        counts[action] = counts.get(action, 0) + 1
    load_ids = {c["result"]["load_id"] for c in controls
                if c["action"] in {"prepare", "prepare_group"} and c["result"].get("load_id")}
    intervals = {}
    for control in controls:
        result = control["result"]
        load_id = result.get("load_id")
        duration = result.get("cuda_elapsed_ms")
        if control["action"] == "load_status" and load_id in load_ids and isinstance(duration, (int, float)):
            intervals[load_id] = duration
    return {"control_wall_ms_by_action": wall, "control_count_by_action": counts,
            "native_cuda_interval_count": len(intervals),
            "native_cuda_interval_total_ms": sum(intervals.values()) if intervals else None}


def analyze(root: Path, trials: list[int], modes: list[str]) -> dict:
    arms, issues, cases = [], [], []
    for trial in trials:
        for mode in modes:
            folder = root / "arms" / f"trial{trial}_{mode}"
            path = folder / "case_results.json"
            if not path.exists():
                issues.append(f"Missing {folder.name}")
                continue
            case = json.loads(path.read_text())
            case["config"].setdefault("restore_style", "group" if any(
                c["action"] == "prepare_group" for c in case["controls"]) else "serial")
            case["config"].setdefault("trace_profile", "kv_lifecycle_lean")
            case["config"].setdefault("control_style", "individual")
            cases.append(case)
            if case["status"] != "complete":
                issues.append(f"{folder.name}: {case.get('error')}")
                continue
            trace = folder / "backend_trace.jsonl.gz"
            if not trace.exists():
                trace = folder / "backend_trace.jsonl"
            matches = first_matches(trace)
            replays = case["turns"]
            if len(replays) != case["config"]["sessions"] * case["config"]["turns"]:
                issues.append(f"{folder.name}: incomplete replay count")
            matched = [matches.get(r["request_id"]) for r in replays]
            if not all(matched):
                issues.append(f"{folder.name}: missing native cache lookup evidence")
            reuse = [m["gpu_tokens"] / r["prompt_tokens"] for r, m in zip(replays, matched) if m]
            if mode in {"coordinated", "resident"} and (not reuse or min(reuse) < .90):
                issues.append(f"{folder.name}: at least one replay reused less than 90% of its prefix on GPU")
            releases = [c for c in case["controls"] if c["action"] == "release_prefix" and c["result"].get("ok")]
            releases += [{"sent_ns": c["sent_ns"], "result": member}
                         for c in case["controls"] if c["action"] == "release_group"
                         for member in c["result"].get("members", []) if member.get("ok")]
            loads = [c for c in case["controls"] if c["action"] in {"prepare", "prepare_group"} and c["result"].get("load_id")]
            measured_loads = [c for c in loads if c["sent_ns"] >= case["started_ns"]]
            restore_readiness = []
            prefetch_readiness = []
            with (folder / "harness_events.jsonl").open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("event") == "coordinated.restore_ready" and row["slot"] >= 0:
                        restore_readiness.append(row["ready_ns"] <= row["tool_due_ns"])
                    if row.get("event") == "coordinated.prefetch_ready":
                        prefetch_readiness.append(row)
            readiness_count = len(prefetch_readiness) if prefetch_readiness else len(restore_readiness)
            ready_margins = [(r["tool_due_ns"] - r["ready_ns"]) / 1e6 for r in prefetch_readiness]
            on_time = (sum(r["ready_ns"] <= r["tool_due_ns"] for r in prefetch_readiness)
                       if prefetch_readiness else sum(restore_readiness))
            arms.append({"trial": trial, "mode": mode, "started_ns": case["started_ns"],
                         "config": case["config"], **case["metrics"],
                         "setup_ms": case["setup_ms"], **measured_controls(case),
                         "minimum_gpu_prefix_fraction": min(reuse) if reuse else None,
                         "native_matches": sum(m is not None for m in matched),
                         "measured_load_count": len(measured_loads),
                         "measured_loaded_tokens": sum(c["result"]["loaded_tokens"] for c in measured_loads),
                         "measured_released_tokens": sum(c["result"]["evicted_tokens"] for c in releases
                                                         if c["sent_ns"] >= case["started_ns"]),
                         "restores_before_due": sum(restore_readiness),
                         "restores_after_due": len(restore_readiness) - sum(restore_readiness),
                         "kv_ready_before_due": on_time,
                         "kv_ready_after_due": readiness_count - on_time,
                         "mean_kv_ready_margin_ms": (sum(ready_margins) / len(ready_margins)
                                                     if ready_margins else None),
                         "already_resident_replays": sum(
                             r.get("source") == "already_resident" for r in prefetch_readiness),
                         "correctness": case.get("correctness"),
                         "per_session_completion_ms": {
                             session: (max(r["request_end_ns"] for r in replays if r["session_id"] == session)
                                       - case["started_ns"]) / 1e6 for session in sorted({r["session_id"] for r in replays})}})
    comparisons = []
    for trial in trials:
        by_mode = {a["mode"]: a for a in arms if a["trial"] == trial}
        if "coordinated" not in by_mode:
            continue
        for reference in ("independent", "resident"):
            if reference in by_mode:
                a, b = by_mode["coordinated"], by_mode[reference]
                comparisons.append({"trial": trial, "reference": reference,
                                    "workload_change_pct": 100 * (a["workload_ms"] / b["workload_ms"] - 1),
                                    "due_to_first_token_change_ms": a["mean_due_to_first_token_ms"] - b["mean_due_to_first_token_ms"],
                                    "sessions_finished_sooner": sum(a["per_session_completion_ms"][s] < value
                                        for s, value in b["per_session_completion_ms"].items()),
                                    "sessions_finished_later": sum(a["per_session_completion_ms"][s] > value
                                        for s, value in b["per_session_completion_ms"].items())})
    return {"schema": "agentic_work_audit.coordinated_swap.summary.v1", "run_id": root.name,
            "status": "blocked" if issues else "complete", "issues": issues, "arms": arms,
            "started_ns": min((c.get("started_ns", c["setup_started_ns"]) for c in cases), default=None),
            "configuration": cases[0]["config"] if cases else {}, "comparisons": comparisons,
            "failures": issues,
            "research_question": "Can ideally coordinated independent sessions hide real CPU/GPU KV swaps?",
            "limitations": ["Optimistic paired timeline; not a fair causal comparison of grouping alone.",
                       "Twenty separate synthetic contexts; fixed tool-result text and forced output length.",
                       "Setup excluded and reported separately; measured control, swap and alignment delays included.",
                       "Matched GPU prefix is reuse evidence, not an independent tensor-level proof of every copy.",
                       "No storage, semantic priorities, or shared model context."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-trials", nargs="+", type=int, required=True)
    parser.add_argument("--expected-modes", nargs="+", required=True)
    args = parser.parse_args()
    summary = analyze(args.run_root, args.expected_trials, args.expected_modes)
    (args.run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "issues": summary["issues"], "comparisons": summary["comparisons"]}, indent=2))
    if summary["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
