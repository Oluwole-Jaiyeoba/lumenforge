#!/usr/bin/env python3
"""Compare pre-return worker loads with and without short-replay overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from agentic_backends.sglang.audit_v0510 import translate_trace
from agentic_work_audit import read_events, write_events
from agentic_work_audit.multisession import analyze_multisession


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _requests(row: dict[str, Any]) -> list[dict[str, Any]]:
    context = row.get("kv_context") or {}
    batch = context.get("batch") or {}
    return batch.get("requests") or context.get("requests") or []


def _has_request(row: dict[str, Any], request_id: str) -> bool:
    return any(request.get("agent_request_id") == request_id or
               request.get("request_id") == request_id for request in _requests(row))


def _short_decode(case: dict[str, Any], trace: list[dict[str, Any]]) -> dict[str, Any]:
    request_id = f"{case['case_id']}-short-replay"
    replay = case["short"]["replay"]
    first = int(replay["first_token_ns"])
    finish = int(replay["request_end_ns"])
    starts: dict[str, dict[str, Any]] = {}
    forward_starts: dict[str, dict[str, Any]] = {}
    batches = []
    forwards = []
    steps = []
    for row in trace:
        event = row.get("event")
        if event == "scheduler.run_batch.start":
            starts[str(row.get("call_id"))] = row
        elif event == "scheduler.run_batch.end":
            start = starts.pop(str(row.get("call_id")), None)
            if start and _has_request(start, request_id):
                batches.append({"start_ns": start["ts_ns"], "end_ns": row["ts_ns"],
                                "duration_ms": round((row["ts_ns"] - start["ts_ns"]) / 1e6, 3),
                                "forward_mode": (start.get("kv_context") or {}).get("batch", {}).get("forward_mode")})
        elif event == "worker.forward_batch_generation.start":
            forward_starts[str(row.get("call_id"))] = row
        elif event == "worker.forward_batch_generation.end":
            start = forward_starts.pop(str(row.get("call_id")), None)
            if start:
                forwards.append({"start_ns": start["ts_ns"], "end_ns": row["ts_ns"]})
        elif event == "scheduler.process_batch_result_decode.end" and _has_request(row, request_id):
            request = next(item for item in _requests(row)
                           if item.get("agent_request_id") == request_id or item.get("request_id") == request_id)
            output_ids = request.get("output_ids") or {}
            steps.append({"ts_ns": row["ts_ns"], "output_ids_count": output_ids.get("count")})
    batches = sorted((row for row in batches if first <= row["end_ns"] <= finish),
                     key=lambda row: row["start_ns"])
    for batch in batches:
        nested = [row for row in forwards if batch["start_ns"] <= row["start_ns"] and
                  row["end_ns"] <= batch["end_ns"]]
        batch["model_forward_ms"] = round(sum((row["end_ns"] - row["start_ns"]) / 1e6
                                              for row in nested), 3) if nested else None
        batch["non_forward_ms"] = (round(batch["duration_ms"] - batch["model_forward_ms"], 3)
                                   if nested else None)
    steps = sorted((row for row in steps if first <= row["ts_ns"] <= finish),
                   key=lambda row: row["ts_ns"])
    gaps = [round((right["start_ns"] - left["end_ns"]) / 1e6, 3)
            for left, right in zip(batches, batches[1:])]
    chunks = replay.get("content_chunks") or []
    output_counts = [row["output_ids_count"] for row in steps
                     if isinstance(row["output_ids_count"], int)]
    return {
        "request_id": request_id,
        "content_chunk_count": len(chunks),
        "content_intervals_ms": [round((right["ts_ns"] - left["ts_ns"]) / 1e6, 3)
                                 for left, right in zip(chunks, chunks[1:])],
        "completion_tokens": replay.get("completion_tokens"),
        "output_characters": replay.get("output_characters"),
        "output_sha256": replay.get("output_sha256"),
        "decode_batches": batches,
        "decode_steps": steps,
        "final_output_ids_count": max(output_counts) if output_counts else None,
        "inter_batch_gaps_ms": gaps,
        "first_token_ns": first,
        "finish_ns": finish,
    }


def _overlap_metrics(case: dict[str, Any], short: dict[str, Any]) -> dict[str, Any]:
    load = case["long"]["load_status"]
    start = load.get("worker_started_ns")
    finish = load.get("committed_ns")
    if not isinstance(start, int) or not isinstance(finish, int) or finish <= start:
        return {"bounds_available": False}
    batches = short["decode_batches"]
    overlapping = [row for row in batches if row["start_ns"] < finish and row["end_ns"] > start]
    nonoverlapping = [row for row in batches if row not in overlapping]
    def med(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        return round(median(values), 3) if values else None
    return {
        "bounds_available": True,
        "worker_started_ns": start,
        "load_committed_ns": finish,
        "worker_active_ms": round((finish - start) / 1e6, 3),
        "short_decode_overlap_ms": round(max(0, min(short["finish_ns"], finish) -
                                              max(short["first_token_ns"], start)) / 1e6, 3),
        "overlapping_batch_count": len(overlapping),
        "overlapping_batch_median_ms": med(overlapping, "duration_ms"),
        "nonoverlapping_batch_median_ms": med(nonoverlapping, "duration_ms"),
        "overlapping_forward_median_ms": med(overlapping, "model_forward_ms"),
        "nonoverlapping_forward_median_ms": med(nonoverlapping, "model_forward_ms"),
        "overlapping_non_forward_median_ms": med(overlapping, "non_forward_ms"),
        "nonoverlapping_non_forward_median_ms": med(nonoverlapping, "non_forward_ms"),
        "cuda_elapsed_ms": load.get("cuda_elapsed_ms"),
    }


def analyze(run_id: str, raw_cases: list[dict[str, Any]], trace: list[dict[str, Any]],
            events: list[Any], *, require_model_forward: bool = False) -> dict[str, Any]:
    analyzed = []
    for raw in raw_cases:
        case_id = raw["case_id"]
        case_events = [row for row in events if row.session_id.startswith(f"{case_id}-") or
                       row.kind in ("runtime_hooks", "hook_error")]
        audit = analyze_multisession(case_events, case_id,
                                     expected_runtime=("0.5.10.post1", "v0510"),
                                     condition=raw["condition"], require_end_eviction=True)
        short = _short_decode(raw, trace)
        analyzed.append({"case_id": case_id, "condition": raw["condition"],
                         "pair": raw["pair"], "warmup": raw.get("warmup", False),
                         "audit": audit, "short_decode": short,
                         "load_overlap": _overlap_metrics(raw, short)})
    measured = [row for row in analyzed if not row["warmup"]]
    pairs = []
    failures = []
    for number in sorted({row["pair"] for row in measured}):
        modes = {row["condition"]: row for row in measured if row["pair"] == number}
        early, post = modes.get("early"), modes.get("post_short")
        reasons = []
        if len([row for row in measured if row["pair"] == number]) != 2 or not early or not post:
            reasons.append("pair must contain exactly one early and one post_short case")
        else:
            for item in (early, post):
                if item["audit"]["status"] != "validated":
                    reasons.append(f"{item['condition']} lifecycle gate failed")
                if len(item["short_decode"]["decode_batches"]) < 2:
                    reasons.append(f"{item['condition']} lacks request-linked decode batches")
                if require_model_forward and sum(row["model_forward_ms"] is not None for row in
                                                 item["short_decode"]["decode_batches"]) < 2:
                    reasons.append(f"{item['condition']} lacks request-linked model forwards")
                if len(item["short_decode"]["decode_steps"]) < 2:
                    reasons.append(f"{item['condition']} lacks request-linked decode steps")
                if not item["short_decode"]["content_chunk_count"]:
                    reasons.append(f"{item['condition']} lacks response content chunks")
                if not item["load_overlap"]["bounds_available"]:
                    reasons.append(f"{item['condition']} lacks worker load bounds")
            if early["short_decode"]["content_chunk_count"] != post["short_decode"]["content_chunk_count"]:
                reasons.append("response content-chunk counts differ; output-token comparability unknown")
            if early["short_decode"]["completion_tokens"] is not None and post["short_decode"]["completion_tokens"] is not None and early["short_decode"]["completion_tokens"] != post["short_decode"]["completion_tokens"]:
                reasons.append("completion token counts differ")
            if (early["short_decode"]["final_output_ids_count"] is not None and
                    post["short_decode"]["final_output_ids_count"] is not None and
                    early["short_decode"]["final_output_ids_count"] !=
                    post["short_decode"]["final_output_ids_count"]):
                reasons.append("backend output-token counts differ")
            if early["load_overlap"].get("short_decode_overlap_ms", 0) <= 0:
                reasons.append("early worker load did not overlap short decode")
            if post["load_overlap"].get("short_decode_overlap_ms", 0) > 0:
                reasons.append("post-short worker load overlapped short decode")
            for label in ("short", "long"):
                delta = abs(early["audit"]["sessions"][label]["observed_tool_wait_ms"] -
                            post["audit"]["sessions"][label]["observed_tool_wait_ms"])
                if delta > 100:
                    reasons.append(f"{label} tool waits differ by {delta:.1f} ms")
        result = {"pair": number, "comparable": not reasons, "reasons": reasons,
                  "early_case_id": early["case_id"] if early else None,
                  "post_short_case_id": post["case_id"] if post else None}
        if early and post:
            result.update({
                "early_short_first_token_ms": early["audit"]["sessions"]["short"]["first_token_after_tool_ms"],
                "post_short_first_token_ms": post["audit"]["sessions"]["short"]["first_token_after_tool_ms"],
                "early_short_finish_ms": early["audit"]["sessions"]["short"]["completion_after_tool_ms"],
                "post_short_finish_ms": post["audit"]["sessions"]["short"]["completion_after_tool_ms"],
                "early_long_first_token_ms": early["audit"]["sessions"]["long"]["first_token_after_tool_ms"],
                "post_short_long_first_token_ms": post["audit"]["sessions"]["long"]["first_token_after_tool_ms"],
                "early_workflow_ms": early["audit"]["workflow_makespan_ms"],
                "post_short_workflow_ms": post["audit"]["workflow_makespan_ms"],
            })
        if reasons:
            failures.append(f"pair {number}: {', '.join(reasons)}")
        pairs.append(result)
    if any(row["audit"]["status"] != "validated" for row in analyzed if row["warmup"]):
        failures.append("warmup lifecycle gate failed")
    if not pairs:
        failures.append("no measured pairs")
    return {
        "schema": "agentic_work_audit.decode_overlap.v1", "run_id": run_id,
        "model_forward_required": require_model_forward,
        "status": "validated" if not failures else "failed", "failures": failures,
        "pairs": pairs, "cases": measured,
        "warmup_cases": [row for row in analyzed if row["warmup"]],
        "limitations": [
            "Worker start-to-commit is an upper bound on GPU-copy activity, not exact HBM overlap.",
            "Client stream content chunks need not equal model tokens; backend decode steps are separate evidence.",
            "Batch wall time includes CPU submission and synchronization; it is not GPU kernel time.",
            "A timing difference is not direct proof of bandwidth contention or recoverable hardware work.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--case-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-model-forward", action="store_true")
    args = parser.parse_args()
    events = sorted(read_events(args.harness) + translate_trace(args.trace), key=lambda row: row.ts_ns)
    write_events(args.out_dir / "normalized_events.jsonl", events)
    summary = analyze(args.run_id, json.loads(args.case_results.read_text(encoding="utf-8")),
                      _rows(args.trace), events,
                      require_model_forward=args.require_model_forward)
    summary["source_paths"] = {"harness": str(args.harness), "backend_trace": str(args.trace)}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "failures": summary["failures"],
                      "pairs": summary["pairs"]}, indent=2))
    if summary["status"] != "validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
