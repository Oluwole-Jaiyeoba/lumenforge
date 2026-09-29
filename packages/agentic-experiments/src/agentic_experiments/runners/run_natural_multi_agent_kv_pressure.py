#!/usr/bin/env python3
"""Observe natural KV reload overlap in a production-shaped agent workload.

No request is assigned frontend priority and this runner never calls the
prepared-prefix control API. Sessions build context, wait for synthetic tools,
and resume through ordinary chat-completions requests. The companion report
correlates those normal replay intervals with SGLang's native load-back trace.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import completion, make_prompt, prompt_hash, write_jsonl


def request_context(session_id: str, phase: str, request_id: str, p_hash: str) -> dict[str, Any]:
    """Correlation metadata only: this workload deliberately has no priority."""

    metadata = {
        "session_id": session_id,
        "prefix_id": f"{session_id}-prefix",
        "phase": phase,
        "mode": "natural_multi_agent_tool_workload",
        "label": request_id,
        "request_id": request_id,
        "parent_run_id": session_id,
        "correlation_id": f"{session_id}:{phase}:{request_id}",
        "case_id": session_id,
        "prompt_hash": p_hash,
    }
    return {"agentic_kv": metadata, "request_context": {"request_id": request_id, "phase": phase, "case_id": session_id}}


def tool_wait_ms(args: argparse.Namespace, session_index: int, step: int) -> int:
    span = args.tool_wait_max_ms - args.tool_wait_min_ms + 1
    return args.tool_wait_min_ms + ((session_index * 251 + step * 617) % span)


def append_tool_result(prompt: str, session_id: str, step: int) -> str:
    return (
        f"{prompt}\n\nTool result {step + 1} for {session_id}: "
        "the repository check returned new evidence; continue the same coding task."
    )


async def run_session(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    events: Path,
    session_index: int,
) -> list[dict[str, Any]]:
    session_id = f"{args.run_id}-session-{session_index:02d}"
    await asyncio.sleep(session_index * args.initial_stagger_ms / 1000)
    prompt = make_prompt(session_id, args.session_prefix_tokens)
    prompt += "\nFirst turn: inspect the repository and decide which tool to use first."
    rows: list[dict[str, Any]] = []

    initial_id = f"{session_id}-initial"
    initial = await completion(
        client,
        base_url=args.base_url,
        model=args.model,
        prompt=prompt,
        request_context=request_context(session_id, "initial", initial_id, prompt_hash(prompt)),
        max_tokens=args.initial_max_tokens,
    )
    rows.append({"event": "natural_kv.session_initial", "session_id": session_id, "request_id": initial_id, **initial})

    for step in range(args.tool_waits):
        wait_ms = tool_wait_ms(args, session_index, step)
        write_jsonl(
            events,
            {
                "event": "natural_kv.tool_wait_start",
                "session_id": session_id,
                "tool_wait_step": step + 1,
                "tool_wait_ms": wait_ms,
            },
        )
        await asyncio.sleep(wait_ms / 1000)
        prompt = append_tool_result(prompt, session_id, step)
        request_id = f"{session_id}-replay-{step + 1}"
        replay = await completion(
            client,
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            request_context=request_context(session_id, f"replay_{step + 1}", request_id, prompt_hash(prompt)),
            max_tokens=args.replay_tokens,
        )
        row = {
            "event": "natural_kv.replay_complete",
            "session_id": session_id,
            "request_id": request_id,
            "tool_wait_step": step + 1,
            "tool_wait_ms": wait_ms,
            **replay,
        }
        write_jsonl(events, row)
        rows.append(row)
    return rows


def load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"backend trace is missing: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"workload event log is missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def native_loads(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loads: list[dict[str, Any]] = []
    for row in trace_rows:
        if row.get("event") != "hiradix.load_back.end":
            continue
        context = row.get("kv_context") or {}
        if context.get("agent_mode") != "natural_multi_agent_tool_workload":
            continue
        phase = str(context.get("agent_phase") or "")
        if not phase.startswith("replay_"):
            continue
        end_ns = int(row.get("ts_ns") or 0)
        duration_ms = float(row.get("duration_ms") or 0.0)
        loads.append(
            {
                "session_id": context.get("agent_session_id"),
                "request_id": context.get("agent_request_id"),
                "phase": phase,
                "start_ns": end_ns - int(duration_ms * 1_000_000),
                "end_ns": end_ns,
                "duration_ms": round(duration_ms, 3),
                "node_id": context.get("node_id"),
            }
        )
    return loads


def bucket_for_load_count(count: int) -> str:
    if count == 0:
        return "control_like"
    if count <= 20:
        return "low_like"
    if count <= 60:
        return "medium_like"
    return "high_like"


def correlate(replays: list[dict[str, Any]], loads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for replay in replays:
        start_ns = int(replay["request_start_ns"])
        end_ns = int(replay["request_end_ns"])
        # Restores before the first token affect TTFT. This study isolates
        # restores that overlap active decode, which starts at first token.
        decode_start_ns = int(
            replay.get("first_token_ns") or start_ns + int(float(replay["ttft_ms"]) * 1_000_000)
        )
        overlaps = [load for load in loads if decode_start_ns < int(load["end_ns"]) and end_ns > int(load["start_ns"])]
        foreign_overlaps = [load for load in overlaps if load.get("session_id") != replay["session_id"]]
        observations.append(
            {
                "session_id": replay["session_id"],
                "request_id": replay["request_id"],
                "tool_wait_step": replay["tool_wait_step"],
                "tool_wait_ms": replay["tool_wait_ms"],
                "ttft_ms": replay["ttft_ms"],
                "total_decode_ms": replay["total_latency_ms"],
                "natural_reload_count": len(overlaps),
                "cross_session_reload_count": len(foreign_overlaps),
                "natural_reload_sessions": sorted({str(load.get("session_id") or "") for load in overlaps}),
                "cross_session_reload_sessions": sorted({str(load.get("session_id") or "") for load in foreign_overlaps}),
                "natural_reload_trace_duration_ms": round(sum(float(load["duration_ms"]) for load in overlaps), 3),
                "pressure_bucket": bucket_for_load_count(len(overlaps)),
            }
        )
    return observations


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend-trace", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--session-count", type=int, default=8)
    parser.add_argument("--tool-waits", type=int, default=3)
    parser.add_argument("--session-prefix-tokens", type=int, default=8192)
    parser.add_argument("--replay-tokens", type=int, default=384)
    parser.add_argument("--initial-max-tokens", type=int, default=1)
    parser.add_argument("--initial-stagger-ms", type=int, default=75)
    parser.add_argument("--tool-wait-min-ms", type=int, default=400)
    parser.add_argument("--tool-wait-max-ms", type=int, default=2000)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--rescore-events",
        type=Path,
        help="Rebuild the summary from an existing event log and backend trace without sending requests.",
    )
    args = parser.parse_args()
    if min(args.session_count, args.tool_waits, args.session_prefix_tokens, args.replay_tokens) < 1:
        parser.error("session count, tool waits, prefix tokens, and replay tokens must be positive")
    if args.tool_wait_min_ms < 0 or args.tool_wait_max_ms < args.tool_wait_min_ms:
        parser.error("tool wait bounds are invalid")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = args.out_dir / "natural_multi_agent_events.jsonl"
    if args.rescore_events:
        replays = [row for row in load_events(args.rescore_events) if row.get("event") == "natural_kv.replay_complete"]
    else:
        events.write_text("", encoding="utf-8")
        timeout = httpx.Timeout(args.timeout_s, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            completed = await asyncio.gather(
                *(run_session(client, args, events, index) for index in range(args.session_count))
            )
        replays = [row for session_rows in completed for row in session_rows if row["event"] == "natural_kv.replay_complete"]
        await asyncio.sleep(1.0)
    loads = native_loads(load_trace(args.backend_trace))
    observations = correlate(replays, loads)
    buckets = {name: sum(row["pressure_bucket"] == name for row in observations) for name in ("control_like", "low_like", "medium_like", "high_like")}
    summary = {
        "schema_version": "natural_multi_agent_kv_pressure.v1",
        "run_id": args.run_id,
        "frontend_priority": "none",
        "workload": {
            "session_count": args.session_count,
            "tool_waits_per_session": args.tool_waits,
            "session_prefix_tokens": args.session_prefix_tokens,
            "replay_tokens": args.replay_tokens,
            "tool_wait_range_ms": [args.tool_wait_min_ms, args.tool_wait_max_ms],
        },
        "replay_count": len(observations),
        "native_reload_events": len(loads),
        "pressure_buckets": buckets,
        "observations": observations,
        "native_loads": loads,
        "interpretation": "Natural means ordinary replay requests caused the observed SGLang load-back events; no prepared-prefix control requests were issued. Overlap counts begin after each replay's first token, so they describe active decode rather than that replay's own TTFT restore.",
    }
    (args.out_dir / "natural_multi_agent_kv_pressure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    asyncio.run(main_async())
