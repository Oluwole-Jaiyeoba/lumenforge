"""Compare equal agent sessions across GPU, host, and storage KV residency."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import context, make_prompt, prompt_hash
from .run_sustained_decode_kv_overlap import stream_decode


def tool_result(session_id: str, turn: int, words: int) -> str:
    marker = f"\nTool result for {session_id}, call {turn}:"
    return marker + " " + " ".join([f"result-{turn}"] * words)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


async def inspect_residency(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    session: dict[str, Any],
) -> dict[str, Any]:
    identity = {
        "session_id": session["session_id"],
        "prefix_id": session["prefix_id"],
        "request_id": session["previous_request_id"],
        "prompt_hash": prompt_hash(session["prompt"]),
        "control_timeout_ms": 15_000,
    }
    action = "storage_status" if args.mode == "storage" else "prefix_residency"
    started_ns = time.time_ns()
    response = await client.post(args.control_url, json={"action": action, **identity})
    result = response.json()
    result.update({
        "action": action,
        "inspection_started_ns": started_ns,
        "inspection_finished_ns": time.time_ns(),
        "http_status": response.status_code,
    })
    if not result.get("ok"):
        raise RuntimeError(f"{session['session_id']}: residency inspection failed: {result}")
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sessions = []
    workload_namespace = getattr(args, "workload_namespace", None)
    for index in range(args.sessions):
        session_id = (f"{workload_namespace}-s{index:02d}" if workload_namespace
                      else f"tier-{args.pattern}-{args.mode}-s{index:02d}")
        prompt_key = (f"{workload_namespace}-private-{index:02d}" if workload_namespace
                      else f"memory-tier-private-{index:02d}")
        prompt = make_prompt(prompt_key, args.initial_tokens)
        sessions.append({
            "session_id": session_id,
            "prefix_id": session_id,
            "prompt": prompt,
            "previous_request_id": "",
            "turns": [],
        })

    started_ns = time.time_ns()
    result: dict[str, Any] = {
        "schema": "agentic_work_audit.memory_tier.arm.v1",
        "run_id": args.run_id,
        "mode": args.mode,
        "pattern": args.pattern,
        "seed": args.seed,
        "status": "running",
        "started_ns": started_ns,
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "sessions": sessions,
        "rounds": [],
    }

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0), limits=httpx.Limits(max_connections=32)
    ) as client:
        request_slots = asyncio.Semaphore(args.max_inflight)

        async def submit(
            session: dict[str, Any], turn: int, due_ns: int | None, *, max_tokens: int | None = None,
        ) -> dict[str, Any]:
            request_id = f"{session['session_id']}-turn{turn:02d}"
            phase = "memory_tier_initial" if turn == 0 else "memory_tier_replay"
            async with request_slots:
                submitted_ns = time.time_ns()
                decoded = await stream_decode(
                    client,
                    base_url=args.base_url,
                    model=args.model,
                    prompt=session["prompt"],
                    request_context=context(
                        session_id=session["session_id"],
                        prefix_id=session["prefix_id"],
                        phase=phase,
                        request_id=request_id,
                        p_hash=prompt_hash(session["prompt"]),
                    ),
                    max_tokens=max_tokens or args.decode_tokens,
                    warmup_chunks=1,
                    warmup_ready=asyncio.Event(),
                )
            if not decoded["chunk_times_ns"]:
                raise RuntimeError(f"{request_id}: no first token")
            first_token_ns = decoded["chunk_times_ns"][0]
            row = {
                "session_id": session["session_id"],
                "request_id": request_id,
                "turn": turn,
                "tool_due_ns": due_ns,
                "submitted_ns": submitted_ns,
                "submission_delay_ms": ((submitted_ns - due_ns) / 1e6 if due_ns else 0.0),
                "due_to_first_token_ms": ((first_token_ns - due_ns) / 1e6
                                           if due_ns else None),
                "prompt_tokens": decoded["usage"].get("prompt_tokens"),
                "completion_tokens": decoded["usage"].get("completion_tokens"),
                "ttft_ms": decoded["ttft_ms"],
                "request_start_ns": decoded["request_start_ns"],
                "first_token_ns": first_token_ns,
                "request_end_ns": decoded["request_end_ns"],
                "total_latency_ms": decoded["total_latency_ms"],
            }
            session["previous_request_id"] = request_id
            session["turns"].append(row)
            return row

        # Initial requests establish identical private prefixes in every arm.
        initial_rows = await asyncio.gather(*(
            submit(session, 0, None, max_tokens=getattr(args, "prime_tokens", None))
            for session in sessions
        ))
        result["initial_setup_ms"] = (max(row["request_end_ns"] for row in initial_rows) - started_ns) / 1e6

        for turn in range(1, args.turns + 1):
            previous_end_ns = max(session["turns"][-1]["request_end_ns"] for session in sessions)
            round_anchor_ns = previous_end_ns + args.wait_ms * 1_000_000
            window_ms = args.burst_window_ms if args.pattern == "burst" else args.spread_window_ms
            offsets_ms = [round(index * window_ms / max(1, args.sessions - 1), 3)
                          for index in range(args.sessions)]
            due_by_session = {
                session["session_id"]: round_anchor_ns + int(offsets_ms[index] * 1_000_000)
                for index, session in enumerate(sessions)
            }

            inspect_at_ns = min(due_by_session.values()) - args.inspect_lead_ms * 1_000_000
            await asyncio.sleep(max(0, (inspect_at_ns - time.time_ns()) / 1e9))
            inspections = []
            if not getattr(args, "skip_residency_inspection", False):
                inspections = await asyncio.gather(*(inspect_residency(client, args, session)
                                                      for session in sessions))

            async def replay(session: dict[str, Any], index: int) -> dict[str, Any]:
                due_ns = due_by_session[session["session_id"]]
                await asyncio.sleep(max(0, (due_ns - time.time_ns()) / 1e9))
                session["prompt"] += tool_result(session["session_id"], turn, args.tool_words)
                return await submit(session, turn, due_ns)

            replay_rows = await asyncio.gather(*(replay(session, index)
                                                 for index, session in enumerate(sessions)))
            result["rounds"].append({
                "turn": turn,
                "anchor_due_ns": round_anchor_ns,
                "return_window_ms": window_ms,
                "offsets_ms": offsets_ms,
                "inspections": inspections,
                "replays": replay_rows,
            })

    ended_ns = time.time_ns()
    replay_rows = [row for session in sessions for row in session["turns"] if row["turn"] > 0]
    due_delays = [row["due_to_first_token_ms"] for row in replay_rows]
    ttfts = [row["ttft_ms"] for row in replay_rows]
    result.update({
        "status": "complete",
        "ended_ns": ended_ns,
        "workflow_duration_ms": (max(row["request_end_ns"] for row in replay_rows) - started_ns) / 1e6,
        "replay_count": len(replay_rows),
        "metrics": {
            "mean_due_to_first_token_ms": statistics.mean(due_delays),
            "p95_due_to_first_token_ms": percentile(due_delays, 0.95),
            "total_due_to_first_token_ms": sum(due_delays),
            "mean_ttft_ms": statistics.mean(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95),
            "total_ttft_ms": sum(ttfts),
            "total_submission_delay_ms": sum(row["submission_delay_ms"] for row in replay_rows),
            "output_tokens": sum(row.get("completion_tokens") or 0 for row in replay_rows),
        },
        "per_session_completion_ms": {
            session["session_id"]: (session["turns"][-1]["request_end_ns"] - started_ns) / 1e6
            for session in sessions
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("resident", "host", "storage"), required=True)
    parser.add_argument("--pattern", choices=("spread", "burst"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--initial-tokens", type=int, default=4096)
    parser.add_argument("--tool-words", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=1000)
    parser.add_argument("--burst-window-ms", type=int, default=75)
    parser.add_argument("--spread-window-ms", type=int, default=1000)
    parser.add_argument("--inspect-lead-ms", type=int, default=300)
    parser.add_argument("--max-inflight", type=int, default=6)
    parser.add_argument("--prime-tokens", type=int)
    parser.add_argument("--workload-namespace")
    parser.add_argument("--skip-residency-inspection", action="store_true")
    args = parser.parse_args()
    positive = (args.sessions, args.turns, args.initial_tokens, args.tool_words,
                args.decode_tokens, args.wait_ms, args.max_inflight)
    if min(positive) <= 0 or min(args.burst_window_ms, args.spread_window_ms,
                                 args.inspect_lead_ms) < 0:
        parser.error("workload sizes must be positive and timing windows non-negative")
    if args.max_inflight > args.sessions:
        parser.error("max inflight cannot exceed session count")
    try:
        value = asyncio.run(run(args))
    except Exception as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "case_failure.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, indent=2) + "\n", encoding="utf-8")
        raise
    args.out.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
