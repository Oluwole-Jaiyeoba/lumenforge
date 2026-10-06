"""Run matched, equal-priority agent sessions with repeated tool returns."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import completion, context, make_prompt, prompt_hash, write_jsonl
from .run_sustained_decode_kv_overlap import stream_decode


def tool_result(session: str, turn: int, words: int) -> str:
    header = f"\nTool result {turn} for {session}: test output and repository inspection. "
    return header + " ".join(["context"] * words) + "\nAssistant: continue the coding task."


def wait_ms(seed: int, index: int, turn: int, base_ms: int) -> int:
    return base_ms + random.Random(f"{seed}:{index}:{turn}").randrange(0, max(1, base_ms // 4))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "harness_events.jsonl"
    case_id = f"toolcycles-seed{args.seed}"
    active: list[dict[str, Any]] = []
    donors: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=httpx.Limits(max_connections=32)) as client:
        for group, count, initial in ((active, args.active_count, args.initial_tokens),
                                      (donors, args.donor_count, args.donor_initial_tokens)):
            for index in range(count):
                kind = "active" if group is active else "donor"
                session = f"{case_id}-{kind}{index}"
                prompt = make_prompt(session, initial)
                request_id = f"{session}-prime"
                prime = await completion(
                    client, base_url=args.base_url, model=args.model, prompt=prompt,
                    request_context=context(session_id=session, prefix_id=f"{session}-prefix",
                                            phase="audit_initial", request_id=request_id,
                                            p_hash=prompt_hash(prompt)),
                    max_tokens=2,
                )
                group.append({"session_id": session, "prompt": prompt, "prime": prime,
                              "turns": [], "kind": kind})
        started_ns = time.time_ns()
        gate = asyncio.Event()

        async def session_loop(row: dict[str, Any], index: int) -> None:
            await gate.wait()
            is_active = row["kind"] == "active"
            for turn in range(1, args.turns + 1):
                delay = wait_ms(args.seed, index + (0 if is_active else args.active_count),
                                turn, args.wait_ms)
                wait_started_ns = time.time_ns()
                await asyncio.sleep(delay / 1000)
                tool_return_ns = time.time_ns()
                row["prompt"] += tool_result(row["session_id"], turn, args.tool_result_words)
                prompt = row["prompt"]
                request_id = f"{row['session_id']}-turn{turn:02d}"
                context_row = context(session_id=row["session_id"],
                                      prefix_id=f"{row['session_id']}-prefix",
                                      phase="audit_replay", request_id=request_id,
                                      p_hash=prompt_hash(prompt))
                write_jsonl(events_path, {"event": "tool_cycles.tool_return", "ts_ns": tool_return_ns,
                                          "request_id": request_id, "session_id": row["session_id"],
                                          "turn": turn, "wait_started_ns": wait_started_ns,
                                          "planned_wait_ms": delay})
                ready = asyncio.Event()
                result = await stream_decode(
                    client, base_url=args.base_url, model=args.model, prompt=prompt,
                    request_context=context_row,
                    max_tokens=args.decode_tokens if is_active else args.donor_decode_tokens,
                    warmup_chunks=1, warmup_ready=ready,
                )
                if not result["chunk_times_ns"] or not result["usage"].get("prompt_tokens"):
                    raise RuntimeError(f"{request_id}: no first token or prompt-token usage")
                turn_row = {
                    "request_id": request_id, "session_id": row["session_id"],
                    "kind": row["kind"], "turn": turn,
                    "wait_started_ns": wait_started_ns, "tool_return_ns": tool_return_ns,
                    "planned_wait_ms": delay, "prompt_hash": prompt_hash(prompt),
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"].get("completion_tokens"),
                    "request_start_ns": result["request_start_ns"],
                    "first_token_ns": result["chunk_times_ns"][0],
                    "request_end_ns": result["request_end_ns"],
                    "ttft_ms": result["ttft_ms"],
                    "first_token_after_tool_ms": round((result["chunk_times_ns"][0] - tool_return_ns) / 1e6, 3),
                    "completion_after_tool_ms": round((result["request_end_ns"] - tool_return_ns) / 1e6, 3),
                }
                row["turns"].append(turn_row)
                write_jsonl(events_path, {"event": "tool_cycles.replay_complete", "ts_ns": result["request_end_ns"],
                                          **turn_row})

        tasks = [asyncio.create_task(session_loop(row, i)) for i, row in enumerate(active)]
        tasks += [asyncio.create_task(session_loop(row, i)) for i, row in enumerate(donors)]
        gate.set()
        await asyncio.gather(*tasks)
        ended_ns = time.time_ns()
    turns = [turn for row in active + donors for turn in row["turns"]]
    summary = {
        "schema": "agentic_work_audit.tool_cycles.v1", "run_id": args.run_id,
        "research_question_id": args.research_question_id, "seed": args.seed,
        "frontend_priority": "none", "active_count": args.active_count,
        "donor_count": args.donor_count, "turn_count": args.turns,
        "started_ns": started_ns, "ended_ns": ended_ns,
        "workflow_makespan_ms": round((ended_ns - started_ns) / 1e6, 3),
        "active_workflow_makespan_ms": round((max(t["request_end_ns"] for r in active for t in r["turns"])
                                              - started_ns) / 1e6, 3),
        "workload": {"initial_tokens": args.initial_tokens,
                     "donor_initial_tokens": args.donor_initial_tokens,
                     "tool_result_words": args.tool_result_words,
                     "decode_tokens": args.decode_tokens,
                     "donor_decode_tokens": args.donor_decode_tokens,
                     "wait_ms": args.wait_ms},
        "turns": turns,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--research-question-id", default="RQ14")
    parser.add_argument("--active-count", type=int, default=2)
    parser.add_argument("--donor-count", type=int, default=0)
    parser.add_argument("--turns", type=int, default=12)
    parser.add_argument("--initial-tokens", type=int, default=768)
    parser.add_argument("--donor-initial-tokens", type=int, default=512)
    parser.add_argument("--tool-result-words", type=int, default=96)
    parser.add_argument("--decode-tokens", type=int, default=24)
    parser.add_argument("--donor-decode-tokens", type=int, default=8)
    parser.add_argument("--wait-ms", type=int, default=800)
    args = parser.parse_args()
    if (min(args.active_count, args.turns, args.initial_tokens, args.tool_result_words,
            args.decode_tokens, args.donor_initial_tokens, args.donor_decode_tokens,
            args.wait_ms) <= 0 or args.donor_count < 0):
        parser.error("Invalid session or workload size")
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in
                      ("run_id", "active_count", "donor_count", "turn_count", "workflow_makespan_ms")}, indent=2))


if __name__ == "__main__":
    main()
