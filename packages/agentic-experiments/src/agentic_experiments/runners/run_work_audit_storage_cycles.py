"""Run equal-priority, multi-turn sessions with naturally displaced storage KV."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import context, make_prompt, prompt_hash
from .run_sustained_decode_kv_overlap import stream_decode
from .run_work_audit_storage import control, stage_storage
from .run_work_audit_tool_cycles import tool_result


async def prepare_during_wait(
    client: httpx.AsyncClient, args: argparse.Namespace, identity: dict[str, str],
    storage_id: str, due_ns: int,
) -> dict[str, Any]:
    staged = await stage_storage(client, args.control_url, identity, storage_id)
    result: dict[str, Any] = {"stage": staged, "stage_before_due": staged["completed_observed_ns"] <= due_ns}
    return result


async def inspect_during_wait(
    client: httpx.AsyncClient, args: argparse.Namespace, identity: dict[str, str],
    storage_id: str, due_ns: int, inspect_delay_ms: int,
) -> dict[str, Any]:
    await asyncio.sleep(inspect_delay_ms / 1000)
    residency = await control(client, args.control_url, "storage_status", identity)
    storage_candidate = int(residency.get("storage_candidate_tokens") or 0) >= args.page_size
    safe_stage_eligible = int(residency["gpu_tokens"]) == 0 and int(residency["host_tokens"]) == 0
    result: dict[str, Any] = {"residency": residency, "natural_storage_candidate": storage_candidate,
                              "safe_stage_eligible": safe_stage_eligible,
                              "inspection_before_due": time.time_ns() < due_ns,
                              "preparation": None}
    if safe_stage_eligible and args.arm != "on_demand" and result["inspection_before_due"]:
        try:
            result["preparation"] = await prepare_during_wait(client, args, identity, storage_id, due_ns)
        except Exception as exc:
            result["preparation"] = {"error": f"{type(exc).__name__}: {exc}"}
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run_started_ns = time.time_ns()
    sessions: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=httpx.Limits(max_connections=32)) as client:
        for index in range(args.sessions):
            session_id = f"storagecycles-seed{args.seed}-session{index}"
            prompt = make_prompt(session_id, args.initial_tokens)
            sessions.append({"session_id": session_id, "prefix_id": f"{session_id}-prefix",
                             "prompt": prompt, "turns": [], "previous_request_id": ""})
        gate = asyncio.Event()

        async def one_session(row: dict[str, Any], index: int) -> None:
            await gate.wait()
            pending: list[tuple[asyncio.Task[dict[str, Any]], dict[str, Any]]] = []
            if index:
                await asyncio.sleep(index * args.stagger_ms / 1000)
            for turn in range(args.turns + 1):
                session_id = row["session_id"]
                if turn:
                    delay_ms = args.wait_ms + random.Random(
                        f"{args.seed}:{index}:{turn}"
                    ).randrange(0, args.wait_spread_ms + 1)
                    wait_started_ns = time.time_ns()
                    due_ns = wait_started_ns + delay_ms * 1_000_000
                    prior_identity = {
                        "session_id": session_id, "prefix_id": row["prefix_id"],
                        "request_id": row["previous_request_id"],
                        "prompt_hash": prompt_hash(row["prompt"]),
                    }
                    inspection_task = asyncio.create_task(inspect_during_wait(
                        client, args, prior_identity,
                        f"{session_id}-turn{turn:02d}-storage", due_ns,
                        int(delay_ms * args.inspect_fraction),
                    ))
                    await asyncio.sleep(max(0, (due_ns - time.time_ns()) / 1e9))
                    tool_return_ns = time.time_ns()
                    row["prompt"] += tool_result(session_id, turn, args.tool_result_words)
                else:
                    wait_started_ns = due_ns = tool_return_ns = None
                    inspection_task = None
                request_id = f"{session_id}-turn{turn:02d}"
                prompt = row["prompt"]
                result = await stream_decode(
                    client, base_url=args.base_url, model=args.model, prompt=prompt,
                    request_context=context(session_id=session_id, prefix_id=row["prefix_id"],
                                            phase="storage_cycles_replay" if turn else "storage_cycles_initial",
                                            request_id=request_id, p_hash=prompt_hash(prompt)),
                    max_tokens=args.decode_tokens, warmup_chunks=1, warmup_ready=asyncio.Event(),
                )
                if not result["chunk_times_ns"] or not result["usage"].get("prompt_tokens"):
                    raise RuntimeError(f"{request_id}: missing first token or usage")
                turn_row = {
                    "request_id": request_id, "session_id": session_id, "turn": turn,
                    "wait_started_ns": wait_started_ns, "due_ns": due_ns,
                    "tool_return_ns": tool_return_ns, "residency_during_wait": None,
                    "natural_storage_candidate": False, "inspection_before_due": None,
                    "safe_stage_eligible": False,
                    "preparation": None,
                    "prompt_tokens": result["usage"]["prompt_tokens"],
                    "completion_tokens": result["usage"].get("completion_tokens"),
                    "request_start_ns": result["request_start_ns"],
                    "first_token_ns": result["chunk_times_ns"][0],
                    "request_end_ns": result["request_end_ns"],
                    "ttft_ms": result["ttft_ms"],
                    "due_to_first_token_ms": round((result["chunk_times_ns"][0] - due_ns) / 1e6, 3)
                    if due_ns is not None else None,
                }
                row["turns"].append(turn_row)
                row["previous_request_id"] = request_id
                if inspection_task is not None:
                    pending.append((inspection_task, turn_row))

            for inspection_task, turn_row in pending:
                try:
                    inspected = await inspection_task
                    turn_row["residency_during_wait"] = inspected["residency"]
                    turn_row["natural_storage_candidate"] = inspected["natural_storage_candidate"]
                    turn_row["safe_stage_eligible"] = inspected["safe_stage_eligible"]
                    turn_row["inspection_before_due"] = inspected["inspection_before_due"]
                    turn_row["preparation"] = inspected["preparation"]
                except Exception as exc:  # Preserve replay data while exposing the control failure.
                    turn_row["preparation"] = {"error": f"{type(exc).__name__}: {exc}"}

        tasks = [asyncio.create_task(one_session(row, index)) for index, row in enumerate(sessions)]
        gate.set()
        await asyncio.gather(*tasks)
    run_ended_ns = time.time_ns()
    turns = [item for row in sessions for item in row["turns"]]
    replay_turns = [item for item in turns if item["turn"] > 0]
    return {
        "schema": "agentic_work_audit.storage_cycles.v1", "run_id": args.run_id,
        "arm": args.arm, "seed": args.seed, "frontend_priority": "equal",
        "session_count": args.sessions, "turn_count_per_session": args.turns,
        "workload": {"initial_tokens": args.initial_tokens, "tool_result_words": args.tool_result_words,
                     "decode_tokens": args.decode_tokens, "wait_ms": args.wait_ms,
                     "wait_spread_ms": args.wait_spread_ms,
                     "stagger_ms": args.stagger_ms, "inspect_fraction": args.inspect_fraction},
        "started_ns": run_started_ns, "ended_ns": run_ended_ns,
        "workflow_duration_ms": round((max(item["request_end_ns"] for item in turns) - run_started_ns) / 1e6, 3),
        "collection_wall_ms": round((run_ended_ns - run_started_ns) / 1e6, 3),
        "natural_storage_candidate_waits": sum(item["natural_storage_candidate"] for item in replay_turns),
        "safe_stage_eligible_waits": sum(item["safe_stage_eligible"] for item in replay_turns),
        "inspections_before_due": sum(bool(item["inspection_before_due"]) for item in replay_turns),
        "stage_before_due_count": sum(bool((item["preparation"] or {}).get("stage_before_due"))
                                      for item in replay_turns),
        "sessions": [{"session_id": row["session_id"],
                      "completion_ms": round((row["turns"][-1]["request_end_ns"] - run_started_ns) / 1e6, 3)}
                     for row in sessions],
        "turns": turns,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=("on_demand", "host_stage"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--initial-tokens", type=int, default=2048)
    parser.add_argument("--tool-result-words", type=int, default=900)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=1000)
    parser.add_argument("--wait-spread-ms", type=int, default=3000)
    parser.add_argument("--stagger-ms", type=int, default=250)
    parser.add_argument("--inspect-fraction", type=float, default=0.4)
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.sessions, args.turns, args.initial_tokens, args.tool_result_words,
           args.decode_tokens, args.wait_ms, args.page_size) <= 0 or min(args.stagger_ms, args.wait_spread_ms) < 0 or not 0 <= args.inspect_fraction < 1:
        parser.error("workload sizes and wait must be positive")
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        (args.out.parent / "case_failure.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        }, indent=2) + "\n", encoding="utf-8")
        raise
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "run_id", "arm", "seed", "workflow_duration_ms", "natural_storage_candidate_waits",
        "stage_before_due_count",
    )}))


if __name__ == "__main__":
    main()
