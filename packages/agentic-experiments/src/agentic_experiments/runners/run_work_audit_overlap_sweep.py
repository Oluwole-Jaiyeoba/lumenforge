"""Run one equal-priority multi-session KV-load overlap dose."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import (
    completion, context, make_prompt, prepare_prefix, prompt_hash, replay_prompt,
    stage_host_resident_prefix, write_jsonl,
)
from .run_sustained_decode_kv_overlap import stream_decode
from .run_work_audit_timing import confirm_load


def _duration_ms(start_ns: int, end_ns: int) -> float:
    return round((end_ns - start_ns) / 1e6, 3)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.out_dir / "harness_events.jsonl"
    case_id = f"rq11-seed{args.seed}"
    started_ns = time.time_ns()
    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0), limits=httpx.Limits(max_connections=32)) as client:
        donors = []
        for index in range(args.donor_count):
            donor_id = f"{case_id}-donor{index}"
            plan, session, prefix, p_hash = await stage_host_resident_prefix(
                client, args, event_path, donor_id)
            donors.append({"session_id": session, "prefix_id": prefix, "prompt_hash": p_hash,
                           "host_tokens": int(plan.get("host_tokens") or 0),
                           "prime_request_id": f"{donor_id}-donor-prime",
                           "prompt": make_prompt(session, args.donor_prompt_tokens)})
        active_count = args.session_count - args.donor_count
        active = []
        for index in range(active_count):
            session = f"{case_id}-active{index}"
            prompt = make_prompt(session, args.active_prompt_tokens)
            p_hash = prompt_hash(prompt)
            prime_id = f"{session}-prime"
            await completion(
                client, base_url=args.base_url, model=args.model, prompt=prompt,
                request_context=context(session_id=session, prefix_id=f"{session}-prefix",
                                        phase="audit_initial", request_id=prime_id, p_hash=p_hash),
                max_tokens=args.prime_max_tokens,
            )
            active.append({"session_id": session, "prefix_id": f"{session}-prefix",
                           "prompt": prompt})
        staged_ns = time.time_ns()
        write_jsonl(event_path, {"event": "rq11.staged", "ts_ns": staged_ns,
                                 "session_count": args.session_count, "host_donors": len(donors),
                                 "host_tokens": [row["host_tokens"] for row in donors]})
        await asyncio.sleep(args.target_wait_ms / 1000)
        tool_return_ns = time.time_ns()
        warmup_ready = asyncio.Event()

        async def decode(index: int) -> dict[str, Any]:
            row = active[index]
            request_id = f"{row['session_id']}-replay"
            prompt = replay_prompt(row["prompt"])
            result = await stream_decode(
                client, base_url=args.base_url, model=args.model, prompt=prompt,
                request_context=context(session_id=row["session_id"], prefix_id=row["prefix_id"],
                                        phase="audit_replay", request_id=request_id,
                                        p_hash=prompt_hash(prompt)),
                max_tokens=args.decode_tokens, warmup_chunks=args.warmup_chunks if index == 0 else 1,
                warmup_ready=warmup_ready if index == 0 else asyncio.Event(),
            )
            result.update({"session_id": row["session_id"], "request_id": request_id,
                           "tool_return_ns": tool_return_ns,
                           "first_token_after_tool_ms": _duration_ms(tool_return_ns,
                                                                      result["chunk_times_ns"][0])
                           if result["chunk_times_ns"] else None,
                           "completion_after_tool_ms": _duration_ms(tool_return_ns,
                                                                      result["request_end_ns"])})
            return result

        tasks = [asyncio.create_task(decode(index)) for index in range(active_count)]
        await warmup_ready.wait()
        if tasks[0].done() and args.planned_overlap:
            raise RuntimeError("Target finished before its warmup boundary; no overlap is possible")

        async def request_load(index: int) -> tuple[dict[str, Any], asyncio.Task]:
            row = donors[index]
            request_start_ns = time.time_ns()
            accepted = await prepare_prefix(
                client, url=args.prepare_control_url, session_id=row["session_id"],
                prefix_id=row["prefix_id"], p_hash=row["prompt_hash"],
                request_id=row["prime_request_id"], plan_only=False,
                min_load_tokens=args.min_load_tokens,
                minimum_host_tokens=args.minimum_host_tokens,
            )
            load_id = str(accepted.get("load_id") or "")
            if not load_id or int(accepted.get("loaded_tokens") or 0) <= 0:
                raise RuntimeError(f"Donor {index} native load not accepted: {accepted}")
            row.update({"load_id": load_id, "load_request_ns": request_start_ns,
                        "load_accepted_ns": time.time_ns(),
                        "loaded_tokens": int(accepted["loaded_tokens"])})
            write_jsonl(event_path, {"event": "rq11.load_accepted", "ts_ns": row["load_accepted_ns"],
                                     "donor_index": index, "load_id": load_id,
                                     "loaded_tokens": row["loaded_tokens"]})
            return row, asyncio.create_task(confirm_load(client, args.prepare_control_url, load_id))

        confirmations = []
        for index in range(args.planned_overlap):
            confirmations.append(await request_load(index))
        target = await tasks[0]
        for index in range(args.planned_overlap, args.donor_count):
            confirmations.append(await request_load(index))
        for row, task in confirmations:
            status = await task
            if status.get("status") != "finished":
                raise RuntimeError(f"Donor load did not finish: {status}")
            row.update({"worker_started_ns": status.get("worker_started_ns"),
                        "worker_enqueued_ns": status.get("worker_enqueued_ns"),
                        "load_committed_ns": status.get("committed_ns"),
                        "cuda_elapsed_ms": status.get("cuda_elapsed_ms")})
        active_results = [target] + list(await asyncio.gather(*tasks[1:]))
        target_first = target["chunk_times_ns"][0]
        target_finish = target["request_end_ns"]
        for row in donors:
            start, finish = row.get("worker_started_ns"), row.get("load_committed_ns")
            row["worker_window_overlap_ms"] = (_duration_ms(max(start, target_first),
                                                            min(finish, target_finish))
                                                if isinstance(start, int) and isinstance(finish, int)
                                                and min(finish, target_finish) > max(start, target_first)
                                                else 0.0)
        realized = sum(row["worker_window_overlap_ms"] > 0 for row in donors)
        if realized != args.planned_overlap:
            raise RuntimeError(f"Requested {args.planned_overlap} worker-window overlaps, "
                               f"but observed {realized}; this dose is not comparable")
        due_ns = staged_ns + args.donor_wait_ms * 1_000_000
        if time.time_ns() > due_ns:
            raise RuntimeError("Loads and target decode overshot the donor tool-return deadline")
        await asyncio.sleep(max(0, (due_ns - time.time_ns()) / 1e9))

        async def donor_replay(row: dict[str, Any]) -> dict[str, Any]:
            text = replay_prompt(row["prompt"])
            return await completion(
                client, base_url=args.base_url, model=args.model, prompt=text,
                request_context=context(session_id=row["session_id"], prefix_id=row["prefix_id"],
                                        phase="audit_replay", request_id=f"{row['session_id']}-replay",
                                        p_hash=prompt_hash(text)), max_tokens=args.donor_replay_tokens,
            )

        donor_results = await asyncio.gather(*(donor_replay(row) for row in donors))
        finished_ns = time.time_ns()
    result = {
        "schema": "agentic_work_audit.overlap_dose.v1", "run_id": args.run_id,
        "status": "worker_window_only", "research_question_id": "RQ11",
        "seed": args.seed, "session_count": args.session_count,
        "donor_count": args.donor_count, "planned_overlap": args.planned_overlap,
        "realized_worker_window_overlap_count": realized,
        "realized_worker_window_overlap_ms": round(sum(row["worker_window_overlap_ms"] for row in donors), 3),
        "target": target, "active_requests": active_results,
        "donors": [{key: value for key, value in row.items() if key != "prompt"} for row in donors],
        "donor_replay_ttft_ms": [row["ttft_ms"] for row in donor_results],
        "workflow_makespan_ms": _duration_ms(started_ns, finished_ns),
        "workload": {"target_wait_ms": args.target_wait_ms,
                     "donor_wait_ms": args.donor_wait_ms,
                     "active_prompt_tokens": args.active_prompt_tokens,
                     "donor_prompt_tokens": args.donor_prompt_tokens,
                     "decode_tokens": args.decode_tokens,
                     "donor_replay_tokens": args.donor_replay_tokens},
        "limitations": ["Worker-start to commit overlap is only an upper bound on physical CUDA-copy overlap.",
                        "Physical copy overlap requires the separate Nsight evidence gate."],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--session-count", type=int, default=6)
    parser.add_argument("--donor-count", type=int, default=4)
    parser.add_argument("--planned-overlap", type=int, required=True)
    parser.add_argument("--active-prompt-tokens", type=int, default=4090)
    parser.add_argument("--donor-prompt-tokens", type=int, default=4090)
    parser.add_argument("--eviction-prompt-tokens", type=int, default=4090)
    parser.add_argument("--prime-max-tokens", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--donor-replay-tokens", type=int, default=8)
    parser.add_argument("--warmup-chunks", type=int, default=3)
    parser.add_argument("--target-wait-ms", type=int, default=900)
    parser.add_argument("--donor-wait-ms", type=int, default=10000)
    parser.add_argument("--min-load-tokens", type=int, default=512)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--eviction-rounds", type=int, default=4)
    args = parser.parse_args()
    args.direct_device_evict_for_stage = True
    if (args.session_count < args.donor_count + 1 or args.donor_count < 1 or
            not 0 <= args.planned_overlap <= args.donor_count or
            min(args.decode_tokens, args.donor_wait_ms, args.target_wait_ms,
                args.active_prompt_tokens, args.donor_prompt_tokens) <= 0):
        parser.error("Invalid session, overlap, or workload size")
    result = asyncio.run(run(args))
    print(json.dumps({key: result[key] for key in
                      ("run_id", "status", "planned_overlap", "realized_worker_window_overlap_count",
                       "realized_worker_window_overlap_ms", "workflow_makespan_ms")}, indent=2))


if __name__ == "__main__":
    main()
