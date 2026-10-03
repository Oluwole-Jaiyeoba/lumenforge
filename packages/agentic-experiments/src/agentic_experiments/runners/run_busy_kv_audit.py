#!/usr/bin/env python3
"""Busy KV audit: ordinary replay, check-only, or timing-only KV preparation.

The controller sees each session's expected tool-return time and the backend's
current host-residency answer. It never assigns frontend importance or forces
eviction. Each arm must start with a fresh backend to keep cache state separate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from agentic_controller.kv_busy_prepare import BusyPreparePolicy

from .run_kv_movement_interference import completion, make_prompt, prepare_prefix, prompt_hash, write_jsonl
from .run_natural_multi_agent_kv_pressure import append_tool_result, load_trace, request_context
from .run_work_audit_timing import confirm_load


def wait_ms(args: argparse.Namespace, session_index: int, step: int) -> int:
    span = args.wait_max_ms - args.wait_min_ms + 1
    return args.wait_min_ms + ((args.seed * 137 + session_index * 251 + step * 617) % span)


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {"median_ms": None, "p95_ms": None, "max_ms": None}
    values = sorted(float(row["tool_return_to_first_token_ms"]) for row in rows)
    return {"median_ms": round(statistics.median(values), 3),
            "p95_ms": round(values[min(len(values) - 1, int(.95 * len(values)))], 3),
            "max_ms": round(values[-1], 3)}


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = args.out_dir / "harness_events.jsonl"
    events.write_text("", encoding="utf-8")
    policy = BusyPreparePolicy(estimated_load_ms=args.estimated_load_ms, margin_ms=args.margin_ms)
    active_replays: set[str] = set()
    decisions: list[dict[str, Any]] = []
    all_rows: list[list[dict[str, Any]]] = []
    timeout = httpx.Timeout(args.timeout_s, connect=60)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async def session(index: int) -> list[dict[str, Any]]:
            # This identity and prompt are identical in both arms for a seed.
            session_id = f"busy-seed{args.seed}-session-{index:02d}"
            prefix_id = f"{session_id}-prefix"
            await asyncio.sleep(index * args.stagger_ms / 1000)
            prompt = make_prompt(session_id, args.prefix_tokens)
            prompt += "\nFirst turn: inspect the repository and decide which tool to use first."
            initial_id = f"{session_id}-initial"
            initial = await completion(
                client, base_url=args.base_url, model=args.model, prompt=prompt,
                request_context=request_context(session_id, "initial", initial_id, prompt_hash(prompt)),
                max_tokens=1,
            )
            write_jsonl(events, {"kind": "initial_finished", "session_id": session_id,
                                 "request_id": initial_id, **initial})
            rows: list[dict[str, Any]] = []
            prepare_tasks: list[asyncio.Task[None]] = []
            for step in range(args.tool_waits):
                prior_prompt_hash = prompt_hash(prompt)
                prior_request_id = initial_id if step == 0 else f"{session_id}-replay-{step}"
                duration_ms = wait_ms(args, index, step)
                wait_start_ns = time.time_ns()
                due_ns = wait_start_ns + duration_ms * 1_000_000
                write_jsonl(events, {"kind": "tool_wait_start", "session_id": session_id,
                                     "step": step + 1, "wait_ms": duration_ms,
                                     "start_ns": wait_start_ns, "expected_return_ns": due_ns})

                async def prepare_during_wait(
                    step: int = step, wait_start_ns: int = wait_start_ns,
                    duration_ms: int = duration_ms, due_ns: int = due_ns,
                    prior_prompt_hash: str = prior_prompt_hash,
                    prior_request_id: str = prior_request_id,
                ) -> None:
                    # Three bounded checks; all control calls are independent of the
                    # harness's due-time sleep and cannot hold replay submission.
                    for fraction in (0.0, .35, .65):
                        target_ns = wait_start_ns + int(fraction * duration_ms * 1_000_000)
                        await asyncio.sleep(max(0, (target_ns - time.time_ns()) / 1e9))
                        plan = await prepare_prefix(
                            client, url=args.prepare_url, session_id=session_id,
                            prefix_id=prefix_id, p_hash=prior_prompt_hash,
                            request_id=prior_request_id, plan_only=True,
                            min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
                            source="agentic_work_audit.busy",
                        )
                        now_ns = time.time_ns()
                        eligible = plan.get("ok") is True and plan.get("status") == "would_load_back"
                        decision = policy.decide(now_ns=now_ns, tool_due_ns=due_ns,
                                                 host_resident=eligible,
                                                 active_replays=len(active_replays))
                        row: dict[str, Any] = {
                            "session_id": session_id, "step": step + 1,
                            "check_fraction": fraction, "plan_status": plan.get("status"),
                            "host_tokens": plan.get("host_tokens"), "node_id": plan.get("node_id"),
                            "plan_duration_ms": plan.get("control_duration_ms"),
                            "decision": decision.action, "reason": decision.reason,
                            "remaining_ms": decision.remaining_ms,
                            "active_replays": decision.active_replays,
                            "decision_ns": now_ns,
                        }
                        decisions.append(row)
                        write_jsonl(events, {"kind": "controller_decision", **row})
                        if decision.action != "load":
                            continue
                        if args.mode == "check_only":
                            placebo = await prepare_prefix(
                                client, url=args.prepare_url, session_id=session_id,
                                prefix_id=prefix_id, p_hash=prior_prompt_hash,
                                request_id=prior_request_id, plan_only=True,
                                min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
                                source="agentic_work_audit.busy",
                            )
                            row.update({"placebo_status": placebo.get("status"),
                                        "placebo_duration_ms": placebo.get("control_duration_ms")})
                            write_jsonl(events, {"kind": "controller_placebo", **row})
                            return
                        load = await prepare_prefix(
                            client, url=args.prepare_url, session_id=session_id,
                            prefix_id=prefix_id, p_hash=prior_prompt_hash,
                            request_id=prior_request_id, plan_only=False,
                            min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
                            source="agentic_work_audit.busy",
                        )
                        row.update({"load_status": load.get("status"),
                                    "load_id": load.get("load_id"),
                                    "loaded_tokens": load.get("loaded_tokens"),
                                    "load_request_ns": load.get("control_request_started_ns"),
                                    "load_response_ns": load.get("control_response_ns")})
                        write_jsonl(events, {"kind": "controller_load", **row})
                        if load.get("load_id"):
                            try:
                                finished = await confirm_load(client, args.prepare_url, str(load["load_id"]))
                                row["load_finished_observed_ns"] = finished.get("finished_observed_ns")
                                row["cuda_elapsed_ms"] = finished.get("cuda_elapsed_ms")
                            except (RuntimeError, TimeoutError) as exc:
                                row["load_confirmation_error"] = str(exc)
                            write_jsonl(events, {"kind": "controller_load_outcome", **row})
                        return

                prep_task = asyncio.create_task(prepare_during_wait()) if args.mode in ("check_only", "controller") else None
                if prep_task:
                    prepare_tasks.append(prep_task)
                await asyncio.sleep(max(0, (due_ns - time.time_ns()) / 1e9))
                returned_ns = time.time_ns()
                write_jsonl(events, {"kind": "tool_return", "session_id": session_id,
                                     "step": step + 1, "returned_ns": returned_ns,
                                     "expected_return_ns": due_ns})
                prompt = append_tool_result(prompt, session_id, step)
                replay_id = f"{session_id}-replay-{step + 1}"
                active_replays.add(replay_id)
                try:
                    replay = await completion(
                        client, base_url=args.base_url, model=args.model, prompt=prompt,
                        request_context=request_context(session_id, f"replay_{step + 1}", replay_id,
                                                        prompt_hash(prompt)),
                        max_tokens=args.replay_tokens,
                    )
                finally:
                    active_replays.discard(replay_id)
                row = {"session_id": session_id, "request_id": replay_id, "step": step + 1,
                       "wait_ms": duration_ms, "tool_return_ns": returned_ns,
                       "tool_return_to_first_token_ms": round((replay["first_token_ns"] - returned_ns) / 1e6, 3),
                       "tool_return_to_finish_ms": round((replay["request_end_ns"] - returned_ns) / 1e6, 3),
                       **replay}
                write_jsonl(events, {"kind": "replay_finished", **row})
                rows.append(row)
            if prepare_tasks:
                await asyncio.gather(*prepare_tasks)
            return [{"session_id": session_id, "request_id": initial_id, **initial}, *rows]

        all_rows = await asyncio.gather(*(session(i) for i in range(args.session_count)))

    initials = [rows[0] for rows in all_rows]
    replays = [row for rows in all_rows for row in rows[1:]]
    trace = load_trace(args.backend_trace)
    native_loads = [row for row in trace if row.get("event") == args.native_load_event]
    prepared = [row for row in decisions if row["decision"] == "load"]
    completed_before_due = [row for row in prepared if isinstance(row.get("load_finished_observed_ns"), int)
                            and row["load_finished_observed_ns"] <= next(
                                replay["tool_return_ns"] for replay in replays
                                if replay["session_id"] == row["session_id"] and replay["step"] == row["step"])]
    summary = {
        "schema": "agentic_work_audit.busy_arm.v1", "run_id": args.run_id,
        "mode": args.mode, "status": "complete", "seed": args.seed,
        "frontend_priority": "none", "forced_eviction": False,
        "native_load_event": args.native_load_event,
        "workload": {"session_count": args.session_count, "tool_waits_per_session": args.tool_waits,
                     "prefix_tokens": args.prefix_tokens, "replay_tokens": args.replay_tokens,
                     "wait_range_ms": [args.wait_min_ms, args.wait_max_ms],
                     "stagger_ms": args.stagger_ms, "estimated_load_ms": args.estimated_load_ms,
                     "margin_ms": args.margin_ms, "minimum_host_tokens": args.minimum_host_tokens},
        "session_count": len(all_rows), "replay_count": len(replays),
        "workflow_makespan_ms": round((max(row["request_end_ns"] for row in replays) -
                                        min(row["request_start_ns"] for row in initials)) / 1e6, 3),
        "total_replay_ttft_ms": round(sum(row["ttft_ms"] for row in replays), 3),
        "total_return_to_first_token_ms": round(sum(row["tool_return_to_first_token_ms"] for row in replays), 3),
        "return_to_first_token": summarize_latency(replays),
        "native_load_events": len(native_loads),
        "controller_plan_checks": len(decisions),
        "controller_load_decisions": len(prepared),
        "controller_load_attempts": len(prepared) if args.mode == "controller" else 0,
        "controller_loads_finished_before_tool_return": len(completed_before_due),
        "controller_loads_finished_after_tool_return": sum(
            isinstance(row.get("load_finished_observed_ns"), int) and
            row["load_finished_observed_ns"] > next(replay["tool_return_ns"] for replay in replays
                                                   if replay["session_id"] == row["session_id"] and replay["step"] == row["step"])
            for row in prepared),
        "decisions": decisions, "replays": replays,
        "interpretation_limit": "Native load counts and cache residency are observations; changed replay timing is a matched-arm association, not proof of HBM interference or universally avoidable work.",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("baseline", "check_only", "controller"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backend-trace", type=Path, required=True)
    parser.add_argument("--native-load-event", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--session-count", type=int, default=12)
    parser.add_argument("--tool-waits", type=int, default=3)
    parser.add_argument("--prefix-tokens", type=int, default=8192)
    parser.add_argument("--replay-tokens", type=int, default=64)
    parser.add_argument("--stagger-ms", type=int, default=75)
    parser.add_argument("--wait-min-ms", type=int, default=800)
    parser.add_argument("--wait-max-ms", type=int, default=3500)
    parser.add_argument("--estimated-load-ms", type=float, default=250)
    parser.add_argument("--margin-ms", type=float, default=150)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--timeout-s", type=float, default=900)
    args = parser.parse_args()
    if min(args.session_count, args.tool_waits, args.prefix_tokens, args.replay_tokens,
           args.wait_min_ms, args.minimum_host_tokens) <= 0 or args.wait_max_ms < args.wait_min_ms:
        parser.error("workload sizes and waits must be positive and ordered")
    if args.stagger_ms < 0:
        parser.error("stagger must be nonnegative")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
