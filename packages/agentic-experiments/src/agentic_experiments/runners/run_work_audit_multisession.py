#!/usr/bin/env python3
"""Compare KV load timings across three equal-importance sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
from agentic_controller.kv_prepare_window import KVPrepareWindowPolicy

from .run_kv_movement_interference import (
    completion, context, eligible_host_prefix, evict_device_prefix, make_prompt,
    prepare_prefix, prompt_hash, replay_prompt,
)
from .run_work_audit_timing import confirm_load
from .run_work_audit_validation import EventLog


async def one_case(client: httpx.AsyncClient, args: argparse.Namespace, log: EventLog,
                   case_id: str, condition: str, *, release_ended_prefix: bool = True) -> dict[str, Any]:
    labels = ("long", "short", "ends")
    sessions = {label: f"{case_id}-{label}" for label in labels}
    prompts = {label: make_prompt(sessions[label], args.prompt_tokens) for label in labels}
    ids = {label: f"{sessions[label]}-initial" for label in labels}

    log.emit("case_start", sessions["long"], ids["long"], condition=condition)
    try:
        async def initial(label: str) -> dict[str, Any]:
            session = sessions[label]
            result = await completion(
                client, base_url=args.base_url, model=args.model, prompt=prompts[label],
                request_context=context(session_id=session, prefix_id=f"{session}-prefix",
                                        phase="audit_initial", request_id=ids[label],
                                        p_hash=prompt_hash(prompts[label])), max_tokens=args.max_tokens,
            )
            log.emit("initial_sent", session, ids[label], at_ns=result["request_start_ns"])
            log.emit("initial_finished", session, ids[label], at_ns=result["request_end_ns"])
            return result

        initials = dict(zip(labels, await asyncio.gather(*(initial(label) for label in labels))))
        # These are harness predictions, not frontend importance ranks.
        for label, wait_ms in (("short", args.short_wait_ms), ("long", args.long_wait_ms)):
            log.emit("tool_start", sessions[label], ids[label], expected_ms=wait_ms,
                     importance="equal")
        started = asyncio.get_running_loop().time()
        long_due = started + args.long_wait_ms / 1000
        short_completion = asyncio.Event()
        log.emit("capacity_policy", sessions["long"], ids["long"],
                 active_prefix_budget=2, candidate_sessions=3,
                 action="explicit_evict_long_prefix")

        async def evict_long() -> dict[str, Any]:
            label = "long"
            session = sessions[label]
            plan: dict[str, Any] = {}
            for attempt in range(args.eviction_rounds):
                evicted = await evict_device_prefix(
                    client, url=args.prepare_control_url, session_id=session,
                    prefix_id=f"{session}-prefix", p_hash=prompt_hash(prompts[label]),
                    request_id=ids[label], tokens=args.prompt_tokens,
                    source="agentic_work_audit.multisession",
                )
                if not evicted.get("ok"):
                    raise RuntimeError(f"explicit device eviction failed: {evicted}")
                plan = await prepare_prefix(
                    client, url=args.prepare_control_url, session_id=session,
                    prefix_id=f"{session}-prefix", p_hash=prompt_hash(prompts[label]),
                    request_id=ids[label], plan_only=True, min_load_tokens=None,
                    minimum_host_tokens=args.minimum_host_tokens,
                )
                if eligible_host_prefix(plan):
                    log.emit("device_evict_proof", session, ids[label], round=attempt,
                             evicted_tokens=evicted.get("evicted_tokens"))
                    log.emit("host_resident_proof", session, ids[label],
                             host_tokens=plan.get("host_tokens"), node_id=plan.get("node_id"))
                    return plan
                await asyncio.sleep(0.15)
            raise RuntimeError(f"long session host residency not proved: {plan}")

        eviction = asyncio.create_task(evict_long())

        async def replay(label: str, wait_ms: int) -> dict[str, Any]:
            await asyncio.sleep(max(0, started + wait_ms / 1000 -
                                    asyncio.get_running_loop().time()))
            session = sessions[label]
            log.emit("tool_end", session, ids[label])
            replay_id = f"{session}-replay"
            text = replay_prompt(prompts[label])
            result = await completion(
                client, base_url=args.base_url, model=args.model, prompt=text,
                request_context=context(session_id=session, prefix_id=f"{session}-prefix",
                                        phase="audit_replay", request_id=replay_id,
                                        p_hash=prompt_hash(text)), max_tokens=args.max_tokens,
            )
            log.emit("replay_sent", session, replay_id, at_ns=result["request_start_ns"])
            log.emit("replay_first_token", session, replay_id, at_ns=result["first_token_ns"])
            log.emit("replay_finished", session, replay_id, at_ns=result["request_end_ns"])
            if label == "short":
                short_completion.set()
            return result

        async def long_path() -> dict[str, Any]:
            plan = await eviction
            session = sessions["long"]

            async def load() -> tuple[dict[str, Any], dict[str, Any]]:
                accepted = await prepare_prefix(
                    client, url=args.prepare_control_url, session_id=session,
                    prefix_id=f"{session}-prefix", p_hash=prompt_hash(prompts["long"]),
                    request_id=ids["long"], plan_only=False, min_load_tokens=None,
                    minimum_host_tokens=args.minimum_host_tokens,
                )
                load_id = str(accepted.get("load_id") or "")
                if not load_id or int(accepted.get("loaded_tokens") or 0) <= 0:
                    raise RuntimeError(f"load not accepted: {accepted}")
                log.emit("load_requested", session, ids["long"],
                         at_ns=accepted["control_request_started_ns"], load_id=load_id)
                log.emit("load_accepted", session, ids["long"],
                         at_ns=accepted["control_response_ns"], load_id=load_id)
                status = await confirm_load(client, args.prepare_control_url, load_id)
                log.emit("client_load_confirmed", session, ids["long"],
                         at_ns=status.get("finished_observed_ns") or status.get("observed_ns"),
                         load_id=load_id, loaded_tokens=status.get("loaded_tokens"),
                         cuda_elapsed_ms=status.get("cuda_elapsed_ms"))
                return accepted, status

            if condition in ("early", "post_short", "controller_window"):
                await end_task
                if condition == "post_short":
                    await short_task
                elif condition == "controller_window":
                    policy = KVPrepareWindowPolicy(
                        estimated_load_ms=args.estimated_load_ms,
                        safety_margin_ms=args.load_margin_ms,
                    )

                    def decide(short_finished: bool) -> str:
                        decision = policy.decide(
                            now_ns=int(asyncio.get_running_loop().time() * 1e9),
                            tool_return_due_ns=int(long_due * 1e9),
                            host_resident=eligible_host_prefix(plan),
                            slot_released=end_task.done() and end_task.exception() is None,
                            active_replays=int(not short_finished and
                                               asyncio.get_running_loop().time() >=
                                               started + args.short_wait_ms / 1000),
                            blocking_replay_finished=short_finished,
                        )
                        log.emit("controller_prepare_decision", session, ids["long"],
                                 action=decision.action, reason=decision.reason,
                                 remaining_ms=decision.remaining_ms,
                                 estimated_load_ms=decision.estimated_load_ms,
                                 safety_margin_ms=decision.safety_margin_ms,
                                 active_replays=decision.active_replays,
                                 short_replay_finished=short_finished)
                        return decision.action

                    short_finished = short_completion.is_set()
                    action = decide(short_finished)
                    if not short_finished:
                        await short_completion.wait()
                        await short_task
                        action = decide(True)
                    else:
                        await short_task
                    if action not in ("load", "defer"):
                        raise RuntimeError(f"controller did not reach an actionable window: {action}")
                else:
                    await asyncio.sleep(max(0, started + args.early_at_ms / 1000 -
                                            asyncio.get_running_loop().time()))
                if condition == "controller_window" and action == "defer":
                    log.emit("controller_prepare_outcome", session, ids["long"],
                             outcome="deferred_to_tool_return")
                elif asyncio.get_running_loop().time() >= long_due:
                    raise RuntimeError(f"{condition} preparation missed the tool-return window")
                else:
                    accepted, status = await load()
                    load_before_due = asyncio.get_running_loop().time() < long_due
                    if condition == "controller_window":
                        log.emit("controller_prepare_outcome", session, ids["long"],
                                 outcome="good_window" if load_before_due else "overshot_tool_return",
                                 overshoot_ms=round(max(0, (asyncio.get_running_loop().time() -
                                                            long_due) * 1000), 3))
                    if not load_before_due:
                        raise RuntimeError(f"{condition} native load did not finish before tool return")
            await asyncio.sleep(max(0, long_due - asyncio.get_running_loop().time()))
            log.emit("tool_end", session, ids["long"])
            if condition == "late_nonblocking" or (condition == "controller_window" and action == "defer"):
                # The control response must not gate replay submission.
                load_task = asyncio.create_task(load())
                await asyncio.sleep(0)
            first = await replay_now("long")
            if condition == "late_nonblocking" or (condition == "controller_window" and action == "defer"):
                accepted, status = await load_task
            second_id = f"{session}-replay-2"
            second_text = replay_prompt(replay_prompt(prompts["long"]))
            second = await completion(
                client, base_url=args.base_url, model=args.model, prompt=second_text,
                request_context=context(session_id=session, prefix_id=f"{session}-prefix",
                                        phase="audit_replay_2", request_id=second_id,
                                        p_hash=prompt_hash(second_text)), max_tokens=args.max_tokens,
            )
            log.emit("replay_2_sent", session, second_id, at_ns=second["request_start_ns"])
            log.emit("replay_2_first_token", session, second_id, at_ns=second["first_token_ns"])
            log.emit("replay_2_finished", session, second_id, at_ns=second["request_end_ns"])
            return {"host_plan": plan, "replay": first, "replay_2": second,
                    "load_acceptance": accepted, "load_status": status}

        async def replay_now(label: str) -> dict[str, Any]:
            session = sessions[label]
            replay_id = f"{session}-replay"
            text = replay_prompt(prompts[label])
            result = await completion(
                client, base_url=args.base_url, model=args.model, prompt=text,
                request_context=context(session_id=session, prefix_id=f"{session}-prefix",
                                        phase="audit_replay", request_id=replay_id,
                                        p_hash=prompt_hash(text)), max_tokens=args.max_tokens,
            )
            log.emit("replay_sent", session, replay_id, at_ns=result["request_start_ns"])
            log.emit("replay_first_token", session, replay_id, at_ns=result["first_token_ns"])
            log.emit("replay_finished", session, replay_id, at_ns=result["request_end_ns"])
            return result

        async def end_path() -> None:
            await asyncio.sleep(max(0, started + args.short_wait_ms / 2000 -
                                    asyncio.get_running_loop().time()))
            log.emit("session_end", sessions["ends"], ids["ends"],
                     reason="synthetic_task_complete_no_replay")
            if not release_ended_prefix:
                return
            evicted = await evict_device_prefix(
                client, url=args.prepare_control_url, session_id=sessions["ends"],
                prefix_id=f"{sessions['ends']}-prefix", p_hash=prompt_hash(prompts["ends"]),
                request_id=ids["ends"], tokens=args.prompt_tokens,
                source="agentic_work_audit.multisession",
            )
            if not evicted.get("ok") or int(evicted.get("evicted_tokens") or 0) <= 0:
                raise RuntimeError(f"ending session's prefix was not released: {evicted}")
            log.emit("ended_prefix_evict_proof", sessions["ends"], ids["ends"],
                     evicted_tokens=evicted["evicted_tokens"])

        short_task = asyncio.create_task(replay("short", args.short_wait_ms))
        long_task = asyncio.create_task(long_path())
        end_task = asyncio.create_task(end_path())
        short, long, _ = await asyncio.gather(short_task, long_task, end_task)
        return {"case_id": case_id, "condition": condition, "initials": initials,
                "short": {"replay": short}, "long": long,
                "ends": {"ended_without_replay": True}}
    finally:
        log.emit("case_end", sessions["long"], ids["long"], condition=condition)


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--prompt-tokens", type=int, default=4090)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--short-wait-ms", type=int, default=900)
    parser.add_argument("--long-wait-ms", type=int, default=2500)
    parser.add_argument("--early-at-ms", type=int, default=1200)
    parser.add_argument("--estimated-load-ms", type=float, default=250)
    parser.add_argument("--load-margin-ms", type=float, default=150)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--eviction-rounds", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--warmup-pairs", type=int, default=1)
    parser.add_argument("--case-order", choices=(
        "early-late_nonblocking", "late_nonblocking-early",
        "late_nonblocking-early-post_short", "post_short-early-late_nonblocking",
        "late_nonblocking-early-post_short-controller_window",
        "controller_window-post_short-early-late_nonblocking",
        "long-short-ends",
    ), default="late_nonblocking-early")
    args = parser.parse_args()
    if (not (0 < args.short_wait_ms < args.early_at_ms < args.long_wait_ms) or
            args.pairs < 1 or args.warmup_pairs < 0 or
            args.estimated_load_ms <= 0 or args.load_margin_ms < 0):
        parser.error("need short wait < early preparation < long wait, measured pairs, and nonnegative warmups")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.out_dir / "harness_events.jsonl")
    try:
        cases = []
        async with httpx.AsyncClient(timeout=45) as client:
            if args.case_order == "long-short-ends":
                result = await one_case(client, args, log, args.run_id, "late_nonblocking",
                                        release_ended_prefix=False)
            else:
                for sequence in range(1, args.warmup_pairs + args.pairs + 1):
                    warmup = sequence <= args.warmup_pairs
                    pair = sequence - args.warmup_pairs if not warmup else sequence
                    order = (args.case_order.split("-") if sequence % 2 else
                             list(reversed(args.case_order.split("-"))))
                    for condition in order:
                        label = "warmup" if warmup else "pair"
                        case_id = f"{args.run_id}-{label}{pair:02d}-{condition}"
                        case = await one_case(client, args, log, case_id, condition)
                        case["pair"] = pair
                        case["warmup"] = warmup
                        cases.append(case)
                result = cases
        (args.out_dir / "case_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        log.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
