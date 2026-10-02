#!/usr/bin/env python3
"""Small concurrent KV timeline with equal-importance agent sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import (
    completion, context, eligible_host_prefix, evict_device_prefix, make_prompt,
    prepare_prefix, prompt_hash, replay_prompt,
)
from .run_work_audit_timing import confirm_load
from .run_work_audit_validation import EventLog


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
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--eviction-rounds", type=int, default=4)
    args = parser.parse_args()
    if not (0 < args.short_wait_ms < args.long_wait_ms):
        parser.error("need positive short wait less than long wait")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.out_dir / "harness_events.jsonl")
    labels = ("long", "short", "ends")
    sessions = {label: f"{args.run_id}-{label}" for label in labels}
    prompts = {label: make_prompt(sessions[label], args.prompt_tokens) for label in labels}
    ids = {label: f"{sessions[label]}-initial" for label in labels}

    async with httpx.AsyncClient(timeout=45) as client:
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

        try:
            initials = dict(zip(labels, await asyncio.gather(*(initial(label) for label in labels))))
            # These are harness predictions, not frontend importance ranks.
            for label, wait_ms in (("short", args.short_wait_ms), ("long", args.long_wait_ms)):
                log.emit("tool_start", sessions[label], ids[label], expected_ms=wait_ms,
                         importance="equal")
            long_due = asyncio.get_running_loop().time() + args.long_wait_ms / 1000
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
                await asyncio.sleep(wait_ms / 1000)
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
                return result

            async def long_path() -> dict[str, Any]:
                plan = await eviction
                # The load is intentionally requested at tool return, but its
                # control response does not gate the replay request.
                await asyncio.sleep(max(0, long_due - asyncio.get_running_loop().time()))
                session = sessions["long"]
                log.emit("tool_end", session, ids["long"])

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

                load_task = asyncio.create_task(load())
                await asyncio.sleep(0)
                first = await replay_now("long")
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
                await asyncio.sleep(args.short_wait_ms / 2000)
                log.emit("session_end", sessions["ends"], ids["ends"],
                         reason="synthetic_task_complete_no_replay")

            short_task = asyncio.create_task(replay("short", args.short_wait_ms))
            long_task = asyncio.create_task(long_path())
            end_task = asyncio.create_task(end_path())
            short, long, _ = await asyncio.gather(short_task, long_task, end_task)
            cases = {"initials": initials, "short": {"replay": short}, "long": long,
                     "ends": {"ended_without_replay": True}}
            (args.out_dir / "case_results.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
        finally:
            log.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
