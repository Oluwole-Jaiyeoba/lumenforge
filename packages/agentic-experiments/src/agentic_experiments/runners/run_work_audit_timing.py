#!/usr/bin/env python3
"""Compare early, response-gated late, and nonblocking late KV preparation."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import (
    completion, context, eligible_host_prefix, evict_device_prefix, make_prompt,
    prepare_prefix, prompt_hash, replay_prompt,
)
from .run_sustained_decode_kv_overlap import load_status
from .run_work_audit_validation import EventLog


async def confirm_load(client: httpx.AsyncClient, url: str, load_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = await load_status(client, url, load_id)
        if not status.get("ok"):
            raise RuntimeError(f"native load status failed: {json.dumps(status, sort_keys=True)}")
        if status.get("status") == "finished":
            if float(status.get("cuda_elapsed_ms") or 0) <= 0:
                raise RuntimeError(f"CUDA completion was not timed: {json.dumps(status, sort_keys=True)}")
            return status
        await asyncio.sleep(0.01)
    raise TimeoutError(f"native load {load_id} did not finish")


async def one_case(
    client: httpx.AsyncClient, args: argparse.Namespace, log: EventLog, pair: int, condition: str,
) -> dict[str, Any]:
    label = f"warmup-pair{abs(pair):02d}" if pair <= 0 else f"pair{pair:02d}"
    session = f"{args.run_id}-{label}-{condition}"
    prefix = f"{session}-prefix"
    initial_id = f"{session}-initial"
    replay_id = f"{session}-replay"
    prompt = make_prompt(session, args.prompt_tokens)
    hashed = prompt_hash(prompt)
    initial = await completion(
        client, base_url=args.base_url, model=args.model, prompt=prompt,
        request_context=context(session_id=session, prefix_id=prefix, phase="audit_initial",
                                request_id=initial_id, p_hash=hashed), max_tokens=args.max_tokens,
    )
    log.emit("initial_sent", session, initial_id, at_ns=initial["request_start_ns"])
    log.emit("initial_finished", session, initial_id, at_ns=initial["request_end_ns"])

    plan: dict[str, Any] = {}
    for round_index in range(args.eviction_rounds):
        evicted = await evict_device_prefix(
            client, url=args.prepare_control_url, session_id=session, prefix_id=prefix,
            p_hash=hashed, request_id=initial_id, tokens=args.prompt_tokens,
            source="agentic_work_audit.timing",
        )
        if not evicted.get("ok"):
            raise RuntimeError(f"device eviction failed: {json.dumps(evicted, sort_keys=True)}")
        plan = await prepare_prefix(
            client, url=args.prepare_control_url, session_id=session, prefix_id=prefix,
            p_hash=hashed, request_id=initial_id, plan_only=True,
            min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
        )
        if eligible_host_prefix(plan):
            log.emit("device_evict_proof", session, initial_id, round=round_index,
                     evicted_tokens=evicted.get("evicted_tokens"))
            log.emit("host_resident_proof", session, initial_id,
                     host_tokens=plan.get("host_tokens"), node_id=plan.get("node_id"))
            break
        await asyncio.sleep(0.15)
    if not eligible_host_prefix(plan):
        raise RuntimeError(f"host residency was not proved: {json.dumps(plan, sort_keys=True)}")

    async def tool_wait() -> None:
        await asyncio.sleep(args.wait_ms / 1000)
        log.emit("tool_end", session, initial_id)

    log.emit("tool_start", session, initial_id, expected_ms=args.wait_ms)
    wait_task = asyncio.create_task(tool_wait())
    if condition != "early":
        await wait_task

    async def start_load() -> tuple[dict[str, Any], asyncio.Task[dict[str, Any]]]:
        accepted = await prepare_prefix(
            client, url=args.prepare_control_url, session_id=session, prefix_id=prefix,
            p_hash=hashed, request_id=initial_id, plan_only=False,
            min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
        )
        load_id = str(accepted.get("load_id") or "")
        if not load_id or int(accepted.get("loaded_tokens") or 0) <= 0:
            raise RuntimeError(f"native load was not accepted: {json.dumps(accepted, sort_keys=True)}")
        log.emit("load_requested", session, initial_id, at_ns=accepted["control_request_started_ns"],
                 load_id=load_id)
        log.emit("load_accepted", session, initial_id, at_ns=accepted["control_response_ns"],
                 load_id=load_id, loaded_tokens=accepted.get("loaded_tokens"))

        async def observe_completion() -> dict[str, Any]:
            status = await confirm_load(client, args.prepare_control_url, load_id)
            log.emit("client_load_confirmed", session, initial_id,
                     at_ns=status.get("finished_observed_ns") or status.get("observed_ns"),
                     load_id=load_id, loaded_tokens=status.get("loaded_tokens"),
                     cuda_elapsed_ms=status.get("cuda_elapsed_ms"))
            return status

        return accepted, asyncio.create_task(observe_completion())

    if condition == "late_nonblocking":
        preparation = asyncio.create_task(start_load())
        await asyncio.sleep(0)
    else:
        accepted, status_task = await start_load()
    if condition == "early":
        status = await status_task
        await wait_task

    replay_text = replay_prompt(prompt)
    replay = await completion(
        client, base_url=args.base_url, model=args.model, prompt=replay_text,
        request_context=context(session_id=session, prefix_id=prefix, phase="audit_replay",
                                request_id=replay_id, p_hash=prompt_hash(replay_text)),
        max_tokens=args.max_tokens,
    )
    log.emit("replay_sent", session, replay_id, at_ns=replay["request_start_ns"])
    log.emit("replay_first_token", session, replay_id, at_ns=replay["first_token_ns"])
    log.emit("replay_finished", session, replay_id, at_ns=replay["request_end_ns"])
    if condition == "late_nonblocking":
        accepted, status_task = await preparation
    if condition != "early":
        status = await status_task

    log.emit("tool_2_start", session, replay_id, expected_ms=args.wait_ms)
    await asyncio.sleep(args.wait_ms / 1000)
    log.emit("tool_2_end", session, replay_id)
    second_id = f"{session}-replay-2"
    second_text = replay_prompt(replay_text)
    second = await completion(
        client, base_url=args.base_url, model=args.model, prompt=second_text,
        request_context=context(session_id=session, prefix_id=prefix, phase="audit_replay_2",
                                request_id=second_id, p_hash=prompt_hash(second_text)),
        max_tokens=args.max_tokens,
    )
    log.emit("replay_2_sent", session, second_id, at_ns=second["request_start_ns"])
    log.emit("replay_2_first_token", session, second_id, at_ns=second["first_token_ns"])
    log.emit("replay_2_finished", session, second_id, at_ns=second["request_end_ns"])
    return {"session_id": session, "pair": pair, "condition": condition,
            "prompt_words": len(prompt.split()),
            "initial": initial, "replay": replay, "replay_2": second,
            "host_plan": plan, "load_acceptance": accepted, "load_status": status}


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--prompt-tokens", type=int, default=4090)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=2000)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--eviction-rounds", type=int, default=4)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--warmup-pairs", type=int, default=1,
                        help="Discard paired native-load warmups before measurement")
    parser.add_argument("--case-order", choices=(
        "early-late", "late-early", "early-late-late_nonblocking",
        "late_nonblocking-late-early",
    ), default="early-late")
    args = parser.parse_args()
    if args.pairs < 1 or args.warmup_pairs < 0 or args.wait_ms < 1 or args.prompt_tokens < 512:
        parser.error("need a measured pair, nonnegative warmups, positive wait, and 512-token prefix")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.out_dir / "harness_events.jsonl")
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            warmups = []
            for pair in range(1 - args.warmup_pairs, 1):
                for condition in args.case_order.split("-"):
                    warmups.append(await one_case(client, args, log, pair, condition))
            cases = []
            for pair in range(1, args.pairs + 1):
                for condition in args.case_order.split("-"):
                    cases.append(await one_case(client, args, log, pair, condition))
        (args.out_dir / "warmup_case_results.json").write_text(json.dumps(warmups, indent=2), encoding="utf-8")
        (args.out_dir / "case_results.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    finally:
        log.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
