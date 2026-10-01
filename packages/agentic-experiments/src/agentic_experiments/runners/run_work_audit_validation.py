#!/usr/bin/env python3
"""Two-session, hand-checkable live evidence probe for the pinned backend."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from agentic_work_audit.events import AuditEvent

from .run_kv_movement_interference import (
    completion,
    context,
    eligible_host_prefix,
    evict_device_prefix,
    make_prompt,
    prepare_prefix,
    prompt_hash,
    replay_prompt,
)
from .run_sustained_decode_kv_overlap import load_status


class EventLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def emit(self, kind: str, session: str, request: str = "", *, at_ns: int | None = None, **evidence: Any) -> None:
        event = AuditEvent(kind=kind, ts_ns=at_ns or time.time_ns(), source="validation_client",
                           session_id=session, request_id=request, evidence=evidence)
        self.handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")

    def close(self) -> None:
        self.handle.close()


async def one_case(
    client: httpx.AsyncClient, args: argparse.Namespace, log: EventLog, case: str
) -> dict[str, Any]:
    session = f"{args.run_id}-{case}"
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
    log.emit("tool_start", session, initial_id, expected_ms=args.wait_ms)
    await asyncio.sleep(args.wait_ms / 1000)
    log.emit("tool_end", session, initial_id)

    plan: dict[str, Any] | None = None
    status: dict[str, Any] | None = None
    if case == "host":
        for round_index in range(args.eviction_rounds):
            evicted = await evict_device_prefix(
                client, url=args.prepare_control_url, session_id=session, prefix_id=prefix,
                p_hash=hashed, request_id=initial_id, tokens=args.prompt_tokens,
                source="agentic_work_audit.validation",
            )
            if not evicted.get("ok"):
                raise RuntimeError(f"device eviction failed: {json.dumps(evicted, sort_keys=True)}")
            log.emit("device_evict_attempt", session, initial_id, round=round_index,
                     status=evicted.get("status"), evicted_tokens=evicted.get("evicted_tokens"))
            plan = await prepare_prefix(
                client, url=args.prepare_control_url, session_id=session, prefix_id=prefix,
                p_hash=hashed, request_id=initial_id, plan_only=True,
                min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
            )
            if eligible_host_prefix(plan):
                log.emit("device_evict_proof", session, initial_id, round=round_index,
                         status=evicted.get("status"), evicted_tokens=evicted.get("evicted_tokens"))
                break
            await asyncio.sleep(0.15)
        if not plan or not eligible_host_prefix(plan):
            raise RuntimeError(f"host residency was not proved: {json.dumps(plan, sort_keys=True)}")
        log.emit("host_resident_proof", session, initial_id,
                 host_tokens=plan.get("host_tokens"), node_id=plan.get("node_id"))

    if case == "host":
        accepted = await prepare_prefix(
            client, url=args.prepare_control_url, session_id=session, prefix_id=prefix,
            p_hash=hashed, request_id=initial_id, plan_only=False,
            min_load_tokens=None, minimum_host_tokens=args.minimum_host_tokens,
        )
        load_id = str(accepted.get("load_id") or "")
        if not load_id:
            raise RuntimeError(f"native reload was not accepted: {json.dumps(accepted, sort_keys=True)}")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = await load_status(client, args.prepare_control_url, load_id)
            if status.get("status") == "finished":
                break
            if not status.get("ok"):
                raise RuntimeError(f"native reload failed: {json.dumps(status, sort_keys=True)}")
            await asyncio.sleep(0.01)
        if not status or status.get("status") != "finished" or float(status.get("cuda_elapsed_ms") or 0) <= 0:
            raise RuntimeError(f"native CUDA reload was not proved: {json.dumps(status, sort_keys=True)}")
        log.emit("client_load_confirmed", session, initial_id, load_id=load_id,
                 loaded_tokens=status.get("loaded_tokens"), cuda_elapsed_ms=status.get("cuda_elapsed_ms"))

    replay = await completion(
        client, base_url=args.base_url, model=args.model, prompt=replay_prompt(prompt),
        request_context=context(session_id=session, prefix_id=prefix, phase="audit_replay",
                                request_id=replay_id, p_hash=prompt_hash(replay_prompt(prompt))),
        max_tokens=args.max_tokens,
    )
    log.emit("replay_sent", session, replay_id, at_ns=replay["request_start_ns"])
    log.emit("replay_first_token", session, replay_id, at_ns=replay["first_token_ns"])
    log.emit("replay_finished", session, replay_id, at_ns=replay["request_end_ns"])
    return {"session_id": session, "case": case, "initial": initial, "replay": replay,
            "host_plan": plan, "load_status": status}


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--prompt-tokens", type=int, default=4090)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=500)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--eviction-rounds", type=int, default=4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = EventLog(args.out_dir / "harness_events.jsonl")
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            cases = []
            for case in ("warm", "host"):
                cases.append(await one_case(client, args, log, case))
        (args.out_dir / "case_results.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    finally:
        log.close()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
