"""Paired L3 storage replay probe. Reject arms without a proven L3 hit."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import (
    completion, context, evict_device_prefix, make_prompt, prepare_prefix, prompt_hash, replay_prompt,
)
from .run_work_audit_timing import confirm_load


async def control(client: httpx.AsyncClient, url: str, action: str, identity: dict[str, str], **extra: Any) -> dict[str, Any]:
    started_ns = time.time_ns()
    response = await client.post(url, json={"action": action, **identity, **extra, "control_timeout_ms": 15000})
    result = response.json()
    result["control_request_ns"] = started_ns
    result["control_response_ns"] = time.time_ns()
    if not result.get("ok"):
        raise RuntimeError(f"{action} failed: {json.dumps(result, sort_keys=True)}")
    return result


async def stage_storage(client: httpx.AsyncClient, url: str, identity: dict[str, str], storage_id: str) -> dict[str, Any]:
    accepted = await control(client, url, "prefetch_storage", identity, storage_request_id=storage_id)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        status = await client.post(url, json={"action": "storage_prefetch_status", **identity,
                                              "storage_request_id": storage_id, "control_timeout_ms": 15000})
        result = status.json()
        if result.get("status") == "storage_prefetch_pending":
            await asyncio.sleep(0.25)
            continue
        if not result.get("ok") or int(result.get("storage_loaded_tokens") or 0) <= 0:
            raise RuntimeError(f"native L3 hit not proved: {json.dumps(result, sort_keys=True)}")
        return {"accepted": accepted, "completed": result, "completed_observed_ns": time.time_ns()}
    raise TimeoutError("native L3 prefetch did not complete")


async def run(args: argparse.Namespace) -> dict[str, Any]:
    session = f"{args.run_id}-{args.arm}"
    prefix = f"{session}-prefix"
    initial_id = f"{session}-initial"
    # The prompt must be byte-identical across arms of one seed; only trace
    # identities differ. Otherwise a storage timing comparison is confounded.
    prompt = make_prompt(args.run_id, args.prompt_tokens)
    identity = {"session_id": session, "prefix_id": prefix,
                "prompt_hash": prompt_hash(prompt), "request_id": initial_id}
    async with httpx.AsyncClient(timeout=90) as client:
        initial = await completion(client, base_url=args.base_url, model=args.model, prompt=prompt,
                                   request_context=context(session_id=session, prefix_id=prefix,
                                                           phase="storage_initial", request_id=initial_id,
                                                           p_hash=identity["prompt_hash"]),
                                   max_tokens=args.max_tokens)
        host_status = None
        for _ in range(4):
            await evict_device_prefix(client, url=args.control_url, session_id=session, prefix_id=prefix,
                                      p_hash=identity["prompt_hash"], request_id=initial_id,
                                      tokens=args.prompt_tokens, source="agentic_work_audit.storage")
            host_status = await control(client, args.control_url, "storage_status", identity)
            if host_status["gpu_tokens"] == 0 and host_status["host_tokens"] > 0:
                break
            await asyncio.sleep(0.2)
        if not host_status or host_status["gpu_tokens"] or not host_status["host_tokens"]:
            raise RuntimeError(f"host-only precursor not proved: {host_status}")
        evicted = await control(client, args.control_url, "evict_host", identity)
        if evicted["gpu_tokens_after"] or evicted["host_tokens_after"]:
            raise RuntimeError(f"storage-only precursor not proved: {evicted}")

        tool_start_ns = time.time_ns()
        due_ns = tool_start_ns + args.wait_ms * 1_000_000

        async def peer(index: int) -> dict[str, Any]:
            await asyncio.sleep(args.peer_start_ms / 1000)
            peer_session = f"{session}-peer{index}"
            peer_prompt = make_prompt(f"{args.run_id}-peer{index}", args.peer_prompt_tokens)
            result = await completion(
                client, base_url=args.base_url, model=args.model, prompt=peer_prompt,
                request_context=context(session_id=peer_session, prefix_id=peer_session,
                                        phase="storage_peer", request_id=f"{peer_session}-request",
                                        p_hash=prompt_hash(peer_prompt)),
                max_tokens=args.peer_max_tokens,
            )
            return {"peer_index": index, **result}

        peer_tasks = [asyncio.create_task(peer(index)) for index in range(args.peer_count)]
        staged: dict[str, Any] | None = None
        device_load: dict[str, Any] | None = None
        if args.arm != "on_demand":
            staged = await stage_storage(client, args.control_url, identity, f"{session}-storage")
            if args.arm == "full_prepare":
                accepted = await prepare_prefix(client, url=args.control_url, session_id=session,
                                                prefix_id=prefix, p_hash=identity["prompt_hash"],
                                                request_id=initial_id, plan_only=False, min_load_tokens=None,
                                                minimum_host_tokens=1, source="agentic_work_audit.storage")
                if not accepted.get("load_id") or int(accepted.get("loaded_tokens") or 0) <= 0:
                    raise RuntimeError(f"native H2D load not accepted: {accepted}")
                device_load = {"accepted": accepted,
                               "completed": await confirm_load(client, args.control_url, str(accepted["load_id"]))}
            if time.time_ns() > due_ns:
                raise RuntimeError(
                    f"{args.arm} did not finish preparation inside the {args.wait_ms} ms tool wait; "
                    "run a longer wait for the paired latency comparison"
                )
        remaining = (due_ns - time.time_ns()) / 1e9
        if remaining > 0:
            await asyncio.sleep(remaining)
        tool_end_ns = time.time_ns()
        replay_id = f"{session}-replay"
        replay = await completion(client, base_url=args.base_url, model=args.model,
                                  prompt=replay_prompt(prompt),
                                  request_context=context(session_id=session, prefix_id=prefix,
                                                          phase="storage_replay", request_id=replay_id,
                                                          p_hash=prompt_hash(replay_prompt(prompt))),
                                  max_tokens=args.max_tokens)
        peers = await asyncio.gather(*peer_tasks)
    return {
        "run_id": args.run_id, "arm": args.arm, "session_id": session, "frontend_priority": "equal",
        "prompt_tokens_target": args.prompt_tokens, "wait_ms": args.wait_ms,
        "initial": initial, "host_precursor": host_status, "storage_eviction": evicted,
        "tool_start_ns": tool_start_ns, "tool_due_ns": due_ns, "tool_end_ns": tool_end_ns,
        "storage_stage": staged, "device_load": device_load, "replay": replay, "peers": peers,
        "due_to_first_token_ms": round((replay["first_token_ns"] - due_ns) / 1e6, 3),
        "tool_end_to_first_token_ms": round((replay["first_token_ns"] - tool_end_ns) / 1e6, 3),
        "stage_completed_before_due": staged is not None and staged["completed_observed_ns"] <= due_ns,
        "workflow_duration_ms": round((max([replay["request_end_ns"]] +
                                           [peer["request_end_ns"] for peer in peers])
                                       - initial["request_start_ns"]) / 1e6, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=("on_demand", "host_stage", "full_prepare"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=5000)
    parser.add_argument("--peer-count", type=int, default=0)
    parser.add_argument("--peer-start-ms", type=int, default=1000)
    parser.add_argument("--peer-prompt-tokens", type=int, default=1024)
    parser.add_argument("--peer-max-tokens", type=int, default=96)
    args = parser.parse_args()
    if args.prompt_tokens < 512 or args.wait_ms < 1 or args.peer_count < 0:
        parser.error("need a 512+ token prompt and a positive wait")
    result = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
