#!/usr/bin/env python3
"""Attribute decode slowdown to SGLang's native host-to-device KV load-back.

The three cases deliberately have equal frontend semantics.  A separately
staged donor is either ignored, made device-resident before target decode, or
reloaded from host memory only after target decode is already active.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import (
    context,
    prepare_prefix,
    prompt_hash,
    stage_host_resident_prefix,
    write_jsonl,
)
from .run_sustained_decode_kv_overlap import stream_decode, wait_for_load


def target_prompt(trial_id: str, target_index: int) -> str:
    return (
        f"Device-memory attribution target {trial_id}-{target_index}. Emit the word token followed by one space repeatedly. "
        "Do not explain, summarize, number, punctuate, or stop early."
    )


def _finished_load(status: dict[str, Any] | None) -> bool:
    return bool(status and status.get("status") == "finished" and float(status.get("cuda_elapsed_ms") or 0) > 0)


async def _load_donor(
    client: httpx.AsyncClient, args: argparse.Namespace, events: Path, trial_id: str,
    donor_session: str, donor_prefix: str, donor_hash: str, label: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    request_id = f"{trial_id}-{label}"
    request = await prepare_prefix(
        client, url=args.prepare_control_url, session_id=donor_session, prefix_id=donor_prefix,
        p_hash=donor_hash, request_id=request_id, plan_only=False,
        min_load_tokens=args.min_load_tokens, minimum_host_tokens=args.minimum_host_tokens,
    )
    load_id = str(request.get("load_id") or "")
    if not load_id:
        raise RuntimeError(f"{trial_id}: {label} was not accepted: {json.dumps(request, sort_keys=True)}")
    final, history = await wait_for_load(
        client, url=args.prepare_control_url, load_id=load_id, timeout_ms=args.load_timeout_ms,
        event_path=events, trial_id=trial_id,
    )
    if not _finished_load(final):
        raise RuntimeError(f"{trial_id}: {label} did not complete native load-back: {json.dumps(final, sort_keys=True)}")
    return request, final, history


async def run_trial(client: httpx.AsyncClient, args: argparse.Namespace, events: Path, index: int) -> dict[str, Any]:
    trial_id = f"{args.sample_set_id}-trial-{index:03d}"
    donor_plan, donor_session, donor_prefix, donor_hash = await stage_host_resident_prefix(client, args, events, trial_id)
    preload: dict[str, Any] | None = None
    preload_final: dict[str, Any] | None = None
    preload_history: list[dict[str, Any]] = []
    if args.condition == "device_resident_control":
        preload, preload_final, preload_history = await _load_donor(
            client, args, events, trial_id, donor_session, donor_prefix, donor_hash, "predecode-device-load"
        )
        write_jsonl(events, {"event": "device_memory_attribution.device_preload", "trial_id": trial_id, "result": preload_final})

    active_events: list[asyncio.Event] = []
    decode_tasks: list[asyncio.Task[dict[str, Any]]] = []
    for target_index in range(args.target_concurrency):
        prompt = target_prompt(trial_id, target_index)
        active = asyncio.Event()
        active_events.append(active)
        decode_tasks.append(
            asyncio.create_task(
                stream_decode(
                    client, base_url=args.base_url, model=args.model, prompt=prompt,
                    request_context=context(
                        session_id=f"{trial_id}-target-{target_index}", prefix_id=f"{trial_id}-target-{target_index}-prefix",
                        phase="device_memory_attribution", request_id=f"{trial_id}-target-{target_index}-decode",
                        p_hash=prompt_hash(prompt),
                    ), max_tokens=args.decode_tokens, warmup_chunks=args.warmup_chunks, warmup_ready=active,
                )
            )
        )
    await asyncio.wait_for(asyncio.gather(*(active.wait() for active in active_events)), timeout=args.decode_timeout_s)

    collision_request: dict[str, Any] | None = None
    collision_final: dict[str, Any] | None = None
    collision_history: list[dict[str, Any]] = []
    resident_probe: dict[str, Any] | None = None
    if args.condition == "host_reload_collision":
        collision_request, collision_final, collision_history = await _load_donor(
            client, args, events, trial_id, donor_session, donor_prefix, donor_hash, "during-decode-host-reload"
        )
    elif args.condition == "device_resident_control":
        # Same coordinator checkpoint after decode begins, but only ask the
        # backend to describe the already resident donor; this must not queue H2D work.
        resident_probe = await prepare_prefix(
            client, url=args.prepare_control_url, session_id=donor_session, prefix_id=donor_prefix,
            p_hash=donor_hash, request_id=f"{trial_id}-during-decode-resident-check", plan_only=True,
            min_load_tokens=args.min_load_tokens, minimum_host_tokens=args.minimum_host_tokens,
        )
        # Once the donor was loaded before decode, the control endpoint no
        # longer finds an *evicted host-backed* node. That specific 409 is the
        # expected no-load proof, not a failed coordinator checkpoint.
        resident_no_load = (
            resident_probe.get("status") == "no_eligible_host_resident_prefix"
            and resident_probe.get("reason") == "No donor cache-path node was both evicted, host-backed, and large enough for this probe."
        )
        if not resident_probe.get("ok") and not resident_no_load:
            for decode_task in decode_tasks:
                decode_task.cancel()
            raise RuntimeError(f"{trial_id}: device-resident checkpoint failed: {json.dumps(resident_probe, sort_keys=True)}")

    target_decodes = await asyncio.wait_for(asyncio.gather(*decode_tasks), timeout=args.decode_timeout_s)
    decode = target_decodes[0]
    active_status = next((row for row in collision_history if row.get("status") == "active"), None)
    load_started_ns = int((active_status or collision_final or {}).get("observed_ns") or 0) or None
    if active_status is None and collision_request:
        load_started_ns = int(collision_final.get("command_started_ns") or collision_request.get("command_started_ns") or 0) or None
    load_finished_ns = int((collision_final or {}).get("observed_ns") or 0) or None
    native_overlap = bool(
        collision_final and load_started_ns and load_finished_ns
        and decode["request_start_ns"] < load_started_ns < load_finished_ns < decode["request_end_ns"]
        and _finished_load(collision_final)
    )
    device_control_valid = bool(
        args.condition != "device_resident_control"
        or (
            _finished_load(preload_final)
            and resident_probe
            and (resident_probe.get("ok") or resident_probe.get("status") == "no_eligible_host_resident_prefix")
            and not resident_probe.get("load_id")
        )
    )
    valid = native_overlap if args.condition == "host_reload_collision" else device_control_valid
    result = {
        "sample_id": f"trial_{index:03d}", "condition": args.condition,
        "decode": decode, "donor_plan": donor_plan,
        "preload_request": preload, "preload_final": preload_final, "preload_history": preload_history,
        "resident_probe": resident_probe,
        "collision_request": collision_request, "collision_final": collision_final,
        "collision_history": collision_history, "load_started_ns": load_started_ns,
        "load_finished_ns": load_finished_ns,
        "target_concurrency": args.target_concurrency, "target_decodes": target_decodes,
        "native_overlap": native_overlap, "valid": valid,
        "cuda_load_duration_ms": (collision_final or {}).get("cuda_elapsed_ms"),
        "loaded_tokens": (collision_request or {}).get("loaded_tokens"),
    }
    write_jsonl(events, {"event": "device_memory_attribution.trial_complete", "trial_id": trial_id, **result})
    if not valid:
        raise RuntimeError(f"{trial_id}: attribution proof failed for {args.condition}: {json.dumps(result, sort_keys=True)}")
    return result


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("target_only", "device_resident_control", "host_reload_collision"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-set-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--workload-id", default="device_memory_attribution_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--warmup-chunks", type=int, default=24)
    parser.add_argument("--target-concurrency", type=int, default=1)
    parser.add_argument("--donor-prompt-tokens", type=int, default=4090)
    parser.add_argument("--eviction-prompt-tokens", type=int, default=8192)
    parser.add_argument("--eviction-rounds", type=int, default=8)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--min-load-tokens", type=int, default=None)
    parser.add_argument("--prime-max-tokens", type=int, default=1)
    parser.add_argument("--load-timeout-ms", type=int, default=30_000)
    parser.add_argument("--decode-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.trials < 1 or args.decode_tokens <= args.warmup_chunks or args.warmup_chunks < 1 or args.target_concurrency < 1:
        parser.error("trials and target concurrency must be positive, and decode tokens must exceed warmup chunks")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = args.out_dir / "device_memory_attribution_events.jsonl"
    output = args.out_dir / "probe_run.json"
    existing: list[dict[str, Any]] = []
    if args.append:
        if not output.is_file():
            parser.error("--append requires an existing probe_run.json")
        existing = list(json.loads(output.read_text(encoding="utf-8")).get("trials") or [])
    else:
        events.write_text("", encoding="utf-8")
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.decode_timeout_s + 60, connect=10.0)) as client:
        results = [await run_trial(client, args, events, index) for index in range(args.sample_offset, args.sample_offset + args.trials)]
    payload = {
        "schema_version": "device_memory_attribution.v1", "run_id": args.run_id, "condition": args.condition,
        "sample_set_id": args.sample_set_id, "hardware_profile": args.hardware_profile,
        "backend_version": args.backend_version, "model": args.model, "workload_id": args.workload_id,
        "seed": args.seed, "trials": existing + results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main_async())
