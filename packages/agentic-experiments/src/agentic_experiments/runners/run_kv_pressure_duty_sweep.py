#!/usr/bin/env python3
"""Measure a controlled spectrum of native KV-reload pressure during decode.

This is deliberately a hardware-pressure experiment, not a priority experiment.
All frontend requests have equal semantics.  The only condition difference is
the number of verified, native host-to-GPU KV reloads performed while the same
target response is decoding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import (
    context,
    eligible_host_prefix,
    prepare_prefix,
    prompt_hash,
    write_jsonl,
)
from .run_sustained_decode_kv_overlap import free_device_cache, stream_decode
from .run_sustained_kv_pressure import (
    count_load_windows_with_visible_overlap,
    interval_metrics_windows,
    load_donor,
    stage_donor_pool,
)


CONDITIONS = ("decode_control", "pressure_low", "pressure_medium", "pressure_high")


def target_prompt(trial_id: str) -> str:
    return (
        f"KV pressure duty sweep target {trial_id}. "
        "Continue by emitting the word token followed by one space repeatedly. "
        "Do not explain, summarize, number, punctuate, or stop early."
    )


async def recycle_donor(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    events: Path,
    trial_id: str,
    donor: dict[str, Any],
    sequence: int,
) -> dict[str, Any]:
    """Release one donor from device and prove its host-backed copy survived."""

    attempts: list[dict[str, Any]] = []
    requested_tokens = max(
        args.recycle_evict_tokens,
        int((donor.get("final_plan") or {}).get("host_tokens") or args.minimum_host_tokens),
    )
    for attempt in range(args.recycle_attempts):
        eviction = await free_device_cache(
            client,
            url=args.prepare_control_url,
            session_id=str(donor["session_id"]),
            prefix_id=str(donor["prefix_id"]),
            p_hash=str(donor["prompt_hash"]),
            request_id=f"{trial_id}-recycle-{sequence:03d}-donor-{int(donor['donor_index']):02d}-{attempt}",
            tokens=requested_tokens,
            source="kv_pressure_duty_sweep",
        )
        plan = await prepare_prefix(
            client,
            url=args.prepare_control_url,
            session_id=str(donor["session_id"]),
            prefix_id=str(donor["prefix_id"]),
            p_hash=str(donor["prompt_hash"]),
            request_id=f"{trial_id}-recycle-plan-{sequence:03d}-donor-{int(donor['donor_index']):02d}-{attempt}",
            plan_only=True,
            min_load_tokens=args.min_load_tokens,
            minimum_host_tokens=args.minimum_host_tokens,
        )
        row = {"attempt": attempt, "eviction": eviction, "plan": plan}
        attempts.append(row)
        if eligible_host_prefix(plan):
            donor["final_plan"] = plan
            result = {
                "ok": True,
                "donor_index": int(donor["donor_index"]),
                "load_sequence": sequence,
                "attempts": attempts,
                "host_tokens": int(plan.get("host_tokens") or 0),
            }
            write_jsonl(events, {"event": "kv_pressure_sweep.donor_recycled", "trial_id": trial_id, **result})
            return result
        if attempt + 1 < args.recycle_attempts:
            await asyncio.sleep(args.recycle_retry_ms / 1000)
    raise RuntimeError(
        f"{trial_id}: donor {donor['donor_index']} could not be recycled while retaining host backing: "
        f"{json.dumps(attempts, sort_keys=True)}"
    )


def validate_pressure(
    args: argparse.Namespace,
    decode: dict[str, Any],
    loads: list[dict[str, Any]],
    recycle_records: list[dict[str, Any]],
    regions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    if len(loads) != args.load_count:
        errors.append(f"expected {args.load_count} completed loads, observed {len(loads)}")
    expected_recycles = max(0, args.load_count - args.donor_count)
    if len(recycle_records) < expected_recycles:
        errors.append(f"expected at least {expected_recycles} donor recycle proofs, observed {len(recycle_records)}")

    request_start = int(decode["request_start_ns"])
    request_end = int(decode["request_end_ns"])
    windows: list[tuple[int, int]] = []
    for load in loads:
        start = load.get("start_ns")
        finish = load.get("finish_ns")
        cuda_ms = load.get("cuda_duration_ms")
        loaded_tokens = int(load.get("loaded_tokens") or 0)
        if not start or not finish or not request_start < int(start) < int(finish) < request_end:
            errors.append(f"load {load.get('load_sequence')} did not occur fully inside decode")
            continue
        windows.append((int(start), int(finish)))
        if not isinstance(cuda_ms, (int, float)) or float(cuda_ms) <= 0:
            errors.append(f"load {load.get('load_sequence')} has no positive CUDA duration")
        if loaded_tokens < args.minimum_host_tokens:
            errors.append(
                f"load {load.get('load_sequence')} moved {loaded_tokens} tokens; minimum is {args.minimum_host_tokens}"
            )

    overlapped = count_load_windows_with_visible_overlap(decode["chunk_times_ns"], windows)
    if args.load_count and overlapped != args.load_count:
        errors.append(f"only {overlapped} of {args.load_count} reloads intersected a visible decode interval")
    total_cuda_ms = sum(float(load.get("cuda_duration_ms") or 0.0) for load in loads)
    total_loaded_tokens = sum(int(load.get("loaded_tokens") or 0) for load in loads)
    first_start = min((start for start, _ in windows), default=None)
    last_finish = max((finish for _, finish in windows), default=None)
    envelope_ms = (last_finish - first_start) / 1_000_000 if first_start and last_finish else 0.0
    decode_ms = float(decode.get("total_latency_ms") or 0.0)
    cuda_share_pct = 100.0 * total_cuda_ms / decode_ms if decode_ms else 0.0
    envelope_share_pct = 100.0 * envelope_ms / decode_ms if decode_ms else 0.0
    if cuda_share_pct < args.minimum_cuda_load_share_pct:
        errors.append(
            f"CUDA load share {cuda_share_pct:.3f}% is below required {args.minimum_cuda_load_share_pct:.3f}%"
        )

    return {
        "valid": not errors,
        "errors": errors,
        "requested_load_count": args.load_count,
        "completed_loads": len(loads),
        "recycle_proofs": len(recycle_records),
        "total_loaded_tokens": total_loaded_tokens,
        "total_cuda_load_ms": round(total_cuda_ms, 3),
        "cuda_load_share_of_decode_pct": round(cuda_share_pct, 3),
        "pressure_start_ns": first_start,
        "pressure_finish_ns": last_finish,
        "pressure_envelope_ms": round(envelope_ms, 3),
        "pressure_envelope_share_of_decode_pct": round(envelope_share_pct, 3),
        "reload_windows_with_visible_overlap": overlapped,
        "overlapping_visible_intervals": int(regions["during"]["interval_count"]),
    }


async def run_trial(
    client: httpx.AsyncClient, args: argparse.Namespace, events: Path, trial_index: int
) -> dict[str, Any]:
    trial_id = f"{args.sample_set_id}-trial-{trial_index:03d}"
    donors, device_cache_prep = await stage_donor_pool(client, args, events, trial_id)
    warmup_ready = asyncio.Event()
    prompt = target_prompt(trial_id)
    request_id = f"{trial_id}-target-decode"
    decode_task = asyncio.create_task(
        stream_decode(
            client,
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            request_context=context(
                session_id=f"{trial_id}-target",
                prefix_id=f"{trial_id}-target-prefix",
                phase="kv_pressure_duty_sweep",
                request_id=request_id,
                p_hash=prompt_hash(prompt),
            ),
            max_tokens=args.decode_tokens,
            warmup_chunks=args.warmup_chunks,
            warmup_ready=warmup_ready,
        )
    )
    loads: list[dict[str, Any]] = []
    recycle_records: list[dict[str, Any]] = []
    try:
        if args.load_count:
            await asyncio.wait_for(warmup_ready.wait(), timeout=args.decode_timeout_s)
            for sequence in range(args.load_count):
                donor = donors[sequence % len(donors)]
                if sequence >= len(donors):
                    recycle_records.append(
                        await recycle_donor(client, args, events, trial_id, donor, sequence)
                    )
                loads.append(
                    await load_donor(
                        client, args, events, trial_id, donor, load_sequence=sequence
                    )
                )
        decode = await asyncio.wait_for(decode_task, timeout=args.decode_timeout_s)
    except Exception:
        decode_task.cancel()
        await asyncio.gather(decode_task, return_exceptions=True)
        raise

    if int(decode.get("completion_tokens") or 0) < args.decode_tokens:
        raise RuntimeError(f"{trial_id}: decode ended early")
    windows = [
        (int(load["start_ns"]), int(load["finish_ns"]))
        for load in loads
        if load.get("start_ns") and load.get("finish_ns")
    ]
    regions = interval_metrics_windows(decode["chunk_times_ns"], windows)
    proof = validate_pressure(args, decode, loads, recycle_records, regions)
    result = {
        "sample_id": f"trial_{trial_index:03d}",
        "condition": args.condition,
        "decode": decode,
        "loads": loads,
        "recycle_records": recycle_records,
        "regions": regions,
        "pressure_proof": proof,
        "donor_count": len(donors),
        "device_cache_prep": device_cache_prep,
    }
    write_jsonl(events, {"event": "kv_pressure_sweep.trial_complete", "trial_id": trial_id, **result})
    if not proof["valid"]:
        raise RuntimeError(f"{trial_id}: pressure proof failed: {'; '.join(proof['errors'])}")
    return result


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--load-count", type=int, required=True)
    parser.add_argument("--minimum-cuda-load-share-pct", type=float, default=0.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-set-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--workload-id", default="kv_pressure_duty_sweep_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=384)
    parser.add_argument("--warmup-chunks", type=int, default=24)
    parser.add_argument("--donor-count", type=int, default=8)
    parser.add_argument("--donor-prompt-tokens", type=int, default=8192)
    parser.add_argument("--eviction-prompt-tokens", type=int, default=8192)
    parser.add_argument("--eviction-rounds", type=int, default=8)
    parser.add_argument(
        "--direct-device-evict-for-stage",
        action="store_true",
        help="Stage donors with native device eviction, then prove host residency.",
    )
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--device-free-tokens", type=int, default=40_000)
    parser.add_argument("--recycle-evict-tokens", type=int, default=8192)
    parser.add_argument("--recycle-attempts", type=int, default=4)
    parser.add_argument("--recycle-retry-ms", type=int, default=25)
    parser.add_argument("--min-load-tokens", type=int, default=None)
    parser.add_argument("--prime-max-tokens", type=int, default=1)
    parser.add_argument("--load-timeout-ms", type=int, default=30_000)
    parser.add_argument("--decode-timeout-s", type=float, default=300.0)
    args = parser.parse_args()
    if args.condition == "decode_control" and args.load_count:
        parser.error("decode_control must use --load-count 0")
    if args.condition != "decode_control" and args.load_count < 1:
        parser.error("pressure conditions require at least one load")
    if min(args.trials, args.donor_count, args.decode_tokens, args.warmup_chunks) < 1:
        parser.error("trials, donors, decode tokens, and warmup chunks must be positive")
    if args.warmup_chunks >= args.decode_tokens:
        parser.error("warmup chunks must be smaller than decode tokens")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.out_dir / "kv_pressure_sweep_events.jsonl"
    probe_path = args.out_dir / "probe_run.json"
    existing: dict[str, Any] = {}
    if args.append:
        if not probe_path.is_file():
            parser.error("--append requires an existing probe_run.json")
        existing = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        event_path.write_text("", encoding="utf-8")
    timeout = httpx.Timeout(args.decode_timeout_s + 120, connect=10.0)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index in range(args.sample_offset, args.sample_offset + args.trials):
            results.append(await run_trial(client, args, event_path, index))
    probe_path.write_text(
        json.dumps(
            {
                "schema_version": "kv_pressure_duty_sweep.v1",
                "run_id": args.run_id,
                "condition": args.condition,
                "load_count": args.load_count,
                "sample_set_id": args.sample_set_id,
                "hardware_profile": args.hardware_profile,
                "backend_version": args.backend_version,
                "model": args.model,
                "workload_id": args.workload_id,
                "seed": args.seed,
                "trials": list(existing.get("trials", [])) + results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(main_async())
