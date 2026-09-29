#!/usr/bin/env python3
"""Measure decode while a train of native host-to-device KV loads runs.

Every request has equal frontend semantics. The only condition difference is
whether zero, one, or several already host-resident donor prefixes are loaded
after target decode has started.
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
    stage_host_resident_prefix,
    write_jsonl,
)
from .run_sustained_decode_kv_overlap import free_device_cache, stream_decode, wait_for_load


CONDITION_LOAD_COUNTS = {
    "decode_control": 0,
    "single_reload": 1,
    "sustained_reload": None,
}


def target_prompt(trial_id: str) -> str:
    return (
        f"Sustained KV pressure probe {trial_id}. "
        "Continue by emitting the word token followed by one space repeatedly. "
        "Do not explain, summarize, number, punctuate, or stop early."
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def interval_metrics_windows(
    chunk_times_ns: list[int], windows: list[tuple[int, int]]
) -> dict[str, dict[str, Any]]:
    """Group client-visible update intervals around exact reload windows."""

    regions: dict[str, list[float]] = {
        "before": [],
        "during": [],
        "between": [],
        "after": [],
    }
    if not windows:
        regions["before"] = [
            (current - previous) / 1_000_000
            for previous, current in zip(chunk_times_ns, chunk_times_ns[1:])
        ]
    else:
        first_start = windows[0][0]
        final_finish = windows[-1][1]
        for previous, current in zip(chunk_times_ns, chunk_times_ns[1:]):
            latency_ms = (current - previous) / 1_000_000
            if current <= first_start:
                region = "before"
            elif previous >= final_finish:
                region = "after"
            elif any(previous < finish and current > start for start, finish in windows):
                region = "during"
            else:
                region = "between"
            regions[region].append(latency_ms)

    result: dict[str, dict[str, Any]] = {}
    for name, values in regions.items():
        result[name] = {
            "interval_count": len(values),
            "median_interval_ms": round(statistics.median(values), 3) if values else None,
            "p95_interval_ms": round(percentile(values, 0.95), 3) if values else None,
            "max_interval_ms": round(max(values), 3) if values else None,
            "visible_updates_per_second": round(1000.0 / statistics.mean(values), 3)
            if values and statistics.mean(values) > 0
            else None,
        }
    return result


def count_load_windows_with_visible_overlap(
    chunk_times_ns: list[int], windows: list[tuple[int, int]]
) -> int:
    """Count physical reload windows intersecting a visible decode interval."""

    intervals = list(zip(chunk_times_ns, chunk_times_ns[1:]))
    return sum(
        1
        for start, finish in windows
        if any(previous < finish and current > start for previous, current in intervals)
    )


async def stage_donor_pool(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    events: Path,
    trial_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    donors: list[dict[str, Any]] = []
    for index in range(args.donor_count):
        donor_id = f"{trial_id}-pool-{index:02d}"
        plan, session_id, prefix_id, p_hash = await stage_host_resident_prefix(
            client, args, events, donor_id
        )
        donors.append(
            {
                "donor_index": index,
                "session_id": session_id,
                "prefix_id": prefix_id,
                "prompt_hash": p_hash,
                "initial_plan": plan,
            }
        )

    device_cache_prep: dict[str, Any] | None = None
    if args.device_free_tokens > 0:
        first = donors[0]
        device_cache_prep = await free_device_cache(
            client,
            url=args.prepare_control_url,
            session_id=first["session_id"],
            prefix_id=first["prefix_id"],
            p_hash=first["prompt_hash"],
            request_id=f"{trial_id}-free-device-cache",
            tokens=args.device_free_tokens,
            source="sustained_kv_pressure",
        )
        write_jsonl(
            events,
            {
                "event": "sustained_kv_pressure.device_cache_prepared",
                "trial_id": trial_id,
                "result": device_cache_prep,
            },
        )

    # Later donor creation and the common device-cache preparation can change
    # residency. Recheck the complete pool immediately before decode and
    # refuse partial pressure trains.
    for donor in donors:
        plan = await prepare_prefix(
            client,
            url=args.prepare_control_url,
            session_id=donor["session_id"],
            prefix_id=donor["prefix_id"],
            p_hash=donor["prompt_hash"],
            request_id=f"{trial_id}-pool-{donor['donor_index']:02d}-final-plan",
            plan_only=True,
            min_load_tokens=args.min_load_tokens,
            minimum_host_tokens=args.minimum_host_tokens,
        )
        donor["final_plan"] = plan
        if not eligible_host_prefix(plan):
            raise RuntimeError(
                f"{trial_id}: donor {donor['donor_index']} is not host-resident before decode: "
                f"{json.dumps(plan, sort_keys=True)}"
            )
    write_jsonl(
        events,
        {
            "event": "sustained_kv_pressure.donor_pool_ready",
            "trial_id": trial_id,
            "donor_count": len(donors),
            "host_tokens": [row["final_plan"].get("host_tokens") for row in donors],
        },
    )
    return donors, device_cache_prep


def load_start_ns(load_result: dict[str, Any], final_status: dict[str, Any]) -> int | None:
    value = final_status.get("command_started_ns") or load_result.get("command_started_ns")
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    return None


async def load_donor(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    events: Path,
    trial_id: str,
    donor: dict[str, Any],
    *,
    load_sequence: int | None = None,
) -> dict[str, Any]:
    index = int(donor["donor_index"])
    sequence = index if load_sequence is None else load_sequence
    result = await prepare_prefix(
        client,
        url=args.prepare_control_url,
        session_id=donor["session_id"],
        prefix_id=donor["prefix_id"],
        p_hash=donor["prompt_hash"],
        request_id=f"{trial_id}-load-{sequence:03d}-donor-{index:02d}",
        plan_only=False,
        min_load_tokens=args.min_load_tokens,
        minimum_host_tokens=args.minimum_host_tokens,
    )
    load_id = str(result.get("load_id") or "")
    if not load_id:
        raise RuntimeError(
            f"{trial_id}: donor {index} load was not accepted: {json.dumps(result, sort_keys=True)}"
        )
    final_status, history = await wait_for_load(
        client,
        url=args.prepare_control_url,
        load_id=load_id,
        timeout_ms=args.load_timeout_ms,
        event_path=events,
        trial_id=f"{trial_id}-load-{sequence:03d}-donor-{index:02d}",
    )
    start_ns = load_start_ns(result, final_status)
    finish_ns = int(final_status.get("observed_ns") or 0) or None
    record = {
        "donor_index": index,
        "load_sequence": sequence,
        "load_id": load_id,
        "result": result,
        "status_history": history,
        "start_ns": start_ns,
        "finish_ns": finish_ns,
        "observed_duration_ms": round((finish_ns - start_ns) / 1_000_000, 3)
        if start_ns and finish_ns
        else None,
        "cuda_duration_ms": final_status.get("cuda_elapsed_ms"),
        "loaded_tokens": final_status.get("loaded_tokens") or result.get("loaded_tokens"),
    }
    write_jsonl(
        events,
        {"event": "sustained_kv_pressure.load_complete", "trial_id": trial_id, **record},
    )
    return record


def validate_pressure(
    args: argparse.Namespace,
    decode: dict[str, Any],
    loads: list[dict[str, Any]],
    regions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected = 0 if args.condition == "decode_control" else (
        1 if args.condition == "single_reload" else args.donor_count
    )
    errors: list[str] = []
    if len(loads) != expected:
        errors.append(f"expected {expected} completed loads, observed {len(loads)}")

    request_start = int(decode["request_start_ns"])
    request_end = int(decode["request_end_ns"])
    for load in loads:
        start = load.get("start_ns")
        finish = load.get("finish_ns")
        cuda_ms = load.get("cuda_duration_ms")
        loaded_tokens = int(load.get("loaded_tokens") or 0)
        if not start or not finish or not request_start < int(start) < int(finish) < request_end:
            errors.append(f"load {load.get('donor_index')} did not occur fully inside decode")
        if not isinstance(cuda_ms, (int, float)) or float(cuda_ms) <= 0:
            errors.append(f"load {load.get('donor_index')} has no positive CUDA duration")
        if loaded_tokens < args.minimum_host_tokens:
            errors.append(
                f"load {load.get('donor_index')} moved {loaded_tokens} tokens; "
                f"minimum is {args.minimum_host_tokens}"
            )

    gaps_ms = [
        max(0.0, (int(current["start_ns"]) - int(previous["finish_ns"])) / 1_000_000)
        for previous, current in zip(loads, loads[1:])
        if previous.get("finish_ns") and current.get("start_ns")
    ]
    total_cuda_ms = sum(float(load.get("cuda_duration_ms") or 0.0) for load in loads)
    total_loaded_tokens = sum(int(load.get("loaded_tokens") or 0) for load in loads)
    during_intervals = int(regions["during"]["interval_count"])
    windows = [
        (int(load["start_ns"]), int(load["finish_ns"]))
        for load in loads
        if load.get("start_ns") and load.get("finish_ns")
    ]
    overlapped_loads = count_load_windows_with_visible_overlap(
        decode["chunk_times_ns"], windows
    )
    if args.condition != "decode_control" and overlapped_loads != expected:
        errors.append(
            f"only {overlapped_loads} of {expected} physical reload windows "
            "overlapped client-visible decode intervals"
        )
    if args.condition == "sustained_reload":
        if total_cuda_ms < args.minimum_total_cuda_load_ms:
            errors.append(
                f"total CUDA load duration {total_cuda_ms:.3f} ms is below "
                f"{args.minimum_total_cuda_load_ms:.3f} ms"
            )
        if gaps_ms and max(gaps_ms) > args.maximum_inter_load_gap_ms:
            errors.append(
                f"maximum inter-load gap {max(gaps_ms):.3f} ms exceeds "
                f"{args.maximum_inter_load_gap_ms:.3f} ms"
            )

    start_ns = min((int(load["start_ns"]) for load in loads if load.get("start_ns")), default=None)
    finish_ns = max((int(load["finish_ns"]) for load in loads if load.get("finish_ns")), default=None)
    envelope_ms = (finish_ns - start_ns) / 1_000_000 if start_ns and finish_ns else 0.0
    proof = {
        "valid": not errors,
        "errors": errors,
        "expected_loads": expected,
        "completed_loads": len(loads),
        "total_loaded_tokens": total_loaded_tokens,
        "total_cuda_load_ms": round(total_cuda_ms, 3),
        "pressure_start_ns": start_ns,
        "pressure_finish_ns": finish_ns,
        "pressure_envelope_ms": round(envelope_ms, 3),
        "pressure_share_of_decode_pct": round(
            100.0 * envelope_ms / float(decode["total_latency_ms"]), 3
        )
        if decode.get("total_latency_ms")
        else None,
        "inter_load_gaps_ms": [round(value, 3) for value in gaps_ms],
        "maximum_inter_load_gap_ms": round(max(gaps_ms), 3) if gaps_ms else None,
        "observed_load_duty_cycle_pct": round(
            100.0
            * sum(float(load.get("observed_duration_ms") or 0.0) for load in loads)
            / envelope_ms,
            3,
        )
        if envelope_ms > 0
        else None,
        "overlapping_visible_intervals": during_intervals,
        "reload_windows_with_visible_overlap": overlapped_loads,
    }
    return proof


async def run_trial(
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    events: Path,
    trial_index: int,
) -> dict[str, Any]:
    trial_id = f"{args.sample_set_id}-trial-{trial_index:03d}"
    donors, device_cache_prep = await stage_donor_pool(client, args, events, trial_id)
    selected_count = CONDITION_LOAD_COUNTS[args.condition]
    selected = donors[: args.donor_count if selected_count is None else selected_count]

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
                phase="sustained_kv_pressure",
                request_id=request_id,
                p_hash=prompt_hash(prompt),
            ),
            max_tokens=args.decode_tokens,
            warmup_chunks=args.warmup_chunks,
            warmup_ready=warmup_ready,
        )
    )

    loads: list[dict[str, Any]] = []
    try:
        if selected:
            await asyncio.wait_for(warmup_ready.wait(), timeout=args.decode_timeout_s)
            for donor in selected:
                loads.append(await load_donor(client, args, events, trial_id, donor))
        decode = await asyncio.wait_for(decode_task, timeout=args.decode_timeout_s)
    except Exception:
        decode_task.cancel()
        await asyncio.gather(decode_task, return_exceptions=True)
        raise

    if int(decode.get("completion_tokens") or 0) < args.decode_tokens:
        raise RuntimeError(
            f"{trial_id}: decode ended at {decode.get('completion_tokens')} tokens; "
            f"expected at least {args.decode_tokens}"
        )
    windows = [
        (int(load["start_ns"]), int(load["finish_ns"]))
        for load in loads
        if load.get("start_ns") and load.get("finish_ns")
    ]
    regions = interval_metrics_windows(decode["chunk_times_ns"], windows)
    proof = validate_pressure(args, decode, loads, regions)
    result = {
        "sample_id": f"trial_{trial_index:03d}",
        "condition": args.condition,
        "request_id": request_id,
        "decode": decode,
        "donor_count": len(donors),
        "device_cache_prep": device_cache_prep,
        "donor_plans": [donor["final_plan"] for donor in donors],
        "loads": loads,
        "regions": regions,
        "pressure_proof": proof,
    }
    write_jsonl(events, {"event": "sustained_kv_pressure.trial_complete", **result})
    if not proof["valid"]:
        raise RuntimeError(f"{trial_id}: pressure proof failed: {'; '.join(proof['errors'])}")
    return result


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=tuple(CONDITION_LOAD_COUNTS), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-set-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--workload-id", default="sustained_kv_pressure_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--warmup-chunks", type=int, default=24)
    parser.add_argument("--donor-count", type=int, default=8)
    parser.add_argument("--donor-prompt-tokens", type=int, default=4090)
    parser.add_argument("--eviction-prompt-tokens", type=int, default=8192)
    parser.add_argument("--eviction-rounds", type=int, default=8)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--device-free-tokens", type=int, default=40_000)
    parser.add_argument("--min-load-tokens", type=int, default=None)
    parser.add_argument("--prime-max-tokens", type=int, default=1)
    parser.add_argument("--minimum-total-cuda-load-ms", type=float, default=500.0)
    parser.add_argument("--maximum-inter-load-gap-ms", type=float, default=250.0)
    parser.add_argument("--load-timeout-ms", type=int, default=30_000)
    parser.add_argument("--decode-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if min(args.trials, args.donor_count, args.decode_tokens, args.warmup_chunks) < 1:
        parser.error("trials, donor count, decode tokens, and warmup chunks must be positive")
    if args.warmup_chunks >= args.decode_tokens:
        parser.error("warmup chunks must be smaller than decode tokens")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = args.out_dir / "sustained_kv_pressure_events.jsonl"
    probe = args.out_dir / "probe_run.json"
    existing: dict[str, Any] = {}
    if args.append:
        if not probe.is_file():
            parser.error("--append requires an existing probe_run.json")
        existing = json.loads(probe.read_text(encoding="utf-8"))
    else:
        events.write_text("", encoding="utf-8")

    timeout = httpx.Timeout(args.decode_timeout_s + 120, connect=10.0)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index in range(args.sample_offset, args.sample_offset + args.trials):
            results.append(await run_trial(client, args, events, index))
    payload = {
        "schema_version": "sustained_kv_pressure.v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "sample_set_id": args.sample_set_id,
        "hardware_profile": args.hardware_profile,
        "backend_version": args.backend_version,
        "model": args.model,
        "workload_id": args.workload_id,
        "seed": args.seed,
        "trials": list(existing.get("trials", [])) + results,
    }
    probe.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main_async())
