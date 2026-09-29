#!/usr/bin/env python3
"""Measure active decode while native SGLang KV load-back is in flight.

All requests have equal frontend semantics.  The three conditions differ only
in whether a separately staged donor prefix is left on the host, loaded before
decode, or loaded after decode has already produced several streamed chunks.
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

from .run_kv_movement_interference import (
    context,
    prepare_prefix,
    prompt_hash,
    stage_host_resident_prefix,
    write_jsonl,
)


def target_prompt(trial_id: str) -> str:
    return (
        f"Decode contention probe {trial_id}. "
        "Continue by emitting the word token followed by one space repeatedly. "
        "Do not explain, summarize, number, punctuate, or stop early."
    )


async def stream_decode(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    prompt: str,
    request_context: dict[str, Any],
    max_tokens: int,
    warmup_chunks: int,
    warmup_ready: asyncio.Event,
) -> dict[str, Any]:
    request_start_ns = time.time_ns()
    chunk_times_ns: list[int] = []
    usage: dict[str, Any] = {}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "custom_params": request_context,
    }
    async with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload) as response:
        if response.is_error:
            detail = (await response.aread()).decode("utf-8", errors="replace")[:1_000]
            raise RuntimeError(f"SGLang decode request failed with HTTP {response.status_code}: {detail}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            body = line.removeprefix("data: ").strip()
            if body == "[DONE]":
                break
            try:
                event = json.loads(body)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                content = event.get("choices", [{}])[0].get("delta", {}).get("content")
            except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
                content = None
            if content in (None, ""):
                continue
            chunk_times_ns.append(time.time_ns())
            if len(chunk_times_ns) >= warmup_chunks:
                warmup_ready.set()
    request_end_ns = time.time_ns()
    if not warmup_ready.is_set():
        warmup_ready.set()
    return {
        "request_start_ns": request_start_ns,
        "request_end_ns": request_end_ns,
        "chunk_times_ns": chunk_times_ns,
        "stream_chunks": len(chunk_times_ns),
        "usage": usage,
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_ms": round((chunk_times_ns[0] - request_start_ns) / 1_000_000, 3) if chunk_times_ns else None,
        "total_latency_ms": round((request_end_ns - request_start_ns) / 1_000_000, 3),
    }


async def load_status(client: httpx.AsyncClient, url: str, load_id: str) -> dict[str, Any]:
    response = await client.post(url, json={"action": "load_status", "load_id": load_id})
    result = response.json()
    result["http_status"] = response.status_code
    return result


async def free_device_cache(
    client: httpx.AsyncClient,
    *,
    url: str,
    session_id: str,
    prefix_id: str,
    p_hash: str,
    request_id: str,
    tokens: int,
) -> dict[str, Any]:
    response = await client.post(
        url,
        json={
            "action": "evict_device",
            "session_id": session_id,
            "prefix_id": prefix_id,
            "prompt_hash": p_hash,
            "request_id": request_id,
            "tokens": tokens,
            "control_timeout_ms": 15_000,
            "source": "sustained_decode_kv_overlap",
        },
    )
    result = response.json()
    result["http_status"] = response.status_code
    if not result.get("ok"):
        raise RuntimeError(f"{request_id}: device-cache preparation failed: {json.dumps(result, sort_keys=True)}")
    return result


async def wait_for_load(
    client: httpx.AsyncClient,
    *,
    url: str,
    load_id: str,
    timeout_ms: int,
    event_path: Path,
    trial_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    deadline = time.monotonic() + timeout_ms / 1000
    history: list[dict[str, Any]] = []
    previous = ""
    while time.monotonic() < deadline:
        status = await load_status(client, url, load_id)
        state = str(status.get("status") or "")
        if state != previous:
            history.append(status)
            write_jsonl(
                event_path,
                {"event": "decode_overlap.load_status", "trial_id": trial_id, "result": status},
            )
            previous = state
        if state == "finished":
            return status, history
        if not status.get("ok"):
            raise RuntimeError(f"{trial_id}: load status failed: {json.dumps(status, sort_keys=True)}")
        await asyncio.sleep(0.001)
    raise RuntimeError(f"{trial_id}: load {load_id} did not finish within {timeout_ms} ms")


def interval_metrics(chunk_times_ns: list[int], start_ns: int | None, finish_ns: int | None) -> dict[str, Any]:
    """Measure client-visible stream updates; these are not claimed as one update per token."""
    regions: dict[str, list[float]] = {"before": [], "during": [], "after": []}
    for previous, current in zip(chunk_times_ns, chunk_times_ns[1:]):
        latency_ms = (current - previous) / 1_000_000
        if start_ns is None or current <= start_ns:
            regions["before"].append(latency_ms)
        elif finish_ns is not None and previous >= finish_ns:
            regions["after"].append(latency_ms)
        else:
            # Count a stream interval when any portion intersects the active
            # CUDA load window, including a short load fully contained between
            # two client-visible updates.
            regions["during"].append(latency_ms)

    out: dict[str, Any] = {}
    for name, values in regions.items():
        ordered = sorted(values)
        p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered)) - 1)) if ordered else 0
        out[name] = {
            "interval_count": len(values),
            "median_itl_ms": round(statistics.median(values), 3) if values else None,
            "p95_itl_ms": round(ordered[p95_index], 3) if values else None,
            "max_itl_ms": round(max(values), 3) if values else None,
            "tokens_per_second": round(1000.0 / statistics.mean(values), 3) if values and statistics.mean(values) > 0 else None,
        }
    return out


async def run_trial(client: httpx.AsyncClient, args: argparse.Namespace, events: Path, index: int) -> dict[str, Any]:
    trial_id = f"{args.sample_set_id}-trial-{index:03d}"
    plan, donor_session, donor_prefix, donor_hash = await stage_host_resident_prefix(client, args, events, trial_id)
    device_cache_prep: dict[str, Any] | None = None
    if args.device_free_tokens > 0:
        device_cache_prep = await free_device_cache(
            client,
            url=args.prepare_control_url,
            session_id=donor_session,
            prefix_id=donor_prefix,
            p_hash=donor_hash,
            request_id=f"{trial_id}-free-device-cache",
            tokens=args.device_free_tokens,
        )
        write_jsonl(
            events,
            {"event": "decode_overlap.device_cache_prepared", "trial_id": trial_id, "result": device_cache_prep},
        )
    load_result: dict[str, Any] | None = None
    final_status: dict[str, Any] | None = None
    status_history: list[dict[str, Any]] = []

    if args.condition == "non_overlap_reload":
        load_result = await prepare_prefix(
            client,
            url=args.prepare_control_url,
            session_id=donor_session,
            prefix_id=donor_prefix,
            p_hash=donor_hash,
            request_id=f"{trial_id}-predecode-load",
            plan_only=False,
            min_load_tokens=args.min_load_tokens,
            minimum_host_tokens=args.minimum_host_tokens,
        )
        load_id = str(load_result.get("load_id") or "")
        if not load_id:
            raise RuntimeError(f"{trial_id}: non-overlap load was not accepted: {json.dumps(load_result)}")
        final_status, status_history = await wait_for_load(
            client,
            url=args.prepare_control_url,
            load_id=load_id,
            timeout_ms=args.load_timeout_ms,
            event_path=events,
            trial_id=trial_id,
        )

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
                phase="sustained_decode",
                request_id=request_id,
                p_hash=prompt_hash(prompt),
            ),
            max_tokens=args.decode_tokens,
            warmup_chunks=args.warmup_chunks,
            warmup_ready=warmup_ready,
        )
    )

    if args.condition == "direct_overlap_reload":
        await asyncio.wait_for(warmup_ready.wait(), timeout=args.decode_timeout_s)
        load_result = await prepare_prefix(
            client,
            url=args.prepare_control_url,
            session_id=donor_session,
            prefix_id=donor_prefix,
            p_hash=donor_hash,
            request_id=f"{trial_id}-overlap-load",
            plan_only=False,
            min_load_tokens=args.min_load_tokens,
            minimum_host_tokens=args.minimum_host_tokens,
        )
        load_id = str(load_result.get("load_id") or "")
        if not load_id:
            decode_task.cancel()
            raise RuntimeError(f"{trial_id}: overlap load was not accepted: {json.dumps(load_result)}")
        final_status, status_history = await wait_for_load(
            client,
            url=args.prepare_control_url,
            load_id=load_id,
            timeout_ms=args.load_timeout_ms,
            event_path=events,
            trial_id=trial_id,
        )

    decode = await asyncio.wait_for(decode_task, timeout=args.decode_timeout_s)
    active_status = next((row for row in status_history if row.get("status") == "active"), None)
    load_started_ns = int((active_status or final_status or {}).get("observed_ns") or 0) or None
    if active_status is None:
        load_started_ns = int(
            (final_status or {}).get("command_started_ns")
            or (load_result or {}).get("command_started_ns")
            or (load_result or {}).get("control_request_started_ns")
            or 0
        ) or None
    load_finished_ns = int((final_status or {}).get("observed_ns") or 0) or None
    cuda_elapsed_ms = (final_status or {}).get("cuda_elapsed_ms")
    regions = interval_metrics(decode["chunk_times_ns"], load_started_ns, load_finished_ns)
    during_count = int(regions["during"]["interval_count"])
    valid_overlap = args.condition != "direct_overlap_reload" or (
        final_status is not None
        and final_status.get("status") == "finished"
        and isinstance(cuda_elapsed_ms, (int, float))
        and cuda_elapsed_ms > 0
        and during_count >= args.minimum_overlap_intervals
        and decode["request_start_ns"] < load_started_ns < load_finished_ns < decode["request_end_ns"]
    )
    result = {
        "sample_id": f"trial_{index:03d}",
        "condition": args.condition,
        "request_id": request_id,
        "decode": decode,
        "regions": regions,
        "donor_plan": plan,
        "device_cache_prep": device_cache_prep,
        "load_result": load_result,
        "load_status_history": status_history,
        "load_started_ns": load_started_ns,
        "load_finished_ns": load_finished_ns,
        "overlap_duration_ms": round((load_finished_ns - load_started_ns) / 1_000_000, 3)
        if load_started_ns and load_finished_ns
        else None,
        "cuda_load_duration_ms": cuda_elapsed_ms,
        "active_status_observed": active_status is not None,
        "valid_overlap": valid_overlap,
    }
    write_jsonl(events, {"event": "decode_overlap.trial_complete", "trial_id": trial_id, **result})
    if not valid_overlap:
        raise RuntimeError(
            f"{trial_id}: direct-overlap proof failed; active_seen={active_status is not None}, "
            f"cuda_elapsed_ms={cuda_elapsed_ms}, during_intervals={during_count}, load_finished={bool(final_status)}"
        )
    return result


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        choices=("decode_control", "non_overlap_reload", "direct_overlap_reload"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample-set-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--workload-id", default="sustained_decode_kv_overlap_v1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--warmup-chunks", type=int, default=24)
    parser.add_argument("--minimum-overlap-intervals", type=int, default=1)
    parser.add_argument("--donor-prompt-tokens", type=int, default=4090)
    parser.add_argument("--eviction-prompt-tokens", type=int, default=8192)
    parser.add_argument("--eviction-rounds", type=int, default=8)
    parser.add_argument("--minimum-host-tokens", type=int, default=512)
    parser.add_argument("--device-free-tokens", type=int, default=0)
    parser.add_argument("--min-load-tokens", type=int, default=None)
    parser.add_argument("--prime-max-tokens", type=int, default=1)
    parser.add_argument("--load-timeout-ms", type=int, default=30_000)
    parser.add_argument("--decode-timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.trials < 1 or args.decode_tokens < 8 or args.warmup_chunks < 1:
        parser.error("trials, decode tokens, and warmup chunks must be positive")
    if args.warmup_chunks >= args.decode_tokens:
        parser.error("warmup chunks must be smaller than decode tokens")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_path = args.out_dir / "decode_overlap_events.jsonl"
    probe_path = args.out_dir / "probe_run.json"
    existing: dict[str, Any] = {}
    if args.append:
        if not probe_path.is_file():
            parser.error("--append requires an existing probe_run.json")
        existing = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        event_path.write_text("", encoding="utf-8")

    results: list[dict[str, Any]] = []
    timeout = httpx.Timeout(args.decode_timeout_s + 60, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index in range(args.sample_offset, args.sample_offset + args.trials):
            results.append(await run_trial(client, args, event_path, index))
    all_results = list(existing.get("trials", [])) + results
    payload = {
        "schema_version": "sustained_decode_kv_overlap.v1",
        "run_id": args.run_id,
        "condition": args.condition,
        "sample_set_id": args.sample_set_id,
        "hardware_profile": args.hardware_profile,
        "backend_version": args.backend_version,
        "model": args.model,
        "workload_id": args.workload_id,
        "seed": args.seed,
        "trials": all_results,
    }
    probe_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main_async())
