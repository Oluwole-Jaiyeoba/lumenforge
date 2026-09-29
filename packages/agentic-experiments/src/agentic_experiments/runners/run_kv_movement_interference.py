#!/usr/bin/env python3
"""Collect one control or competing-KV-movement hardware probe run.

This driver deliberately has no frontend priority labels.  It creates a replay
request and, only in the interference condition, asks the backend's prepared
prefix control path to load a separately staged, host-resident KV prefix.
The run fails if that prefix cannot be verified as eligible for load-back.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def make_prompt(label: str, target_tokens: int, *, unique: bool = False) -> str:
    header = (
        f"Agentic coding task {label}. Track repository context, tool results, failing tests, "
        "candidate files, and the next patch hypothesis. "
    )
    # One simple token per appended word keeps the configured prompt length
    # close to the tokenizer length. The older compound/indexed words expanded
    # into many sub-tokens and could accidentally exceed model context limits.
    token = "cache" if unique else "context"
    words = [header]
    while len(" ".join(words).split()) < max(1, target_tokens):
        words.append(token)
    return " ".join(words)


def replay_prompt(shared_prefix: str) -> str:
    """Keep the replay's large prefix byte-for-byte identical to its initial turn."""

    return shared_prefix + "\nTool result: the last test failed; resume the same task from the saved context."


def context(*, session_id: str, prefix_id: str, phase: str, request_id: str, p_hash: str) -> dict[str, Any]:
    # These are experiment identity fields, not semantic priority fields.
    return {
        "agentic_kv": {
            "session_id": session_id,
            "prefix_id": prefix_id,
            "phase": phase,
            "mode": "hardware_kv_movement_probe",
            "label": request_id,
            "request_id": request_id,
            "parent_run_id": session_id,
            "correlation_id": f"{session_id}:{phase}:{request_id}",
            "case_id": session_id,
            "prompt_hash": p_hash,
        },
        "request_context": {
            "request_id": request_id,
            "parent_run_id": session_id,
            "phase": phase,
            "case_id": session_id,
        },
    }


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row.setdefault("ts_ns", time.time_ns())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


async def completion(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    model: str,
    prompt: str,
    request_context: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    started_ns = time.time_ns()
    started = time.perf_counter()
    first_token: float | None = None
    chunks = 0
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "custom_params": request_context,
    }
    async with client.stream("POST", f"{base_url.rstrip('/')}/chat/completions", json=payload) as response:
        if response.is_error:
            detail = (await response.aread()).decode("utf-8", errors="replace")[:1_000]
            raise RuntimeError(f"SGLang request failed with HTTP {response.status_code}: {detail}")
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            if line.removeprefix("data: ").strip() == "[DONE]":
                break
            chunks += 1
            if first_token is None:
                first_token = time.perf_counter()
    ended = time.perf_counter()
    ended_ns = time.time_ns()
    if first_token is None:
        first_token = ended
    return {
        "request_start_ns": started_ns,
        "request_end_ns": ended_ns,
        "ttft_ms": round((first_token - started) * 1000, 3),
        "total_latency_ms": round((ended - started) * 1000, 3),
        "stream_chunks": chunks,
    }


async def prepare_prefix(
    client: httpx.AsyncClient,
    *,
    url: str,
    session_id: str,
    prefix_id: str,
    p_hash: str,
    request_id: str,
    plan_only: bool,
    min_load_tokens: int | None,
    minimum_host_tokens: int,
) -> dict[str, Any]:
    request_started_ns = time.time_ns()
    started = time.perf_counter()
    response = await client.post(
        url,
        json={
            "session_id": session_id,
            "prefix_id": prefix_id,
            "prompt_hash": p_hash,
            "request_id": request_id,
            "plan_only": plan_only,
            "wait": False,
            "wait_timeout_ms": 10_000,
            "control_timeout_ms": 15_000,
            "source": "hardware_kv_movement_interference",
            "min_load_tokens": min_load_tokens,
            "minimum_host_tokens": minimum_host_tokens,
        },
    )
    try:
        result = response.json()
    except json.JSONDecodeError:
        result = {"status": "non_json_response", "body": response.text[:500]}
    result["http_status"] = response.status_code
    result["control_request_started_ns"] = request_started_ns
    result["control_response_ns"] = time.time_ns()
    result["control_duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return result


def eligible_host_prefix(plan: dict[str, Any]) -> bool:
    return (
        bool(plan.get("ok"))
        and plan.get("status") == "would_load_back"
        and int(plan.get("host_tokens") or 0) >= int(plan.get("minimum_host_tokens") or 1)
    )


async def stage_host_resident_prefix(
    client: httpx.AsyncClient, args: argparse.Namespace, event_path: Path, trial_id: str
) -> tuple[dict[str, Any], str, str, str]:
    donor_session = f"{trial_id}-donor"
    donor_prefix = f"{trial_id}-donor-prefix"
    donor_request = f"{trial_id}-donor-prime"
    donor_prompt = make_prompt(donor_session, args.donor_prompt_tokens)
    donor_hash = prompt_hash(donor_prompt)
    await completion(
        client,
        base_url=args.base_url,
        model=args.model,
        prompt=donor_prompt,
        request_context=context(
            session_id=donor_session,
            prefix_id=donor_prefix,
            phase="donor_prime",
            request_id=donor_request,
            p_hash=donor_hash,
        ),
        max_tokens=args.prime_max_tokens,
    )
    write_jsonl(event_path, {"event": "hardware_probe.donor_primed", "trial_id": trial_id, "request_id": donor_request})

    latest_plan: dict[str, Any] = {}
    for round_index in range(args.eviction_rounds + 1):
        latest_plan = await prepare_prefix(
            client,
            url=args.prepare_control_url,
            session_id=donor_session,
            prefix_id=donor_prefix,
            p_hash=donor_hash,
            request_id=donor_request,
            plan_only=True,
            min_load_tokens=args.min_load_tokens,
            minimum_host_tokens=args.minimum_host_tokens,
        )
        write_jsonl(
            event_path,
            {"event": "hardware_probe.donor_plan", "trial_id": trial_id, "round": round_index, "result": latest_plan},
        )
        if eligible_host_prefix(latest_plan):
            return latest_plan, donor_session, donor_prefix, donor_hash
        if round_index == args.eviction_rounds:
            break
        evictor_id = f"{trial_id}-evictor-{round_index:02d}"
        evictor_prompt = make_prompt(evictor_id, args.eviction_prompt_tokens, unique=True)
        await completion(
            client,
            base_url=args.base_url,
            model=args.model,
            prompt=evictor_prompt,
            request_context=context(
                session_id=evictor_id,
                prefix_id=evictor_id,
                phase="eviction_pressure",
                request_id=evictor_id,
                p_hash=prompt_hash(evictor_prompt),
            ),
            max_tokens=args.prime_max_tokens,
        )
    raise RuntimeError(
        f"{trial_id}: donor prefix never became host-resident; final plan={json.dumps(latest_plan, sort_keys=True)}"
    )


async def run_trial(client: httpx.AsyncClient, args: argparse.Namespace, events: Path, trial_index: int) -> dict[str, Any]:
    # `sample_set_id`, unlike `run_id`, is deliberately shared by control and
    # interference. It anchors prompt text, cache keys, and sample identity.
    trial_id = f"{args.sample_set_id}-trial-{trial_index:03d}"
    plan, donor_session, donor_prefix, donor_hash = await stage_host_resident_prefix(client, args, events, trial_id)

    target_session = f"{trial_id}-target"
    target_prefix = f"{trial_id}-target-prefix"
    target_initial_id = f"{trial_id}-target-initial"
    target_replay_id = f"{trial_id}-target-replay"
    target_initial = make_prompt(target_session, args.target_prompt_tokens)
    target_replay = replay_prompt(target_initial)
    await completion(
        client,
        base_url=args.base_url,
        model=args.model,
        prompt=target_initial,
        request_context=context(
            session_id=target_session,
            prefix_id=target_prefix,
            phase="target_initial",
            request_id=target_initial_id,
            p_hash=prompt_hash(target_initial),
        ),
        max_tokens=args.prime_max_tokens,
    )
    await asyncio.sleep(args.replay_wait_ms / 1000)

    prepare_result: dict[str, Any] | None = None
    if args.condition == "interference":
        prepare_result = await prepare_prefix(
            client,
            url=args.prepare_control_url,
            session_id=donor_session,
            prefix_id=donor_prefix,
            p_hash=donor_hash,
            request_id=f"{trial_id}-interference-load",
            plan_only=False,
            min_load_tokens=args.min_load_tokens,
            minimum_host_tokens=args.minimum_host_tokens,
        )
        write_jsonl(
            events,
            {"event": "hardware_probe.interference_load", "trial_id": trial_id, "plan": plan, "result": prepare_result},
        )
        if int(prepare_result.get("loaded_tokens") or 0) <= 0 or prepare_result.get("status") not in {"queued", "ready"}:
            raise RuntimeError(f"{trial_id}: interference load was not admitted: {json.dumps(prepare_result, sort_keys=True)}")

    replay_ready_ns = time.time_ns()
    replay = await completion(
        client,
        base_url=args.base_url,
        model=args.model,
        prompt=target_replay,
        request_context=context(
            session_id=target_session,
            prefix_id=target_prefix,
            phase="replay",
            request_id=target_replay_id,
            p_hash=prompt_hash(target_replay),
        ),
        max_tokens=args.replay_max_tokens,
    )
    write_jsonl(
        events,
        {
            "event": "hardware_probe.replay_complete",
            "trial_id": trial_id,
            "condition": args.condition,
            "request_id": target_replay_id,
            "replay_ready_ns": replay_ready_ns,
            "metrics": replay,
            "interference_load": prepare_result,
        },
    )
    return {
        "sample_id": f"trial_{trial_index:03d}",
        "metrics_ms": {
            "replay_ttft_ms": replay["ttft_ms"],
            "replay_total_latency_ms": replay["total_latency_ms"],
            "replay_lateness_ms": round(max(0.0, replay["ttft_ms"] - args.replay_deadline_ms), 3),
        },
        "request_id": target_replay_id,
        "interference_load": prepare_result,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("control", "interference"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--prepare-control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware-profile", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--workload-id", default="hardware_kv_movement_v1")
    parser.add_argument(
        "--sample-set-id",
        required=True,
        help="Stable logical workload identifier shared by both paired conditions.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append distinct sample IDs to an existing condition result.",
    )
    parser.add_argument("--target-prompt-tokens", type=int, default=2048)
    # The chat wrapper contributes six tokens on the pinned reference runtime.
    # 4090 user tokens therefore lands the donor on a 4096-token boundary,
    # avoiding a tiny final radix node below SGLang's 10-token load threshold.
    parser.add_argument("--donor-prompt-tokens", type=int, default=4090)
    parser.add_argument("--eviction-prompt-tokens", type=int, default=8192)
    parser.add_argument("--eviction-rounds", type=int, default=8)
    parser.add_argument("--replay-wait-ms", type=int, default=1000)
    parser.add_argument("--replay-deadline-ms", type=float, default=2000.0)
    parser.add_argument(
        "--minimum-host-tokens",
        type=int,
        default=512,
        help="Reject a donor unless its selected evicted host-backed KV segment reaches this size.",
    )
    parser.add_argument(
        "--min-load-tokens",
        type=int,
        default=None,
        help="Explicit benchmark-only override for SGLang's native load-back threshold.",
    )
    parser.add_argument("--prime-max-tokens", type=int, default=1)
    parser.add_argument("--replay-max-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least one")
    if args.sample_offset < 0:
        parser.error("--sample-offset must be nonnegative")
    if min(args.target_prompt_tokens, args.donor_prompt_tokens, args.eviction_prompt_tokens) < 1:
        parser.error("prompt sizes must be positive")
    if args.eviction_rounds < 0:
        parser.error("--eviction-rounds must be nonnegative")
    if args.replay_deadline_ms < 0:
        parser.error("--replay-deadline-ms must be nonnegative")
    if args.minimum_host_tokens < 1:
        parser.error("--minimum-host-tokens must be positive")
    if args.min_load_tokens is not None and args.min_load_tokens < 1:
        parser.error("--min-load-tokens must be positive when set")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = args.out_dir / "hardware_probe_events.jsonl"
    probe_path = args.out_dir / "probe_run.json"
    existing_samples: list[dict[str, Any]] = []
    existing_details: list[dict[str, Any]] = []
    if args.append:
        if not probe_path.is_file():
            parser.error("--append requires an existing probe_run.json")
        existing = json.loads(probe_path.read_text(encoding="utf-8"))
        for field in ("condition", "hardware_profile", "backend_version", "model", "workload_id", "seed"):
            if str(existing.get(field)) != str(getattr(args, field)):
                parser.error(f"--append contract mismatch for {field}")
        existing_samples = list(existing.get("samples", []))
        existing_details = list(existing.get("probe_metadata", {}).get("trial_details", []))
    else:
        events.write_text("", encoding="utf-8")
    write_jsonl(
        events,
        {
            "event": "hardware_probe.start",
            "condition": args.condition,
            "run_id": args.run_id,
            "sample_set_id": args.sample_set_id,
            "min_load_tokens": args.min_load_tokens,
            "minimum_host_tokens": args.minimum_host_tokens,
        },
    )
    samples: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        for index in range(args.sample_offset, args.sample_offset + args.trials):
            samples.append(await run_trial(client, args, events, index))
    all_samples = existing_samples + [{"sample_id": row["sample_id"], "metrics_ms": row["metrics_ms"]} for row in samples]
    all_details = existing_details + samples
    if len({str(row["sample_id"]) for row in all_samples}) != len(all_samples):
        parser.error("combined samples contain duplicate sample IDs")
    probe_run = {
        "run_id": args.run_id,
        "condition": args.condition,
        "hardware_profile": args.hardware_profile,
        "backend_version": args.backend_version,
        "model": args.model,
        "workload_id": args.workload_id,
        "seed": args.seed,
        "instrumentation_profile": "lightweight_backend_trace",
        "samples": all_samples,
        "probe_metadata": {
            "sample_set_id": args.sample_set_id,
            "trial_details": all_details,
            "events": str(events),
        },
    }
    probe_path.write_text(json.dumps(probe_run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_jsonl(events, {"event": "hardware_probe.complete", "condition": args.condition, "sample_count": len(all_samples)})


if __name__ == "__main__":
    asyncio.run(main_async())
