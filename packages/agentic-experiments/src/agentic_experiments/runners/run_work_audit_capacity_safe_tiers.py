"""Run capacity-safe GPU, CPU, and storage KV-tier comparisons."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

import httpx

from .run_kv_movement_interference import context, make_prompt, prompt_hash
from .run_sustained_decode_kv_overlap import stream_decode
from .run_work_audit_storage import StorageControlRejected, control, stage_storage
from .run_work_audit_tool_cycles import tool_result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * fraction + 0.999999) - 1))]


async def sleep_until_ns(deadline_ns: int) -> None:
    await asyncio.sleep(max(0, (deadline_ns - time.time_ns()) / 1e9))


async def run(args: argparse.Namespace) -> dict[str, Any]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sessions = [{
        "session_id": f"capacity-seed{args.seed}-s{index:02d}",
        "prefix_id": f"capacity-seed{args.seed}-s{index:02d}",
        "prompt": make_prompt(f"capacity-seed{args.seed}-s{index:02d}", args.initial_tokens),
        "previous_request_id": "",
        "prompt_tokens": 0,
        "turns": [],
    } for index in range(args.sessions)]
    controls: list[dict[str, Any]] = []
    active: dict[str, int] = {}
    active_lock = asyncio.Lock()
    restore_lock = asyncio.Lock()
    slots = asyncio.Semaphore(args.max_active)
    max_active_observed = 0
    max_active_tokens_observed = 0
    capacity_violations: list[dict[str, Any]] = []
    workload_started_ns: int | None = None

    async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=32)) as client:
        def identity(session: dict[str, Any]) -> dict[str, str]:
            return {
                "session_id": session["session_id"], "prefix_id": session["prefix_id"],
                "request_id": session["previous_request_id"],
                "prompt_hash": prompt_hash(session["prompt"]),
            }

        async def recorded_control(session: dict[str, Any], action: str, **extra: Any) -> dict[str, Any]:
            started_ns = time.time_ns()
            try:
                value = await control(client, args.control_url, action, identity(session), **extra)
            except StorageControlRejected as exc:
                value = exc.result
                controls.append({"action": action, "session_id": session["session_id"],
                                 "started_ns": started_ns, "ended_ns": time.time_ns(), "result": value})
                raise
            controls.append({"action": action, "session_id": session["session_id"],
                             "started_ns": started_ns, "ended_ns": time.time_ns(), "result": value})
            return value

        async def residency(session: dict[str, Any]) -> dict[str, Any]:
            base = await recorded_control(session, "prefix_residency")
            if args.mode != "storage":
                return base
            storage = await recorded_control(session, "storage_status")
            return {**base, **storage,
                    "gpu_free_tokens": base.get("gpu_free_tokens"),
                    "cached_tokens": base.get("cached_tokens", storage.get("prefix_tokens")),
                    "registered_tokens": base.get("registered_tokens", storage.get("prefix_tokens"))}

        async def release_to_lower_tier(session: dict[str, Any]) -> dict[str, Any]:
            if args.mode == "resident":
                return {"release_ms": 0.0, "residency": await residency(session)}
            started_ns = time.time_ns()
            if args.mode == "storage":
                deadline = time.monotonic() + 30
                state = await residency(session)
                while int(state.get("gpu_tokens") or 0) > args.page_size:
                    try:
                        await recorded_control(
                            session, "evict_device",
                            tokens=max(session["prompt_tokens"], args.initial_tokens),
                        )
                    except StorageControlRejected as exc:
                        if exc.result.get("status") != "device_prefix_not_evicted":
                            raise
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"GPU-to-host storage precursor timed out: {state}")
                    await asyncio.sleep(0.02)
                    state = await residency(session)
                if int(state.get("host_tokens") or 0) >= args.page_size:
                    while True:
                        try:
                            await recorded_control(session, "evict_host")
                            break
                        except StorageControlRejected as exc:
                            if (exc.result.get("status") != "host_eviction_incomplete" or
                                    time.monotonic() >= deadline):
                                raise
                            await asyncio.sleep(0.05)
                    state = await residency(session)
                elif int(state.get("storage_candidate_tokens") or 0) < args.page_size:
                    raise RuntimeError(f"storage precursor reached neither CPU nor storage: {state}")
                if (int(state.get("gpu_tokens") or 0) > args.page_size or
                        int(state.get("host_tokens") or 0) > args.page_size or
                        int(state.get("storage_candidate_tokens") or 0) < args.page_size):
                    raise RuntimeError(f"storage-only release was not proved: {state}")
                return {"release_ms": (time.time_ns() - started_ns) / 1e6,
                        "residency": state}

            deadline = time.monotonic() + 30
            while True:
                try:
                    await recorded_control(session, "release_prefix")
                    break
                except StorageControlRejected as exc:
                    if exc.result.get("status") not in {
                        "prefix_busy", "host_backup_pending", "prefix_not_registered"
                    } or time.monotonic() >= deadline:
                        raise
                    await asyncio.sleep(0.02)
            state = await residency(session)
            if int(state.get("gpu_tokens") or 0) > args.page_size:
                raise RuntimeError(f"released session still has device KV: {state}")
            if args.mode == "host" and int(state.get("host_tokens") or 0) < (
                int(state.get("cached_tokens") or 0) - args.page_size
            ):
                raise RuntimeError(f"CPU-tier release did not preserve host KV: {state}")
            return {"release_ms": (time.time_ns() - started_ns) / 1e6, "residency": state}

        async def restore_to_gpu(session: dict[str, Any], turn: int) -> dict[str, Any]:
            started_ns = time.time_ns()
            state = await residency(session)
            before = dict(state)
            if int(state.get("storage_candidate_tokens") or 0) >= args.page_size:
                source_tier = "storage"
            elif int(state.get("host_tokens") or 0) >= args.page_size:
                source_tier = "host"
            else:
                source_tier = "gpu"
            staged = None
            host_room_evictions: list[dict[str, Any]] = []
            if int(state.get("storage_candidate_tokens") or 0) >= args.page_size:
                room_attempts = 0
                storage_id = f"{session['session_id']}-turn{turn:02d}-storage"
                while True:
                    try:
                        staged = await stage_storage(
                            client, args.control_url, identity(session), storage_id,
                        )
                        break
                    except StorageControlRejected as exc:
                        if exc.result.get("status") != "host_capacity_insufficient":
                            raise
                        room_attempts += 1
                        if room_attempts > len(sessions) * 2:
                            raise RuntimeError(f"host staging room did not increase enough: {exc.result}")
                    async with active_lock:
                        active_ids = set(active)
                    donor = None
                    donor_state = None
                    for candidate in sessions:
                        if candidate is session or candidate["session_id"] in active_ids:
                            continue
                        candidate_state = await residency(candidate)
                        if int(candidate_state.get("host_tokens") or 0) >= args.page_size:
                            donor, donor_state = candidate, candidate_state
                            break
                    if donor is None:
                        raise RuntimeError(
                            f"no inactive host-backed prefix can make staging room: {exc.result}"
                        )
                    free_before = int(exc.result.get("host_free_tokens") or 0)
                    try:
                        evicted = await recorded_control(donor, "evict_host")
                    except StorageControlRejected as exc:
                        if exc.result.get("status") != "host_eviction_incomplete":
                            raise
                        evicted = exc.result
                    host_room_evictions.append({
                        "session_id": donor["session_id"], "before": donor_state, "result": evicted,
                    })
                    # The next native prefetch attempt reports whether global
                    # host capacity actually increased. Do not infer it from
                    # the nominated donor, because HiCache evicts globally.
                    host_room_evictions[-1]["reported_free_before"] = free_before
                completed = staged["completed"]
                # storage_prefetch_status publishes the exact host node that
                # prepare must consume. A generic residency lookup here can
                # replace that pointer with the radix root for partial tails.
                state = {
                    **state,
                    "gpu_tokens": int(completed.get("gpu_tokens_after") or 0),
                    "host_tokens": int(completed.get("host_tokens_after") or 0),
                    "cached_tokens": int(state.get("prefix_tokens") or
                                         state.get("registered_tokens") or 0),
                }
            cached = int(state.get("cached_tokens") or state.get("registered_tokens") or
                         state.get("prefix_tokens") or 0)
            gpu = int(state.get("gpu_tokens") or 0)
            if cached <= 0:
                raise RuntimeError(f"prefix disappeared before admission: {state}")
            load = None
            if gpu < cached - args.page_size:
                missing = cached - gpu
                if missing > int(state.get("gpu_free_tokens") or 0):
                    raise RuntimeError(f"restoration would exceed free GPU KV: {missing} > {state.get('gpu_free_tokens')}")
                try:
                    load = await recorded_control(
                        session, "prepare", whole_prefix=True, wait=False,
                        min_load_tokens=1, minimum_host_tokens=1, mem_quota=missing,
                    )
                except StorageControlRejected as exc:
                    # The v0.5.10 adapter reports an accepted asynchronous
                    # load as ok=false/status=queued. Its load_id is the
                    # authoritative handle that must be followed to completion.
                    if exc.result.get("status") != "queued" or not exc.result.get("load_id"):
                        raise
                    load = exc.result
                load_id = load.get("load_id")
                if not load_id:
                    raise RuntimeError(f"native load was not accepted: {load}")
                deadline = time.monotonic() + 60
                while True:
                    status = await recorded_control(session, "load_status", load_id=load_id)
                    if status.get("native_finished") or status.get("status") == "finished":
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"device restoration timed out: {session['session_id']}")
                    await asyncio.sleep(0.01)
                state = await residency(session)
            ready_ns = time.time_ns()
            cached = int(state.get("cached_tokens") or state.get("registered_tokens") or
                         state.get("prefix_tokens") or 0)
            gpu = int(state.get("gpu_tokens") or 0)
            if gpu < cached - args.page_size or state.get("pending_loads"):
                raise RuntimeError(f"complete GPU prefix was not ready before admission: {state}")
            return {
                "started_ns": started_ns, "ready_ns": ready_ns,
                "duration_ms": (ready_ns - started_ns) / 1e6,
                "source_tier": source_tier, "before": before,
                "host_room_evictions": host_room_evictions,
                "storage_stage": staged, "device_load": load,
                "residency": state,
            }

        async def request(session: dict[str, Any], turn: int, due_ns: int | None) -> dict[str, Any]:
            request_id = f"{session['session_id']}-turn{turn:02d}"
            response = await stream_decode(
                client, base_url=args.base_url, model=args.model, prompt=session["prompt"],
                request_context=context(
                    session_id=session["session_id"], prefix_id=session["prefix_id"],
                    request_id=request_id, p_hash=prompt_hash(session["prompt"]),
                    phase="capacity_prime" if turn == 0 else "capacity_replay",
                ),
                max_tokens=2 if turn == 0 else args.decode_tokens,
                warmup_chunks=1, warmup_ready=asyncio.Event(),
            )
            if not response["chunk_times_ns"] or not response["usage"].get("prompt_tokens"):
                raise RuntimeError(f"missing response timing or usage: {request_id}")
            if turn and response["completion_tokens"] != args.decode_tokens:
                raise RuntimeError(f"unequal output work: {request_id}")
            session["previous_request_id"] = request_id
            session["prompt_tokens"] = int(response["usage"]["prompt_tokens"])
            return {
                "request_id": request_id, "session_id": session["session_id"], "turn": turn,
                "request_start_ns": response["request_start_ns"],
                "first_token_ns": response["chunk_times_ns"][0],
                "request_end_ns": response["request_end_ns"],
                "prompt_tokens": session["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "ttft_ms": response["ttft_ms"],
                "submission_delay_ms": max(0.0, (response["request_start_ns"] - due_ns) / 1e6)
                if due_ns is not None else None,
                "due_to_first_token_ms": (response["chunk_times_ns"][0] - due_ns) / 1e6
                if due_ns is not None else None,
            }

        # Initial population is excluded from workload time, as in RQ23.
        for session in sessions:
            prime = await request(session, 0, None)
            session["turns"].append(prime)
            if args.mode != "resident":
                prime["lower_tier_release"] = await release_to_lower_tier(session)

        # Prove the initial placement before measuring.
        initial_residency = [await residency(session) for session in sessions]
        if args.mode == "resident" and any(
            int(row.get("gpu_tokens") or 0) < int(row.get("cached_tokens") or 0) - args.page_size
            for row in initial_residency
        ):
            raise RuntimeError("all-GPU reference did not retain every initial prefix")

        workload_started_ns = time.time_ns()
        all_replays: list[dict[str, Any]] = []

        async def replay_one(session: dict[str, Any], turn: int, due_ns: int) -> dict[str, Any]:
            nonlocal max_active_observed, max_active_tokens_observed
            await sleep_until_ns(due_ns)
            ready_ns = time.time_ns()
            slot_wait_started_ns = ready_ns
            async with slots:
                slot_acquired_ns = time.time_ns()
                preparation = None
                if args.mode != "resident":
                    async with restore_lock:
                        preparation = await restore_to_gpu(session, turn)
                else:
                    state = await residency(session)
                    cached = int(state.get("cached_tokens") or 0)
                    if int(state.get("gpu_tokens") or 0) < cached - args.page_size:
                        raise RuntimeError(f"resident prefix left GPU: {state}")
                    preparation = {"started_ns": slot_acquired_ns, "ready_ns": time.time_ns(),
                                   "duration_ms": 0.0, "source_tier": "gpu", "before": state,
                                   "residency": state}

                # Account conservatively for the new tool-result suffix before submission.
                projected_tokens = session["prompt_tokens"] + args.tool_words + args.admission_token_margin
                async with active_lock:
                    projected_total = sum(active.values()) + projected_tokens
                    projected_count = len(active) + 1
                    if projected_count > args.max_active or projected_total > args.active_token_limit:
                        violation = {"session_id": session["session_id"], "turn": turn,
                                     "projected_count": projected_count,
                                     "projected_tokens": projected_total,
                                     "limit": args.active_token_limit}
                        capacity_violations.append(violation)
                        raise RuntimeError(f"active capacity invariant failed: {violation}")
                    active[session["session_id"]] = projected_tokens
                    max_active_observed = max(max_active_observed, projected_count)
                    max_active_tokens_observed = max(max_active_tokens_observed, projected_total)

                session["prompt"] += tool_result(session["session_id"], turn, args.tool_words)
                try:
                    row = await request(session, turn, due_ns)
                finally:
                    async with active_lock:
                        active.pop(session["session_id"], None)
                release = await release_to_lower_tier(session) if args.mode != "resident" else None
                row.update({
                    "tool_ready_ns": ready_ns, "slot_wait_started_ns": slot_wait_started_ns,
                    "slot_acquired_ns": slot_acquired_ns,
                    "kv_ready_ns": preparation["ready_ns"],
                    "slot_wait_ms": (slot_acquired_ns - slot_wait_started_ns) / 1e6,
                    "kv_prepare_ms": preparation["duration_ms"],
                    "admission_to_first_token_ms":
                        (row["first_token_ns"] - preparation["ready_ns"]) / 1e6,
                    "projected_active_tokens_at_admission": projected_tokens,
                    "preparation": preparation, "lower_tier_release": release,
                })
                session["turns"].append(row)
                all_replays.append(row)
                return row

        for turn in range(1, args.turns + 1):
            anchor_ns = time.time_ns() + args.wait_ms * 1_000_000
            window_ms = args.burst_window_ms if args.pattern == "burst" else args.spread_window_ms
            offsets_ms = [round(index * window_ms / max(1, args.sessions - 1), 3)
                          for index in range(args.sessions)]
            await asyncio.gather(*(
                replay_one(session, turn, anchor_ns + int(offsets_ms[index] * 1_000_000))
                for index, session in enumerate(sessions)
            ))

    due_delays = [row["due_to_first_token_ms"] for row in all_replays]
    ttfts = [row["ttft_ms"] for row in all_replays]
    slot_waits = [row["slot_wait_ms"] for row in all_replays]
    preparations = [row["kv_prepare_ms"] for row in all_replays]
    return {
        "schema": "agentic_work_audit.capacity_safe_tiers.arm.v1",
        "status": "complete", "run_id": args.run_id, "seed": args.seed,
        "mode": args.mode, "pattern": args.pattern, "started_ns": workload_started_ns,
        "ended_ns": time.time_ns(),
        "config": vars(args) | {"out": str(args.out)},
        "workflow_duration_ms": (max(row["request_end_ns"] for row in all_replays) - workload_started_ns) / 1e6,
        "replay_count": len(all_replays), "sessions": sessions,
        "initial_residency": initial_residency, "controls": controls,
        "capacity": {"max_active_allowed": args.max_active,
                     "active_token_limit": args.active_token_limit,
                     "max_active_observed": max_active_observed,
                     "max_active_tokens_observed": max_active_tokens_observed,
                     "violations": capacity_violations},
        "metrics": {
            "mean_due_to_first_token_ms": statistics.mean(due_delays),
            "p95_due_to_first_token_ms": percentile(due_delays, 0.95),
            "total_due_to_first_token_ms": sum(due_delays),
            "mean_ttft_ms": statistics.mean(ttfts),
            "p95_ttft_ms": percentile(ttfts, 0.95), "total_ttft_ms": sum(ttfts),
            "mean_slot_wait_ms": statistics.mean(slot_waits),
            "p95_slot_wait_ms": percentile(slot_waits, 0.95),
            "total_slot_wait_ms": sum(slot_waits),
            "mean_kv_prepare_ms": statistics.mean(preparations),
            "p95_kv_prepare_ms": percentile(preparations, 0.95),
            "total_kv_prepare_ms": sum(preparations),
            "total_submission_delay_ms": sum(row["submission_delay_ms"] for row in all_replays),
            "output_tokens": sum(row["completion_tokens"] for row in all_replays),
        },
        "per_session_completion_ms": {
            session["session_id"]: (session["turns"][-1]["request_end_ns"] - workload_started_ns) / 1e6
            for session in sessions
        },
        "limitations": [
            "Synthetic equal-priority coding sessions and fixed tool waits.",
            "Lower-tier restoration begins only after tool return; this is an unprepared worst case.",
            "File-backed storage can be served by the operating-system page cache.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("resident", "host", "storage"), required=True)
    parser.add_argument("--pattern", choices=("spread", "burst"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--initial-tokens", type=int, default=4096)
    parser.add_argument("--tool-words", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=1000)
    parser.add_argument("--burst-window-ms", type=int, default=75)
    parser.add_argument("--spread-window-ms", type=int, default=1000)
    parser.add_argument("--max-active", type=int, default=2)
    parser.add_argument("--active-token-limit", type=int, default=12288)
    parser.add_argument("--admission-token-margin", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()
    if min(args.sessions, args.turns, args.initial_tokens, args.tool_words, args.decode_tokens,
           args.wait_ms, args.max_active, args.active_token_limit, args.page_size) <= 0:
        parser.error("workload sizes and capacity limits must be positive")
    try:
        result = asyncio.run(run(args))
    except Exception as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "case_failure.json").write_text(json.dumps({
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        }, indent=2) + "\n", encoding="utf-8")
        raise
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mode": result["mode"], "pattern": result["pattern"],
                      "workflow_duration_ms": result["workflow_duration_ms"],
                      "capacity": result["capacity"], "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
