"""Optimistic paired KV rotation; independent contexts, real native transfers."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

from agentic_backends.coordinated_audit import validated_runtime_settings
from agentic_controller.coordinated_swap import PairRotation
from .run_kv_movement_interference import context, make_prompt, prompt_hash, write_jsonl
from .run_sustained_decode_kv_overlap import stream_decode


def metrics(rows: list[dict], started_ns: int) -> dict:
    delays = sorted(row["due_to_first_token_ms"] for row in rows)
    return {
        "replays": len(rows),
        "workload_ms": (max(r["request_end_ns"] for r in rows) - started_ns) / 1e6,
        "mean_due_to_first_token_ms": statistics.mean(delays),
        "p95_due_to_first_token_ms": delays[max(0, (95 * len(delays) + 99) // 100 - 1)],
        "total_due_to_first_token_ms": sum(delays),
        "total_ttft_ms": sum(r["ttft_ms"] for r in rows),
        "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in rows),
        "total_submission_delay_ms": sum(r["submission_delay_ms"] for r in rows),
        "total_alignment_padding_ms": sum(r.get("alignment_padding_ms", 0) for r in rows),
        "output_tokens": sum(r["completion_tokens"] for r in rows),
    }


async def sleep_until_ns(deadline: int) -> None:
    await asyncio.sleep(max(0, (deadline - time.time_ns()) / 1e9))


def initial_prompt(index: int, tokens: int) -> str:
    header = make_prompt(f"private-session-{index:02d}", 1)
    return " ".join([header] + ["context"] * max(0, tokens - len(header.split())))


async def run(args: argparse.Namespace) -> dict:
    rotation = PairRotation(args.sessions, args.wait_ms)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    events = args.out_dir / "harness_events.jsonl"
    rows: list[dict] = []
    controls: list[dict] = []
    sessions = [{"id": f"swap-s{index:02d}", "pair": rotation.pair(index),
                 "prompt": initial_prompt(index, args.initial_tokens)}
                for index in range(args.sessions)]
    setup_started_ns = time.time_ns()
    result = {"schema": "agentic_work_audit.coordinated_swap.arm.v1", "status": "running",
              "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "mode": args.mode, "trial": args.trial, "turns": rows, "controls": controls,
              "setup_started_ns": setup_started_ns}

    async with httpx.AsyncClient(timeout=180, limits=httpx.Limits(max_connections=64)) as client:
        async def control(session: dict, action: str, **extra) -> dict:
            sent = time.time_ns()
            payload = {"action": action, "session_id": session["id"],
                       "prefix_id": session["id"], "prompt_hash": prompt_hash(session["prompt"]),
                       "request_id": f"{session['id']}-{action}-{sent}",
                       "control_timeout_ms": 30000, **extra}
            response = await client.post(args.control_url, json=payload)
            response.raise_for_status()
            value = response.json()
            record = {"event": "coordinated.control", "action": action, "session_id": session["id"],
                      "sent_ns": sent, "returned_ns": time.time_ns(), "result": value}
            controls.append(record)
            write_jsonl(events, record)
            return value

        async def residency(session: dict) -> dict:
            value = await control(session, "prefix_residency")
            if not value.get("ok"):
                raise RuntimeError(f"residency: {value}")
            if value["cached_tokens"] < value["registered_tokens"] - 64:
                raise RuntimeError(f"Prefix disappeared from both cache tiers: {value}")
            return value

        async def release(session: dict) -> dict:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                value = await control(session, "release_prefix")
                if value.get("ok"):
                    state = await residency(session)
                    if state["gpu_tokens"] > 64 or state["host_tokens"] < state["cached_tokens"]:
                        raise RuntimeError(f"Release did not leave host-backed private KV: {state}")
                    return value
                if value.get("status") not in {"prefix_busy", "host_backup_pending", "prefix_not_registered"}:
                    raise RuntimeError(f"release: {value}")
                await asyncio.sleep(.02)
            raise TimeoutError(f"Backup/release timeout for {session['id']}")

        async def restore(session: dict) -> dict:
            before = await residency(session)
            if before["gpu_tokens"] == before["cached_tokens"] and not before["pending_loads"]:
                return {"already_resident": True, "ready_ns": time.time_ns(), "loaded_tokens": 0}
            missing = before["cached_tokens"] - before["gpu_tokens"]
            if missing > before["gpu_free_tokens"]:
                raise RuntimeError(f"Restore would evict another session: {missing} > {before['gpu_free_tokens']}")
            value = await control(session, "prepare", whole_prefix=True, wait=False,
                                  min_load_tokens=1, minimum_host_tokens=1, mem_quota=missing)
            if not value.get("load_id"):
                raise RuntimeError(f"No native load accepted: {value}")
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                status = await control(session, "load_status", load_id=value["load_id"])
                if status.get("native_finished") or status.get("status") == "finished":
                    after = await residency(session)
                    if after["gpu_tokens"] != after["cached_tokens"] or after["pending_loads"]:
                        raise RuntimeError(f"Native completion did not publish the full prefix: {after}")
                    return {"ready_ns": time.time_ns(), "loaded_tokens": value["loaded_tokens"],
                            "cuda_ms": status.get("cuda_elapsed_ms"), "load_id": value["load_id"]}
                await asyncio.sleep(.01)
            raise TimeoutError(f"Restore timeout for {session['id']}")

        def member_commands(selected: list[dict], slot: int) -> list[dict]:
            return [{"session_id": s["id"], "prefix_id": s["id"],
                     "prompt_hash": prompt_hash(s["prompt"]), "request_id": f"{s['id']}-group-{slot}"}
                    for s in selected]

        async def release_group(selected: list[dict], slot: int) -> None:
            if args.control_style == "individual":
                for session in selected:
                    await release(session)
                return
            members = member_commands(selected, slot)
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                result = await control(selected[0], "release_group", prefixes=members)
                if result.get("ok"):
                    states = await control(selected[0], "residency_group", prefixes=members)
                    if not states.get("ok") or any(s["gpu_tokens"] > 64 or s["host_tokens"] < s["cached_tokens"]
                                                   for s in states["members"]):
                        raise RuntimeError(f"Group release did not preserve host-backed KV: {states}")
                    return
                if not result.get("members") or any(r.get("status") not in
                    {"released", "prefix_busy", "host_backup_pending"} for r in result["members"]):
                    raise RuntimeError(f"Group release failed: {result}")
                await asyncio.sleep(.02)
            raise TimeoutError("Group backup/release did not finish")

        async def restore_group(selected: list[dict], slot: int, due: int) -> None:
            if args.restore_style == "serial":
                for session in selected:
                    ready = await restore(session)
                    write_jsonl(events, {"event": "coordinated.restore_ready", "session_id": session["id"],
                                          "slot": slot, "tool_due_ns": due, **ready, "grouped": False})
                return
            members = member_commands(selected, slot)
            value = await control(selected[0], "prepare_group", prefixes=members)
            if not value.get("ok"):
                raise RuntimeError(f"Group restore failed: {value}")
            if value.get("load_id"):
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    status = await control(selected[0], "load_status", load_id=value["load_id"])
                    if status.get("native_finished"):
                        break
                    await asyncio.sleep(.01)
                else:
                    raise TimeoutError("Group copy did not finish")
            if args.control_style == "group":
                verified = await control(selected[0], "residency_group", prefixes=members)
                if not verified.get("ok"):
                    raise RuntimeError(f"Group residency failed: {verified}")
                afters = verified["members"]
            else:
                afters = [await residency(session) for session in selected]
            for session, after in zip(selected, afters):
                if after["gpu_tokens"] != after["cached_tokens"] or after["pending_loads"]:
                    raise RuntimeError(f"Group copy not ready for {session['id']}: {after}")
                write_jsonl(events, {"event": "coordinated.restore_ready", "session_id": session["id"],
                                      "slot": slot, "tool_due_ns": due, "ready_ns": time.time_ns(),
                                      "load_id": value.get("load_id"), "grouped": True})

        async def request(session: dict, turn: int, due_ns: int, prime: bool = False) -> dict:
            if not prime:
                session["prompt"] += f"\nTool result {turn}: " + "context " * args.tool_words
            request_id = f"{session['id']}-{'prime' if prime else turn}"
            response = await stream_decode(
                client, base_url=args.base_url, model=args.model, prompt=session["prompt"],
                request_context=context(session_id=session["id"], prefix_id=session["id"],
                                        request_id=request_id, p_hash=prompt_hash(session["prompt"]),
                                        phase="swap_prime" if prime else "swap_replay"),
                max_tokens=2 if prime else args.decode_tokens, warmup_chunks=1, warmup_ready=asyncio.Event())
            if not response["chunk_times_ns"] or not response["usage"].get("prompt_tokens"):
                raise RuntimeError(f"Missing response/usage: {request_id}")
            if not prime and response["completion_tokens"] != args.decode_tokens:
                raise RuntimeError(f"Unequal output work: {request_id}: {response['completion_tokens']}")
            row = {"event": "coordinated.replay", "request_id": request_id, "session_id": session["id"],
                   "pair": session["pair"], "turn": turn, "tool_due_ns": due_ns,
                   "request_start_ns": response["request_start_ns"],
                   "first_token_ns": response["chunk_times_ns"][0], "request_end_ns": response["request_end_ns"],
                   "prompt_tokens": response["usage"]["prompt_tokens"],
                   "completion_tokens": response["completion_tokens"], "usage": response["usage"],
                   "ttft_ms": response["ttft_ms"],
                   "submission_delay_ms": max(0, (response["request_start_ns"] - due_ns) / 1e6),
                   "due_to_first_token_ms": (response["chunk_times_ns"][0] - due_ns) / 1e6}
            if not prime:
                rows.append(row)
                write_jsonl(events, row)
            return row

        try:
            response = await client.get(args.base_url.removesuffix("/v1") + "/get_server_info")
            response.raise_for_status()
            info = response.json()
            result["backend_settings"] = validated_runtime_settings(
                info, io_backend=args.io_backend,
                gpu_tokens=args.resident_gpu_tokens if args.mode == "resident" else args.restricted_gpu_tokens)
            # Only setup is excluded. All measured restore/control/alignment time is retained.
            primes = []
            for session in sessions:
                primes.append(await request(session, 0, time.time_ns(), prime=True))
                # Warm both native copy directions before timing every mode,
                # including the normally managed baseline and resident control.
                if session is sessions[0]:
                    async def correctness_reply() -> dict:
                        response = await client.post(f"{args.base_url}/chat/completions", json={
                            "model": args.model, "messages": [{"role": "user", "content": session["prompt"]}],
                            "max_tokens": 8, "temperature": 0, "ignore_eos": True, "stream": False,
                            "custom_params": context(session_id=session["id"], prefix_id=session["id"],
                                phase="swap_correctness", request_id=f"check-{time.time_ns()}",
                                p_hash=prompt_hash(session["prompt"]))})
                        response.raise_for_status()
                        return response.json()
                    original = await correctness_reply()
                    await release(session)
                    await restore(session)
                    restored = await correctness_reply()
                    content = original["choices"][0]["message"]["content"]
                    result["correctness"] = {"before": original, "after": restored,
                        "equal_output": content == restored["choices"][0]["message"]["content"]}
                    if not content or not result["correctness"]["equal_output"]:
                        raise RuntimeError("Restored-prefix diagnostic output mismatch")
                if args.mode == "coordinated":
                    await release(session)
            initial_tokens = sum(p["prompt_tokens"] for p in primes)
            if initial_tokens <= args.restricted_gpu_tokens:
                raise RuntimeError("All twenty prefixes fit: this does not test CPU spill")
            peak_pair = (max(p["prompt_tokens"] for p in primes)
                         + args.turns * (args.tool_words + 12) + args.decode_tokens) * args.sessions // 2
            if peak_pair > args.restricted_gpu_tokens:
                raise RuntimeError(f"Pair needs approximately {peak_pair} tokens, over restricted cap")
            result["prime_prompt_tokens"] = initial_tokens
            result["estimated_peak_pair_tokens"] = peak_pair
            if args.mode == "coordinated":
                await restore_group(sessions[:args.sessions // 2], -1, time.time_ns() + 30_000_000_000)
            elif args.mode == "resident":
                for session in sessions:
                    state = await residency(session)
                    if state["gpu_tokens"] != state["cached_tokens"]:
                        raise RuntimeError("Resident reference is not resident")
            started = time.time_ns()
            result["started_ns"] = started
            result["setup_ms"] = (started - setup_started_ns) / 1e6
            wait_ns = args.wait_ms * 1_000_000

            if args.mode == "independent":
                async def independent_loop(session: dict, index: int) -> None:
                    due = started + (index * wait_ns // args.sessions)
                    for turn in range(1, args.turns + 1):
                        await sleep_until_ns(due)
                        row = await request(session, turn, due)
                        due = row["request_end_ns"] + wait_ns
                await asyncio.gather(*(independent_loop(s, i) for i, s in enumerate(sessions)))
            else:
                due = [started, started + wait_ns]
                for slot in range(args.turns * 2):
                    pair = slot % 2
                    selected = [s for s in sessions if s["pair"] == pair]
                    await sleep_until_ns(due[pair])
                    slot_started = time.time_ns()
                    replies = await asyncio.gather(*(request(s, slot // 2 + 1, due[pair]) for s in selected))
                    finished = max(r["request_end_ns"] for r in replies)
                    boundary = max(slot_started + wait_ns, finished)
                    for row in replies:
                        row["alignment_padding_ms"] = (boundary - row["request_end_ns"]) / 1e6
                    due[pair] = boundary + wait_ns
                    if slot == args.turns * 2 - 1:
                        break
                    # Start swapping after useful GPU work completes; hide it in remaining slot padding.
                    # The next pair's tool deadline stays fixed even if these real transfers run late.
                    if args.mode == "coordinated":
                        await release_group(selected, slot)
                        await restore_group([s for s in sessions if s["pair"] != pair], slot, due[1 - pair])
                    await sleep_until_ns(boundary)
                    print(f"{args.mode}: slot {slot + 1}/{args.turns * 2}, replays {len(rows)}", flush=True)
            result.update(status="complete", ended_ns=time.time_ns(), metrics=metrics(rows, started))
            result["final_residency"] = [await residency(s) for s in sessions]
            if args.mode == "resident" and any(s["gpu_tokens"] != s["cached_tokens"] for s in result["final_residency"]):
                raise RuntimeError("Resident reference lost KV")
        except Exception as exc:
            result.update(status="blocked", error=f"{type(exc).__name__}: {exc}", ended_ns=time.time_ns())
            raise
        finally:
            (args.out_dir / "case_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("independent", "coordinated", "resident"), required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--turns", type=int, default=40)
    parser.add_argument("--wait-ms", type=int, default=1000)
    parser.add_argument("--initial-tokens", type=int, default=8192)
    parser.add_argument("--tool-words", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--restricted-gpu-tokens", type=int, default=110592)
    parser.add_argument("--resident-gpu-tokens", type=int, default=262144)
    parser.add_argument("--io-backend", choices=("direct", "kernel"), default="direct")
    parser.add_argument("--restore-style", choices=("serial", "group"), default="group")
    parser.add_argument("--control-style", choices=("individual", "group"), default="group")
    parser.add_argument("--trace-profile", default="kv_lifecycle_counts")
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    if args.sessions != 20 or args.wait_ms != 1000:
        parser.error("This study fixes twenty sessions and 1,000 ms tool waits")
    if min(args.turns, args.initial_tokens, args.decode_tokens, args.tool_words, args.restricted_gpu_tokens) <= 0:
        parser.error("Sizes must be positive")
    print(json.dumps(asyncio.run(run(args))["metrics"], indent=2))


if __name__ == "__main__":
    main()
