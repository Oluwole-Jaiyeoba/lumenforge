"""Compare identical SGLang decodes with independent GPU copy pressure off/on."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from .run_sustained_decode_kv_overlap import stream_decode


def window_gaps_ms(chunks: list[int], start_ns: int, end_ns: int) -> list[float]:
    return [
        (right - left) / 1_000_000
        for left, right in zip(chunks, chunks[1:])
        if start_ns <= left and right <= end_ns
    ]


async def trial(client: httpx.AsyncClient, args: argparse.Namespace, mode: str, index: int) -> dict[str, Any]:
    active = asyncio.Event()
    prompt = (
        "Repeat the word token with a space until you reach the output limit. "
        "Do not add punctuation or stop early. Trial ID: " + f"{index:04d}"
    )
    decode_task = asyncio.create_task(
        stream_decode(
            client, base_url=args.base_url, model=args.model, prompt=prompt,
            request_context={"session_id": f"copy-contention-{index:04d}", "agent_phase": "decode"},
            max_tokens=args.decode_tokens, warmup_chunks=args.warmup_chunks, warmup_ready=active,
        )
    )
    try:
        await asyncio.wait_for(active.wait(), timeout=args.timeout_s)
        if decode_task.done():
            raise RuntimeError("Target finished before the pressure window could start")
        response = await client.post(
            f"{args.worker_url}/start", json={"mode": mode, "duration_s": args.window_s}
        )
        response.raise_for_status()
        decode = await asyncio.wait_for(decode_task, timeout=args.timeout_s)
    except BaseException:
        decode_task.cancel()
        raise
    deadline = time.monotonic() + args.window_s + 5
    while True:
        response = await client.get(f"{args.worker_url}/status")
        response.raise_for_status()
        pressure = response.json()
        if pressure.get("status") == "finished":
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Copy worker did not finish the pressure window")
        await asyncio.sleep(0.1)
    if not (decode["request_start_ns"] < pressure["started_ns"] < pressure["finished_ns"] < decode["request_end_ns"]):
        raise RuntimeError("Pressure window did not fit inside the target decode")
    gaps = window_gaps_ms(decode["chunk_times_ns"], pressure["started_ns"], pressure["finished_ns"])
    if len(gaps) < 3:
        raise RuntimeError("Too few decode chunks fell inside the pressure window")
    return {
        "mode": mode, "index": index, "decode": decode, "pressure": pressure,
        "window_chunk_gaps_ms": gaps,
        "window_mean_chunk_gap_ms": round(statistics.mean(gaps), 3),
        "window_median_chunk_gap_ms": round(statistics.median(gaps), 3),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if not modes or any(mode not in {"idle", "copy"} for mode in modes):
        raise ValueError("modes must be a comma-separated sequence of idle and copy")
    async with httpx.AsyncClient(timeout=httpx.Timeout(args.timeout_s + 20, connect=10.0)) as client:
        rows = []
        for index, mode in enumerate(modes):
            row = await trial(client, args, mode, index)
            rows.append(row)
            print(
                f"{mode} {index}: total={row['decode']['total_latency_ms']}ms, "
                f"window_gap={row['window_mean_chunk_gap_ms']}ms, copies={row['pressure']['copies']}",
                flush=True,
            )
    result = {
        "schema_version": "gpu_copy_contention.v1", "model": args.model,
        "decode_tokens": args.decode_tokens, "warmup_chunks": args.warmup_chunks,
        "window_s": args.window_s, "modes": modes, "trials": rows,
    }
    for mode in ("idle", "copy"):
        subset = [row for row in rows if row["mode"] == mode]
        if subset:
            result[f"{mode}_summary"] = {
                "trials": len(subset),
                "median_total_latency_ms": round(statistics.median(row["decode"]["total_latency_ms"] for row in subset), 3),
                "median_window_mean_chunk_gap_ms": round(statistics.median(row["window_mean_chunk_gap_ms"] for row in subset), 3),
                "total_copies": sum(row["pressure"]["copies"] for row in subset),
                "total_bytes_copied": sum(row["pressure"]["bytes_copied"] for row in subset),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--worker-url", default="http://127.0.0.1:31992")
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--decode-tokens", type=int, default=160)
    parser.add_argument("--warmup-chunks", type=int, default=16)
    parser.add_argument("--window-s", type=float, default=8.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--modes", default="idle,copy,copy,idle")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.decode_tokens < 64 or args.warmup_chunks < 1 or args.window_s <= 0:
        parser.error("decode-tokens must be at least 64; warmup-chunks and window-s must be positive")
    result = asyncio.run(run(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
