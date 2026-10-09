"""Run one shared workload under native SGLang or capacity-safe admission."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import traceback
from pathlib import Path

from . import run_work_audit_capacity_safe_tiers as capacity
from . import run_work_audit_memory_tiers as native


def workload_contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "seed": args.seed,
        "pattern": args.pattern,
        "model": args.model,
        "sessions": args.sessions,
        "turns": args.turns,
        "initial_tokens": args.initial_tokens,
        "prime_tokens": args.prime_tokens,
        "tool_words": args.tool_words,
        "decode_tokens": args.decode_tokens,
        "wait_ms": args.wait_ms,
        "burst_window_ms": args.burst_window_ms,
        "spread_window_ms": args.spread_window_ms,
        "workload_namespace": args.workload_namespace,
        "frontend_priority": "equal",
        "prefetch_before_tool_return": False,
        "measurement_boundary": "after_initial_prefix_population",
    }


def fingerprint(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


async def run(args: argparse.Namespace) -> dict[str, object]:
    common = dict(
        run_id=args.run_id, out=args.out, mode=args.mode, pattern=args.pattern,
        seed=args.seed, model=args.model, base_url=args.base_url,
        control_url=args.control_url, sessions=args.sessions, turns=args.turns,
        initial_tokens=args.initial_tokens, prime_tokens=args.prime_tokens,
        tool_words=args.tool_words, decode_tokens=args.decode_tokens,
        wait_ms=args.wait_ms, burst_window_ms=args.burst_window_ms,
        spread_window_ms=args.spread_window_ms,
        workload_namespace=args.workload_namespace,
    )
    if args.policy == "native_sglang":
        value = await native.run(argparse.Namespace(
            **common, inspect_lead_ms=0, max_inflight=args.sessions,
            skip_residency_inspection=True, exclude_initial_setup=True,
        ))
    else:
        value = await capacity.run(argparse.Namespace(
            **common, max_active=args.max_active,
            active_token_limit=args.active_token_limit,
            admission_token_margin=args.admission_token_margin,
            page_size=args.page_size,
        ))
    contract = workload_contract(args)
    value["policy"] = args.policy
    value["workload_contract"] = contract
    value["workload_fingerprint"] = fingerprint(contract)
    value["backend_contract"] = {
        "tier_mode": args.mode,
        "gpu_token_limit": args.active_token_limit,
        "host_cache_gb": args.host_cache_gb,
        "storage_enabled": args.storage_enabled,
        "storage_backend": "file" if args.storage_enabled else None,
        "cuda_graph": True,
        "overlap_schedule": True,
        "trace_profile": args.trace_profile,
    }
    value["policy_contract"] = {
        "policy": args.policy,
        "max_active": args.sessions if args.policy == "native_sglang" else args.max_active,
        "restore_before_admission": args.policy == "capacity_safe",
    }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--policy", choices=("native_sglang", "capacity_safe"), required=True)
    parser.add_argument("--mode", choices=("resident", "host", "storage"), default="host")
    parser.add_argument("--pattern", choices=("spread", "burst"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--control-url", default="http://127.0.0.1:31991/prepare_prefix_kv")
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--initial-tokens", type=int, default=4096)
    parser.add_argument("--prime-tokens", type=int, default=2)
    parser.add_argument("--tool-words", type=int, default=16)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--wait-ms", type=int, default=1000)
    parser.add_argument("--burst-window-ms", type=int, default=75)
    parser.add_argument("--spread-window-ms", type=int, default=1000)
    parser.add_argument("--max-active", type=int, default=2)
    parser.add_argument("--active-token-limit", type=int, default=12288)
    parser.add_argument("--admission-token-margin", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--host-cache-gb", type=float, default=2.0)
    parser.add_argument("--storage-enabled", action="store_true")
    parser.add_argument("--trace-profile", default="kv_lifecycle_lean")
    parser.add_argument("--workload-namespace", required=True)
    args = parser.parse_args()
    try:
        value = asyncio.run(run(args))
    except Exception as exc:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        (args.out.parent / "case_failure.json").write_text(json.dumps({
            "policy": args.policy, "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }, indent=2) + "\n", encoding="utf-8")
        raise
    args.out.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "policy": args.policy, "pattern": args.pattern,
        "workflow_duration_ms": value["workflow_duration_ms"],
        "metrics": value["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
