#!/usr/bin/env python3
"""Small paired SGLang tracing-off/on latency check on an otherwise idle GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request


REPO = Path(__file__).resolve().parents[1]
for package_src in (REPO / "packages").glob("*/src"):
    sys.path.insert(0, str(package_src))

from agentic_backends.sglang.instrumentation_profiles import profile_flags  # noqa: E402


def tracing_flags(profile: str, enabled: bool) -> dict[str, str]:
    selected = profile_flags(profile)
    return selected if enabled else {name: "0" for name in selected}


def _request(model: str, port: int, index: int) -> dict[str, float | int]:
    prompt = "Write the integers 1 through 100, one per line. Do not add commentary. " + ("context " * 120) + f"Trial {index}."
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": 64, "temperature": 0, "stream": False,
                       "cache_salt": f"instrumentation-overhead-{index}"}).encode()
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    usage = result.get("usage") or {}
    return {"latency_ms": round((time.perf_counter() - start) * 1000, 3),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0)}


def _ready(port: int, process: subprocess.Popen[bytes], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"backend exited during startup with code {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError("SGLang did not become healthy")


def _case(args: argparse.Namespace, enabled: bool) -> dict[str, object]:
    label = "on" if enabled else "off"
    output = args.out_dir / label
    output.mkdir(parents=True, exist_ok=True)
    name = f"agentic-trace-overhead-{label}-{os.getpid()}"
    env = os.environ.copy()
    env.update({
        "SGLANG_DOCKER_IMAGE": args.image,
        "SGLANG_DOCKER_EXTRA_ARGS": f"--name {name} -v {args.model_cache}:/tmp/hfcache -e HF_HOME=/tmp/hfcache",
        "AGENTIC_KV_TRACE_ENABLE": "1" if enabled else "0",
        "AGENTIC_KV_TRACE_CONTROL_ONLY": "1" if enabled and args.control_only_pump else "0",
        "AGENTIC_KV_PREPARE_CONTROL_ENABLE": "1" if enabled and args.control_only_pump else "0",
        "MEM_FRACTION_STATIC": str(args.mem_fraction_static),
        "HICACHE_SIZE_GB": str(args.hicache_size_gb),
        "AGENTIC_KV_TRACE_PATH": str(output / "backend_trace.jsonl"),
        "AGENTIC_RUNTIME_TELEMETRY_PATH": str(output / "runtime_telemetry.jsonl"),
        "PORT": str(args.port),
        "PYTHON_BIN": "python3",
    })
    env.update(tracing_flags(args.profile, enabled))
    command = ["bash", str(REPO / "sglang_direct_kv/scripts/run_sglang_hicache_server.sh"), args.model]
    with (output / "server.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=REPO / "sglang_direct_kv", env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            _ready(args.port, process, args.startup_timeout)
            rows = [_request(args.model, args.port, index) for index in range(args.repetitions + 1)]
        finally:
            subprocess.run(["docker", "stop", "--time", "10", name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=False)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
    measured = rows[1:]
    return {"tracing": label, "warmup": rows[0], "measurements": measured,
            "median_latency_ms": round(statistics.median(row["latency_ms"] for row in measured), 3),
            "trace_bytes": (output / "backend_trace.jsonl").stat().st_size if (output / "backend_trace.jsonl").exists() else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--model-cache", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--profile", default="full_debug",
                        help="Shared instrumentation profile to measure (default: full_debug)")
    parser.add_argument("--control-only-pump", action="store_true",
                        help="Include the lean work-audit control command pump")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--startup-timeout", type=int, default=240)
    parser.add_argument("--mem-fraction-static", type=float, default=float(os.environ.get("MEM_FRACTION_STATIC", "0.72")))
    parser.add_argument("--hicache-size-gb", type=int, default=int(os.environ.get("HICACHE_SIZE_GB", "8")))
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    args.model_cache = args.model_cache.resolve()
    if args.repetitions < 2 or not args.model_cache.is_dir():
        parser.error("need at least two repetitions and an existing model cache")
    profile_flags(args.profile)
    if args.control_only_pump and args.profile != "kv_lifecycle_lean":
        parser.error("--control-only-pump requires --profile kv_lifecycle_lean")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            parser.error(f"port {args.port} is in use; run only when the GPU experiment is idle")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [_case(args, False), _case(args, True)]
    baseline, traced = (item["median_latency_ms"] for item in cases)
    result = {"schema_version": "agentic.instrumentation.overhead.v1", "model": args.model,
              "profile": args.profile, "control_only_pump": args.control_only_pump,
              "image": args.image, "order": ["off", "on"], "cases": cases,
              "median_latency_delta_ms": round(traced - baseline, 3),
              "median_latency_delta_percent": round((traced / baseline - 1) * 100, 2),
              "limitation": "Single off-then-on pair on an idle GPU; indicative, not a statistically controlled performance claim."}
    (args.out_dir / "overhead.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("median_latency_delta_ms", "median_latency_delta_percent", "limitation")}, indent=2))


if __name__ == "__main__":
    main()
