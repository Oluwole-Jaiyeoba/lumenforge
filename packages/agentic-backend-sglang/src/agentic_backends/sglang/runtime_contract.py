from __future__ import annotations

import argparse
import json
import platform
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from agentic_backend_api import BackendCapabilities, BackendRuntimeInfo

from .capabilities import collect_sglang_capabilities


def _hook_supports(raw: dict[str, Any], event_suffixes: Iterable[str]) -> bool:
    suffixes = tuple(event_suffixes)
    for row in raw.get("hook_probe", []):
        if not row.get("class_found") or row.get("missing_methods"):
            continue
        module = str(row.get("module") or "")
        class_name = str(row.get("class") or "")
        if any(module.endswith(suffix) or class_name.endswith(suffix) for suffix in suffixes):
            return True
    return False


def normalize_capabilities(raw: dict[str, Any]) -> BackendCapabilities:
    launch = raw.get("launch_capabilities") or {}
    priority = bool(
        launch.get("enable_priority_scheduling_flag_supported")
        or launch.get("priority_schedule_policy_supported")
    )
    has_hicache_controller = _hook_supports(raw, ("cache_controller", "HiCacheController"))
    has_hiradix = _hook_supports(raw, ("hiradix_cache", "HiRadixCache"))
    has_scheduler = _hook_supports(raw, ("managers.scheduler", "Scheduler"))
    controllable = priority or has_hicache_controller or has_hiradix
    return BackendCapabilities(
        priority_queue=priority,
        # A visible scheduler is enough for telemetry, not proof that this
        # backend exposes a portable admission-budget control surface.
        background_prefill_budget=False,
        safe_preemption=False,
        kv_demote=has_hicache_controller,
        kv_prefetch=has_hicache_controller or has_hiradix,
        kv_release=has_hicache_controller or has_hiradix,
        live_metrics=has_scheduler,
        observe_only=not controllable,
        backend_name="sglang",
        backend_version=str(raw.get("sglang_version") or ""),
    )


def build_runtime_info(
    raw: dict[str, Any],
    *,
    runtime_profile: str,
    endpoint: str = "",
    healthy: bool | None = None,
    container_image: str = "",
    container_image_digest: str = "",
    gpu_vendor: str = "",
    gpu_architecture: str = "",
    host_architecture: str = "",
    metadata: dict[str, Any] | None = None,
) -> BackendRuntimeInfo:
    version = str(raw.get("sglang_version") or "")
    health_status = "not_checked" if healthy is None else ("healthy" if healthy else "unhealthy")
    return BackendRuntimeInfo(
        runtime_profile=runtime_profile,
        backend_name="sglang",
        backend_version=version,
        adapter=str(raw.get("selected_adapter") or ""),
        endpoint=endpoint,
        probe_ok=bool(version),
        healthy=bool(healthy),
        health_status=health_status,
        container_image=container_image,
        container_image_digest=container_image_digest,
        gpu_vendor=gpu_vendor,
        gpu_architecture=gpu_architecture,
        host_architecture=host_architecture or platform.machine(),
        capabilities=normalize_capabilities(raw),
        raw_capabilities=raw,
        metadata=dict(metadata or {}),
    )


def missing_capabilities(info: BackendRuntimeInfo, required: Iterable[str]) -> list[str]:
    capabilities = info.capabilities.to_dict()
    return sorted(name for name in set(required) if not bool(capabilities.get(name)))


def _health_ok(url: str, timeout_s: float) -> bool:
    if not url:
        return False
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return 200 <= int(response.status) < 300
    except Exception:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Emit the portable SGLang runtime handshake.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-profile", required=True)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--health-url", default="")
    parser.add_argument("--health-timeout-s", type=float, default=2.0)
    parser.add_argument("--container-image", default="")
    parser.add_argument("--container-image-digest", default="")
    parser.add_argument("--gpu-vendor", default="")
    parser.add_argument("--gpu-architecture", default="")
    parser.add_argument("--host-architecture", default="")
    parser.add_argument("--require-capability", action="append", default=[])
    args = parser.parse_args(argv)

    raw = collect_sglang_capabilities()
    health_checked = bool(args.health_url)
    info = build_runtime_info(
        raw,
        runtime_profile=args.runtime_profile,
        endpoint=args.endpoint,
        healthy=_health_ok(args.health_url, args.health_timeout_s) if health_checked else None,
        container_image=args.container_image,
        container_image_digest=args.container_image_digest,
        gpu_vendor=args.gpu_vendor,
        gpu_architecture=args.gpu_architecture,
        host_architecture=args.host_architecture,
        metadata={"health_checked": health_checked, "health_url": args.health_url},
    )
    missing = missing_capabilities(info, args.require_capability)
    row = info.to_dict()
    row["required_capabilities"] = sorted(set(args.require_capability))
    row["missing_required_capabilities"] = missing
    text = json.dumps(row, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote backend runtime contract to {args.out}")
    else:
        print(text, end="")
    if not info.probe_ok:
        print("SGLang runtime capability probe failed.")
        return 2
    if health_checked and not info.healthy:
        print("SGLang service health check failed.")
        return 2
    if missing:
        print(f"Missing required backend capabilities: {', '.join(missing)}")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
