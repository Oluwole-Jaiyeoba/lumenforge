"""SGLang launch flags for shared profiles; historical controller defaults are frozen."""

from __future__ import annotations

import argparse
import json
from typing import Any

from agentic_instrumentation import PROFILES

from .hook_registry import resolve_hook


_FLAGS = ("AGENTIC_KV_TRACE_SCHEDULER", "AGENTIC_KV_TRACE_KV_POOL",
          "AGENTIC_RUNTIME_TELEMETRY", "AGENTIC_KV_GPU_UTIL_SAMPLER",
          "TRACE_CONTROLLER_DECISIONS_DEFAULT", "TRACE_IDLE_GAP_AUDIT_DEFAULT",
          "AGENTIC_KV_COPY_TELEMETRY_ENABLE")

# The first six names preserve the existing launcher semantics exactly.
_VALUES = {
    "minimal": (0, 0, 0, 0, 0, 0, 1),
    "deadline": (1, 0, 1, 0, 0, 0, 1),
    "controller_decision": (1, 0, 1, 1, 1, 1, 1),
    "idle_gap": (1, 0, 1, 1, 1, 1, 1),
    "cache_debug": (1, 1, 1, 1, 1, 1, 1),
    "full_debug": (1, 1, 1, 1, 1, 1, 1),
    "controller_queue": (1, 0, 1, 0, 0, 0, 0),
    "request_boundary": (1, 0, 0, 0, 0, 0, 0),
    "kv_lifecycle": (1, 0, 0, 0, 0, 0, 0),
    "copy_timing": (1, 1, 0, 0, 0, 0, 1),
}

_HOOK_IDS = {
    "request.accepted": "request_lifecycle",
    "batch.scheduled": "scheduler_batch_observation",
    "batch.completed": "request_completion_timing",
    "kv.write_host": "kv_host_write",
    "kv.evict_gpu": "kv_gpu_evict",
    "kv.evict_host": "kv_host_evict",
    "kv.load_gpu": "kv_gpu_load",
    "kv.layer_copy": "kv_layer_copy",
    "kv.prefix_match": "kv_prefix_match",
    "model.forward": "model_forward",
}


def profile_flags(name: str) -> dict[str, str]:
    if name not in _VALUES:
        raise ValueError(f"Unknown instrumentation profile {name!r}; choose from {sorted(_VALUES)}")
    return dict(zip(_FLAGS, map(str, _VALUES[name])))


def validate_installation(name: str, adapter: str, installation: dict[str, Any]) -> dict[str, Any]:
    """Require at least one installed adapter target for each profile signal."""
    if name not in PROFILES:
        raise ValueError(f"Unknown shared evidence profile {name!r}")
    if installation.get("adapter") != adapter:
        return {"profile": name, "adapter": adapter, "valid": False,
                "missing": [{"signal": "adapter", "reason": f"installation reports {installation.get('adapter')!r}"}]}
    installed = set(installation.get("installed_hooks") or [])
    missing = []
    for signal in PROFILES[name].required_signals:
        hook_id = _HOOK_IDS.get(signal)
        if hook_id is None:
            missing.append({"signal": signal, "reason": "no adapter hook mapping"})
            continue
        targets = [item["target"] for item in resolve_hook(hook_id, adapter)["targets"]]
        if not targets or not installed.intersection(targets):
            missing.append({"signal": signal, "reason": "hook not installed", "targets": targets})
    return {"profile": name, "adapter": adapter, "valid": not missing, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=sorted(_VALUES))
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    flags = profile_flags(args.profile)
    if args.shell:
        print("\n".join(f"{key}={value}" for key, value in flags.items()))
    else:
        print(json.dumps({"profile": args.profile, "flags": flags}, indent=2))


if __name__ == "__main__":
    main()
