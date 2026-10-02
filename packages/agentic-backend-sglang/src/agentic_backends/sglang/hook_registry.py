"""Stable instrumentation capabilities resolved through version adapters.

Experiment contracts use these IDs; they never name SGLang-private modules or
methods.  This package resolves each ID through the selected adapter's hook
table, keeping version drift behind the backend boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .versions import get_adapter


@dataclass(frozen=True)
class HookCapability:
    hook_id: str
    description: str
    selectors: tuple[tuple[str, str], ...]
    read_only: bool = True


HOOK_REGISTRY: dict[str, HookCapability] = {
    "request_lifecycle": HookCapability(
        "request_lifecycle",
        "SGLang accepts and queues a request.",
        (("Scheduler", "handle_generate_request"), ("Scheduler", "process_input_requests")),
    ),
    "scheduler_batch_observation": HookCapability(
        "scheduler_batch_observation",
        "SGLang selects and completes scheduler batches.",
        (("Scheduler", "get_next_batch_to_run"), ("Scheduler", "process_batch_result")),
    ),
    "request_completion_timing": HookCapability(
        "request_completion_timing",
        "SGLang emits batch completion evidence used for request timing.",
        (("Scheduler", "process_batch_result"),),
    ),
    "kv_cache_lifecycle": HookCapability(
        "kv_cache_lifecycle",
        "Prefix-cache matching is visible to the trace.",
        (("HiRadixCache", "match_prefix"), ("RadixCache", "match_prefix")),
    ),
    "hicache_transfer_observation": HookCapability(
        "hicache_transfer_observation",
        "Hierarchical-cache transfer operations are visible to the trace.",
        (("HiCacheController", "load"), ("HiCacheController", "write")),
    ),
    "kv_host_write": HookCapability(
        "kv_host_write", "Host cache write transition.", (("HiCacheController", "write"),),
    ),
    "kv_gpu_evict": HookCapability(
        "kv_gpu_evict", "Device cache eviction transition.", (("HiCacheController", "evict_device"),),
    ),
    "kv_host_evict": HookCapability(
        "kv_host_evict", "Host cache eviction transition.", (("HiCacheController", "evict_host"),),
    ),
    "kv_gpu_load": HookCapability(
        "kv_gpu_load", "Semantic host-to-device cache load.", (("HiCacheController", "load"),),
    ),
    "kv_layer_copy": HookCapability(
        "kv_layer_copy", "Per-layer host-to-device copy.",
        tuple((name, "load_to_device_per_layer") for name in (
            "HostPoolGroup", "MHATokenToKVPoolHost", "MLATokenToKVPoolHost",
            "NSATokenToKVPoolHost", "MambaPoolHost",
        )),
    ),
    "kv_prefix_match": HookCapability(
        "kv_prefix_match", "Prefix-cache match with index evidence.", (("HiRadixCache", "match_prefix"),),
    ),
    "model_forward": HookCapability(
        "model_forward", "Model forward batch begins or ends.", (("TpModelWorker", "forward_batch_generation"),),
    ),
}


def resolve_hook(hook_id: str, adapter_name: str) -> dict[str, Any]:
    """Resolve a stable hook ID to installed-adapter targets and event names."""

    try:
        capability = HOOK_REGISTRY[hook_id]
    except KeyError as exc:
        raise KeyError(f"unknown instrumentation hook ID {hook_id!r}; known: {sorted(HOOK_REGISTRY)}") from exc
    adapter = get_adapter(adapter_name)
    targets: list[dict[str, str]] = []
    wanted = set(capability.selectors)
    for target in adapter.hook_targets:
        for method, event_name in target.methods.items():
            if (target.class_name, method) in wanted:
                targets.append(
                    {
                        "target": f"{target.module}.{target.class_name}.{method}",
                        "event_prefix": event_name,
                        "class_name": target.class_name,
                        "method": method,
                    }
                )
    return {
        "hook_id": capability.hook_id,
        "description": capability.description,
        "read_only": capability.read_only,
        "adapter": adapter_name,
        "targets": targets,
    }


def resolve_contract_hook_ids(hook_ids: list[str], adapter_name: str) -> list[dict[str, Any]]:
    return [resolve_hook(hook_id, adapter_name) for hook_id in hook_ids]
