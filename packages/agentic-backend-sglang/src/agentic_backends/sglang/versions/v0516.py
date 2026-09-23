"""Adapter for SGLang 0.5.16 - 0.5.20 (static verification only).

Changes vs v0513 (found by the static surface check):
- Host KV pools were split out of ``mem_cache/memory_pool_host.py`` into
  ``mem_cache/pool_host/{mha,mla,mamba,group}.py`` over 0.5.16 - 0.5.19.
  MHA (the path Qwen/Llama use) moved in 0.5.16 -> hooks follow it.
- ``MambaPoolHost`` moved in 0.5.18; both locations are listed (the tracer
  wraps each method at most once, so a re-export cannot double-wrap).
- ``HostPoolGroup.load_to_device_per_layer`` / ``backup_from_device_all_layer``
  exist only up to 0.5.17; 0.5.18+ redesigned hybrid-pool transfers
  (``resolve_host_transfers``).  Hybrid-pool transfer tracing is therefore
  unavailable on 0.5.18+ until someone writes hooks for the new API.
- ``Scheduler.cur_batch`` removed (optional trace field).
"""

from __future__ import annotations

from .base import AdapterSpec, SGLangHookTarget, replace_hook_targets
from .v0510 import OPTIONAL_HOOKS, RAW_EVENT_MAP, REQUEST_FIELDS, SURFACE
from .v0513 import HOOK_TARGETS as _V0513_HOOKS

_HOSTPOOL_METHODS = {
    "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
    "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
}

HOOK_TARGETS = replace_hook_targets(
    _V0513_HOOKS,
    move_classes={
        "MHATokenToKVPoolHost": "sglang.srt.mem_cache.pool_host.mha",
        "MLATokenToKVPoolHost": "sglang.srt.mem_cache.pool_host.mla",
    },
    add=(
        SGLangHookTarget(
            module="sglang.srt.mem_cache.pool_host.mamba",
            class_name="MambaPoolHost",
            methods=_HOSTPOOL_METHODS,
        ),
    ),
)

ADAPTER = AdapterSpec(
    name="v0516",
    tested_versions=(),
    version_range=("0.5.16", "0.5.21"),
    verification="static",
    hook_targets=HOOK_TARGETS,
    raw_event_map=RAW_EVENT_MAP,
    surface=SURFACE,
    optional_hooks=OPTIONAL_HOOKS,
    request_fields=REQUEST_FIELDS,
    notes=(
        "Static surface check passes for 0.5.16 - 0.5.20; never run on a GPU.",
        "Hybrid (HostPoolGroup) transfer hooks are absent on 0.5.18+.",
    ),
)
