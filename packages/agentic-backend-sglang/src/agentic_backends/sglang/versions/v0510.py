"""Adapter for SGLang 0.5.10 / 0.5.10.post1 (the version most remote A10G host results used).

Hook targets and the raw event map are the original
``agentic_kv/sglang_adapters/v0510.py`` tables, unchanged.  ``SURFACE`` lists
every other SGLang internal this project reads or calls, grouped by feature;
``python -m agentic_backends.sglang.surface`` checks them statically.
"""

from __future__ import annotations

from .base import (
    ATTRIBUTE,
    CLASS,
    CLI_FLAG,
    FEATURE_LAUNCH,
    FEATURE_PREPARE_PREFIX,
    FEATURE_PRIORITY_EVICTION,
    FEATURE_REQUEST_LOWERING,
    FEATURE_TRACE_REQUEST_CONTEXT,
    FEATURE_TRACE_SCHEDULER,
    METHOD,
    MODULE_ATTR,
    AdapterSpec,
    SGLangHookTarget,
    SurfaceRequirement,
)


HOOK_TARGETS: tuple[SGLangHookTarget, ...] = (
    SGLangHookTarget(
        module="sglang.srt.managers.cache_controller",
        class_name="HiCacheController",
        methods={
            "load": "hicache.load",
            "write": "hicache.write",
            "evict_device": "hicache.evict_device",
            "evict_host": "hicache.evict_host",
            "prefetch": "hicache.prefetch",
            "start_loading": "hicache.start_loading",
            "start_writing": "hicache.start_writing",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.hiradix_cache",
        class_name="HiRadixCache",
        methods={
            "match_prefix": "hiradix.match_prefix",
            "cache_finished_req": "hiradix.cache_finished_req",
            "cache_unfinished_req": "hiradix.cache_unfinished_req",
            "evict": "hiradix.evict",
            "load_back": "hiradix.load_back",
            "init_load_back": "hiradix.init_load_back",
            "ready_to_load_host_cache": "hiradix.ready_to_load_host_cache",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.radix_cache",
        class_name="RadixCache",
        methods={
            "match_prefix": "radix.match_prefix",
            "cache_finished_req": "radix.cache_finished_req",
            "cache_unfinished_req": "radix.cache_unfinished_req",
            "evict": "radix.evict",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="HostPoolGroup",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="MHATokenToKVPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="MLATokenToKVPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="NSATokenToKVPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.mem_cache.memory_pool_host",
        class_name="MambaPoolHost",
        methods={
            "load_to_device_per_layer": "hostpool.load_to_device_per_layer",
            "backup_from_device_all_layer": "hostpool.backup_from_device_all_layer",
        },
    ),
    SGLangHookTarget(
        module="sglang.srt.managers.scheduler",
        class_name="Scheduler",
        methods={
            "handle_generate_request": "scheduler.handle_generate_request",
            "_add_request_to_queue": "scheduler.add_request_to_queue",
            "_prefetch_kvcache": "scheduler.prefetch_kvcache",
            "_run_batch_prebuilt": "scheduler.run_batch_prebuilt",
            "process_batch_result": "scheduler.process_batch_result",
            "process_batch_result_prefill": "scheduler.process_batch_result_prefill",
            "process_batch_result_decode": "scheduler.process_batch_result_decode",
            "run_batch": "scheduler.run_batch",
            "process_input_requests": "scheduler.process_input_requests",
            "get_next_batch_to_run": "scheduler.get_next_batch_to_run",
            "get_new_batch_prefill": "scheduler.get_new_batch_prefill",
            "event_loop_overlap": "scheduler.event_loop_overlap",
            "event_loop_normal": "scheduler.event_loop_normal",
        },
        scheduler_required=True,
    ),
    SGLangHookTarget(
        module="sglang.srt.managers.tp_worker",
        class_name="TpModelWorker",
        methods={
            "forward_batch_generation": "worker.forward_batch_generation",
            "forward_batch_split_prefill": "worker.forward_batch_split_prefill",
            "_forward_batch_generation_dllm": "worker.forward_batch_generation_dllm",
            "forward_batch_embedding": "worker.forward_batch_embedding",
        },
        scheduler_required=True,
    ),
)


RAW_EVENT_MAP: dict[str, str] = {
    "hicache.write.end": "KV_WRITE_HOST",
    "hicache.evict_device.end": "KV_EVICT_GPU",
    "hicache.evict_host.end": "KV_EVICT_HOST",
    "hicache.load.end": "KV_LOAD_GPU",
    "hostpool.load_to_device_per_layer.end": "KV_LOAD_GPU",
    "hostpool.backup_from_device_all_layer.end": "KV_WRITE_HOST",
    "hiradix.init_load_back.end": "KV_LOAD_GPU",
    "hiradix.load_back.end": "KV_LOAD_GPU",
    "hiradix.match_prefix.end": "KV_MATCH_PREFIX",
}


_SCHED = "sglang.srt.managers.scheduler"
_BATCH = "sglang.srt.managers.schedule_batch"
_RADIX = "sglang.srt.mem_cache.radix_cache"
_HIRADIX = "sglang.srt.mem_cache.hiradix_cache"
_CACHE_CONTROLLER = "sglang.srt.managers.cache_controller"
_BASE_PREFIX_CACHE = "sglang.srt.mem_cache.base_prefix_cache"
_PROTOCOL = "sglang.srt.entrypoints.openai.protocol"


def _attrs(module: str, class_name: str, names: str, feature: str, required: bool = True, note: str = "") -> list[SurfaceRequirement]:
    return [
        SurfaceRequirement(kind=ATTRIBUTE, module=module, class_name=class_name, name=name, feature=feature, required=required, note=note)
        for name in names.split()
    ]


SURFACE: tuple[SurfaceRequirement, ...] = tuple(
    # Request identity/prefill attribution read from scheduler Req objects.
    _attrs(_BATCH, "Req", "rid origin_input_ids output_ids prefix_indices last_node", FEATURE_TRACE_REQUEST_CONTEXT)
    + _attrs(_BATCH, "Req", "priority fill_ids extra_key", FEATURE_TRACE_REQUEST_CONTEXT, required=False)
    + _attrs(_BATCH, "ScheduleBatch", "reqs forward_mode", FEATURE_TRACE_SCHEDULER)
    # Scheduler queue/pool state summarized in scheduler trace events.
    + _attrs(_SCHED, "Scheduler", "waiting_queue running_batch tree_cache max_total_num_tokens", FEATURE_TRACE_SCHEDULER)
    + _attrs(_SCHED, "Scheduler", "cur_batch last_batch chunked_req token_to_kv_pool_allocator req_to_token_pool", FEATURE_TRACE_SCHEDULER, required=False)
    # Direct host->device prefix preparation (prepare-prefix control server).
    + [
        SurfaceRequirement(kind=METHOD, module=_HIRADIX, class_name="HiRadixCache", name=name, feature=FEATURE_PREPARE_PREFIX)
        for name in ("load_back", "ready_to_load_host_cache", "loading_check")
    ]
    + _attrs(_HIRADIX, "HiRadixCache", "ongoing_load_back cache_controller", FEATURE_PREPARE_PREFIX)
    + _attrs(_CACHE_CONTROLLER, "HiCacheController", "layer_done_counter", FEATURE_PREPARE_PREFIX)
    + [SurfaceRequirement(kind=CLASS, module=_BASE_PREFIX_CACHE, name="EvictParams", feature=FEATURE_PREPARE_PREFIX)]
    + _attrs(_RADIX, "TreeNode", "id evicted backuped host_value", FEATURE_PREPARE_PREFIX)
    # Optional launch choice radix-eviction-policy=priority (sglang_compat).
    + [
        SurfaceRequirement(kind=CLASS, module="sglang.srt.mem_cache.evict_policy", name="PriorityStrategy", feature=FEATURE_PRIORITY_EVICTION),
        SurfaceRequirement(kind=MODULE_ATTR, module="sglang.srt.server_args", name="RADIX_EVICTION_POLICY_CHOICES", feature=FEATURE_PRIORITY_EVICTION),
    ]
    # Fields the gateway puts on /v1/chat/completions bodies.
    + _attrs(_PROTOCOL, "ChatCompletionRequest", "priority custom_params cache_salt stream max_tokens messages temperature", FEATURE_REQUEST_LOWERING)
    # Server flags used by the launch scripts (run_sglang_*server.sh and the
    # EXTRA_SERVER_ARGS defaults in run_harness_deadline_pressure.sh).
    + [
        SurfaceRequirement(kind=CLI_FLAG, module="sglang.launch_server", name=flag, feature=FEATURE_LAUNCH)
        for flag in (
            "--model-path",
            "--trust-remote-code",
            "--attention-backend",
            "--prefill-attention-backend",
            "--decode-attention-backend",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--disable-overlap-schedule",
            "--default-priority-value",
            "--hicache-storage-backend-extra-config",
            "--enable-priority-scheduling",
            "--schedule-policy",
            "--radix-eviction-policy",
            "--enable-cache-report",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "--hicache-io-backend",
            "--hicache-mem-layout",
            "--hicache-storage-backend",
            "--hicache-storage-prefetch-policy",
            "--mem-fraction-static",
            "--chunked-prefill-size",
        )
    ]
    + [
        SurfaceRequirement(
            kind=CLI_FLAG,
            module="sglang.launch_server",
            name="--file-storage-path",
            feature=FEATURE_LAUNCH,
            required=False,
            note="only passed with HICACHE_STORAGE_BACKEND=file",
        )
    ]
)

# Alternates the tracer wraps only when the model/config uses them.  Missing
# ones do not break the MHA (Qwen/Llama) experiment path.
OPTIONAL_HOOKS: frozenset[tuple[str, str]] = frozenset(
    {
        (cls, method)
        for cls in ("HostPoolGroup", "MLATokenToKVPoolHost", "NSATokenToKVPoolHost", "MambaPoolHost")
        for method in ("load_to_device_per_layer", "backup_from_device_all_layer")
    }
    | {
        ("TpModelWorker", "forward_batch_split_prefill"),
        ("TpModelWorker", "_forward_batch_generation_dllm"),
        ("TpModelWorker", "forward_batch_embedding"),
        ("Scheduler", "event_loop_overlap"),
        ("Scheduler", "_run_batch_prebuilt"),
        ("Scheduler", "_prefetch_kvcache"),
        ("HiCacheController", "prefetch"),
    }
)

REQUEST_FIELDS = {"priority": "priority", "metadata": "custom_params", "cache_salt": "cache_salt"}

ADAPTER = AdapterSpec(
    name="v0510",
    tested_versions=("0.5.10", "0.5.10.post1"),
    version_range=("0.5.10", "0.5.11"),
    verification="runtime",
    hook_targets=HOOK_TARGETS,
    raw_event_map=RAW_EVENT_MAP,
    surface=SURFACE,
    optional_hooks=OPTIONAL_HOOKS,
    request_fields=REQUEST_FIELDS,
    notes=("Reference adapter: ~151 recorded remote A10G host runs used sglang 0.5.10.post1.",),
)
