"""Interpret the pinned runtime fields without adding any instrumentation hooks."""

BACKEND_IMAGE_ENV = "SGLANG_DOCKER_IMAGE"
BACKEND_PRIORITY_FIELD = "sglang_priority"
HOST_CACHE_ENV = "HICACHE_SIZE_GB"
TESTBED_DIRECTORY = "sglang_direct_kv"

PAIR_FIELDS = (
    "model_path", "dtype", "quantization", "max_total_tokens", "enable_hierarchical_cache", "hicache_size", "hicache_ratio",
    "hicache_io_backend", "hicache_mem_layout", "hicache_write_policy", "hicache_storage_backend",
    "mem_fraction_static", "chunked_prefill_size", "max_prefill_tokens", "page_size", "max_running_requests",
    "disable_cuda_graph", "disable_overlap_schedule", "cuda_graph_bs",
)


def runtime_issues(args: dict, *, queue_ranking: bool, retention_ranking: bool) -> list[str]:
    issues = []
    if args.get("disable_cuda_graph") is not False or args.get("disable_overlap_schedule") is not False:
        issues.append("CUDA graphs/overlap setting mismatch")
    if args.get("enable_priority_scheduling") is not queue_ranking:
        issues.append("Queue isolation mismatch")
    if retention_ranking and args.get("radix_eviction_policy") != "priority":
        issues.append("Retention policy missing")
    return issues


def runtime_pair_issues(baseline: dict, treatment: dict) -> list[str]:
    return [f"Backend {key} differs" for key in PAIR_FIELDS if baseline.get(key) != treatment.get(key)]
