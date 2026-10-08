"""Scheduler-thread cache inspection/release for the opt-in coordinated audit."""

from __future__ import annotations

from typing import Any
import time


def replay_prefix_match(row: dict) -> tuple[str, dict] | None:
    if row.get("event") != "hiradix.match_prefix.end":
        return None
    ctx = row.get("kv_context") or {}
    req = ctx.get("request") or {}
    request_id = ctx.get("agent_request_id") or req.get("agent_request_id")
    result = row.get("result")
    if request_id and isinstance(result, list) and result:
        return request_id, {"gpu_tokens": result[0].get("index_count", 0),
                            "host_tokens": result[3] if len(result) > 3 else None}
    return None


def validated_runtime_settings(info: dict, *, io_backend: str, gpu_tokens: int) -> dict:
    expected = {"disable_cuda_graph": False, "disable_overlap_schedule": False,
                "enable_priority_scheduling": False, "hicache_io_backend": io_backend,
                "hicache_storage_backend": None, "max_total_tokens": gpu_tokens}
    if any(info.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Unexpected backend flags: { {k: info.get(k) for k in expected} }")
    return {k: info.get(k) for k in (
        *expected, "hicache_size", "hicache_mem_layout", "hicache_write_policy",
        "max_total_num_tokens", "page_size", "model_path", "version", "random_seed")}


def assert_private_prefix(entry: dict, entries: list[dict]) -> None:
    ids = entry.get("coordinated_token_ids") or []
    identity = (entry.get("request") or {}).get("agent_session_id")
    if len(ids) < 128 or not identity:
        raise ValueError("Missing session identity or private prefix")
    for other in entries:
        if (other.get("request") or {}).get("agent_session_id") == identity:
            continue
        if (other.get("coordinated_token_ids") or [])[:64] == ids[:64]:
            raise ValueError("Prefix anchor is shared by another session")


def release_subtree(cache: Any, anchor: Any) -> dict:
    """Release only an unlocked, durably host-backed private subtree."""
    nodes = []
    stack = [(anchor, False)]
    while stack:
        node, visited = stack.pop()
        if visited:
            if not node.evicted:
                nodes.append(node)
        else:
            stack.append((node, True))
            stack.extend((child, False) for child in node.children.values())
    # Check the entire set before making any mutation.
    for node in nodes:
        if node.lock_ref or node.id in cache.ongoing_load_back:
            return {"ok": False, "status": "prefix_busy", "node_id": str(node.id)}
        if not node.backuped or node.id in cache.ongoing_write_through:
            return {"ok": False, "status": "host_backup_pending", "node_id": str(node.id)}
    count = sum(cache._evict_backuped(node) for node in nodes)
    return {"ok": True, "status": "released", "evicted_tokens": count,
            "evicted_node_ids": [str(node.id) for node in nodes]}


def prefix_control(entry: dict, command: dict) -> dict:
    from importlib.metadata import version
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
    from sglang.srt.mem_cache.radix_cache import RadixKey

    cache = entry["tree_cache"]
    if version("sglang") != "0.5.10.post1":
        raise ValueError("Coordinated cache control is validated only for 0.5.10.post1")
    if type(cache).__name__ != "HiRadixCache" or cache.tp_world_size != 1:
        raise ValueError("Coordinated cache control requires single-rank HiRadixCache")
    ids = entry.get("coordinated_token_ids") or []
    if len(ids) < 128:
        raise ValueError("Registered coordinated prefix is missing or too short")
    cache.loading_check()
    cache.writing_check()
    match = cache.match_prefix(MatchPrefixParams(key=RadixKey(ids, entry.get("coordinated_extra_key"))))
    entry["last_host_node"] = match.last_host_node
    entry["last_node"] = match.last_device_node
    nodes = []
    node = match.last_host_node
    while node is not cache.root_node:
        nodes.append(node)
        node = node.parent
    resident = sum(len(n.key) for n in nodes if not n.evicted)
    host = sum(len(n.key) for n in nodes if n.backuped)
    result = {"ok": True, "status": "residency", "cached_tokens": sum(len(n.key) for n in nodes),
              "gpu_tokens": resident, "host_tokens": host,
              "pending_loads": sum(n.id in cache.ongoing_load_back for n in nodes),
              "registered_tokens": len(ids), "gpu_free_tokens": cache.cache_controller.mem_pool_device_allocator.available_size()}
    if command["action"] == "release_prefix":
        # The runner gives every session a different first 64-token prefix.
        # Splitting at that boundary avoids touching common chat/template nodes.
        anchor_match = cache.match_prefix(MatchPrefixParams(
            key=RadixKey(ids[:64], entry.get("coordinated_extra_key"))))
        anchor = anchor_match.last_host_node
        if anchor is cache.root_node or len(anchor.key) == 0:
            raise ValueError("No private prefix anchor")
        result.update(release_subtree(cache, anchor))
    return result


def prepare_group(entries: list[dict], commands: list[dict], enqueue) -> tuple[dict, dict]:
    """Reserve each private prefix, then use the native merged load queue once."""
    import torch

    if not entries or len(entries) > 20 or len(entries) != len(commands):
        raise ValueError("A load group must contain 1..20 registered prefixes")
    cache = entries[0]["tree_cache"]
    if any(entry["tree_cache"] is not cache for entry in entries):
        raise ValueError("Group spans cache instances")
    for entry in entries:
        assert_private_prefix(entry, entries)
    if cache.cache_controller.load_queue:
        raise ValueError("Native load queue must be empty before a group restore")
    states = [prefix_control(e, {"action": "prefix_residency"}) for e in entries]
    if any(s["cached_tokens"] < s["registered_tokens"] - 64 for s in states):
        raise ValueError("Group prefix is missing from both cache tiers")
    missing = [s["cached_tokens"] - s["gpu_tokens"] for s in states]
    if any(s["pending_loads"] or s["host_tokens"] < s["cached_tokens"] for s in states):
        raise ValueError("Group contains pending loads or incomplete host backups")
    if sum(missing) > states[-1]["gpu_free_tokens"]:
        raise ValueError("Group restore would evict another session")
    started = time.time_ns()
    responses = []
    failure = None
    producer = -1
    start_event, finish_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    try:
        for command, tokens in zip(commands, missing):
            if not tokens:
                responses.append({"ok": True, "loaded_tokens": 0, "session_id": command["session_id"]})
                continue
            response = enqueue({**command, "action": "prepare", "whole_prefix": True,
                                "min_load_tokens": 1, "minimum_host_tokens": 1, "mem_quota": tokens,
                                "_coordinated_defer": True})
            if response.get("status") != "reserved_for_group":
                raise RuntimeError(f"Native group reservation failed: {response}")
            responses.append({**response, "session_id": command["session_id"]})
    except Exception as exc:
        failure = str(exc)
    finally:
        # Even a partial reservation must be submitted, so no locked, unfilled
        # slots are stranded if a later member fails its precondition.
        if cache.cache_controller.load_queue:
            with torch.cuda.stream(cache.cache_controller.load_stream):
                start_event.record()
            producer = int(cache.ready_to_load_host_cache())
            with torch.cuda.stream(cache.cache_controller.load_stream):
                finish_event.record()
    if failure:
        raise RuntimeError(failure)
    if producer < 0:
        return {"ok": True, "status": "already_resident", "members": responses}, {}
    native = cache.cache_controller.layer_done_counter.events[producer]
    load_id = f"coordinated-group:{started}:{producer}"
    tokens = sum(r["loaded_tokens"] for r in responses)
    tracking = {"load_id": load_id, "node_id": "group", "producer_id": producer,
                "loaded_tokens": tokens, "command_started_ns": started, "queued_ns": time.time_ns(),
                "started_observed_ns": None, "finished_observed_ns": None, "cuda_elapsed_ms": None,
                "last_reported_status": None, "start_event": start_event, "finish_event": finish_event,
                "native_start_event": native.start_event, "native_finish_event": native.finish_event}
    return {"ok": True, "status": "queued", "load_id": load_id, "loaded_tokens": tokens,
            "members": responses, "control_path": "native_load_back+merged_start_loading"}, tracking
