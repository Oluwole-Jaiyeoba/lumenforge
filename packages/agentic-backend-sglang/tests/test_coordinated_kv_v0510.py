from types import SimpleNamespace
from contextlib import nullcontext
import sys

import pytest

from agentic_backends.sglang.versions.v0510_coordinated_kv import assert_private_prefix, release_subtree
from agentic_backends.coordinated_audit import validated_runtime_settings


def node(identity, **changes):
    return SimpleNamespace(id=identity, children={}, evicted=False, backuped=True, lock_ref=0, **changes)


def test_release_checks_whole_tree_before_mutation():
    root, child = node(1), node(2)
    root.children[2] = child
    child.lock_ref = 1
    freed = []
    cache = SimpleNamespace(ongoing_load_back={}, ongoing_write_through={},
                            _evict_backuped=lambda n: freed.append(n.id) or 64)
    assert release_subtree(cache, root)["status"] == "prefix_busy"
    assert freed == []
    child.lock_ref = 0
    assert release_subtree(cache, root)["evicted_tokens"] == 128
    assert freed == [2, 1]


def test_pending_backup_is_not_released():
    root = node(1)
    cache = SimpleNamespace(ongoing_load_back={}, ongoing_write_through={1: root})
    assert release_subtree(cache, root)["status"] == "host_backup_pending"


def test_shared_anchor_is_rejected():
    entry = {"request": {"agent_session_id": "a"}, "coordinated_token_ids": list(range(128))}
    other = {**entry, "request": {"agent_session_id": "b"}}
    with pytest.raises(ValueError, match="shared"):
        assert_private_prefix(entry, [entry, other])
    assert_private_prefix(entry, [entry, entry])


def test_runtime_rejects_wrong_capacity_or_priority_flags():
    info = dict(disable_cuda_graph=False, disable_overlap_schedule=False,
                enable_priority_scheduling=False, hicache_io_backend="kernel",
                hicache_storage_backend=None, max_total_tokens=110592)
    assert validated_runtime_settings(info, io_backend="kernel", gpu_tokens=110592)["max_total_tokens"] == 110592
    for changed in ({"max_total_tokens": 262144}, {"enable_priority_scheduling": True},
                    {"disable_cuda_graph": True}, {"disable_overlap_schedule": True}):
        with pytest.raises(RuntimeError, match="Unexpected backend"):
            validated_runtime_settings({**info, **changed}, io_backend="kernel", gpu_tokens=110592)


def test_group_uses_one_native_start_and_drains_partial_failure(monkeypatch):
    from agentic_backends.sglang.versions import v0510_coordinated_kv as module

    event = SimpleNamespace(record=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(
        Event=lambda **kw: event, stream=lambda stream: nullcontext())))
    native = SimpleNamespace(start_event=event, finish_event=event)
    controller = SimpleNamespace(load_queue=[], load_stream=object(),
                                 layer_done_counter=SimpleNamespace(events=[native]))
    starts = []

    def start():
        starts.append(len(controller.load_queue))
        controller.load_queue.clear()
        return 0

    cache = SimpleNamespace(cache_controller=controller, ready_to_load_host_cache=start)
    entries = [{"tree_cache": cache, "request": {"agent_session_id": str(i)},
                "coordinated_token_ids": [i] * 128} for i in range(2)]
    monkeypatch.setattr(module, "prefix_control", lambda *a: dict(
        cached_tokens=128, registered_tokens=128, gpu_tokens=0, host_tokens=128, pending_loads=0, gpu_free_tokens=1024))

    def enqueue(command):
        controller.load_queue.append(command)
        return {"status": "reserved_for_group", "loaded_tokens": 128}

    commands = [{"session_id": str(i)} for i in range(2)]
    result, tracking = module.prepare_group(entries, commands, enqueue)
    assert starts == [2]
    assert result["loaded_tokens"] == tracking["loaded_tokens"] == 256

    def fail_second(command):
        if command["session_id"] == "1":
            return {"status": "load_back_not_admitted"}
        return enqueue(command)

    with pytest.raises(RuntimeError, match="reservation failed"):
        module.prepare_group(entries, commands, fail_second)
    assert starts == [2, 1]
    assert controller.load_queue == []
