from __future__ import annotations

import queue
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from agentic_backends.sglang.versions.v0510_async_kv import AsyncHiCacheLoader


class FakeEvent:
    def __init__(self, *, ready: bool = False, enable_timing: bool = False) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, _other: FakeEvent) -> float:
        return 7.5


class FakeNode:
    def __init__(self, node_id: int, parent: FakeNode | None, value: object | None) -> None:
        self.id = node_id
        self.parent = parent
        self.value = value
        self.host_value = object()
        self.key = [node_id]
        self.host_refs = 0

    @property
    def evicted(self) -> bool:
        return self.value is None

    def protect_host(self) -> None:
        self.host_refs += 1

    def release_host(self) -> None:
        self.host_refs -= 1


def test_async_load_hides_prefix_until_copy_completes() -> None:
    root = FakeNode(0, None, object())
    leaf = FakeNode(7, root, object())
    value = leaf.value
    op = SimpleNamespace(node_ids=[7], host_indices=[0, 1])
    controller = SimpleNamespace(load_queue=[op])
    cache = SimpleNamespace(
        ongoing_load_back={7: leaf},
        _update_leaf_status=Mock(),
        _update_host_leaf_status=Mock(),
        dec_lock_ref=Mock(),
    )
    loader = AsyncHiCacheLoader.__new__(AsyncHiCacheLoader)
    loader.cache = cache
    loader.controller = controller
    loader.jobs = {}
    loader.pending_node_ids = set()
    loader.queue = queue.Queue()
    loader.lock = threading.Lock()
    with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(Event=FakeEvent))}):
        job = loader.submit("load-7", leaf, [leaf])

    assert leaf.value is None
    assert leaf.host_refs == 1
    assert id(leaf) in loader.pending_node_ids
    assert controller.load_queue == []
    assert loader.poll() == []

    job.enqueued.set()
    assert loader.poll() == []
    job.finish_event.ready = True
    assert loader.poll() == [job]
    assert leaf.value is value
    assert leaf.host_refs == 0
    assert loader.pending_node_ids == set()
    assert cache.ongoing_load_back == {}
    cache.dec_lock_ref.assert_called_once_with(leaf)
    assert loader.snapshot("load-7")["status"] == "finished"


def test_async_load_failure_never_exposes_prefix() -> None:
    root = FakeNode(0, None, object())
    leaf = FakeNode(7, root, object())
    controller = SimpleNamespace(load_queue=[SimpleNamespace(node_ids=[7], host_indices=[0])])
    cache = SimpleNamespace(
        ongoing_load_back={7: leaf},
        _update_leaf_status=Mock(),
        _update_host_leaf_status=Mock(),
    )
    loader = AsyncHiCacheLoader.__new__(AsyncHiCacheLoader)
    loader.cache = cache
    loader.controller = controller
    loader.jobs = {}
    loader.pending_node_ids = set()
    loader.queue = queue.Queue()
    loader.lock = threading.Lock()
    with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=SimpleNamespace(Event=FakeEvent))}):
        job = loader.submit("load-7", leaf, [leaf])
    job.error = "copy_failed"
    job.enqueued.set()
    with pytest.raises(RuntimeError, match="failed closed"):
        loader.poll()
    assert leaf.value is None
    assert id(leaf) in loader.pending_node_ids
    assert loader.snapshot("load-7")["status"] == "failed"
