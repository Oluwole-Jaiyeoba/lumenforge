"""Opt-in, single-rank async HiCache load experiment for SGLang 0.5.10.

Only the CUDA copy loop runs on a worker. Cache and allocator mutations remain
on the scheduler thread. Pending nodes are hidden until the copy event completes.
This is a research prototype, not a general SGLang scheduling replacement.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any


@dataclass
class AsyncLoadJob:
    load_id: str
    node_id: str
    op: Any
    nodes: list[tuple[Any, Any, Any]]
    submitted_ns: int
    start_event: Any
    finish_event: Any
    enqueued: threading.Event = field(default_factory=threading.Event)
    worker_started_ns: int | None = None
    worker_enqueued_ns: int | None = None
    committed_ns: int | None = None
    verified: bool = False
    error: str | None = None


class AsyncHiCacheLoader:
    """Keep a reserved prefix invisible while a dedicated stream copies it."""

    def __init__(self, tree_cache: Any) -> None:
        import torch

        if version("sglang") != "0.5.10.post1":
            raise ValueError("Async KV prototype requires SGLang 0.5.10.post1")
        if type(tree_cache).__name__ != "HiRadixCache":
            raise ValueError("Async KV requires the pinned HiRadixCache implementation")
        if int(getattr(tree_cache, "tp_world_size", 1)) != 1:
            raise ValueError("Async KV prototype supports only one tensor-parallel rank")
        controller = tree_cache.cache_controller
        if controller.io_backend != "direct":
            raise ValueError("Async KV prototype requires the direct HiCache I/O backend")
        if not hasattr(controller.mem_pool_host, "load_to_device_per_layer"):
            raise ValueError("HiCache host pool lacks per-layer native transfer")
        self.cache = tree_cache
        self.controller = controller
        self.device_index = torch.cuda.current_device()
        self.stream = torch.cuda.Stream(device=controller.device)
        self.jobs: dict[str, AsyncLoadJob] = {}
        self.pending_node_ids: set[int] = set()
        self.queue: queue.Queue[AsyncLoadJob] = queue.Queue()
        self.lock = threading.Lock()
        self.worker_error: str | None = None
        original_load_back = tree_cache.load_back

        def guard_pending_load(node: Any, mem_quota: int | None = None) -> Any:
            current = node
            while current is not None:
                if id(current) in self.pending_node_ids:
                    return None
                current = getattr(current, "parent", None)
            return original_load_back(node, mem_quota)

        tree_cache.load_back = guard_pending_load
        self.worker = threading.Thread(target=self._run, name="agentic-v0510-kv-copy", daemon=True)
        self.worker.start()

    def capture_evicted_chain(self, node: Any) -> list[Any]:
        chain = []
        while getattr(node, "evicted", False):
            chain.insert(0, node)
            node = node.parent
        return chain

    def submit(self, load_id: str, node: Any, chain: list[Any]) -> AsyncLoadJob:
        import torch

        if not chain or node is not chain[-1]:
            raise RuntimeError("Async KV expected an evicted host-backed chain")
        if len(self.controller.load_queue) != 1:
            raise RuntimeError("Async KV requires one isolated reserved load operation")
        op = self.controller.load_queue[0]
        if list(op.node_ids) != [node.id]:
            raise RuntimeError("Async KV reserved operation does not match the selected node")
        if node.id not in self.cache.ongoing_load_back:
            raise RuntimeError("Async KV lost SGLang's load-back lock")
        if load_id in self.jobs:
            raise RuntimeError("Async KV duplicate load ID")
        saved = []
        for part in chain:
            if part.value is None or part.host_value is None:
                raise RuntimeError("Async KV reservation did not create both cache copies")
            saved.append((part, part.value, part.key))
        job = AsyncLoadJob(
            load_id=load_id,
            node_id=str(node.id),
            op=op,
            nodes=saved,
            submitted_ns=time.time_ns(),
            start_event=torch.cuda.Event(enable_timing=True),
            finish_event=torch.cuda.Event(enable_timing=True),
        )
        for part, _value, _key in saved:
            part.protect_host()
            part.value = None
            self.pending_node_ids.add(id(part))
            self.cache._update_leaf_status(part)
            self.cache._update_host_leaf_status(part)
        self.controller.load_queue.clear()
        with self.lock:
            self.jobs[load_id] = job
        self.queue.put(job)
        return job

    def _run(self) -> None:
        import torch

        try:
            torch.cuda.set_device(self.device_index)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.worker_error = f"{type(exc).__name__}: {exc}"
        while True:
            job = self.queue.get()
            try:
                if self.worker_error is not None:
                    raise RuntimeError(f"Worker startup failed: {self.worker_error}")
                with self.lock:
                    job.worker_started_ns = time.time_ns()
                host_indices, device_indices = self.controller.move_indices(job.op)
                with torch.cuda.stream(self.stream):
                    job.start_event.record(self.stream)
                    for layer_id in range(self.controller.layer_num):
                        args = (
                            self.controller.mem_pool_device,
                            host_indices,
                            device_indices,
                            layer_id,
                            self.controller.io_backend,
                        )
                        pool_transfers = getattr(job.op, "pool_transfers", None)
                        if pool_transfers is None:
                            self.controller.mem_pool_host.load_to_device_per_layer(*args)
                        else:
                            self.controller.mem_pool_host.load_to_device_per_layer(
                                *args, pool_transfers=pool_transfers
                            )
                    job.finish_event.record(self.stream)
                    if host_indices.is_cuda:
                        host_indices.record_stream(self.stream)
                    if device_indices.is_cuda:
                        device_indices.record_stream(self.stream)
                with self.lock:
                    job.worker_enqueued_ns = time.time_ns()
            except Exception as exc:  # noqa: BLE001
                with self.lock:
                    job.error = f"{type(exc).__name__}: {exc}"
            finally:
                job.enqueued.set()
                self.queue.task_done()

    def poll(self) -> list[AsyncLoadJob]:
        """Run only on the scheduler thread; publish completed prefixes."""

        completed = []
        for job in list(self.jobs.values()):
            if job.committed_ns is not None or not job.enqueued.is_set():
                continue
            if job.error is not None:
                raise RuntimeError(f"Async KV copy failed closed: {job.error}")
            if not job.finish_event.query():
                continue
            for part, _value, original_key in job.nodes:
                if part.value is not None or part.key != original_key:
                    raise RuntimeError("Async KV cache node changed during the in-flight copy")
            if os.environ.get("AGENTIC_KV_ASYNC_VERIFY_FULL_COPY") == "1":
                self._verify_full_copy(job)
                job.verified = True
            for part, value, _key in job.nodes:
                part.value = value
                self.pending_node_ids.remove(id(part))
                self.cache._update_leaf_status(part)
                self.cache._update_host_leaf_status(part)
                part.release_host()
            end_node = self.cache.ongoing_load_back.pop(job.op.node_ids[0])
            self.cache.dec_lock_ref(end_node)
            with self.lock:
                job.committed_ns = time.time_ns()
            completed.append(job)
        return completed

    def _verify_full_copy(self, job: AsyncLoadJob) -> None:
        """Diagnostic-only exact host/GPU comparison before exposing the KV."""

        import torch

        host_pool = self.controller.mem_pool_host
        device_pool = self.controller.mem_pool_device
        if host_pool.layout != "layer_first" or host_pool.page_size != 1:
            raise RuntimeError("Full-copy verification requires layer-first, page-size-one KV")
        host_indices = job.op.host_indices.cpu()
        device_indices = job.op.device_indices.to(self.controller.device)
        for layer_id in range(self.controller.layer_num):
            for name in ("k_buffer", "v_buffer"):
                host_values = getattr(host_pool, name)[layer_id].index_select(0, host_indices)
                device_values = getattr(device_pool, name)[layer_id].index_select(0, device_indices).cpu()
                if not torch.equal(host_values, device_values):
                    raise RuntimeError(f"Async KV mismatch in {name} layer {layer_id}")

    def snapshot(self, load_id: str) -> dict[str, Any] | None:
        with self.lock:
            job = self.jobs.get(load_id)
            if job is None:
                return None
            snapshot = {
                "load_id": load_id,
                "node_id": job.node_id,
                "loaded_tokens": int(len(job.op.host_indices)),
                "submitted_ns": job.submitted_ns,
                "worker_started_ns": job.worker_started_ns,
                "worker_enqueued_ns": job.worker_enqueued_ns,
                "committed_ns": job.committed_ns,
                "full_copy_verified": job.verified,
                "finished_observed_ns": job.committed_ns,
                "observed_ns": time.time_ns(),
                "error": job.error,
            }
        if job.error is not None:
            snapshot["status"] = "failed"
        elif job.committed_ns is not None:
            snapshot["status"] = "finished"
            snapshot["cuda_elapsed_ms"] = round(float(job.start_event.elapsed_time(job.finish_event)), 6)
        elif job.worker_started_ns is not None:
            snapshot["status"] = "active"
        else:
            snapshot["status"] = "queued"
        return snapshot
