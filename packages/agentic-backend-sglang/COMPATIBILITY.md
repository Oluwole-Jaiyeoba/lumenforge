# SGLang compatibility matrix

Generated on 2026-09-23 by running the static surface check
(`python -m agentic_backends.sglang.surface`) against the unpacked PyPI wheel
of every SGLang release from 0.5.10.post1 to 0.5.20 (no install, no GPU).

- **Selected adapter**: what `selection.select_adapter(<version>)` picks.
- **Verification**: `runtime` = full EC2 experiments ran on it;
  `capability_probe` = `probe_sglang_capabilities_docker.sh` ran in the real
  container; `static` = only this static check. Treat `static` as "should
  work, must be confirmed with one reference experiment on a GPU".
- **Required / optional missing**: surface elements the selected adapter
  needs that the release does not have. Optional = alternates (MLA/NSA/Mamba/
  hybrid host pools, `Req.fill_ids`, `Scheduler.cur_batch`, ...).
- **Last column**: what the *pre-refactor* code would have lost on that
  release (it used the v0510 tables for every version -- ``v0511`` was an
  alias). Missing hooks were skipped without any message; a missing CLI flag
  makes ``sglang.launch_server`` exit at startup.

| SGLang | Selected adapter | Verification | Required missing | Optional missing | Lost silently by the pre-refactor code (v0510/v0511 tables) |
| --- | --- | --- | ---: | ---: | --- |
| 0.5.10.post1 | v0510 | runtime | 0 | 0 | nothing |
| 0.5.11 | v0511 | capability_probe | 0 | 0 | 2: mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.12 | v0511 | capability_probe | 0 | 0 | 2: mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.12.post1 | v0511 | capability_probe | 0 | 0 | 2: mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.13 | v0513 | static | 0 | 1 | 5: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.13.post1 | v0513 | static | 0 | 1 | 5: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.14 | v0513 | static | 0 | 1 | 5: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.15 | v0513 | static | 0 | 1 | 5: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.15.post1 | v0513 | static | 0 | 1 | 5: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.16 | v0516 | static | 0 | 4 | 10: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.MHATokenToKVPoolHost, mem_cache.memory_pool_host.MLATokenToKVPoolHost, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.17 | v0516 | static | 0 | 4 | 10: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.MHATokenToKVPoolHost, mem_cache.memory_pool_host.MLATokenToKVPoolHost, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.18 | v0516 | static | 0 | 6 | 14: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.HostPoolGroup, mem_cache.memory_pool_host.MHATokenToKVPoolHost, mem_cache.memory_pool_host.MLATokenToKVPoolHost, mem_cache.memory_pool_host.MambaPoolHost, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.19 | v0516 | static | 0 | 6 | 14: managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.HostPoolGroup, mem_cache.memory_pool_host.MHATokenToKVPoolHost, mem_cache.memory_pool_host.MLATokenToKVPoolHost, mem_cache.memory_pool_host.MambaPoolHost, mem_cache.memory_pool_host.NSATokenToKVPoolHost |
| 0.5.20 | v0520 | static | 0 | 6 | 15: --disable-piecewise-cuda-graph, managers.schedule_batch.Req, managers.scheduler.Scheduler, mem_cache.memory_pool_host.HostPoolGroup, mem_cache.memory_pool_host.MHATokenToKVPoolHost, mem_cache.memory_pool_host.MLATokenToKVPoolHost, mem_cache.memory_pool_host.MambaPoolHost, mem_cache.memory_pool_host.NSATokenToKVPoolHost |

## What changed between releases (found by the checker)

| Release | Change in SGLang | Adapter response |
| --- | --- | --- |
| 0.5.11 | `NSATokenToKVPoolHost` removed from `mem_cache/memory_pool_host.py` | `v0511` drops the NSA host-pool hooks |
| 0.5.13 | `Scheduler.process_batch_result_prefill/decode` moved to `managers/scheduler_components/batch_result_processor.py` (`SchedulerBatchResultProcessor`, composition not mixin); `Req.fill_ids` removed | `v0513` hooks the new class with the **same event names**; inside those hooks `self` is the processor, so scheduler-queue summaries on those two events may be thinner |
| 0.5.16 | `MHATokenToKVPoolHost` / `MLATokenToKVPoolHost` moved to `mem_cache/pool_host/{mha,mla}.py`; `Scheduler.cur_batch` removed | `v0516` follows the move |
| 0.5.18 | `MambaPoolHost` moved to `pool_host/mamba.py`; `HostPoolGroup` lost `load_to_device_per_layer` / `backup_from_device_all_layer` (hybrid transfer redesign: `resolve_host_transfers`) | `v0516` lists both Mamba locations; hybrid-pool transfer tracing is unavailable on 0.5.18+ (optional) |
| 0.5.19 | `HostPoolGroup` moved to `pool_host/group.py` | optional hook, no action |
| 0.5.20 | `--disable-piecewise-cuda-graph` removed (CUDA-graph config redesign); `"priority"` is now a built-in `--radix-eviction-policy` choice; most flags are generated from dataclass fields | `v0520` (same hooks as v0516). **Blocker:** `run_harness_deadline_pressure.sh` and `run_milestone22_live_agentbench_bridge.sh` pass `--disable-piecewise-cuda-graph` by default, so the server will not start on 0.5.20 until that default changes. `scripts/sglang_preflight.py` reports it before the model loads. |

Wire fields used by the gateway (`priority`, `custom_params`, `cache_salt` on
`ChatCompletionRequest`) are present in all releases above, and so is every
CLI flag the launch scripts use except `--disable-piecewise-cuda-graph` on 0.5.20.

## Re-generating this table

```bash
pip download "sglang==X.Y.Z" --no-deps -d /tmp/w && python -m zipfile -e /tmp/w/*.whl /tmp/sgl-X.Y.Z
python -m agentic_backends.sglang.surface --sglang-src /tmp/sgl-X.Y.Z          # adapter picked by version
python -m agentic_backends.sglang.surface --sglang-src /tmp/sgl-X.Y.Z --all    # every adapter
```
