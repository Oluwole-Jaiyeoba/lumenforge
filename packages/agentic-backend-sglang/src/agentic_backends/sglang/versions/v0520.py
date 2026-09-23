"""Adapter for SGLang 0.5.20 (static verification only).

Same hook tables as ``v0516``.  Launch-level change found by the static check:

- ``--disable-piecewise-cuda-graph`` no longer exists (CUDA-graph config was
  redesigned; see ``sglang/srt/arg_groups/cuda_graph_hook.py``).  The default
  ``EXTRA_SERVER_ARGS`` in ``sglang_direct_kv/scripts/run_harness_deadline_pressure.sh``
  (and ``run_milestone22_live_agentbench_bridge.sh``) still pass it, so the
  server will refuse to start on 0.5.20 until those defaults drop the flag.
  ``scripts/sglang_preflight.py`` prints this before the model loads.
- ``"priority"`` is now a built-in ``--radix-eviction-policy`` choice, so the
  ``compat.py`` registration is a no-op.
"""

from __future__ import annotations

from .base import AdapterSpec
from .v0510 import OPTIONAL_HOOKS, RAW_EVENT_MAP, REQUEST_FIELDS, SURFACE
from .v0516 import HOOK_TARGETS

REMOVED_FLAGS = ("--disable-piecewise-cuda-graph",)

SURFACE_V0520 = tuple(req for req in SURFACE if req.name not in REMOVED_FLAGS)

ADAPTER = AdapterSpec(
    name="v0520",
    tested_versions=(),
    version_range=("0.5.20", "0.5.21"),
    verification="static",
    hook_targets=HOOK_TARGETS,
    raw_event_map=RAW_EVENT_MAP,
    surface=SURFACE_V0520,
    optional_hooks=OPTIONAL_HOOKS,
    request_fields=REQUEST_FIELDS,
    notes=(
        "Static surface check passes for 0.5.20; never run on a GPU.",
        "Launch scripts must stop passing --disable-piecewise-cuda-graph.",
    ),
)
