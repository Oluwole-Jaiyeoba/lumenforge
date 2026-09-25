"""SGLang backend integration.

The ONLY package in this repository allowed to know SGLang module paths,
class/method/attribute names, CLI flags or wire fields.  Layout:

- ``versions/``     per-release adapter data (hook tables, surface lists)
- ``selection``     pick an adapter for an SGLang version (range, then static probe)
- ``surface``       static AST compatibility check (no torch/GPU needed)
- ``lowering``      ``BackendRequest`` -> SGLang ``/v1/chat/completions`` body
- ``launch``        server command lines + flag preflight
- ``telemetry``     raw trace rows -> ``BackendObservation``
- ``adapters``      controller-command adapters (with explicit effect levels)
- ``trace/patch``   in-server tracing + prepare-prefix control (moved verbatim)
- ``compat``        optional ``--radix-eviction-policy priority`` registration
- ``capabilities``  runtime capability probe (imports SGLang; run in the server env)
- ``hooks``         pre-refactor hook-table API used by ``agentic_kv`` shims
"""

from __future__ import annotations

from .lowering import SGLangRequestLowering, lower_translation
from .hook_registry import HOOK_REGISTRY, resolve_hook
from .selection import AdapterSelection, adapter_for_version, select_adapter
from .telemetry import SGLangTelemetryNormalizer
from .versions import ADAPTERS, AdapterSpec, get_adapter, newest_adapter

BACKEND_NAME = "sglang"

__all__ = [
    "ADAPTERS",
    "AdapterSelection",
    "AdapterSpec",
    "BACKEND_NAME",
    "HOOK_REGISTRY",
    "SGLangRequestLowering",
    "SGLangTelemetryNormalizer",
    "adapter_for_version",
    "get_adapter",
    "lower_translation",
    "newest_adapter",
    "select_adapter",
    "resolve_hook",
]
