"""Compatibility package: SGLang adapters moved to ``agentic_backends.sglang``.

The version tables now live in ``agentic_backends/sglang/versions`` and
selection in ``agentic_backends.sglang.selection``.  The functions below keep
their historical signatures (see ``agentic_backends.sglang.hooks``).
"""

from __future__ import annotations

from agentic_backends.sglang.hooks import (
    SGLangHookTarget,
    get_hook_targets,
    get_raw_event_map,
    installed_sglang_version,
    select_adapter_name,
)

__all__ = [
    "SGLangHookTarget",
    "get_hook_targets",
    "get_raw_event_map",
    "installed_sglang_version",
    "select_adapter_name",
]
