"""Hook-table accessors with the pre-refactor function signatures.

``agentic_kv.sglang_adapters`` re-exports these so historical imports keep
working.  New code should use ``selection.select_adapter`` and the returned
``AdapterSpec`` directly.
"""

from __future__ import annotations

from typing import Iterable

from .selection import adapter_for_version, installed_sglang_version, legacy_adapter_name
from .versions import ADAPTERS, AdapterSpec
from .versions.base import SGLangHookTarget


def select_adapter_name(version: str | None = None) -> str:
    """Adapter name for ``version`` (default: installed SGLang). Never raises."""

    return legacy_adapter_name(version if version is not None else installed_sglang_version())


def _adapter(version: str | None) -> AdapterSpec:
    return ADAPTERS[select_adapter_name(version)]


def get_hook_targets(
    *,
    include_scheduler: bool = False,
    version: str | None = None,
) -> tuple[SGLangHookTarget, ...]:
    targets: Iterable[SGLangHookTarget] = _adapter(version).hook_targets
    if include_scheduler:
        return tuple(targets)
    return tuple(target for target in targets if not target.scheduler_required)


def get_raw_event_map(version: str | None = None) -> dict[str, str]:
    return dict(_adapter(version).raw_event_map)


__all__ = [
    "SGLangHookTarget",
    "adapter_for_version",
    "get_hook_targets",
    "get_raw_event_map",
    "installed_sglang_version",
    "select_adapter_name",
]
