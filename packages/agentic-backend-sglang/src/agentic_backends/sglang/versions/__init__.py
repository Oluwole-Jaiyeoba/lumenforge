"""Registry of SGLang version adapters (see ``base.AdapterSpec``)."""

from __future__ import annotations

from .base import AdapterSpec, SGLangHookTarget, SurfaceRequirement
from . import v0510, v0511, v0513, v0516

# Ordered oldest -> newest.  Add new adapters at the end.
ADAPTERS: dict[str, AdapterSpec] = {
    adapter.name: adapter
    for adapter in (v0510.ADAPTER, v0511.ADAPTER, v0513.ADAPTER, v0516.ADAPTER)
}


def get_adapter(name: str) -> AdapterSpec:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(f"unknown SGLang adapter {name!r}; known: {sorted(ADAPTERS)}") from None


def newest_adapter() -> AdapterSpec:
    return list(ADAPTERS.values())[-1]


__all__ = ["ADAPTERS", "AdapterSpec", "SGLangHookTarget", "SurfaceRequirement", "get_adapter", "newest_adapter"]
