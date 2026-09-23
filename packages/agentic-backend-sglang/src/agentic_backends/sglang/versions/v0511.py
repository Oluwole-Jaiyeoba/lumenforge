"""Adapter for SGLang 0.5.11 - 0.5.12.post1.

Change vs v0510 (found by the static surface check):
- ``memory_pool_host.NSATokenToKVPoolHost`` no longer exists (NSA host-pool
  hooks dropped; it was an optional alternate anyway).

0.5.11 was exercised with ``probe_sglang_capabilities_docker.sh`` (image
``lmsysorg/sglang:v0.5.11-cu129-runtime``); no full experiment has run on it.
"""

from __future__ import annotations

from .base import AdapterSpec, replace_hook_targets
from .v0510 import HOOK_TARGETS as _V0510_HOOKS
from .v0510 import OPTIONAL_HOOKS, RAW_EVENT_MAP, REQUEST_FIELDS, SURFACE

HOOK_TARGETS = replace_hook_targets(_V0510_HOOKS, drop_classes=("NSATokenToKVPoolHost",))

ADAPTER = AdapterSpec(
    name="v0511",
    tested_versions=("0.5.11",),
    version_range=("0.5.11", "0.5.13"),
    verification="capability_probe",
    hook_targets=HOOK_TARGETS,
    raw_event_map=RAW_EVENT_MAP,
    surface=SURFACE,
    optional_hooks=OPTIONAL_HOOKS,
    request_fields=REQUEST_FIELDS,
    notes=("Static surface check passes for 0.5.11, 0.5.12, 0.5.12.post1.",),
)
