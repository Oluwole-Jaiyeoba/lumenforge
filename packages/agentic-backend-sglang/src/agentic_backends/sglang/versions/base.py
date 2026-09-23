"""Data model for per-version SGLang adapters.

An adapter is *data*, not code: it lists which SGLang internals this project
touches for one SGLang release line.  Everything that knows SGLang module
paths, class names, private method names, attribute names, CLI flags or
request-body field names lives in a ``versions/vXXXX.py`` module built from
these types.

Upgrading SGLang = add/adjust one adapter module + run the surface check
(``python -m agentic_backends.sglang.surface --sglang-src <path>``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SGLangHookTarget:
    """A version-specific SGLang class/method group to trace."""

    module: str
    class_name: str
    methods: Mapping[str, str]
    scheduler_required: bool = False


RawEventMap = Mapping[str, str]


# Surface requirement kinds understood by ``agentic_backends.sglang.surface``.
MODULE = "module"  # importable module file exists
CLASS = "class"  # class defined in module
METHOD = "method"  # method on class (own or inherited)
ATTRIBUTE = "attribute"  # instance/class attribute (``self.x = ...``, field, property or method)
MODULE_ATTR = "module_attr"  # module-level name (assignment, import, def, class)
CLI_FLAG = "cli_flag"  # ``--flag`` accepted by ``sglang.launch_server``


@dataclass(frozen=True)
class SurfaceRequirement:
    """One piece of SGLang that a feature of this project depends on.

    ``required=True`` means the feature is broken (or silently loses data)
    without it.  ``required=False`` marks alternates the code probes
    defensively (e.g. MLA/NSA host pools) -- missing ones are reported as
    warnings only.
    """

    kind: str
    module: str
    name: str = ""
    class_name: str = ""
    feature: str = ""
    required: bool = True
    note: str = ""

    def label(self) -> str:
        if self.kind == MODULE:
            return self.module
        if self.kind == CLI_FLAG:
            return self.name
        if self.kind in (METHOD, ATTRIBUTE):
            return f"{self.module}.{self.class_name}.{self.name}"
        if self.kind == CLASS:
            return f"{self.module}.{self.name}"
        return f"{self.module}:{self.name}"


# Feature names used in surface requirements.  Keep in sync with the README.
FEATURE_TRACE_KV = "trace.kv_movement"
FEATURE_TRACE_SCHEDULER = "trace.scheduler"
FEATURE_TRACE_REQUEST_CONTEXT = "trace.request_context"
FEATURE_PREPARE_PREFIX = "control.prepare_prefix"
FEATURE_PRIORITY_EVICTION = "compat.priority_radix_eviction"
FEATURE_REQUEST_LOWERING = "request.lowering"
FEATURE_LAUNCH = "launch.flags"


@dataclass(frozen=True)
class AdapterSpec:
    """Everything version-specific about one SGLang release line."""

    name: str
    tested_versions: tuple[str, ...]
    # Half-open range [min, max) of SGLang versions this adapter is meant for.
    version_range: tuple[str, str] = ("0", "0")
    # How the adapter was verified:
    #   "runtime"           -- full experiments ran on this SGLang version
    #   "capability_probe"  -- probe_sglang_capabilities ran in a real container
    #   "static"            -- only the static surface check was run
    verification: str = "static"
    hook_targets: tuple[SGLangHookTarget, ...] = ()
    raw_event_map: Mapping[str, str] = field(default_factory=dict)
    surface: tuple[SurfaceRequirement, ...] = ()
    launch_flags: Mapping[str, str] = field(default_factory=dict)
    request_fields: Mapping[str, str] = field(default_factory=dict)
    optional_hooks: frozenset[tuple[str, str]] = frozenset()
    notes: tuple[str, ...] = ()

    def hook_surface_with_overrides(self) -> tuple[SurfaceRequirement, ...]:
        """Derive method requirements from the hook table.

        Every hooked ``(class, method)`` is required for its feature unless it
        is listed in ``optional_hooks`` (alternates the tracer wraps when
        present, e.g. MLA/NSA host pools or dLLM forward paths).
        """

        out: list[SurfaceRequirement] = []
        for target in self.hook_targets:
            feature = FEATURE_TRACE_SCHEDULER if target.scheduler_required else FEATURE_TRACE_KV
            for method in target.methods:
                out.append(
                    SurfaceRequirement(
                        kind=METHOD,
                        module=target.module,
                        class_name=target.class_name,
                        name=method,
                        feature=feature,
                        required=(target.class_name, method) not in self.optional_hooks,
                    )
                )
        return tuple(out)


def replace_hook_targets(
    targets: tuple[SGLangHookTarget, ...],
    *,
    drop_classes: tuple[str, ...] = (),
    drop_methods: Mapping[str, tuple[str, ...]] | None = None,
    move_classes: Mapping[str, str] | None = None,
    add: tuple[SGLangHookTarget, ...] = (),
) -> tuple[SGLangHookTarget, ...]:
    """Derive a new hook table from an older one with explicit, reviewable edits."""

    drop_methods = drop_methods or {}
    move_classes = move_classes or {}
    out: list[SGLangHookTarget] = []
    for target in targets:
        if target.class_name in drop_classes:
            continue
        methods = {k: v for k, v in target.methods.items() if k not in drop_methods.get(target.class_name, ())}
        module = move_classes.get(target.class_name, target.module)
        out.append(
            SGLangHookTarget(
                module=module,
                class_name=target.class_name,
                methods=methods,
                scheduler_required=target.scheduler_required,
            )
        )
    out.extend(add)
    return tuple(out)
