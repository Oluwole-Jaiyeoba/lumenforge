"""Build a tiny, importable fake ``sglang`` source tree from an adapter spec.

The fake contains exactly the modules/classes/methods/attributes/flags an
adapter says it needs (no torch).  Tests use it to exercise the static surface
check, adapter selection and the real in-server trace installer without a
GPU.  ``drop`` removes elements to simulate an incompatible SGLang release.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from agentic_backends.sglang.versions.base import ATTRIBUTE, CLASS, CLI_FLAG, METHOD, MODULE_ATTR, AdapterSpec


def build_fake_sglang(root: Path, adapter: AdapterSpec, version: str, drop: set[str] | None = None) -> Path:
    """``drop`` entries: ``"Class.method"``, ``"Class.attr"``, ``"Class"``, ``"--flag"`` or ``"module:NAME"``."""

    drop = drop or set()
    classes: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: defaultdict(lambda: {"methods": set(), "attrs": set()}))
    module_attrs: dict[str, set[str]] = defaultdict(set)
    flags: list[str] = []
    for target in adapter.hook_targets:
        for method in target.methods:
            classes[target.module][target.class_name]["methods"].add(method)
    for req in adapter.surface:
        if req.kind == METHOD:
            classes[req.module][req.class_name]["methods"].add(req.name)
        elif req.kind == ATTRIBUTE:
            classes[req.module][req.class_name]["attrs"].add(req.name)
        elif req.kind == CLASS:
            classes[req.module][req.name]
        elif req.kind == MODULE_ATTR:
            module_attrs[req.module].add(req.name)
        elif req.kind == CLI_FLAG:
            flags.append(req.name)
    module_attrs.setdefault("sglang.srt.server_args", set())

    all_modules = set(classes) | set(module_attrs)
    for module in sorted(all_modules):
        parts = module.split(".")
        for depth in range(1, len(parts)):
            package_dir = root.joinpath(*parts[:depth])
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / "__init__.py").touch()
        lines = ["# generated fake sglang module", ""]
        for name in sorted(module_attrs.get(module, ())):
            if f"{module}:{name}" not in drop:
                lines.append(f"{name} = ['lru', 'lfu']")
        if module == "sglang.srt.server_args":
            kept = [flag for flag in flags if flag not in drop]
            lines.append("CLI_FLAGS = [" + ", ".join(repr(f) for f in kept) + "]")
        for class_name, members in sorted(classes.get(module, {}).items()):
            if class_name in drop:
                continue
            lines.append(f"class {class_name}:")
            attrs = sorted(a for a in members["attrs"] if f"{class_name}.{a}" not in drop)
            lines.append("    def __init__(self):")
            if attrs:
                for attr in attrs:
                    lines.append(f"        self.{attr} = None")
            else:
                lines.append("        pass")
            for method in sorted(members["methods"]):
                if f"{class_name}.{method}" in drop or method == "__init__":
                    continue
                lines.append(f"    def {method}(self, *args, **kwargs):")
                lines.append(f"        return {method!r}")
            lines.append("")
        root.joinpath(*parts).with_suffix(".py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    dist = root / f"sglang-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: sglang\nVersion: {version}\n", encoding="utf-8")
    return root
