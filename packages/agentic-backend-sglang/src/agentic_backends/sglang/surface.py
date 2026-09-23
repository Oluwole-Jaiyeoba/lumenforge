"""Static compatibility check of an SGLang source tree against an adapter.

Why static?  Importing ``sglang.srt`` pulls in torch/CUDA, which is not
available on laptops or CI.  Parsing the SGLang source with ``ast`` answers the
questions that matter for an upgrade without a GPU:

- do the modules/classes/methods our trace hooks wrap still exist?
- do the attributes our trace/control code reads still exist?
- does the OpenAI request model still accept ``priority``/``custom_params``/
  ``cache_salt``?
- does ``launch_server`` still accept the CLI flags our scripts pass?

Usage::

    # installed SGLang (no import of sglang is performed)
    python -m agentic_backends.sglang.surface

    # an unpacked wheel / git checkout, against a specific adapter
    python -m agentic_backends.sglang.surface --sglang-src /path/to/site-packages --adapter v0510

    # check every registered adapter and print JSON
    python -m agentic_backends.sglang.surface --sglang-src ... --all --json

Exit code is 0 when every *required* element is present, 1 otherwise.

Limitations: attributes created dynamically (``setattr``, ``__getattr__``) are
invisible to a static check and are reported missing; mark such requirements
``required=False`` with a note.  Inheritance is resolved through ``from x
import Y`` imports and same-module classes.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from agentic_backend_api import CompatibilityFinding, CompatibilityReport

from .versions.base import (
    ATTRIBUTE,
    CLASS,
    CLI_FLAG,
    METHOD,
    MODULE,
    MODULE_ATTR,
    AdapterSpec,
    SurfaceRequirement,
)


@dataclass
class _ClassInfo:
    module: str
    name: str
    bases: list[str]
    methods: set[str] = field(default_factory=set)
    attributes: set[str] = field(default_factory=set)


@dataclass
class _ModuleInfo:
    name: str
    path: Path
    tree: ast.Module
    classes: dict[str, _ClassInfo]
    imports: dict[str, str]  # local name -> fully qualified name
    names: set[str]  # module-level names


class SourceIndex:
    """Lazy AST index of an SGLang source tree.

    ``root`` is the directory that *contains* the ``sglang`` package
    (``site-packages`` or an unpacked wheel), or the ``sglang`` package
    directory itself.
    """

    def __init__(self, root: str | Path) -> None:
        root = Path(root).resolve()
        if (root / "sglang" / "__init__.py").exists() or (root / "sglang").is_dir():
            self.root = root
        elif root.name == "sglang":
            self.root = root.parent
        elif (root / "python" / "sglang").is_dir():  # git checkout layout
            self.root = root / "python"
        else:
            raise FileNotFoundError(f"no sglang package under {root}")
        self._modules: dict[str, _ModuleInfo | None] = {}

    def version(self) -> str:
        # Wheel/site-packages metadata is authoritative; version.py in source
        # checkouts is often a dev placeholder ("0.0.0.dev0").
        for dist in sorted(self.root.glob("sglang-*.dist-info/METADATA")):
            match = re.search(r"^Version:\s*(\S+)", dist.read_text(encoding="utf-8"), re.M)
            if match:
                return match.group(1)
        for candidate in (self.root / "sglang" / "_version.py", self.root / "sglang" / "version.py"):
            if candidate.exists():
                match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", candidate.read_text(encoding="utf-8"))
                if match and match.group(1) != "0.0.0.dev0":
                    return match.group(1)
        return ""

    def module_path(self, module: str) -> Path | None:
        base = self.root.joinpath(*module.split("."))
        for candidate in (base.with_suffix(".py"), base / "__init__.py"):
            if candidate.exists():
                return candidate
        return None

    def module(self, module: str) -> _ModuleInfo | None:
        if module in self._modules:
            return self._modules[module]
        path = self.module_path(module)
        info: _ModuleInfo | None = None
        if path is not None:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                tree = None
            if tree is not None:
                info = _index_module(module, path, tree)
        self._modules[module] = info
        return info

    def find_class(self, module: str, name: str, _seen: frozenset[str] = frozenset()) -> _ClassInfo | None:
        info = self.module(module)
        if info is None:
            return None
        if name in info.classes:
            return info.classes[name]
        target = info.imports.get(name)
        key = f"{module}:{name}"
        if target and key not in _seen and "." in target:
            parent, _, cls = target.rpartition(".")
            return self.find_class(parent, cls, _seen | {key})
        return None

    def class_members(self, module: str, name: str) -> tuple[set[str], set[str]] | None:
        """Methods and attributes including inherited ones (best effort)."""

        cls = self.find_class(module, name)
        if cls is None:
            return None
        methods: set[str] = set()
        attributes: set[str] = set()
        stack = [cls]
        seen: set[tuple[str, str]] = set()
        while stack:
            current = stack.pop()
            if (current.module, current.name) in seen:
                continue
            seen.add((current.module, current.name))
            methods |= current.methods
            attributes |= current.attributes
            owner = self.module(current.module)
            for base in current.bases:
                base_name = base.split(".")[-1]
                resolved = None
                if owner is not None and base in owner.classes:
                    resolved = owner.classes[base]
                elif owner is not None and base_name in owner.imports:
                    target = owner.imports[base_name]
                    parent, _, cls_name = target.rpartition(".")
                    resolved = self.find_class(parent, cls_name)
                if resolved is not None:
                    stack.append(resolved)
        return methods, attributes

    def iter_py_files(self, subdir: str) -> Iterable[Path]:
        base = self.root.joinpath(*subdir.split("."))
        if base.is_dir():
            yield from base.rglob("*.py")


def _index_module(module: str, path: Path, tree: ast.Module) -> _ModuleInfo:
    classes: dict[str, _ClassInfo] = {}
    imports: dict[str, str] = {}
    names: set[str] = set()
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    type_checking_only: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for sub in node.body:
                for inner in ast.walk(sub):
                    type_checking_only.add(id(inner))
    for node in ast.walk(tree):
        if id(node) in type_checking_only:
            continue  # typing-only imports do not exist at runtime
        if isinstance(node, ast.ImportFrom):
            source = node.module or ""
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                source = ".".join(p for p in parts + ([source] if source else []) if p)
            for alias in node.names:
                imports[alias.asname or alias.name] = f"{source}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = alias.name
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.FunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    for target in sub.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
    names |= set(imports)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        info = _ClassInfo(module=module, name=node.name, bases=[ast.unparse(b) for b in node.bases])
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info.methods.add(item.name)
                info.attributes.add(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        info.attributes.add(target.id)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                info.attributes.add(item.target.id)
        for sub in ast.walk(node):
            targets: list[ast.expr] = []
            if isinstance(sub, ast.Assign):
                targets = list(sub.targets)
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                targets = [sub.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if (
                        isinstance(leaf, ast.Attribute)
                        and isinstance(leaf.value, ast.Name)
                        and leaf.value.id == "self"
                    ):
                        info.attributes.add(leaf.attr)
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "setattr"
                and len(sub.args) >= 2
                and isinstance(sub.args[0], ast.Name)
                and sub.args[0].id == "self"
                and isinstance(sub.args[1], ast.Constant)
                and isinstance(sub.args[1].value, str)
            ):
                info.attributes.add(sub.args[1].value)
        classes.setdefault(node.name, info)
    return _ModuleInfo(name=module, path=path, tree=tree, classes=classes, imports=imports, names=names)


def _flag_present(index: SourceIndex, flag: str) -> tuple[bool, str]:
    """A CLI flag counts as present if it is spelled literally or declared as a
    server-args dataclass field (SGLang >= 0.5.1x generates flags from fields)."""

    literal = f'"{flag}"'
    literal_single = f"'{flag}'"
    field_name = flag.lstrip("-").replace("-", "_")
    field_re = re.compile(rf"^\s+{re.escape(field_name)}\s*:", re.M)
    candidates = [index.module_path("sglang.srt.server_args")]
    candidates += list(index.iter_py_files("sglang.srt.arg_groups"))
    for path in candidates:
        if path is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if literal in text or literal_single in text:
            return True, f"literal in {path.name}"
        if field_re.search(text):
            return True, f"field {field_name} in {path.name}"
    return False, "flag not found in server_args or arg_groups"


def check_requirement(index: SourceIndex, req: SurfaceRequirement) -> CompatibilityFinding:
    present = False
    detail = ""
    if req.kind == MODULE:
        present = index.module_path(req.module) is not None
        detail = "" if present else "module file not found"
    elif req.kind == MODULE_ATTR:
        info = index.module(req.module)
        present = info is not None and req.name in info.names
        detail = "" if present else ("module missing" if info is None else "name not defined at module level")
    elif req.kind == CLASS:
        present = index.find_class(req.module, req.name) is not None
        detail = "" if present else "class not found"
    elif req.kind in (METHOD, ATTRIBUTE):
        members = index.class_members(req.module, req.class_name)
        if members is None:
            detail = "class not found"
        else:
            methods, attributes = members
            present = req.name in (methods if req.kind == METHOD else attributes | methods)
            detail = "" if present else f"{req.kind} not found on class or resolvable bases"
    elif req.kind == CLI_FLAG:
        present, detail = _flag_present(index, req.name)
    else:
        detail = f"unknown requirement kind {req.kind!r}"
    return CompatibilityFinding(
        requirement=f"{req.kind}:{req.label()}",
        feature=req.feature,
        required=req.required,
        present=present,
        detail=detail if not present else "",
    )


def check_adapter(index: SourceIndex, adapter: AdapterSpec) -> CompatibilityReport:
    requirements = list(adapter.hook_surface_with_overrides()) + list(adapter.surface)
    findings = tuple(check_requirement(index, req) for req in requirements)
    return CompatibilityReport(
        backend_name="sglang",
        backend_version=index.version(),
        adapter=adapter.name,
        findings=findings,
    )


@lru_cache(maxsize=None)
def installed_source_root() -> str:
    """Directory containing the installed ``sglang`` package, without importing it."""

    spec = importlib.util.find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        return ""
    return str(Path(list(spec.submodule_search_locations)[0]).parent)


def check(source_root: str | None = None, adapter: AdapterSpec | None = None) -> CompatibilityReport:
    from .versions import get_adapter, newest_adapter

    root = source_root or installed_source_root()
    if not root:
        raise FileNotFoundError("sglang is not installed and no --sglang-src was given")
    index = SourceIndex(root)
    if adapter is None:
        from .selection import adapter_for_version

        adapter = adapter_for_version(index.version()) or newest_adapter()
    elif isinstance(adapter, str):  # type: ignore[unreachable]
        adapter = get_adapter(adapter)
    return check_adapter(index, adapter)


def _print_report(report: CompatibilityReport) -> None:
    status = "OK" if report.ok else "INCOMPATIBLE"
    print(f"[{status}] sglang {report.backend_version or '?'} vs adapter {report.adapter}: "
          f"{len(report.findings)} checks, {len(report.missing_required)} required missing, "
          f"{len(report.missing_optional)} optional missing")
    for finding in report.missing_required:
        print(f"  REQUIRED MISSING  [{finding.feature}] {finding.requirement}  ({finding.detail})")
    for finding in report.missing_optional:
        print(f"  optional missing  [{finding.feature}] {finding.requirement}")
    if report.broken_features():
        print("  broken features: " + ", ".join(report.broken_features()))


def main(argv: list[str] | None = None) -> int:
    from .versions import ADAPTERS, get_adapter

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sglang-src", default="", help="dir containing the sglang package (default: installed)")
    parser.add_argument("--adapter", default="", help="adapter name (default: best match for the source version)")
    parser.add_argument("--all", action="store_true", help="check against every registered adapter")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    root = args.sglang_src or installed_source_root()
    if not root:
        print("sglang is not installed; pass --sglang-src", file=sys.stderr)
        return 2
    index = SourceIndex(root)
    if args.all:
        adapters = list(ADAPTERS.values())
    elif args.adapter:
        adapters = [get_adapter(args.adapter)]
    else:
        from .selection import adapter_for_version
        from .versions import newest_adapter

        adapters = [adapter_for_version(index.version()) or newest_adapter()]
    reports = [check_adapter(index, adapter) for adapter in adapters]
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
    else:
        for report in reports:
            _print_report(report)
    return 0 if any(report.ok for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
