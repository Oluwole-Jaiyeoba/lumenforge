"""Static architecture rules for the portable packages (see docs/architecture/README_RESTRUCTURING.md).

Kept separate from the test so the same scanner can print a report::

    python tests/architecture/boundaries.py            # human-readable summary
    python tests/architecture/boundaries.py --baseline # print current vocabulary counts (JSON)
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# import name -> (source dir, allowed first-party imports)
PACKAGES: dict[str, tuple[str, frozenset[str]]] = {
    "agentic_core": ("packages/agentic-core/src/agentic_core", frozenset()),
    "agentic_work_audit": ("packages/agentic-work-audit/src/agentic_work_audit", frozenset()),
    "agentic_backend_api": ("packages/agentic-backend-api/src/agentic_backend_api", frozenset({"agentic_core"})),
    "agentic_controller": (
        "packages/agentic-controller/src/agentic_controller",
        frozenset({"agentic_core", "agentic_backend_api"}),
    ),
    "agentic_harnesses": ("packages/agentic-harnesses/src/agentic_harnesses", frozenset({"agentic_core"})),
    "agentic_prompt_codec": ("packages/agentic-prompt-codec/src/agentic_prompt_codec", frozenset({"agentic_core"})),
    "agentic_gateway": ("packages/agentic-gateway/src/agentic_gateway", frozenset({"agentic_core", "agentic_harnesses"})),
    "agentic_backends": (
        "packages/agentic-backend-sglang/src/agentic_backends",
        frozenset({"agentic_core", "agentic_backend_api", "agentic_work_audit"}),
    ),
    "agentic_harness_scenarios": ("packages/agentic-harness-scenarios/src/agentic_harness_scenarios", frozenset()),
    "agentic_hardware_probes": ("packages/agentic-hardware-probes/src/agentic_hardware_probes", frozenset()),
    # Report layer: reads raw SGLang trace rows through the backend's event map
    # (agentic_backends.sglang.hooks) until reports move onto normalized
    # observations; uses controller runtime classes and prompt-codec reporting.
    "agentic_reports": (
        "packages/agentic-reports/src/agentic_reports",
        frozenset({"agentic_core", "agentic_backends", "agentic_controller", "agentic_prompt_codec", "agentic_hardware_probes"}),
    ),
    # Composition root for experiments: may use every portable package and the
    # SGLang backend, but never the report layer or the testbed (agentic_kv).
    "agentic_experiments": (
        "packages/agentic-experiments/src/agentic_experiments",
        frozenset(
            {
                "agentic_core",
                "agentic_backend_api",
                "agentic_controller",
                "agentic_harnesses",
                "agentic_gateway",
                "agentic_prompt_codec",
                "agentic_harness_scenarios",
                "agentic_backends",
                "agentic_work_audit",
            }
        ),
    ),
}

FIRST_PARTY = set(PACKAGES) | {"agentic_kv", "sitecustomize"}
# Only this package may import SGLang or spell its private module paths.
SGLANG_OWNER = "agentic_backends"
VOCABULARY = re.compile(r"sglang|hicache|radix", re.IGNORECASE)
ALLOWLIST_PATH = Path(__file__).with_name("vocabulary_allowlist.json")
# Testbed files (outside packages/) allowed to touch SGLang internals directly.
LEGACY_SGLANG_ALLOWLIST_PATH = Path(__file__).with_name("legacy_sglang_internal_users.json")


def iter_py(directory: Path):
    yield from sorted(p for p in directory.rglob("*.py") if "__pycache__" not in p.parts)


def top_level_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def docstring_nodes(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
                if isinstance(body[0].value.value, str):
                    ids.add(id(body[0].value))
    return ids


def vocabulary_hits(tree: ast.AST) -> int:
    """Occurrences of backend vocabulary in identifiers and non-docstring strings."""

    skip = docstring_nodes(tree)
    hits = 0
    for node in ast.walk(tree):
        text = ""
        if isinstance(node, ast.Name):
            text = node.id
        elif isinstance(node, ast.Attribute):
            text = node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = node.name
        elif isinstance(node, ast.arg):
            text = node.arg
        elif isinstance(node, ast.alias):
            text = node.asname or node.name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip:
            text = node.value
        hits += len(VOCABULARY.findall(text))
    return hits


def scan() -> dict[str, object]:
    import_violations: list[str] = []
    sglang_violations: list[str] = []
    vocabulary: dict[str, int] = {}
    for name, (rel, allowed) in PACKAGES.items():
        for path in iter_py(REPO / rel):
            rel_path = str(path.relative_to(REPO))
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=rel_path)
            imported = top_level_imports(tree)
            for other in sorted((imported & FIRST_PARTY) - allowed - {name}):
                import_violations.append(f"{rel_path}: {name} must not import {other}")
            if name != SGLANG_OWNER:
                if "sglang" in imported:
                    sglang_violations.append(f"{rel_path}: imports sglang")
                if "sglang.srt" in source:
                    sglang_violations.append(f"{rel_path}: references sglang.srt")
                count = vocabulary_hits(tree)
                if count:
                    vocabulary[rel_path] = count
    legacy_users: list[str] = []
    for base in ("sglang_direct_kv/src", "sglang_direct_kv/scripts"):
        for path in iter_py(REPO / base):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            if "sglang" in top_level_imports(tree) or "sglang.srt" in source:
                legacy_users.append(str(path.relative_to(REPO)))
    return {
        "import_violations": import_violations,
        "sglang_violations": sglang_violations,
        "vocabulary": vocabulary,
        "legacy_sglang_internal_users": sorted(legacy_users),
    }


def main() -> int:
    result = scan()
    if "--baseline" in sys.argv:
        print(json.dumps({"vocabulary": result["vocabulary"], "legacy": result["legacy_sglang_internal_users"]}, indent=2, sort_keys=True))
        return 0
    for key in ("import_violations", "sglang_violations"):
        print(f"{key}: {len(result[key])}")
        for line in result[key]:
            print("  " + line)
    print(f"backend vocabulary in portable packages: {sum(result['vocabulary'].values())} hits in {len(result['vocabulary'])} files")
    print(f"testbed files touching SGLang internals: {len(result['legacy_sglang_internal_users'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
