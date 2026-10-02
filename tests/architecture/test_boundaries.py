"""Architecture guard rails.  Run: python -m pytest tests/architecture

Fails when:
- a portable package imports a first-party package it is not allowed to
  (dependency direction: core <- backend_api <- controller; core <- harnesses
  <- gateway; core/backend_api <- agentic_backends.sglang);
- any package other than ``agentic_backends`` imports ``sglang`` or spells
  ``sglang.srt``;
- backend vocabulary (sglang/hicache/radix) GROWS in a portable package
  (see vocabulary_allowlist.json -- counts may only go down);
- a new testbed file starts touching SGLang internals directly.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from boundaries import ALLOWLIST_PATH, LEGACY_SGLANG_ALLOWLIST_PATH, PACKAGES, REPO, iter_py, scan

RESULT = scan()


class BoundaryTest(unittest.TestCase):
    def test_package_dependency_direction(self) -> None:
        self.assertEqual(RESULT["import_violations"], [])

    def test_only_backend_package_touches_sglang(self) -> None:
        self.assertEqual(RESULT["sglang_violations"], [])

    def test_backend_vocabulary_does_not_grow(self) -> None:
        allowed = json.loads(ALLOWLIST_PATH.read_text())["counts"]
        grown = {
            path: (count, allowed.get(path, 0))
            for path, count in RESULT["vocabulary"].items()
            if count > allowed.get(path, 0)
        }
        self.assertEqual(
            grown,
            {},
            "backend-specific words leaked into portable packages (found, allowed). Move the code to "
            "packages/agentic-backend-sglang or use backend-neutral names.",
        )

    def test_no_new_testbed_files_touch_sglang_internals(self) -> None:
        allowed = set(json.loads(LEGACY_SGLANG_ALLOWLIST_PATH.read_text())["files"])
        self.assertEqual(sorted(set(RESULT["legacy_sglang_internal_users"]) - allowed), [])

    def test_no_lane_local_backend_hook_tables(self) -> None:
        violations = []
        for name, (relative, _) in PACKAGES.items():
            if name == "agentic_backends":
                continue
            for path in iter_py(REPO / relative):
                source = path.read_text(encoding="utf-8")
                if "SGLangHookTarget" in source or "HOOK_TARGETS" in source:
                    violations.append(str(path.relative_to(REPO)))
        legacy_shim = "sglang_direct_kv/src/agentic_kv/sglang_adapters/__init__.py"
        for relative in ("sglang_direct_kv/src", "sglang_direct_kv/scripts"):
            for path in iter_py(REPO / relative):
                name = str(path.relative_to(REPO))
                if name == legacy_shim:
                    continue
                source = path.read_text(encoding="utf-8")
                if "SGLangHookTarget" in source or "HOOK_TARGETS" in source:
                    violations.append(name)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
