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

from boundaries import ALLOWLIST_PATH, LEGACY_SGLANG_ALLOWLIST_PATH, scan

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


if __name__ == "__main__":
    unittest.main()
