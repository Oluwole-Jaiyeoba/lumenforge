import importlib
import os
import pkgutil
import unittest
from pathlib import Path
from unittest import mock

import agentic_experiments
from agentic_experiments import paths

REPO = Path(__file__).resolve().parents[3]


class PackageTest(unittest.TestCase):
    def test_every_module_imports(self) -> None:
        failures = {}
        for info in pkgutil.walk_packages(agentic_experiments.__path__, "agentic_experiments."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # noqa: BLE001
                failures[info.name] = f"{type(exc).__name__}: {exc}"
        self.assertEqual(failures, {})

    def test_testbed_root_is_the_former_scripts_parent(self) -> None:
        paths.testbed_root.cache_clear()
        with mock.patch.dict(os.environ, {"AGENTIC_TESTBED_ROOT": ""}):
            self.assertEqual(paths.testbed_root(), REPO / "sglang_direct_kv")
        self.assertTrue(paths.testbed_script("harness_sglang_gateway.py").is_file())

    def test_testbed_root_env_override(self) -> None:
        paths.testbed_root.cache_clear()
        try:
            with mock.patch.dict(os.environ, {"AGENTIC_TESTBED_ROOT": "/tmp/somewhere"}):
                self.assertEqual(paths.testbed_root(), Path("/tmp/somewhere").resolve())
        finally:
            paths.testbed_root.cache_clear()


if __name__ == "__main__":
    unittest.main()
