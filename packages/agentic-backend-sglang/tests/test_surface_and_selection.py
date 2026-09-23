from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_sglang import build_fake_sglang  # noqa: E402

from agentic_backend_api import CompatibilityProbe, UnsupportedBackendVersion  # noqa: E402
from agentic_backends.sglang import selection, surface  # noqa: E402
from agentic_backends.sglang.versions import ADAPTERS, get_adapter  # noqa: E402


class SurfaceCheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_every_adapter_passes_against_its_own_fake(self) -> None:
        for name, adapter in ADAPTERS.items():
            root = build_fake_sglang(self.tmp / name, adapter, adapter.version_range[0])
            report = surface.check_adapter(surface.SourceIndex(root), adapter)
            self.assertTrue(report.ok, f"{name}: {report.to_dict()}")
            self.assertEqual(report.missing_optional, ())

    def test_missing_required_method_is_reported_with_feature(self) -> None:
        adapter = get_adapter("v0510")
        root = build_fake_sglang(self.tmp, adapter, "0.5.10", drop={"HiRadixCache.load_back"})
        report = surface.check_adapter(surface.SourceIndex(root), adapter)
        self.assertFalse(report.ok)
        self.assertIn("control.prepare_prefix", report.broken_features())
        self.assertTrue(any("HiRadixCache.load_back" in f.requirement for f in report.missing_required))

    def test_missing_optional_hook_does_not_fail(self) -> None:
        adapter = get_adapter("v0510")
        root = build_fake_sglang(self.tmp, adapter, "0.5.10", drop={"NSATokenToKVPoolHost"})
        report = surface.check_adapter(surface.SourceIndex(root), adapter)
        self.assertTrue(report.ok)
        self.assertTrue(report.missing_optional)

    def test_missing_cli_flag_and_request_field(self) -> None:
        adapter = get_adapter("v0510")
        root = build_fake_sglang(
            self.tmp, adapter, "0.5.10", drop={"--enable-priority-scheduling", "ChatCompletionRequest.priority"}
        )
        report = surface.check_adapter(surface.SourceIndex(root), adapter)
        self.assertEqual(set(report.broken_features()), {"launch.flags", "request.lowering"})

    def test_version_read_from_dist_info(self) -> None:
        root = build_fake_sglang(self.tmp, get_adapter("v0510"), "0.5.10.post1")
        self.assertEqual(surface.SourceIndex(root).version(), "0.5.10.post1")

    def test_inheritance_import_resolution_and_type_checking_imports(self) -> None:
        pkg = self.tmp / "sglang" / "srt"
        pkg.mkdir(parents=True)
        (self.tmp / "sglang" / "__init__.py").touch()
        (pkg / "__init__.py").touch()
        (pkg / "base.py").write_text("class Base:\n    def inherited(self):\n        self.from_base = 1\n")
        (pkg / "child.py").write_text(
            textwrap.dedent(
                """
                from typing import TYPE_CHECKING
                from sglang.srt.base import Base
                if TYPE_CHECKING:
                    from sglang.srt.base import Base as OnlyForTyping
                class Child(Base):
                    field: int = 0
                    def own(self):
                        setattr(self, "dynamic", 1)
                """
            )
        )
        index = surface.SourceIndex(self.tmp)
        methods, attributes = index.class_members("sglang.srt.child", "Child")
        self.assertIn("inherited", methods)
        self.assertIn("from_base", attributes)
        self.assertIn("field", attributes)
        self.assertIn("dynamic", attributes)
        self.assertIsNone(index.find_class("sglang.srt.child", "OnlyForTyping"))

    def test_cli_flag_declared_as_dataclass_field(self) -> None:
        srt = self.tmp / "sglang" / "srt"
        (srt / "arg_groups").mkdir(parents=True)
        (self.tmp / "sglang" / "__init__.py").touch()
        (srt / "__init__.py").touch()
        (srt / "server_args.py").write_text("x = 1\n")
        (srt / "arg_groups" / "fields.py").write_text("class A:\n    enable_priority_scheduling: bool = False\n")
        present, detail = surface._flag_present(surface.SourceIndex(self.tmp), "--enable-priority-scheduling")
        self.assertTrue(present, detail)

    def test_cli_exit_codes(self) -> None:
        adapter = get_adapter("v0510")
        good = build_fake_sglang(self.tmp / "good", adapter, "0.5.10")
        bad = build_fake_sglang(self.tmp / "bad", adapter, "0.5.10", drop={"HiRadixCache"})
        with mock.patch("sys.stdout"):
            self.assertEqual(surface.main(["--sglang-src", str(good), "--adapter", "v0510"]), 0)
            self.assertEqual(surface.main(["--sglang-src", str(bad), "--adapter", "v0510"]), 1)


class SelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for key in ("AGENTIC_SGLANG_ADAPTER", "AGENTIC_SGLANG_STRICT"):
            os.environ.pop(key, None)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        self._env.stop()

    def test_version_ranges(self) -> None:
        expected = {
            "0.5.10": "v0510",
            "0.5.10.post1": "v0510",
            "0.5.11": "v0511",
            "0.5.12.post1": "v0511",
            "0.5.13": "v0513",
            "0.5.15.post1": "v0513",
            "0.5.16": "v0516",
            "0.5.19": "v0516",
            "0.5.20": "v0520",
        }
        for version, name in expected.items():
            self.assertEqual(selection.adapter_for_version(version).name, name, version)
        self.assertIsNone(selection.adapter_for_version("0.5.9"))
        self.assertIsNone(selection.adapter_for_version("0.6.0"))
        self.assertIsNone(selection.adapter_for_version("garbage"))

    def test_status_tested_vs_in_range(self) -> None:
        self.assertEqual(selection.select_adapter("0.5.10.post1").status, "tested")
        sel = selection.select_adapter("0.5.12")
        self.assertEqual(sel.status, "in_range")
        self.assertIn("capability_probe", sel.warning())
        self.assertEqual(selection.select_adapter("0.5.10.post1").warning(), "")

    def test_legacy_names_match_pre_refactor_behavior(self) -> None:
        # Pre-refactor: unparsable -> v0510, < 0.5.11 -> v0510, >= 0.5.11 -> v0511.
        self.assertEqual(selection.legacy_adapter_name(""), "v0510")
        self.assertEqual(selection.legacy_adapter_name("weird"), "v0510")
        self.assertEqual(selection.legacy_adapter_name("0.4.0"), "v0510")
        self.assertEqual(selection.legacy_adapter_name("0.5.10.post1"), "v0510")
        self.assertEqual(selection.legacy_adapter_name("0.5.11"), "v0511")
        # New: unknown future versions go to the newest adapter instead of v0511.
        self.assertEqual(selection.legacy_adapter_name("9.9.9"), list(ADAPTERS)[-1])

    def test_forced_adapter(self) -> None:
        with mock.patch.dict(os.environ, {"AGENTIC_SGLANG_ADAPTER": "v0511"}):
            sel = selection.select_adapter("0.5.10")
        self.assertEqual((sel.adapter.name, sel.status), ("v0511", "forced"))
        with mock.patch.dict(os.environ, {"AGENTIC_SGLANG_ADAPTER": "nope"}):
            with self.assertRaises(UnsupportedBackendVersion):
                selection.select_adapter("0.5.10")

    def test_unknown_version_probes_source_then_falls_back(self) -> None:
        root = build_fake_sglang(self.tmp / "ok", get_adapter("v0513"), "0.7.0")
        sel = selection.select_adapter("0.7.0", source_root=str(root))
        self.assertEqual(sel.status, "probed")
        self.assertTrue(sel.report.ok)

        broken = build_fake_sglang(self.tmp / "bad", get_adapter("v0510"), "0.7.0", drop={"HiRadixCache", "RadixCache"})
        with mock.patch.object(surface, "installed_source_root", return_value=""):
            sel = selection.select_adapter("0.7.0", source_root=str(broken))
        self.assertEqual(sel.status, "unverified")
        self.assertIn("not covered", sel.warning())
        with self.assertRaises(UnsupportedBackendVersion):
            selection.select_adapter("0.7.0", source_root=str(broken), strict=True)

    def test_strict_rejects_missing_required_surface(self) -> None:
        root = build_fake_sglang(self.tmp, get_adapter("v0510"), "0.5.10", drop={"HiRadixCache.load_back"})
        sel = selection.select_adapter("0.5.10", source_root=str(root), surface_check=True)
        self.assertFalse(sel.report.ok)
        self.assertIn("control.prepare_prefix", sel.warning())
        with self.assertRaises(UnsupportedBackendVersion):
            selection.select_adapter("0.5.10", source_root=str(root), surface_check=True, strict=True)


class ProtocolConformanceTest(unittest.TestCase):
    def test_module_level_check_satisfies_probe_protocol_shape(self) -> None:
        class Probe:
            def check(self, source_root=None):
                return surface.check(source_root)

        self.assertIsInstance(Probe(), CompatibilityProbe)


if __name__ == "__main__":
    unittest.main()
