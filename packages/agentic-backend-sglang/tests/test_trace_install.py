"""Run the real in-server trace installer against a fake SGLang (no torch/GPU).

Each case runs in a subprocess so hook installation and module globals never
leak between tests -- the same way sitecustomize runs inside a fresh SGLang
server process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fake_sglang import build_fake_sglang  # noqa: E402

from agentic_backends.sglang.versions import get_adapter  # noqa: E402

REPO = HERE.parents[2]
PACKAGE_SRC = [str(p) for p in sorted((REPO / "packages").glob("*/src"))]

SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    from agentic_backends.sglang.trace import install
    install()
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
    result = HiRadixCache().match_prefix("key")
    rows = [json.loads(line) for line in open(os.environ["AGENTIC_KV_TRACE_PATH"])]
    print(json.dumps({"result": result, "wrapped": getattr(HiRadixCache.match_prefix, "_agentic_kv_wrapped", False), "events": [r.get("event") for r in rows], "rows": rows}))
    """
)


def run_install(root: Path, trace: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": os.pathsep.join([str(root), *PACKAGE_SRC]),
            "AGENTIC_KV_TRACE_ENABLE": "1",
            "AGENTIC_KV_TRACE_PATH": str(trace),
            "AGENTIC_KV_TRACE_SCHEDULER": "1",
        }
    )
    env.pop("AGENTIC_SGLANG_STRICT", None)
    env.pop("AGENTIC_SGLANG_ADAPTER", None)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=120)


class TraceInstallTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_all_hooks_install_and_emit_events_on_reference_version(self) -> None:
        root = build_fake_sglang(self.tmp / "site", get_adapter("v0510"), "0.5.10.post1")
        proc = run_install(root, self.tmp / "trace.jsonl")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("WARNING", proc.stderr)
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(out["result"], "match_prefix")  # wrapper returns the original result
        self.assertTrue(out["wrapped"])
        rows = {row["event"]: row for row in out["rows"]}
        self.assertEqual(rows["trace.adapter.selected"]["adapter"], "v0510")
        self.assertEqual(rows["trace.adapter.selected"]["selection"]["status"], "tested")
        summary = rows["trace.install.summary"]
        self.assertEqual(summary["missing_required_hooks"], [])
        self.assertGreater(summary["installed_hook_count"], 30)
        self.assertIn("hiradix.match_prefix.start", out["events"])
        self.assertIn("hiradix.match_prefix.end", out["events"])

    def test_missing_required_hook_is_loud_not_silent(self) -> None:
        root = build_fake_sglang(self.tmp / "site", get_adapter("v0510"), "0.5.10.post1", drop={"HiCacheController.load"})
        proc = run_install(root, self.tmp / "trace.jsonl")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("required SGLang trace hooks could not be installed", proc.stderr)
        rows = {row["event"]: row for row in json.loads(proc.stdout.strip().splitlines()[-1])["rows"]}
        self.assertTrue(any("HiCacheController.load" in item for item in rows["trace.install.summary"]["missing_required_hooks"]))

    def test_strict_mode_refuses_broken_instrumentation(self) -> None:
        root = build_fake_sglang(self.tmp / "site", get_adapter("v0510"), "0.5.10.post1", drop={"HiCacheController.load"})
        proc = run_install(root, self.tmp / "trace.jsonl", {"AGENTIC_SGLANG_STRICT": "1"})
        self.assertNotEqual(proc.returncode, 0)

    def test_newer_version_uses_moved_hook_locations(self) -> None:
        root = build_fake_sglang(self.tmp / "site", get_adapter("v0520"), "0.5.20")
        proc = run_install(root, self.tmp / "trace.jsonl")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = {row["event"]: row for row in json.loads(proc.stdout.strip().splitlines()[-1])["rows"]}
        self.assertEqual(rows["trace.adapter.selected"]["adapter"], "v0520")
        self.assertEqual(rows["trace.install.summary"]["missing_required_hooks"], [])
        self.assertIn("static", proc.stderr)  # warns that 0.5.20 support is static-verified only


if __name__ == "__main__":
    unittest.main()
