"""Compatibility guarantees of the SGLang-portability refactor.

1. Every script under ``scripts/`` still imports (catches broken re-exports).
2. Old ``agentic_kv`` module paths are aliases of the moved modules.
3. The SGLang server bootstrap (``sitecustomize`` -> ``agentic_kv`` sys.path
   fallback -> ``agentic_backends.sglang``) installs hooks in a fresh process
   with only ``src`` and SGLang on PYTHONPATH -- exactly how the launch
   scripts start the server.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for extra in (str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)


class ScriptImportTest(unittest.TestCase):
    def test_every_script_imports(self) -> None:
        failures = {}
        for path in sorted((ROOT / "scripts").glob("*.py")):
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    importlib.import_module(path.stem)
            except BaseException as exc:  # noqa: BLE001 - report everything
                failures[path.name] = f"{type(exc).__name__}: {exc}"
        self.assertEqual(failures, {})


class ModuleAliasTest(unittest.TestCase):
    def test_old_paths_alias_new_modules(self) -> None:
        pairs = {
            "agentic_kv.sglang_trace_patch": "agentic_backends.sglang.trace.patch",
            "agentic_kv.sglang_compat": "agentic_backends.sglang.compat",
            "agentic_kv.nvtx": "agentic_backends.sglang.instrumentation.nvtx",
            "agentic_kv.runtime_telemetry": "agentic_backends.sglang.instrumentation.runtime_telemetry",
            "agentic_kv.torch_cuda_profiler": "agentic_backends.sglang.instrumentation.torch_cuda_profiler",
            "agentic_kv.controller.backend": "agentic_backends.sglang.adapters",
            "agentic_kv.sglang_adapters.capabilities": "agentic_backends.sglang.capabilities",
            "agentic_kv.sglang_adapters.v0510": "agentic_backends.sglang.versions.v0510",
            # testbed split (second pass)
            "agentic_kv.block_ledger.normalizer": "agentic_reports.block_ledger.normalizer",
            "agentic_kv.evidence_audit": "agentic_reports.evidence_audit",
            "agentic_kv.evidence_schema": "agentic_reports.evidence_schema",
            "agentic_kv.policies": "agentic_experiments.basic_workload.policies",
            "agentic_kv.sglang_client": "agentic_experiments.basic_workload.sglang_client",
            "agentic_kv.harness_scenarios.real_runner": "agentic_experiments.real_runner",
            "agentic_kv.block_ledger.events": "agentic_reports.block_ledger.events",
            "agentic_kv.harness_scenarios.policies.baseline": "agentic_harness_scenarios.policies.baseline",
            "agentic_kv.harness_scenarios.adapters.controller_signal": "agentic_harness_scenarios.adapters.controller_signal",
        }
        for old, new in pairs.items():
            self.assertIs(importlib.import_module(old), importlib.import_module(new), old)

    def test_compat_packages_reexport_without_duplicating_modules(self) -> None:
        old = importlib.import_module("agentic_kv.block_ledger")
        new = importlib.import_module("agentic_reports.block_ledger")
        self.assertIs(old.KVEventType, new.KVEventType)
        self.assertIs(old.build_block_ledger, new.build_block_ledger)

    def test_script_wrappers_alias_package_modules(self) -> None:
        pairs = {
            "run_multi_harness_replay_driver": "agentic_experiments.runners.run_multi_harness_replay_driver",
            "harness_sglang_gateway": "agentic_experiments.gateway.harness_sglang_gateway",
            "build_multi_harness_deadline_summary": "agentic_reports.builders.build_multi_harness_deadline_summary",
            "replay_path_classifier": "agentic_reports.analysis.replay_path_classifier",
            "smoke_priority_radix_eviction": "agentic_backends.sglang.tools.smoke_priority_radix_eviction",
        }
        for old, new in pairs.items():
            self.assertIs(importlib.import_module(old), importlib.import_module(new), old)

    def test_private_state_is_shared_through_alias(self) -> None:
        old = importlib.import_module("agentic_kv.sglang_trace_patch")
        new = importlib.import_module("agentic_backends.sglang.trace.patch")
        self.assertIs(old._PREPARABLE_PREFIXES, new._PREPARABLE_PREFIXES)


class ServerBootstrapTest(unittest.TestCase):
    def test_sitecustomize_installs_hooks_in_fresh_process(self) -> None:
        sys.path.insert(0, str(REPO / "packages" / "agentic-backend-sglang" / "tests"))
        from fake_sglang import build_fake_sglang
        from agentic_backends.sglang.versions import get_adapter

        with tempfile.TemporaryDirectory() as tmp:
            site = build_fake_sglang(Path(tmp) / "site", get_adapter("v0510"), "0.5.10.post1")
            trace = Path(tmp) / "trace.jsonl"
            env = dict(os.environ)
            env.update(
                PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(site)]),
                AGENTIC_KV_TRACE_ENABLE="1",
                AGENTIC_KV_TRACE_PATH=str(trace),
            )
            env.pop("AGENTIC_SGLANG_STRICT", None)
            code = (
                "from sglang.srt.mem_cache.radix_cache import RadixCache;"
                "print(getattr(RadixCache.evict, '_agentic_kv_wrapped', False))"
            )
            proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), "True", proc.stderr)
            events = [json.loads(line)["event"] for line in trace.read_text().splitlines()]
            self.assertIn("trace.install.summary", events)


if __name__ == "__main__":
    unittest.main()
