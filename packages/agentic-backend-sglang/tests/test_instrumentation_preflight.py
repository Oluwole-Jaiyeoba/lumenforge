from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_backends.sglang.instrumentation_preflight import run_preflight


class InstrumentationPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.contract = self.tmp / "contract.json"
        self.contract.write_text(
            json.dumps(
                {
                    "supported_adapters": ["v0510"],
                    "required": [
                        {"id": "scheduler", "installed_hooks_any": ["Scheduler.process_batch_result"], "source_event_prefixes_any": ["scheduler.process_batch_result"]},
                        {"id": "ledger", "environment": {"TRACE_CONTROLLER_DECISIONS": "1"}},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.runtime = self.tmp / "runtime.json"
        self.runtime.write_text(json.dumps({"adapter": "v0510"}), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, stage: str) -> argparse.Namespace:
        return argparse.Namespace(
            contract=str(self.contract),
            out=str(self.tmp / "out.json"),
            stage=stage,
            runtime_contract=str(self.runtime),
            installation_report=str(self.tmp / "install.json"),
            gateway_base="http://gateway",
            model="model",
            trace=str(self.tmp / "trace.jsonl"),
            timeout=1,
            trace_wait=0,
            policy="strict",
        )

    def test_static_accepts_supported_adapter(self) -> None:
        result = run_preflight(self._args("static"))
        self.assertTrue(result["valid"])

    def test_live_rejects_missing_required_hook(self) -> None:
        Path(self._args("live").installation_report).write_text(json.dumps({"installed_hooks": []}), encoding="utf-8")
        with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}):
            result = run_preflight(self._args("live"))
        self.assertFalse(result["valid"])
        self.assertFalse(result["experiment_allowed"])

    def test_live_requires_installation_and_live_event(self) -> None:
        args = self._args("live")
        Path(args.installation_report).write_text(json.dumps({"installed_hooks": ["Scheduler.process_batch_result"]}), encoding="utf-8")
        Path(args.trace).write_text(json.dumps({"ts_ns": 2, "source_event": "scheduler.process_batch_result.end"}) + "\n", encoding="utf-8")
        old = os.environ.get("TRACE_CONTROLLER_DECISIONS")
        os.environ["TRACE_CONTROLLER_DECISIONS"] = "1"
        try:
            with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}):
                result = run_preflight(args)
        finally:
            if old is None:
                os.environ.pop("TRACE_CONTROLLER_DECISIONS", None)
            else:
                os.environ["TRACE_CONTROLLER_DECISIONS"] = old
        self.assertTrue(result["valid"])
        self.assertTrue(result["sentinel"]["excluded_from_measurement"])


if __name__ == "__main__":
    unittest.main()
