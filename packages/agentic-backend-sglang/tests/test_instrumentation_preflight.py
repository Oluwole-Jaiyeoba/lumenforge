from __future__ import annotations

import argparse
import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentic_backends.sglang.instrumentation_preflight import _send_sentinel, run_preflight
from agentic_backends.sglang.hook_registry import resolve_hook


SCHEDULER_HOOK = "sglang.srt.managers.scheduler.Scheduler.process_batch_result"


class InstrumentationPreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.contract = self.tmp / "contract.json"
        self.contract.write_text(json.dumps({
            "supported_adapters": ["v0510"],
            "required": [
                {"id": "gateway", "trace_event_prefixes_all": ["m27.request.start", "m27.request.end"], "required_fields": ["ttft_ms"]},
                {"id": "scheduler", "hook_id": "scheduler_batch_observation"},
                {"id": "ledger", "controller_probe": True},
                {"id": "action_acknowledgement", "controller_probe": True},
            ],
            "optional": [{"id": "transfer", "hook_id": "hicache_transfer_observation"}],
        }), encoding="utf-8")
        self.runtime = self.tmp / "runtime.json"
        self.runtime.write_text(json.dumps({"adapter": "v0510"}), encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _args(self, stage: str, policy: str = "strict") -> argparse.Namespace:
        return argparse.Namespace(
            contract=str(self.contract), out=str(self.tmp / "out.json"), stage=stage,
            runtime_contract=str(self.runtime), installation_report=str(self.tmp / "install.json"),
            gateway_base="http://gateway", model="model", gateway_trace=str(self.tmp / "gateway.jsonl"),
            backend_trace=str(self.tmp / "backend.jsonl"),
            sentinel_trace=str(self.tmp / "sentinel.jsonl"), controller_probe_report=str(self.tmp / "controller.jsonl"),
            controller_mode="controller_scheduler_priority", timeout=1, trace_wait=0, policy=policy,
        )

    @staticmethod
    def _controller_rows() -> dict[str, object]:
        return {"ran": True, "rows": [{"event": "m27.controller.preflight.decision"}, {"event": "m27.controller.preflight.action_ack", "acted_count": 1}]}

    def test_static_accepts_supported_adapter(self) -> None:
        self.assertTrue(run_preflight(self._args("static"))["valid"])

    def test_static_blocks_version_adapter_mismatch(self) -> None:
        self.runtime.write_text(json.dumps({"adapter": "v0520"}), encoding="utf-8")
        self.assertFalse(run_preflight(self._args("static"))["valid"])

    def test_live_rejects_missing_required_hook(self) -> None:
        args = self._args("live")
        Path(args.installation_report).write_text(json.dumps({"adapter": "v0510", "installed_hooks": [], "hook_statuses": {SCHEDULER_HOOK: "missing_method"}}), encoding="utf-8")
        with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}), patch("agentic_backends.sglang.instrumentation_preflight._controller_probe", return_value=self._controller_rows()):
            result = run_preflight(args)
        self.assertFalse(result["valid"])
        self.assertFalse(result["experiment_allowed"])

    def test_live_rejects_installed_hook_without_live_event(self) -> None:
        args = self._args("live")
        Path(args.installation_report).write_text(json.dumps({"adapter": "v0510", "installed_hooks": [SCHEDULER_HOOK]}), encoding="utf-8")
        Path(args.gateway_trace).write_text("\n".join(json.dumps(row) for row in [{"ts_ns": 2, "event": "m27.request.start"}, {"ts_ns": 3, "event": "m27.request.end", "ttft_ms": 1}]) + "\n", encoding="utf-8")
        with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}), patch("agentic_backends.sglang.instrumentation_preflight._controller_probe", return_value=self._controller_rows()):
            result = run_preflight(args)
        self.assertFalse(result["valid"])

    def test_live_requires_installation_and_full_evidence(self) -> None:
        args = self._args("live")
        Path(args.installation_report).write_text(json.dumps({"adapter": "v0510", "installed_hooks": [SCHEDULER_HOOK]}), encoding="utf-8")
        gateway_rows = [{"ts_ns": 2, "event": "m27.request.start"}, {"ts_ns": 3, "event": "m27.request.end", "ttft_ms": 1}]
        backend_rows = [{"ts_ns": 4, "source_event": "scheduler.process_batch_result.end"}]
        Path(args.gateway_trace).write_text("\n".join(json.dumps(row) for row in gateway_rows) + "\n", encoding="utf-8")
        Path(args.backend_trace).write_text("\n".join(json.dumps(row) for row in backend_rows) + "\n", encoding="utf-8")
        with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}), patch("agentic_backends.sglang.instrumentation_preflight._controller_probe", return_value=self._controller_rows()):
            result = run_preflight(args)
        self.assertTrue(result["valid"])
        self.assertTrue(Path(args.sentinel_trace).exists())

    def test_observe_only_keeps_invalid_result_explicit(self) -> None:
        args = self._args("live", policy="observe_only")
        Path(args.installation_report).write_text(json.dumps({"adapter": "v0510", "installed_hooks": []}), encoding="utf-8")
        with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}), patch("agentic_backends.sglang.instrumentation_preflight._controller_probe", return_value=self._controller_rows()):
            result = run_preflight(args)
        self.assertFalse(result["valid"])
        self.assertTrue(result["experiment_allowed"])

    def test_shared_queue_profile_rejects_missing_request_identity(self) -> None:
        args = self._args("live")
        args.shared_profile = "controller_queue"
        installed = [item["target"] for hook_id in ("request_lifecycle", "scheduler_batch_observation", "request_completion_timing")
                     for item in resolve_hook(hook_id, "v0510")["targets"]]
        Path(args.installation_report).write_text(json.dumps({"adapter": "v0510", "installed_hooks": installed}), encoding="utf-8")
        Path(args.gateway_trace).write_text("\n".join(json.dumps(row) for row in [
            {"ts_ns": 2, "event": "m27.request.start"},
            {"ts_ns": 3, "event": "m27.request.end", "ttft_ms": 1},
        ]) + "\n", encoding="utf-8")
        Path(args.backend_trace).write_text("\n".join(json.dumps(row) for row in [
            {"ts_ns": 4, "source_event": "scheduler.handle_generate_request.end"},
            {"ts_ns": 5, "source_event": "scheduler.get_next_batch_to_run.end"},
            {"ts_ns": 6, "source_event": "scheduler.process_batch_result.end"},
        ]) + "\n", encoding="utf-8")
        with patch("agentic_backends.sglang.instrumentation_preflight._send_sentinel", return_value={"sent": True, "started_ns": 1}), patch("agentic_backends.sglang.instrumentation_preflight._controller_probe", return_value=self._controller_rows()):
            result = run_preflight(args)
        self.assertFalse(result["valid"])
        shared = next(item for item in result["checks"] if item["id"] == "shared_profile:controller_queue")
        self.assertEqual(shared["details"]["invalid_fields"], ["request.accepted"])

    def test_sentinel_retry_uses_fresh_request_identity(self) -> None:
        response = MagicMock()
        with patch(
            "agentic_backends.sglang.instrumentation_preflight.urllib.request.urlopen",
            side_effect=[OSError("temporarily unavailable"), response],
        ) as urlopen, patch("agentic_backends.sglang.instrumentation_preflight.time.sleep"):
            result = _send_sentinel("http://gateway", "model", 1)
        request_ids = []
        for call in urlopen.call_args_list:
            body = json.loads(call.args[0].data)
            encoded = body["messages"][0]["content"].split("HARNESS_REPLAY_EXPERIMENT_JSON:", 1)[1]
            marker = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
            self.assertEqual(body["cache_salt"], marker["label"])
            request_ids.append(marker["label"])
        self.assertEqual(len(request_ids), 2)
        self.assertNotEqual(*request_ids)
        self.assertEqual(result["request_id"], request_ids[-1])


if __name__ == "__main__":
    unittest.main()
