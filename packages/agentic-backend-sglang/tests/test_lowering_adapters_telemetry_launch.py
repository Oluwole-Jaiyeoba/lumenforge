from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_sglang import build_fake_sglang  # noqa: E402

from agentic_backend_api import (  # noqa: E402
    BackendAdapter,
    EffectLevel,
    LaunchSpec,
    RequestLowering,
    TelemetryNormalizer,
)
from agentic_core import BackendRequest, ControllerCommand, KVAction, SchedulerAction  # noqa: E402
from agentic_backends.sglang import adapters, launch  # noqa: E402
from agentic_backends.sglang.lowering import SGLangRequestLowering, lower_translation  # noqa: E402
from agentic_backends.sglang.telemetry import SGLangTelemetryNormalizer  # noqa: E402
from agentic_backends.sglang.versions import get_adapter  # noqa: E402


class LoweringTest(unittest.TestCase):
    def test_priority_and_cache_salt_are_top_level_fields(self) -> None:
        request = BackendRequest(
            prompt_text="hi",
            max_tokens=4,
            priority=7,
            cache_salt="tenant",
            request_context={"request_id": "r"},
            agent_hints={"priority": 7},
            request_metadata={"session_id": "s"},
            native_cache_bridge={"schema": "x"},
        )
        body = lower_translation(request, "m")
        self.assertEqual(body["priority"], 7)
        self.assertEqual(body["cache_salt"], "tenant")
        self.assertEqual(body["custom_params"]["agentic_kv"], {"session_id": "s"})
        self.assertIs(body["custom_params"]["native_cache_bridge"], body["custom_params"]["nvext"]["cache_control"])
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])

    def test_no_priority_means_no_priority_field(self) -> None:
        body = SGLangRequestLowering().lower(BackendRequest(prompt_text="x", max_tokens=1), model="m")
        self.assertNotIn("priority", body)
        self.assertNotIn("nvext", body)
        self.assertNotIn("cache_salt", body)
        self.assertIsInstance(SGLangRequestLowering(), RequestLowering)


class AdapterEffectLevelTest(unittest.TestCase):
    def command(self, **kwargs) -> ControllerCommand:
        return ControllerCommand(command_id="c", session_id="s", prefix_id="p", session_generation=0, **kwargs)

    def test_gateway_adapters_report_lowering_not_confirmation(self) -> None:
        adapter = adapters.GatewayPriorityBackendAdapter()
        self.assertIsInstance(adapter, BackendAdapter)
        result = adapter.apply(self.command(scheduler_action=SchedulerAction.SET_PRIORITY, priority=5))
        self.assertTrue(result.acted)  # unchanged legacy meaning
        self.assertEqual(result.effect_level, EffectLevel.LOWERED_AT_REQUEST_BOUNDARY)
        self.assertEqual(result.to_dict()["effect_level"], "lowered_at_request_boundary")
        idle = adapter.apply(self.command())
        self.assertEqual(idle.effect_level, EffectLevel.RECORDED_ONLY)

    def test_targeted_prefetch_without_hook_is_unsupported(self) -> None:
        adapter = adapters.SGLangTargetedKVPrefetchBackendAdapter(direct_hook_available=False)
        result = adapter.apply(self.command(kv_action=KVAction.PREFETCH))
        self.assertFalse(result.acted)
        self.assertEqual(result.effect_level, EffectLevel.UNSUPPORTED)
        with_hook = adapters.SGLangTargetedKVPrefetchBackendAdapter(direct_hook_available=True)
        self.assertEqual(with_hook.apply(self.command(kv_action=KVAction.PREFETCH)).effect_level, EffectLevel.DISPATCHED_TO_BACKEND)


class TelemetryTest(unittest.TestCase):
    def test_raw_event_maps_to_stable_kind(self) -> None:
        normalizer = SGLangTelemetryNormalizer(get_adapter("v0510"), "0.5.10.post1")
        self.assertIsInstance(normalizer, TelemetryNormalizer)
        obs = normalizer.normalize(
            {
                "event": "hiradix.load_back.end",
                "ts_ns": 5_000_000,
                "duration_ms": 1.5,
                "kv_context": {"agent_session_id": "s1", "agent_request_id": "r1", "node_id": 9},
            }
        )
        self.assertEqual((obs.kind, obs.session_id, obs.request_id), ("KV_LOAD_GPU", "s1", "r1"))
        self.assertEqual(obs.observed_at_ms, 5)
        self.assertEqual(obs.payload["node_id"], 9)
        self.assertIsNone(normalizer.normalize({"event": "something.else"}))
        meta = normalizer.normalize({"event": "trace.install.summary", "missing_required_hooks": []})
        self.assertEqual(meta.kind, "backend_hook_install")


class LaunchTest(unittest.TestCase):
    def test_launch_spec_builds_argv(self) -> None:
        spec = launch.launch_spec(
            model="Qwen/x",
            port=31000,
            options={"enable_priority_scheduling": True, "schedule-policy": "lpm", "disable_radix_cache": False},
            sglang_version="0.5.10.post1",
        )
        self.assertIsInstance(spec, LaunchSpec)
        self.assertEqual(spec.argv[:3], ("python", "-m", "sglang.launch_server"))
        self.assertIn("--enable-priority-scheduling", spec.argv)
        self.assertEqual(spec.argv[spec.argv.index("--schedule-policy") + 1], "lpm")
        self.assertNotIn("--disable-radix-cache", spec.argv)
        self.assertIn("adapter=v0510", spec.notes)

    def test_preflight_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = build_fake_sglang(Path(tmp), get_adapter("v0510"), "0.5.10", drop={"--schedule-policy"})
            argv = ["--enable-priority-scheduling", "--schedule-policy", "lpm", "--hicache-size=4"]
            self.assertEqual(launch.unknown_flags(argv, str(root)), ["--schedule-policy"])
            with mock.patch("sys.stderr"):
                self.assertEqual(launch.preflight(argv, source_root=str(root)), 0)
                self.assertEqual(launch.preflight(argv, source_root=str(root), strict=True), 1)


if __name__ == "__main__":
    unittest.main()
