from __future__ import annotations

import unittest

from agentic_backends.sglang.runtime_contract import (
    build_runtime_info,
    missing_capabilities,
    normalize_capabilities,
)


def probe(*, priority: bool = True, complete_hooks: bool = True) -> dict:
    missing = [] if complete_hooks else ["run_batch"]
    return {
        "sglang_version": "0.5.10.post1",
        "selected_adapter": "v0510",
        "launch_capabilities": {
            "enable_priority_scheduling_flag_supported": priority,
            "priority_schedule_policy_supported": False,
        },
        "hook_probe": [
            {
                "module": "sglang.srt.managers.cache_controller",
                "class": "HiCacheController",
                "class_found": True,
                "missing_methods": missing,
            },
            {
                "module": "sglang.srt.managers.scheduler",
                "class": "Scheduler",
                "class_found": True,
                "missing_methods": missing,
            },
        ],
    }


class RuntimeContractTest(unittest.TestCase):
    def test_normalizes_backend_specific_probe(self) -> None:
        capabilities = normalize_capabilities(probe())
        self.assertTrue(capabilities.priority_queue)
        self.assertTrue(capabilities.kv_prefetch)
        self.assertFalse(capabilities.background_prefill_budget)
        self.assertFalse(capabilities.observe_only)

    def test_incomplete_hooks_are_not_claimed(self) -> None:
        capabilities = normalize_capabilities(probe(priority=False, complete_hooks=False))
        self.assertFalse(capabilities.kv_prefetch)
        self.assertFalse(capabilities.live_metrics)
        self.assertTrue(capabilities.observe_only)

    def test_required_capabilities_fail_closed(self) -> None:
        info = build_runtime_info(probe(priority=False), runtime_profile="test")
        self.assertTrue(info.probe_ok)
        self.assertEqual(info.health_status, "not_checked")
        self.assertEqual(missing_capabilities(info, ["priority_queue", "kv_prefetch"]), ["priority_queue"])


if __name__ == "__main__":
    unittest.main()
