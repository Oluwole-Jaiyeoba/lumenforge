from __future__ import annotations

import unittest

from agentic_backends.sglang.hook_registry import resolve_hook


class HookRegistryTest(unittest.TestCase):
    def test_contract_hook_resolves_through_adapter(self) -> None:
        resolved = resolve_hook("scheduler_batch_observation", "v0510")
        self.assertEqual(resolved["hook_id"], "scheduler_batch_observation")
        self.assertTrue(any(target["target"].endswith("Scheduler.process_batch_result") for target in resolved["targets"]))
        self.assertTrue(any(target["event_prefix"] == "scheduler.process_batch_result" for target in resolved["targets"]))

    def test_unknown_hook_is_loud(self) -> None:
        with self.assertRaises(KeyError):
            resolve_hook("not_a_real_hook", "v0510")


if __name__ == "__main__":
    unittest.main()
