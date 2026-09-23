import os
import unittest
from unittest import mock

from agentic_controller import lead_times


class LeadTimesTest(unittest.TestCase):
    def test_prepare_lead(self) -> None:
        self.assertEqual(lead_times.controller_prepare_lead_ms(100), 0)
        self.assertEqual(lead_times.controller_prepare_lead_ms(2_000), 500)
        self.assertEqual(lead_times.controller_prepare_lead_ms(8_000), 1_000)
        self.assertEqual(lead_times.controller_prepare_lead_ms(60_000), 1_500)

    def test_deadline_rank_orders_by_due_time(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            early = lead_times.deadline_fair_priority_for_due(100)
            late = lead_times.deadline_fair_priority_for_due(5_000)
        self.assertGreater(early, late)

    def test_direct_load_window_rejects_when_no_slack(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            window = lead_times.controller_direct_load_window(tool_start_ms=0, replay_due_ms=500, prompt_tokens=10)
        self.assertFalse(window["admitted"])


if __name__ == "__main__":
    unittest.main()
