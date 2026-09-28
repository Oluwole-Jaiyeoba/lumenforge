from __future__ import annotations

import unittest

from agentic_experiments.runners.run_multi_harness_replay_driver import apply_equal_importance_contract
from agentic_gateway.translation import resolve_backend_priority
from agentic_harnesses.signals import build_harness_controller_signal
from agentic_reports.builders.build_multi_harness_deadline_summary import (
    CHART_SIGNAL_ORDER,
    chart_signal_bucket,
    collect_equal_replay_summary,
    target_replay_rows,
)


class EqualImportanceScenarioTest(unittest.TestCase):
    def test_rejects_frontend_importance_for_every_request_role(self) -> None:
        for phase, field, value in (
            ("replay", "priority_intent", {"class": "urgent"}),
            ("pressure_filler", "high_priority", 100),
            ("pressure_filler_initial", "priority_label", "low"),
        ):
            with self.subTest(phase=phase, field=field):
                with self.assertRaisesRegex(ValueError, "forbids semantic priority"):
                    apply_equal_importance_contract(
                        {"mode": "controller_ready_time_gpu_backfill", "phase": phase, field: value},
                        enabled=True,
                    )

    def test_both_replay_roles_have_normal_urgency_and_due_time_ranks(self) -> None:
        priorities = []
        for phase, due, rank in (("replay", 1000, 99900), ("pressure_filler", 2000, 99800)):
            meta = apply_equal_importance_contract({
                "mode": "controller_ready_time_gpu_backfill",
                "phase": phase,
                "session_id": phase,
                "controller_predictive_deadline_queue": True,
                "controller_sglang_priority": rank,
                "deadline_offset_ms": due,
                "priority_label": "equal",
            }, enabled=True)
            signal = build_harness_controller_signal(meta)
            self.assertEqual(signal["phase"]["work_class"], "peer")
            self.assertEqual(signal["scheduling"]["urgency"], "normal")
            self.assertEqual(resolve_backend_priority(meta), rank)
            priorities.append(rank)
        self.assertGreater(priorities[0], priorities[1])

    def test_report_counts_all_replays_and_excludes_sentinel(self) -> None:
        rows = [
            {"harness": "hatcher", "pressure_level": "p3_high", "mode": "no_prefetch", "phase": phase,
             "request_group": group, "has_replay_deadline": "yes", "ttft_ms": ttft,
             "first_token_lateness_ms": lateness}
            for phase, group, ttft, lateness in (
                ("replay", "target", 100, 20),
                ("pressure_filler", "filler", 200, -10),
            )
        ]
        rows.append({"harness": "instrumentation", "phase": "replay", "has_replay_deadline": "yes"})
        self.assertEqual(len(target_replay_rows(rows, equal_importance=True)), 2)
        self.assertEqual(len(target_replay_rows(rows)), 1)
        summary = collect_equal_replay_summary(rows)[0]
        self.assertEqual(summary["mode"], "no_prefetch")
        self.assertEqual(summary["replays"], 2)
        self.assertEqual(summary["total_ttft_ms"], 300)
        self.assertEqual(summary["total_deadline_debt_ms"], 20)

    def test_ready_time_mode_is_visible_in_pressure_chart(self) -> None:
        bucket = chart_signal_bucket({"mode": "controller_ready_time_gpu_backfill"})
        self.assertEqual(bucket, "controller_ready_time_gpu_backfill")
        self.assertIn(bucket, CHART_SIGNAL_ORDER)


if __name__ == "__main__":
    unittest.main()
