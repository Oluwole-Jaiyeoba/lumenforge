import unittest

from agentic_experiments.runners.analyze_work_audit_tool_cycles import analyze


def event(name, ts_ns, **fields):
    return {"event": name, "ts_ns": ts_ns, **fields}


class ToolCycleAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "active_count": 1,
            "turn_count": 1,
            "turns": [{
                "request_id": "replay-1", "kind": "active", "turn": 1,
                "tool_return_ns": 1_000_000,
                "request_start_ns": 2_000_000,
                "first_token_ns": 12_000_000,
                "prompt_tokens": 120,
                "ttft_ms": 10.0,
                "first_token_after_tool_ms": 11.0,
            }],
        }
        self.trace = [
            event("kv_telemetry.request_stage", 3_000_000, request_id="replay-1",
                  stage="cache_match_prefix", phase="start"),
            event("hiradix.match_prefix.end", 4_000_000,
                  kv_context={"request": {"agent_request_id": "replay-1"}},
                  result=[{"index_count": 80}]),
            event("kv_telemetry.request_stage", 5_000_000, request_id="replay-1",
                  stage="cache_match_prefix", phase="end"),
            event("scheduler.run_batch.start", 9_000_000,
                  kv_context={"batch": {"requests": [{"agent_request_id": "replay-1"}]}}),
            event("scheduler.run_batch.end", 10_000_000,
                  kv_context={"batch": {"requests": [{"agent_request_id": "replay-1"}]}}),
        ]

    def test_no_load_and_first_stage_wins(self):
        self.trace.append(event("kv_telemetry.request_stage", 15_000_000,
                                request_id="replay-1", stage="cache_match_prefix", phase="start"))
        result = analyze(self.summary, self.trace)
        row = result["turns"][0]
        self.assertEqual(row["lookup_start_ns"], 3_000_000)
        self.assertEqual(row["lookup_to_batch_ms"], 4.0)
        self.assertEqual(row["uncached_prompt_tokens"], 40)
        self.assertEqual(row["kv_load_back_count"], 0)
        self.assertEqual(result["measurements"]["active_replays_with_kv_load_back"], 0)

    def test_load_call_does_not_absorb_other_gap(self):
        self.trace[3:3] = [
            event("hiradix.load_back.start", 6_000_000,
                  kv_context={"agent_request_id": "replay-1"}),
            event("hiradix.load_back.end", 7_000_000, duration_ms=1.0,
                  kv_context={"agent_request_id": "replay-1"}),
        ]
        result = analyze(self.summary, self.trace)
        row = result["turns"][0]
        self.assertEqual(row["lookup_to_load_ms"], 1.0)
        self.assertEqual(row["kv_load_back_call_ms"], 1.0)
        self.assertEqual(row["load_end_to_batch_ms"], 2.0)
        self.assertEqual(result["measurements"]["kv_load_back_operations"], 1)

    def test_missing_batch_evidence_fails(self):
        with self.assertRaisesRegex(ValueError, "missing stage evidence"):
            analyze(self.summary, self.trace[:-2])

    def test_out_of_order_evidence_fails(self):
        self.trace[3]["ts_ns"] = 4_000_000
        with self.assertRaisesRegex(ValueError, "out of order"):
            analyze(self.summary, self.trace)


if __name__ == "__main__":
    unittest.main()
