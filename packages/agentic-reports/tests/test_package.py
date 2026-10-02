import importlib
import pkgutil
import unittest

import agentic_reports
from agentic_reports.block_ledger import KVEventType, NormalizedKVEvent, block_ledger_rows, build_block_ledger, normalize_sglang_trace_events


class PackageTest(unittest.TestCase):
    def test_every_module_imports(self) -> None:
        failures = {}
        for info in pkgutil.walk_packages(agentic_reports.__path__, "agentic_reports."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # noqa: BLE001
                failures[info.name] = f"{type(exc).__name__}: {exc}"
        self.assertEqual(failures, {})

    def test_block_ledger_builds_from_normalized_events(self) -> None:
        event = NormalizedKVEvent(
            event_type=KVEventType.LOAD_GPU,
            session_id="s1",
            phase="replay",
            time_ms=1.0,
            duration_ms=0.5,
            token_start=0,
            token_end=16,
            token_count=16,
            node_id="n1",
        )
        rows = block_ledger_rows(build_block_ledger([event]))
        self.assertTrue(rows)

    def test_trace_normalizer_uses_recorded_adapter(self) -> None:
        row = {"event": "hicache.load.end", "ts_ns": 20,
               "kv_context": {"agent_session_id": "s1", "device_indices": {"values": [1]}}}
        result = normalize_sglang_trace_events([row], adapter_name="v0510")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].event_type, KVEventType.LOAD_GPU)
        with self.assertRaises(ValueError):
            normalize_sglang_trace_events([row], version="0.5.10.post1", adapter_name="v0510")


if __name__ == "__main__":
    unittest.main()
