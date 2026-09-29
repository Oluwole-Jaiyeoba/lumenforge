from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_backends.sglang.trace import patch as trace_patch


class _Event:
    def __init__(self, ready: bool) -> None:
        self.ready = ready

    def query(self) -> bool:
        return self.ready

    def elapsed_time(self, _other: object) -> float:
        return 12.5


class PrepareLoadStatusTest(unittest.TestCase):
    def tearDown(self) -> None:
        with trace_patch._PREPARE_LOADS_LOCK:
            trace_patch._PREPARE_LOADS.clear()

    def test_native_finish_event_gates_load_completion(self) -> None:
        native_finish = _Event(False)
        with trace_patch._PREPARE_LOADS_LOCK:
            trace_patch._PREPARE_LOADS["load-1"] = {
                "load_id": "load-1",
                "start_event": _Event(True),
                "finish_event": _Event(True),
                "native_start_event": _Event(True),
                "native_finish_event": native_finish,
            }

        with patch.object(trace_patch, "_write_event"):
            active = trace_patch._prepare_load_status("load-1")
            native_finish.ready = True
            finished = trace_patch._prepare_load_status("load-1")

        self.assertEqual(active["status"], "active")
        self.assertTrue(active["timing_finished"])
        self.assertFalse(active["native_finished"])
        self.assertNotIn("cuda_elapsed_ms", active)
        self.assertEqual(finished["status"], "finished")
        self.assertTrue(finished["native_finished"])
        self.assertEqual(finished["cuda_elapsed_ms"], 12.5)


if __name__ == "__main__":
    unittest.main()
