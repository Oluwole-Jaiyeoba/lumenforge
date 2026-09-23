"""Behavior-preservation tests for the SGLang-portability refactor.

The fixtures were recorded from the pre-refactor code (commit 491aea5).  The
refactor moved gateway translation and driver policy helpers out of scripts
into packages; these tests prove the moved code still produces byte-identical
results through BOTH the old script entry points and the new package APIs.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for extra in (str(HERE), str(ROOT / "scripts"), str(ROOT / "src")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from golden_eval import compare, evaluate_driver, evaluate_gateway  # noqa: E402


def _load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def _assert_same(test: unittest.TestCase, expected: dict, actual: dict) -> None:
    actual = json.loads(json.dumps(actual, sort_keys=True, default=repr))
    problems = compare(expected, actual)
    test.assertEqual(problems, [], "behavior drifted from the recorded golden output:\n" + "\n".join(problems[:20]))


class GatewayGoldenTest(unittest.TestCase):
    def test_gateway_script_entry_points_match_golden(self) -> None:
        import harness_sglang_gateway as gateway

        _assert_same(self, _load("gateway.golden.json"), evaluate_gateway(gateway))

    def test_package_translation_plus_sglang_lowering_match_golden(self) -> None:
        """Same matrix, but calling the new packages directly (no script)."""

        from agentic_backends.sglang import lowering
        from agentic_gateway import translation
        from agentic_harnesses import request_hints

        facade = SimpleNamespace(
            sglang_priority=translation.resolve_backend_priority,
            priority_translation_context=translation.priority_translation_context,
            cache_translation_context=translation.cache_translation_context,
            emitted_priority_signal=request_hints.emitted_priority_signal,
            emitted_cache_signal=request_hints.emitted_cache_signal,
            emitted_signal_is_urgent=request_hints.emitted_signal_is_urgent,
            metadata_context=translation.metadata_context,
            build_sglang_payload=lambda payload, meta, api_kind, model: lowering.lower_translation(
                translation.translate_request(payload, meta, api_kind), model
            ),
        )
        _assert_same(self, _load("gateway.golden.json"), evaluate_gateway(facade))


class DriverGoldenTest(unittest.TestCase):
    def test_driver_script_helpers_match_golden(self) -> None:
        import run_multi_harness_replay_driver as driver

        _assert_same(self, _load("driver.golden.json"), evaluate_driver(driver))

    def test_package_helpers_match_golden(self) -> None:
        from agentic_controller import eviction_value, lead_times
        from agentic_harnesses import emission_emulation

        facade = SimpleNamespace(
            **{name: getattr(lead_times, name) for name in lead_times.__all__},
            value_aware_eviction_metadata=eviction_value.value_aware_eviction_metadata,
            outbound_priority_fields=emission_emulation.outbound_priority_fields,
        )
        _assert_same(self, _load("driver.golden.json"), evaluate_driver(facade))


if __name__ == "__main__":
    unittest.main()
