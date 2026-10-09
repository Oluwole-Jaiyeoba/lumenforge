import argparse

from agentic_experiments.runners.run_work_audit_native_vs_capacity_safe import (
    fingerprint,
    workload_contract,
)
from agentic_experiments.runners.analyze_work_audit_native_vs_capacity_safe import (
    _measurement_boundary,
)


def _args(**overrides):
    values = {
        "seed": 1,
        "pattern": "burst",
        "model": "model",
        "sessions": 6,
        "turns": 10,
        "initial_tokens": 4096,
        "prime_tokens": 2,
        "tool_words": 16,
        "decode_tokens": 16,
        "wait_ms": 1000,
        "burst_window_ms": 75,
        "spread_window_ms": 1000,
        "workload_namespace": "paired-seed1-burst",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_workload_fingerprint_is_policy_independent():
    native = workload_contract(_args())
    capacity_safe = workload_contract(_args())
    assert fingerprint(native) == fingerprint(capacity_safe)


def test_workload_fingerprint_changes_with_actual_work():
    assert fingerprint(workload_contract(_args())) != fingerprint(
        workload_contract(_args(turns=11))
    )


def test_legacy_unequal_measurement_boundaries_are_detected():
    assert _measurement_boundary({
        "policy": "native_sglang",
        "initial_setup_ms": 7774.0,
    }) == "includes_initial_prefix_population"
    assert _measurement_boundary({
        "policy": "capacity_safe",
    }) == "after_initial_prefix_population"
