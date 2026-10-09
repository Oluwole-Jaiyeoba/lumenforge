import json
from pathlib import Path

import pytest

from agentic_controller.eviction_value import value_aware_eviction_metadata
from agentic_experiments.runners.run_multi_harness_replay_driver import apply_equal_importance_contract
from agentic_experiments.runners.run_controller_audit_pivots import arm_environment, clean_environment


ROOT = Path(__file__).resolve().parents[2]


def test_reuse_time_retention_preserves_equal_importance(monkeypatch):
    monkeypatch.setenv("CONTROLLER_EVICTION_SIGNAL_POLICY", "reuse_time")
    meta = {"mode": "controller_value_aware_eviction", "session_id": "a"}
    short = value_aware_eviction_metadata(meta, next_wait_ms=100, future_replay=True)
    long = value_aware_eviction_metadata(meta, next_wait_ms=1000, future_replay=True)
    assert short["controller_sglang_priority"] > long["controller_sglang_priority"]
    assert apply_equal_importance_contract(dict(short), enabled=True)["work_class"] == "peer"
    assert value_aware_eviction_metadata(meta, future_replay=False)["controller_sglang_priority"] == 0
    assert value_aware_eviction_metadata({**meta, "mode": "no_prefetch"}) == {}
    assert short == value_aware_eviction_metadata({**meta, "session_id": "b"}, next_wait_ms=100, future_replay=True)


@pytest.mark.parametrize("wait", [None, -1, float("nan"), float("inf")])
def test_reuse_time_rejects_unknown_or_invalid_wait(monkeypatch, wait):
    monkeypatch.setenv("CONTROLLER_EVICTION_SIGNAL_POLICY", "reuse_time")
    with pytest.raises(ValueError):
        value_aware_eviction_metadata({"mode": "controller_value_aware_eviction"}, next_wait_ms=wait, future_replay=True)


def test_spec_isolation_and_identical_pair_configuration():
    spec = json.loads((ROOT / "configs/experiment_specs/controller_audit_pivots.json").read_text())
    for scenario in ("1", "2", "3"):
        baseline = arm_environment(ROOT, spec, scenario, 1, "no_prefetch", "test", Path("/models"), "pinned")
        treatment = arm_environment(ROOT, spec, scenario, 1, spec["scenarios"][scenario]["mode"], "test", Path("/models"), "pinned")
        assert {k for k in baseline if baseline[k] != treatment[k]} == {"MODES", "REPORT_LABEL"}
        assert "--disable-cuda-graph" not in baseline["EXTRA_SERVER_ARGS"]
        assert "--disable-overlap-schedule" not in baseline["EXTRA_SERVER_ARGS"]
        assert baseline["AGENTIC_EQUAL_IMPORTANCE_WORKLOAD"] == "1"
        if scenario != "2":
            assert "AGENTIC_KV_PREPARE_CONTROL_ENABLE" not in baseline


def test_old_flags_and_secrets_not_inherited(monkeypatch):
    monkeypatch.setenv("CONTROLLER_DIRECT_LOAD_MECHANISM", "bad")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    assert "CONTROLLER_DIRECT_LOAD_MECHANISM" not in clean_environment()
    assert "ANTHROPIC_API_KEY" not in clean_environment()
