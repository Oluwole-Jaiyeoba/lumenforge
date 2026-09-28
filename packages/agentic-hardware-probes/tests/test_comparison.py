from __future__ import annotations

from dataclasses import replace

import pytest

from agentic_hardware_probes import ProbeRun, Sample, compare_runs, load_run


def run(condition: str, values: tuple[float, ...]) -> ProbeRun:
    return ProbeRun(
        run_id=condition,
        condition=condition,
        hardware_profile="standard_gpu",
        backend_version="pinned_backend",
        model="test_model",
        workload_id="same_workload",
        seed=7,
        instrumentation_profile="lightweight",
        samples=tuple(
            Sample(sample_id=str(index), metrics_ms={"replay_ttft_ms": value})
            for index, value in enumerate(values)
        ),
    )


def test_paired_change_is_interference_minus_control() -> None:
    summary = compare_runs(run("control", (10, 20)), run("interference", (12, 25)), "replay_ttft_ms")
    assert summary.sample_count == 2
    assert summary.control_median_ms == 15
    assert summary.interference_median_ms == 18.5
    assert summary.median_paired_change_ms == 3.5


def test_rejects_different_workload_contract() -> None:
    other = replace(run("interference", (12,)), workload_id="different")
    with pytest.raises(ValueError, match="runs differ"):
        compare_runs(run("control", (10,)), other, "replay_ttft_ms")


def test_rejects_unmatched_samples() -> None:
    with pytest.raises(ValueError, match="same sample IDs"):
        compare_runs(run("control", (10,)), run("interference", (12, 20)), "replay_ttft_ms")


def test_rejects_profiler_overhead_comparison() -> None:
    profiled = replace(run("interference", (12,)), instrumentation_profile="profiler")
    with pytest.raises(ValueError, match="runs differ"):
        compare_runs(run("control", (10,)), profiled, "replay_ttft_ms")


def test_rejects_invalid_measurement() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_runs(run("control", (10,)), run("interference", (float("nan"),)), "replay_ttft_ms")


def test_loads_collected_run_format(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        '{"run_id":"r1","condition":"control","hardware_profile":"gpu",'
        '"backend_version":"v1","model":"model","workload_id":"workload",'
        '"seed":7,"instrumentation_profile":"lightweight",'
        '"samples":[{"sample_id":"one","metrics_ms":{"replay_ttft_ms":12}}]}',
        encoding="utf-8",
    )
    loaded = load_run(path)
    assert loaded.samples[0].metrics_ms["replay_ttft_ms"] == 12
