"""Pair identical workload samples without asserting a physical bottleneck."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class Sample:
    sample_id: str
    metrics_ms: Mapping[str, float]


@dataclass(frozen=True)
class ProbeRun:
    run_id: str
    condition: str
    hardware_profile: str
    backend_version: str
    model: str
    workload_id: str
    seed: int
    instrumentation_profile: str
    samples: tuple[Sample, ...]

    def contract(self) -> tuple[str, str, str, str, int, str]:
        return (
            self.hardware_profile,
            self.backend_version,
            self.model,
            self.workload_id,
            self.seed,
            self.instrumentation_profile,
        )


@dataclass(frozen=True)
class Comparison:
    metric: str
    sample_count: int
    control_median_ms: float
    interference_median_ms: float
    median_paired_change_ms: float
    mean_paired_change_ms: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "metric": self.metric,
            "sample_count": self.sample_count,
            "control_median_ms": self.control_median_ms,
            "interference_median_ms": self.interference_median_ms,
            "median_paired_change_ms": self.median_paired_change_ms,
            "mean_paired_change_ms": self.mean_paired_change_ms,
        }


def _samples_by_id(run: ProbeRun) -> dict[str, Sample]:
    samples = {sample.sample_id: sample for sample in run.samples}
    if len(samples) != len(run.samples) or not samples or "" in samples:
        raise ValueError(f"{run.run_id}: sample IDs must be nonempty and unique")
    return samples


def compare_runs(control: ProbeRun, interference: ProbeRun, metric: str) -> Comparison:
    """Positive paired change means the interference condition took longer."""
    if control.condition != "control" or interference.condition != "interference":
        raise ValueError("expected control and interference conditions")
    if control.run_id == interference.run_id:
        raise ValueError("control and interference must be distinct runs")
    if control.contract() != interference.contract():
        raise ValueError("runs differ in hardware, backend, workload, seed, or instrumentation")
    control_samples = _samples_by_id(control)
    interference_samples = _samples_by_id(interference)
    if control_samples.keys() != interference_samples.keys():
        raise ValueError("runs must contain the same sample IDs")

    control_values: list[float] = []
    interference_values: list[float] = []
    changes: list[float] = []
    for sample_id in sorted(control_samples):
        pair = (control_samples[sample_id], interference_samples[sample_id])
        values: list[float] = []
        for sample in pair:
            if metric not in sample.metrics_ms:
                raise ValueError(f"{sample_id}: missing {metric}")
            value = float(sample.metrics_ms[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{sample_id}: {metric} must be finite and nonnegative")
            values.append(value)
        control_values.append(values[0])
        interference_values.append(values[1])
        changes.append(values[1] - values[0])

    return Comparison(
        metric=metric,
        sample_count=len(changes),
        control_median_ms=statistics.median(control_values),
        interference_median_ms=statistics.median(interference_values),
        median_paired_change_ms=statistics.median(changes),
        mean_paired_change_ms=statistics.mean(changes),
    )


def load_run(path: Path) -> ProbeRun:
    data = json.loads(path.read_text(encoding="utf-8"))
    return ProbeRun(
        run_id=str(data["run_id"]),
        condition=str(data["condition"]),
        hardware_profile=str(data["hardware_profile"]),
        backend_version=str(data["backend_version"]),
        model=str(data["model"]),
        workload_id=str(data["workload_id"]),
        seed=int(data["seed"]),
        instrumentation_profile=str(data["instrumentation_profile"]),
        samples=tuple(
            Sample(sample_id=str(row["sample_id"]), metrics_ms=row["metrics_ms"])
            for row in data["samples"]
        ),
    )
