"""Portable comparison contracts for controlled hardware measurements."""

from .comparison import Comparison, ProbeRun, Sample, compare_runs, load_run

__all__ = ["Comparison", "ProbeRun", "Sample", "compare_runs", "load_run"]
