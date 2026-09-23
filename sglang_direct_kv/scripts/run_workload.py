#!/usr/bin/env python
"""Moved to ``agentic_experiments.runners.run_workload``.

This file keeps ``python scripts/run_workload.py ...`` and ``import run_workload`` working
(see docs/architecture/README_RESTRUCTURING.md, "Testbed split").
"""

from _moved import run_or_alias

run_or_alias(__name__, "agentic_experiments.runners.run_workload")
