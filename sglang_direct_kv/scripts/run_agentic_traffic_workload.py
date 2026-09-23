#!/usr/bin/env python
"""Moved to ``agentic_experiments.runners.run_agentic_traffic_workload``.

This file keeps ``python scripts/run_agentic_traffic_workload.py ...`` and ``import run_agentic_traffic_workload`` working
(see docs/architecture/README_RESTRUCTURING.md, "Testbed split").
"""

from _moved import run_or_alias

run_or_alias(__name__, "agentic_experiments.runners.run_agentic_traffic_workload")
