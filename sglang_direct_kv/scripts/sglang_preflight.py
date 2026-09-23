#!/usr/bin/env python
"""Warn (or fail with AGENTIC_SGLANG_STRICT=1) when SGLang does not accept a server flag.

Thin wrapper so launch scripts can call the check with only ``src`` on
PYTHONPATH: importing ``agentic_kv`` puts ``<repo>/packages/*/src`` on sys.path.

    python scripts/sglang_preflight.py -- --enable-hierarchical-cache --hicache-size 14
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import agentic_kv  # noqa: E402,F401  (sys.path fallback for packages/)

from agentic_backends.sglang.launch import main  # noqa: E402

if __name__ == "__main__":
    args = ["preflight"]
    if os.environ.get("AGENTIC_SGLANG_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}:
        args.append("--strict")
    raise SystemExit(main(args + sys.argv[1:]))
