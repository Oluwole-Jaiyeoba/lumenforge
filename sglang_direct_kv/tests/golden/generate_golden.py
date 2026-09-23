"""Record golden fixtures from the *current* implementation.

Run from ``sglang_direct_kv``::

    python tests/golden/generate_golden.py

Only regenerate when a behavior change is intentional, and document why in
``docs/architecture/README_RESTRUCTURING.md``.  The fixtures committed with the refactor were
recorded from commit 491aea5 (pre-refactor checkpoint).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "scripts"), str(ROOT / "src")]

from golden_eval import evaluate_driver, evaluate_gateway  # noqa: E402


def main() -> None:
    import harness_sglang_gateway as gateway
    import run_multi_harness_replay_driver as driver

    (HERE / "gateway.golden.json").write_text(
        json.dumps(evaluate_gateway(gateway), sort_keys=True, indent=1, default=repr) + "\n", encoding="utf-8"
    )
    (HERE / "driver.golden.json").write_text(
        json.dumps(evaluate_driver(driver), sort_keys=True, indent=1, default=repr) + "\n", encoding="utf-8"
    )
    print("wrote", HERE / "gateway.golden.json", HERE / "driver.golden.json")


if __name__ == "__main__":
    main()
