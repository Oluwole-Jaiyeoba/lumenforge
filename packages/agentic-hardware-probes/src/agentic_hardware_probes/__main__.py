"""Compare two collected, matched probe-run JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .comparison import compare_runs, load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("interference", type=Path)
    parser.add_argument("--metric", required=True, help="Metric key in samples[*].metrics_ms")
    args = parser.parse_args()
    summary = compare_runs(load_run(args.control), load_run(args.interference), args.metric)
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
