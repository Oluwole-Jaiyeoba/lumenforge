"""List the shared signals and evidence profiles without a backend runtime."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .catalog import PROFILES, SIGNALS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("signals", "profiles"))
    args = parser.parse_args()
    items = SIGNALS if args.kind == "signals" else PROFILES
    print(json.dumps({name: asdict(item) for name, item in items.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
