"""Locate the SGLang testbed directory (``sglang_direct_kv``).

Experiment code used to live in ``sglang_direct_kv/scripts`` and computed the
testbed root as ``Path(__file__).parents[1]``.  After the move into this
package it calls ``testbed_root()`` instead, which returns the same directory:

1. ``$AGENTIC_TESTBED_ROOT`` if set;
2. ``<repo>/sglang_direct_kv`` when this package is an editable install inside
   the repository (the normal case);
3. the first ``sglang_direct_kv``-shaped directory found from the current
   working directory upwards.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

_MARKER = Path("scripts") / "run_harness_deadline_pressure.sh"


def _is_testbed(path: Path) -> bool:
    return (path / _MARKER).is_file()


@lru_cache(maxsize=None)
def testbed_root() -> Path:
    env = os.environ.get("AGENTIC_TESTBED_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    in_repo = Path(__file__).resolve().parents[4] / "sglang_direct_kv"
    if _is_testbed(in_repo):
        return in_repo
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        for candidate in (base, base / "sglang_direct_kv"):
            if _is_testbed(candidate):
                return candidate
    raise FileNotFoundError("cannot locate the sglang_direct_kv testbed; set AGENTIC_TESTBED_ROOT")


def testbed_script(name: str) -> Path:
    """Path of a command-line script in the testbed (``scripts/<name>``)."""

    return testbed_root() / "scripts" / name
