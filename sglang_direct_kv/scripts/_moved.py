"""Helper for script files whose code moved into ``packages/``.

Every ``scripts/<name>.py`` that became a thin wrapper calls
``run_or_alias(__name__, "<package.module>")``:

- run as a program (``python scripts/<name>.py ...``): the package module is
  executed as ``__main__`` with ``runpy`` -- same argv, same ``--help`` text
  (the module file keeps the script's name), same exit codes;
- imported (``import <name>`` with ``scripts/`` on ``sys.path``, as tests and
  a few scripts do): the old name becomes an alias of the package module, so
  attributes, monkeypatching and module state are shared.

It also makes the repository packages importable when they are not
pip-installed: ``sglang_direct_kv/src`` goes on ``sys.path`` and importing
``agentic_kv`` adds ``<repo>/packages/*/src``.
"""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import agentic_kv  # noqa: E402,F401  (sys.path fallback for <repo>/packages/*/src)


def run_or_alias(name: str, target: str) -> None:
    if name == "__main__":
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    else:
        sys.modules[name] = importlib.import_module(target)
