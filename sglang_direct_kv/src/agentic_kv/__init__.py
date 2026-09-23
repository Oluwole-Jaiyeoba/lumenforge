"""Compatibility layer for the SGLang testbed (``sglang_direct_kv``).

Since the 2026-09-23 restructuring this package contains no real code: every
module is an alias of (or re-exports) its new home in ``packages/``
(``agentic_backends.sglang``, ``agentic_controller``, ``agentic_harnesses``,
``agentic_experiments``, ``agentic_reports`` ...). See
docs/architecture/README_RESTRUCTURING.md.

Importing it also puts ``<repo>/packages/*/src`` on ``sys.path`` so historical
commands work without ``pip install``; normal installations should use
``scripts/install_workspace.sh``.
"""

from pathlib import Path
import sys


_repo_root = Path(__file__).resolve().parents[3]
_package_root = _repo_root / "packages"
if _package_root.is_dir():
    for _source_dir in sorted(_package_root.glob("*/src")):
        _source_text = str(_source_dir)
        if _source_text not in sys.path:
            sys.path.insert(0, _source_text)
