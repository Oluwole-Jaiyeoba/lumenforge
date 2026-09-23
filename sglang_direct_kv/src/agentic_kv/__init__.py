"""Agentic KV direct instrumentation testbed.

Portable packages live at the repository's top-level ``packages`` directory.
The path fallback keeps source-tree compatibility for historical commands;
normal installations should use ``scripts/install_workspace.sh``.
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
