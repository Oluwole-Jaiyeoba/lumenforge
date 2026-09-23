"""Keep pre-refactor import paths working after code moved into ``packages/``.

A shim module calls ``alias(__name__, "new.module")``; the old dotted name then
refers to the *same module object* as the new one (private names and module
globals included), so monkeypatching or reading ``_PRIVATE`` state through the
old path keeps working.  See docs/architecture/README_RESTRUCTURING.md, "Compatibility shims".
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


def alias(old_name: str, new_name: str) -> ModuleType:
    module = importlib.import_module(new_name)
    sys.modules[old_name] = module
    return module
