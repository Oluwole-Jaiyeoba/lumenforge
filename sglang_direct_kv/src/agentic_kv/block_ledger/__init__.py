"""Compatibility package: the KV block ledger moved to ``agentic_reports.block_ledger``.

This package is deliberately NOT a ``sys.modules`` alias (an aliased package
would make ``agentic_kv.block_ledger.<sub>`` load a second copy of each
submodule).  It re-exports the public API; each submodule file here aliases
its ``agentic_reports.block_ledger.<sub>`` counterpart.
"""

from agentic_reports.block_ledger import *  # noqa: F401,F403
from agentic_reports.block_ledger import __all__  # noqa: F401
