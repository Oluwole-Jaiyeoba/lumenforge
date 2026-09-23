"""Compatibility alias: this module is ``agentic_harness_scenarios.models`` (single copy since the refactor).

``agentic_kv.harness_scenarios`` used to hold a duplicate of the portable
scenario package; the duplicate was removed.  Only ``real_runner.py`` (the
testbed-specific launcher) still lives here.
"""

from agentic_kv._compat_alias import alias

alias(__name__, "agentic_harness_scenarios.models")
