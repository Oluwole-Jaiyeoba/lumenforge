"""Compatibility package for ``agentic_harness_scenarios.policies``.

Not a ``sys.modules`` alias on purpose: an aliased *package* would make
``agentic_kv.harness_scenarios.policies.<sub>`` load a second copy of each
submodule.  Each submodule file here aliases its counterpart instead.
"""

from agentic_harness_scenarios.policies import *  # noqa: F401,F403
