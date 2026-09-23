"""Compatibility alias: this module moved to ``agentic_experiments.basic_workload.agent_trace``.

Kept so historical imports keep working. New code should import the new path.
"""

from agentic_kv._compat_alias import alias

alias(__name__, "agentic_experiments.basic_workload.agent_trace")
