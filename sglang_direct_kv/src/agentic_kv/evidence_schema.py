"""Compatibility alias: this module moved to ``agentic_reports.evidence_schema``.

Kept so historical imports keep working. New code should import the new path.
"""

from agentic_kv._compat_alias import alias

alias(__name__, "agentic_reports.evidence_schema")
