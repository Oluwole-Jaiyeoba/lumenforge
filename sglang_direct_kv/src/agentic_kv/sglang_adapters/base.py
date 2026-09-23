"""Compatibility alias: this module moved to ``agentic_backends.sglang.versions.base``.

Kept so historical imports, scripts and sitecustomize hooks keep working.
New code should import ``agentic_backends.sglang.versions.base`` directly.
"""

from agentic_kv._compat_alias import alias

alias(__name__, "agentic_backends.sglang.versions.base")
