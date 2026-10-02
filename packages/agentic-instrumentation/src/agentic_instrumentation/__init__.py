"""Shared evidence contracts; backend-specific hooks belong in adapters."""

from .catalog import SIGNALS, PROFILES, validate_events, validate_profile
from .events import EvidenceEvent
from .proofs import assess_loaded_match

__all__ = ["SIGNALS", "PROFILES", "validate_events", "validate_profile", "EvidenceEvent", "assess_loaded_match"]
