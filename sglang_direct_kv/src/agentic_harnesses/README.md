# Agentic Harnesses

This package owns harness-native capture and normalization. It records what a
harness emitted, where the signal was attached, and how the observation was
produced. It does not choose scheduling or KV actions.

- `signals.py` normalizes request/session facts for controller consumption.
- `hint_benchmark/` loads hint manifests, builds observations, and validates
  evidence from real, configuration-driven, replayed, or simulated lanes.

The package may depend on `agentic_core`. It must not import
`agentic_controller`, SGLang, or a backend implementation.
