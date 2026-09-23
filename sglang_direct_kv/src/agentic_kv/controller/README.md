# Compatibility Controller Package

The portable controller implementation now lives in `agentic_controller`.
These modules preserve existing `agentic_kv.controller` imports while scripts
and experiments migrate incrementally.

Gateway/backend adapters remain here until the SGLang adapter extraction phase.
Harness signal normalization now lives in `agentic_harnesses`.
