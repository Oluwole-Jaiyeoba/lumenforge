"""Report builders, summaries, audits and the KV block ledger.

Moved out of ``sglang_direct_kv/scripts`` and ``agentic_kv`` (see
docs/architecture/README_RESTRUCTURING.md, "Testbed split").  Modules keep
their former script names.

- ``builders``        HTML/CSV report builders (master report, deadline summary, deep dives, audits)
- ``summaries``       smaller summary tools
- ``audits``          evidence audits and validators
- ``analysis``        shared analysis code (replay-path classifier, torch-profile correlation)
- ``block_ledger``    KV block ledger (normalizes raw trace rows via the SGLang adapter's event map)
- ``evidence_audit``, ``evidence_schema``  report evidence rules

Reports read raw SGLang trace rows today; the backend-neutral path is
``agentic_backends.sglang.telemetry`` (migration listed as follow-up work).
"""
