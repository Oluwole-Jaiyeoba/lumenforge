"""Experiment orchestration for the agentic-hardware testbed.

Moved out of ``sglang_direct_kv/scripts`` (see
docs/architecture/README_RESTRUCTURING.md, "Testbed split").  Modules keep
their former script names so ``python scripts/<name>.py`` wrappers,
``--help`` text and documentation stay valid.

- ``runners``            experiment drivers (multi-harness replay driver, workloads, probes, smoke tests)
- ``gateway``            client-side processes: harness gateway, OpenAI proxy logger, NAT wrapper, live prefetch controller
- ``workloads``          workload generation/extraction and timing models
- ``environment``        run-environment capture (packages, GPU sampling)
- ``prompt_codec_eval``  prompt-codec evaluation tools
- ``basic_workload``     the original smoke workload library (formerly ``agentic_kv`` top level)
- ``real_runner``        real-backend scenario command builder
- ``paths``              ``testbed_root()``

Composition root: may import every portable package and the SGLang backend.
"""
