"""Adapter for SGLang 0.5.13 - 0.5.15.post1 (static verification only).

Change vs v0511 (found by the static surface check):
- ``Scheduler.process_batch_result_prefill`` / ``process_batch_result_decode``
  moved to ``managers/scheduler_components/batch_result_processor.py``
  (class ``SchedulerBatchResultProcessor``, composed into the scheduler, not a
  mixin).  The hooks follow them; event names are unchanged so reports keep
  working.  NOTE: inside those hooks ``self`` is now the processor, not the
  Scheduler, so scheduler-queue summaries attached to those two events may be
  thinner.  Verify on a GPU before trusting scheduler-stage timing.
- ``Req.fill_ids`` removed (optional trace field).
"""

from __future__ import annotations

from .base import AdapterSpec, SGLangHookTarget, replace_hook_targets
from .v0510 import OPTIONAL_HOOKS, RAW_EVENT_MAP, REQUEST_FIELDS, SURFACE
from .v0511 import HOOK_TARGETS as _V0511_HOOKS

BATCH_RESULT_PROCESSOR = SGLangHookTarget(
    module="sglang.srt.managers.scheduler_components.batch_result_processor",
    class_name="SchedulerBatchResultProcessor",
    methods={
        "process_batch_result_prefill": "scheduler.process_batch_result_prefill",
        "process_batch_result_decode": "scheduler.process_batch_result_decode",
    },
    scheduler_required=True,
)

HOOK_TARGETS = replace_hook_targets(
    _V0511_HOOKS,
    drop_methods={"Scheduler": ("process_batch_result_prefill", "process_batch_result_decode")},
    add=(BATCH_RESULT_PROCESSOR,),
)

ADAPTER = AdapterSpec(
    name="v0513",
    tested_versions=(),
    version_range=("0.5.13", "0.5.16"),
    verification="static",
    hook_targets=HOOK_TARGETS,
    raw_event_map=RAW_EVENT_MAP,
    surface=SURFACE,
    optional_hooks=OPTIONAL_HOOKS,
    request_fields=REQUEST_FIELDS,
    notes=("Static surface check passes for 0.5.13 - 0.5.15.post1; never run on a GPU.",),
)
