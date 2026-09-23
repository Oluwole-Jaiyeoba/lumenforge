from __future__ import annotations

"""Controller-command adapters for the SGLang experiment path.

Moved from ``agentic_kv/controller/backend.py``.  Every result now carries an
explicit ``effect_level`` (see ``agentic_backend_api.EffectLevel``): the
historical ``acted=True`` of the ``Gateway*`` adapters means "accepted for
lowering into request fields", never "SGLang confirmed it".
"""

import os

from agentic_backend_api import BackendActionResult, BackendAdapter, BackendCapabilities, EffectLevel
from agentic_core import ControllerCommand, KVAction, SchedulerAction


class ObserveOnlyBackendAdapter:
    """Portable adapter that records intent without mutating SGLang."""

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            observe_only=True,
            live_metrics=True,
            backend_name="observe_only",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="observe-only adapter recorded command intent",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.RECORDED_ONLY,
        )


class GatewayPriorityBackendAdapter:
    """Adapter for scheduler-only mode.

    The actual SGLang mutation happens at the request boundary, where the
    gateway lowers this accepted controller decision into the request priority
    field. Keeping this adapter side-effect-free makes the controller portable.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            observe_only=False,
            backend_name="controller_scheduler_priority",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller priority accepted for gateway lowering",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in scheduler-only mode",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.RECORDED_ONLY,
        )


class GatewayDemoteRestoreBackendAdapter:
    """Adapter for portable demote/restore experiments.

    Demotion and restoration are lowered at the gateway request boundary. This
    adapter records that the controller command is accepted, without depending
    on a private SGLang API for in-place queue manipulation.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            kv_demote=True,
            kv_release=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_demote_restore",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action is KVAction.DEMOTE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller demote accepted for gateway lowering of background traffic",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller replay priority accepted during demote/restore window",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.kv_action is KVAction.RELEASE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller restore/release accepted after replay",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in demote/restore mode",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.RECORDED_ONLY,
        )


class GatewaySpeculativePreloadBackendAdapter:
    """Adapter for controller-driven gateway speculative KV preload.

    The controller remains backend-neutral: it emits a KV prefetch command, and
    the experiment gateway lowers that accepted command into the existing
    background warmup request path.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            kv_prefetch=True,
            observe_only=False,
            backend_name="controller_speculative_preload",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action.value == "prefetch":
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller KV prefetch accepted for gateway speculative preload",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in preload-only mode",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.RECORDED_ONLY,
        )


class GatewayAdmissionControlBackendAdapter:
    """Adapter for controller-driven speculative work admission.

    The adapter accepts priority, prefetch, and background-budget commands. The
    actual admit/skip choice is made by the portable experiment driver using
    pressure knobs and the accepted command envelope.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            background_prefill_budget=True,
            kv_prefetch=True,
            kv_release=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_admission_control",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action is KVAction.PREFETCH:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller prefetch accepted for admission-gated warmup",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.scheduler_action is SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller background prefill budget accepted for admission gating",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller replay priority accepted for admission-control mode",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.kv_action is KVAction.RELEASE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="controller admission-control release recorded after replay",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="controller command recorded but not active in admission-control mode",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.RECORDED_ONLY,
        )


class GatewayFullControllerBackendAdapter:
    """Adapter for the first combined portable controller policy.

    Full controller v1 deliberately combines only the controller pieces that
    have shown value on EC2: replay priority, background demotion/restoration,
    and explicit admission/budget decisions. It does not expose KV prefetch in
    its default capabilities, so speculative preload cannot sneak into the full
    mode unless a later policy version opts into it.
    """

    def __init__(self, capabilities: BackendCapabilities | None = None) -> None:
        self._capabilities = capabilities or BackendCapabilities(
            priority_queue=True,
            background_prefill_budget=True,
            kv_demote=True,
            kv_prefetch=False,
            kv_release=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_full",
            backend_version="v1:no_speculative_preload",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action is KVAction.DEMOTE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="full controller demote accepted for gateway lowering of background traffic",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.scheduler_action is SchedulerAction.SET_PRIORITY and command.priority is not None:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="full controller replay priority accepted for gateway lowering",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.scheduler_action is SchedulerAction.SET_BACKGROUND_PREFILL_BUDGET:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="full controller background prefill budget accepted for admission gating",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        if command.kv_action is KVAction.RELEASE:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=True,
                reason="full controller restore/release accepted after replay",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.LOWERED_AT_REQUEST_BOUNDARY,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=False,
            reason="full controller recorded command but did not enable speculative preload in v1",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.RECORDED_ONLY,
        )


class SGLangTargetedKVPrefetchBackendAdapter:
    """Adapter for a future direct SGLang host-to-device KV prefetch hook.

    This adapter intentionally separates controller intent from SGLang internals.
    If a stable direct hook is unavailable, the command is accepted and recorded
    but not reported as acted. That keeps Phase 4 portable across SGLang
    versions while still producing honest proof rows.
    """

    def __init__(
        self,
        capabilities: BackendCapabilities | None = None,
        *,
        direct_hook_available: bool | None = None,
    ) -> None:
        if direct_hook_available is None:
            direct_hook_available = os.environ.get("AGENTIC_KV_TARGETED_PREFETCH_HOOK", "").lower() in {
                "1",
                "true",
                "yes",
            }
        self.direct_hook_available = direct_hook_available
        self._capabilities = capabilities or BackendCapabilities(
            kv_prefetch=True,
            live_metrics=True,
            observe_only=False,
            backend_name="controller_targeted_kv_prefetch",
            backend_version="direct_hook_available=1" if direct_hook_available else "direct_hook_available=0",
        )
        self.commands: list[ControllerCommand] = []

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def apply(self, command: ControllerCommand) -> BackendActionResult:
        self.commands.append(command)
        if command.kv_action.value != "prefetch":
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=False,
                reason="controller command recorded but not active in targeted-prefetch mode",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.RECORDED_ONLY,
            )
        if not self.direct_hook_available:
            return BackendActionResult(
                command_id=command.command_id,
                accepted=True,
                acted=False,
                reason="targeted SGLang KV prefetch hook unavailable in this SGLang version",
                backend_name=self._capabilities.backend_name,
                effect_level=EffectLevel.UNSUPPORTED,
            )
        return BackendActionResult(
            command_id=command.command_id,
            accepted=True,
            acted=True,
            reason="targeted SGLang KV prefetch hook accepted",
            backend_name=self._capabilities.backend_name,
            effect_level=EffectLevel.DISPATCHED_TO_BACKEND,
        )
