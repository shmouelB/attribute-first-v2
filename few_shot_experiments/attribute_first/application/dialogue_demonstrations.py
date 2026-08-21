"""Just-in-time demonstration history for dialogue stages."""

from copy import deepcopy
from dataclasses import dataclass

from ..runtime.conversation import Conversation


@dataclass(frozen=True, slots=True)
class DialogueDemonstrationLease:
    """Exact removable range occupied by one stage's temporary examples."""

    stage_name: str
    start_index: int
    message_count: int
    history_sha256: str


class DialogueDemonstrationService:
    """Validate and append one stage's demonstrations to a live session."""

    _STATE_FIELDS = {
        "ambiguity_highlight": "ambiguity_highlight_demos",
        "fusion_in_context": "fusion_demos",
    }
    _TRACE_PREFIXES = {
        "ambiguity_highlight": "ah",
        "fusion_in_context": "fic",
    }

    def __init__(self, dependencies):
        self._dependencies = dependencies

    def inject(
        self,
        *,
        state,
        instance,
        stage_name,
        demos,
        role_messages,
        demo_count,
    ):
        """Append a stable demo history and record its exact digest."""

        try:
            target = getattr(state, self._STATE_FIELDS[stage_name])
            prefix = self._TRACE_PREFIXES[stage_name]
        except KeyError as exc:
            raise ValueError(
                f"unsupported dialogue demonstration stage: {stage_name}"
            ) from exc
        history = self._dependencies.dialogue_demo_histories(
            role_messages,
            [instance.uid],
            stage_name,
            demo_count,
        )[instance.uid]
        if not target:
            target.extend(deepcopy(demos))
        elif (
            self._dependencies.stable_value_sha256(target)
            != self._dependencies.stable_value_sha256(demos)
        ):
            raise ValueError(
                f"{stage_name}: demonstration set drifted "
                f"for {instance.uid!r}"
            )
        instance.protocol.update(
            {
                f"{prefix}_demo_history": deepcopy(history),
                f"{prefix}_demo_history_sha256": (
                    self._dependencies.stable_value_sha256(history)
                ),
            }
        )
        conversation = Conversation.wrap(instance.session)
        start_index = len(conversation.history)
        self._dependencies.append_dialogue_history(
            conversation.provider_session,
            history,
        )
        return DialogueDemonstrationLease(
            stage_name=stage_name,
            start_index=start_index,
            message_count=len(history),
            history_sha256=self._dependencies.stable_value_sha256(
                self._dependencies.jsonable_dialogue_value(history)
            ),
        )

    def remove(self, *, instance, lease):
        """Remove only a completed stage's temporary demonstrations."""

        if not isinstance(lease, DialogueDemonstrationLease):
            raise TypeError("lease must be a DialogueDemonstrationLease")
        conversation = Conversation.wrap(instance.session)
        history = conversation.history
        stop_index = lease.start_index + lease.message_count
        if stop_index > len(history):
            raise ValueError(
                f"{lease.stage_name}: demonstration lease exceeds history"
            )
        observed = self._dependencies.jsonable_dialogue_value(
            history[lease.start_index:stop_index]
        )
        if (
            self._dependencies.stable_value_sha256(observed)
            != lease.history_sha256
        ):
            raise ValueError(
                f"{lease.stage_name}: demonstration history drifted "
                "before pruning"
            )
        conversation.reset(
            history[:lease.start_index] + history[stop_index:]
        )
        prefix = self._TRACE_PREFIXES[lease.stage_name]
        instance.protocol.update(
            {
                f"{prefix}_previous_task_demos_removed": True,
                f"{prefix}_removed_demo_history_sha256": (
                    lease.history_sha256
                ),
            }
        )


__all__ = [
    "DialogueDemonstrationLease",
    "DialogueDemonstrationService",
]
