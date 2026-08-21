"""Content-selection stage service for stateful dialogue runs."""

import logging
from copy import deepcopy


class DialogueContentSelectionService:
    """Execute CS, including cache fallback and terminal propagation."""

    def __init__(self, dependencies, session_service):
        self._dependencies = dependencies
        self._session_service = session_service

    def run(self, state, instance, turn, source_row):
        """Execute one content-selection turn and build its pipeline row."""

        plan = state.plan
        stage = plan.content_selection
        parsed, raw = self._dependencies.dialogue_turn(
            instance.session,
            turn,
            stage.parse_fn,
            instance.content_selection_prompt,
            plan.num_retries,
            plan.temperature,
            response_schema=stage.schema,
            output_max_length=getattr(
                stage.args,
                "output_max_length",
                4096,
            ),
            model_name=stage.args.model_name,
            attempt_trace=instance.trace["content_selection"],
            stop_on_cache_transport_failure=(
                state.cache_state.cache is not None
            ),
            call_records=state.call_records,
            call_context={
                "unique_id": instance.uid,
                "stage": "content_selection",
                "cache_bound": instance.cache_bound,
            },
        )
        remaining_retries = (
            plan.num_retries
            - len(instance.trace["content_selection"])
        )
        if (
            raw is None
            and state.cache_state.cache is not None
            and remaining_retries > 0
            and self._dependencies.cache_related_transport_failure(
                instance.trace["content_selection"]
            )
        ):
            parsed, raw = (
                self._session_service
                .retry_content_selection_without_cache(
                    state,
                    instance,
                    remaining_retries,
                )
            )
        if raw is None:
            self.record_failure(state, instance, source_row)
            return False, None

        result = dict(
            plan.content_selection_additional.get(instance.uid, {})
        )
        result.update(parsed)
        state.content_selection_results[instance.uid] = result
        row = self._dependencies.single_pipeline_row(
            stage.pipeline_fn,
            instance.uid,
            result,
            [source_row],
            "content_selection",
        )
        state.content_selection_rows[instance.uid] = row
        self._session_service.remove_completed_content_selection_demos(
            state,
            instance,
        )
        return True, row

    def record_failure(self, state, instance, source_row):
        """Propagate a terminal CS failure through every downstream result."""

        plan = state.plan
        uid = instance.uid
        logging.warning(
            "[dialogue] CS failed for %s — emitting ERROR row",
            uid,
        )
        error = dict(plan.content_selection_additional.get(uid, {}))
        error.update(
            {
                "final_output": "ERROR - dialogue CS failed",
                "alignments": [],
                "dialogue_attempt_trace": deepcopy(instance.trace),
                "dialogue_protocol_trace": deepcopy(instance.protocol),
            }
        )
        state.content_selection_results[uid] = error
        error_row = self._dependencies.single_pipeline_row(
            plan.content_selection.pipeline_fn,
            uid,
            error,
            [source_row],
            "content_selection",
        )
        state.content_selection_rows[uid] = error_row
        if plan.has_ambiguity_highlight:
            ah_error = {
                "final_output": (
                    "ERROR - upstream content_selection failed"
                ),
                "alignments": [],
                "upstream_skipped_reason": "model_error",
                "dialogue_attempt_trace": deepcopy(instance.trace),
                "dialogue_protocol_trace": deepcopy(instance.protocol),
            }
            state.ambiguity_highlight_results[uid] = ah_error
            ah_row = self._dependencies.single_pipeline_row(
                plan.ambiguity_highlight.pipeline_fn,
                uid,
                ah_error,
                [error_row],
                "ambiguity_highlight",
            )
            state.ambiguity_highlight_rows[uid] = ah_row
            state.fusion_source_rows[uid] = ah_row
        else:
            state.fusion_source_rows[uid] = error_row
        final_error = dict(error)
        final_error.pop("prompt_budget_trace", None)
        final_error["upstream_skipped_reason"] = "model_error"
        state.fusion_results[uid] = self._dependencies.with_gold_summary(
            final_error,
            source_row,
        )


__all__ = ["DialogueContentSelectionService"]
