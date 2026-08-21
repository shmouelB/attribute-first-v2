"""Ambiguity-highlight stage service for stateful dialogue runs."""

import logging
from copy import deepcopy


class DialogueAmbiguityHighlightService:
    """Execute AH and propagate a terminal failure without running FiC."""

    def __init__(
        self,
        dependencies,
        prompt_builder,
        demonstration_service,
    ):
        self._dependencies = dependencies
        self._prompt_builder = prompt_builder
        self._demonstration_service = demonstration_service

    def run(
        self,
        state,
        instance,
        cs_row,
        source_row,
    ):
        """Execute one ambiguity-highlight turn."""

        plan = state.plan
        stage = plan.ambiguity_highlight
        prepared = self._prompt_builder.build(
            state,
            instance,
            stage,
            cs_row,
        )
        live_message = stage.continuation
        instance.protocol["ah_live_message_sha256"] = (
            self._dependencies.stable_value_sha256(live_message)
        )
        instance.protocol["ah_validation_prompt_sha256"] = (
            self._dependencies.stable_value_sha256(
                prepared.validation_prompt
            )
        )
        if instance.uses_roles:
            demo_lease = self._demonstration_service.inject(
                state=state,
                instance=instance,
                stage_name="ambiguity_highlight",
                demos=prepared.demos,
                role_messages=prepared.role_messages,
                demo_count=prepared.live_demo_count,
            )
        else:
            demo_lease = None
        parsed, raw = self._dependencies.dialogue_turn(
            instance.session,
            live_message,
            stage.parse_fn,
            prepared.validation_prompt,
            getattr(stage.args, "num_retries", plan.num_retries),
            getattr(stage.args, "temperature", plan.temperature),
            response_schema=stage.schema,
            output_max_length=getattr(
                stage.args,
                "output_max_length",
                4096,
            ),
            model_name=stage.args.model_name,
            attempt_trace=instance.trace["ambiguity_highlight"],
            call_records=state.call_records,
            call_context={
                "unique_id": instance.uid,
                "stage": "ambiguity_highlight",
                "cache_bound": instance.cache_bound,
            },
        )
        if raw is None:
            self.record_failure(
                state,
                instance,
                cs_row,
                source_row,
                prepared.additional,
            )
            return False, None
        if demo_lease is not None:
            self._demonstration_service.remove(
                instance=instance,
                lease=demo_lease,
            )
        result = dict(prepared.additional.get(instance.uid, {}))
        result.update(parsed)
        state.ambiguity_highlight_results[instance.uid] = result
        row = self._dependencies.single_pipeline_row(
            stage.pipeline_fn,
            instance.uid,
            result,
            [cs_row],
            "ambiguity_highlight",
        )
        state.ambiguity_highlight_rows[instance.uid] = row
        return True, row

    def record_failure(
        self,
        state,
        instance,
        cs_row,
        source_row,
        additional,
    ):
        """Persist AH failure evidence and prevent downstream generation."""

        uid = instance.uid
        logging.warning(
            "[dialogue] AH failed for %s — skipping FiC and "
            "emitting ERROR row",
            uid,
        )
        error_result = {
            "final_output": "ERROR - dialogue AH failed",
            "alignments": [],
            "dialogue_attempt_trace": deepcopy(instance.trace),
            "dialogue_protocol_trace": deepcopy(instance.protocol),
        }
        ah_error = dict(additional.get(uid, {}))
        ah_error.update(error_result)
        state.ambiguity_highlight_results[uid] = ah_error
        ah_row = self._dependencies.single_pipeline_row(
            state.plan.ambiguity_highlight.pipeline_fn,
            uid,
            ah_error,
            [cs_row],
            "ambiguity_highlight",
        )
        state.ambiguity_highlight_rows[uid] = ah_row
        state.fusion_source_rows[uid] = ah_row
        final_error = dict(error_result)
        final_error["upstream_skipped_reason"] = "model_error"
        state.fusion_results[uid] = self._dependencies.with_gold_summary(
            final_error,
            source_row,
        )


__all__ = ["DialogueAmbiguityHighlightService"]
