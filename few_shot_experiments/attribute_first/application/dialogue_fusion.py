"""Fusion-in-context stage service for stateful dialogue runs."""

import logging
from copy import deepcopy


class DialogueFusionService:
    """Execute one attributed fusion turn from the live upstream row."""

    def __init__(
        self,
        dependencies,
        prompt_builder,
        demonstration_service,
    ):
        self._dependencies = dependencies
        self._prompt_builder = prompt_builder
        self._demonstration_service = demonstration_service

    def run(self, state, instance, source_row, population_row):
        """Execute FiC and always leave one terminal result for the UID."""

        plan = state.plan
        stage = plan.fusion
        prepared = self._prompt_builder.build(
            state,
            instance,
            stage,
            source_row,
        )
        additional_for_uid = prepared.additional[instance.uid]
        registry = self._dependencies.fic_highlight_registry(
            additional_for_uid
        )
        live_message = (
            f"{stage.continuation}\n\n"
            "### CANONICAL HIGHLIGHT IDS FOR THIS TURN ###\n"
            f"{registry}\n\n"
            "Use every canonical highlight ID exactly once in the "
            "structured response."
        )
        hash_value = self._dependencies.stable_value_sha256
        instance.protocol.update(
            {
                "fic_validation_prompt_sha256": hash_value(
                    prepared.validation_prompt
                ),
                "fic_live_message_sha256": hash_value(live_message),
                "fic_highlight_registry": registry,
                "fic_highlight_registry_sha256": hash_value(registry),
            }
        )
        if instance.uses_roles:
            self._demonstration_service.inject(
                state=state,
                instance=instance,
                stage_name="fusion_in_context",
                demos=prepared.demos,
                role_messages=prepared.role_messages,
                demo_count=prepared.live_demo_count,
            )
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
            attempt_trace=instance.trace["fusion_in_context"],
            call_records=state.call_records,
            call_context={
                "unique_id": instance.uid,
                "stage": "fusion_in_context",
                "cache_bound": instance.cache_bound,
            },
        )
        if raw is not None:
            result = dict(additional_for_uid)
            result.update(parsed)
            result = self._dependencies.with_gold_summary(
                result,
                population_row,
            )
            result["dialogue_attempt_trace"] = deepcopy(instance.trace)
            result["dialogue_protocol_trace"] = deepcopy(
                instance.protocol
            )
            state.fusion_results[instance.uid] = result
            return
        logging.warning(
            "[dialogue] FiC failed for %s — emitting ERROR row",
            instance.uid,
        )
        error_result = dict(additional_for_uid)
        error_result.update(
            {
                "final_output": "ERROR - dialogue FiC failed",
                "alignments": [],
                "dialogue_attempt_trace": deepcopy(instance.trace),
                "dialogue_protocol_trace": deepcopy(
                    instance.protocol
                ),
            }
        )
        state.fusion_results[instance.uid] = (
            self._dependencies.with_gold_summary(
                error_result,
                population_row,
            )
        )


__all__ = ["DialogueFusionService"]
