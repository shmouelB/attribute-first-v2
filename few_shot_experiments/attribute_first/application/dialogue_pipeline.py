"""Stateful dialogue-pipeline orchestration facade."""

import logging
import os

from .dialogue_ambiguity import DialogueAmbiguityHighlightService
from .dialogue_coherence import DialogueCoherenceService
from .dialogue_cache import DialogueCacheManager
from .dialogue_content_selection import (
    DialogueContentSelectionService,
)
from .dialogue_demonstrations import DialogueDemonstrationService
from .dialogue_fusion import DialogueFusionService
from .dialogue_persistence import DialogueResultPersister
from .dialogue_preparation import DialoguePlanBuilder
from .dialogue_rerun import DialogueRerunService
from .dialogue_sessions import DialogueSessionService
from .dialogue_shared_content_selection import (
    DialogueContentSelectionCheckpointService,
)
from .dialogue_stage_prompts import DialogueStagePromptBuilder
from .dialogue_state import DialogueInstanceState, DialogueRunState


class DialoguePipelineRunner:
    """Coordinate a plan while delegating each application responsibility."""

    def __init__(self, dependencies):
        self._dependencies = dependencies
        self._plan_builder = DialoguePlanBuilder(dependencies)
        self._cache_manager = DialogueCacheManager(
            caching_module=dependencies.caching_module,
            normalize_model_name=dependencies.normalize_model_name,
            context_cache_target=dependencies.context_cache_target,
        )
        self._session_service = DialogueSessionService(dependencies)
        self._demonstration_service = DialogueDemonstrationService(
            dependencies
        )
        self._stage_prompt_builder = DialogueStagePromptBuilder(
            dependencies
        )
        self._content_selection_service = (
            DialogueContentSelectionService(
                dependencies,
                self._session_service,
            )
        )
        self._ambiguity_service = (
            DialogueAmbiguityHighlightService(
                dependencies,
                self._stage_prompt_builder,
                self._demonstration_service,
            )
        )
        self._fusion_service = DialogueFusionService(
            dependencies,
            self._stage_prompt_builder,
            self._demonstration_service,
        )
        self._coherence_service = DialogueCoherenceService(
            dependencies,
            self._stage_prompt_builder,
        )
        self._result_persister = DialogueResultPersister(dependencies)
        self._rerun_service = DialogueRerunService()
        self._shared_content_selection = (
            DialogueContentSelectionCheckpointService(
                dependencies,
                self._session_service,
            )
        )

    def run(
        self,
        args,
        full_configs,
        original_args_dict,
        outdir,
        intermediate_outdir,
    ):
        """Execute one complete dialogue run and finalize cache ownership."""

        plan = self._plan_builder.build(
            args=args,
            full_configs=full_configs,
            original_args_dict=original_args_dict,
            outdir=outdir,
            intermediate_outdir=intermediate_outdir,
        )
        state = DialogueRunState(plan=plan)
        shared_entries = self._shared_content_selection.load(plan)
        if getattr(plan, "rerun_context", None) is not None:
            self._rerun_service.hydrate(state, shared_entries)
        self._dependencies.artifact_store.write_json(
            os.path.join(outdir, "args.json"),
            state.args_snapshot,
        )
        self._dependencies.reset_token_usage()
        state.cache_state = self._cache_manager.create(
            requested=(
                False
                if shared_entries is not None
                else self._dependencies.env_flag("AF_CONTEXT_CACHE")
            ),
            cs_prompts=plan.content_selection_prompts,
            role_messages=plan.dialogue_role_messages,
            roles_requested=plan.roles_requested,
            model_name=plan.model_name,
        )

        try:
            logging.info(
                "[dialogue] running per-instance chat sessions ..."
            )
            for uid, prompt in self._dependencies.tqdm(
                plan.content_selection_prompts.items()
            ):
                self._run_instance(
                    state,
                    uid,
                    prompt,
                    (
                        shared_entries[uid]
                        if shared_entries is not None
                        else None
                    ),
                )
            self._save_results(state)
            self._shared_content_selection.persist(state)
        finally:
            cache_trace = self._cache_manager.finalize(
                state.cache_state,
                state.call_records,
            )
            self._persist_runtime_artifacts(state, cache_trace)

    def _run_instance(
        self,
        state,
        uid,
        cs_prompt,
        shared_entry=None,
    ):
        plan = state.plan
        source_by_uid = {
            instance["unique_id"]: instance
            for instance in plan.alignments
        }
        instance = DialogueInstanceState(
            uid=uid,
            content_selection_prompt=cs_prompt,
            role_payload=(
                plan.dialogue_role_messages.get(uid)
                if plan.roles_requested
                else None
            ),
            uses_roles=plan.roles_requested,
            protocol={
                "cs_parse_prompt_sha256": (
                    self._dependencies.stable_value_sha256(cs_prompt)
                )
            },
        )
        uses_coherence_planning = (
            getattr(plan, "uses_coherence_planning", False) is True
        )
        if not uses_coherence_planning:
            instance.trace.pop("clustering", None)
            instance.trace.pop("reorder", None)
        if shared_entry is None:
            cs_turn = self._session_service.start(state, instance)
            cs_ok, cs_row = self._content_selection_service.run(
                state,
                instance,
                cs_turn,
                source_by_uid[uid],
            )
            self._shared_content_selection.capture(state, instance)
        else:
            cs_ok, cs_row = self._shared_content_selection.restore(
                state,
                instance,
                shared_entry,
                source_by_uid[uid],
            )
        if not cs_ok:
            if uses_coherence_planning:
                self._coherence_service.record_upstream_failure(
                    state,
                    instance,
                    source_by_uid[uid],
                    "content_selection",
                )
            return
        if plan.has_ambiguity_highlight:
            ah_ok, fusion_source = self._ambiguity_service.run(
                state,
                instance,
                cs_row,
                source_by_uid[uid],
            )
            if not ah_ok:
                if uses_coherence_planning:
                    self._coherence_service.record_upstream_failure(
                        state,
                        instance,
                        source_by_uid[uid],
                        "ambiguity_highlight",
                    )
                return
        else:
            fusion_source = cs_row
        state.fusion_source_rows[uid] = fusion_source
        if uses_coherence_planning:
            self._coherence_service.run(
                state,
                instance,
                fusion_source,
                source_by_uid[uid],
            )
        else:
            self._fusion_service.run(
                state,
                instance,
                fusion_source,
                source_by_uid[uid],
            )

    # Patch-compatible persistence seams retained by the public facade.
    def _save_results(self, state):
        return self._result_persister.save_results(state)

    def _persist_runtime_artifacts(self, state, cache_trace):
        return self._result_persister.persist_runtime_artifacts(
            state,
            cache_trace,
        )


__all__ = ["DialoguePipelineRunner"]
