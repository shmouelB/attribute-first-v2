"""Explicit dependency contract for dialogue-pipeline orchestration."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..ports import ArtifactStore, DialogueGateway


@dataclass(frozen=True)
class DialoguePipelineDependencies:
    """All patchable collaborators captured from the legacy façade."""

    update_args: Callable
    get_token_counter: Callable
    env_flag: Callable
    get_af_environment_flags: Callable
    stable_value_sha256: Callable
    artifact_sha256: Callable
    artifact_store: ArtifactStore
    get_subtask_prompt_structures: Callable
    get_data: Callable
    construct_prompts: Callable
    get_subtask_funcs: Callable
    dialogue_gateway: DialogueGateway
    save_results: Callable
    get_token_usage: Callable
    reset_token_usage: Callable
    tqdm: Callable
    subtask_schemas: Mapping[str, Any]
    caching_module: Any
    normalize_model_name: Callable
    context_cache_target: str
    build_subtask_args: Callable
    load_subtask_prompt_dict: Callable
    dialogue_role_view: Callable
    dialogue_demo_histories: Callable
    append_dialogue_history: Callable
    dialogue_turn: Callable
    jsonable_dialogue_value: Callable
    cache_related_transport_failure: Callable
    single_pipeline_row: Callable
    fallback_error_pipeline_row: Callable
    with_gold_summary: Callable
    assert_uid_coverage: Callable
    content_selection_live_state: Callable
    fic_highlight_registry: Callable
    system_instruction: str

    @classmethod
    def from_namespace(cls, namespace):
        """Capture current façade bindings at the start of every run."""
        return cls(
            update_args=namespace["update_args"],
            get_token_counter=namespace["get_token_counter"],
            env_flag=namespace["env_flag"],
            get_af_environment_flags=namespace[
                "get_af_environment_flags"
            ],
            stable_value_sha256=namespace["stable_value_sha256"],
            artifact_sha256=namespace["artifact_sha256"],
            artifact_store=namespace["_legacy_artifact_store"](),
            get_subtask_prompt_structures=namespace[
                "get_subtask_prompt_structures"
            ],
            get_data=namespace["get_data"],
            construct_prompts=namespace["construct_prompts"],
            get_subtask_funcs=namespace["get_subtask_funcs"],
            dialogue_gateway=namespace["_legacy_dialogue_gateway"](),
            save_results=namespace["save_results"],
            get_token_usage=namespace["get_token_usage"],
            reset_token_usage=namespace["reset_token_usage"],
            tqdm=namespace["tqdm"],
            subtask_schemas=namespace["SUBTASK_SCHEMAS"],
            caching_module=namespace["genai_caching"],
            normalize_model_name=namespace["_normalize_model_name"],
            context_cache_target=namespace["_CONTEXT_CACHE_TARGET"],
            build_subtask_args=namespace["_build_subtask_args"],
            load_subtask_prompt_dict=namespace[
                "_load_subtask_prompt_dict"
            ],
            dialogue_role_view=namespace["_dialogue_role_view"],
            dialogue_demo_histories=namespace[
                "_dialogue_demo_histories"
            ],
            append_dialogue_history=namespace[
                "_append_dialogue_history"
            ],
            dialogue_turn=namespace["_dialogue_turn"],
            jsonable_dialogue_value=namespace[
                "_jsonable_dialogue_value"
            ],
            cache_related_transport_failure=namespace[
                "_cache_related_transport_failure"
            ],
            single_pipeline_row=namespace["_single_pipeline_row"],
            fallback_error_pipeline_row=namespace[
                "_fallback_error_pipeline_row"
            ],
            with_gold_summary=namespace["_with_gold_summary"],
            assert_uid_coverage=namespace["_assert_uid_coverage"],
            content_selection_live_state=namespace[
                "_content_selection_live_state"
            ],
            fic_highlight_registry=namespace[
                "_fic_highlight_registry"
            ],
            system_instruction=namespace[
                "DIALOGUE_SYSTEM_INSTRUCTION"
            ],
        )


__all__ = ["DialoguePipelineDependencies"]
