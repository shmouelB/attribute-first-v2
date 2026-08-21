"""Legacy compatibility façade for full pipeline generation.

Application orchestration and artifact persistence live under
``attribute_first``.  Every wrapper captures this module's current globals so
existing tests and notebooks can continue patching historical entry points.
"""

import argparse
import json
import logging
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

from google.generativeai import caching as genai_caching
from tqdm import tqdm

from utils import (
    IncompleteGenerationError,
    _CONTEXT_CACHE_TARGET,
    _normalize_model_name,
    artifact_sha256,
    atomic_write_json,
    atomic_write_jsonl,
    config_protocol_environment,
    create_chat_session,
    ensure_parseable_finish_reason,
    env_flag,
    gemini_chat_call,
    get_af_environment_flags,
    get_last_call_metadata,
    get_last_call_usage,
    get_token_counter,
    get_token_usage,
    protocol_environment,
    reset_last_call_usage,
    reset_token_usage,
    save_results,
    stable_value_sha256,
    summarize_response_metadata,
    update_args,
    validate_protocol_environment_flags,
)
from run_script import main as main_func
from run_iterative_sentence_generation import (
    main as iterative_sent_gen_main,
)
from subtask_specific_utils import (
    construct_non_demo_part,
    construct_prompts,
    convert_FiC_CoT_results_to_pipeline_format,
    convert_ambiguity_highlight_results_to_pipeline_format,
    get_data,
    get_subtask_funcs,
    get_subtask_prompt_structures,
    parse_FiC_response,
    parse_ambiguity_highlight_response,
)
from schemas import SUBTASK_SCHEMAS

from attribute_first.application.dialogue_dependencies import (
    DialoguePipelineDependencies,
)
from attribute_first.application.dialogue_pipeline import (
    DialoguePipelineRunner,
)
from attribute_first.application.dialogue_turns import (
    DIALOGUE_SYSTEM_INSTRUCTION,
    DialogueTurnDependencies,
    DialogueTurnExecutor,
    append_dialogue_history,
    assert_uid_coverage,
    cache_related_transport_failure,
    content_selection_live_state,
    dialogue_demo_histories,
    dialogue_role_view,
    fallback_error_pipeline_row,
    fic_highlight_registry,
    jsonable_dialogue_value,
    single_pipeline_row,
    transport_only_failure,
    with_gold_summary,
)
from attribute_first.application.pipeline_application import (
    PipelineApplicationDependencies,
    PipelineApplicationRunner,
)
from attribute_first.application.protocol import (
    dialogue_protocol_environment as _dialogue_environment,
)
from attribute_first.artifacts.pipeline_artifacts import (
    ArtifactDependencies,
    PipelineArtifactService,
)
from attribute_first.artifacts.standard_run_artifacts import RerunPolicy
from attribute_first.infrastructure import (
    CallableArtifactStore,
    CallableDialogueGateway,
)
from attribute_first.stages.configuration import DEFAULT_GENERATION


logging.basicConfig(level=logging.INFO)


def _legacy_artifact_store():
    """Expose patchable legacy writes through the persistence port."""
    return CallableArtifactStore(
        write_json=atomic_write_json,
        write_jsonl=atomic_write_jsonl,
    )


def _legacy_dialogue_gateway():
    """Expose patchable legacy chat calls through the dialogue port."""
    return CallableDialogueGateway(
        create_chat=create_chat_session,
        send_message=gemini_chat_call,
    )


def _artifact_service():
    return PipelineArtifactService(
        ArtifactDependencies.from_namespace(globals())
    )


def _dialogue_role_view(payload, stage_name):
    return dialogue_role_view(payload, stage_name)


def _dialogue_demo_histories(
    role_messages,
    instance_ids,
    stage_name,
    n_demos,
):
    return dialogue_demo_histories(
        role_messages,
        instance_ids,
        stage_name,
        n_demos,
    )


def _append_dialogue_history(session, stage_history):
    return append_dialogue_history(session, stage_history)


def _jsonable_dialogue_value(value):
    return jsonable_dialogue_value(value)


def _transport_only_failure(attempt_trace):
    return transport_only_failure(attempt_trace)


def _cache_related_transport_failure(attempt_trace):
    return cache_related_transport_failure(attempt_trace)


def _single_pipeline_row(
    converter,
    instance_id,
    result,
    source_rows,
    stage,
):
    return single_pipeline_row(
        converter,
        instance_id,
        result,
        source_rows,
        stage,
    )


def _fallback_error_pipeline_row(source_row, stage, error_text):
    return fallback_error_pipeline_row(
        source_row,
        stage,
        error_text,
    )


def _with_gold_summary(result, source_row):
    return with_gold_summary(result, source_row)


def _assert_uid_coverage(label, mapping, expected_ids):
    return assert_uid_coverage(label, mapping, expected_ids)


def _content_selection_live_state(final_output):
    return content_selection_live_state(final_output)


def _fic_highlight_registry(fic_additional):
    return fic_highlight_registry(fic_additional)


@contextmanager
def dialogue_protocol_environment(full_configs):
    with _dialogue_environment(
        full_configs,
        validate_protocol_environment_flags=(
            validate_protocol_environment_flags
        ),
        protocol_environment=protocol_environment,
    ) as effective:
        yield effective


def _log_stage_health(pipeline_format_path, label):
    return _artifact_service().log_stage_health(
        pipeline_format_path,
        label,
    )


def persist_pipeline_token_usage(outdir):
    return _artifact_service().persist_token_usage(outdir)


def persist_pipeline_response_metadata(outdir, dialogue_mode):
    return _artifact_service().persist_response_metadata(
        outdir,
        dialogue_mode,
    )


def persist_dialogue_content_selection_usage(outdir):
    return _artifact_service().persist_dialogue_content_selection_usage(
        outdir
    )


def _resolved_input_path(config, args, key, default):
    return _artifact_service().resolved_input_path(
        config,
        args,
        key,
        default,
    )


def _load_fixed_population(input_path, max_examples, payload=None):
    return _artifact_service().load_fixed_population(
        input_path,
        max_examples,
        payload=payload,
    )


def _capture_provenance_file(source_path, outdir, relative_path):
    return _artifact_service().capture_provenance_file(
        source_path,
        outdir,
        relative_path,
    )


def persist_pipeline_provenance(args, full_configs, outdir):
    return _artifact_service().persist_provenance(
        args,
        full_configs,
        outdir,
    )


def prepare_shared_content_selection(args, outdir):
    return _artifact_service().prepare_shared_content_selection(
        args,
        outdir,
    )


def prepare_dialogue_rerun(args, outdir):
    """Load an immutable parent before the child output is claimed."""

    return RerunPolicy(artifact_sha256).load(args, outdir)


def run_subtask(
    full_configs,
    subtask_name,
    curr_outdir,
    original_args_dict,
    indir_alignments=None,
):
    dependencies = PipelineApplicationDependencies.from_namespace(
        globals()
    )
    return PipelineApplicationRunner(dependencies).run_subtask(
        full_configs,
        subtask_name,
        curr_outdir,
        original_args_dict,
        indir_alignments,
    )


def _dialogue_default_role_protocol(args):
    """Return a CLI role default only when stage configs are silent."""

    if not getattr(args, "dialogue_mode", False):
        return None
    with open(args.config_file, "r", encoding="utf-8") as pipeline_file:
        full_configs = json.load(pipeline_file)
    for stage in full_configs:
        with open(
            stage["config_file"],
            "r",
            encoding="utf-8",
        ) as config_file:
            stage_config = json.load(config_file)
        protocol = stage_config.get("protocol")
        flags = (
            protocol.get("environment_flags")
            if isinstance(protocol, dict)
            else None
        )
        if isinstance(flags, dict) and "AF_USE_ROLES" in flags:
            return None
    use_roles = getattr(
        args,
        "use_roles",
        DEFAULT_GENERATION.use_roles,
    )
    if (
        use_roles is DEFAULT_GENERATION.use_roles
        and "AF_USE_ROLES" in os.environ
    ):
        use_roles = env_flag("AF_USE_ROLES")
    return DEFAULT_GENERATION.protocol_with_role_default(
        None,
        use_roles=use_roles,
    )


def main(args):
    dependencies = PipelineApplicationDependencies.from_namespace(
        globals()
    )
    with protocol_environment(_dialogue_default_role_protocol(args)):
        return PipelineApplicationRunner(dependencies).run(args)


def _subtask_cfg(full_configs, subtask_name):
    matches = [
        entry
        for entry in full_configs
        if entry["subtask"] == subtask_name
    ]
    return matches[0] if matches else None


def _build_subtask_args(
    full_configs,
    subtask_name,
    original_args_dict,
    curr_outdir,
    indir_alignments=None,
):
    config = _subtask_cfg(full_configs, subtask_name)
    function_args = deepcopy(original_args_dict)
    function_args.update(config)
    function_args["outdir"] = curr_outdir
    function_args["indir_alignments"] = indir_alignments
    return argparse.Namespace(**function_args)


def _load_subtask_prompt_dict(args):
    indir_prompt = (
        args.indir_prompt
        if hasattr(args, "indir_prompt") and args.indir_prompt
        else f"prompts/{args.setting}.json"
    )
    with open(indir_prompt, "r") as prompt_file:
        return json.loads(prompt_file.read())


def _dialogue_turn(
    session,
    message,
    parse_fn,
    prompt_for_parse,
    num_retries,
    temperature,
    response_schema=None,
    output_max_length=4096,
    model_name=None,
    attempt_trace=None,
    stop_on_cache_transport_failure=False,
    call_records=None,
    call_context=None,
):
    dependencies = DialogueTurnDependencies.from_namespace(globals())
    return DialogueTurnExecutor(dependencies).execute(
        session,
        message,
        parse_fn,
        prompt_for_parse,
        num_retries,
        temperature,
        response_schema=response_schema,
        output_max_length=output_max_length,
        model_name=model_name,
        attempt_trace=attempt_trace,
        stop_on_cache_transport_failure=(
            stop_on_cache_transport_failure
        ),
        call_records=call_records,
        call_context=call_context,
    )


def run_dialogue_pipeline(
    args,
    full_configs,
    original_args_dict,
    outdir,
    intermediate_outdir,
):
    dependencies = DialoguePipelineDependencies.from_namespace(
        globals()
    )
    return DialoguePipelineRunner(dependencies).run(
        args,
        full_configs,
        original_args_dict,
        outdir,
        intermediate_outdir,
    )


def _build_argparser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="path to json config file.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        help="path to output csv.",
    )
    parser.add_argument(
        "--indir-alignments",
        type=str,
        default=None,
        help=(
            "path to json file with alignments (if nothing is passed - "
            "goes to default under data/{setting}/{split}.json)."
        ),
    )
    parser.add_argument(
        "--indir-prompt",
        type=str,
        default=None,
        help=(
            "path to json file with the prompt structure and ICL examples "
            "(if nothing is passed - goes to default under "
            "prompts/{setting}.json)."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_GENERATION.model_name,
        help="full provider model ID",
    )
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=30000,
        help=(
            "Explicit fixed prompt-token budget; required by controlled "
            "configs."
        ),
    )
    parser.add_argument(
        "--n-demos",
        type=int,
        default=2,
        help="number of ICL examples (default 2)",
    )
    parser.add_argument(
        "--num-retries",
        type=int,
        default=1,
        help="number of retries of running the model.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature of generation",
    )
    parser.add_argument(
        "--debugging",
        action="store_true",
        default=False,
        help="if debugging mode.",
    )
    parser.add_argument(
        "--merge-cross-sents-highlights",
        action="store_true",
        default=False,
        help=(
            "whether to merge consecutive highlights that span across "
            "several sentences."
        ),
    )
    parser.add_argument(
        "--CoT",
        action="store_true",
        default=False,
        help=(
            "whether to use a CoT approach (relevant for FiC and "
            "clustering)."
        ),
    )
    parser.add_argument(
        "--cut-surplus",
        action="store_true",
        default=False,
        help=(
            "whether to cut surplus text from prompts (in subtask with "
            "given highlights - everything after last highlight, and in "
            "tasks without - last prct_surplus sentences)."
        ),
    )
    parser.add_argument(
        "--prct-surplus",
        type=float,
        default=None,
        help=(
            "for subtasks without given highlights (e.g. "
            "content_selection, e2e_only_setting, or ALCE) - what "
            "percentage of top document sents to drop in cases when the "
            "prompts are too long."
        ),
    )
    parser.add_argument(
        "--always-with-question",
        action="store_true",
        default=False,
        help=(
            "relevant for LFQA - whether to always add the question "
            "(also to clustering and FiC)"
        ),
    )
    parser.add_argument(
        "--num-demo-changes",
        type=int,
        default=4,
        help=(
            "number of changing demos when the currently-chosen set of "
            "demos returns an ERROR."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="if need to rerun on instances that had errors",
    )
    parser.add_argument(
        "--rerun-path",
        type=str,
        default=None,
        help="path to rerun on (where the results are)",
    )
    parser.add_argument(
        "--rerun-n-demos",
        type=int,
        default=None,
        help=(
            "new n_demos for rerun in cases when the current n_demos "
            "doesnt work."
        ),
    )
    parser.add_argument(
        "--rerun-temperature",
        type=float,
        default=None,
        help=(
            "new temperature for rerun in cases when the current "
            "temperature doesnt work."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help=(
            "cap total examples processed (for small smoke tests without "
            "--debugging, which forces fixed demos)."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "concurrent model calls within each stage (1 = sequential). "
            "Cuts wall-clock ~Nx for I/O-bound API calls."
        ),
    )
    parser.add_argument(
        "--no-prefix",
        action="store_true",
        default=False,
        help="ablation study where the prefix is not add.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260728,
        help="Seed used for reproducible demonstration selection.",
    )
    structured_output = parser.add_mutually_exclusive_group()
    structured_output.add_argument(
        "--structured-output",
        dest="structured_output",
        action="store_true",
        help=(
            "Use Gemini JSON mode (response_schema) for each subtask. "
            "FiC still requires a complete attributed response."
        ),
    )
    structured_output.add_argument(
        "--no-structured-output",
        dest="structured_output",
        action="store_false",
        help="Legacy opt-out: request the historical free-text response.",
    )
    parser.set_defaults(
        structured_output=DEFAULT_GENERATION.structured_output
    )
    parser.add_argument(
        "--no-roles",
        dest="use_roles",
        action="store_false",
        help="Legacy opt-out: use the historical flat-prompt transport.",
    )
    parser.set_defaults(use_roles=DEFAULT_GENERATION.use_roles)
    parser.add_argument(
        "--dialogue-mode",
        action="store_true",
        default=False,
        help=(
            "Run the CoT pipeline as a stateful multi-turn dialogue. Each "
            "instance shares one ChatSession; downstream turns receive "
            "just-in-time demos plus the new live state/task, without "
            "re-sending documents in application code. Only compatible "
            "with CoT (fusion_in_context) pipelines."
        ),
    )
    parser.add_argument(
        "--planned-dialogue",
        action="store_true",
        default=False,
        help=(
            "Append coherence clustering, reordering, and planned fusion "
            "turns to the same LFQA dialogue session. Requires "
            "--dialogue-mode and --concurrency 1."
        ),
    )
    parser.add_argument(
        "--shared-content-selection-source",
        default=None,
        help=(
            "completed baseline cell whose exactly equivalent "
            "content-selection stage is reused without another provider "
            "call"
        ),
    )
    parser.add_argument(
        "--canonical-cell-id",
        default=None,
        help="catalog identity injected by the controlled campaign",
    )
    parser.add_argument(
        "--generation-strategy",
        choices=("direct", "planned"),
        default=None,
        help="declared generation factor for controlled provenance",
    )
    parser.add_argument(
        "--demonstration-mode",
        choices=("few_shot", "zero_shot"),
        default=None,
        help="declared demonstration factor for controlled provenance",
    )
    parser.add_argument(
        "--context-augmentation",
        choices=("enabled", "disabled"),
        default=None,
        help="declared context-augmentation factor",
    )
    parser.add_argument(
        "--transport-mode",
        choices=("independent", "dialogue"),
        default=None,
        help="declared provider-transport factor",
    )
    return parser


if __name__ == "__main__":
    main(_build_argparser().parse_args())
