"""Compatibility facade for the sequential-dialogue experiment.

The object-oriented implementation lives in
``attribute_first.application.sequential_dialogue``.  This module retains the
historical functions, signatures, constants, CLI, and monkey-patch boundaries
used by notebooks and offline tests.

Usage:
  python run_dialogue_sequential.py --setting MDS --split test \
      -o results/test/MDS/dialogue_seq \
      --model models/gemini-3-flash-preview --concurrency 4
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

import utils
from attribute_first.infrastructure import (
    CallableArtifactStore,
    CallableDialogueGateway,
)
from prompt_utils import (
    construct_prompts,
    get_data,
    get_subtask_prompt_structures,
)
from schemas import (
    CLUSTERING_SCHEMA,
    CONTENT_SELECTION_SCHEMA,
    SENTENCE_FUSION_SCHEMA,
)
from utils import (
    IncompleteGenerationError,
    create_chat_session,
    ensure_parseable_finish_reason,
    gemini_chat_call,
    get_last_call_metadata,
    get_last_call_usage,
    get_token_counter,
    get_token_usage,
    reset_last_call_usage,
    reset_token_usage,
    stable_value_sha256,
)

from attribute_first.application.sequential_dialogue import (
    HIGHLIGHT_END,
    HIGHLIGHT_START,
    SEQUENTIAL_PROTOCOL,
    SequentialDialogueInstanceRunner,
    SequentialDialoguePipelineRunner,
    SequentialInstanceDependencies,
    SequentialPipelineDependencies,
    SequentialPipelineResultAssembler,
    parse_clusters,
    parse_content_selection_spans,
    source_span_metadata,
    usage_from_attempt_trace,
)


logging.basicConfig(level=logging.INFO)
HS, HE = HIGHLIGHT_START, HIGHLIGHT_END


def _parse_cs_spans(cs_out):
    """Compatibility wrapper for the historical CS parser helper."""

    return parse_content_selection_spans(cs_out)


def _clusters_from(out, n_spans):
    """Compatibility wrapper for the historical clustering helper."""

    return parse_clusters(out, n_spans)


def _source_span_metadata(inst, document_file, span_text):
    """Compatibility wrapper for evaluator-required source coordinates."""

    return source_span_metadata(inst, document_file, span_text)


def _delta(u0):
    """Compatibility wrapper for trace-scoped per-instance usage."""

    return usage_from_attempt_trace(u0)


def _build_pipeline_format_results(source_instances, results):
    """Compatibility wrapper for evaluator-ready pipeline rows."""

    return SequentialPipelineResultAssembler().build(
        source_instances,
        results,
    )


def _instance_dependencies():
    """Capture current facade globals so legacy monkeypatches remain effective."""

    return SequentialInstanceDependencies(
        dialogue_gateway=_legacy_dialogue_gateway(),
        reset_last_call_usage=reset_last_call_usage,
        get_last_call_usage=get_last_call_usage,
        get_last_call_metadata=get_last_call_metadata,
        ensure_parseable_finish_reason=ensure_parseable_finish_reason,
        stable_value_sha256=stable_value_sha256,
        incomplete_generation_error=IncompleteGenerationError,
        time_module=time,
        content_selection_schema=CONTENT_SELECTION_SCHEMA,
        clustering_schema=CLUSTERING_SCHEMA,
        sentence_fusion_schema=SENTENCE_FUSION_SCHEMA,
        parse_content_selection=_parse_cs_spans,
        parse_clustering=_clusters_from,
        source_metadata=_source_span_metadata,
    )


def _legacy_dialogue_gateway():
    """Adapt patchable legacy chat functions to the dialogue port."""

    return CallableDialogueGateway(
        create_chat=lambda model_name, **_kwargs: create_chat_session(
            model_name
        ),
        send_message=gemini_chat_call,
    )


def _legacy_artifact_store():
    """Adapt atomic facade persistence to the artifact-store port."""

    return CallableArtifactStore(
        write_json=utils.atomic_write_json,
        write_jsonl=utils.atomic_write_jsonl,
    )


def run_instance(
    inst,
    cs_prompt,
    clustering_instr,
    fusion_instr,
    model,
    num_retries=3,
):
    """Drive one instance while preserving the historical public signature."""

    return SequentialDialogueInstanceRunner(_instance_dependencies()).run(
        inst,
        cs_prompt,
        clustering_instr,
        fusion_instr,
        model,
        num_retries=num_retries,
    )


def _pipeline_dependencies():
    """Capture all facade-owned boundaries immediately before one run."""

    return SequentialPipelineDependencies(
        get_data=get_data,
        get_prompt_structures=get_subtask_prompt_structures,
        construct_prompts=construct_prompts,
        get_token_counter=get_token_counter,
        reset_token_usage=reset_token_usage,
        get_token_usage=get_token_usage,
        run_instance=run_instance,
        save_results=utils.save_results,
        get_environment_flags=utils.get_af_environment_flags,
        artifact_store=_legacy_artifact_store(),
        build_pipeline_results=_build_pipeline_format_results,
        executor_factory=ThreadPoolExecutor,
        completed_futures=as_completed,
        log_info=logging.info,
        log_exception=logging.exception,
        protocol=SEQUENTIAL_PROTOCOL,
        highlight_start=HS,
        highlight_end=HE,
    )


def main(args):
    """Run the sequential pipeline through freshly captured dependencies."""

    return SequentialDialoguePipelineRunner(_pipeline_dependencies()).run(args)


def _argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--indir-alignments",
        dest="indir_alignments",
        default=None,
    )
    parser.add_argument(
        "--indir-prompt",
        dest="indir_prompt",
        default=None,
    )
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument(
        "--model",
        default="models/gemini-3-flash-preview",
    )
    parser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=30000,
        help="Fixed maximum prompt-token budget used by prompt shortening.",
    )
    parser.add_argument("--n-demos", dest="n_demos", type=int, default=3)
    parser.add_argument(
        "--num-retries",
        dest="num_retries",
        type=int,
        default=3,
        help="Positive number of attempts allowed for each dialogue turn.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--max-examples",
        dest="max_examples",
        type=int,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


if __name__ == "__main__":
    main(_argument_parser().parse_args())
