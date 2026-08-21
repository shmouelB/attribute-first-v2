"""Legacy facade for the iterative sentence-generation application.

The public functions in this module intentionally retain their historical
signatures.  Their implementation is delegated to dependency-injected
application objects so callers and tests can continue monkeypatching the
legacy module boundaries.
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict

import numpy as np
from tqdm import tqdm

from attribute_first.stages.configuration import DEFAULT_GENERATION
from attribute_first.application.iterative_sentence_generation import (
    IterativeApplicationDependencies,
    IterativeExecutionDependencies,
    IterativeGenerationExecutor,
    IterativePersistenceDependencies,
    IterativePromptBuilder,
    IterativePromptDependencies,
    IterativeResultConverter,
    IterativeResultPersister,
    IterativeSentenceGenerationApplication,
    parse_iterative_sentence_response,
)
from subtask_specific_utils import get_data
from utils import (
    artifact_sha256,
    atomic_write_json,
    extract_highlights,
    get_highlighted_doc,
    get_token_counter,
    get_token_usage,
    make_demo,
    prompt_model,
    reset_token_usage,
    rmv_txt_after_last_highlight,
    save_results,
    update_args,
)


logging.basicConfig(level=logging.INFO)
MODEL_DEFAULT = DEFAULT_GENERATION.model_name


def _prompt_builder():
    """Create a builder from the facade's current, patchable dependencies."""

    return IterativePromptBuilder(
        IterativePromptDependencies(
            extract_highlights=extract_highlights,
            get_highlighted_doc=get_highlighted_doc,
            make_demo=make_demo,
            remove_after_last_highlight=rmv_txt_after_last_highlight,
        )
    )


def keep_specific_occurrences(s, substring, occurrences):
    """Keep only specific occurrences of a substring in a string."""

    return IterativePromptBuilder.keep_specific_occurrences(
        s,
        substring,
        occurrences,
    )


def adapt_demo(train_item, curr_cluster_ind, no_prefix):
    return _prompt_builder().adapt_demo(
        train_item,
        curr_cluster_ind,
        no_prefix,
    )


def construct_curr_non_demo_part(
    curr_docs,
    alignments,
    highlight_start_tkn_placeholder,
    highlight_end_tkn_placeholder,
    merge_cross_sents_highlights,
    prompt_dict,
    prefix,
    instruction_prompt,
    prompt_structure,
    always_with_question: bool,
    instance_question: str,
    cut_surplus: bool = False,
):
    return _prompt_builder().construct_non_demo_part(
        curr_docs,
        alignments,
        highlight_start_tkn_placeholder,
        highlight_end_tkn_placeholder,
        merge_cross_sents_highlights,
        prompt_dict,
        prefix,
        instruction_prompt,
        prompt_structure,
        always_with_question,
        instance_question,
        cut_surplus,
    )


def construct_curr_prompt(
    prompt_dict,
    alignments,
    curr_docs,
    used_demos,
    curr_cluster_ind,
    prefix,
    merge_cross_sents_highlights,
    tkn_counter: Dict,
    cut_surplus: bool = False,
    always_with_question: bool = False,
    instance_question: str = None,
    no_prefix: bool = False,
):
    return _prompt_builder().construct_prompt(
        prompt_dict,
        alignments,
        curr_docs,
        used_demos,
        curr_cluster_ind,
        prefix,
        merge_cross_sents_highlights,
        tkn_counter,
        cut_surplus,
        always_with_question,
        instance_question,
        no_prefix,
    )


def parse_itertive_sent_gen_response(response, prompt):
    """Preserve the original misspelled parser entry point."""

    return parse_iterative_sentence_response(response, prompt)


def iterative_sent_gen_prompting(
    alignments_dict,
    prompt_dict,
    used_demos,
    model_name,
    num_retries,
    debugging,
    n_demos,
    num_demo_changes,
    temperature,
    merge_cross_sents_highlights,
    tkn_counter,
    cut_surplus: bool = False,
    always_with_question: bool = False,
    no_prefix: bool = False,
    rng=None,
):
    executor = IterativeGenerationExecutor(
        IterativeExecutionDependencies(
            construct_prompt=construct_curr_prompt,
            prompt_model=prompt_model,
            parse_response=parse_itertive_sent_gen_response,
            progress=tqdm,
            log_info=logging.info,
            rng_factory=np.random.default_rng,
        )
    )
    return executor.generate(
        alignments_dict=alignments_dict,
        prompt_dict=prompt_dict,
        used_demos=used_demos,
        model_name=model_name,
        num_retries=num_retries,
        debugging=debugging,
        n_demos=n_demos,
        num_demo_changes=num_demo_changes,
        temperature=temperature,
        merge_cross_sents_highlights=merge_cross_sents_highlights,
        tkn_counter=tkn_counter,
        cut_surplus=cut_surplus,
        always_with_question=always_with_question,
        no_prefix=no_prefix,
        rng=rng,
    )


def get_set_of_highlights_in_context_iterative_sent_gen(
    curr_instance,
    *args,
    **kwargs,
):
    return IterativeResultConverter().highlights_in_context(
        curr_instance,
        *args,
        **kwargs,
    )


def convert_iterative_sent_gen_to_pipeline_format(
    results,
    alignments_dict,
    *args,
    **kwargs,
):
    return IterativeResultConverter().convert(
        results,
        alignments_dict,
        *args,
        **kwargs,
    )


def main(args):
    persister = IterativeResultPersister(
        IterativePersistenceDependencies(
            save_results=save_results,
            get_token_usage=get_token_usage,
            write_json=atomic_write_json,
        )
    )
    application = IterativeSentenceGenerationApplication(
        IterativeApplicationDependencies(
            update_args=update_args,
            get_data=get_data,
            generate=iterative_sent_gen_prompting,
            get_token_counter=get_token_counter,
            reset_token_usage=reset_token_usage,
            convert_results=(
                convert_iterative_sent_gen_to_pipeline_format
            ),
            persister=persister,
            rng_factory=np.random.default_rng,
            log_info=logging.info,
            artifact_sha256=artifact_sha256,
        )
    )
    return application.run(args)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=(
            "path to json config file. Should come instead of all the other "
            "parameters"
        ),
    )
    argparser.add_argument(
        "--indir-alignments",
        type=str,
        default=None,
        help=(
            "path to json file with alignments (if nothing is passed - goes "
            "to default under data/{setting}/{split}.json)."
        ),
    )
    argparser.add_argument(
        "--indir-prompt",
        type=str,
        default=None,
        help=(
            "path to json file with the prompt structure and ICL examples "
            "(if nothing is passed - goes to default under "
            "prompts/{setting}.json)."
        ),
    )
    argparser.add_argument(
        "--setting",
        type=str,
        default=None,
        help="setting (MDS or LFQA)",
    )
    argparser.add_argument(
        "--split",
        type=str,
        default=None,
        help="data split (test or dev)",
    )
    argparser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=None,
        help="path to output csv.",
    )
    argparser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_DEFAULT,
        help="full provider model ID",
    )
    argparser.add_argument(
        "--n-demos",
        type=int,
        default=2,
        help="number of ICL examples (default 2)",
    )
    argparser.add_argument(
        "--num-retries",
        type=int,
        default=1,
        help="number of retries of running the model.",
    )
    argparser.add_argument(
        "--num-demo-changes",
        type=int,
        default=4,
        help=(
            "number of changing demos when the currently-chosen set of demos "
            "returns an ERROR."
        ),
    )
    argparser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="temperature of generation",
    )
    argparser.add_argument(
        "--rerun",
        action="store_true",
        default=False,
        help="if need to rerun on instances that had errors",
    )
    argparser.add_argument(
        "--rerun-path",
        type=str,
        default=None,
        help="path to rerun on (where the results are)",
    )
    argparser.add_argument(
        "--rerun-n-demos",
        type=int,
        default=None,
        help=(
            "new n_demos for rerun in cases when the current n_demos doesnt "
            "work."
        ),
    )
    argparser.add_argument(
        "--rerun-temperature",
        type=float,
        default=None,
        help=(
            "new temperature for rerun in cases when the current temperature "
            "doesnt work."
        ),
    )
    argparser.add_argument(
        "--debugging",
        action="store_true",
        default=False,
        help="if debugging mode.",
    )
    argparser.add_argument(
        "--merge-cross-sents-highlights",
        action="store_true",
        default=False,
        help=(
            "whether to merge consecutive highlights that span across "
            "several sentences."
        ),
    )
    argparser.add_argument(
        "--cut-surplus",
        action="store_true",
        default=False,
        help=(
            "whether to cut surplus text from prompts (everything after last "
            "highlight)."
        ),
    )
    argparser.add_argument(
        "--always-with-question",
        action="store_true",
        default=False,
        help="relevant for LFQA - whether to add the question",
    )
    argparser.add_argument(
        "--no-prefix",
        action="store_true",
        default=False,
        help="ablation study where the prefix is not add.",
    )
    argparser.add_argument(
        "--seed",
        type=int,
        default=20260728,
        help="seed used for demonstration sampling",
    )
    argparser.add_argument(
        "--prompt-token-budget",
        type=int,
        default=30000,
        help="maximum prompt size accepted by iterative prompt construction",
    )
    main(argparser.parse_args())
