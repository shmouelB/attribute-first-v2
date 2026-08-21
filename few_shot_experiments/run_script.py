"""Compatibility facade for the standard single-stage generation runner."""

import argparse
from collections.abc import Mapping
import logging
import os

from schemas import SUBTASK_SCHEMAS
from subtask_specific_utils import (
    construct_prompts,
    get_data,
    get_subtask_funcs,
    get_subtask_prompt_structures,
)
from utils import (
    SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS,
    artifact_sha256,
    atomic_write_json,
    atomic_write_jsonl,
    get_af_environment_flags,
    get_token_counter,
    get_token_usage,
    prompt_model,
    protocol_environment,
    remove_pipeline_artifact,
    save_results,
    stable_value_sha256,
    update_args,
)
from attribute_first.application.standard_pipeline import (
    StandardPipelineDependencies,
    StandardPipelineRunner,
)
from attribute_first.infrastructure import (
    CallableArtifactStore,
    CallableBatchGenerationGateway,
)
from attribute_first.artifacts.standard_run_artifacts import (
    DemonstrationDescriptorFactory,
    RerunPolicy,
    RerunProvenanceBuilder,
)
from attribute_first.stages.configuration import DEFAULT_GENERATION
from attribute_first.stages.registry import LegacyStageRegistryAdapter
from attribute_first.stages.standard_registry import (
    DEFAULT_STAGE_REGISTRY,
)


logging.basicConfig(level=logging.INFO)
_DEFAULT_GET_SUBTASK_FUNCS = get_subtask_funcs


def _effective_generation_settings(args):
    return RerunPolicy.effective_generation_settings(args)


def _load_rerun_source(args, outdir):
    return RerunPolicy(artifact_sha256).load(args, outdir)


def _demonstration_descriptors(used_demos):
    return DemonstrationDescriptorFactory(
        stable_value_sha256
    ).build(used_demos)


def _rerun_provenance(
    rerun_context,
    *,
    prompts,
    role_messages,
    used_demos,
    args,
    effective_n_demos,
    effective_temperature,
    environment_flags,
):
    return RerunProvenanceBuilder(
        stable_value_sha256,
        demonstration_descriptors=_demonstration_descriptors,
    ).build(
        rerun_context,
        prompts=prompts,
        role_messages=role_messages,
        used_demos=used_demos,
        args=args,
        effective_n_demos=effective_n_demos,
        effective_temperature=effective_temperature,
        environment_flags=environment_flags,
    )


def _runner_dependencies():
    """Capture every patchable legacy global immediately before execution."""
    stage_registry = DEFAULT_STAGE_REGISTRY
    if get_subtask_funcs is not _DEFAULT_GET_SUBTASK_FUNCS:
        stage_registry = LegacyStageRegistryAdapter(
            DEFAULT_STAGE_REGISTRY,
            get_subtask_funcs,
            SUBTASK_SCHEMAS,
        )
    return StandardPipelineDependencies(
        subtasks_without_given_highlights=(
            SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS
        ),
        effective_generation_settings=_effective_generation_settings,
        load_rerun_source=_load_rerun_source,
        build_rerun_provenance=_rerun_provenance,
        get_environment_flags=get_af_environment_flags,
        get_data=get_data,
        stage_registry=stage_registry,
        get_subtask_prompt_structures=get_subtask_prompt_structures,
        construct_prompts=construct_prompts,
        get_token_counter=get_token_counter,
        generation_gateway=CallableBatchGenerationGateway(prompt_model),
        artifact_store=CallableArtifactStore(
            write_json=atomic_write_json,
            write_jsonl=atomic_write_jsonl,
        ),
        save_results=save_results,
        remove_pipeline_artifact=remove_pipeline_artifact,
        artifact_sha256=artifact_sha256,
        get_token_usage=get_token_usage,
    )


def _run_with_effective_environment(args):
    return StandardPipelineRunner(_runner_dependencies()).run(args)


def _effective_protocol(args):
    """Merge universal role transport into an ad-hoc stage protocol."""

    protocol = getattr(args, "protocol", None)
    declared_flags = (
        protocol.get("environment_flags")
        if isinstance(protocol, Mapping)
        else None
    )
    use_roles = getattr(
        args,
        "use_roles",
        DEFAULT_GENERATION.use_roles,
    )
    if (
        use_roles is DEFAULT_GENERATION.use_roles
        and not (
            isinstance(declared_flags, Mapping)
            and "AF_USE_ROLES" in declared_flags
        )
        and "AF_USE_ROLES" in os.environ
    ):
        use_roles = get_af_environment_flags()["AF_USE_ROLES"]
    return DEFAULT_GENERATION.protocol_with_role_default(
        protocol,
        use_roles=use_roles,
    )


def main(args):
    if not args.config_file and (
        not args.setting or not args.subtask or not args.split
    ):
        raise Exception(
            "If no config file is passed, then must explicitly determine "
            "setting, subtask, and split."
        )
    if args.config_file:
        args = update_args(args)
    with protocol_environment(_effective_protocol(args)):
        return _run_with_effective_environment(args)


def _argument_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config-file",
        type=str,
        default=None,
        help="path to json config file. Should come instead of all the other parameters",
    )
    parser.add_argument("--split", type=str, default=None, help="data split (test or dev)")
    parser.add_argument("--setting", type=str, default=None, help="setting (MDS or LFQA)")
    parser.add_argument(
        "--subtask",
        type=str,
        default=None,
        help="subtask to run (content_selection, clustering, FiC, e2e_only_setting, ALCE)",
    )
    parser.add_argument(
        "--indir-alignments",
        type=str,
        default=None,
        help="path to json file with alignments (if nothing is passed - goes to default under data/{setting}/{split}.json).",
    )
    parser.add_argument(
        "--indir-prompt",
        type=str,
        default=None,
        help="path to json file with the prompt structure and ICL examples (if nothing is passed - goes to default under prompts/{setting}.json).",
    )
    parser.add_argument("-o", "--outdir", type=str, default=None, help="path to output csv.")
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
        help="Explicit fixed prompt-token budget; required by controlled configs.",
    )
    parser.add_argument("--n-demos", type=int, default=2, help="number of ICL examples (default 2)")
    parser.add_argument("--num-retries", type=int, default=1, help="number of retries of running the model.")
    parser.add_argument("--temperature", type=float, default=0.2, help="temperature of generation")
    parser.add_argument("--debugging", action="store_true", default=False, help="if debugging mode.")
    parser.add_argument(
        "--merge-cross-sents-highlights",
        action="store_true",
        default=False,
        help="whether to merge consecutive highlights that span across several sentences.",
    )
    parser.add_argument("--CoT", action="store_true", default=False, help="whether to use a CoT approach (relevant for FiC and clustering).")
    parser.add_argument(
        "--cut-surplus",
        action="store_true",
        default=False,
        help="whether to cut surplus text from prompts (in subtask with given highlights - everything after last highlight, and in tasks without - last prct_surplus sentences).",
    )
    parser.add_argument(
        "--prct-surplus",
        type=float,
        default=None,
        help="for subtasks without given highlights (e.g. content_selection, e2e_only_setting, or ALCE) - what percentage of top document sents to drop in cases when the prompts are too long.",
    )
    parser.add_argument("--always-with-question", action="store_true", default=False, help="relevant for LFQA - whether to always add the question (also to clustering and FiC)")
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
    parser.add_argument("--rerun", action="store_true", default=False, help="Re-run only instances that had ERROR outputs.")
    parser.add_argument(
        "--rerun-path",
        type=str,
        default=None,
        help="Required immutable parent results.json for --rerun; the derived --outdir must be new and distinct.",
    )
    parser.add_argument("--rerun-n-demos", type=int, default=None, help="Override n_demos for the rerun.")
    parser.add_argument("--rerun-temperature", type=float, default=None, help="Override temperature for the rerun.")
    parser.add_argument("--num-demo-changes", type=int, default=4, help="Number of demo changes on ERROR before giving up.")
    parser.add_argument("--max-examples", type=int, default=None, help="Cap total examples processed.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent model calls within a stage (1 = sequential). Bounded thread pool; cuts wall-clock ~Nx for I/O-bound API calls.",
    )
    parser.add_argument("--no-prefix", action="store_true", default=False, help="Ablation: omit the prefix.")
    parser.add_argument("--seed", type=int, default=20260728, help="Seed used for reproducible demonstration selection.")
    parser.add_argument("--dialogue-mode", action="store_true", default=False, help="Run pipeline as multi-turn chat (FiC only).")
    return parser


if __name__ == "__main__":
    main(_argument_parser().parse_args())
