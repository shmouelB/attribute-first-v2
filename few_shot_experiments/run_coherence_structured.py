"""Compatibility facade for the controlled coherence and MEGA variants.

The implementation lives in focused ``attribute_first`` services:

* stages build and execute the shared clustering -> reorder -> fusion plan;
* artifacts validate populations and preserve byte-exact provenance;
* the application runner coordinates one complete controlled cell.

This module intentionally retains the historical functions, constants, CLI,
and monkey-patch boundaries used by notebooks and tests.
"""

import argparse
import logging
from pathlib import Path
import sys
import time


EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_ROOT.parent
for import_root in (EXPERIMENT_ROOT, PROJECT_ROOT):
    import_root_text = str(import_root)
    if import_root_text not in sys.path:
        sys.path.insert(0, import_root_text)

from few_shot_experiments import schemas as schemas  # noqa: E402
from few_shot_experiments import utils as utils  # noqa: E402
from attribute_first.application.planned_pipeline import (  # noqa: E402
    PlannedPipelineDependencies,
    PlannedPipelineRunner,
)
from attribute_first.domain import (  # noqa: E402
    supported_derived_treatments,
    validate_supported_derived_pair,
)
from attribute_first.infrastructure import (  # noqa: E402
    CallableArtifactStore,
    CallableGenerationGateway,
)
from attribute_first.artifacts.population import (  # noqa: E402
    EXPECTED_TEST_POPULATIONS,
    PopulationLoader,
    read_jsonl_snapshot,
    validated_unique_ids,
)
from attribute_first.artifacts.provenance import (  # noqa: E402
    DEPENDENCY_MANIFEST_NAMES,
    EXPECTED_UPSTREAM_PIPELINE_CONFIGS,
    SOURCE_FILE_NAMES,
    ProvenanceBuilder,
    ProvenanceDependencies,
    ProvenanceRepository,
)
from attribute_first.artifacts.results import (  # noqa: E402
    PipelineResultBuilder,
)
from attribute_first.stages.planned import (  # noqa: E402
    CONTROLLED_STAGE_PROTOCOLS,
    LFQA_CLUSTER_INSTR,
    LFQA_FUSION_INSTR,
    LFQA_REORDER_INSTR,
    PROMPT_BUDGET_SCOPE,
    CLUSTER_INSTR,
    FUSION_INSTR,
    REORDER_INSTR,
    PlannedInstanceDependencies,
    PlannedInstanceRunner,
    ProtocolDefinition,
    ProtocolFactory,
    StageExecutionDependencies,
    StageExecutor,
    StageRequest,
    all_results_trace_usage,
    empty_stage_traces,
    parse_clusters,
    parse_reorder,
    source_upstream_failure,
    terminal_result,
    trace_usage_summary,
    validate_fusion_plan,
)
from attribute_first.stages.structured_fusion import (  # noqa: E402
    parse_structured_fusion,
)


logging.basicConfig(level=logging.INFO)
HS, HE = "<highlight_start>", "<highlight_end>"
MODEL_DEFAULT = "models/gemini-3-flash-preview"
FIC_COT_SCHEMA = schemas.FIC_COT_SCHEMA
CLUSTERING_SCHEMA = schemas.CLUSTERING_SCHEMA
CLUSTER_REORDER_SCHEMA = schemas.CLUSTER_REORDER_SCHEMA

gemini_call = utils.gemini_call
get_token_usage = utils.get_token_usage
reset_token_usage = utils.reset_token_usage
get_last_call_usage = utils.get_last_call_usage
get_last_call_metadata = utils.get_last_call_metadata
reset_last_call_usage = utils.reset_last_call_usage
IncompleteGenerationError = utils.IncompleteGenerationError


DERIVED_VARIANTS = supported_derived_treatments()


def save_results(output_dir, used_demonstrations, results, pipeline_results):
    """Patch-friendly compatibility boundary around the legacy serializer."""
    return utils.save_results(
        output_dir,
        used_demonstrations,
        results,
        pipeline_results,
    )


def _effective_protocol(model, *, setting="MDS"):
    """Return the complete downstream treatment shared by both variants."""
    definition = ProtocolDefinition(
        stage_protocols=CONTROLLED_STAGE_PROTOCOLS,
        response_schemas={
            "clustering": CLUSTERING_SCHEMA,
            "reorder": CLUSTER_REORDER_SCHEMA,
            "fusion": FIC_COT_SCHEMA,
        },
        prompt_budget_scope=PROMPT_BUDGET_SCOPE,
    )
    return ProtocolFactory(
        definition,
        utils.stable_value_sha256,
    ).build(model, setting=setting)


def _role_call(
    system,
    user,
    model,
    schema=None,
    max_out=8192,
    temperature=0.3,
):
    """Send one independent Gemini-native system/user role request."""
    return gemini_call(
        prompt=user,
        model_name=model,
        output_max_length=max_out,
        temperature=temperature,
        response_schema=schema,
        contents=[{"role": "user", "parts": [user]}],
        system_instruction=system,
    )


def _parse_clusters(output, highlight_count):
    return parse_clusters(output, highlight_count)


def _parse_reorder(output, cluster_count):
    return parse_reorder(output, cluster_count)


def _validate_fusion_plan(parsed, clusters):
    return validate_fusion_plan(parsed, clusters)


def _execute_stage(
    *,
    stage,
    system_instruction,
    user_message,
    model,
    response_schema,
    max_output_tokens,
    prompt_token_budget,
    prompt_budget_scope,
    temperature,
    num_retries,
    parser,
):
    """Execute one stage using dependencies captured from this facade."""
    gateway = CallableGenerationGateway(
        lambda request: _role_call(
            request.system_instruction,
            request.prompt,
            request.model_name,
            schema=request.response_schema,
            max_out=request.output_max_length,
            temperature=request.temperature,
        )
    )
    dependencies = StageExecutionDependencies(
        generation_gateway=gateway,
        stable_value_sha256=utils.stable_value_sha256,
        reset_last_call_usage=reset_last_call_usage,
        get_last_call_usage=get_last_call_usage,
        get_last_call_metadata=get_last_call_metadata,
        ensure_parseable_finish_reason=utils.ensure_parseable_finish_reason,
        incomplete_generation_error=IncompleteGenerationError,
        sleep=time.sleep,
        prompt_budget_scope=PROMPT_BUDGET_SCOPE,
    )
    request = StageRequest(
        stage=stage,
        system_instruction=system_instruction,
        user_message=user_message,
        model=model,
        response_schema=response_schema,
        max_output_tokens=max_output_tokens,
        prompt_token_budget=prompt_token_budget,
        prompt_budget_scope=prompt_budget_scope,
        temperature=temperature,
        num_retries=num_retries,
        parser=parser,
    )
    return StageExecutor(dependencies).execute(request)


def _terminal_result(stage, message, stage_traces, **metadata):
    return terminal_result(stage, message, stage_traces, **metadata)


def _source_upstream_failure(instance):
    return source_upstream_failure(instance)


def run_instance(inst, model, *, setting="MDS"):
    """Run the common planned downstream stages for one source instance."""
    dependencies = PlannedInstanceDependencies(
        effective_protocol=_effective_protocol,
        stable_value_sha256=utils.stable_value_sha256,
        execute_stage=_execute_stage,
        parse_clusters=_parse_clusters,
        parse_reorder=_parse_reorder,
        parse_structured_fusion=parse_structured_fusion,
        validate_fusion_plan=_validate_fusion_plan,
        terminal_result=_terminal_result,
        source_upstream_failure=_source_upstream_failure,
        clustering_schema=CLUSTERING_SCHEMA,
        reorder_schema=CLUSTER_REORDER_SCHEMA,
        fusion_schema=FIC_COT_SCHEMA,
    )
    return PlannedInstanceRunner(dependencies).run(
        inst,
        model,
        setting=setting,
    )


def _build_pipeline_format_results(source_instances, results):
    return PipelineResultBuilder().build(source_instances, results)


def _read_jsonl_snapshot(path, label):
    return read_jsonl_snapshot(path, label)


def _validated_unique_ids(rows, label):
    return validated_unique_ids(rows, label)


def _active_experiment_root():
    """Resolve the live facade location, including isolated archive fixtures."""
    return Path(__file__).resolve().parent


def _load_controlled_population(args):
    return PopulationLoader(
        _active_experiment_root(),
        utils.stable_value_sha256,
        EXPECTED_TEST_POPULATIONS,
    ).load(args)


def _provenance_repository():
    dependencies = ProvenanceDependencies(
        stable_value_sha256=utils.stable_value_sha256,
        artifact_sha256=utils.artifact_sha256,
        get_environment_flags=utils.get_af_environment_flags,
    )
    return ProvenanceRepository(
        _active_experiment_root(),
        dependencies,
        DERIVED_VARIANTS,
        SOURCE_FILE_NAMES,
        DEPENDENCY_MANIFEST_NAMES,
    )


def _prepare_output_directory(path):
    return _provenance_repository().prepare_output_directory(path)


def _load_upstream_provenance(args, input_path):
    return _provenance_repository().load_upstream_provenance(
        args,
        input_path,
    )


def _expected_upstream_contract(setting, variant):
    return _provenance_repository().expected_upstream_contract(
        setting,
        variant,
    )


def _resolve_recorded_path(value):
    return _provenance_repository().resolve_recorded_path(value)


def _validate_upstream_treatment(
    upstream_snapshot,
    *,
    variant,
    setting,
    split,
    model,
    input_path,
    population_reference_sha256=None,
    population_ids=None,
    expected_contract=None,
):
    return _provenance_repository().validate_upstream_treatment(
        upstream_snapshot,
        variant=variant,
        setting=setting,
        split=split,
        model=model,
        input_path=input_path,
        population_reference_sha256=population_reference_sha256,
        population_ids=population_ids,
        expected_contract=expected_contract,
    )


def _snapshot_provenance_payload(payload, outdir, relative_path):
    return _provenance_repository().snapshot_payload(
        payload,
        outdir,
        relative_path,
    )


def _capture_source_provenance(outdir):
    return _provenance_repository().capture_source(outdir)


def _build_pipeline_provenance(
    args,
    *,
    variant,
    protocol,
    population,
    upstream_snapshot,
    expected_contract,
    outdir,
):
    return ProvenanceBuilder(_provenance_repository()).build(
        args,
        variant=variant,
        protocol=protocol,
        population=population,
        upstream_snapshot=upstream_snapshot,
        expected_contract=expected_contract,
        outdir=outdir,
    )


def _empty_stage_traces():
    return empty_stage_traces()


def _trace_usage_summary(stage_traces):
    return trace_usage_summary(stage_traces)


def _all_results_trace_usage(results):
    return all_results_trace_usage(results)


def _application_dependencies():
    """Capture patchable facade globals immediately before one run."""
    return PlannedPipelineDependencies(
        derived_variants=DERIVED_VARIANTS,
        controlled_stage_protocols=CONTROLLED_STAGE_PROTOCOLS,
        effective_protocol=_effective_protocol,
        stable_value_sha256=utils.stable_value_sha256,
        load_population=_load_controlled_population,
        load_upstream_provenance=_load_upstream_provenance,
        expected_upstream_contract=_expected_upstream_contract,
        validate_upstream_treatment=_validate_upstream_treatment,
        prepare_output_directory=_prepare_output_directory,
        build_pipeline_provenance=_build_pipeline_provenance,
        get_environment_flags=utils.get_af_environment_flags,
        artifact_store=CallableArtifactStore(
            write_json=utils.atomic_write_json,
            write_jsonl=utils.atomic_write_jsonl,
        ),
        reset_token_usage=reset_token_usage,
        source_upstream_failure=_source_upstream_failure,
        terminal_result=_terminal_result,
        empty_stage_traces=_empty_stage_traces,
        trace_usage_summary=_trace_usage_summary,
        run_instance=run_instance,
        build_pipeline_format_results=_build_pipeline_format_results,
        save_results=save_results,
        get_token_usage=get_token_usage,
        all_results_trace_usage=_all_results_trace_usage,
        summarize_response_metadata=utils.summarize_response_metadata,
        artifact_sha256=utils.artifact_sha256,
    )


def main(args):
    """Run one controlled derived cell through the application service."""
    return PlannedPipelineRunner(_application_dependencies()).run(args)


class _DerivedArgumentParser(argparse.ArgumentParser):
    """Reject catalog-invalid variant/setting pairs at the CLI boundary."""

    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        try:
            validate_supported_derived_pair(
                parsed.variant,
                parsed.setting,
                treatments=DERIVED_VARIANTS,
            )
        except ValueError as exc:
            self.error(str(exc))
        return parsed


def _argument_parser():
    parser = _DerivedArgumentParser()
    parser.add_argument(
        "--variant",
        required=True,
        choices=sorted(DERIVED_VARIANTS),
    )
    parser.add_argument(
        "--setting",
        required=True,
        choices=sorted(CONTROLLED_STAGE_PROTOCOLS),
    )
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument(
        "--cs",
        required=True,
        help=(
            "upstream pipeline_format JSONL: few-shot content selection for "
            "coherence, zero-shot ambiguity-highlight output for mega"
        ),
    )
    parser.add_argument(
        "--upstream-provenance",
        required=True,
        help=(
            "pipeline_provenance.json for the upstream full pipeline; used "
            "to verify the few-shot or zero-shot treatment"
        ),
    )
    parser.add_argument("-o", "--outdir", required=True)
    parser.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        choices=[MODEL_DEFAULT],
        help="fixed controlled generation model",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(_argument_parser().parse_args())
