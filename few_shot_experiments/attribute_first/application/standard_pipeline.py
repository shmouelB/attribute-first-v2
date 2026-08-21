"""Application service for one standard, schema-aware generation stage."""

from dataclasses import dataclass
import logging
import os
from typing import Any, Callable

from ..artifacts.output_directory import OutputDirectoryClaim
from ..artifacts.standard_run_artifacts import StandardResultAssembler
from ..ports import (
    ArtifactStore,
    BatchGenerationGateway,
    BatchGenerationRequest,
)
from ..stages.registry import (
    ResolvedStageProtocol,
    StageProtocolRegistry,
)
from ..stages.configuration import DEFAULT_GENERATION


@dataclass(frozen=True)
class StandardPipelineDependencies:
    """All legacy globals captured by ``run_script`` at call time."""

    subtasks_without_given_highlights: object
    effective_generation_settings: Callable[[Any], tuple]
    load_rerun_source: Callable[[Any, str], Any]
    build_rerun_provenance: Callable[..., dict]
    get_environment_flags: Callable[[], dict]
    get_data: Callable[[Any], tuple]
    stage_registry: StageProtocolRegistry
    get_subtask_prompt_structures: Callable[..., Any]
    construct_prompts: Callable[..., tuple]
    get_token_counter: Callable[..., Any]
    generation_gateway: BatchGenerationGateway
    artifact_store: ArtifactStore
    save_results: Callable[[str, list, dict, Any], None]
    remove_pipeline_artifact: Callable[[str], None]
    artifact_sha256: Callable[[Any], str]
    get_token_usage: Callable[[], dict]


@dataclass
class StandardRunState:
    """Prepared input and prompt state for one standard stage."""

    args: Any
    outdir: str
    rerun_context: Any
    effective_n_demos: int
    effective_temperature: float
    environment_flags: dict
    all_alignments: list
    upstream_failures: dict
    stage_protocol: ResolvedStageProtocol
    used_demos: list
    prompts: dict
    additional_data: dict
    role_messages: dict
    existing_results: dict


class StandardPipelineRunner:
    """Coordinate prompt construction, generation, and durable artifacts."""

    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.result_assembler = StandardResultAssembler()

    @staticmethod
    def _output_directory(args):
        cot_suffix = "_CoT" if args.CoT else ""
        merge_suffix = (
            "_merged_cross_sents_sep"
            if args.merge_cross_sents_highlights
            else ""
        )
        return (
            args.outdir
            if args.outdir
            else (
                f"results/{args.split}/{args.setting}/{args.subtask}"
                f"{cot_suffix}{merge_suffix}"
            )
        )

    @staticmethod
    def _rerun_alignments(rerun_context, all_alignments):
        if rerun_context is None:
            return all_alignments
        error_ids = set(rerun_context["error_ids"])
        selected = [
            item
            for item in all_alignments
            if str(item.get("unique_id")) in error_ids
        ]
        available_ids = {
            str(item.get("unique_id")) for item in selected
        }
        missing_ids = sorted(error_ids - available_ids)
        if missing_ids:
            raise ValueError(
                "rerun parent contains ERROR IDs absent from the current "
                "input: "
                + ", ".join(missing_ids)
            )
        return selected

    def _write_args_snapshot(
        self,
        args,
        outdir,
        effective_n_demos,
        effective_temperature,
        environment_flags,
    ):
        snapshot = {
            key: value
            for key, value in args.__dict__.items()
            if not key.startswith("_")
        }
        snapshot.update(
            {
                "effective_n_demos": effective_n_demos,
                "effective_temperature": effective_temperature,
                "environment_flags": environment_flags,
            }
        )
        self.dependencies.artifact_store.write_json(
            os.path.join(outdir, "args.json"),
            snapshot,
        )

    def _subtask_protocol(self, args, prompt_dict, structured_output):
        protocol = self.dependencies.stage_registry.resolve(
            args.subtask,
            structured_output=structured_output,
        )
        prompt_subtask = protocol.prompt_subtask_name
        if (
            args.cut_surplus
            and prompt_subtask
            in self.dependencies.subtasks_without_given_highlights
            and not args.prct_surplus
        ):
            logging.error(
                "when passing --cut-surplus with the subtask %s - you need "
                "to also pass --prct-surplus",
                args.subtask,
            )
            raise SystemExit(1)
        prompt_details = (
            self.dependencies.get_subtask_prompt_structures(
                prompt_dict=prompt_dict,
                setting=args.setting,
                subtask=prompt_subtask,
                CoT=args.CoT,
                always_with_question=args.always_with_question,
                structured_output=structured_output,
            )
        )
        return protocol, prompt_details

    def _construct_prompts(
        self,
        args,
        prompt_dict,
        active_alignments,
        prompt_details,
        effective_n_demos,
        stage_protocol,
    ):
        return self.dependencies.construct_prompts(
            prompt_dict=prompt_dict,
            alignments_dict=active_alignments,
            n_demos=effective_n_demos,
            debugging=args.debugging,
            merge_cross_sents_highlights=(
                args.merge_cross_sents_highlights
            ),
            specific_prompt_details=prompt_details,
            tkn_counter=self.dependencies.get_token_counter(
                args.model_name,
                getattr(args, "prompt_token_budget", None),
            ),
            no_highlights=(
                stage_protocol.prompt_subtask_name
                in self.dependencies.subtasks_without_given_highlights
            ),
            cut_surplus=args.cut_surplus,
            prct_surplus=args.prct_surplus,
            seed=getattr(args, "seed", None),
        )

    @staticmethod
    def _filter_rerun_prompts(
        rerun_context,
        prompts,
        role_messages,
    ):
        if rerun_context is None:
            return prompts, role_messages
        retry_ids = set(rerun_context["error_ids"])
        filtered_prompts = {
            str(key): value
            for key, value in prompts.items()
            if str(key) in retry_ids
        }
        filtered_roles = {
            str(key): value
            for key, value in role_messages.items()
            if str(key) in filtered_prompts
        }
        logging.info(
            "Rerun mode: %s instances with ERROR outputs to retry.",
            len(filtered_prompts),
        )
        return filtered_prompts, filtered_roles

    def _prepare(self, args):
        outdir = self._output_directory(args)
        rerun_context = self.dependencies.load_rerun_source(args, outdir)
        effective_n_demos, effective_temperature = (
            self.dependencies.effective_generation_settings(args)
        )
        environment_flags = self.dependencies.get_environment_flags()
        prompt_dict, all_alignments = self.dependencies.get_data(args)
        selected_alignments = self._rerun_alignments(
            rerun_context,
            all_alignments,
        )
        active_alignments, upstream_failures = (
            self.result_assembler.partition_upstream_failures(
                selected_alignments
            )
        )
        structured_output = DEFAULT_GENERATION.structured_output_for(args)
        stage_protocol, prompt_details = (
            self._subtask_protocol(
                args,
                prompt_dict,
                structured_output,
            )
        )

        logging.info("saving results to %s", outdir)
        pipeline_root = getattr(
            args,
            "_pipeline_run_root",
            None,
        )
        if pipeline_root is None:
            outdir = str(
                OutputDirectoryClaim.claim(
                    outdir,
                    owner="standard-generation-stage-v1",
                )
            )
        else:
            outdir = str(
                OutputDirectoryClaim.prepare_child(
                    outdir,
                    owner_root=pipeline_root,
                )
            )
        self._write_args_snapshot(
            args,
            outdir,
            effective_n_demos,
            effective_temperature,
            environment_flags,
        )
        (
            used_demos,
            prompts,
            additional_data,
            role_messages,
        ) = self._construct_prompts(
            args,
            prompt_dict,
            active_alignments,
            prompt_details,
            effective_n_demos,
            stage_protocol,
        )
        prompts, role_messages = self._filter_rerun_prompts(
            rerun_context,
            prompts,
            role_messages,
        )
        existing_results = (
            rerun_context["existing_results"]
            if rerun_context is not None
            else {}
        )
        return StandardRunState(
            args=args,
            outdir=outdir,
            rerun_context=rerun_context,
            effective_n_demos=effective_n_demos,
            effective_temperature=effective_temperature,
            environment_flags=environment_flags,
            all_alignments=all_alignments,
            upstream_failures=upstream_failures,
            stage_protocol=stage_protocol,
            used_demos=used_demos,
            prompts=prompts,
            additional_data=additional_data,
            role_messages=role_messages,
            existing_results=existing_results,
        )

    def _generate(self, state):
        args = state.args
        responses = self.dependencies.generation_gateway.generate_batch(
            BatchGenerationRequest(
                prompts=state.prompts,
                model_name=args.model_name,
                parse_response=state.stage_protocol.parser,
                num_retries=args.num_retries,
                temperature=state.effective_temperature,
                response_schema=state.stage_protocol.response_schema,
                output_max_length=getattr(
                    args,
                    "output_max_length",
                    4096,
                ),
                concurrency=getattr(args, "concurrency", 1),
                role_messages=state.role_messages,
            )
        )
        self.result_assembler.add_upstream_failures(
            responses,
            state.additional_data,
            state.upstream_failures,
        )
        return responses

    def _convert(self, state, final_results):
        converter = state.stage_protocol.converter
        if not converter:
            return None
        try:
            pipeline_results = converter(
                final_results,
                state.all_alignments,
                structured_output=(
                    state.stage_protocol.structured_output
                ),
            )
            if pipeline_results is None:
                raise RuntimeError(
                    "pipeline conversion returned no result"
                )
            return pipeline_results
        except Exception:
            logging.exception(
                "The conversion to pipeline format failed; any stale "
                "pipeline artifact will be removed."
            )
            self.dependencies.remove_pipeline_artifact(state.outdir)
            raise

    def _write_rerun_provenance(self, state):
        context = state.rerun_context
        if context is None:
            return
        if (
            self.dependencies.artifact_sha256(context["source_path"])
            != context["source_sha256"]
        ):
            raise RuntimeError(
                "rerun parent changed while the derived run executed"
            )
        provenance = self.dependencies.build_rerun_provenance(
            context,
            prompts=state.prompts,
            role_messages=state.role_messages,
            used_demos=state.used_demos,
            args=state.args,
            effective_n_demos=state.effective_n_demos,
            effective_temperature=state.effective_temperature,
            environment_flags=state.environment_flags,
        )
        self.dependencies.artifact_store.write_json(
            os.path.join(state.outdir, "rerun_provenance.json"),
            provenance,
        )

    def _write_token_usage(self, state):
        usage = self.dependencies.get_token_usage()
        usage["subtask"] = state.args.subtask
        usage["model"] = state.args.model_name
        self.dependencies.artifact_store.write_json(
            os.path.join(state.outdir, "token_usage.json"),
            usage,
        )

    def _persist_generation_evidence(self, state, final_results):
        """Commit provider evidence before fallible pipeline conversion."""

        self.dependencies.artifact_store.write_json(
            os.path.join(state.outdir, "results.json"),
            final_results,
        )
        self._write_token_usage(state)
        self.dependencies.save_results(
            state.outdir,
            state.used_demos,
            final_results,
            None,
        )
        self._write_rerun_provenance(state)

    def _publish_pipeline_results(self, state, pipeline_results):
        if pipeline_results is None:
            return
        self.dependencies.artifact_store.write_jsonl(
            os.path.join(
                state.outdir,
                "pipeline_format_results.json",
            ),
            pipeline_results,
        )

    def run(self, args):
        state = self._prepare(args)
        responses = self._generate(state)
        final_results = self.result_assembler.assemble(
            state.existing_results,
            responses,
            state.additional_data,
            state.all_alignments,
            state.args,
        )
        self._persist_generation_evidence(state, final_results)
        pipeline_results = self._convert(state, final_results)
        self._publish_pipeline_results(state, pipeline_results)
