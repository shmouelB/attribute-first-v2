"""Application service for one controlled coherence or MEGA run."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Callable, Mapping

from ..ports import ArtifactStore


@dataclass(frozen=True)
class PlannedPipelineDependencies:
    """All mutable boundaries captured by the legacy compatibility facade."""

    derived_variants: Mapping[str, Mapping[str, Any]]
    controlled_stage_protocols: Mapping[str, Any]
    effective_protocol: Callable[..., dict]
    stable_value_sha256: Callable[[Any], str]
    load_population: Callable[[Any], dict]
    load_upstream_provenance: Callable[[Any, Path], dict]
    expected_upstream_contract: Callable[[str, str], dict]
    validate_upstream_treatment: Callable[..., Any]
    prepare_output_directory: Callable[[Any], Path]
    build_pipeline_provenance: Callable[..., dict]
    get_environment_flags: Callable[[], dict]
    artifact_store: ArtifactStore
    reset_token_usage: Callable[[], None]
    source_upstream_failure: Callable[[dict], Any]
    terminal_result: Callable[..., dict]
    empty_stage_traces: Callable[[], dict]
    trace_usage_summary: Callable[[Any], dict]
    run_instance: Callable[..., dict]
    build_pipeline_format_results: Callable[[list, dict], list]
    save_results: Callable[[str, list, dict, list], None]
    get_token_usage: Callable[[], dict]
    all_results_trace_usage: Callable[[dict], dict]
    summarize_response_metadata: Callable[[list], dict]
    artifact_sha256: Callable[[Path], str]


@dataclass
class PlannedRunContext:
    """State shared across the small application-service phases."""

    args: Any
    variant: str
    canonical_cell_id: str
    protocol: dict
    protocol_sha256: str
    population: dict
    upstream_snapshot: dict
    expected_contract: dict
    outdir: Path
    provenance: dict
    args_snapshot: dict


class PlannedPipelineRunner:
    """Coordinate fixed inputs, instance work, and immutable artifacts."""

    OUTPUT_ARTIFACTS = (
        ".controlled_run_claim",
        "args.json",
        "pipeline_provenance.json",
        "results.json",
        "results.csv",
        "pipeline_format_results.json",
        "response_metadata.json",
        "token_usage.json",
        "used_demonstrations.json",
    )

    def __init__(self, dependencies):
        self.dependencies = dependencies

    def _validate_args(self, args):
        variant = getattr(args, "variant", None)
        if variant not in self.dependencies.derived_variants:
            raise ValueError(
                "variant must be one of "
                + ", ".join(sorted(self.dependencies.derived_variants))
            )
        if args.setting not in self.dependencies.controlled_stage_protocols:
            raise ValueError(
                "setting must be one of "
                + ", ".join(
                    sorted(self.dependencies.controlled_stage_protocols)
                )
            )
        treatment = self.dependencies.derived_variants[variant]
        upstream_by_setting = treatment.get(
            "upstream_canonical_id_by_setting"
        )
        if (
            not isinstance(upstream_by_setting, Mapping)
            or args.setting not in upstream_by_setting
        ):
            raise ValueError(
                f"derived variant {variant!r} is not supported for "
                f"setting {args.setting!r}"
            )
        if args.split != "test":
            raise ValueError(
                "controlled derived variants require split=test"
            )
        if type(args.concurrency) is not int or args.concurrency < 1:
            raise ValueError("concurrency must be a positive integer")
        return variant

    def _args_snapshot(
        self,
        args,
        variant,
        protocol,
        protocol_sha256,
        population,
        upstream_snapshot,
    ):
        treatment = self.dependencies.derived_variants[variant]
        canonical_cell_id = (
            f"{args.setting.lower()}."
            f"{treatment['canonical_factor_id']}"
        )
        return {
            "variant": variant,
            "cell_id": f"{args.setting}.{variant}",
            "canonical_cell_id": canonical_cell_id,
            "setting": args.setting,
            "split": args.split,
            "input_path": str(population["input"]["path"]),
            "input_stage": treatment["input_stage"],
            "upstream_treatment": treatment["upstream_treatment"],
            "population_reference_path": str(
                population["reference"]["path"]
            ),
            "upstream_provenance_path": str(upstream_snapshot["path"]),
            "model": args.model,
            "concurrency": args.concurrency,
            "max_examples": population["max_examples"],
            "effective_stage_parameters": protocol["stage_parameters"],
            "prompt_budget_scope": protocol["prompt_budget_scope"],
            "effective_randomness": protocol["randomness"],
            "effective_protocol_sha256": protocol_sha256,
            "environment_flags": (
                self.dependencies.get_environment_flags()
            ),
            "append_policy": "new_or_empty_directory_only",
        }

    def _prepare(self, args):
        variant = self._validate_args(args)
        protocol = self.dependencies.effective_protocol(
            args.model,
            setting=args.setting,
        )
        protocol_sha256 = self.dependencies.stable_value_sha256(protocol)
        population = self.dependencies.load_population(args)
        input_path = population["input"]["path"]
        upstream_snapshot = self.dependencies.load_upstream_provenance(
            args,
            input_path,
        )
        expected_contract = self.dependencies.expected_upstream_contract(
            args.setting,
            variant,
        )
        matched_contract = self.dependencies.validate_upstream_treatment(
            upstream_snapshot,
            variant=variant,
            setting=args.setting,
            split=args.split,
            model=args.model,
            input_path=input_path,
            population_reference_sha256=population["reference"]["sha256"],
            population_ids=population["reference_ids"],
            expected_contract=expected_contract,
        )
        if matched_contract is not None:
            expected_contract = matched_contract
        outdir = self.dependencies.prepare_output_directory(args.outdir)
        provenance = self.dependencies.build_pipeline_provenance(
            args,
            variant=variant,
            protocol=protocol,
            population=population,
            upstream_snapshot=upstream_snapshot,
            expected_contract=expected_contract,
            outdir=outdir,
        )
        args_snapshot = self._args_snapshot(
            args,
            variant,
            protocol,
            protocol_sha256,
            population,
            upstream_snapshot,
        )
        context = PlannedRunContext(
            args=args,
            variant=variant,
            canonical_cell_id=args_snapshot["canonical_cell_id"],
            protocol=protocol,
            protocol_sha256=protocol_sha256,
            population=population,
            upstream_snapshot=upstream_snapshot,
            expected_contract=expected_contract,
            outdir=outdir,
            provenance=provenance,
            args_snapshot=args_snapshot,
        )
        self._write_initial_state(context)
        return context

    def _write_initial_state(self, context):
        write_json = self.dependencies.artifact_store.write_json
        write_json(context.outdir / "args.json", context.args_snapshot)
        write_json(
            context.outdir / "pipeline_provenance.json",
            context.provenance,
        )
        write_json(
            context.outdir / "run_state.json",
            {
                "status": "in_progress",
                "variant": context.variant,
                "cell_id": (
                    f"{context.args.setting}.{context.variant}"
                ),
                "canonical_cell_id": context.canonical_cell_id,
                "protocol_sha256": context.protocol_sha256,
            },
        )

    def _gold_by_id(self, context):
        return {
            unique_id: context.population["dataset_by_id"][
                unique_id
            ].get("response")
            for unique_id in context.population["selected_ids"]
        }

    def _terminal_upstream_result(
        self,
        instance,
        upstream_failure,
        context,
        gold,
    ):
        unique_id = instance["unique_id"]
        if upstream_failure is None:
            upstream_failure = {
                "skipped_reason": "no_highlights",
                "upstream_error": (
                    "ERROR - upstream stage skipped: no_highlights"
                ),
            }
        upstream_error = upstream_failure["upstream_error"]
        result = self.dependencies.terminal_result(
            "upstream",
            upstream_error.removeprefix("ERROR - ").strip(),
            self.dependencies.empty_stage_traces(),
            protocol_sha256=context.protocol_sha256,
        )
        result["upstream_skipped_reason"] = upstream_failure[
            "skipped_reason"
        ]
        result["unique_id"] = unique_id
        result["gold_summary"] = gold[unique_id]
        result["usage_summary"] = self.dependencies.trace_usage_summary(
            result["plan_metadata"]["stage_traces"]
        )
        return result

    def _partition_instances(self, context, gold):
        completed = {}
        active = []
        for instance in context.population["selected_rows"]:
            unique_id = instance["unique_id"]
            failure = self.dependencies.source_upstream_failure(instance)
            highlights = (
                instance.get("set_of_highlights_in_context") or []
            )
            if failure is not None or not highlights:
                completed[unique_id] = self._terminal_upstream_result(
                    instance,
                    failure,
                    context,
                    gold,
                )
            else:
                active.append(instance)
        return completed, active

    def _work(self, instance, context, gold):
        unique_id = instance["unique_id"]
        result = self.dependencies.run_instance(
            instance,
            context.args.model,
            setting=context.args.setting,
        )
        result["unique_id"] = unique_id
        result["gold_summary"] = gold[unique_id]
        result["usage_summary"] = self.dependencies.trace_usage_summary(
            result.get("plan_metadata", {}).get("stage_traces")
        )
        return unique_id, result

    def _execute_instances(self, context):
        gold = self._gold_by_id(context)
        completed, active = self._partition_instances(context, gold)
        with ThreadPoolExecutor(
            max_workers=context.args.concurrency
        ) as executor:
            futures = [
                executor.submit(self._work, instance, context, gold)
                for instance in active
            ]
            for index, future in enumerate(
                as_completed(futures),
                start=1,
            ):
                unique_id, result = future.result()
                completed[unique_id] = result
                if index % 5 == 0:
                    logging.info(
                        "  %s/%s active examples done",
                        index,
                        len(futures),
                    )
        return self._ordered_results(context, completed, gold)

    def _ordered_results(self, context, completed, gold):
        results = {}
        for unique_id in context.population["selected_ids"]:
            result = completed.get(unique_id)
            if result is None:
                result = self.dependencies.terminal_result(
                    "runtime",
                    "missing worker result",
                    self.dependencies.empty_stage_traces(),
                    protocol_sha256=context.protocol_sha256,
                )
                result["unique_id"] = unique_id
                result["gold_summary"] = gold[unique_id]
                result[
                    "usage_summary"
                ] = self.dependencies.trace_usage_summary(
                    result["plan_metadata"]["stage_traces"]
                )
            results[unique_id] = result
        return results

    def _persist_primary_results(self, context, results):
        source_rows = context.population["selected_rows"]
        pipeline_results = (
            self.dependencies.build_pipeline_format_results(
                source_rows,
                results,
            )
        )
        selected_ids = context.population["selected_ids"]
        if [row["unique_id"] for row in pipeline_results] != selected_ids:
            raise AssertionError(
                "pipeline output order differs from input order"
            )
        if list(results) != selected_ids:
            raise AssertionError(
                "results output order differs from input order"
            )
        self.dependencies.save_results(
            str(context.outdir),
            [],
            results,
            pipeline_results,
        )

    def _usage_report(self, context, results):
        usage = self.dependencies.get_token_usage()
        trace_usage = self.dependencies.all_results_trace_usage(results)
        provider_fields = (
            "provider_total",
            "provider_total_calls",
        )
        provider_totals_declared = any(
            field in usage for field in provider_fields
        )
        provider_total_fields_complete = all(
            field in usage for field in provider_fields
        )
        reconciliation = {
            "prompt_matches": (
                usage["prompt"] == trace_usage["prompt_token_count"]
            ),
            "completion_matches": (
                usage["completion"]
                == trace_usage["candidates_token_count"]
            ),
            "cached_matches": (
                usage["cached"]
                == trace_usage["cached_content_token_count"]
            ),
            "calls_match": (
                usage["calls"] == trace_usage["usage_record_count"]
            ),
            "response_usage_records_match": (
                trace_usage["response_count"]
                == trace_usage["usage_record_count"]
            ),
            "provider_total_fields_complete": (
                not provider_totals_declared
                or provider_total_fields_complete
            ),
            "provider_total_matches": (
                not provider_totals_declared
                or (
                    provider_total_fields_complete
                    and usage["provider_total"]
                    == trace_usage["total_token_count"]
                )
            ),
            "provider_total_calls_match": (
                not provider_totals_declared
                or (
                    provider_total_fields_complete
                    and usage["provider_total_calls"]
                    == trace_usage[
                        "total_token_count_record_count"
                    ]
                )
            ),
        }
        reconciliation["all_aggregate_counters_match"] = all(
            reconciliation.values()
        )
        errors = sum(
            str(result.get("final_output", "")).startswith("ERROR")
            for result in results.values()
        )
        usage.update(
            {
                "subtask": "controlled_derived_variant",
                "variant": context.variant,
                "cell_id": (
                    f"{context.args.setting}.{context.variant}"
                ),
                "canonical_cell_id": context.canonical_cell_id,
                "model": context.args.model,
                "protocol_sha256": context.protocol_sha256,
                "population_total": len(results),
                "population_valid": len(results) - errors,
                "population_errors": errors,
                "per_call_trace_usage": trace_usage,
                "per_call_trace_reconciliation": reconciliation,
            }
        )
        return usage, errors

    def _write_usage(self, context, usage):
        write_json = self.dependencies.artifact_store.write_json
        write_json(context.outdir / "token_usage.json", usage)
        reconciliation = usage["per_call_trace_reconciliation"]
        if reconciliation["all_aggregate_counters_match"]:
            return
        write_json(
            context.outdir / "run_state.json",
            {
                "status": "invalid_usage_reconciliation",
                "variant": context.variant,
                "cell_id": (
                    f"{context.args.setting}.{context.variant}"
                ),
                "canonical_cell_id": context.canonical_cell_id,
                "protocol_sha256": context.protocol_sha256,
                "per_call_trace_usage": usage["per_call_trace_usage"],
                "per_call_trace_reconciliation": reconciliation,
            },
        )
        raise RuntimeError(
            "aggregate token usage does not reconcile with per-call traces"
        )

    @staticmethod
    def _all_attempts(results):
        attempts = []
        for result in results.values():
            stage_traces = result.get("plan_metadata", {}).get(
                "stage_traces",
                {},
            )
            if not isinstance(stage_traces, dict):
                continue
            for stage_attempts in stage_traces.values():
                if isinstance(stage_attempts, list):
                    attempts.extend(stage_attempts)
        return attempts

    def _complete_provenance(self, context, results):
        metadata = self.dependencies.summarize_response_metadata(
            self._all_attempts(results)
        )
        write_json = self.dependencies.artifact_store.write_json
        write_json(context.outdir / "response_metadata.json", metadata)
        context.args_snapshot["observed_response_metadata"] = metadata
        write_json(context.outdir / "args.json", context.args_snapshot)
        context.provenance["observed_response_metadata"] = metadata
        context.provenance["run"]["completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        write_json(
            context.outdir / "pipeline_provenance.json",
            context.provenance,
        )
        return metadata

    def _summary(
        self,
        context,
        results,
        usage,
        errors,
        response_metadata,
    ):
        output_artifacts = {
            name: self.dependencies.artifact_sha256(
                context.outdir / name
            )
            for name in self.OUTPUT_ARTIFACTS
        }
        return {
            "variant": context.variant,
            "cell_id": f"{context.args.setting}.{context.variant}",
            "canonical_cell_id": context.canonical_cell_id,
            "setting": context.args.setting,
            "split": context.args.split,
            "population": {
                "total": len(results),
                "valid": len(results) - errors,
                "errors": errors,
                "unique_ids_order_sha256": (
                    self.dependencies.stable_value_sha256(list(results))
                ),
            },
            "protocol_sha256": context.protocol_sha256,
            "provenance_snapshot_bundle_sha256": context.provenance[
                "provenance_snapshot"
            ]["bundle_sha256"],
            "observed_response_metadata": response_metadata,
            "token_usage": usage,
            "output_artifacts": output_artifacts,
        }

    def run(self, args):
        context = self._prepare(args)
        self.dependencies.reset_token_usage()
        results = self._execute_instances(context)
        self._persist_primary_results(context, results)
        usage, errors = self._usage_report(context, results)
        self._write_usage(context, usage)
        metadata = self._complete_provenance(context, results)
        summary = self._summary(
            context,
            results,
            usage,
            errors,
            metadata,
        )
        self.dependencies.artifact_store.write_json(
            context.outdir / "run_state.json",
            {"status": "completed", **summary},
        )
        logging.info(
            "[%s] %s: %s/%s valid | prompt=%s completion=%s "
            "cached=%s calls=%s",
            context.variant,
            context.args.setting,
            len(results) - errors,
            len(results),
            usage["prompt"],
            usage["completion"],
            usage["cached"],
            usage["calls"],
        )
        return summary
