"""Single source of truth for archived generation-code bundles."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationSourceCatalog:
    """Immutable inventories for standard and controlled-derived runs."""

    shared: tuple[str, ...]
    standard_entrypoints: tuple[str, ...]
    derived_entrypoints: tuple[str, ...]
    legacy_derived: tuple[str, ...]

    @property
    def standard(self) -> tuple[str, ...]:
        return self._unique(self.standard_entrypoints + self.shared)

    @property
    def derived(self) -> tuple[str, ...]:
        return self._unique(self.derived_entrypoints + self.shared)

    @staticmethod
    def _unique(names: tuple[str, ...]) -> tuple[str, ...]:
        if len(names) != len(set(names)):
            raise ValueError("source inventories must not contain duplicates")
        return names


SHARED_GENERATION_SOURCE_FILES = (
    "__init__.py",
    "utils.py",
    "schemas.py",
    "response_parsers.py",
    "attribute_first/__init__.py",
    "attribute_first/compatibility/__init__.py",
    "attribute_first/compatibility/legacy_names.py",
    "attribute_first/compatibility/stage_aliases.py",
    "attribute_first/domain/__init__.py",
    "attribute_first/domain/catalog.py",
    "attribute_first/domain/enums.py",
    "attribute_first/domain/identifiers.py",
    "attribute_first/domain/models.py",
    "attribute_first/domain/policies.py",
    "attribute_first/ports/__init__.py",
    "attribute_first/ports/artifact_store.py",
    "attribute_first/ports/model_gateway.py",
    "attribute_first/infrastructure/__init__.py",
    "attribute_first/infrastructure/json_artifact_store.py",
    "attribute_first/infrastructure/model_gateways.py",
    "attribute_first/prompting/__init__.py",
    "attribute_first/prompting/highlights.py",
    "attribute_first/prompting/templates.py",
    "attribute_first/runtime/__init__.py",
    "attribute_first/runtime/attempts.py",
    "attribute_first/runtime/conversation.py",
    "attribute_first/runtime/environment.py",
    "attribute_first/runtime/retry_policy.py",
    "attribute_first/runtime/system_resources.py",
    "attribute_first/runtime/usage.py",
    "attribute_first/campaign/__init__.py",
    "attribute_first/campaign/catalog.py",
    "attribute_first/campaign/cli.py",
    "attribute_first/campaign/manifest_accounting.py",
    "attribute_first/campaign/manifest_cell_accounting.py",
    "attribute_first/campaign/manifest_contract.py",
    "attribute_first/campaign/manifest_shared_graph.py",
    "attribute_first/campaign/manifest_writer.py",
    "attribute_first/campaign/population.py",
    "attribute_first/campaign/runner.py",
    "attribute_first/campaign/scheduler.py",
    "attribute_first/application/__init__.py",
    "attribute_first/application/dialogue_cache.py",
    "attribute_first/application/dialogue_content_selection.py",
    "attribute_first/application/dialogue_ambiguity.py",
    "attribute_first/application/dialogue_demonstrations.py",
    "attribute_first/application/dialogue_dependencies.py",
    "attribute_first/application/dialogue_fusion.py",
    "attribute_first/application/dialogue_pipeline.py",
    "attribute_first/application/dialogue_persistence.py",
    "attribute_first/application/dialogue_preparation.py",
    "attribute_first/application/dialogue_sessions.py",
    "attribute_first/application/dialogue_shared_content_selection.py",
    "attribute_first/application/dialogue_stage_prompts.py",
    "attribute_first/application/dialogue_state.py",
    "attribute_first/application/dialogue_turns.py",
    "attribute_first/application/iterative_application.py",
    "attribute_first/application/iterative_results.py",
    "attribute_first/application/iterative_sentence_generation.py",
    "attribute_first/application/pipeline_application.py",
    "attribute_first/application/planned_pipeline.py",
    "attribute_first/application/protocol.py",
    "attribute_first/application/sequential_contracts.py",
    "attribute_first/application/sequential_dialogue.py",
    "attribute_first/application/sequential_instance.py",
    "attribute_first/application/sequential_pipeline.py",
    "attribute_first/application/sequential_results.py",
    "attribute_first/application/standard_pipeline.py",
    "attribute_first/stages/__init__.py",
    "attribute_first/stages/configuration.py",
    "attribute_first/stages/fic_canonical_highlights.py",
    "attribute_first/stages/planned.py",
    "attribute_first/stages/registry.py",
    "attribute_first/stages/structured_fusion.py",
    "attribute_first/stages/structured_highlights.py",
    "attribute_first/artifacts/__init__.py",
    "attribute_first/artifacts/dialogue_usage.py",
    "attribute_first/artifacts/output_directory.py",
    "attribute_first/artifacts/pipeline_artifacts.py",
    "attribute_first/artifacts/population.py",
    "attribute_first/artifacts/provenance.py",
    "attribute_first/artifacts/results.py",
    "attribute_first/artifacts/shared_content_runtime.py",
    "attribute_first/artifacts/shared_content_selection.py",
    "attribute_first/artifacts/source_catalog.py",
    "attribute_first/artifacts/standard_provenance.py",
    "attribute_first/artifacts/standard_run_artifacts.py",
    "attribute_first/artifacts/token_usage.py",
    "attribute_first/validation/__init__.py",
    "attribute_first/validation/artifacts.py",
    "attribute_first/validation/catalog_identity.py",
    "attribute_first/validation/core.py",
    "attribute_first/validation/derived_metrics.py",
    "attribute_first/validation/derived_population.py",
    "attribute_first/validation/derived_run.py",
    "attribute_first/validation/derived_support.py",
    "attribute_first/validation/dialogue.py",
    "attribute_first/validation/independent.py",
    "attribute_first/validation/metrics.py",
    "attribute_first/validation/provenance.py",
    "attribute_first/validation/shared_provenance.py",
    "attribute_first/validation/shared_trace.py",
    "attribute_first/validation/standard_run.py",
    "attribute_first/validation/terminal.py",
    "evaluation/calc_rouge_l.py",
)

STANDARD_ENTRYPOINT_FILES = (
    "run_all_variants.sh",
    "campaign_manifest.py",
    "run_full_pipeline.py",
    "run_script.py",
    "run_iterative_sentence_generation.py",
    "prompt_utils.py",
    "pipeline_converters.py",
    "subtask_specific_utils.py",
    "attribute_first/stages/standard_registry.py",
)

V4_DERIVED_ENTRYPOINT_FILES = (
    "run_all_variants.sh",
    "campaign_manifest.py",
    "run_coherence_structured.py",
    "validate_controlled_derived_run.py",
)
DERIVED_ENTRYPOINT_FILES = (
    *V4_DERIVED_ENTRYPOINT_FILES,
    "attribute_first/domain/evidence_designed.py",
)

LEGACY_DERIVED_SOURCE_FILES = (
    "run_coherence_structured.py",
    "validate_controlled_derived_run.py",
    "run_all_variants.sh",
    "campaign_manifest.py",
    "utils.py",
    "schemas.py",
    "response_parsers.py",
    "evaluation/calc_rouge_l.py",
)

DEFAULT_SOURCE_CATALOG = GenerationSourceCatalog(
    shared=SHARED_GENERATION_SOURCE_FILES,
    standard_entrypoints=STANDARD_ENTRYPOINT_FILES,
    derived_entrypoints=DERIVED_ENTRYPOINT_FILES,
    legacy_derived=LEGACY_DERIVED_SOURCE_FILES,
)
V4_SOURCE_CATALOG = GenerationSourceCatalog(
    shared=SHARED_GENERATION_SOURCE_FILES,
    standard_entrypoints=STANDARD_ENTRYPOINT_FILES,
    derived_entrypoints=V4_DERIVED_ENTRYPOINT_FILES,
    legacy_derived=LEGACY_DERIVED_SOURCE_FILES,
)
STANDARD_SOURCE_FILE_NAMES = DEFAULT_SOURCE_CATALOG.standard
DERIVED_SOURCE_FILE_NAMES = DEFAULT_SOURCE_CATALOG.derived
V4_DERIVED_SOURCE_FILE_NAMES = V4_SOURCE_CATALOG.derived


__all__ = [
    "DEFAULT_SOURCE_CATALOG",
    "DERIVED_SOURCE_FILE_NAMES",
    "GenerationSourceCatalog",
    "LEGACY_DERIVED_SOURCE_FILES",
    "SHARED_GENERATION_SOURCE_FILES",
    "STANDARD_SOURCE_FILE_NAMES",
    "V4_DERIVED_SOURCE_FILE_NAMES",
    "V4_SOURCE_CATALOG",
]
