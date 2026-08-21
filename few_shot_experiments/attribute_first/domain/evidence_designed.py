"""Append-only experiment treatments selected from observed evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .catalog import (
    CampaignCellKind,
    CatalogCell,
    ExperimentCatalog,
)
from .enums import (
    ContextAugmentation,
    Dataset,
    DemonstrationMode,
    GenerationStrategy,
    TransportMode,
)
from .models import PipelineFactors


PLANNED_ZERO_SHOT_WITHOUT_CONTEXT = (
    "planned_zero_shot_without_context"
)
LFQA_PLANNED_FEW_SHOT_CONTEXT_INDEPENDENT = (
    "planned_few_shot_context_augmented_independent"
)
LFQA_FEW_SHOT_CONTEXT_COHERENCE_DIALOGUE = (
    "planned_few_shot_context_augmented_dialogue"
)
_LEGACY_PIPELINE_CONFIG_BY_CANONICAL = {
    "direct_few_shot.json": "fullcot_g3flash_pipeline.json",
    "direct_zero_shot.json": "fullcot_zeroshot_g3flash_pipeline.json",
    "direct_zero_shot_context_augmented.json": (
        "decontext_zeroshot_g3flash_pipeline.json"
    ),
    "direct_few_shot_context_augmented.json": (
        "decontext_g3flash_pipeline.json"
    ),
}


class EvidenceDesignedCatalog:
    """Construct isolated catalogs without changing the frozen campaign."""

    @classmethod
    def planned_zero_shot_without_context(
        cls,
    ) -> ExperimentCatalog:
        """Return the MDS zero-shot upstream and planned downstream pair."""

        direct_factors = PipelineFactors(
            generation=GenerationStrategy.DIRECT,
            demonstrations=DemonstrationMode.ZERO_SHOT,
            context_augmentation=ContextAugmentation.DISABLED,
            transport=TransportMode.INDEPENDENT,
        )
        planned_factors = PipelineFactors(
            generation=GenerationStrategy.PLANNED,
            demonstrations=DemonstrationMode.ZERO_SHOT,
            context_augmentation=ContextAugmentation.DISABLED,
            transport=TransportMode.INDEPENDENT,
        )
        upstream = CatalogCell(
            setting=Dataset.MDS,
            kind=CampaignCellKind.STANDARD,
            name="full_CoT_pipeline_zeroshot",
            canonical_id=f"mds.{direct_factors.canonical_id}",
            factors=direct_factors,
            pipeline_config=(
                "configs/controlled/test/MDS/pipelines/"
                "direct_zero_shot.json"
            ),
            upstream_canonical_id=None,
        )
        derived = CatalogCell(
            setting=Dataset.MDS,
            kind=CampaignCellKind.DERIVED,
            name=PLANNED_ZERO_SHOT_WITHOUT_CONTEXT,
            canonical_id=f"mds.{planned_factors.canonical_id}",
            factors=planned_factors,
            pipeline_config=None,
            upstream_canonical_id=upstream.canonical_id,
        )
        return ExperimentCatalog(cells=(upstream, derived))

    @classmethod
    def planned_few_shot_context_augmented_independent(
        cls,
    ) -> ExperimentCatalog:
        """Return the LFQA few-shot AH upstream and planned downstream."""

        direct_factors = PipelineFactors(
            generation=GenerationStrategy.DIRECT,
            demonstrations=DemonstrationMode.FEW_SHOT,
            context_augmentation=ContextAugmentation.ENABLED,
            transport=TransportMode.INDEPENDENT,
        )
        planned_factors = PipelineFactors(
            generation=GenerationStrategy.PLANNED,
            demonstrations=DemonstrationMode.FEW_SHOT,
            context_augmentation=ContextAugmentation.ENABLED,
            transport=TransportMode.INDEPENDENT,
        )
        upstream = CatalogCell(
            setting=Dataset.LFQA,
            kind=CampaignCellKind.STANDARD,
            name="full_decontextualization_CoT_pipeline_structured",
            canonical_id=f"lfqa.{direct_factors.canonical_id}",
            factors=direct_factors,
            pipeline_config=(
                "configs/controlled/test/LFQA/pipelines/"
                "direct_few_shot_context_augmented.json"
            ),
            upstream_canonical_id=None,
        )
        derived = CatalogCell(
            setting=Dataset.LFQA,
            kind=CampaignCellKind.DERIVED,
            name=LFQA_PLANNED_FEW_SHOT_CONTEXT_INDEPENDENT,
            canonical_id=f"lfqa.{planned_factors.canonical_id}",
            factors=planned_factors,
            pipeline_config=None,
            upstream_canonical_id=upstream.canonical_id,
        )
        return ExperimentCatalog(cells=(upstream, derived))

@dataclass(frozen=True, slots=True)
class EvidenceDesignedDialogueSpec:
    """One append-only composite that is not part of the frozen campaign."""

    name: str
    setting: Dataset
    factors: PipelineFactors
    pipeline_config: str
    stage_order: tuple[str, ...]
    concurrency: int

    @classmethod
    def lfqa_few_shot_context_coherence(
        cls,
    ) -> "EvidenceDesignedDialogueSpec":
        """Return the user-requested LFQA MEGA composition."""

        return cls(
            name=LFQA_FEW_SHOT_CONTEXT_COHERENCE_DIALOGUE,
            setting=Dataset.LFQA,
            factors=PipelineFactors(
                generation=GenerationStrategy.PLANNED,
                demonstrations=DemonstrationMode.FEW_SHOT,
                context_augmentation=ContextAugmentation.ENABLED,
                transport=TransportMode.DIALOGUE,
            ),
            pipeline_config=(
                "configs/evidence_designed/test/LFQA/pipelines/"
                "planned_few_shot_context_augmented_dialogue.json"
            ),
            stage_order=(
                "content_selection",
                "ambiguity_highlight",
                "clustering",
                "reorder",
                "fusion_in_context",
            ),
            concurrency=1,
        )


@dataclass(frozen=True, slots=True)
class EvidenceDesignedStandardCell:
    """Validator identity for a standard run outside the frozen campaign."""

    setting: Dataset
    kind: CampaignCellKind
    name: str
    canonical_id: str
    factors: PipelineFactors
    pipeline_config: str


def supported_experiment_catalogs() -> tuple[ExperimentCatalog, ...]:
    """Return frozen and append-only catalogs as separate identities."""

    return (
        ExperimentCatalog.controlled(),
        EvidenceDesignedCatalog.planned_zero_shot_without_context(),
        (
            EvidenceDesignedCatalog
            .planned_few_shot_context_augmented_independent()
        ),
    )


def supported_standard_validation_cells() -> tuple[
    EvidenceDesignedStandardCell,
    ...,
]:
    """Return append-only standard identities without mutating v4."""

    spec = EvidenceDesignedDialogueSpec.lfqa_few_shot_context_coherence()
    return (
        EvidenceDesignedStandardCell(
            setting=spec.setting,
            kind=CampaignCellKind.STANDARD,
            name=spec.name,
            canonical_id=(
                f"{spec.setting.value.lower()}.{spec.factors.canonical_id}"
            ),
            factors=spec.factors,
            pipeline_config=spec.pipeline_config,
        ),
    )


def supported_derived_cells() -> tuple[CatalogCell, ...]:
    """Return every supported derived identity without merging catalogs."""

    return tuple(
        cell
        for catalog in supported_experiment_catalogs()
        for cell in catalog.cells
        if cell.kind is CampaignCellKind.DERIVED
    )


def supported_derived_treatments() -> dict[str, dict[str, object]]:
    """Project supported cells into the planned-runner vocabulary."""

    treatments: dict[str, dict[str, object]] = {}
    for catalog in supported_experiment_catalogs():
        derived_cells = tuple(
            cell
            for cell in catalog.cells
            if cell.kind is CampaignCellKind.DERIVED
        )
        for cell in derived_cells:
            has_context = (
                cell.factors.context_augmentation
                is ContextAugmentation.ENABLED
            )
            demonstration_prefix = (
                "few_shot"
                if cell.factors.demonstrations
                is DemonstrationMode.FEW_SHOT
                else "zero_shot"
            )
            treatment: dict[str, object] = {
                "canonical_factor_id": cell.factors.canonical_id,
                "factors": {
                    "generation": cell.factors.generation.value,
                    "demonstrations": (
                        cell.factors.demonstrations.value
                    ),
                    "context_augmentation": (
                        cell.factors.context_augmentation.value
                    ),
                    "transport": cell.factors.transport.value,
                },
                "demonstration_mode": (
                    cell.factors.demonstrations.value
                ),
                "context_augmentation": has_context,
                "input_stage": (
                    "ambiguity_highlight"
                    if has_context
                    else "content_selection"
                ),
                "upstream_treatment": (
                    demonstration_prefix
                    + "_content_selection"
                    + (
                        "_then_ambiguity_highlight"
                        if has_context
                        else ""
                    )
                ),
            }
            previous = treatments.get(cell.name)
            if previous is None:
                previous = {
                    **treatment,
                    "upstream_canonical_id_by_setting": {},
                }
                treatments[cell.name] = previous
            elif {
                key: value
                for key, value in previous.items()
                if key != "upstream_canonical_id_by_setting"
            } != treatment:
                raise ValueError(
                    "derived catalog treatment differs between datasets "
                    f"for {cell.name!r}"
                )
            upstream_by_setting = previous[
                "upstream_canonical_id_by_setting"
            ]
            if not isinstance(upstream_by_setting, dict):
                raise TypeError("derived treatment registry is malformed")
            upstream_by_setting[cell.setting.value] = (
                cell.upstream_canonical_id
            )
    return treatments


def supported_derived_validation_specs() -> dict[str, dict[str, object]]:
    """Add validator-only config aliases without changing provenance."""

    specs = supported_derived_treatments()
    for catalog in supported_experiment_catalogs():
        by_canonical_id = {
            cell.canonical_id: cell for cell in catalog.cells
        }
        for cell in catalog.cells:
            if cell.kind is not CampaignCellKind.DERIVED:
                continue
            upstream = by_canonical_id[cell.upstream_canonical_id]
            pipeline_config = upstream.pipeline_config
            if not isinstance(pipeline_config, str):
                raise TypeError(
                    "derived upstream pipeline config is missing"
                )
            pipeline_config_name = pipeline_config.rsplit("/", 1)[-1]
            legacy_pipeline_config = (
                _LEGACY_PIPELINE_CONFIG_BY_CANONICAL.get(
                    pipeline_config_name
                )
            )
            if legacy_pipeline_config is None:
                raise ValueError(
                    "derived upstream has no historical config identity: "
                    f"{pipeline_config_name}"
                )
            spec = specs[cell.name]
            previous_pipeline = spec.get("pipeline_config")
            previous_legacy = spec.get("legacy_pipeline_config")
            if previous_pipeline not in {None, pipeline_config_name} or (
                previous_legacy not in {None, legacy_pipeline_config}
            ):
                raise ValueError(
                    "derived validator config differs between datasets "
                    f"for {cell.name!r}"
                )
            spec["pipeline_config"] = pipeline_config_name
            spec["legacy_pipeline_config"] = legacy_pipeline_config
    return specs


def resolve_supported_derived_cell(
    setting: str,
    variant: str,
) -> CatalogCell | None:
    """Resolve one derived identity across the isolated catalogs."""

    matches = tuple(
        cell
        for cell in supported_derived_cells()
        if cell.setting.value == setting and cell.name == variant
    )
    if len(matches) > 1:
        raise ValueError(
            f"duplicate supported derived identity: {setting}.{variant}"
        )
    return matches[0] if matches else None


def validate_supported_derived_pair(
    variant: object,
    setting: object,
    *,
    treatments: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Reject a variant outside its catalog-declared dataset support."""

    registry = (
        supported_derived_treatments()
        if treatments is None
        else treatments
    )
    treatment = registry.get(variant) if isinstance(variant, str) else None
    if treatment is None:
        raise ValueError(f"unsupported derived variant: {variant}")
    upstream_by_setting = treatment.get(
        "upstream_canonical_id_by_setting"
    )
    if (
        not isinstance(setting, str)
        or not isinstance(upstream_by_setting, dict)
        or setting not in upstream_by_setting
    ):
        raise ValueError(
            f"derived variant {variant!r} is not supported for setting "
            f"{setting!r}"
        )


__all__ = [
    "EvidenceDesignedCatalog",
    "EvidenceDesignedDialogueSpec",
    "LFQA_FEW_SHOT_CONTEXT_COHERENCE_DIALOGUE",
    "LFQA_PLANNED_FEW_SHOT_CONTEXT_INDEPENDENT",
    "PLANNED_ZERO_SHOT_WITHOUT_CONTEXT",
    "resolve_supported_derived_cell",
    "supported_derived_cells",
    "supported_derived_treatments",
    "supported_derived_validation_specs",
    "supported_experiment_catalogs",
    "validate_supported_derived_pair",
]
