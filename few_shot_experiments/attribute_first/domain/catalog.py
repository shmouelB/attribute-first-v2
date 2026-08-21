"""Typed declaration of the controlled sixteen-cell campaign."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from .enums import (
    ContextAugmentation,
    Dataset,
    DemonstrationMode,
    GenerationStrategy,
    TransportMode,
)
from .models import PipelineFactors


class CampaignCellKind(str, Enum):
    """How a campaign cell obtains its generation input."""

    STANDARD = "standard"
    DERIVED = "derived"


@dataclass(frozen=True, slots=True)
class _PipelineConfigurationKey:
    """Scientific factors encoded by one controlled pipeline file."""

    generation: GenerationStrategy
    demonstrations: DemonstrationMode
    context_augmentation: ContextAugmentation

    @classmethod
    def from_factors(
        cls,
        factors: PipelineFactors,
    ) -> "_PipelineConfigurationKey":
        return cls(
            generation=factors.generation,
            demonstrations=factors.demonstrations,
            context_augmentation=factors.context_augmentation,
        )


class _ControlledPipelineConfigRegistry:
    """Derive the only valid controlled config path for standard factors."""

    _STEMS = MappingProxyType(
        {
            _PipelineConfigurationKey(
                GenerationStrategy.DIRECT,
                DemonstrationMode.ZERO_SHOT,
                ContextAugmentation.DISABLED,
            ): "direct_zero_shot",
            _PipelineConfigurationKey(
                GenerationStrategy.DIRECT,
                DemonstrationMode.ZERO_SHOT,
                ContextAugmentation.ENABLED,
            ): "direct_zero_shot_context_augmented",
            _PipelineConfigurationKey(
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.DISABLED,
            ): "direct_few_shot",
            _PipelineConfigurationKey(
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.ENABLED,
            ): "direct_few_shot_context_augmented",
        }
    )

    @classmethod
    def path_for(
        cls,
        setting: Dataset,
        factors: PipelineFactors,
    ) -> str:
        """Return the config path whose contents represent the given factors."""

        if not isinstance(setting, Dataset):
            raise ValueError("setting must be a Dataset")
        if not isinstance(factors, PipelineFactors):
            raise ValueError("factors must be PipelineFactors")
        key = _PipelineConfigurationKey.from_factors(factors)
        try:
            stem = cls._STEMS[key]
        except KeyError as exc:
            raise ValueError(
                "no controlled standard pipeline config represents factors "
                f"{factors.canonical_id!r}"
            ) from exc
        return (
            f"configs/controlled/test/{setting.value}/pipelines/"
            f"{stem}.json"
        )


@dataclass(frozen=True, slots=True)
class CatalogCell:
    """One catalog entry with legacy persistence and canonical identities."""

    setting: Dataset
    kind: CampaignCellKind
    name: str
    canonical_id: str
    factors: PipelineFactors
    pipeline_config: str | None
    upstream_canonical_id: str | None
    content_selection_source_canonical_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.setting, Dataset):
            raise ValueError("setting must be a Dataset")
        if not isinstance(self.kind, CampaignCellKind):
            raise ValueError("kind must be a CampaignCellKind")
        if (
            not isinstance(self.name, str)
            or not self.name
            or self.name.strip() != self.name
        ):
            raise ValueError("name must be a non-empty legacy output alias")
        if not isinstance(self.factors, PipelineFactors):
            raise ValueError("factors must be PipelineFactors")
        expected_id = (
            f"{self.setting.value.lower()}."
            f"{self.factors.canonical_id}"
        )
        if self.canonical_id != expected_id:
            raise ValueError(
                f"canonical_id must equal {expected_id!r}"
            )
        if self.kind is CampaignCellKind.STANDARD:
            if (
                not isinstance(self.pipeline_config, str)
                or not self.pipeline_config
                or self.pipeline_config.strip()
                != self.pipeline_config
            ):
                raise ValueError(
                    "standard cells require a pipeline_config"
                )
            expected_config = _ControlledPipelineConfigRegistry.path_for(
                self.setting,
                self.factors,
            )
            if self.pipeline_config != expected_config:
                raise ValueError(
                    "pipeline_config is incompatible with factors: "
                    f"expected {expected_config!r}, got "
                    f"{self.pipeline_config!r}"
                )
            if self.upstream_canonical_id is not None:
                raise ValueError(
                    "standard cells cannot declare an upstream cell"
                )
            source_id = self.content_selection_source_canonical_id
            if source_id is not None:
                if (
                    not isinstance(source_id, str)
                    or not source_id
                    or source_id.strip() != source_id
                ):
                    raise ValueError(
                        "content-selection source must be a non-empty "
                        "canonical ID"
                    )
                if not source_id.startswith(
                    f"{self.setting.value.lower()}."
                ):
                    raise ValueError(
                        "content-selection consumer and source must "
                        "share a setting"
                    )
                if source_id == self.canonical_id:
                    raise ValueError(
                        "standard cell cannot reuse content selection "
                        "from itself"
                    )
        else:
            if (
                self.factors.generation
                is not GenerationStrategy.PLANNED
                or self.factors.transport
                is not TransportMode.INDEPENDENT
            ):
                raise ValueError(
                    "derived cells require planned generation with "
                    "independent transport"
                )
            if self.pipeline_config is not None:
                raise ValueError(
                    "derived cells cannot declare a pipeline_config"
                )
            if (
                not isinstance(self.upstream_canonical_id, str)
                or not self.upstream_canonical_id
                or self.upstream_canonical_id.strip()
                != self.upstream_canonical_id
            ):
                raise ValueError(
                    "derived cells require an upstream canonical ID"
                )
            if not self.upstream_canonical_id.startswith(
                f"{self.setting.value.lower()}."
            ):
                raise ValueError(
                    "derived cell and upstream cell must share a setting"
                )
            if self.upstream_canonical_id == self.canonical_id:
                raise ValueError(
                    "derived cell cannot use itself as its upstream"
                )
            if self.content_selection_source_canonical_id is not None:
                raise ValueError(
                    "derived cells cannot declare a content-selection "
                    "source"
                )


@dataclass(frozen=True, slots=True)
class SharedContentSelectionPair:
    """One exact producer-consumer edge for a shared CS realization."""

    producer: CatalogCell
    consumer: CatalogCell

    def __post_init__(self) -> None:
        if not isinstance(self.producer, CatalogCell):
            raise ValueError("producer must be a CatalogCell")
        if not isinstance(self.consumer, CatalogCell):
            raise ValueError("consumer must be a CatalogCell")
        producer = self.producer
        consumer = self.consumer
        if (
            producer.kind is not CampaignCellKind.STANDARD
            or consumer.kind is not CampaignCellKind.STANDARD
        ):
            raise ValueError(
                "shared content selection requires two standard cells"
            )
        if (
            consumer.content_selection_source_canonical_id
            != producer.canonical_id
        ):
            raise ValueError(
                "consumer does not reference the declared producer"
            )
        if producer.content_selection_source_canonical_id is not None:
            raise ValueError(
                "shared content-selection producer cannot be a consumer"
            )
        if producer.setting is not consumer.setting:
            raise ValueError(
                "shared content-selection pair must share a setting"
            )
        if (
            producer.factors.generation
            is not GenerationStrategy.DIRECT
            or consumer.factors.generation
            is not GenerationStrategy.DIRECT
        ):
            raise ValueError(
                "shared content selection requires direct pipelines"
            )
        if (
            producer.factors.demonstrations
            is not consumer.factors.demonstrations
            or producer.factors.transport
            is not consumer.factors.transport
        ):
            raise ValueError(
                "shared content-selection pair changes an experimental "
                "factor other than context augmentation"
            )
        if (
            producer.factors.context_augmentation
            is not ContextAugmentation.DISABLED
            or consumer.factors.context_augmentation
            is not ContextAugmentation.ENABLED
        ):
            raise ValueError(
                "shared content selection must connect baseline to its "
                "context-augmented treatment"
            )


@dataclass(frozen=True, slots=True)
class ExperimentCatalog:
    """Immutable ordered collection of predeclared experiment cells."""

    cells: tuple[CatalogCell, ...]

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if not cells or any(
            not isinstance(cell, CatalogCell) for cell in cells
        ):
            raise ValueError("cells must contain CatalogCell values")
        object.__setattr__(self, "cells", cells)

        legacy_keys = tuple(
            (cell.setting, cell.kind, cell.name) for cell in cells
        )
        canonical_ids = tuple(cell.canonical_id for cell in cells)
        if len(legacy_keys) != len(set(legacy_keys)):
            raise ValueError("catalog contains duplicate legacy cell keys")
        if len(canonical_ids) != len(set(canonical_ids)):
            raise ValueError("catalog contains duplicate canonical IDs")
        by_canonical_id = {
            cell.canonical_id: cell for cell in cells
        }
        position_by_canonical_id = {
            cell.canonical_id: index
            for index, cell in enumerate(cells)
        }
        for cell in cells:
            if cell.kind is CampaignCellKind.DERIVED:
                upstream = by_canonical_id.get(
                    cell.upstream_canonical_id
                )
                if (
                    upstream is None
                    or upstream.kind is not CampaignCellKind.STANDARD
                ):
                    raise ValueError(
                        f"derived cell {cell.canonical_id!r} must "
                        "reference one standard catalog cell"
                    )
                expected_upstream_factors = PipelineFactors(
                    generation=GenerationStrategy.DIRECT,
                    demonstrations=cell.factors.demonstrations,
                    context_augmentation=(
                        cell.factors.context_augmentation
                    ),
                    transport=TransportMode.INDEPENDENT,
                )
                if upstream.factors != expected_upstream_factors:
                    raise ValueError(
                        f"derived cell {cell.canonical_id!r} upstream "
                        "factors do not match its declared treatment"
                    )
                if (
                    position_by_canonical_id[upstream.canonical_id]
                    >= position_by_canonical_id[cell.canonical_id]
                ):
                    raise ValueError(
                        f"derived cell {cell.canonical_id!r} must "
                        "appear after its standard upstream cell"
                    )
            source_id = (
                cell.content_selection_source_canonical_id
            )
            if source_id is None:
                continue
            source = by_canonical_id.get(source_id)
            if source is None:
                raise ValueError(
                    f"standard cell {cell.canonical_id!r} must "
                    "reference one catalog content-selection source"
                )
            SharedContentSelectionPair(
                producer=source,
                consumer=cell,
            )
            if (
                position_by_canonical_id[source.canonical_id]
                >= position_by_canonical_id[cell.canonical_id]
            ):
                raise ValueError(
                    f"content-selection source {source.canonical_id!r} "
                    f"must appear before consumer {cell.canonical_id!r}"
                )

    @classmethod
    def controlled(cls) -> "ExperimentCatalog":
        """Return the fixed campaign in the legacy launch order."""

        cells = tuple(
            cell
            for setting in (Dataset.MDS, Dataset.LFQA)
            for cell in _controlled_setting_cells(setting)
        )
        if len(cells) != 16:
            raise AssertionError(
                "controlled campaign must declare exactly 16 cells"
            )
        catalog = cls(cells=cells)
        if len(catalog.shared_content_selection_pairs) != 6:
            raise AssertionError(
                "controlled campaign must declare exactly six shared "
                "content-selection pairs"
            )
        return catalog

    @property
    def shared_content_selection_pairs(
        self,
    ) -> tuple[SharedContentSelectionPair, ...]:
        """Return immutable shared-CS edges in consumer catalog order."""

        by_canonical_id = {
            cell.canonical_id: cell for cell in self.cells
        }
        return tuple(
            SharedContentSelectionPair(
                producer=by_canonical_id[source_id],
                consumer=cell,
            )
            for cell in self.cells
            if (
                source_id
                := cell.content_selection_source_canonical_id
            )
        )


@dataclass(frozen=True, slots=True)
class CampaignPlan:
    """Execution-ready immutable projection of an experiment catalog."""

    cells: tuple[CatalogCell, ...]

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if not cells or any(
            not isinstance(cell, CatalogCell) for cell in cells
        ):
            raise ValueError("cells must contain CatalogCell values")
        object.__setattr__(self, "cells", cells)

    @classmethod
    def from_catalog(
        cls,
        catalog: ExperimentCatalog,
    ) -> "CampaignPlan":
        """Preserve the catalog's declared order without reinterpretation."""

        if not isinstance(catalog, ExperimentCatalog):
            raise TypeError("catalog must be an ExperimentCatalog")
        return cls(cells=catalog.cells)


def _factors(
    *,
    generation: GenerationStrategy,
    demonstrations: DemonstrationMode,
    context: ContextAugmentation,
    transport: TransportMode,
) -> PipelineFactors:
    return PipelineFactors(
        generation=generation,
        demonstrations=demonstrations,
        context_augmentation=context,
        transport=transport,
    )


def _catalog_cell(
    *,
    setting: Dataset,
    kind: CampaignCellKind,
    name: str,
    factors: PipelineFactors,
    upstream_canonical_id: str | None,
    content_selection_source_canonical_id: str | None = None,
) -> CatalogCell:
    return CatalogCell(
        setting=setting,
        kind=kind,
        name=name,
        canonical_id=(
            f"{setting.value.lower()}.{factors.canonical_id}"
        ),
        factors=factors,
        pipeline_config=(
            _ControlledPipelineConfigRegistry.path_for(setting, factors)
            if kind is CampaignCellKind.STANDARD
            else None
        ),
        upstream_canonical_id=upstream_canonical_id,
        content_selection_source_canonical_id=(
            content_selection_source_canonical_id
        ),
    )


def _canonical_id(
    setting: Dataset,
    factors: PipelineFactors,
) -> str:
    return f"{setting.value.lower()}.{factors.canonical_id}"


def _controlled_setting_cells(
    setting: Dataset,
) -> tuple[CatalogCell, ...]:
    direct_zs = _factors(
        generation=GenerationStrategy.DIRECT,
        demonstrations=DemonstrationMode.ZERO_SHOT,
        context=ContextAugmentation.DISABLED,
        transport=TransportMode.INDEPENDENT,
    )
    direct_zs_context = _factors(
        generation=GenerationStrategy.DIRECT,
        demonstrations=DemonstrationMode.ZERO_SHOT,
        context=ContextAugmentation.ENABLED,
        transport=TransportMode.INDEPENDENT,
    )
    direct_fs_dialogue = _factors(
        generation=GenerationStrategy.DIRECT,
        demonstrations=DemonstrationMode.FEW_SHOT,
        context=ContextAugmentation.DISABLED,
        transport=TransportMode.DIALOGUE,
    )
    direct_fs_context_dialogue = _factors(
        generation=GenerationStrategy.DIRECT,
        demonstrations=DemonstrationMode.FEW_SHOT,
        context=ContextAugmentation.ENABLED,
        transport=TransportMode.DIALOGUE,
    )
    direct_fs = _factors(
        generation=GenerationStrategy.DIRECT,
        demonstrations=DemonstrationMode.FEW_SHOT,
        context=ContextAugmentation.DISABLED,
        transport=TransportMode.INDEPENDENT,
    )
    direct_fs_context = _factors(
        generation=GenerationStrategy.DIRECT,
        demonstrations=DemonstrationMode.FEW_SHOT,
        context=ContextAugmentation.ENABLED,
        transport=TransportMode.INDEPENDENT,
    )
    planned_fs = _factors(
        generation=GenerationStrategy.PLANNED,
        demonstrations=DemonstrationMode.FEW_SHOT,
        context=ContextAugmentation.DISABLED,
        transport=TransportMode.INDEPENDENT,
    )
    planned_zs_context = _factors(
        generation=GenerationStrategy.PLANNED,
        demonstrations=DemonstrationMode.ZERO_SHOT,
        context=ContextAugmentation.ENABLED,
        transport=TransportMode.INDEPENDENT,
    )

    standard = CampaignCellKind.STANDARD
    derived = CampaignCellKind.DERIVED
    return (
        _catalog_cell(
            setting=setting,
            kind=standard,
            name="full_CoT_pipeline_zeroshot",
            factors=direct_zs,
            upstream_canonical_id=None,
        ),
        _catalog_cell(
            setting=setting,
            kind=standard,
            name="full_decontextualization_CoT_pipeline_zeroshot",
            factors=direct_zs_context,
            upstream_canonical_id=None,
            content_selection_source_canonical_id=_canonical_id(
                setting,
                direct_zs,
            ),
        ),
        _catalog_cell(
            setting=setting,
            kind=standard,
            name="full_CoT_pipeline_dialogue",
            factors=direct_fs_dialogue,
            upstream_canonical_id=None,
        ),
        _catalog_cell(
            setting=setting,
            kind=standard,
            name="full_decontextualization_CoT_pipeline_dialogue",
            factors=direct_fs_context_dialogue,
            upstream_canonical_id=None,
            content_selection_source_canonical_id=_canonical_id(
                setting,
                direct_fs_dialogue,
            ),
        ),
        _catalog_cell(
            setting=setting,
            kind=standard,
            name="full_CoT_pipeline_structured",
            factors=direct_fs,
            upstream_canonical_id=None,
        ),
        _catalog_cell(
            setting=setting,
            kind=standard,
            name="full_decontextualization_CoT_pipeline_structured",
            factors=direct_fs_context,
            upstream_canonical_id=None,
            content_selection_source_canonical_id=_canonical_id(
                setting,
                direct_fs,
            ),
        ),
        _catalog_cell(
            setting=setting,
            kind=derived,
            name="coherence",
            factors=planned_fs,
            upstream_canonical_id=_canonical_id(
                setting,
                direct_fs,
            ),
        ),
        _catalog_cell(
            setting=setting,
            kind=derived,
            name="mega",
            factors=planned_zs_context,
            upstream_canonical_id=_canonical_id(
                setting,
                direct_zs_context,
            ),
        ),
    )


__all__ = [
    "CampaignCellKind",
    "CampaignPlan",
    "CatalogCell",
    "ExperimentCatalog",
    "SharedContentSelectionPair",
]
