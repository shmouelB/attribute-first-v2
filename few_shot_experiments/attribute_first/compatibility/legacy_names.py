"""Translate historical names at the boundary of the typed domain."""

from types import MappingProxyType

from ..domain import (
    ContextAugmentation,
    DemonstrationMode,
    GenerationStrategy,
    ModelLifecycle,
    ModelProvider,
    ModelSpec,
    PipelineFactors,
    StageKind,
    TransportMode,
)
from .stage_aliases import resolve_stage_alias


class UnknownLegacyNameError(ValueError):
    """A historical identifier has no declared canonical meaning."""

    def __init__(self, category: str, name: object) -> None:
        super().__init__(f"unknown legacy {category}: {name!r}")
        self.category = category
        self.name = name


def _normalize(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return "_".join(
        name.strip().casefold().replace("-", "_").split()
    )


_GEMINI_3_FLASH_PREVIEW = ModelSpec(
    provider=ModelProvider.GOOGLE,
    requested_model_id="models/gemini-3-flash-preview",
    lifecycle=ModelLifecycle.PREVIEW,
)
_GEMINI_25_FLASH_LITE = ModelSpec(
    provider=ModelProvider.GOOGLE,
    requested_model_id="models/gemini-2.5-flash-lite",
    lifecycle=ModelLifecycle.STABLE,
)
_GEMINI_PRO_LATEST = ModelSpec(
    provider=ModelProvider.GOOGLE,
    requested_model_id="models/gemini-pro-latest",
    lifecycle=ModelLifecycle.MUTABLE_ALIAS,
)


class LegacyNameResolver:
    """The only component allowed to interpret historical shorthand.

    ``mega`` is retained as an opaque historical identifier. The repository
    declares its factors, but does not define an expansion of the name.
    """

    _MODEL_ALIASES = MappingProxyType(
        {
            "g3flash": _GEMINI_3_FLASH_PREVIEW,
            "liteweak": _GEMINI_25_FLASH_LITE,
            "prolatest": _GEMINI_PRO_LATEST,
        }
    )
    _PIPELINE_ALIASES = MappingProxyType(
        {
            "fullcot": (
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.DISABLED,
            ),
            "full_cot": (
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.DISABLED,
            ),
            "decontex": (
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.ENABLED,
            ),
            "decontext": (
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.ENABLED,
            ),
            "decontextualization": (
                GenerationStrategy.DIRECT,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.ENABLED,
            ),
            "coherence": (
                GenerationStrategy.PLANNED,
                DemonstrationMode.FEW_SHOT,
                ContextAugmentation.DISABLED,
            ),
            "mega": (
                GenerationStrategy.PLANNED,
                DemonstrationMode.ZERO_SHOT,
                ContextAugmentation.ENABLED,
            ),
        }
    )

    @classmethod
    def resolve_model(cls, name: object) -> ModelSpec:
        """Resolve a model shorthand to its full provider request."""

        for spec in cls._MODEL_ALIASES.values():
            if name == spec.requested_model_id:
                return spec
        normalized = _normalize(name)
        try:
            return cls._MODEL_ALIASES[normalized]
        except KeyError as exc:
            raise UnknownLegacyNameError("model", name) from exc

    @classmethod
    def resolve_stage(cls, name: object) -> StageKind:
        """Resolve a stage shorthand to the canonical semantic stage."""

        try:
            return resolve_stage_alias(name)
        except ValueError as exc:
            raise UnknownLegacyNameError("stage", name) from exc

    @classmethod
    def resolve_pipeline(
        cls,
        name: object,
        *,
        transport: TransportMode = TransportMode.INDEPENDENT,
    ) -> PipelineFactors:
        """Resolve a historical pipeline label into explicit factors."""

        if not isinstance(transport, TransportMode):
            raise ValueError("transport must be a TransportMode")
        normalized = _normalize(name)
        try:
            generation, demonstrations, context = (
                cls._PIPELINE_ALIASES[normalized]
            )
        except KeyError as exc:
            raise UnknownLegacyNameError("pipeline", name) from exc
        return PipelineFactors(
            generation=generation,
            demonstrations=demonstrations,
            context_augmentation=context,
            transport=transport,
        )
