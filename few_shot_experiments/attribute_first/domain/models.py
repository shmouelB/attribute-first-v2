"""Immutable domain objects for experiment configuration."""

from dataclasses import dataclass

from .enums import (
    ContextAugmentation,
    Dataset,
    DemonstrationMode,
    GenerationStrategy,
    ModelLifecycle,
    ModelProvider,
    StageKind,
    TransportMode,
)
from .policies import CachePolicy, RetryPolicy, TokenBudget


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """A provider model request with its reproducibility lifecycle."""

    provider: ModelProvider
    requested_model_id: str
    lifecycle: ModelLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.provider, ModelProvider):
            raise ValueError("provider must be a ModelProvider")
        try:
            inferred_provider = ModelProvider.from_model_id(
                self.requested_model_id
            )
        except ValueError as exc:
            raise ValueError(
                "requested_model_id must be a full provider model id"
            ) from exc
        if inferred_provider is not self.provider:
            raise ValueError(
                "provider and requested_model_id disagree"
            )
        if not isinstance(self.lifecycle, ModelLifecycle):
            raise ValueError("lifecycle must be a ModelLifecycle")


@dataclass(frozen=True, slots=True)
class PipelineFactors:
    """Independent scientific and execution factors of one pipeline."""

    generation: GenerationStrategy
    demonstrations: DemonstrationMode
    context_augmentation: ContextAugmentation
    transport: TransportMode

    def __post_init__(self) -> None:
        expected_types = (
            (self.generation, GenerationStrategy, "generation"),
            (
                self.demonstrations,
                DemonstrationMode,
                "demonstrations",
            ),
            (
                self.context_augmentation,
                ContextAugmentation,
                "context_augmentation",
            ),
            (self.transport, TransportMode, "transport"),
        )
        for value, expected_type, name in expected_types:
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"{name} must be a {expected_type.__name__}"
                )

    @property
    def canonical_id(self) -> str:
        """Build a readable ID from the independent factors."""

        parts = [
            self.generation.value,
            self.demonstrations.short_name,
        ]
        if self.context_augmentation is ContextAugmentation.ENABLED:
            parts.append("context_augmented")
        parts.append(self.transport.value)
        return "_".join(parts)


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Complete immutable configuration of one semantic stage."""

    kind: StageKind
    model: ModelSpec
    demonstration_count: int
    retry_policy: RetryPolicy
    token_budget: TokenBudget
    cache_policy: CachePolicy
    temperature: float
    output_token_limit: int
    structured_output: bool
    schema_name: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StageKind):
            raise ValueError("kind must be a StageKind")
        if not isinstance(self.model, ModelSpec):
            raise ValueError("model must be a ModelSpec")
        if (
            type(self.demonstration_count) is not int
            or self.demonstration_count < 0
        ):
            raise ValueError(
                "demonstration_count must be a non-negative integer"
            )
        if not isinstance(self.retry_policy, RetryPolicy):
            raise ValueError("retry_policy must be a RetryPolicy")
        if not isinstance(self.token_budget, TokenBudget):
            raise ValueError("token_budget must be a TokenBudget")
        if not isinstance(self.cache_policy, CachePolicy):
            raise ValueError("cache_policy must be a CachePolicy")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if (
            type(self.output_token_limit) is not int
            or self.output_token_limit <= 0
        ):
            raise ValueError(
                "output_token_limit must be a positive integer"
            )
        if type(self.structured_output) is not bool:
            raise ValueError("structured_output must be boolean")
        if self.structured_output and (
            not isinstance(self.schema_name, str)
            or not self.schema_name
        ):
            raise ValueError(
                "schema_name is required for structured output"
            )
        if not self.structured_output and self.schema_name is not None:
            raise ValueError(
                "schema_name must be absent for unstructured output"
            )


@dataclass(frozen=True, slots=True)
class ExperimentCell:
    """One dataset population and one fully specified pipeline."""

    cell_id: str
    dataset: Dataset
    population_size: int
    factors: PipelineFactors
    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.cell_id, str)
            or not self.cell_id
            or self.cell_id.strip() != self.cell_id
        ):
            raise ValueError("cell_id must be a non-empty canonical ID")
        if not isinstance(self.dataset, Dataset):
            raise ValueError("dataset must be a Dataset")
        if (
            type(self.population_size) is not int
            or self.population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer")
        if not isinstance(self.factors, PipelineFactors):
            raise ValueError("factors must be PipelineFactors")

        stages = tuple(self.stages)
        if not stages or any(
            not isinstance(stage, StageSpec) for stage in stages
        ):
            raise ValueError("stages must contain StageSpec values")
        object.__setattr__(self, "stages", stages)

        kinds = self.stage_kinds
        if len(kinds) != len(set(kinds)):
            raise ValueError("a stage kind cannot occur more than once")
        self._validate_stage_composition(kinds)
        self._validate_demonstration_mode()

    @property
    def stage_kinds(self) -> tuple[StageKind, ...]:
        """Return the canonical stage sequence."""

        return tuple(stage.kind for stage in self.stages)

    def _validate_stage_composition(
        self,
        kinds: tuple[StageKind, ...],
    ) -> None:
        has_context = (
            self.factors.context_augmentation
            is ContextAugmentation.ENABLED
        )
        context_stages = (
            (StageKind.CONTEXT_AUGMENTATION,)
            if has_context
            else ()
        )
        if self.factors.generation is GenerationStrategy.DIRECT:
            expected = (
                StageKind.CONTENT_SELECTION,
                *context_stages,
                StageKind.FUSION_IN_CONTEXT,
            )
            failure = "direct generation stage composition is invalid"
        else:
            expected = (
                StageKind.CONTENT_SELECTION,
                *context_stages,
                StageKind.CLUSTERING,
                StageKind.REORDERING,
                StageKind.FUSION_IN_CONTEXT,
            )
            failure = "planned generation requires clustering and reordering"

        observed_context = (
            StageKind.CONTEXT_AUGMENTATION in kinds
        )
        if observed_context != has_context:
            raise ValueError(
                "context augmentation factor and stage disagree"
            )
        if kinds != expected:
            raise ValueError(failure)

    def _validate_demonstration_mode(self) -> None:
        counts = tuple(
            stage.demonstration_count for stage in self.stages
        )
        if (
            self.factors.demonstrations
            is DemonstrationMode.ZERO_SHOT
            and any(counts)
        ):
            raise ValueError(
                "zero-shot cells cannot contain demonstrations"
            )
        if (
            self.factors.demonstrations
            is DemonstrationMode.FEW_SHOT
            and not any(counts)
        ):
            raise ValueError(
                "few-shot cells require at least one demonstration"
            )
