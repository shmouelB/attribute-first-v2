"""Canonical vocabulary for controlled Attribute-First experiments."""

from enum import Enum


class Dataset(str, Enum):
    """Supported experimental datasets."""

    MDS = "MDS"
    LFQA = "LFQA"


class ModelProvider(str, Enum):
    """External model providers."""

    GOOGLE = "google"
    OPENAI = "openai"

    @classmethod
    def from_model_id(cls, model_id: object) -> "ModelProvider":
        """Resolve a full model ID without substring heuristics."""

        if (
            not isinstance(model_id, str)
            or not model_id
            or model_id.strip() != model_id
        ):
            raise ValueError("model_id must be a non-empty full model ID")
        normalized = model_id.casefold()
        if normalized.startswith("models/") and len(model_id) > len(
            "models/"
        ):
            return cls.GOOGLE
        if normalized.startswith("gpt-") and len(model_id) > len("gpt-"):
            return cls.OPENAI
        raise ValueError(
            f"unsupported provider model ID: {model_id!r}"
        )


class ModelLifecycle(str, Enum):
    """Reproducibility status of a requested provider model identifier."""

    STABLE = "stable"
    PREVIEW = "preview"
    MUTABLE_ALIAS = "mutable_alias"


class StageKind(str, Enum):
    """Semantic stage names, independent of historical abbreviations."""

    CONTENT_SELECTION = "content_selection"
    CONTEXT_AUGMENTATION = "context_augmentation"
    CLUSTERING = "clustering"
    REORDERING = "reordering"
    FUSION_IN_CONTEXT = "fusion_in_context"
    END_TO_END = "end_to_end"
    ALCE = "alce"


class GenerationStrategy(str, Enum):
    """How selected evidence is transformed into attributed text."""

    DIRECT = "direct"
    PLANNED = "planned"


class DemonstrationMode(str, Enum):
    """Whether any configurable stage receives demonstrations."""

    FEW_SHOT = "few_shot"
    ZERO_SHOT = "zero_shot"

    @property
    def short_name(self) -> str:
        """Return the compact, unambiguous factor label."""

        return {
            DemonstrationMode.FEW_SHOT: "fs",
            DemonstrationMode.ZERO_SHOT: "zs",
        }[self]


class ContextAugmentation(str, Enum):
    """Whether ambiguity-aware source context is added before fusion."""

    DISABLED = "disabled"
    ENABLED = "enabled"


class TransportMode(str, Enum):
    """Provider interaction mode for stages belonging to one example."""

    INDEPENDENT = "independent"
    DIALOGUE = "dialogue"
