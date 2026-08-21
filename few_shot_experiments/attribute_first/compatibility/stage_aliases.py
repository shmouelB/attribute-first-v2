"""Canonical stage names accepted at the legacy command boundary."""

from types import MappingProxyType

from ..domain.enums import StageKind


def normalize_stage_name(name: object) -> str:
    """Normalize a user-facing stage label without guessing substrings."""

    if not isinstance(name, str) or not name.strip():
        return ""
    return "_".join(
        name.strip().casefold().replace("-", "_").split()
    )


STAGE_ALIASES = MappingProxyType(
    {
        "cs": StageKind.CONTENT_SELECTION,
        "content_selection": StageKind.CONTENT_SELECTION,
        "ah": StageKind.CONTEXT_AUGMENTATION,
        "ambiguity_highlight": StageKind.CONTEXT_AUGMENTATION,
        "context_augmentation": StageKind.CONTEXT_AUGMENTATION,
        "clustering": StageKind.CLUSTERING,
        "reorder": StageKind.REORDERING,
        "reordering": StageKind.REORDERING,
        "fic": StageKind.FUSION_IN_CONTEXT,
        "fusion_in_context": StageKind.FUSION_IN_CONTEXT,
        "topic_outline_fusion": StageKind.FUSION_IN_CONTEXT,
        "topic_cluster_fusion": StageKind.FUSION_IN_CONTEXT,
        "fic_v2": StageKind.FUSION_IN_CONTEXT,
        "fusion_in_context_v2": StageKind.FUSION_IN_CONTEXT,
        "e2e_only_setting": StageKind.END_TO_END,
        "end_to_end": StageKind.END_TO_END,
        "alce": StageKind.ALCE,
        "iterative_sentence_generation": StageKind.REORDERING,
    }
)


def resolve_stage_alias(name: object) -> StageKind:
    """Return the semantic stage kind for one declared legacy alias."""

    normalized = normalize_stage_name(name)
    try:
        return STAGE_ALIASES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported stage alias: {name!r}") from exc


def aliases_for_stage(kind: StageKind) -> tuple[str, ...]:
    """Return every normalized alias declared for one stage kind."""

    if not isinstance(kind, StageKind):
        raise TypeError("kind must be a StageKind")
    return tuple(
        alias
        for alias, candidate in STAGE_ALIASES.items()
        if candidate is kind
    )


__all__ = [
    "STAGE_ALIASES",
    "aliases_for_stage",
    "normalize_stage_name",
    "resolve_stage_alias",
]
