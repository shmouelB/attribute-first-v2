"""Concrete parser/converter/schema bindings for the standard runtime."""

from pipeline_converters import (
    convert_ALCE_to_pipeline_format,
    convert_FiC_CoT_results_to_pipeline_format,
    convert_ambiguity_highlight_results_to_pipeline_format,
    convert_clustering_results_to_pipeline_format,
    convert_content_selection_results_to_pipeline_format,
    convert_e2e_only_setting_to_pipeline_format,
)
from response_parsers import (
    parse_ALCE_response,
    parse_FiC_response,
    parse_FiC_structured_response,
    parse_ambiguity_highlight_response,
    parse_ambiguity_highlight_structured_response,
    parse_clustering_response,
    parse_content_selection_response,
    parse_content_selection_structured_response,
    parse_e2e_only_setting_response,
)
from schemas import (
    AMBIGUITY_HIGHLIGHT_SCHEMA,
    CONTENT_SELECTION_SCHEMA,
    FIC_COT_SCHEMA,
)

from ..domain import StageKind
from .registry import StageBinding, StageRegistry


DEFAULT_STAGE_REGISTRY = StageRegistry(
    (
        StageBinding(
            kind=StageKind.CONTENT_SELECTION,
            parser=parse_content_selection_response,
            prompt_subtask_name="content_selection",
            structured_parser=(
                parse_content_selection_structured_response
            ),
            converter=(
                convert_content_selection_results_to_pipeline_format
            ),
            response_schema=CONTENT_SELECTION_SCHEMA,
            schema_name="SUBTASK_SCHEMAS.content_selection",
            schema_aliases=("CS", "content_selection"),
        ),
        StageBinding(
            kind=StageKind.CONTEXT_AUGMENTATION,
            parser=parse_ambiguity_highlight_response,
            prompt_subtask_name="ambiguity_highlight",
            structured_parser=(
                parse_ambiguity_highlight_structured_response
            ),
            converter=(
                convert_ambiguity_highlight_results_to_pipeline_format
            ),
            response_schema=AMBIGUITY_HIGHLIGHT_SCHEMA,
            schema_name="SUBTASK_SCHEMAS.ambiguity_highlight",
            schema_aliases=(
                "AH",
                "ambiguity_highlight",
                "context_augmentation",
            ),
        ),
        StageBinding(
            kind=StageKind.CLUSTERING,
            parser=parse_clustering_response,
            converter=convert_clustering_results_to_pipeline_format,
            prompt_subtask_name="clustering",
        ),
        StageBinding(
            kind=StageKind.FUSION_IN_CONTEXT,
            parser=parse_FiC_response,
            structured_parser=parse_FiC_structured_response,
            converter=convert_FiC_CoT_results_to_pipeline_format,
            prompt_subtask_name="FiC",
            prompt_name_overrides={
                "topic_outline_fusion": "topic_outline_fusion",
                "topic_cluster_fusion": "topic_cluster_fusion",
                "FiC_v2": "FiC_v2",
                "fusion_in_context_v2": "FiC_v2",
            },
            response_schema=FIC_COT_SCHEMA,
            schema_name="SUBTASK_SCHEMAS.FiC",
            schema_aliases=("FiC", "fusion_in_context"),
        ),
        StageBinding(
            kind=StageKind.END_TO_END,
            parser=parse_e2e_only_setting_response,
            converter=convert_e2e_only_setting_to_pipeline_format,
            prompt_subtask_name="e2e_only_setting",
        ),
        StageBinding(
            kind=StageKind.ALCE,
            parser=parse_ALCE_response,
            converter=convert_ALCE_to_pipeline_format,
            prompt_subtask_name="ALCE",
        ),
    )
)


__all__ = ["DEFAULT_STAGE_REGISTRY"]
