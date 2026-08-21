"""Compatibility facade for the stage-specific generation functions."""

from pipeline_converters import (
    convert_ALCE_to_pipeline_format,
    convert_FiC_CoT_results_to_pipeline_format,
    convert_ambiguity_highlight_results_to_pipeline_format,
    convert_clustering_results_to_pipeline_format,
    convert_content_selection_results_to_pipeline_format,
    convert_e2e_only_setting_to_pipeline_format,
)
from prompt_utils import (
    construct_non_demo_part,
    construct_prompts,
    get_data,
    get_subtask_prompt_structures,
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
from attribute_first.stages.standard_registry import (
    DEFAULT_STAGE_REGISTRY,
)


def get_subtask_funcs(subtask, structured_output: bool = False):
    """Return the legacy parser/converter tuple for one registered stage."""

    protocol = DEFAULT_STAGE_REGISTRY.resolve(
        subtask,
        structured_output=structured_output,
    )
    return protocol.parser, protocol.converter
