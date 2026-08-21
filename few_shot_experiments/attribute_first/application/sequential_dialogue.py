"""Compatibility facade for the sequential-dialogue application.

The legacy :mod:`run_dialogue_sequential` module and existing imports keep
their public names. Implementations are split by contract, per-instance
conversation, result conversion, and population orchestration.
"""

from .sequential_contracts import (
    HIGHLIGHT_END,
    HIGHLIGHT_START,
    SEQUENTIAL_PROTOCOL,
    concrete_offsets as _concrete_offsets,
    parse_clusters,
    parse_content_selection_spans,
    parse_fusion_output,
    source_span_metadata,
    usage_delta,
    usage_from_attempt_trace,
)
from .sequential_instance import (
    SequentialDialogueInstanceRunner,
    SequentialInstanceDependencies,
)
from .sequential_pipeline import (
    SequentialDialoguePipelineRunner,
    SequentialPipelineDependencies,
)
from .sequential_results import SequentialPipelineResultAssembler


__all__ = [
    "HIGHLIGHT_END",
    "HIGHLIGHT_START",
    "SEQUENTIAL_PROTOCOL",
    "SequentialDialogueInstanceRunner",
    "SequentialDialoguePipelineRunner",
    "SequentialInstanceDependencies",
    "SequentialPipelineDependencies",
    "SequentialPipelineResultAssembler",
    "parse_clusters",
    "parse_content_selection_spans",
    "parse_fusion_output",
    "source_span_metadata",
    "usage_delta",
    "usage_from_attempt_trace",
]
