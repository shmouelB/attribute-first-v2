"""Explicit immutable plan and mutable state for a dialogue run."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DialogueStage:
    """Resolved, immutable runtime contract for one pipeline stage."""

    name: str
    args: Any
    token_counter: Any
    prompt_dict: dict
    structures: dict
    parse_fn: Any
    pipeline_fn: Any
    schema: Any
    continuation: str | None = None


@dataclass(frozen=True)
class DialoguePlan:
    """Immutable configuration and prompt inputs shared by all instances."""

    has_ambiguity_highlight: bool
    uses_coherence_planning: bool
    content_selection: DialogueStage
    ambiguity_highlight: DialogueStage | None
    fusion: DialogueStage
    content_selection_outdir: str
    ambiguity_highlight_outdir: str | None
    clustering_outdir: str | None
    reorder_outdir: str | None
    final_outdir: str
    shared_content_selection_reference: Any
    roles_requested: bool
    no_demos: bool
    model_name: str
    temperature: float
    num_retries: int
    planning_stage_parameters: dict
    alignments: list
    content_selection_prompts: dict
    content_selection_additional: dict
    content_selection_role_messages: dict
    dialogue_role_messages: dict
    initial_content_selection_demos: list
    initial_fusion_demos: list
    initial_args_snapshot: dict
    rerun_context: Any = None


@dataclass
class DialogueRunState:
    """Mutable aggregate produced while instances progress through stages."""

    plan: DialoguePlan
    content_selection_demos: list = field(default_factory=list)
    ambiguity_highlight_demos: list = field(default_factory=list)
    fusion_demos: list = field(default_factory=list)
    content_selection_results: dict = field(default_factory=dict)
    ambiguity_highlight_results: dict = field(default_factory=dict)
    fusion_results: dict = field(default_factory=dict)
    content_selection_rows: dict = field(default_factory=dict)
    ambiguity_highlight_rows: dict = field(default_factory=dict)
    fusion_source_rows: dict = field(default_factory=dict)
    clustering_results: dict = field(default_factory=dict)
    clustering_rows: dict = field(default_factory=dict)
    reorder_results: dict = field(default_factory=dict)
    reorder_rows: dict = field(default_factory=dict)
    call_records: list = field(default_factory=list)
    content_selection_checkpoints: dict = field(default_factory=dict)
    cache_state: Any = None
    args_snapshot: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.args_snapshot:
            self.args_snapshot = deepcopy(
                self.plan.initial_args_snapshot
            )
        if not self.content_selection_demos:
            self.content_selection_demos = list(
                self.plan.initial_content_selection_demos
            )
        if not self.fusion_demos:
            self.fusion_demos = list(self.plan.initial_fusion_demos)


@dataclass
class DialogueInstanceState:
    """Mutable causal state for one fixed-population instance."""

    uid: str
    content_selection_prompt: str
    role_payload: dict | None
    uses_roles: bool = False
    trace: dict = field(
        default_factory=lambda: {
            "content_selection": [],
            "ambiguity_highlight": [],
            "clustering": [],
            "reorder": [],
            "fusion_in_context": [],
        }
    )
    protocol: dict = field(default_factory=dict)
    session: Any = None
    cache_bound: bool = False
    content_selection_demo_history: list = field(default_factory=list)


__all__ = [
    "DialogueInstanceState",
    "DialoguePlan",
    "DialogueRunState",
    "DialogueStage",
]
