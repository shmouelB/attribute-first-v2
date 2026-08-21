"""Shared construction of validation prompts for live dialogue stages."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreparedDialogueStage:
    """Prompt evidence generated for one instance and one stage."""

    demos: list
    validation_prompt: object
    additional: dict
    role_messages: dict
    live_demo_count: int


class DialogueStagePromptBuilder:
    """Build AH/FiC validation inputs from the live upstream row."""

    def __init__(self, dependencies):
        self._dependencies = dependencies

    @staticmethod
    def _live_demo_count(plan, instance, stage) -> int:
        return (
            getattr(stage.args, "n_demos", 0)
            if instance.uses_roles and not plan.no_demos
            else 0
        )

    def build(self, state, instance, stage, source_row):
        """Construct prompts with the exact stage configuration."""

        live_demo_count = self._live_demo_count(
            state.plan,
            instance,
            stage,
        )
        demos, validation_prompts, additional, role_messages = (
            self._dependencies.construct_prompts(
                prompt_dict=stage.prompt_dict,
                alignments_dict=[source_row],
                n_demos=live_demo_count,
                debugging=getattr(stage.args, "debugging", False),
                merge_cross_sents_highlights=getattr(
                    stage.args,
                    "merge_cross_sents_highlights",
                    False,
                ),
                specific_prompt_details=stage.structures,
                tkn_counter=stage.token_counter,
                no_highlights=False,
                cut_surplus=getattr(
                    stage.args,
                    "cut_surplus",
                    False,
                ),
                prct_surplus=getattr(
                    stage.args,
                    "prct_surplus",
                    None,
                ),
                seed=getattr(stage.args, "seed", None),
            )
        )
        return PreparedDialogueStage(
            demos=demos,
            validation_prompt=validation_prompts[instance.uid],
            additional=additional,
            role_messages=role_messages,
            live_demo_count=live_demo_count,
        )


__all__ = [
    "DialogueStagePromptBuilder",
    "PreparedDialogueStage",
]
