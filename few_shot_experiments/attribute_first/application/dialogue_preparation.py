"""Build the immutable plan shared by every dialogue instance."""

import json
import logging
import os
from pathlib import Path

from ..stages.configuration import DEFAULT_GENERATION
from .dialogue_rerun import DialogueRerunService
from .dialogue_state import DialoguePlan, DialogueStage


class DialoguePlanBuilder:
    """Resolve stage configs and prompt contracts before provider calls."""

    def __init__(self, dependencies):
        self._dependencies = dependencies

    def build(
        self,
        *,
        args,
        full_configs,
        original_args_dict,
        outdir,
        intermediate_outdir,
    ):
        has_ah = "ambiguity_highlight" in [
            entry["subtask"] for entry in full_configs
        ]
        cs_outdir = os.path.join(
            intermediate_outdir,
            "content_selection",
        )
        ah_outdir = (
            os.path.join(intermediate_outdir, "ambiguity_highlight")
            if has_ah
            else None
        )
        planned_dialogue = (
            getattr(args, "planned_dialogue", False) is True
        )
        clustering_outdir = (
            os.path.join(intermediate_outdir, "clustering")
            if planned_dialogue
            else None
        )
        reorder_outdir = (
            os.path.join(intermediate_outdir, "reorder")
            if planned_dialogue
            else None
        )
        shared_reference = getattr(
            args,
            "_shared_content_selection_reference",
            None,
        )
        if shared_reference is None:
            Path(cs_outdir).mkdir(parents=True, exist_ok=True)
        if ah_outdir:
            Path(ah_outdir).mkdir(parents=True, exist_ok=True)
        if clustering_outdir:
            Path(clustering_outdir).mkdir(parents=True, exist_ok=True)
        if reorder_outdir:
            Path(reorder_outdir).mkdir(parents=True, exist_ok=True)

        resolved = self._resolve_stage_args(
            args=args,
            full_configs=full_configs,
            original_args_dict=original_args_dict,
            outdir=outdir,
            cs_outdir=cs_outdir,
            ah_outdir=ah_outdir,
            has_ah=has_ah,
        )
        roles_requested = self._dependencies.env_flag("AF_USE_ROLES")
        no_demos = self._dependencies.env_flag("AF_DIALOGUE_NO_DEMOS")
        if shared_reference is None:
            (
                cs_stage,
                alignments,
                cs_demos,
                cs_prompts,
                cs_additional,
                cs_role_messages,
            ) = self._prepare_content_selection(
                resolved["content_selection"],
                no_demos=no_demos,
            )
        else:
            (
                cs_stage,
                alignments,
                cs_demos,
                cs_prompts,
                cs_additional,
                cs_role_messages,
            ) = self._prepare_shared_content_selection(
                resolved["content_selection"],
                shared_reference=shared_reference,
                no_demos=no_demos,
            )
        rerun_context = getattr(args, "_dialogue_rerun_context", None)
        if rerun_context is not None:
            active_ids = set(
                DialogueRerunService.active_ids(
                    rerun_context,
                    [str(row["unique_id"]) for row in alignments],
                )
            )
            cs_prompts = {
                unique_id: prompt
                for unique_id, prompt in cs_prompts.items()
                if unique_id in active_ids
            }
            cs_role_messages = {
                unique_id: payload
                for unique_id, payload in cs_role_messages.items()
                if unique_id in active_ids
            }
        ah_stage = self._prepare_ambiguity_highlight(
            resolved.get("ambiguity_highlight")
        )
        fic_stage, fic_demos = self._prepare_fusion(
            resolved["fusion"],
            alignments=alignments,
            roles_requested=roles_requested,
            no_demos=no_demos,
        )
        role_messages = (
            self._scope_content_selection_roles(
                cs_prompts=cs_prompts,
                cs_role_messages=cs_role_messages,
                roles_requested=roles_requested,
            )
            if shared_reference is None
            else {}
        )
        model_name = cs_stage.args.model_name
        temperature = getattr(cs_stage.args, "temperature", 0.1)
        num_retries = getattr(cs_stage.args, "num_retries", 3)
        planning_stage_parameters = (
            self._planning_stage_parameters(resolved, fic_stage)
            if planned_dialogue
            else {}
        )
        snapshot = self._build_args_snapshot(
            original_args_dict=original_args_dict,
            full_configs=full_configs,
            roles_requested=roles_requested,
            no_demos=no_demos,
            cs_stage=cs_stage,
            ah_stage=ah_stage,
            fic_stage=fic_stage,
            cs_demos=cs_demos,
            fic_demos=fic_demos,
            model_name=model_name,
            temperature=temperature,
            num_retries=num_retries,
            planning_stage_parameters=planning_stage_parameters,
        )
        return DialoguePlan(
            has_ambiguity_highlight=has_ah,
            uses_coherence_planning=planned_dialogue,
            content_selection=cs_stage,
            ambiguity_highlight=ah_stage,
            fusion=fic_stage,
            content_selection_outdir=cs_outdir,
            ambiguity_highlight_outdir=ah_outdir,
            clustering_outdir=clustering_outdir,
            reorder_outdir=reorder_outdir,
            final_outdir=outdir,
            shared_content_selection_reference=shared_reference,
            roles_requested=roles_requested,
            no_demos=no_demos,
            model_name=model_name,
            temperature=temperature,
            num_retries=num_retries,
            planning_stage_parameters=planning_stage_parameters,
            alignments=alignments,
            content_selection_prompts=cs_prompts,
            content_selection_additional=cs_additional,
            content_selection_role_messages=cs_role_messages,
            dialogue_role_messages=role_messages,
            initial_content_selection_demos=cs_demos,
            initial_fusion_demos=fic_demos,
            initial_args_snapshot=snapshot,
            rerun_context=rerun_context,
        )

    def _resolve_stage_args(
        self,
        *,
        args,
        full_configs,
        original_args_dict,
        outdir,
        cs_outdir,
        ah_outdir,
        has_ah,
    ):
        build_args = self._dependencies.build_subtask_args
        update_args = self._dependencies.update_args
        content_selection = update_args(
            build_args(
                full_configs,
                "content_selection",
                original_args_dict,
                curr_outdir=cs_outdir,
                indir_alignments=args.indir_alignments,
            )
        )
        fusion = update_args(
            build_args(
                full_configs,
                "fusion_in_context",
                original_args_dict,
                curr_outdir=outdir,
                indir_alignments=None,
            )
        )
        resolved = {
            "content_selection": content_selection,
            "fusion": fusion,
        }
        if getattr(args, "planned_dialogue", False) is True:
            for stage_name in ("clustering", "reorder"):
                resolved[stage_name] = update_args(
                    build_args(
                        full_configs,
                        stage_name,
                        original_args_dict,
                        curr_outdir=outdir,
                        indir_alignments=None,
                    )
                )
        if has_ah:
            resolved["ambiguity_highlight"] = update_args(
                build_args(
                    full_configs,
                    "ambiguity_highlight",
                    original_args_dict,
                    curr_outdir=ah_outdir,
                    indir_alignments=None,
                )
            )
        return resolved

    @staticmethod
    def _planning_stage_parameters(resolved, fusion_stage):
        """Materialize runtime parameters from the archived stage configs."""

        parameters = {}
        for stage_name, args in (
            ("clustering", resolved["clustering"]),
            ("reorder", resolved["reorder"]),
            ("fusion", fusion_stage.args),
        ):
            parameters[stage_name] = {
                "temperature": getattr(args, "temperature"),
                "num_retries": getattr(args, "num_retries"),
                "prompt_token_budget": getattr(
                    args,
                    "prompt_token_budget",
                ),
                "max_output_tokens": getattr(
                    args,
                    "output_max_length",
                ),
            }
        return parameters

    def _prepare_shared_content_selection(
        self,
        args,
        *,
        shared_reference,
        no_demos,
    ):
        """Hydrate logical CS state without rebuilding provider prompts."""

        _, alignments = self._dependencies.get_data(args)
        if not isinstance(alignments, list) or not alignments:
            raise ValueError(
                "shared dialogue content selection has no input population"
            )
        unique_ids = [
            row.get("unique_id")
            for row in alignments
            if isinstance(row, dict)
        ]
        if (
            len(unique_ids) != len(alignments)
            or any(
                not isinstance(unique_id, str) or not unique_id
                for unique_id in unique_ids
            )
            or len(set(unique_ids)) != len(unique_ids)
        ):
            raise ValueError(
                "shared dialogue content-selection population has "
                "invalid unique IDs"
            )
        demos_path = shared_reference.snapshot_for(
            "used_demonstrations.json"
        )
        try:
            demos = json.loads(demos_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "invalid shared content-selection demonstrations"
            ) from exc
        expected_demo_count = (
            0 if no_demos else getattr(args, "n_demos", 0)
        )
        if (
            not isinstance(demos, list)
            or len(demos) != expected_demo_count
        ):
            raise ValueError(
                "shared content-selection demonstrations conflict with "
                "the consumer config"
            )
        structured = DEFAULT_GENERATION.structured_output_for(args)
        stage = DialogueStage(
            name="content_selection",
            args=args,
            token_counter=None,
            prompt_dict={},
            structures={},
            parse_fn=None,
            pipeline_fn=None,
            schema=(
                self._dependencies.subtask_schemas.get(
                    "content_selection"
                )
                if structured
                else None
            ),
        )
        prompts = {
            unique_id: "" for unique_id in unique_ids
        }
        return stage, alignments, demos, prompts, {}, {}

    def _prepare_content_selection(self, args, *, no_demos):
        dependencies = self._dependencies
        structured = DEFAULT_GENERATION.structured_output_for(args)
        token_counter = dependencies.get_token_counter(
            args.model_name,
            getattr(args, "prompt_token_budget", None),
        )
        prompt_dict = dependencies.load_subtask_prompt_dict(args)
        structures = dependencies.get_subtask_prompt_structures(
            prompt_dict,
            args.setting,
            "content_selection",
            args.CoT,
            getattr(args, "always_with_question", False),
            structured_output=structured,
        )
        _, alignments = dependencies.get_data(args)
        demos, prompts, additional, role_messages = (
            dependencies.construct_prompts(
                prompt_dict=prompt_dict,
                alignments_dict=alignments,
                n_demos=0 if no_demos else getattr(args, "n_demos", 0),
                debugging=getattr(args, "debugging", False),
                merge_cross_sents_highlights=getattr(
                    args,
                    "merge_cross_sents_highlights",
                    False,
                ),
                specific_prompt_details=structures,
                tkn_counter=token_counter,
                no_highlights=True,
                cut_surplus=getattr(args, "cut_surplus", False),
                prct_surplus=getattr(args, "prct_surplus", None),
                seed=getattr(args, "seed", None),
            )
        )
        parse_fn, pipeline_fn = dependencies.get_subtask_funcs(
            "content_selection",
            structured_output=structured,
        )
        stage = DialogueStage(
            name="content_selection",
            args=args,
            token_counter=token_counter,
            prompt_dict=prompt_dict,
            structures=structures,
            parse_fn=parse_fn,
            pipeline_fn=pipeline_fn,
            schema=(
                dependencies.subtask_schemas.get("content_selection")
                if structured
                else None
            ),
        )
        return (
            stage,
            alignments,
            demos,
            prompts,
            additional,
            role_messages,
        )

    def _prepare_ambiguity_highlight(self, args):
        if args is None:
            return None
        dependencies = self._dependencies
        structured = DEFAULT_GENERATION.structured_output_for(args)
        token_counter = dependencies.get_token_counter(
            args.model_name,
            getattr(args, "prompt_token_budget", None),
        )
        prompt_dict = dependencies.load_subtask_prompt_dict(args)
        structures = dependencies.get_subtask_prompt_structures(
            prompt_dict,
            args.setting,
            "ambiguity_highlight",
            getattr(args, "CoT", False),
            getattr(args, "always_with_question", False),
            structured_output=structured,
        )
        instruction = structures.get(
            "instruction_prompt",
            prompt_dict.get("instruction-ambiguity-highlight", ""),
        )
        answer_details = structures.get("answer_related_prompts", {})
        answer_prefix = (
            ""
            if structured
            else answer_details.get(
                "answer_prompt",
                prompt_dict.get(
                    "answer_ambiguity_highlight_prompt",
                    "",
                ),
            )
        )
        answer_format = (
            ""
            if structured
            else answer_details.get(
                "answer_ambiguity_highlight_format",
                prompt_dict.get(
                    "answer_ambiguity_highlight_format",
                    "",
                ),
            )
        )
        continuation = (
            "### LIVE INSTANCE — AMBIGUITY_HIGHLIGHT ###\n"
            "Use only the selected content and documents already present in "
            "the earlier live content-selection exchange. Do not require "
            "that state to be copied into this message.\n\n"
            f"{instruction}\n\n{answer_prefix}{answer_format}"
        )
        parse_fn, pipeline_fn = dependencies.get_subtask_funcs(
            "ambiguity_highlight",
            structured_output=structured,
        )
        return DialogueStage(
            name="ambiguity_highlight",
            args=args,
            token_counter=token_counter,
            prompt_dict=prompt_dict,
            structures=structures,
            parse_fn=parse_fn,
            pipeline_fn=pipeline_fn,
            schema=(
                dependencies.subtask_schemas.get("ambiguity_highlight")
                if structured
                else None
            ),
            continuation=continuation,
        )

    def _prepare_fusion(
        self,
        args,
        *,
        alignments,
        roles_requested,
        no_demos,
    ):
        dependencies = self._dependencies
        structured = DEFAULT_GENERATION.structured_output_for(args)
        token_counter = dependencies.get_token_counter(
            args.model_name,
            getattr(args, "prompt_token_budget", None),
        )
        prompt_dict = dependencies.load_subtask_prompt_dict(args)
        structures = dependencies.get_subtask_prompt_structures(
            prompt_dict,
            args.setting,
            "FiC",
            getattr(args, "CoT", True),
            getattr(args, "always_with_question", False),
            structured_output=structured,
        )
        instruction = structures.get(
            "instruction_prompt",
            prompt_dict.get(
                "instruction-FiC-CoT",
                prompt_dict.get("instruction-FiC", ""),
            ),
        )
        answer_details = structures.get("answer_related_prompts", {})
        answer_prefix = (
            ""
            if structured
            else answer_details.get(
                "answer_prompt",
                prompt_dict.get(
                    "answer_FiC-CoT_prompt",
                    prompt_dict.get("answer_FiC_prompt", ""),
                ),
            )
        )
        format_example = (
            ""
            if structured
            else prompt_dict.get(
                "dialogue-FiC-CoT-format-example",
                "",
            )
        )
        demos = []
        demo_block = ""
        if no_demos:
            logging.info(
                "[dialogue] downstream stage demonstrations disabled by "
                "AF_DIALOGUE_NO_DEMOS"
            )
        elif not roles_requested and getattr(args, "n_demos", 0):
            demos, demo_prompts, _, _ = dependencies.construct_prompts(
                prompt_dict=prompt_dict,
                alignments_dict=alignments,
                n_demos=args.n_demos,
                debugging=getattr(args, "debugging", False),
                merge_cross_sents_highlights=getattr(
                    args,
                    "merge_cross_sents_highlights",
                    False,
                ),
                specific_prompt_details=structures,
                tkn_counter=token_counter,
                no_highlights=False,
                cut_surplus=getattr(args, "cut_surplus", False),
                prct_surplus=getattr(args, "prct_surplus", None),
                seed=getattr(args, "seed", None),
            )
            target_header = (
                "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###\n"
            )
            first_prompt = next(iter(demo_prompts.values()), "")
            if target_header not in first_prompt:
                raise ValueError(
                    "cannot isolate the FiC demonstration prefix"
                )
            demo_block = first_prompt.split(target_header)[0].strip()
            logging.info(
                "[dialogue] flat FiC continuation includes "
                f"{args.n_demos} FiC demonstrations "
                f"({len(demo_block)} chars)"
            )
        demo_prefix = (demo_block + "\n\n") if demo_block else ""
        continuation = (
            f"{demo_prefix}"
            "### LIVE INSTANCE — FUSION_IN_CONTEXT ###\n"
            "Use the highlights in the latest successful LIVE output already "
            "present in conversation history. Assign highlight_ids in their "
            "1-based appearance order (document order, then span order for a "
            "content-selection map). A canonical ID table is appended to "
            "this task when local offset normalization changes that "
            "appearance order.\n\n"
            f"{instruction}{format_example}\n\n{answer_prefix}"
        )
        parse_fn, pipeline_fn = dependencies.get_subtask_funcs(
            "FiC",
            structured_output=structured,
        )
        return (
            DialogueStage(
                name="fusion_in_context",
                args=args,
                token_counter=token_counter,
                prompt_dict=prompt_dict,
                structures=structures,
                parse_fn=parse_fn,
                pipeline_fn=pipeline_fn,
                schema=(
                    dependencies.subtask_schemas.get("FiC")
                    if structured
                    else None
                ),
                continuation=continuation,
            ),
            demos,
        )

    def _scope_content_selection_roles(
        self,
        *,
        cs_prompts,
        cs_role_messages,
        roles_requested,
    ):
        if not roles_requested:
            return {}
        if cs_prompts and not cs_role_messages:
            raise ValueError(
                "AF_USE_ROLES is enabled but content-selection prompt "
                "construction produced no role payloads"
            )
        scoped = {}
        for instance_id in cs_prompts:
            view = self._dependencies.dialogue_role_view(
                cs_role_messages[instance_id],
                "content_selection",
            )
            scoped[instance_id] = {
                "system": self._dependencies.system_instruction,
                "contents": view["contents"],
            }
        return scoped

    def _build_args_snapshot(
        self,
        *,
        original_args_dict,
        full_configs,
        roles_requested,
        no_demos,
        cs_stage,
        ah_stage,
        fic_stage,
        cs_demos,
        fic_demos,
        model_name,
        temperature,
        num_retries,
        planning_stage_parameters,
    ):
        hash_value = self._dependencies.stable_value_sha256
        cs_demo_count = (
            0
            if no_demos
            else getattr(cs_stage.args, "n_demos", 0)
        )
        ah_demo_count = (
            getattr(ah_stage.args, "n_demos", 0)
            if roles_requested and ah_stage is not None and not no_demos
            else 0
        )
        fic_demo_count = (
            getattr(fic_stage.args, "n_demos", 0)
            if roles_requested and not no_demos
            else 0
        )
        snapshot = dict(original_args_dict)
        snapshot.update(
            {
                "effective_model": model_name,
                "effective_temperature": temperature,
                "effective_num_retries": num_retries,
                "environment_flags": (
                    self._dependencies.get_af_environment_flags()
                ),
                "pipeline_configs": full_configs,
                "planned_dialogue": {
                    "enabled": (
                        getattr(
                            cs_stage.args,
                            "planned_dialogue",
                            False,
                        )
                        is True
                    ),
                    "session_scope": "one_chat_per_instance",
                    "concurrency": 1,
                    "stage_order": (
                        [
                            "content_selection",
                            "ambiguity_highlight",
                            "clustering",
                            "reorder",
                            "fusion_in_context",
                        ]
                        if (
                            getattr(
                                cs_stage.args,
                                "planned_dialogue",
                                False,
                            )
                            is True
                        )
                        else [
                            "content_selection",
                            *(
                                ["ambiguity_highlight"]
                                if ah_stage is not None
                                else []
                            ),
                            "fusion_in_context",
                        ]
                    ),
                    "demonstration_counts": {
                        "content_selection": cs_demo_count,
                        "ambiguity_highlight": ah_demo_count,
                        "clustering": 0,
                        "reorder": 0,
                        "fusion_in_context": (
                            0
                            if (
                                getattr(
                                    cs_stage.args,
                                    "planned_dialogue",
                                    False,
                                )
                                is True
                            )
                            else fic_demo_count
                        ),
                    },
                    "stage_parameters": planning_stage_parameters,
                },
                "dialogue_role_contract": {
                    "enabled": roles_requested,
                    "system_instruction": (
                        self._dependencies.system_instruction
                        if roles_requested
                        else None
                    ),
                    "demonstration_delivery": {
                        "content_selection": {
                            "count": (
                                cs_demo_count if roles_requested else 0
                            ),
                            "timing": (
                                "initial_history"
                                if roles_requested and cs_demo_count
                                else "disabled"
                            ),
                        },
                        "ambiguity_highlight": {
                            "count": ah_demo_count,
                            "timing": (
                                "just_in_time_history"
                                if ah_demo_count
                                else "disabled"
                            ),
                        },
                        "fusion_in_context": {
                            "count": fic_demo_count,
                            "timing": (
                                "just_in_time_history"
                                if fic_demo_count
                                else "disabled"
                            ),
                        },
                    },
                    "demonstration_sets": {
                        "content_selection": {
                            "count": len(cs_demos),
                            "sha256": hash_value(cs_demos),
                        },
                        "ambiguity_highlight": {
                            "count": 0,
                            "sha256": hash_value([]),
                        },
                        "fusion_in_context": {
                            "count": len(fic_demos),
                            "sha256": hash_value(fic_demos),
                        },
                    },
                },
            }
        )
        return snapshot


__all__ = ["DialoguePlanBuilder"]
