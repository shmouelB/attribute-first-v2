"""Application service behind the legacy full-pipeline entry point."""

import argparse
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..artifacts.output_directory import OutputDirectoryClaim
from ..stages.configuration import StageConfigContract


@dataclass(frozen=True)
class PipelineApplicationDependencies:
    """Patch-aware collaborators captured by the compatibility façade."""

    config_protocol_environment: Callable
    main_func: Callable
    iterative_sent_gen_main: Callable
    persist_pipeline_provenance: Callable
    prepare_shared_content_selection: Callable
    prepare_dialogue_rerun: Callable
    dialogue_protocol_environment: Callable
    run_dialogue_pipeline: Callable
    persist_pipeline_token_usage: Callable
    persist_dialogue_content_selection_usage: Callable
    persist_pipeline_response_metadata: Callable
    run_subtask: Callable
    log_stage_health: Callable

    @classmethod
    def from_namespace(cls, namespace):
        return cls(
            config_protocol_environment=namespace[
                "config_protocol_environment"
            ],
            main_func=namespace["main_func"],
            iterative_sent_gen_main=namespace[
                "iterative_sent_gen_main"
            ],
            persist_pipeline_provenance=namespace[
                "persist_pipeline_provenance"
            ],
            prepare_shared_content_selection=namespace[
                "prepare_shared_content_selection"
            ],
            prepare_dialogue_rerun=namespace["prepare_dialogue_rerun"],
            dialogue_protocol_environment=namespace[
                "dialogue_protocol_environment"
            ],
            run_dialogue_pipeline=namespace["run_dialogue_pipeline"],
            persist_pipeline_token_usage=namespace[
                "persist_pipeline_token_usage"
            ],
            persist_dialogue_content_selection_usage=namespace[
                "persist_dialogue_content_selection_usage"
            ],
            persist_pipeline_response_metadata=namespace[
                "persist_pipeline_response_metadata"
            ],
            run_subtask=namespace["run_subtask"],
            log_stage_health=namespace["_log_stage_health"],
        )


class PipelineApplicationRunner:
    """Run standard or dialogue pipelines through injected boundaries."""

    CONTROLLED_SPLITS = frozenset({"test", "dev"})
    CONTROLLED_SETTINGS = frozenset({"MDS", "LFQA"})
    ALLOWED_SUBTASK_SEQUENCES = (
        (
            "content_selection",
            "clustering",
            "iterative_sentence_generation",
        ),
        ("content_selection", "fusion_in_context"),
        ("content_selection", "topic_outline_fusion"),
        ("content_selection", "topic_cluster_fusion"),
        (
            "content_selection",
            "ambiguity_highlight",
            "clustering",
            "iterative_sentence_generation",
        ),
        (
            "content_selection",
            "ambiguity_highlight",
            "fusion_in_context",
        ),
        (
            "content_selection",
            "ambiguity_highlight",
            "topic_outline_fusion",
        ),
        (
            "content_selection",
            "ambiguity_highlight",
            "topic_cluster_fusion",
        ),
        ("content_selection", "fusion_in_context_v2"),
        (
            "content_selection",
            "ambiguity_highlight",
            "fusion_in_context_v2",
        ),
    )
    DIALOGUE_SUBTASK_SEQUENCES = (
        ("content_selection", "fusion_in_context"),
        (
            "content_selection",
            "ambiguity_highlight",
            "fusion_in_context",
        ),
    )
    PLANNED_DIALOGUE_SUBTASK_SEQUENCE = (
        "content_selection",
        "ambiguity_highlight",
        "clustering",
        "reorder",
        "fusion_in_context",
    )

    def __init__(self, dependencies):
        self._dependencies = dependencies

    @classmethod
    def _validated_stage_sequence(
        cls,
        full_configs,
        *,
        dialogue_mode: bool,
        planned_dialogue: bool = False,
    ) -> tuple[str, ...]:
        if not isinstance(full_configs, list) or not full_configs:
            raise ValueError(
                "pipeline config must be a non-empty list of stages"
            )
        if any(not isinstance(entry, dict) for entry in full_configs):
            raise ValueError("every pipeline stage must be an object")
        sequence = tuple(entry.get("subtask") for entry in full_configs)
        if any(
            not isinstance(stage, str) or not stage
            for stage in sequence
        ):
            raise ValueError(
                "every pipeline stage requires a non-empty subtask"
            )
        if len(set(sequence)) != len(sequence):
            raise ValueError(
                "pipeline stages must be unique; duplicate subtask found"
            )
        allowed_sequences = cls.ALLOWED_SUBTASK_SEQUENCES
        if planned_dialogue:
            allowed_sequences = (
                *allowed_sequences,
                cls.PLANNED_DIALOGUE_SUBTASK_SEQUENCE,
            )
        if sequence not in allowed_sequences:
            raise ValueError(
                "pipeline config must follow an exact stage sequence"
            )
        if (
            dialogue_mode
            and sequence not in (
                *cls.DIALOGUE_SUBTASK_SEQUENCES,
                *(
                    (cls.PLANNED_DIALOGUE_SUBTASK_SEQUENCE,)
                    if planned_dialogue
                    else ()
                ),
            )
        ):
            raise ValueError(
                "dialogue mode requires content_selection, optional "
                "ambiguity_highlight, then fusion_in_context"
            )
        return sequence

    def run_subtask(
        self,
        full_configs,
        subtask_name,
        curr_outdir,
        original_args_dict,
        indir_alignments=None,
    ):
        current = deepcopy(
            [
                element
                for element in full_configs
                if element["subtask"] == subtask_name
            ][0]
        )
        current.update(
            {
                "outdir": curr_outdir,
                "indir_alignments": indir_alignments,
            }
        )
        function_args = deepcopy(original_args_dict)
        function_args.update(current)
        with self._dependencies.config_protocol_environment(
            current.get("config_file")
        ):
            if subtask_name != "iterative_sentence_generation":
                self._dependencies.main_func(
                    argparse.Namespace(**function_args)
                )
            else:
                self._dependencies.iterative_sent_gen_main(
                    argparse.Namespace(**function_args)
                )

    def run(self, args):
        original_args_dict = deepcopy(args.__dict__)
        with open(args.config_file, "r") as config_file:
            full_configs = json.loads(config_file.read())
        dialogue_mode = getattr(args, "dialogue_mode", False)
        planned_dialogue = getattr(args, "planned_dialogue", False)
        if planned_dialogue and not dialogue_mode:
            raise ValueError(
                "--planned-dialogue requires --dialogue-mode"
            )
        if planned_dialogue and getattr(args, "concurrency", 1) != 1:
            raise ValueError(
                "planned dialogue requires --concurrency 1"
            )
        self._validated_stage_sequence(
            full_configs,
            dialogue_mode=dialogue_mode,
            planned_dialogue=planned_dialogue,
        )

        if dialogue_mode:
            with self._dependencies.dialogue_protocol_environment(
                full_configs
            ):
                self._run_validated_pipeline(
                    args=args,
                    full_configs=full_configs,
                    original_args_dict=original_args_dict,
                    dialogue_mode=True,
                )
            return
        self._run_validated_pipeline(
            args=args,
            full_configs=full_configs,
            original_args_dict=original_args_dict,
            dialogue_mode=False,
        )

    def _run_validated_pipeline(
        self,
        *,
        args,
        full_configs,
        original_args_dict,
        dialogue_mode,
    ):
        controlled_config = (
            bool(getattr(args, "canonical_cell_id", None))
            or bool(
                {
                    "controlled",
                    "evidence_designed",
                }
                & set(
                    Path(args.config_file).expanduser().resolve().parts
                )
            )
        )
        splits, settings = self._stage_population_contract(
            full_configs,
            defaults=original_args_dict,
            strict=controlled_config,
        )
        if getattr(args, "planned_dialogue", False):
            if settings != ["LFQA"] * len(settings):
                raise ValueError(
                    "planned dialogue MEGA is supported only for LFQA"
                )
            required_factors = {
                "generation_strategy": "planned",
                "demonstration_mode": "few_shot",
                "context_augmentation": "enabled",
                "transport_mode": "dialogue",
            }
            for field, expected in required_factors.items():
                observed = getattr(args, field, None)
                if observed not in {None, expected}:
                    raise ValueError(
                        "planned dialogue LFQA MEGA requires "
                        f"{field}={expected}"
                    )
            canonical_id = getattr(args, "canonical_cell_id", None)
            if canonical_id not in {
                None,
                "lfqa.planned_fs_context_augmented_dialogue",
            }:
                raise ValueError(
                    "planned dialogue LFQA MEGA has an incompatible "
                    "canonical_cell_id"
                )
        pipeline_subdir = Path(args.config_file).stem
        outdir = (
            args.outdir
            if args.outdir
            else (
                f"results/{splits[0]}/{settings[0]}/"
                f"{pipeline_subdir}"
            )
        )
        logging.info(f"saving results to {outdir}")
        dialogue_rerun_context = None
        if dialogue_mode and getattr(args, "rerun", False):
            prepare_rerun = getattr(
                self._dependencies,
                "prepare_dialogue_rerun",
                None,
            )
            if not callable(prepare_rerun):
                raise RuntimeError(
                    "dialogue rerun preparation is unavailable"
                )
            dialogue_rerun_context = prepare_rerun(args, outdir)
        outdir = str(
            OutputDirectoryClaim.claim(
                outdir,
                owner="full-generation-pipeline-v1",
            )
        )
        original_args_dict["_pipeline_run_root"] = outdir
        intermediate_outdir = os.path.join(
            outdir,
            "itermediate_results",
        )
        Path(intermediate_outdir).mkdir(parents=True, exist_ok=True)
        self._dependencies.persist_pipeline_provenance(
            args,
            full_configs,
            outdir,
        )
        shared_reference = None
        if getattr(
            args,
            "shared_content_selection_source",
            None,
        ) is not None:
            prepare = getattr(
                self._dependencies,
                "prepare_shared_content_selection",
                None,
            )
            if not callable(prepare):
                raise RuntimeError(
                    "shared content-selection preparation is unavailable"
                )
            shared_reference = prepare(args, outdir)
            if shared_reference is None:
                raise RuntimeError(
                    "shared content-selection preparation returned no "
                    "reference"
                )
        setattr(
            args,
            "_shared_content_selection_reference",
            shared_reference,
        )
        setattr(
            args,
            "_dialogue_rerun_context",
            dialogue_rerun_context,
        )

        if dialogue_mode:
            self._run_dialogue(
                args=args,
                full_configs=full_configs,
                original_args_dict=original_args_dict,
                outdir=outdir,
                intermediate_outdir=intermediate_outdir,
            )
            return
        self._run_independent(
            args=args,
            full_configs=full_configs,
            original_args_dict=original_args_dict,
            outdir=outdir,
            intermediate_outdir=intermediate_outdir,
        )

    @classmethod
    def _stage_population_contract(
        cls,
        full_configs,
        *,
        defaults=None,
        strict=False,
    ):
        splits = []
        settings = []
        model_names = []
        for stage_index, element in enumerate(
            full_configs,
            start=1,
        ):
            with open(element["config_file"], "r") as config_file:
                current = json.loads(config_file.read())
            try:
                stage_config = StageConfigContract.from_mapping(
                    current,
                    declared_subtask=element["subtask"],
                    defaults=defaults,
                    strict=strict,
                )
            except ValueError as exc:
                raise ValueError(
                    f"pipeline stage {stage_index} is invalid: {exc}"
                ) from exc
            if (
                strict
                and current.get("split")
                not in cls.CONTROLLED_SPLITS
            ):
                raise ValueError(
                    f"pipeline stage {stage_index} is invalid: "
                    "split must be test or dev"
                )
            if (
                strict
                and current.get("setting")
                not in cls.CONTROLLED_SETTINGS
            ):
                raise ValueError(
                    f"pipeline stage {stage_index} is invalid: "
                    "setting must be MDS or LFQA"
                )
            splits.append(current["split"])
            settings.append(current["setting"])
            model_names.append(stage_config.model_name)
        if len(set(splits)) != 1 or len(set(settings)) != 1:
            raise Exception(
                "all subtasks must have the same split (test/dev) "
                "and the same setting (MDS/LFQA)"
            )
        if strict and len(set(model_names)) != 1:
            raise ValueError(
                "all controlled pipeline stages must declare the same "
                "model_name"
            )
        return splits, settings

    def _run_dialogue(
        self,
        *,
        args,
        full_configs,
        original_args_dict,
        outdir,
        intermediate_outdir,
    ):
        subtasks = {entry["subtask"] for entry in full_configs}
        if (
            "clustering" in subtasks
            or "iterative_sentence_generation" in subtasks
        ) and not getattr(args, "planned_dialogue", False):
            raise ValueError(
                "--dialogue-mode only supports CoT "
                "(fusion_in_context) pipelines."
            )
        self._dependencies.run_dialogue_pipeline(
            args,
            full_configs,
            original_args_dict,
            outdir,
            intermediate_outdir,
        )
        self._dependencies.persist_pipeline_token_usage(outdir)
        persist_cs_usage = getattr(
            self._dependencies,
            "persist_dialogue_content_selection_usage",
            None,
        )
        if callable(persist_cs_usage):
            persist_cs_usage(outdir)
        self._dependencies.persist_pipeline_response_metadata(
            outdir,
            dialogue_mode=True,
        )

    def _run_independent(
        self,
        *,
        args,
        full_configs,
        original_args_dict,
        outdir,
        intermediate_outdir,
    ):
        cs_outdir = os.path.join(
            intermediate_outdir,
            "content_selection",
        )
        shared_reference = getattr(
            args,
            "_shared_content_selection_reference",
            None,
        )
        if shared_reference is None:
            logging.info("running content selection:")
            self._dependencies.run_subtask(
                full_configs=full_configs,
                subtask_name="content_selection",
                curr_outdir=cs_outdir,
                original_args_dict=original_args_dict,
                indir_alignments=args.indir_alignments,
            )
            previous = os.path.join(
                cs_outdir,
                "pipeline_format_results.json",
            )
        else:
            logging.info(
                "reusing exactly equivalent content selection: %s",
                shared_reference.source_root,
            )
            previous = str(
                shared_reference.snapshot_for(
                    "pipeline_format_results.json"
                )
            )
        self._dependencies.log_stage_health(
            previous,
            "content_selection",
        )

        if "ambiguity_highlight" in [
            element["subtask"] for element in full_configs
        ]:
            ah_outdir = os.path.join(
                intermediate_outdir,
                "ambiguity_highlight",
            )
            logging.info("running ambiguity highlight:")
            self._dependencies.run_subtask(
                full_configs=full_configs,
                subtask_name="ambiguity_highlight",
                curr_outdir=ah_outdir,
                original_args_dict=original_args_dict,
                indir_alignments=previous,
            )
            previous = os.path.join(
                ah_outdir,
                "pipeline_format_results.json",
            )
            self._dependencies.log_stage_health(
                previous,
                "ambiguity_highlight",
            )

        if "clustering" in [
            element["subtask"] for element in full_configs
        ]:
            self._run_decomposed(
                full_configs,
                original_args_dict,
                intermediate_outdir,
                outdir,
                previous,
            )
        else:
            self._run_fusion(
                full_configs,
                original_args_dict,
                outdir,
                previous,
            )
        self._dependencies.persist_pipeline_token_usage(outdir)
        self._dependencies.persist_pipeline_response_metadata(
            outdir,
            dialogue_mode=False,
        )

    def _run_decomposed(
        self,
        full_configs,
        original_args_dict,
        intermediate_outdir,
        outdir,
        previous,
    ):
        clustering_outdir = os.path.join(
            intermediate_outdir,
            "clustering",
        )
        logging.info("running clustering:")
        self._dependencies.run_subtask(
            full_configs=full_configs,
            subtask_name="clustering",
            curr_outdir=clustering_outdir,
            original_args_dict=original_args_dict,
            indir_alignments=previous,
        )
        logging.info("running final iterative sentence generation:")
        self._dependencies.run_subtask(
            full_configs=full_configs,
            subtask_name="iterative_sentence_generation",
            curr_outdir=outdir,
            original_args_dict=original_args_dict,
            indir_alignments=os.path.join(
                clustering_outdir,
                "pipeline_format_results.json",
            ),
        )

    def _run_fusion(
        self,
        full_configs,
        original_args_dict,
        outdir,
        previous,
    ):
        fusion_subtasks = {
            "fusion_in_context",
            "fusion_in_context_v2",
            "topic_outline_fusion",
            "topic_cluster_fusion",
        }
        fusion_subtask = next(
            name
            for name in [
                element["subtask"] for element in full_configs
            ]
            if name in fusion_subtasks
        )
        logging.info(
            f"running CoT-style fusion ({fusion_subtask}):"
        )
        self._dependencies.run_subtask(
            full_configs=full_configs,
            subtask_name=fusion_subtask,
            curr_outdir=outdir,
            original_args_dict=original_args_dict,
            indir_alignments=previous,
        )
        self._dependencies.log_stage_health(
            os.path.join(outdir, "pipeline_format_results.json"),
            f"{fusion_subtask}(final)",
        )


__all__ = [
    "PipelineApplicationDependencies",
    "PipelineApplicationRunner",
]
