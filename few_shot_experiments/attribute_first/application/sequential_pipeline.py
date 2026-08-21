"""Population orchestration and fail-closed persistence for sequential dialogue."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Callable, Mapping

from .sequential_contracts import (
    HIGHLIGHT_END,
    HIGHLIGHT_START,
    SEQUENTIAL_PROTOCOL,
)
from ..artifacts.output_directory import OutputDirectoryClaim
from ..ports import ArtifactStore


@dataclass(frozen=True)
class SequentialPipelineDependencies:
    """All facade-owned boundaries used by the full sequential run."""

    get_data: Callable[[object], tuple[dict[str, object], list[dict]]]
    get_prompt_structures: Callable[..., dict[str, object]]
    construct_prompts: Callable[..., tuple]
    get_token_counter: Callable[..., object]
    reset_token_usage: Callable[[], None]
    get_token_usage: Callable[[], Mapping[str, int]]
    run_instance: Callable[..., tuple[dict[str, object], dict[str, int]]]
    save_results: Callable[[str, list, dict, object], None]
    get_environment_flags: Callable[[], Mapping[str, bool]]
    artifact_store: ArtifactStore
    build_pipeline_results: Callable[
        [list[dict[str, object]], Mapping[str, Mapping[str, object]]],
        list[dict[str, object]],
    ]
    executor_factory: Callable[..., object] = ThreadPoolExecutor
    completed_futures: Callable[..., object] = as_completed
    log_info: Callable[[str], None] = logging.info
    log_exception: Callable[[str], None] = logging.exception
    protocol: Mapping[str, object] | None = None
    highlight_start: str = HIGHLIGHT_START
    highlight_end: str = HIGHLIGHT_END


class SequentialDialoguePipelineRunner:
    """Prepare, execute, and persist one exact fixed population."""

    def __init__(self, dependencies: SequentialPipelineDependencies) -> None:
        self._dependencies = dependencies

    @staticmethod
    def _require_exact_coverage(
        label: str,
        expected_ids,
        actual_ids,
    ) -> None:
        expected = tuple(str(value) for value in expected_ids)
        actual = tuple(str(value) for value in actual_ids)
        if len(set(expected)) != len(expected):
            raise ValueError(
                f"{label}: source population contains duplicate unique_id"
            )
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise ValueError(
                f"{label}: fixed-population coverage mismatch "
                f"(missing={sorted(set(expected) - set(actual))}, "
                f"extra={sorted(set(actual) - set(expected))})"
            )

    @staticmethod
    def _validated_num_retries(args: object) -> int:
        value = getattr(args, "num_retries", 3)
        if type(value) is not int or value <= 0:
            raise ValueError(
                "--num-retries must be a positive integer"
            )
        return value

    def run(self, args: object) -> None:
        """Claim a new output directory and execute the full population."""

        if getattr(args, "max_examples", None) is not None:
            raise ValueError(
                "--max-examples is forbidden for the fixed population"
            )
        num_retries = self._validated_num_retries(args)
        prompt_dict, alignments = self._dependencies.get_data(args)
        always_with_question = args.setting == "LFQA"
        prompt_structures = self._dependencies.get_prompt_structures(
            prompt_dict,
            args.setting,
            "content_selection",
            False,
            always_with_question,
            True,
        )
        used_demos, content_selection_prompts, _, _ = (
            self._dependencies.construct_prompts(
                prompt_dict=prompt_dict,
                alignments_dict=alignments,
                n_demos=args.n_demos,
                debugging=False,
                merge_cross_sents_highlights=False,
                specific_prompt_details=prompt_structures,
                tkn_counter=self._dependencies.get_token_counter(
                    args.model,
                    getattr(args, "prompt_token_budget", None),
                ),
                no_highlights=True,
                cut_surplus=False,
                prct_surplus=None,
                seed=getattr(args, "seed", None),
            )
        )
        clustering_instruction = prompt_dict[
            "instruction-clustering"
        ].replace(
            "{HS}",
            self._dependencies.highlight_start,
        ).replace(
            "{HE}",
            self._dependencies.highlight_end,
        )
        fusion_instruction = prompt_dict[
            "instruction-next-cluster-fusion"
        ].replace(
            "{HS}",
            self._dependencies.highlight_start,
        ).replace(
            "{HE}",
            self._dependencies.highlight_end,
        )
        gold = {
            element["unique_id"]: element.get("response")
            for element in alignments
        }

        source_ids = [element["unique_id"] for element in alignments]
        self._require_exact_coverage(
            "content-selection prompts",
            source_ids,
            content_selection_prompts,
        )
        items = [
            (
                element["unique_id"],
                content_selection_prompts[element["unique_id"]],
            )
            for element in alignments
        ]
        pipeline_root = getattr(args, "_pipeline_run_root", None)
        if pipeline_root is None:
            outdir = OutputDirectoryClaim.claim(
                args.outdir,
                owner="sequential-dialogue-generation-v1",
            )
        else:
            outdir = OutputDirectoryClaim.prepare_child(
                args.outdir,
                owner_root=pipeline_root,
            )

        self._dependencies.reset_token_usage()
        results = self._execute(
            args=args,
            alignments=alignments,
            items=items,
            gold=gold,
            clustering_instruction=clustering_instruction,
            fusion_instruction=fusion_instruction,
            num_retries=num_retries,
        )
        self._require_exact_coverage(
            "dialogue results",
            source_ids,
            results,
        )
        self._persist(
            args=args,
            outdir=outdir,
            alignments=alignments,
            used_demos=used_demos,
            results=results,
        )

    def _execute(
        self,
        *,
        args: object,
        alignments: list[dict[str, object]],
        items: list[tuple[str, str]],
        gold: Mapping[str, object],
        clustering_instruction: str,
        fusion_instruction: str,
        num_retries: int = 3,
    ) -> dict[str, dict[str, object]]:
        source_by_uid = {
            element["unique_id"]: element for element in alignments
        }

        def work(item: tuple[str, str]):
            unique_id, content_selection_prompt = item
            try:
                result, _ = self._dependencies.run_instance(
                    source_by_uid[unique_id],
                    content_selection_prompt,
                    clustering_instruction,
                    fusion_instruction,
                    args.model,
                    num_retries=num_retries,
                )
            except (KeyError, AttributeError, TypeError):
                raise
            except Exception as exc:
                result = {
                    "final_output": f"ERROR - {exc}",
                    "alignments": [],
                    "protocol_trace": {
                        "content_selection_raw": None,
                        "clustering_raw": None,
                        "content_selection": [],
                        "clustering": [],
                        "fusion": [],
                        "runtime_error": (
                            f"{type(exc).__name__}: {exc}"
                        ),
                    },
                }
            result["gold_summary"] = gold.get(unique_id)
            return unique_id, result

        results: dict[str, dict[str, object]] = {}
        with self._dependencies.executor_factory(
            max_workers=args.concurrency
        ) as executor:
            futures = [executor.submit(work, item) for item in items]
            for index, future in enumerate(
                self._dependencies.completed_futures(futures),
                start=1,
            ):
                unique_id, result = future.result()
                results[unique_id] = result
                if index % 5 == 0:
                    self._dependencies.log_info(
                        f"  {index}/{len(items)} done"
                    )
        return results

    def _persist(
        self,
        *,
        args: object,
        outdir: Path,
        alignments: list[dict[str, object]],
        used_demos: list,
        results: dict[str, dict[str, object]],
    ) -> None:
        conversion_error = None
        try:
            pipeline_results = self._dependencies.build_pipeline_results(
                alignments,
                results,
            )
            if pipeline_results is None:
                raise ValueError(
                    "sequential conversion returned no pipeline results"
                )
        except Exception as exc:
            conversion_error = exc
            self._dependencies.log_exception(
                "[dialogue_seq] pipeline conversion failed; raw results "
                "will be retained and the run will report failure"
            )
            pipeline_results = None
        self._dependencies.save_results(
            str(outdir),
            used_demos,
            results,
            pipeline_results,
        )

        environment_flags = dict(
            self._dependencies.get_environment_flags()
        )
        protocol = deepcopy(
            self._dependencies.protocol or SEQUENTIAL_PROTOCOL
        )
        protocol["environment_flags"] = environment_flags
        args_snapshot = dict(vars(args))
        args_snapshot.update(
            {
                "environment_flags": environment_flags,
                "protocol": protocol,
            }
        )
        self._dependencies.artifact_store.write_json(
            outdir / "args.json",
            args_snapshot,
        )

        usage = dict(self._dependencies.get_token_usage())
        usage["subtask"] = "dialogue_sequential"
        usage["model"] = args.model
        self._dependencies.artifact_store.write_json(
            outdir / "token_usage.json",
            usage,
        )
        valid = sum(
            1
            for result in results.values()
            if not str(result["final_output"]).startswith("ERROR")
        )
        cached_percentage = (
            100 * usage["cached"] / usage["prompt"]
            if usage["prompt"]
            else 0
        )
        self._dependencies.log_info(
            f"[dialogue_seq] {args.setting}: {valid}/{len(results)} valid | "
            f"prompt={usage['prompt']} cached={usage['cached']} "
            f"({cached_percentage:.1f}%) calls={usage['calls']}"
        )
        if conversion_error is not None:
            raise RuntimeError(
                "sequential dialogue pipeline conversion failed"
            ) from conversion_error


__all__ = [
    "SequentialDialoguePipelineRunner",
    "SequentialPipelineDependencies",
]
