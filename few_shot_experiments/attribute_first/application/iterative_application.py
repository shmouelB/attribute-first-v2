"""Context preparation and orchestration for iterative generation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..artifacts.output_directory import OutputDirectoryClaim
from .iterative_results import (
    IterativeRerunEvidence,
    IterativeResultPersister,
)


@dataclass
class IterativeRunContext:
    """Prepared mutable state shared across application phases."""

    args: Any
    outdir: str
    model_name: str
    n_demos: int
    debugging: bool
    num_retries: int
    num_demo_changes: int
    temperature: float
    cut_surplus: bool
    prompt_token_budget: int
    merge_cross_sents_highlights: bool
    always_with_question: bool
    no_prefix: bool
    prompt_dict: dict
    alignments: list
    original_alignments: list
    used_demos: list
    original_results: dict | None
    rerun_evidence: IterativeRerunEvidence | None
    rng: Any


@dataclass(frozen=True)
class IterativeApplicationDependencies:
    """Use-case boundaries captured by the legacy facade at call time."""

    update_args: Callable[[Any], Any]
    get_data: Callable[[Any], tuple]
    generate: Callable[..., dict]
    get_token_counter: Callable[..., Any]
    reset_token_usage: Callable[[], None]
    convert_results: Callable[..., list]
    persister: IterativeResultPersister
    rng_factory: Callable[..., Any]
    log_info: Callable[[str], None]
    artifact_sha256: Callable[[str | Path], str]


class IterativeRunContextFactory:
    """Load and validate a run without claiming its output directory."""

    def __init__(self, dependencies: IterativeApplicationDependencies):
        self._dependencies = dependencies

    @staticmethod
    def _control_value(source, name):
        if isinstance(source, dict):
            if name not in source:
                raise ValueError(
                    f"iterative protocol is missing {name}"
                )
            return source[name]
        if not hasattr(source, name):
            raise ValueError(f"iterative protocol is missing {name}")
        return getattr(source, name)

    @classmethod
    def validate_generation_controls(cls, source, label) -> None:
        """Validate the effective generation protocol before output claim."""

        n_demos = cls._control_value(source, "n_demos")
        if type(n_demos) is not int or n_demos < 0:
            raise ValueError(
                f"{label} n_demos must be a non-negative integer"
            )
        num_retries = cls._control_value(source, "num_retries")
        if type(num_retries) is not int or num_retries < 1:
            raise ValueError(
                f"{label} num_retries must be a positive integer"
            )
        num_demo_changes = cls._control_value(
            source,
            "num_demo_changes",
        )
        if type(num_demo_changes) is not int or num_demo_changes < 0:
            raise ValueError(
                f"{label} num_demo_changes must be a non-negative integer"
            )
        temperature = cls._control_value(source, "temperature")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError(
                f"{label} temperature must be numeric between 0 and 2"
            )

    @staticmethod
    def _optional_control(source, name, default):
        if isinstance(source, dict):
            return source.get(name, default)
        return getattr(source, name, default)

    @classmethod
    def prompt_controls(cls, source, label) -> tuple[bool, int]:
        """Resolve prompt-shaping controls from one effective protocol."""

        cut_surplus = cls._optional_control(
            source,
            "cut_surplus",
            False,
        )
        if type(cut_surplus) is not bool:
            raise ValueError(f"{label} cut_surplus must be boolean")
        prompt_token_budget = cls._optional_control(
            source,
            "prompt_token_budget",
            30000,
        )
        if (
            type(prompt_token_budget) is not int
            or prompt_token_budget < 1
        ):
            raise ValueError(
                f"{label} prompt_token_budget must be a positive integer"
            )
        return cut_surplus, prompt_token_budget

    @classmethod
    def validate_invocation_controls(cls, args) -> None:
        """Validate controls owned by the current invocation."""

        seed = getattr(args, "seed", 20260728)
        if type(seed) is not int:
            raise ValueError("iterative seed must be an integer")
        cut_surplus, prompt_token_budget = cls.prompt_controls(
            args,
            "iterative",
        )
        if not hasattr(args, "seed"):
            args.seed = seed
        if not hasattr(args, "cut_surplus"):
            args.cut_surplus = cut_surplus
        if not hasattr(args, "prompt_token_budget"):
            args.prompt_token_budget = prompt_token_budget

    @staticmethod
    def output_directory(args) -> str:
        no_prefix_suffix = "_no_prefix" if args.no_prefix else ""
        merged_suffix = (
            "_merged_cross_sents_sep"
            if args.merge_cross_sents_highlights
            else ""
        )
        if args.rerun:
            if not args.rerun_path:
                raise ValueError(
                    "--rerun requires a source --rerun-path"
                )
            if not args.outdir:
                raise ValueError(
                    "--rerun requires a distinct derived --outdir"
                )
            source = Path(args.rerun_path).expanduser().resolve()
            derived = Path(args.outdir).expanduser().resolve()
            if source == derived:
                raise ValueError(
                    "rerun --outdir must be distinct from its parent"
                )
            return str(derived)
        if args.outdir:
            return args.outdir
        return (
            f"results/{args.split}/{args.setting}/"
            "iterative_sentence_generation"
            f"{no_prefix_suffix}{merged_suffix}"
        )

    @staticmethod
    def _validated_fresh_data(prompt_dict, alignments, args):
        if not isinstance(prompt_dict, dict):
            raise ValueError("iterative prompt data must be an object")
        demos = prompt_dict.get("demos")
        if not isinstance(demos, list):
            raise ValueError(
                "iterative prompt data must contain a demos list"
            )
        if not isinstance(alignments, list):
            raise ValueError("iterative alignments must be a list")
        IterativeRunContextFactory._alignment_ids(
            alignments,
            "iterative alignments",
        )
        if type(args.n_demos) is not int or args.n_demos < 0:
            raise ValueError("n_demos must be a non-negative integer")
        required_demos = 3 if args.debugging else args.n_demos
        if required_demos > len(demos):
            raise ValueError(
                "not enough demonstrations for iterative generation"
            )
        return demos

    @staticmethod
    def _alignment_ids(alignments, label):
        identifiers = set()
        for index, alignment in enumerate(alignments, start=1):
            if not isinstance(alignment, dict):
                raise ValueError(
                    f"{label} row {index} must be an object"
                )
            unique_id = alignment.get("unique_id")
            if not isinstance(unique_id, str) or not unique_id:
                raise ValueError(
                    f"{label} row {index} has no non-empty unique_id"
                )
            if unique_id in identifiers:
                raise ValueError(
                    f"{label} contains duplicate unique_id {unique_id!r}"
                )
            identifiers.add(unique_id)
        return identifiers

    @staticmethod
    def _validated_rerun_payload(
        original_args,
        prompt_dict,
        alignments,
        used_demos,
        original_results,
    ):
        if not isinstance(original_args, dict):
            raise ValueError("iterative rerun args must be an object")
        if not isinstance(prompt_dict, dict):
            raise ValueError("iterative rerun prompt data must be an object")
        demos = prompt_dict.get("demos")
        if not isinstance(demos, list):
            raise ValueError(
                "iterative rerun prompt data must contain a demos list"
            )
        if not isinstance(alignments, list):
            raise ValueError("iterative rerun alignments must be a list")
        alignment_ids = IterativeRunContextFactory._alignment_ids(
            alignments,
            "iterative rerun alignments",
        )
        if not isinstance(used_demos, list):
            raise ValueError(
                "iterative rerun demonstrations must be a list"
            )
        if not isinstance(original_results, dict):
            raise ValueError("iterative rerun results must be an object")
        for unique_id, result in original_results.items():
            if not isinstance(unique_id, str) or not unique_id:
                raise ValueError(
                    "iterative rerun results contain an invalid unique_id"
                )
            if not isinstance(result, dict):
                raise ValueError(
                    f"iterative rerun result {unique_id!r} must be an object"
                )
            sentences = result.get("generated_summary_sents", [])
            if not isinstance(sentences, list):
                raise ValueError(
                    "iterative rerun generated_summary_sents must be a list"
                )
        return demos, alignment_ids

    def fresh_context(self, args, outdir) -> IterativeRunContext:
        prompt_dict, alignments = self._dependencies.get_data(args)
        demos = self._validated_fresh_data(
            prompt_dict,
            alignments,
            args,
        )
        original_alignments = deepcopy(alignments)
        rng = self._dependencies.rng_factory(
            getattr(args, "seed", 20260728)
        )
        demo_ids = (
            rng.choice(
                len(demos),
                args.n_demos,
                replace=False,
            )
            if not args.debugging
            else [0, 2]
        )
        used_demos = [demos[demo_id] for demo_id in demo_ids]
        return IterativeRunContext(
            args=args,
            outdir=outdir,
            model_name=args.model_name,
            n_demos=args.n_demos,
            debugging=args.debugging,
            num_retries=args.num_retries,
            num_demo_changes=args.num_demo_changes,
            temperature=args.temperature,
            cut_surplus=args.cut_surplus,
            prompt_token_budget=args.prompt_token_budget,
            merge_cross_sents_highlights=(
                args.merge_cross_sents_highlights
            ),
            always_with_question=args.always_with_question,
            no_prefix=args.no_prefix,
            prompt_dict=prompt_dict,
            alignments=alignments,
            original_alignments=original_alignments,
            used_demos=used_demos,
            original_results=None,
            rerun_evidence=None,
            rng=rng,
        )

    def rerun_context(self, args, outdir) -> IterativeRunContext:
        if not args.rerun_path:
            raise ValueError(
                "if passing --rerun, also pass relevant --rerun-path."
            )
        source_directory = Path(args.rerun_path).expanduser().resolve()
        if not source_directory.is_dir():
            raise ValueError(
                f"rerun path is not a run directory: {source_directory}"
            )
        persister = self._dependencies.persister
        results_path = source_directory / "results.json"
        source_sha256 = self._dependencies.artifact_sha256(results_path)
        original_args = persister.load_json(
            source_directory / "args.json"
        )
        self.validate_generation_controls(
            original_args,
            "iterative rerun parent",
        )
        cut_surplus, prompt_token_budget = self.prompt_controls(
            original_args,
            "iterative rerun parent",
        )
        prompt_dict, alignments = self._dependencies.get_data(original_args)
        original_alignments = deepcopy(alignments)
        used_demos = persister.load_json(
            source_directory / "used_demonstrations.json"
        )
        original_results = persister.load_json(results_path)
        demos, alignment_ids = self._validated_rerun_payload(
            original_args,
            prompt_dict,
            alignments,
            used_demos,
            original_results,
        )
        if self._dependencies.artifact_sha256(results_path) != source_sha256:
            raise RuntimeError(
                "iterative rerun parent changed while it was loaded"
            )
        error_ids = [
            unique_id
            for unique_id, value in original_results.items()
            if any(
                isinstance(element, str)
                and element.startswith("ERROR")
                for element in value.get("generated_summary_sents", [])
            )
        ]
        missing_error_ids = sorted(set(error_ids) - alignment_ids)
        if missing_error_ids:
            raise ValueError(
                "iterative rerun ERROR parent UIDs are missing from "
                f"current alignments: {missing_error_ids}"
            )
        selected_alignments = [
            element
            for element in alignments
            if element["unique_id"] in error_ids
        ]
        n_demos = (
            args.rerun_n_demos
            if args.rerun_n_demos is not None
            else original_args["n_demos"]
        )
        if type(n_demos) is not int or n_demos < 0:
            raise ValueError(
                "rerun_n_demos must be a non-negative integer"
            )
        temperature = (
            args.rerun_temperature
            if args.rerun_temperature is not None
            else original_args["temperature"]
        )
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError(
                "rerun_temperature must be numeric between 0 and 2"
            )
        rng = self._dependencies.rng_factory(
            getattr(args, "seed", 20260728)
        )
        if n_demos > len(demos):
            raise ValueError(
                "not enough demonstrations for iterative rerun: "
                f"requested {n_demos}, available {len(demos)}"
            )
        if args.rerun_n_demos is None:
            if len(used_demos) != n_demos:
                raise ValueError(
                    "iterative rerun archived demonstrations conflict "
                    f"with n_demos={n_demos}"
                )
            if any(demo not in demos for demo in used_demos):
                raise ValueError(
                    "iterative rerun archived demonstrations are absent "
                    "from the prompt data"
                )
        else:
            selected_demo_ids = rng.choice(
                len(demos),
                n_demos,
                replace=False,
            )
            used_demos = [
                demos[demo_id] for demo_id in selected_demo_ids
            ]
        return IterativeRunContext(
            args=args,
            outdir=outdir,
            model_name=original_args["model_name"],
            n_demos=n_demos,
            debugging=original_args["debugging"],
            num_retries=original_args["num_retries"],
            num_demo_changes=original_args["num_demo_changes"],
            temperature=temperature,
            cut_surplus=cut_surplus,
            prompt_token_budget=prompt_token_budget,
            merge_cross_sents_highlights=original_args.get(
                "merge_cross_sents_highlights",
                False,
            ),
            always_with_question=original_args.get(
                "always_with_question",
                False,
            ),
            no_prefix=original_args.get("no_prefix", False),
            prompt_dict=prompt_dict,
            alignments=selected_alignments,
            original_alignments=original_alignments,
            used_demos=used_demos,
            original_results=original_results,
            rerun_evidence=IterativeRerunEvidence(
                source_directory=source_directory,
                results_path=results_path,
                results_sha256=source_sha256,
                retried_ids=tuple(sorted(error_ids)),
            ),
            rng=rng,
        )

    @staticmethod
    def args_snapshot(context: IterativeRunContext) -> dict:
        snapshot = {
            key: value
            for key, value in vars(context.args).items()
            if not str(key).startswith("_")
        }
        snapshot.update(
            {
                "model_name": context.model_name,
                "n_demos": context.n_demos,
                "temperature": context.temperature,
                "cut_surplus": context.cut_surplus,
                "prompt_token_budget": context.prompt_token_budget,
            }
        )
        if context.rerun_evidence is not None:
            snapshot["rerun_parent"] = str(
                context.rerun_evidence.source_directory
            )
        return snapshot


class IterativeSentenceGenerationApplication:
    """Coordinate preparation, generation, conversion, and persistence."""

    def __init__(self, dependencies: IterativeApplicationDependencies):
        self._dependencies = dependencies
        self._context_factory = IterativeRunContextFactory(dependencies)

    @staticmethod
    def _output_directory(args) -> str:
        return IterativeRunContextFactory.output_directory(args)

    def _fresh_context(self, args, outdir) -> IterativeRunContext:
        return self._context_factory.fresh_context(args, outdir)

    def _rerun_context(self, args, outdir) -> IterativeRunContext:
        return self._context_factory.rerun_context(args, outdir)

    @staticmethod
    def _args_snapshot(context: IterativeRunContext) -> dict:
        return IterativeRunContextFactory.args_snapshot(context)

    @staticmethod
    def _claim_output(context: IterativeRunContext) -> None:
        pipeline_root = getattr(
            context.args,
            "_pipeline_run_root",
            None,
        )
        if pipeline_root is None:
            claimed = OutputDirectoryClaim.claim(
                context.outdir,
                owner="iterative-sentence-generation-v1",
            )
        else:
            claimed = OutputDirectoryClaim.prepare_child(
                context.outdir,
                owner_root=pipeline_root,
            )
        context.outdir = str(claimed)

    def _prepare(self, args) -> IterativeRunContext:
        if not args.config_file and (not args.setting or not args.split):
            raise ValueError(
                "If no config file is passed, then must explicitly determine "
                "setting and split."
            )
        if args.config_file:
            args = self._dependencies.update_args(args)
        self._context_factory.validate_invocation_controls(args)
        if not args.rerun:
            self._context_factory.validate_generation_controls(
                args,
                "iterative generation",
            )
        outdir = self._output_directory(args)
        self._dependencies.log_info(f"saving results to {outdir}")
        context = (
            self._rerun_context(args, outdir)
            if args.rerun
            else self._fresh_context(args, outdir)
        )
        self._claim_output(context)
        self._dependencies.persister.write_args(
            context.outdir,
            self._args_snapshot(context),
        )
        return context

    @staticmethod
    def _merge_results(context, responses):
        final_results = (
            context.original_results
            if context.original_results is not None
            else {unique_id: {} for unique_id in responses}
        )
        for unique_id, response in responses.items():
            final_results[unique_id]["gold_summary"] = [
                element["response"]
                for element in context.alignments
                if element["unique_id"] == unique_id
            ][0]
            final_results[unique_id].update(response)
        return final_results

    def run(self, args) -> None:
        context = self._prepare(args)
        self._dependencies.reset_token_usage()
        responses = self._dependencies.generate(
            alignments_dict=context.alignments,
            prompt_dict=context.prompt_dict,
            used_demos=context.used_demos,
            model_name=context.model_name,
            num_retries=context.num_retries,
            debugging=context.debugging,
            n_demos=context.n_demos,
            num_demo_changes=context.num_demo_changes,
            temperature=context.temperature,
            merge_cross_sents_highlights=(
                context.merge_cross_sents_highlights
            ),
            tkn_counter=self._dependencies.get_token_counter(
                context.model_name,
                context.prompt_token_budget,
            ),
            cut_surplus=context.cut_surplus,
            always_with_question=context.always_with_question,
            no_prefix=context.no_prefix,
            rng=context.rng,
        )
        final_results = self._merge_results(context, responses)
        conversion_error = None
        try:
            pipeline_results = self._dependencies.convert_results(
                final_results,
                context.original_alignments,
            )
            if pipeline_results is None:
                raise ValueError(
                    "iterative conversion returned no pipeline results"
                )
        except Exception as exc:
            conversion_error = exc
            pipeline_results = None
            self._dependencies.log_info(
                "The conversion to the pipeline format failed; raw results "
                "will be retained and the run will report failure."
            )
        if context.rerun_evidence is not None:
            context.rerun_evidence.verify_unchanged(
                self._dependencies.artifact_sha256
            )
        self._dependencies.persister.persist(
            outdir=context.outdir,
            used_demos=context.used_demos,
            final_results=final_results,
            pipeline_format_results=pipeline_results,
            model_name=context.model_name,
        )
        if context.rerun_evidence is not None:
            context.rerun_evidence.verify_unchanged(
                self._dependencies.artifact_sha256
            )
            self._dependencies.persister.write_rerun_provenance(
                context.outdir,
                context.rerun_evidence.provenance(
                    model_name=context.model_name,
                    n_demos=context.n_demos,
                    temperature=context.temperature,
                    seed=getattr(context.args, "seed", 20260728),
                ),
            )
        if conversion_error is not None:
            raise RuntimeError(
                "iterative pipeline conversion failed"
            ) from conversion_error


__all__ = [
    "IterativeApplicationDependencies",
    "IterativeRunContext",
    "IterativeRunContextFactory",
    "IterativeSentenceGenerationApplication",
]
