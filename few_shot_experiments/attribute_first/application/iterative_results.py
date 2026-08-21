"""Conversion and persistence for iterative sentence-generation results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class IterativeRerunEvidence:
    """Immutable link from a derived iterative run to its source results."""

    source_directory: Path
    results_path: Path
    results_sha256: str
    retried_ids: tuple[str, ...]

    @classmethod
    def capture(
        cls,
        source_directory,
        *,
        artifact_sha256: Callable[[str | Path], str],
        retried_ids,
    ) -> "IterativeRerunEvidence":
        source = Path(source_directory).expanduser().resolve()
        results_path = source / "results.json"
        return cls(
            source_directory=source,
            results_path=results_path,
            results_sha256=artifact_sha256(results_path),
            retried_ids=tuple(sorted(str(value) for value in retried_ids)),
        )

    def verify_unchanged(
        self,
        artifact_sha256: Callable[[str | Path], str],
    ) -> None:
        """Fail if the parent results changed during the derived run."""

        if artifact_sha256(self.results_path) != self.results_sha256:
            raise RuntimeError(
                "iterative rerun parent changed while the derived run "
                "executed"
            )

    def provenance(
        self,
        *,
        model_name: str,
        n_demos: int,
        temperature: float,
        seed: int,
    ) -> dict:
        """Build the portable append-only lineage artifact."""

        return {
            "schema_version": 1,
            "kind": "iterative_sentence_generation_rerun",
            "parent": {
                "run_directory": str(self.source_directory),
                "results_path": str(self.results_path),
                "sha256": self.results_sha256,
            },
            "retried_ids": list(self.retried_ids),
            "effective": {
                "model_name": model_name,
                "n_demos": n_demos,
                "temperature": temperature,
                "seed": seed,
            },
        }


class IterativeResultConverter:
    """Convert sentence-level history into the historical pipeline schema."""

    @staticmethod
    def highlights_in_context(curr_instance, *_args, **_kwargs):
        highlights_in_context = []
        final_output = ""
        current_sentence_offset = 0
        for generated_instance in curr_instance["generation_history"]:
            current_alignments = [
                {
                    key: value
                    for key, value in element.items()
                    if key != "prefix"
                }
                for element in generated_instance["curr_alignments"]
            ]
            current_alignments = [
                {
                    key: (
                        generated_instance["final_output"]
                        if key == "scuSentence"
                        else value
                    )
                    for key, value in element.items()
                }
                for element in current_alignments
            ]
            current_alignments = [
                {
                    key: (
                        float(current_sentence_offset)
                        if key == "scuSentCharIdx"
                        else value
                    )
                    for key, value in element.items()
                }
                for element in current_alignments
            ]
            highlights_in_context += current_alignments
            final_output += generated_instance["final_output"] + " "
            current_sentence_offset += (
                len(generated_instance["final_output"]) + 1
            )

        if not all(
            final_output[int(element["scuSentCharIdx"]) :].startswith(
                element["scuSentence"]
            )
            for element in highlights_in_context
        ):
            raise ValueError("scuSentence doesn't match scuSentCharIdx")
        if sum(
            len(element["curr_alignments"])
            for element in curr_instance["generation_history"]
        ) != len(highlights_in_context):
            raise ValueError(
                "num of final highlights in context doesn't match original "
                "number of highlights in context"
            )
        return highlights_in_context, final_output.strip()

    def convert(self, results, alignments_dict, *_args, **_kwargs):
        pipeline_style_data = []
        for unique_id, value in results.items():
            original_instance = deepcopy(
                [
                    element
                    for element in alignments_dict
                    if element["unique_id"] == unique_id
                ][0]
            )
            highlights, final_output = self.highlights_in_context(
                value,
                original_instance,
            )
            original_instance.update(
                {
                    "set_of_highlights_in_context": highlights,
                    "response": final_output,
                    "response_sents": value["generated_summary_sents"],
                    "gold_summary": value["gold_summary"],
                }
            )
            pipeline_style_data.append(original_instance)
        return pipeline_style_data


@dataclass(frozen=True)
class IterativePersistenceDependencies:
    """Filesystem and serializer boundaries for iterative run artifacts."""

    save_results: Callable[..., None]
    get_token_usage: Callable[[], dict]
    write_json: Callable[[str, Any], None]
    read_json: Callable[[str], Any] | None = None


class IterativeResultPersister:
    """Load and persist the artifacts owned by an iterative run."""

    def __init__(self, dependencies: IterativePersistenceDependencies):
        self._dependencies = dependencies

    def load_json(self, path) -> Any:
        if self._dependencies.read_json is not None:
            return self._dependencies.read_json(str(path))
        with open(path, "r", encoding="utf-8") as source:
            return json.load(source)

    def write_args(self, outdir, args_dict) -> None:
        self._dependencies.write_json(
            str(Path(outdir) / "args.json"),
            args_dict,
        )

    def write_rerun_provenance(self, outdir, provenance) -> None:
        self._dependencies.write_json(
            str(Path(outdir) / "rerun_provenance.json"),
            provenance,
        )

    def persist(
        self,
        *,
        outdir,
        used_demos,
        final_results,
        pipeline_format_results,
        model_name,
    ) -> None:
        self._dependencies.save_results(
            outdir,
            used_demos,
            final_results,
            pipeline_format_results,
        )
        usage = dict(self._dependencies.get_token_usage())
        usage["subtask"] = "iterative_sentence_generation"
        usage["model"] = model_name
        self._dependencies.write_json(
            str(Path(outdir) / "token_usage.json"),
            usage,
        )


__all__ = [
    "IterativePersistenceDependencies",
    "IterativeRerunEvidence",
    "IterativeResultConverter",
    "IterativeResultPersister",
]
