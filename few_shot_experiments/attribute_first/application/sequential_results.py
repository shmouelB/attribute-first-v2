"""Evaluator-ready result assembly for sequential dialogue."""

from copy import deepcopy
from typing import Mapping


class SequentialPipelineResultAssembler:
    """Build evaluator-ready rows while preserving the source population."""

    def build(
        self,
        source_instances: list[dict[str, object]],
        results: Mapping[str, Mapping[str, object]],
    ) -> list[dict[str, object]]:
        """Convert terminal results without dropping any source row."""

        pipeline_results = []
        for source in source_instances:
            unique_id = source["unique_id"]
            current = deepcopy(source)
            result = results.get(unique_id)
            if result is None:
                current.update(
                    {
                        "set_of_highlights_in_context": [],
                        "response": "ERROR - missing dialogue result",
                        "skipped_reason": "missing_result",
                    }
                )
                pipeline_results.append(current)
                continue

            final_output = result.get("final_output", "")
            alignments = result.get("alignments") or []
            if str(final_output).startswith("ERROR") or not alignments:
                current.update(
                    {
                        "set_of_highlights_in_context": [],
                        "response": final_output,
                        "skipped_reason": (
                            "model_error"
                            if str(final_output).startswith("ERROR")
                            else "no_alignments"
                        ),
                    }
                )
                pipeline_results.append(current)
                continue

            highlights = []
            search_from = 0
            for alignment in alignments:
                sentence = str(alignment.get("sent_text", "")).strip()
                sentence_start = str(final_output).find(
                    sentence,
                    search_from,
                )
                if not sentence or sentence_start < 0:
                    raise ValueError(
                        f"{unique_id}: validated sentence "
                        f"{alignment.get('sent_id')} is absent from "
                        "final_output"
                    )
                search_from = sentence_start + len(sentence)
                for source_highlight in alignment.get(
                    "highlight_spans",
                    [],
                ):
                    attributed = deepcopy(source_highlight)
                    attributed["scuSentCharIdx"] = sentence_start
                    attributed["scuSentence"] = sentence
                    highlights.append(attributed)
            current.update(
                {
                    "set_of_highlights_in_context": highlights,
                    "response": final_output,
                }
            )
            current.pop("skipped_reason", None)
            pipeline_results.append(current)
        return pipeline_results


__all__ = ["SequentialPipelineResultAssembler"]
