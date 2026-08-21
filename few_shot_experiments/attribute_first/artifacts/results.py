"""Conversion of fusion alignments into attributable pipeline rows."""

from copy import deepcopy


class PipelineResultBuilder:
    """Build pipeline JSONL rows while preserving source order and errors."""

    @staticmethod
    def _error_row(source, result):
        current = deepcopy(source)
        final_output = result.get("final_output", "")
        skipped_reason = result.get("upstream_skipped_reason")
        if not skipped_reason:
            skipped_reason = (
                "model_error"
                if str(final_output).startswith("ERROR")
                else "no_alignments"
            )
        current.update(
            {
                "set_of_highlights_in_context": [],
                "response": final_output,
                "skipped_reason": skipped_reason,
                "upstream_error": (
                    final_output
                    if str(final_output).startswith("ERROR")
                    else "ERROR - coherence result has no alignments"
                ),
            }
        )
        return current

    @staticmethod
    def _attributed_highlights(final_output, alignments):
        highlights = []
        search_from = 0
        for alignment in alignments:
            sentence = alignment["sent_text"].strip()
            sentence_start = final_output.find(sentence, search_from)
            if sentence_start < 0:
                raise ValueError(
                    f"validated sentence {alignment['sent_id']} is absent "
                    "from final_output"
                )
            search_from = sentence_start + len(sentence)
            for source_highlight in alignment.get("highlight_spans", []):
                attributed = deepcopy(source_highlight)
                attributed["scuSentCharIdx"] = sentence_start
                attributed["scuSentence"] = sentence
                highlights.append(attributed)
        return highlights

    def build(self, source_instances, results):
        pipeline_results = []
        for source in source_instances:
            unique_id = source["unique_id"]
            result = results.get(unique_id)
            if result is None:
                missing = deepcopy(source)
                missing.update(
                    {
                        "set_of_highlights_in_context": [],
                        "response": "ERROR - missing coherence result",
                        "skipped_reason": "missing_result",
                    }
                )
                pipeline_results.append(missing)
                continue

            final_output = result.get("final_output", "")
            alignments = result.get("alignments") or []
            if str(final_output).startswith("ERROR") or not alignments:
                pipeline_results.append(self._error_row(source, result))
                continue
            current = deepcopy(source)
            try:
                highlights = self._attributed_highlights(
                    final_output,
                    alignments,
                )
            except Exception as exc:
                conversion_error = (
                    "ERROR - attribution conversion failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                result.update(
                    {"final_output": conversion_error, "alignments": []}
                )
                current.update(
                    {
                        "set_of_highlights_in_context": [],
                        "response": conversion_error,
                        "skipped_reason": "attribution_conversion_error",
                        "upstream_error": conversion_error,
                    }
                )
                pipeline_results.append(current)
                continue
            current.update(
                {
                    "set_of_highlights_in_context": highlights,
                    "response": final_output,
                }
            )
            current.pop("skipped_reason", None)
            current.pop("upstream_error", None)
            pipeline_results.append(current)
        return pipeline_results
