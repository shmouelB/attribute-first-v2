"""Application objects for legacy iterative sentence generation.

The public compatibility functions remain in
``run_iterative_sentence_generation.py``.  This module owns the actual
prompt-building, execution, conversion, and persistence responsibilities while
receiving every mutable boundary explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from .iterative_application import (
    IterativeApplicationDependencies,
    IterativeRunContext,
    IterativeRunContextFactory,
    IterativeSentenceGenerationApplication,
)
from .iterative_results import (
    IterativePersistenceDependencies,
    IterativeRerunEvidence,
    IterativeResultConverter,
    IterativeResultPersister,
)


@dataclass(frozen=True)
class IterativePromptDependencies:
    """Text and prompt helpers required by iterative prompt construction."""

    extract_highlights: Callable[..., list]
    get_highlighted_doc: Callable[..., dict]
    make_demo: Callable[..., tuple]
    remove_after_last_highlight: Callable[[str, str], str]


class IterativePromptBuilder:
    """Build one sentence-generation prompt without owning model execution."""

    def __init__(self, dependencies: IterativePromptDependencies):
        self._dependencies = dependencies

    @staticmethod
    def keep_specific_occurrences(
        text: str,
        substring: str,
        occurrences,
    ) -> str:
        """Keep only the selected one-based occurrences of ``substring``."""

        parts = []
        count = 0
        last_index = 0
        while True:
            index = text.find(substring, last_index)
            if index == -1:
                parts.append(text[last_index:])
                break
            count += 1
            if count in occurrences:
                parts.append(text[last_index : index + len(substring)])
            else:
                parts.append(text[last_index:index])
            last_index = index + len(substring)
        return "".join(parts)

    def adapt_demo(
        self,
        train_item: Mapping[str, Any],
        curr_cluster_ind: int,
        no_prefix: bool,
    ) -> dict:
        """Project one full demonstration onto the current cluster."""

        if no_prefix:
            division_ind = 0
        else:
            division_ind = (
                curr_cluster_ind
                if curr_cluster_ind < len(train_item["planning"]) - 1
                else len(train_item["planning"]) - 1
            )
        prefix = " ".join(
            element["output"]
            for element in train_item["planning"][:division_ind]
        )
        next_sentence = train_item["planning"][division_ind]["output"]
        relevant_highlights = train_item["planning"][division_ind][
            "highlights_cluster"
        ]
        adapted_docs = []
        for document_index, document in enumerate(train_item["docs"]):
            document_relevant_highlights = [
                element
                for element in relevant_highlights
                if element["doc"] == document_index + 1
            ]
            if document_relevant_highlights:
                document_relevant_highlights = sorted(
                    document_relevant_highlights[0]["relative_highlights"]
                )

            adapted_document = self.keep_specific_occurrences(
                document["text"],
                "{HS}",
                document_relevant_highlights,
            )
            adapted_document = self.keep_specific_occurrences(
                adapted_document,
                "{HE}",
                document_relevant_highlights,
            )

            if (
                adapted_document.replace("{HS}", "").replace("{HE}", "")
                != document["text"].replace("{HS}", "").replace("{HE}", "")
            ):
                raise ValueError(
                    "adapting a demonstration changed its document text"
                )
            if adapted_document.count("{HS}") != len(
                document_relevant_highlights
            ):
                raise ValueError(
                    "adapted demonstration has an invalid number of "
                    "highlight-start markers"
                )
            if adapted_document.count("{HE}") != len(
                document_relevant_highlights
            ):
                raise ValueError(
                    "adapted demonstration has an invalid number of "
                    "highlight-end markers"
                )
            original_highlights = self._dependencies.extract_highlights(
                document["text"],
                "{HS}",
                "{HE}",
            )
            if self._dependencies.extract_highlights(
                adapted_document,
                "{HS}",
                "{HE}",
            ) != [
                element
                for highlight_index, element in enumerate(original_highlights)
                if highlight_index + 1 in document_relevant_highlights
            ]:
                raise ValueError(
                    "adapted demonstration retained the wrong highlights"
                )
            adapted_docs.append({"text": adapted_document})

        adapted = {
            "answer": next_sentence,
            "docs": adapted_docs,
            "prefix": prefix,
        }
        if "question" in train_item:
            adapted["question"] = train_item["question"]
        return adapted

    def construct_non_demo_part(
        self,
        curr_docs,
        alignments,
        highlight_start_tkn_placeholder,
        highlight_end_tkn_placeholder,
        merge_cross_sents_highlights,
        prompt_dict,
        prefix,
        instruction_prompt,
        prompt_structure,
        always_with_question,
        instance_question,
        cut_surplus=False,
    ):
        """Render the live instance and retain its document ordering."""

        highlighted_texts = self._dependencies.get_highlighted_doc(
            docs=curr_docs,
            highlights=alignments,
            highlight_start_tkn=highlight_start_tkn_placeholder,
            highlight_end_tkn=highlight_end_tkn_placeholder,
            merge_cross_sents_highlights=merge_cross_sents_highlights,
        )
        highlighted_texts = {
            key: value
            for key, value in highlighted_texts.items()
            if highlight_start_tkn_placeholder in value
        }
        if cut_surplus:
            highlighted_texts = {
                document_name: (
                    self._dependencies.remove_after_last_highlight(
                        document_text,
                        highlight_end_tkn_placeholder,
                    )
                )
                for document_name, document_text in highlighted_texts.items()
            }

        docs_order = [
            {"doc_name": document_name, "doc_text": document_text}
            for document_name, document_text in highlighted_texts.items()
        ]
        evaluation_item = {
            "docs": [
                {"text": document["doc_text"]} for document in docs_order
            ],
            "prefix": prefix,
        }
        if always_with_question and instance_question:
            evaluation_item["question"] = instance_question

        answer_prompts = {
            "answer_prompt": prompt_dict[
                "answer_next_cluster_fusion_prompt"
            ],
            "answer_highlights_listing_prompt": prompt_dict[
                "answer_highlights_listing_prompt"
            ],
        }
        current_prompt, _ = self._dependencies.make_demo(
            item=evaluation_item,
            prompt=prompt_structure,
            doc_prompt=prompt_dict["doc_prompt"],
            instruction=instruction_prompt,
            answer_related_prompts=answer_prompts,
            highlight_start_tkn=prompt_dict["highlight_start_tkn"],
            highlight_end_tkn=prompt_dict["highlight_end_tkn"],
            test=True,
        )
        return current_prompt, docs_order

    @staticmethod
    def _prompt_protocol(
        prompt_dict,
        adapted_train_item,
        always_with_question,
    ):
        if always_with_question and "question" in adapted_train_item:
            return (
                prompt_dict[
                    "instruction-next-cluster-fusion-with-question"
                ],
                prompt_dict[
                    "demo_prompt_next_cluster_fusion_with_question"
                ],
            )
        return (
            prompt_dict["instruction-next-cluster-fusion"],
            prompt_dict["demo_prompt_next_cluster_fusion"],
        )

    def _render_demo(
        self,
        adapted_train_item,
        prompt_dict,
        instruction_prompt,
        prompt_structure,
    ):
        answer_prompts = {
            "answer_prompt": prompt_dict[
                "answer_next_cluster_fusion_prompt"
            ],
            "answer_highlights_listing_prompt": prompt_dict[
                "answer_highlights_listing_prompt"
            ],
        }
        return self._dependencies.make_demo(
            item=adapted_train_item,
            prompt=prompt_structure,
            doc_prompt=prompt_dict["doc_prompt"],
            instruction=instruction_prompt,
            answer_related_prompts=answer_prompts,
            highlight_start_tkn=prompt_dict["highlight_start_tkn"],
            highlight_end_tkn=prompt_dict["highlight_end_tkn"],
        )[0]

    def _shorten_demo(self, adapted_train_item, highlight_end):
        shortened = {
            key: [
                {
                    document_key: (
                        self._dependencies.remove_after_last_highlight(
                            document_value,
                            highlight_end,
                        )
                        if document_key == "text"
                        else document_value
                    )
                    for document_key, document_value in element.items()
                }
                for element in value
            ]
            if key == "docs"
            else value
            for key, value in adapted_train_item.items()
        }
        shortened["docs"] = [
            document for document in shortened["docs"] if document["text"]
        ]
        return shortened

    @staticmethod
    def _document_views(
        docs_order,
        prompt_dict,
        highlight_start,
        highlight_end,
    ):
        highlighted = [
            {
                "doc_name": element["doc_name"],
                "doc_text": (
                    element["doc_text"]
                    .replace(
                        highlight_start,
                        prompt_dict["highlight_start_tkn"],
                    )
                    .replace(
                        highlight_end,
                        prompt_dict["highlight_end_tkn"],
                    )
                ),
            }
            for element in docs_order
        ]
        plain = [
            {
                "doc_name": element["doc_name"],
                "doc_text": (
                    element["doc_text"]
                    .replace(highlight_start, "")
                    .replace(highlight_end, "")
                ),
            }
            for element in docs_order
        ]
        return highlighted, plain

    def construct_prompt(
        self,
        prompt_dict,
        alignments,
        curr_docs,
        used_demos,
        curr_cluster_ind,
        prefix,
        merge_cross_sents_highlights,
        tkn_counter,
        cut_surplus=False,
        always_with_question=False,
        instance_question=None,
        no_prefix=False,
    ):
        """Build the complete long and shortened prompt alternatives."""

        highlight_start = "{HS}"
        highlight_end = "{HE}"
        head_prompt = ""
        head_prompt_shorter = ""
        protocol_source = (
            {"question": instance_question}
            if instance_question
            else {}
        )
        instruction_prompt, prompt_structure = self._prompt_protocol(
            prompt_dict,
            protocol_source,
            always_with_question,
        )
        for train_item in used_demos:
            adapted_train_item = self.adapt_demo(
                train_item,
                curr_cluster_ind,
                no_prefix,
            )
            head_prompt += self._render_demo(
                adapted_train_item,
                prompt_dict,
                instruction_prompt,
                prompt_structure,
            )
            head_prompt += prompt_dict["demo_sep"]

            shortened_demo = self._shorten_demo(
                adapted_train_item,
                highlight_end,
            )
            head_prompt_shorter += self._render_demo(
                shortened_demo,
                prompt_dict,
                instruction_prompt,
                prompt_structure,
            )
            head_prompt_shorter += prompt_dict["demo_sep"]

        current_prompt, docs_order = self.construct_non_demo_part(
            curr_docs,
            alignments,
            highlight_start,
            highlight_end,
            merge_cross_sents_highlights,
            prompt_dict,
            prefix,
            instruction_prompt,
            prompt_structure,
            always_with_question,
            instance_question,
        )
        if (
            cut_surplus
            or tkn_counter["tkn_counter"].token_count(
                head_prompt + current_prompt
            )
            >= tkn_counter["tkn_max_limit"]
        ):
            current_prompt, docs_order_shorter = (
                self.construct_non_demo_part(
                    curr_docs,
                    alignments,
                    highlight_start,
                    highlight_end,
                    merge_cross_sents_highlights,
                    prompt_dict,
                    prefix,
                    instruction_prompt,
                    prompt_structure,
                    always_with_question,
                    instance_question,
                    cut_surplus=True,
                )
            )
            final_prompt = head_prompt_shorter + current_prompt
        else:
            final_prompt = head_prompt + current_prompt
            docs_order_shorter = []

        highlighted_docs, non_highlighted_docs = self._document_views(
            docs_order,
            prompt_dict,
            highlight_start,
            highlight_end,
        )
        (
            highlighted_docs_shorter,
            non_highlighted_docs_shorter,
        ) = self._document_views(
            docs_order_shorter,
            prompt_dict,
            highlight_start,
            highlight_end,
        )
        additional_data = {
            "highlighted_docs": highlighted_docs,
            "non_highlighted_docs": non_highlighted_docs,
            "highlighted_docs_shorter": highlighted_docs_shorter,
            "non_highlighted_docs_shorter": non_highlighted_docs_shorter,
            "curr_alignments": alignments,
            "curr_prefix": prefix,
        }
        return final_prompt, additional_data


def parse_iterative_sentence_response(response, prompt):
    """Parse the legacy next-sentence response shape."""

    del prompt
    if not isinstance(response, str):
        raise ValueError(
            "iterative response must contain a non-empty sentence"
        )
    parsed_response = response.strip()
    if parsed_response.lower().strip().startswith("answer:"):
        parsed_response = parsed_response.strip()[len("answer:") :].strip()
    if parsed_response.lower().strip().startswith(
        "the next sentence is:"
    ):
        parsed_response = parsed_response.strip()[
            len("the next sentence is:") :
        ].strip()
    if not parsed_response:
        raise ValueError(
            "iterative response must contain a non-empty sentence"
        )
    if "next sentence" in parsed_response.casefold():
        raise ValueError(
            '"some variation of "next sentence" appeared in the response: '
            f"{parsed_response}"
        )
    return {
        "final_output": parsed_response,
        "full_model_response": response,
    }


@dataclass(frozen=True)
class IterativeExecutionDependencies:
    """Mutable boundaries used while generating sentence by sentence."""

    construct_prompt: Callable[..., tuple]
    prompt_model: Callable[..., dict]
    parse_response: Callable[..., dict]
    progress: Callable[[Any], Any]
    log_info: Callable[[str], None]
    rng_factory: Callable[[], Any] = np.random.default_rng


class IterativeGenerationExecutor:
    """Execute the legacy per-instance, per-cluster generation state machine."""

    def __init__(self, dependencies: IterativeExecutionDependencies):
        self._dependencies = dependencies

    @staticmethod
    def _source_documents(instance):
        return {
            element["documentFile"]: element["rawDocumentText"]
            for element in instance["documents"]
        }

    @staticmethod
    def _new_instance_result(instance):
        return {
            "non_highlighted_docs_full": {
                element["documentFile"]: element["rawDocumentText"]
                for element in instance["documents"]
            }
        }

    def _call_model(
        self,
        *,
        topic_name,
        prompt,
        model_name,
        num_retries,
        temperature,
    ):
        return self._dependencies.prompt_model(
            prompts={topic_name: prompt},
            model_name=model_name,
            parse_response_fn=self._dependencies.parse_response,
            num_retries=num_retries,
            verbose=False,
            temperature=temperature,
            reset_usage=False,
        )

    def generate(
        self,
        alignments_dict,
        prompt_dict,
        used_demos,
        model_name,
        num_retries,
        debugging,
        n_demos,
        num_demo_changes,
        temperature,
        merge_cross_sents_highlights,
        tkn_counter,
        cut_surplus=False,
        always_with_question=False,
        no_prefix=False,
        rng=None,
    ):
        """Generate all instances while preserving terminal-error behavior."""

        if rng is None:
            rng = self._dependencies.rng_factory()
        active_alignments = list(alignments_dict)
        if debugging:
            active_alignments = active_alignments[:3]

        final_data_instances = {}
        for instance in self._dependencies.progress(active_alignments):
            topic_name = instance["unique_id"]
            sentence_indices = sorted(
                {
                    element["scuSentCharIdx"]
                    for element in instance[
                        "set_of_highlights_in_context"
                    ]
                }
            )
            final_data_instances[topic_name] = self._new_instance_result(
                instance
            )
            generated_summary_sents = []
            generation_history = []
            for cluster_index, sentence_index in enumerate(sentence_indices):
                current_alignments = [
                    element
                    for element in instance[
                        "set_of_highlights_in_context"
                    ]
                    if element["scuSentCharIdx"] == sentence_index
                ]
                current_used_demos = used_demos
                no_errors = False
                for demo_change_index in range(num_demo_changes + 1):
                    current_prompt, additional_data = (
                        self._dependencies.construct_prompt(
                            prompt_dict=prompt_dict,
                            alignments=current_alignments,
                            curr_docs=self._source_documents(instance),
                            used_demos=current_used_demos,
                            curr_cluster_ind=cluster_index,
                            prefix=(
                                ""
                                if no_prefix
                                else " ".join(generated_summary_sents)
                            ),
                            merge_cross_sents_highlights=(
                                merge_cross_sents_highlights
                            ),
                            tkn_counter=tkn_counter,
                            cut_surplus=cut_surplus,
                            always_with_question=always_with_question,
                            instance_question=instance.get("query"),
                            no_prefix=no_prefix,
                        )
                    )
                    responses = self._call_model(
                        topic_name=topic_name,
                        prompt=current_prompt,
                        model_name=model_name,
                        num_retries=num_retries,
                        temperature=temperature,
                    )
                    response = responses[topic_name]
                    if not response["final_output"].startswith("ERROR"):
                        generated_summary_sents.append(
                            response["final_output"]
                        )
                        history_entry = dict(additional_data)
                        history_entry.update(response)
                        history_entry["updated_demos"] = (
                            current_used_demos
                            if demo_change_index > 0
                            else []
                        )
                        generation_history.append(history_entry)
                        no_errors = True
                        break

                    if demo_change_index < num_demo_changes:
                        replacement_ids = rng.choice(
                            len(prompt_dict["demos"]),
                            n_demos,
                            replace=False,
                        )
                        current_used_demos = [
                            prompt_dict["demos"][demo_id]
                            for demo_id in replacement_ids
                        ]

                if not no_errors:
                    self._dependencies.log_info(
                        f"in instance {topic_name} couldn't generate output"
                    )
                    generated_summary_sents.append(
                        responses[topic_name]["final_output"]
                    )
                    history_entry = dict(additional_data)
                    history_entry.update(responses[topic_name])
                    history_entry["updated_demos"] = []
                    generation_history.append(history_entry)
                    break

            final_data_instances[topic_name].update(
                {
                    "generated_summary_sents": generated_summary_sents,
                    "generation_history": generation_history,
                }
            )
        return final_data_instances


__all__ = [
    "IterativeApplicationDependencies",
    "IterativeExecutionDependencies",
    "IterativeGenerationExecutor",
    "IterativePersistenceDependencies",
    "IterativeRerunEvidence",
    "IterativePromptBuilder",
    "IterativePromptDependencies",
    "IterativeResultConverter",
    "IterativeResultPersister",
    "IterativeRunContext",
    "IterativeRunContextFactory",
    "IterativeSentenceGenerationApplication",
    "parse_iterative_sentence_response",
]
