from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import threading

from attribute_first.stages.fic_canonical_highlights import (
    FiCCanonicalHighlightRegistry,
)
from attribute_first.stages.structured_highlights import (
    AmbiguousSourceSpan,
    SourceSpanNotFound,
    UniqueSourceSpanLocator,
)
from utils import (
    SPAN_SEP,
    find_substring,
    get_consecutive_subspans,
    longest_common_subsequence,
    rmv_spaces_and_punct,
)

COSINE_SIMILARITY_THR = 0.6
sent_transformer_model_name = "sentence-transformers/paraphrase-distilroberta-base-v1"
_sentence_transformer_resources = None
_semantic_math_resources = None
_sentence_transformer_lock = threading.Lock()


@dataclass(frozen=True)
class _InBandModelError:
    """Canonical pipeline projection for terminal model-error results."""

    message: str
    skipped_reason: str

    @classmethod
    def from_result(cls, result):
        final_output = result.get("final_output")
        if not (
            isinstance(final_output, str)
            and final_output.lstrip().startswith("ERROR")
        ):
            return None

        skipped_reason = result.get("upstream_skipped_reason")
        if not isinstance(skipped_reason, str) or not skipped_reason.strip():
            skipped_reason = "model_error"
        return cls(
            message=final_output,
            skipped_reason=skipped_reason,
        )

    def to_pipeline_row(self, source_row, result):
        pipeline_row = deepcopy(source_row)
        if "gold_summary" in result:
            pipeline_row["gold_summary"] = deepcopy(
                result["gold_summary"]
            )
        pipeline_row.update(
            {
                "set_of_highlights_in_context": [],
                "response": self.message,
                "skipped_reason": self.skipped_reason,
                "upstream_error": self.message,
            }
        )
        return pipeline_row


def _get_spacy_nlp():
    """Load the parser module only when a converter needs sentence parsing."""

    from response_parsers import _get_spacy_nlp as get_spacy_nlp

    return get_spacy_nlp()


def adapt_highlights_to_doc_alignments(*args, **kwargs):
    """Preserve the legacy re-export without importing parsers eagerly."""

    from response_parsers import (
        adapt_highlights_to_doc_alignments as adapt_highlights,
    )

    return adapt_highlights(*args, **kwargs)


def _load_sentence_transformer():
    """Load the semantic fallback backend only when a fuzzy match needs it."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(sent_transformer_model_name)
    model = AutoModel.from_pretrained(sent_transformer_model_name)
    return model, tokenizer


def _get_sentence_transformer():
    """Return one process-wide backend, publishing it atomically after loading."""
    global _sentence_transformer_resources
    if _sentence_transformer_resources is None:
        with _sentence_transformer_lock:
            if _sentence_transformer_resources is None:
                _sentence_transformer_resources = _load_sentence_transformer()
    return _sentence_transformer_resources


def _load_semantic_math_resources():
    """Import tensor and distance libraries only for semantic fallback."""

    import torch
    from scipy.spatial.distance import cosine

    return torch, cosine


def _get_semantic_math_resources():
    """Publish the lazy math backend once under the shared backend lock."""

    global _semantic_math_resources
    if _semantic_math_resources is None:
        with _sentence_transformer_lock:
            if _semantic_math_resources is None:
                _semantic_math_resources = (
                    _load_semantic_math_resources()
                )
    return _semantic_math_resources


def get_sentence_embedding(sentence, model, tokenizer):
    torch_module, _ = _get_semantic_math_resources()
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    with torch_module.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def _highlights_in_context_from_spans(doc_name, doc_text, spans, nlp, doc_sents=None):
    """Convert a list[str] spans (verbatim) into the pipeline highlight objects."""
    if not spans:
        return []
    return get_set_of_highlights_in_context_content_selection(
        doc_name=doc_name,
        doc_text=doc_text,
        highlights=spans,
        nlp=nlp,
        doc_sents=doc_sents,
    )


def _docspan_ranges(hic_item: dict):
    """Return list[(start,end)] ranges for a single highlight-in-context item."""
    offsets = hic_item.get("docSpanOffsets")
    if not offsets:
        return []
    ranges = []
    for off in offsets:
        if isinstance(off, (list, tuple)) and len(off) == 2:
            try:
                s, e = int(off[0]), int(off[1])
            except Exception:
                continue
            if s < e:
                ranges.append((s, e))
    return ranges


def _ranges_overlap(a, b):
    """True iff half-open intervals a=[s1,e1), b=[s2,e2) overlap."""
    return a[0] < b[1] and b[0] < a[1]


def _merge_set_of_highlights_in_context(existing_list, new_list):
    """Merge HA highlights into existing HS highlights, deduplicating and removing overlaps."""
    existing_list = existing_list if isinstance(existing_list, list) else []
    new_list = new_list if isinstance(new_list, list) else []

    def _key(h):
        try:
            return json.dumps(h, sort_keys=True)
        except Exception:
            return str(h)

    merged = []
    seen = set()
    ranges_by_doc = {}

    for h in existing_list:
        if not isinstance(h, dict):
            continue
        k = _key(h)
        if k in seen:
            continue
        seen.add(k)
        merged.append(h)
        doc = h.get("documentFile")
        if doc:
            ranges_by_doc.setdefault(doc, []).extend(_docspan_ranges(h))

    kept_new = []
    for h in new_list:
        if not isinstance(h, dict):
            continue
        k = _key(h)
        if k in seen:
            continue

        doc = h.get("documentFile")
        new_ranges = _docspan_ranges(h)

        if doc and new_ranges:
            existing_ranges = ranges_by_doc.get(doc, [])
            if any(_ranges_overlap(r, er) for r in new_ranges for er in existing_ranges):
                continue
            ranges_by_doc.setdefault(doc, []).extend(new_ranges)

        seen.add(k)
        kept_new.append(h)
        merged.append(h)

    return kept_new, merged


def convert_ambiguity_highlight_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    """Merge ambiguity-added highlights (HA) into the existing HS highlights."""
    nlp = _get_spacy_nlp()
    pipeline_style_data = []

    for unique_id, value in results.items():
        curr_original_inst = deepcopy([e for e in alignments_dict if e.get("unique_id") == unique_id][0])
        final_output = value.get("final_output")
        if (
            isinstance(final_output, str)
            and final_output.strip().startswith("ERROR")
        ):
            curr_original_inst.update(
                {
                    "set_of_highlights_in_context": [],
                    "context_set_of_highlights_in_context": [],
                    "new_set_of_highlights_in_context": [],
                    "skipped_reason": value.get(
                        "upstream_skipped_reason", "model_error"
                    ),
                    "upstream_error": final_output,
                }
            )
            pipeline_style_data.append(curr_original_inst)
            continue

        carry_gold_fields = [
            "gold_highlights",
            "gold_highlighted_docs",
            "gold_highlights_shorter",
            "gold_highlighted_docs_shorter",
            "gold_summary",
        ]
        for k in carry_gold_fields:
            if k in value:
                curr_original_inst[k] = value[k]

        non_highlighted_docs = value.get("non_highlighted_docs")
        if not isinstance(non_highlighted_docs, list):
            non_highlighted_docs = []

        doc_file_to_raw = {}
        doc_file_to_sents = {}
        for d in curr_original_inst.get("documents", []):
            if not isinstance(d, dict):
                continue
            df = d.get("documentFile")
            if not df:
                continue
            doc_file_to_raw[df] = d.get("rawDocumentText")
            if "documentText" in d:
                doc_file_to_sents[df] = d.get("documentText")

        ha_final = value.get("final_output", {})
        ha_hic_all = []

        if isinstance(ha_final, dict):
            for doc_key, spans in ha_final.items():
                if not isinstance(spans, list) or not doc_key.startswith("Document ["):
                    continue
                try:
                    doc_i = int(doc_key.replace("Document [", "").replace("]", "")) - 1
                except Exception:
                    continue
                if doc_i < 0 or doc_i >= len(non_highlighted_docs):
                    continue

                doc_name = non_highlighted_docs[doc_i].get("doc_name") if isinstance(non_highlighted_docs[doc_i], dict) else None
                if not doc_name:
                    continue

                raw = doc_file_to_raw.get(doc_name)
                if not raw:
                    continue

                doc_sents = doc_file_to_sents.get(doc_name)
                ha_hic_all.extend(_highlights_in_context_from_spans(doc_name, raw, spans, nlp, doc_sents=doc_sents))

        existing_hic = curr_original_inst.get(
            "set_of_highlights_in_context",
            [],
        )
        kept_new_hic, merged_hic = (
            _merge_set_of_highlights_in_context(
                existing_hic,
                ha_hic_all,
            )
        )

        # Minimal verbatim context is evidence too; overlap/deduplication is
        # the only removal policy.
        curr_original_inst["set_of_highlights_in_context"] = merged_hic
        curr_original_inst["context_set_of_highlights_in_context"] = kept_new_hic
        curr_original_inst["new_set_of_highlights_in_context"] = kept_new_hic

        if "full_model_response" in value:
            curr_original_inst["ambiguity_full_model_response"] = value["full_model_response"]
        if "removed_nested_or_duplicate" in value:
            curr_original_inst["removed_nested_or_duplicate"] = value["removed_nested_or_duplicate"]

        pipeline_style_data.append(curr_original_inst)

    return pipeline_style_data


def _unique_context_span_offsets(
    doc_name: str,
    doc_text: str,
    span_text: str,
) -> tuple[int, int]:
    """Locate one source span without silently choosing an occurrence."""

    try:
        located = UniqueSourceSpanLocator().locate(
            doc_name,
            doc_text,
            span_text,
        )
    except AmbiguousSourceSpan as exc:
        raise ValueError(
            f"ambiguous context span in {doc_name!r}: "
            f"{span_text[:80]!r} has multiple occurrences"
        ) from exc
    except SourceSpanNotFound as exc:
        raise ValueError(
            f"context span is not verbatim in {doc_name!r}: "
            f"{span_text[:80]!r}"
        ) from exc
    return located.start, located.end


def _lcs_doc_span_offsets(
    *,
    sentence: str,
    highlight: str,
    sentence_start: int,
) -> list[list[int]]:
    """Convert inclusive LCS runs into half-open document offsets."""

    lcs_details = longest_common_subsequence(sentence, highlight)
    inclusive_runs = get_consecutive_subspans(
        sorted(lcs_details[1])
    )
    return [
        [
            sentence_start + start,
            sentence_start + inclusive_end + 1,
        ]
        for start, inclusive_end in inclusive_runs
    ]


def _ordered_sentence_offsets(
    document_text: str,
    document_sentences: list[str],
) -> list[tuple[int, int]]:
    """Map declared sentence segments to their source occurrences in order."""

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for sentence_index, sentence in enumerate(document_sentences):
        if not isinstance(sentence, str):
            raise TypeError(
                f"documentText[{sentence_index}] must be a string"
            )
        if not rmv_spaces_and_punct(sentence):
            offsets.append((cursor, cursor))
            continue

        relative_start, relative_end = find_substring(
            document_text[cursor:],
            sentence,
        )
        if relative_start < 0 or relative_end <= relative_start:
            raise ValueError(
                "documentText segment cannot be mapped in source order: "
                f"index={sentence_index}"
            )
        start = cursor + relative_start
        end = cursor + relative_end
        offsets.append((start, end))
        cursor = end
    return offsets


def get_set_of_highlights_in_context_content_selection(doc_name, doc_text, highlights, nlp, doc_sents, *args, **kwargs):
    if not doc_sents:
        doc_sents = [sent.text for sent in nlp(doc_text).sents]
    sents_idx_limits = _ordered_sentence_offsets(doc_text, doc_sents)

    highlights_in_context_list = []
    for h in highlights:
        if not rmv_spaces_and_punct(h):
            raise ValueError(
                f"context span for {doc_name!r} must be non-empty"
            )
        h = h.strip()
        h_idx_limits = _unique_context_span_offsets(
            doc_name,
            doc_text,
            h,
        )

        relevant_sents_i = sorted([i for i,lims in enumerate(sents_idx_limits) if set(range(lims[0],lims[1])).intersection(set(range(h_idx_limits[0], h_idx_limits[1]))) and rmv_spaces_and_punct(doc_sents[i])])
        relevant_sents_i = sorted([i for i in relevant_sents_i if not any(rmv_spaces_and_punct(doc_sents[i]) in rmv_spaces_and_punct(doc_sents[j]) and rmv_spaces_and_punct(doc_sents[i])!=rmv_spaces_and_punct(doc_sents[j]) for j in relevant_sents_i)])

        if len(relevant_sents_i)>1:
            for sentence_index in relevant_sents_i:
                curr_sents_idx_limits = sents_idx_limits[sentence_index]
                curr_h_idx_limits = get_consecutive_subspans(sorted(list(set(range(h_idx_limits[0],h_idx_limits[1]+1)).intersection(range(curr_sents_idx_limits[0], curr_sents_idx_limits[1]+1)))))
                docSentCharIdx = str(curr_sents_idx_limits[0])
                docSentText = doc_text[curr_sents_idx_limits[0]:curr_sents_idx_limits[1]]
                docSpanOffsets = [list(subspan) for subspan in curr_h_idx_limits]
                docSpanText = SPAN_SEP.join([doc_text[subspan[0]:subspan[1]] for subspan in curr_h_idx_limits])

                highlights_in_context_list.append({"documentFile" : doc_name,
                                                "scuSentCharIdx" : None,
                                                "scuSentence" : None,
                                                "docSentCharIdx" : docSentCharIdx,
                                                "docSentText" : docSentText,
                                                "docSpanText" : docSpanText,
                                                "docSpanOffsets" : docSpanOffsets,
                                                "sent_idx" : sentence_index})

        elif len(relevant_sents_i)==0:
            potentially_containing_sents_i = [i for i,sent in enumerate(doc_sents) if rmv_spaces_and_punct(h) in rmv_spaces_and_punct(sent)]
            if not potentially_containing_sents_i:
                raise ValueError("highlight wasn't found")
            relevant_sents_i = potentially_containing_sents_i[0]
            docSentCharIdx = str(sents_idx_limits[relevant_sents_i][0])
            docSentText = doc_text[sents_idx_limits[relevant_sents_i][0]:sents_idx_limits[relevant_sents_i][1]]

            docSpanOffsets = _lcs_doc_span_offsets(
                sentence=doc_sents[relevant_sents_i],
                highlight=h,
                sentence_start=int(docSentCharIdx),
            )
            docSpanText = SPAN_SEP.join([doc_text[subspan[0]:subspan[1]] for subspan in docSpanOffsets])

            highlights_in_context_list.append({"documentFile" : doc_name,
                                            "scuSentCharIdx" : None,
                                            "scuSentence" : None,
                                            "docSentCharIdx" : docSentCharIdx,
                                            "docSentText" : docSentText,
                                            "docSpanText" : docSpanText,
                                            "docSpanOffsets" : docSpanOffsets,
                                            "sent_idx" : relevant_sents_i})
        else:
            relevant_sents_i = relevant_sents_i[0]
            docSentCharIdx = [str(sent[0]) for sent_i,sent in enumerate(sents_idx_limits) if sent_i==relevant_sents_i][0]
            docSentText = [doc_text[sent_span[0]:sent_span[1]] for sent_i,sent_span in enumerate(sents_idx_limits) if sent_i==relevant_sents_i][0]
            docSpanOffsets = [list(h_idx_limits)]
            docSpanText = doc_text[h_idx_limits[0]:h_idx_limits[1]]
            highlights_in_context_list.append({"documentFile" : doc_name,
                                            "scuSentCharIdx" : None,
                                            "scuSentence" : None,
                                            "docSentCharIdx" : docSentCharIdx,
                                            "docSentText" : docSentText,
                                            "docSpanText" : docSpanText,
                                            "docSpanOffsets" : docSpanOffsets,
                                            "sent_idx" : relevant_sents_i})

    deduplicated_highlights = []
    seen_highlights = set()
    for highlight in highlights_in_context_list:
        canonical_highlight = json.dumps(highlight, sort_keys=True)
        if canonical_highlight in seen_highlights:
            continue
        seen_highlights.add(canonical_highlight)
        deduplicated_highlights.append(highlight)

    # The caller visits source documents in their declared order. Within each
    # document, use source offsets as the canonical order so planned variants
    # receive stable highlight IDs independently of PYTHONHASHSEED.
    return sorted(
        deduplicated_highlights,
        key=lambda highlight: (
            tuple(
                tuple(span)
                for span in highlight.get("docSpanOffsets", [])
            ),
            str(highlight.get("docSpanText", "")),
        ),
    )


def convert_content_selection_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = _get_spacy_nlp()
    pipeline_style_data = []

    for unique_id, value in results.items():
        curr_original_inst = deepcopy([elem for elem in alignments_dict if elem.get("unique_id") == unique_id][0])
        curr_documents = curr_original_inst.get("documents", [])
        raw_final_output = value.get("final_output")
        if (
            isinstance(raw_final_output, str)
            and raw_final_output.strip().startswith("ERROR")
        ):
            curr_original_inst.update(
                {
                    "set_of_highlights_in_context": [],
                    "skipped_reason": value.get(
                        "upstream_skipped_reason", "model_error"
                    ),
                    "upstream_error": raw_final_output,
                }
            )
            pipeline_style_data.append(curr_original_inst)
            continue

        doc_by_file = {}
        for d in curr_documents:
            if not isinstance(d, dict):
                continue
            df = d.get("documentFile") or d.get("documentUrl")
            if df:
                doc_by_file[str(df)] = d

        non_highlighted_docs = value.get("non_highlighted_docs")
        if not isinstance(non_highlighted_docs, list):
            non_highlighted_docs = []

        final_output = value.get("final_output", {})
        if not isinstance(final_output, dict):
            final_output = {}

        highlights_in_context = []

        for i, doc in enumerate(non_highlighted_docs):
            doc_key = f"Document [{str(i+1)}]"
            if doc_key not in final_output:
                continue

            if not isinstance(doc, dict):
                continue

            doc_name = doc.get("doc_name") or doc.get("documentFile") or doc.get("url") or doc.get("source_url")
            if not doc_name:
                continue
            doc_name = str(doc_name)

            matching_doc = doc_by_file.get(doc_name)
            if matching_doc is None:
                matching_doc = next(
                    (
                        dd
                        for dd in curr_documents
                        if isinstance(dd, dict)
                        and (
                            str(dd.get("documentUrl")) == doc_name
                            or str(dd.get("documentFile")) == doc_name
                        )
                    ),
                    None,
                )

            if matching_doc is None:
                continue

            doc_text = matching_doc.get("rawDocumentText") or matching_doc.get("source_raw_text")
            if not doc_text:
                continue

            doc_sents = None
            if isinstance(matching_doc.get("documentText"), list):
                doc_sents = matching_doc.get("documentText")

            spans = final_output.get(doc_key)
            if not isinstance(spans, list):
                continue

            hic = get_set_of_highlights_in_context_content_selection(
                doc_name=doc_name,
                doc_text=doc_text,
                highlights=spans,
                nlp=nlp,
                doc_sents=doc_sents,
            )
            highlights_in_context.extend(hic)

        curr_original_inst["set_of_highlights_in_context"] = highlights_in_context
        if not highlights_in_context:
            curr_original_inst.update(
                {
                    "skipped_reason": "no_highlights",
                    "upstream_error": (
                        "ERROR - content selection produced no highlights"
                    ),
                }
            )
        else:
            curr_original_inst.pop("skipped_reason", None)
            curr_original_inst.pop("upstream_error", None)
        pipeline_style_data.append(curr_original_inst)

    return pipeline_style_data


def get_set_of_highlights_in_context_clustering(curr_instance, nlp, doc_sents, *args, **kwargs):
    highlights_in_context_list = []
    highlight_global_index = 1
    for doc_i,doc_highlights in enumerate(curr_instance['highlights']):
        curr_doc_name = curr_instance['highlighted_docs'][doc_i]['doc_name']
        curr_doc_text = next(iter([elem['rawDocumentText'] for elem in doc_sents if elem['documentFile']==curr_doc_name]), None)
        curr_doc_sents = next(iter([elem['documentText'] for elem in doc_sents if elem['documentFile']==curr_doc_name]), None)

        for highlight in doc_highlights:
            curr_highlights_in_context = get_set_of_highlights_in_context_content_selection(doc_name=curr_doc_name,
                                                                                            doc_text=curr_doc_text,
                                                                                            highlights=[highlight],
                                                                                            nlp=nlp,
                                                                                            doc_sents=curr_doc_sents)

            relevant_clusters = [cluster_i for cluster_i,elem in enumerate(curr_instance['final_output']) if highlight_global_index in elem['cluster']]

            for cluster_index in relevant_clusters:
                highlights_in_context_list+=[{k: cluster_index if k == 'scuSentCharIdx' else v for k, v in d.items()} for d in curr_highlights_in_context]

            highlight_global_index+=1

    return highlights_in_context_list


def get_set_of_highlights_in_context_ALCE(curr_instance):
    highlights_in_context_list = []
    final_output = ""
    curr_scuSentCharIdx = 0
    for sentwise_results in curr_instance["final_output"]:
        if sentwise_results['cited_docs']:
            for doc_i in sentwise_results['cited_docs']:
                highlights_in_context_list.append({"documentFile" : curr_instance['non_highlighted_docs'][doc_i-1]['doc_name'],
                                                   "scuSentCharIdx" : curr_scuSentCharIdx,
                                                   "scuSentence" : sentwise_results['sent'],
                                                   "docSentCharIdx" : None,
                                                   "docSentText" : None,
                                                   "docSpanText" : None,
                                                   "docSpanOffsets" : None,
                                                   "sent_idx" : None})
        else:
                highlights_in_context_list.append({"documentFile" : None,
                                                   "scuSentCharIdx" : curr_scuSentCharIdx,
                                                   "scuSentence" : sentwise_results['sent'],
                                                   "docSentCharIdx" : None,
                                                   "docSentText" : None,
                                                   "docSpanText" : None,
                                                   "docSpanOffsets" : None,
                                                   "sent_idx" : None})

        curr_scuSentCharIdx = curr_scuSentCharIdx + len(sentwise_results['sent']) + 1
        final_output = final_output + sentwise_results['sent'] + " "

    if not all(
        final_output[element["scuSentCharIdx"] :].startswith(
            element["scuSentence"]
        )
        for element in highlights_in_context_list
    ):
        raise ValueError("scuSentence doesn't match scuSentCharIdx")
    return highlights_in_context_list, final_output.strip()


def convert_clustering_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    nlp = None
    pipeline_style_data = []
    for key,value in results.items():
        curr_pipeline_style_data = deepcopy(
            [
                elem
                for elem in alignments_dict
                if elem["unique_id"] == key
            ][0]
        )
        model_error = _InBandModelError.from_result(value)
        if model_error is not None:
            pipeline_style_data.append(
                model_error.to_pipeline_row(
                    curr_pipeline_style_data,
                    value,
                )
            )
            continue

        if nlp is None:
            nlp = _get_spacy_nlp()
        curr_documents = curr_pipeline_style_data["documents"]
        highlights_in_context = get_set_of_highlights_in_context_clustering(curr_instance=value,
                                                                            nlp=nlp,
                                                                            doc_sents=curr_documents)
        curr_pipeline_style_data.update({"set_of_highlights_in_context":highlights_in_context,
                                       "response" : value["gold_summary"]})
        pipeline_style_data.append(curr_pipeline_style_data)
    return pipeline_style_data


def convert_e2e_only_setting_to_pipeline_format(results, alignments_dict, *args, **kwargs):
        pipeline_style_data = []
        for key,value in results.items():
            original_alignments_dict = deepcopy([elem for elem in alignments_dict if elem['unique_id']==key][0])
            model_error = _InBandModelError.from_result(value)
            if model_error is not None:
                pipeline_style_data.append(
                    model_error.to_pipeline_row(
                        original_alignments_dict,
                        value,
                    )
                )
                continue
            original_alignments_dict.update({"set_of_highlights_in_context":[],
                                             "response" : value["final_output"],
                                             "gold_summary" : value["gold_summary"]})
            pipeline_style_data.append(original_alignments_dict)
        return pipeline_style_data


def convert_ALCE_to_pipeline_format(results, alignments_dict, *args, **kwargs):
        pipeline_style_data = []
        for key,value in results.items():
            original_alignments_dict = deepcopy([elem for elem in alignments_dict if elem['unique_id']==key][0])
            model_error = _InBandModelError.from_result(value)
            if model_error is not None:
                pipeline_style_data.append(
                    model_error.to_pipeline_row(
                        original_alignments_dict,
                        value,
                    )
                )
                continue
            curr_documents = [elem["documents"] for elem in alignments_dict if elem["unique_id"]==key][0]
            highlights_in_context, final_output = get_set_of_highlights_in_context_ALCE(curr_instance=value)
            original_alignments_dict.update({"set_of_highlights_in_context" : highlights_in_context,
                                             "response" : final_output,
                                             "gold_summary" : value["gold_summary"]})
            pipeline_style_data.append(original_alignments_dict)
        return pipeline_style_data


def get_set_of_highlights_in_context_FiC_CoT(curr_instance, nlp, doc_sents, *args, **kwargs):
    strict_alignment = kwargs.get("strict_alignment", False)
    canonical_registry = kwargs.get("canonical_registry")
    if canonical_registry is not None:
        return canonical_registry.project_structured_alignments(
            final_output=curr_instance["final_output"],
            alignments=curr_instance["alignments"],
        )
    clustering_style_instance_format = {
        "highlights": curr_instance["highlights"],
        "highlighted_docs": curr_instance["highlighted_docs"],
        "final_output": [
            {"cluster": elem["highlights"]}
            for elem in curr_instance["alignments"]
        ],
    }
    clustered_set_of_highlights = (
        get_set_of_highlights_in_context_clustering(
            curr_instance=clustering_style_instance_format,
            nlp=nlp,
            doc_sents=doc_sents,
        )
    )
    highlights_in_context_list = []
    alignments_sents_embedding = None
    for sent in nlp(curr_instance['final_output']).sents:
        relevant_sent_id = [elem['sent_id'] for elem in curr_instance['alignments'] if rmv_spaces_and_punct(sent.text) and rmv_spaces_and_punct(sent.text)==rmv_spaces_and_punct(elem['sent_text'])]
        if not relevant_sent_id:
            relevant_sent_id = [elem['sent_id'] for elem in curr_instance['alignments'] if (rmv_spaces_and_punct(sent.text) and rmv_spaces_and_punct(sent.text) in rmv_spaces_and_punct(elem['sent_text'])) or (rmv_spaces_and_punct(elem['sent_text']) and rmv_spaces_and_punct(elem['sent_text']) in rmv_spaces_and_punct(sent.text))]

        if not relevant_sent_id:
            if strict_alignment:
                raise ValueError(
                    "structured FiC sentence has no exact or substring "
                    f"alignment: {sent.text!r}"
                )
            sent_transformer_model, sent_transformer_tokenizer = _get_sentence_transformer()
            curr_sent_embedding = get_sentence_embedding(
                sent.text,
                sent_transformer_model,
                sent_transformer_tokenizer,
            )
            if not alignments_sents_embedding:
                alignments_sents_embedding = [get_sentence_embedding(elem["sent_text"], sent_transformer_model, sent_transformer_tokenizer) for elem in curr_instance['alignments']]
            _, cosine_distance = _get_semantic_math_resources()
            similarities = [1 - cosine_distance(curr_sent_embedding , s_embedding) for s_embedding in alignments_sents_embedding]
            relevant_sent_id = [curr_instance['alignments'][i]['sent_id'] for i,scr in enumerate(similarities) if scr>=COSINE_SIMILARITY_THR]

        if not relevant_sent_id:
            highlights_in_context_list.append({"documentFile" : None,
                                               "scuSentCharIdx" : sent.start_char,
                                               "scuSentence" : sent.text,
                                               "docSentCharIdx" : None,
                                               "docSentText" : None,
                                               "docSpanText" : None,
                                               "docSpanOffsets" : None,
                                               "sent_idx" : None})
        else:
            relevant_clustered_highlights_in_context = deepcopy(
                [
                    elem
                    for elem in clustered_set_of_highlights
                    if elem["scuSentCharIdx"] + 1 in relevant_sent_id
                ]
            )
            relevant_clustered_highlights_in_context = [
                {
                    key: sent.text if key == "scuSentence" else item
                    for key, item in elem.items()
                }
                for elem in relevant_clustered_highlights_in_context
            ]
            relevant_clustered_highlights_in_context = [
                {
                    key: (
                        sent.start_char
                        if key == "scuSentCharIdx"
                        else item
                    )
                    for key, item in elem.items()
                }
                for elem in relevant_clustered_highlights_in_context
            ]
            highlights_in_context_list.extend(
                relevant_clustered_highlights_in_context
            )
    return highlights_in_context_list


def convert_FiC_CoT_results_to_pipeline_format(results, alignments_dict, *args, **kwargs):
    strict_alignment = bool(
        kwargs.get("strict_alignment")
        or kwargs.get("structured_output")
    )
    nlp = None if strict_alignment else _get_spacy_nlp()
    pipeline_style_data = []
    n_skipped_error = 0
    n_skipped_no_align = 0
    for key,value in results.items():
        curr_pipeline_style_data = deepcopy([elem for elem in alignments_dict if elem["unique_id"]==key][0])

        final_output = value.get("final_output", "")
        alignments = value.get("alignments", []) or []
        # IMPROVEMENTS 1.1: never feed a model-error string (e.g. "ERROR - 429 Resource
        # exhausted...") through the spaCy sentence splitter — doing so fabricates one
        # fake `documentFile=None` highlight per error sentence and corrupts the metric
        # (the reported F1 becomes metric-on-error-strings). Skip such instances
        # explicitly with an empty highlight set + a skipped_reason marker so they can be
        # excluded or reported separately by the scorer.
        is_error = isinstance(final_output, str) and final_output.strip().startswith("ERROR")
        if is_error or not alignments:
            curr_pipeline_style_data.update({
                "set_of_highlights_in_context": [],
                "response": final_output if isinstance(final_output, str) else "",
                "skipped_reason": "model_error" if is_error else "no_alignments",
            })
            pipeline_style_data.append(curr_pipeline_style_data)
            if is_error:
                n_skipped_error += 1
            else:
                n_skipped_no_align += 1
            continue

        curr_documents = curr_pipeline_style_data["documents"]
        canonical_registry = None
        if strict_alignment:
            canonical_registry = FiCCanonicalHighlightRegistry.build(
                marked_documents=value.get("highlighted_docs", []),
                source_documents=curr_documents,
                upstream_highlights=curr_pipeline_style_data.get(
                    "set_of_highlights_in_context",
                    [],
                ),
                allow_controlled_prefix=False,
            )
            canonical_registry.assert_declared_highlights(
                value.get("highlights")
            )
            canonical_registry.assert_alignment_coverage(alignments)
        highlights_in_context = get_set_of_highlights_in_context_FiC_CoT(curr_instance=value,
                                                                         nlp=nlp,
                                                                         doc_sents=curr_documents,
                                                                         strict_alignment=strict_alignment,
                                                                         canonical_registry=canonical_registry)
        curr_pipeline_style_data.update({"set_of_highlights_in_context":highlights_in_context,
                                       "response" : value["final_output"]})
        pipeline_style_data.append(curr_pipeline_style_data)
    if n_skipped_error or n_skipped_no_align:
        logging.warning(
            f"convert_FiC_CoT: emitted empty highlights for "
            f"{n_skipped_error} model_error + {n_skipped_no_align} no_alignments instance(s) "
            f"(did NOT fabricate highlights from text)."
        )
    return pipeline_style_data
