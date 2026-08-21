"""Strict parsers, attribution, and usage contracts for sequential dialogue."""

from copy import deepcopy
import json
from typing import Mapping


HIGHLIGHT_START = "<highlight_start>"
HIGHLIGHT_END = "<highlight_end>"
SEQUENTIAL_PROTOCOL = {
    "runner": "dialogue_sequential",
    "session_scope": "one_chat_per_instance",
    "turns": [
        "content_selection",
        "clustering",
        "sentence_fusion_per_cluster",
    ],
    "structured_output": {
        "content_selection": True,
        "clustering": True,
        "sentence_fusion": True,
    },
    "fusion_history_policy": "rollback_after_each_cluster",
}
_USAGE_FIELDS = ("prompt", "completion", "cached", "calls")


def parse_content_selection_spans(output: str) -> list[tuple[str, str]]:
    """Parse only the declared content-selection JSON contract."""

    try:
        structured = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if (
        not isinstance(structured, dict)
        or set(structured) != {"highlights"}
        or not isinstance(structured["highlights"], list)
    ):
        return []

    spans: list[tuple[str, str]] = []
    for highlight in structured["highlights"]:
        if (
            not isinstance(highlight, dict)
            or set(highlight) != {"doc_id", "span_text"}
        ):
            return []
        document_id = highlight["doc_id"]
        span_text = highlight["span_text"]
        if (
            type(document_id) is not str
            or not document_id
            or not document_id.isascii()
            or not document_id.isdecimal()
            or str(int(document_id)) != document_id
            or int(document_id) < 1
            or type(span_text) is not str
            or not span_text.strip()
        ):
            return []
        spans.append((document_id, span_text.strip()))
    return spans


def parse_clusters(output: str, span_count: int) -> list[list[int]]:
    """Parse a clustering response and require one exact span partition."""

    try:
        structured = json.loads(output)
        if (
            not isinstance(structured, dict)
            or set(structured) != {"clusters"}
            or not isinstance(structured["clusters"], list)
        ):
            return []
        clusters = structured["clusters"]
        if any(
            not isinstance(cluster, list) or not cluster
            for cluster in clusters
        ):
            return []
        if any(
            type(index) is not int
            for cluster in clusters
            for index in cluster
        ):
            return []
        flattened = [index for cluster in clusters for index in cluster]
        if sorted(flattened) != list(range(1, span_count + 1)):
            return []
        return clusters
    except Exception:
        return []


def parse_fusion_output(
    output: str,
    expected_highlight_ids: list[int],
) -> dict[str, object] | None:
    """Parse one fusion response without coercing text or highlight IDs."""

    try:
        structured = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    if (
        not isinstance(structured, dict)
        or set(structured) != {"sentence_text", "highlight_ids"}
    ):
        return None
    sentence = structured["sentence_text"]
    returned_ids = structured["highlight_ids"]
    if (
        type(sentence) is not str
        or not sentence.strip()
        or not isinstance(returned_ids, list)
        or any(type(index) is not int for index in returned_ids)
        or len(returned_ids) != len(set(returned_ids))
        or returned_ids != expected_highlight_ids
    ):
        return None
    return {
        "sentence_text": sentence.strip(),
        "highlight_ids": list(returned_ids),
    }


def concrete_offsets(
    offsets: object,
    raw_document: str,
    expected_span: str | None = None,
) -> bool:
    """Require bounded offsets and, when supplied, exact source text."""

    structurally_valid = (
        isinstance(offsets, list)
        and bool(offsets)
        and all(
            isinstance(offset, (list, tuple))
            and len(offset) == 2
            and type(offset[0]) is int
            and type(offset[1]) is int
            and 0 <= offset[0] < offset[1] <= len(raw_document)
            for offset in offsets
        )
    )
    if not structurally_valid:
        return False
    if expected_span is None:
        return True
    return (
        len(offsets) == 1
        and raw_document[offsets[0][0] : offsets[0][1]]
        == expected_span
    )


def source_span_metadata(
    instance: Mapping[str, object],
    document_file: object,
    span_text: str,
) -> dict[str, object]:
    """Recover evaluator-required source coordinates for one selected span."""

    documents = instance.get("documents", [])
    document = next(
        (
            candidate
            for candidate in documents
            if str(candidate.get("documentFile")) == str(document_file)
        ),
        None,
    )
    if not isinstance(document, Mapping):
        raise ValueError(
            f"source document {document_file!r} is unavailable"
        )

    raw_document = str(document.get("rawDocumentText", ""))
    trusted_matches = [
        source_highlight
        for source_highlight in instance.get(
            "set_of_highlights_in_context",
            [],
        )
        if (
            isinstance(source_highlight, Mapping)
            and str(source_highlight.get("documentFile"))
            == str(document_file)
            and source_highlight.get("docSpanText") == span_text
            and concrete_offsets(
                source_highlight.get("docSpanOffsets"),
                raw_document,
                span_text,
            )
        )
    ]
    if len(trusted_matches) == 1:
        return deepcopy(dict(trusted_matches[0]))
    if len(trusted_matches) > 1:
        raise ValueError(
            f"multiple trusted occurrences of {span_text!r} exist in "
            f"source document {document_file!r}"
        )

    occurrence_starts = []
    search_from = 0
    while True:
        current_start = raw_document.find(span_text, search_from)
        if current_start < 0:
            break
        occurrence_starts.append(current_start)
        search_from = current_start + max(1, len(span_text))
    if not occurrence_starts:
        raise ValueError(
            f"exact span is absent from source document {document_file!r}"
        )
    if len(occurrence_starts) > 1:
        raise ValueError(
            f"ambiguous span {span_text!r} has multiple occurrences in "
            f"source document {document_file!r}"
        )
    start = occurrence_starts[0]

    metadata: dict[str, object] = {
        "documentFile": str(document_file),
        "docSpanText": span_text,
        "docSpanOffsets": [[start, start + len(span_text)]],
    }
    sentence_starts = document.get("docSentCharIdxToSentIdx") or []
    sentence_texts = document.get("documentText") or []
    containing_sentence = None
    for index, sentence_start in enumerate(sentence_starts):
        try:
            if int(sentence_start) <= start:
                containing_sentence = index
            else:
                break
        except (TypeError, ValueError):
            continue
    if (
        containing_sentence is not None
        and containing_sentence < len(sentence_texts)
    ):
        metadata.update(
            {
                "docSentCharIdx": sentence_starts[containing_sentence],
                "docSentText": sentence_texts[containing_sentence],
                "sent_idx": containing_sentence,
            }
        )
    return metadata


def usage_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    """Subtract two aggregate usage snapshots using stable artifact keys."""

    return {
        key: after.get(key, 0) - before.get(key, 0)
        for key in _USAGE_FIELDS
    }


def usage_from_attempt_trace(
    trace: Mapping[str, object],
) -> dict[str, int]:
    """Sum exact per-call provider usage without reading shared totals."""

    usage_total = {key: 0 for key in _USAGE_FIELDS}
    for stage in ("content_selection", "clustering", "fusion"):
        attempts = trace.get(stage, [])
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            usage = attempt.get("usage")
            if not isinstance(usage, Mapping):
                continue
            usage_total["calls"] += 1
            for target, source in (
                ("prompt", "prompt_token_count"),
                ("completion", "candidates_token_count"),
                ("cached", "cached_content_token_count"),
            ):
                value = usage.get(source)
                if type(value) is int and value >= 0:
                    usage_total[target] += value
    return usage_total


__all__ = [
    "HIGHLIGHT_END",
    "HIGHLIGHT_START",
    "SEQUENTIAL_PROTOCOL",
    "concrete_offsets",
    "parse_clusters",
    "parse_content_selection_spans",
    "parse_fusion_output",
    "source_span_metadata",
    "usage_delta",
    "usage_from_attempt_trace",
]
