"""Canonical FiC highlight IDs backed by already-resolved upstream offsets."""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from ..prompting.highlights import rmv_spaces_and_punct
from .structured_highlights import HighlightMarkupParser


_SPAN_SEPARATORS = ("<HIGHLIGHT_SEP>", "<SENT_SEP>")


def _normalized_span_text(value: str) -> str:
    """Normalize only formatting ignored by the existing HIC contract."""

    normalized = value
    for separator in _SPAN_SEPARATORS:
        normalized = normalized.replace(separator, "")
    return "".join(normalized.split()).casefold()


@dataclass(frozen=True, slots=True)
class FiCSourceProjection:
    """One immutable upstream source interval used by a FiC marker."""

    document_file: str
    start: int
    end: int
    span_text: str
    document_sentence_start: object
    document_sentence_text: object
    sentence_index: object

    def as_pipeline_highlight(
        self,
        *,
        scu_sentence_start: int,
        scu_sentence: str,
    ) -> dict:
        """Project the canonical source offset into one generated sentence."""

        return {
            "documentFile": self.document_file,
            "scuSentCharIdx": scu_sentence_start,
            "scuSentence": scu_sentence,
            "docSentCharIdx": self.document_sentence_start,
            "docSentText": self.document_sentence_text,
            "docSpanText": self.span_text,
            "docSpanOffsets": [[self.start, self.end]],
            "sent_idx": self.sentence_index,
        }


@dataclass(frozen=True, slots=True)
class FiCCanonicalHighlight:
    """One numbered marker and its exact upstream source projections."""

    highlight_id: int
    document_file: str
    start: int
    end: int
    span_text: str
    projections: tuple[FiCSourceProjection, ...]


class FiCCanonicalHighlightRegistry:
    """Bind FiC highlight IDs to marked source occurrences, never text search."""

    def __init__(
        self,
        highlights: Sequence[FiCCanonicalHighlight],
        marker_text_groups: Sequence[Sequence[str]],
    ):
        self._highlights = tuple(highlights)
        self._marker_text_groups = tuple(
            tuple(group) for group in marker_text_groups
        )
        expected_ids = tuple(range(1, len(self._highlights) + 1))
        actual_ids = tuple(
            highlight.highlight_id for highlight in self._highlights
        )
        if actual_ids != expected_ids:
            raise ValueError(
                "FiC canonical highlight IDs must follow marker order"
            )
        self._by_id = {
            highlight.highlight_id: highlight
            for highlight in self._highlights
        }

    @property
    def highlights(self) -> tuple[FiCCanonicalHighlight, ...]:
        return self._highlights

    @staticmethod
    def _source_documents(
        source_documents: Iterable[Mapping],
    ) -> dict[str, str]:
        documents: dict[str, str] = {}
        for source_document in source_documents:
            if not isinstance(source_document, Mapping):
                raise TypeError("FiC source document must be an object")
            document_file = (
                source_document.get("documentFile")
                or source_document.get("documentUrl")
            )
            raw_text = (
                source_document.get("rawDocumentText")
                or source_document.get("source_raw_text")
            )
            if not isinstance(document_file, str) or not document_file:
                raise ValueError("FiC source document has no documentFile")
            if not isinstance(raw_text, str):
                raise ValueError(
                    f"FiC source document {document_file!r} has no raw text"
                )
            if document_file in documents:
                raise ValueError(
                    f"duplicate FiC source document {document_file!r}"
                )
            documents[document_file] = raw_text
        return documents

    @staticmethod
    def _validated_offsets(
        document_file: str,
        raw_text: str,
        upstream_highlight: Mapping,
    ) -> tuple[tuple[int, int], ...]:
        offsets = upstream_highlight.get("docSpanOffsets")
        if not isinstance(offsets, list) or not offsets:
            raise ValueError(
                f"upstream HIC for {document_file!r} has no docSpanOffsets"
            )
        validated = []
        for offset in offsets:
            if (
                not isinstance(offset, (list, tuple))
                or len(offset) != 2
                or type(offset[0]) is not int
                or type(offset[1]) is not int
            ):
                raise ValueError(
                    f"invalid upstream docSpanOffsets for {document_file!r}"
                )
            start, end = offset
            if start < 0 or start >= end or end > len(raw_text):
                raise ValueError(
                    f"upstream docSpanOffsets are outside {document_file!r}"
                )
            validated.append((start, end))
        if validated != sorted(validated):
            raise ValueError(
                f"upstream docSpanOffsets are not ordered for "
                f"{document_file!r}"
            )
        if any(
            current_start < previous_end
            for (_, previous_end), (current_start, _) in zip(
                validated,
                validated[1:],
            )
        ):
            raise ValueError(
                f"upstream docSpanOffsets overlap for {document_file!r}"
            )
        return tuple(validated)

    @classmethod
    def _upstream_projections(
        cls,
        source_by_file: Mapping[str, str],
        upstream_highlights: Iterable[Mapping],
    ) -> dict[str, tuple[FiCSourceProjection, ...]]:
        projections_by_document: dict[
            str, dict[tuple[int, int], FiCSourceProjection]
        ] = {}
        for upstream_highlight in upstream_highlights:
            if not isinstance(upstream_highlight, Mapping):
                raise TypeError("upstream HIC must be an object")
            document_file = upstream_highlight.get("documentFile")
            if not isinstance(document_file, str) or not document_file:
                raise ValueError("upstream HIC has no documentFile")
            if document_file not in source_by_file:
                raise ValueError(
                    "upstream HIC references an unknown source document "
                    f"{document_file!r}"
                )
            raw_text = source_by_file[document_file]
            offsets = cls._validated_offsets(
                document_file,
                raw_text,
                upstream_highlight,
            )
            claimed_text = upstream_highlight.get("docSpanText")
            if not isinstance(claimed_text, str):
                raise ValueError(
                    f"upstream HIC for {document_file!r} has no docSpanText"
                )
            resolved_text = "".join(
                raw_text[start:end] for start, end in offsets
            )
            if _normalized_span_text(
                claimed_text
            ) != _normalized_span_text(resolved_text):
                raise ValueError(
                    "upstream HIC docSpanText does not match its canonical "
                    f"offsets in {document_file!r}"
                )

            document_projections = projections_by_document.setdefault(
                document_file,
                {},
            )
            for start, end in offsets:
                projection = FiCSourceProjection(
                    document_file=document_file,
                    start=start,
                    end=end,
                    span_text=raw_text[start:end],
                    document_sentence_start=upstream_highlight.get(
                        "docSentCharIdx"
                    ),
                    document_sentence_text=upstream_highlight.get(
                        "docSentText"
                    ),
                    sentence_index=upstream_highlight.get("sent_idx"),
                )
                key = (start, end)
                previous = document_projections.get(key)
                if previous is not None and previous != projection:
                    raise ValueError(
                        "conflicting upstream HIC metadata for canonical "
                        f"offset {document_file!r}:{start}-{end}"
                    )
                if previous is None and any(
                    start < existing_end and existing_start < end
                    for existing_start, existing_end in document_projections
                ):
                    raise ValueError(
                        "overlapping upstream HIC offsets for "
                        f"{document_file!r}:{start}-{end}"
                    )
                document_projections[key] = projection

        return {
            document_file: tuple(
                projections[offset]
                for offset in sorted(projections)
            )
            for document_file, projections in projections_by_document.items()
        }

    @staticmethod
    def _marker_projections(
        document_file: str,
        raw_text: str,
        marker_start: int,
        marker_end: int,
        upstream: Sequence[FiCSourceProjection],
    ) -> tuple[FiCSourceProjection, ...]:
        candidates = tuple(
            projection
            for projection in upstream
            if marker_start <= projection.start
            and projection.end <= marker_end
        )
        if (
            not candidates
            or candidates[0].start != marker_start
            or candidates[-1].end != marker_end
        ):
            raise ValueError(
                "FiC marker/upstream mismatch for "
                f"{document_file!r}:{marker_start}-{marker_end}"
            )
        cursor = marker_start
        for projection in candidates:
            if projection.start > cursor:
                # ``merge_spans`` includes each half-open end while merging.
                # Therefore it can absorb exactly one raw separator between
                # two source fragments, but never an arbitrary-length gap.
                absorbed_separator = raw_text[cursor : projection.start]
                if len(absorbed_separator) != 1:
                    raise ValueError(
                        "FiC marker/upstream mismatch for "
                        f"{document_file!r}:{marker_start}-{marker_end}"
                    )
            cursor = max(cursor, projection.end)
        if cursor != marker_end:
            raise ValueError(
                "FiC marker/upstream mismatch for "
                f"{document_file!r}:{marker_start}-{marker_end}"
            )
        evidence = tuple(
            projection
            for projection in candidates
            if rmv_spaces_and_punct(projection.span_text)
        )
        if not evidence:
            raise ValueError(
                "FiC marker contains no attributable evidence for "
                f"{document_file!r}:{marker_start}-{marker_end}"
            )
        return evidence

    @classmethod
    def build(
        cls,
        *,
        marked_documents: Iterable[Mapping],
        source_documents: Iterable[Mapping],
        upstream_highlights: Iterable[Mapping],
        allow_controlled_prefix: bool = False,
    ) -> "FiCCanonicalHighlightRegistry":
        """Build marker-ordered IDs after exact source/upstream validation."""

        source_by_file = cls._source_documents(source_documents)
        upstream_by_document = cls._upstream_projections(
            source_by_file,
            upstream_highlights,
        )
        parser = HighlightMarkupParser()
        canonical_highlights: list[FiCCanonicalHighlight] = []
        marker_text_groups: list[tuple[str, ...]] = []
        covered_offsets: set[tuple[str, int, int]] = set()
        seen_documents: set[str] = set()

        for marked_document in marked_documents:
            if not isinstance(marked_document, Mapping):
                raise TypeError("FiC marked document must be an object")
            document_file = (
                marked_document.get("doc_name")
                or marked_document.get("documentFile")
            )
            marked_text = marked_document.get("doc_text")
            if not isinstance(document_file, str) or not document_file:
                raise ValueError("FiC marked document has no doc_name")
            if document_file in seen_documents:
                raise ValueError(
                    f"duplicate FiC marked document {document_file!r}"
                )
            seen_documents.add(document_file)
            if document_file not in source_by_file:
                raise ValueError(
                    f"FiC marked document {document_file!r} is not a source"
                )
            if not isinstance(marked_text, str):
                raise ValueError(
                    f"FiC marked document {document_file!r} has no text"
                )

            parsed = parser.parse(marked_text)
            raw_text = source_by_file[document_file]
            exact_source = parsed.text == raw_text
            controlled_prefix = (
                allow_controlled_prefix
                and raw_text.startswith(parsed.text)
                and bool(parsed.highlighted_intervals)
                and parsed.highlighted_intervals[-1].end == len(parsed.text)
            )
            if not exact_source and not controlled_prefix:
                raise ValueError(
                    "FiC marker removal does not restore the raw source or "
                    f"a controlled highlighted prefix for {document_file!r}"
                )

            upstream = upstream_by_document.get(document_file, ())
            marker_text_groups.append(
                tuple(
                    raw_text[interval.start : interval.end]
                    for interval in parsed.highlighted_intervals
                )
            )
            for interval in parsed.highlighted_intervals:
                projections = cls._marker_projections(
                    document_file,
                    raw_text,
                    interval.start,
                    interval.end,
                    upstream,
                )
                for projection in projections:
                    covered_offsets.add(
                        (
                            projection.document_file,
                            projection.start,
                            projection.end,
                        )
                    )
                canonical_highlights.append(
                    FiCCanonicalHighlight(
                        highlight_id=len(canonical_highlights) + 1,
                        document_file=document_file,
                        start=interval.start,
                        end=interval.end,
                        span_text=raw_text[interval.start : interval.end],
                        projections=projections,
                    )
                )

            for projection in upstream:
                if not rmv_spaces_and_punct(projection.span_text):
                    continue
                if (
                    projection.end <= len(parsed.text)
                    and (
                        projection.document_file,
                        projection.start,
                        projection.end,
                    )
                    not in covered_offsets
                ):
                    raise ValueError(
                        "FiC marker/upstream mismatch: rendered source omits "
                        "canonical offset "
                        f"{document_file!r}:{projection.start}-"
                        f"{projection.end}"
                    )

        for document_file, projections in upstream_by_document.items():
            for projection in projections:
                if not rmv_spaces_and_punct(projection.span_text):
                    continue
                if (
                    projection.document_file,
                    projection.start,
                    projection.end,
                ) not in covered_offsets:
                    raise ValueError(
                        "FiC marker/upstream mismatch: no source marker "
                        "covers canonical offset "
                        f"{document_file!r}:{projection.start}-"
                        f"{projection.end}"
                    )

        if not canonical_highlights:
            raise ValueError("FiC canonical highlight registry is empty")
        return cls(canonical_highlights, marker_text_groups)

    def assert_declared_highlights(self, declared_highlights) -> None:
        """Cross-check legacy prompt metadata without using it to locate text."""

        if not isinstance(declared_highlights, list):
            raise ValueError("FiC result has no declared highlights")
        actual_groups = []
        for document_highlights in declared_highlights:
            if not isinstance(document_highlights, list):
                raise ValueError(
                    "FiC declared highlights must be grouped by document"
                )
            if any(
                not isinstance(span_text, str)
                for span_text in document_highlights
            ):
                raise ValueError("FiC declared highlight must be text")
            actual_groups.append(tuple(document_highlights))
        if tuple(actual_groups) != self._marker_text_groups:
            raise ValueError(
                "FiC declared highlight order does not match source markers"
            )

    def assert_alignment_coverage(self, alignments) -> None:
        """Require each marker ID exactly once in structured FiC output."""

        if not isinstance(alignments, list):
            raise ValueError("FiC alignments must be a list")
        used_ids = []
        for alignment in alignments:
            if not isinstance(alignment, Mapping):
                raise ValueError("FiC alignment must be an object")
            highlight_ids = alignment.get("highlights")
            if not isinstance(highlight_ids, list):
                raise ValueError("FiC alignment has no highlight IDs")
            for highlight_id in highlight_ids:
                if type(highlight_id) is not int:
                    raise ValueError("FiC highlight ID must be an integer")
                used_ids.append(highlight_id)
        expected_ids = list(range(1, len(self._highlights) + 1))
        if sorted(used_ids) != expected_ids:
            raise ValueError(
                "structured FiC alignments must use every canonical "
                "highlight ID exactly once"
            )

    def project(
        self,
        highlight_ids: Iterable[int],
        *,
        scu_sentence_start: int,
        scu_sentence: str,
    ) -> list[dict]:
        """Project numbered marker offsets into one generated SCU."""

        projected = []
        seen_ids: set[int] = set()
        for highlight_id in highlight_ids:
            if type(highlight_id) is not int:
                raise ValueError("FiC highlight ID must be an integer")
            canonical_id = highlight_id
            if canonical_id in seen_ids:
                continue
            seen_ids.add(canonical_id)
            highlight = self._by_id.get(canonical_id)
            if highlight is None:
                raise ValueError(
                    f"FiC alignment references unknown highlight "
                    f"{canonical_id}"
                )
            projected.extend(
                projection.as_pipeline_highlight(
                    scu_sentence_start=scu_sentence_start,
                    scu_sentence=scu_sentence,
                )
                for projection in highlight.projections
            )
        return projected

    def project_structured_alignments(
        self,
        *,
        final_output: str,
        alignments: Sequence[Mapping],
    ) -> list[dict]:
        """Project ordered structured sentences without text-based matching."""

        if not isinstance(final_output, str):
            raise ValueError("structured FiC final_output must be text")
        projected = []
        cursor = 0
        for sentence_index, alignment in enumerate(alignments, start=1):
            if not isinstance(alignment, Mapping):
                raise ValueError("FiC alignment must be an object")
            if alignment.get("sent_id") != sentence_index:
                raise ValueError(
                    "structured FiC sentence IDs must be consecutive"
                )
            sentence_text = alignment.get("sent_text")
            if not isinstance(sentence_text, str) or not sentence_text:
                raise ValueError(
                    "structured FiC alignment has no sentence text"
                )

            sentence_start = cursor
            while (
                sentence_start < len(final_output)
                and final_output[sentence_start].isspace()
            ):
                sentence_start += 1
            if not final_output.startswith(sentence_text, sentence_start):
                raise ValueError(
                    "structured FiC final_output does not match its ordered "
                    f"sentence {sentence_index}"
                )
            cursor = sentence_start + len(sentence_text)
            projected.extend(
                self.project(
                    alignment["highlights"],
                    scu_sentence_start=sentence_start,
                    scu_sentence=sentence_text,
                )
            )

        if final_output[cursor:].strip():
            raise ValueError(
                "structured FiC final_output has text outside alignments"
            )
        return projected


__all__ = [
    "FiCCanonicalHighlight",
    "FiCCanonicalHighlightRegistry",
    "FiCSourceProjection",
]
