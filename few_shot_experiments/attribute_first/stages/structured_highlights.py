"""Strict value objects and parser for schema-bound highlight responses."""

import json
from dataclasses import dataclass
import re


class MalformedStructuredHighlightResponse(ValueError):
    """The provider response is not one complete highlight JSON envelope."""


class AmbiguousSourceSpan(ValueError):
    """A verbatim span identifies more than one source occurrence."""


class SourceSpanNotFound(ValueError):
    """A claimed verbatim span cannot be traced to its declared source."""


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """One uniquely attributable half-open source interval."""

    start: int
    end: int
    match_mode: str


@dataclass(frozen=True, slots=True)
class MarkedSourceDocument:
    """Marker-free source text plus exact highlighted source intervals."""

    text: str
    highlighted_intervals: tuple[SourceSpan, ...]

    def overlaps_highlight(self, span: SourceSpan) -> bool:
        """Return whether ``span`` touches any already-highlighted character."""

        return any(
            span.start < highlighted.end
            and highlighted.start < span.end
            for highlighted in self.highlighted_intervals
        )


class HighlightMarkupParser:
    """Remove supported highlight markers while preserving source offsets."""

    _START_TO_END = {
        "<highlight_start>": "<highlight_end>",
        "{HS}": "{HE}",
    }
    _MARKER_PATTERN = re.compile(
        "|".join(
            re.escape(marker)
            for marker in (
                "<highlight_start>",
                "<highlight_end>",
                "{HS}",
                "{HE}",
            )
        )
    )

    def parse(self, source_text: str) -> MarkedSourceDocument:
        """Parse non-nested marker pairs into a marker-free document."""

        if not isinstance(source_text, str):
            raise TypeError("source_text must be a string")

        plain_parts: list[str] = []
        highlighted_intervals: list[SourceSpan] = []
        plain_length = 0
        active_start: int | None = None
        expected_end: str | None = None
        cursor = 0

        for marker_match in self._MARKER_PATTERN.finditer(source_text):
            segment = source_text[cursor : marker_match.start()]
            plain_parts.append(segment)
            plain_length += len(segment)
            marker = marker_match.group(0)

            if marker in self._START_TO_END:
                if active_start is not None:
                    raise ValueError("nested highlight markers are invalid")
                active_start = plain_length
                expected_end = self._START_TO_END[marker]
            else:
                if active_start is None or marker != expected_end:
                    raise ValueError("unbalanced highlight markers are invalid")
                highlighted_intervals.append(
                    SourceSpan(
                        start=active_start,
                        end=plain_length,
                        match_mode="highlight",
                    )
                )
                active_start = None
                expected_end = None
            cursor = marker_match.end()

        if active_start is not None:
            raise ValueError("unbalanced highlight markers are invalid")

        trailing = source_text[cursor:]
        plain_parts.append(trailing)
        return MarkedSourceDocument(
            text="".join(plain_parts),
            highlighted_intervals=tuple(highlighted_intervals),
        )


class UniqueSourceSpanLocator:
    """Locate one source span without choosing an occurrence.

    Provider JSON may collapse source line breaks into spaces. Treat only
    whitespace runs as equivalent; case, punctuation, and every non-whitespace
    character must remain exact.
    """

    @staticmethod
    def _all_offsets(haystack: str, needle: str) -> list[tuple[int, int]]:
        offsets = []
        search_from = 0
        while needle:
            start = haystack.find(needle, search_from)
            if start < 0:
                break
            offsets.append((start, start + len(needle)))
            search_from = start + 1
        return offsets

    @staticmethod
    def _all_whitespace_equivalent_offsets(
        haystack: str,
        needle: str,
    ) -> list[tuple[int, int]]:
        """Map a span through whitespace runs while preserving source offsets."""

        parts = re.split(r"\s+", needle)
        if not parts or any(not part for part in parts):
            return []
        pattern = r"\s+".join(re.escape(part) for part in parts)
        overlapping = re.compile(f"(?=({pattern}))")
        return [
            (match.start(1), match.end(1))
            for match in overlapping.finditer(haystack)
        ]

    def locate(
        self,
        source_name: str,
        document_text: str,
        span_text: str,
    ) -> SourceSpan:
        """Return the only exact or whitespace-equivalent source interval."""

        if not isinstance(document_text, str):
            raise TypeError("document_text must be a string")
        if not isinstance(span_text, str):
            raise TypeError("span_text must be a string")
        needle = span_text.strip()
        offsets = self._all_whitespace_equivalent_offsets(
            document_text,
            needle,
        )
        if len(offsets) == 1:
            start, end = offsets[0]
            match_mode = (
                "exact"
                if document_text[start:end] == needle
                else "whitespace-normalized"
            )
            return SourceSpan(start=start, end=end, match_mode=match_mode)
        if len(offsets) > 1:
            raise AmbiguousSourceSpan(
                f"ambiguous source span in {source_name!r}: "
                f"{span_text[:80]!r} has multiple occurrences"
            )
        raise SourceSpanNotFound(
            "source span is not traceable to its declared document "
            f"{source_name!r}: {span_text[:80]!r}"
        )


class UniqueSourceSpanExtender:
    """Extend one known source occurrence to a deterministic unique span."""

    @staticmethod
    def _previous_word_start(text: str, start: int) -> int | None:
        cursor = start
        while cursor > 0 and text[cursor - 1].isspace():
            cursor -= 1
        if cursor == 0:
            return None
        while cursor > 0 and not text[cursor - 1].isspace():
            cursor -= 1
        return cursor

    @staticmethod
    def _next_word_end(text: str, end: int) -> int | None:
        cursor = end
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor == len(text):
            return None
        while cursor < len(text) and not text[cursor].isspace():
            cursor += 1
        return cursor

    @staticmethod
    def _overlaps_any(
        candidate: SourceSpan,
        forbidden: tuple[SourceSpan, ...],
    ) -> bool:
        return any(
            candidate.start < interval.end
            and interval.start < candidate.end
            for interval in forbidden
        )

    def extend(
        self,
        source_name: str,
        document_text: str,
        target: SourceSpan,
        *,
        forbidden_intervals: tuple[SourceSpan, ...] = (),
    ) -> SourceSpan:
        """Return the shortest word-aligned unique span containing ``target``."""

        if (
            target.start < 0
            or target.start >= target.end
            or target.end > len(document_text)
        ):
            raise ValueError("target source interval is outside the document")

        pending = [(target.start, target.end)]
        visited: set[tuple[int, int]] = set()
        while pending:
            candidates = []
            for start, end in pending:
                if (start, end) in visited:
                    continue
                visited.add((start, end))
                span = SourceSpan(start, end, "exact")
                if not self._overlaps_any(span, forbidden_intervals):
                    text = document_text[start:end]
                    offsets = UniqueSourceSpanLocator._all_offsets(
                        document_text,
                        text,
                    )
                    if offsets == [(start, end)]:
                        candidates.append(span)
            if candidates:
                return min(
                    candidates,
                    key=lambda span: (
                        (span.end - span.start)
                        - (target.end - target.start),
                        span.start,
                        span.end,
                    ),
                )

            expanded: set[tuple[int, int]] = set()
            for start, end in pending:
                previous_start = self._previous_word_start(
                    document_text,
                    start,
                )
                if previous_start is not None:
                    expanded.add((previous_start, end))
                next_end = self._next_word_end(document_text, end)
                if next_end is not None:
                    expanded.add((start, next_end))
            pending = sorted(
                expanded - visited,
                key=lambda bounds: (
                    (bounds[1] - bounds[0])
                    - (target.end - target.start),
                    bounds[0],
                    bounds[1],
                ),
            )

        raise AmbiguousSourceSpan(
            f"could not extend the intended span in {source_name!r} "
            "to one unique non-overlapping occurrence"
        )


@dataclass(frozen=True, slots=True)
class StructuredHighlight:
    """One immutable highlight selected by a schema-bound stage."""

    doc_id: str
    span_text: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.doc_id, str)
            or not self.doc_id
            or self.doc_id.strip() != self.doc_id
        ):
            raise MalformedStructuredHighlightResponse(
                "structured highlight doc_id must be a non-empty string"
            )
        if not isinstance(self.span_text, str):
            raise MalformedStructuredHighlightResponse(
                "structured highlight span_text must be a string"
            )
        if not self.span_text.strip():
            raise MalformedStructuredHighlightResponse(
                "structured highlight span_text must be non-empty"
            )


class StructuredHighlightParser:
    """Parse a complete JSON envelope without partial-response recovery."""

    def parse(self, response: str) -> tuple[StructuredHighlight, ...]:
        try:
            payload = json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise MalformedStructuredHighlightResponse(
                "structured highlight JSON is malformed or truncated"
            ) from exc
        if not isinstance(payload, dict):
            raise MalformedStructuredHighlightResponse(
                "structured highlight response must be a JSON object"
            )

        if set(payload) != {"highlights"}:
            raise MalformedStructuredHighlightResponse(
                "structured highlight response contains undeclared fields"
            )
        highlights = payload["highlights"]
        if not isinstance(highlights, list):
            raise MalformedStructuredHighlightResponse(
                "structured highlight response requires a highlights list"
            )

        parsed: list[StructuredHighlight] = []
        for item in highlights:
            if not isinstance(item, dict):
                raise MalformedStructuredHighlightResponse(
                    "every structured highlight must be a JSON object"
                )
            if set(item) != {"doc_id", "span_text"}:
                raise MalformedStructuredHighlightResponse(
                    "every structured highlight must contain exactly "
                    "doc_id and span_text"
                )
            parsed.append(
                StructuredHighlight(
                    doc_id=item["doc_id"],
                    span_text=item["span_text"],
                )
            )
        return tuple(parsed)


__all__ = [
    "AmbiguousSourceSpan",
    "MalformedStructuredHighlightResponse",
    "SourceSpan",
    "SourceSpanNotFound",
    "StructuredHighlight",
    "StructuredHighlightParser",
    "UniqueSourceSpanLocator",
]
