"""Strict parser for schema-bound fusion responses.

This focused parser is intentionally independent from the legacy
``response_parsers`` module.  The controlled runner can therefore be imported
as a package even when another top-level module called ``utils`` is present.
"""

import json
import re

import spacy


class IncompleteFiCResponseError(ValueError):
    """The response cannot prove a complete FiC attribution plan."""


class StructuredFusionParser:
    """Parse and validate one structured fusion response."""

    _cot_prefix = re.compile(
        r"^\s*highlights?\s+[\d,\sand]*\s+(?:is|are)\s+combined "
        r"to form sentence\s+\d+\s*:\s*",
        re.IGNORECASE,
    )
    _sentinel = re.compile(
        r"^\s*so the final (?:summary|answer) is\s*:",
        re.IGNORECASE,
    )
    _nlp = None

    @classmethod
    def _sentence_parser(cls):
        if cls._nlp is None:
            cls._nlp = spacy.load("en_core_web_sm")
        return cls._nlp

    @staticmethod
    def _expected_highlight_ids(prompt):
        if not prompt or "The highlighted spans are:" not in prompt:
            return None
        target = prompt
        target_header = "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###"
        if target_header in target:
            target = target.rsplit(target_header, 1)[-1]
        numbered = target.rsplit("The highlighted spans are:", 1)[-1]
        if "SENTENCE PLAN" in numbered:
            numbered = numbered.split("SENTENCE PLAN", 1)[0]
        identifiers = {
            int(match.group(1))
            for match in re.finditer(r"(?m)^\s*(\d+)\.\s+\S", numbered)
        }
        return identifiers or None

    @staticmethod
    def _validate_alignments(alignments, expected_highlight_ids=None):
        if not isinstance(alignments, list) or not alignments:
            raise IncompleteFiCResponseError(
                "incomplete FiC output: no attributed sentences"
            )
        sentence_ids = [
            alignment.get("sent_id") for alignment in alignments
        ]
        if sentence_ids != list(range(1, len(alignments) + 1)):
            raise IncompleteFiCResponseError(
                "incomplete FiC output: sentence IDs must be consecutive "
                "from 1"
            )

        seen = []
        for alignment in alignments:
            text = alignment.get("sent_text")
            highlight_ids = alignment.get("highlights")
            if not isinstance(text, str) or not text.strip():
                raise IncompleteFiCResponseError(
                    "incomplete FiC output: every sentence needs text"
                )
            if not isinstance(highlight_ids, list) or not highlight_ids:
                raise IncompleteFiCResponseError(
                    "incomplete FiC output: every sentence needs highlight IDs"
                )
            if any(
                type(identifier) is not int or identifier < 1
                for identifier in highlight_ids
            ):
                raise IncompleteFiCResponseError(
                    "incomplete FiC output: highlight IDs must be positive "
                    "integers"
                )
            seen.extend(highlight_ids)

        if len(seen) != len(set(seen)):
            raise IncompleteFiCResponseError(
                "incomplete FiC output: highlight IDs must be used exactly "
                "once"
            )
        if (
            expected_highlight_ids is not None
            and set(seen) != set(expected_highlight_ids)
        ):
            missing = sorted(set(expected_highlight_ids) - set(seen))
            extra = sorted(set(seen) - set(expected_highlight_ids))
            raise IncompleteFiCResponseError(
                "incomplete FiC highlight coverage: "
                f"missing={missing}, extra={extra}"
            )

    def parse(self, response, prompt=None):
        """Return the legacy parser shape after strict completeness checks."""
        try:
            data = json.loads(response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise IncompleteFiCResponseError(
                "incomplete FiC output: structured JSON is malformed or "
                "truncated"
            ) from exc
        if not isinstance(data, dict):
            raise IncompleteFiCResponseError(
                "incomplete FiC output: structured response must be a JSON "
                "object"
            )
        if set(data) != {"sentences"}:
            raise IncompleteFiCResponseError(
                "incomplete FiC output: structured response contains "
                "undeclared fields"
            )
        sentences = data["sentences"]
        if not isinstance(sentences, list):
            raise IncompleteFiCResponseError(
                "incomplete FiC output: structured response has no sentences "
                "list"
            )

        alignments = []
        for sentence in sentences:
            if not isinstance(sentence, dict):
                raise IncompleteFiCResponseError(
                    "incomplete FiC output: every structured sentence must "
                    "be an object"
                )
            if set(sentence) != {
                "sentence_id",
                "sentence_text",
                "highlight_ids",
            }:
                raise IncompleteFiCResponseError(
                    "incomplete FiC output: a structured sentence contains "
                    "undeclared or missing fields"
                )
            raw_text = sentence["sentence_text"]
            if self._sentinel.match(raw_text or ""):
                continue
            clean_text = self._cot_prefix.sub("", raw_text or "").strip()
            physical_sentences = [
                physical.text.strip()
                for physical in self._sentence_parser()(clean_text).sents
                if physical.text.strip()
            ]
            if len(physical_sentences) != 1:
                raise IncompleteFiCResponseError(
                    "incomplete FiC output: every structured item must "
                    "contain exactly one physical sentence"
                )
            alignments.append(
                {
                    "sent_id": sentence.get("sentence_id"),
                    "highlights": sentence["highlight_ids"],
                    "sent_text": clean_text,
                }
            )

        self._validate_alignments(
            alignments,
            expected_highlight_ids=self._expected_highlight_ids(prompt),
        )
        return {
            "alignments": alignments,
            "final_output": " ".join(
                alignment["sent_text"] for alignment in alignments
            ),
            "full_model_response": response,
            "abstained": False,
        }


def parse_structured_fusion(response, prompt=None):
    """Compatibility function around :class:`StructuredFusionParser`."""
    return StructuredFusionParser().parse(response, prompt=prompt)
