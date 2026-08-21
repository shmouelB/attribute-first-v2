"""Exact-envelope contracts for schema-bound controlled stages."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


from attribute_first.stages.structured_fusion import (  # noqa: E402
    IncompleteFiCResponseError,
    StructuredFusionParser,
)
from attribute_first.stages.structured_highlights import (  # noqa: E402
    MalformedStructuredHighlightResponse,
    StructuredHighlightParser,
)


class StrictStructuredHighlightTests(unittest.TestCase):
    def test_rejects_extra_envelope_and_item_fields(self):
        parser = StructuredHighlightParser()
        invalid = (
            {
                "highlights": [
                    {"doc_id": "1", "span_text": "Alpha"}
                ],
                "commentary": "undeclared",
            },
            {
                "highlights": [
                    {
                        "doc_id": "1",
                        "span_text": "Alpha",
                        "confidence": 0.9,
                    }
                ]
            },
        )

        for payload in invalid:
            with self.subTest(payload=payload):
                with self.assertRaises(
                    MalformedStructuredHighlightResponse
                ):
                    parser.parse(json.dumps(payload))

    def test_rejects_an_empty_item_instead_of_partially_accepting(self):
        with self.assertRaisesRegex(
            MalformedStructuredHighlightResponse,
            "non-empty",
        ):
            StructuredHighlightParser().parse(
                json.dumps(
                    {
                        "highlights": [
                            {"doc_id": "1", "span_text": "Alpha"},
                            {"doc_id": "2", "span_text": "   "},
                        ]
                    }
                )
            )


class StrictStructuredFusionTests(unittest.TestCase):
    @staticmethod
    def _single_sentence_parser():
        return lambda text: SimpleNamespace(
            sents=[SimpleNamespace(text=text)]
        )

    def test_rejects_extra_envelope_and_sentence_fields(self):
        invalid = (
            {
                "sentences": [
                    {
                        "sentence_id": 1,
                        "sentence_text": "Alpha.",
                        "highlight_ids": [1],
                    }
                ],
                "commentary": "undeclared",
            },
            {
                "sentences": [
                    {
                        "sentence_id": 1,
                        "sentence_text": "Alpha.",
                        "highlight_ids": [1],
                        "confidence": 0.9,
                    }
                ]
            },
        )

        with mock.patch.object(
            StructuredFusionParser,
            "_sentence_parser",
            return_value=self._single_sentence_parser(),
        ):
            for payload in invalid:
                with self.subTest(payload=payload):
                    with self.assertRaises(IncompleteFiCResponseError):
                        StructuredFusionParser().parse(
                            json.dumps(payload),
                            prompt=(
                                "The highlighted spans are:\n"
                                "1. Alpha"
                            ),
                        )


if __name__ == "__main__":
    unittest.main()
