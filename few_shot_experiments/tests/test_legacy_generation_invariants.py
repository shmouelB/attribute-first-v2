import json
from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import response_parsers  # noqa: E402
from attribute_first.application.iterative_sentence_generation import (  # noqa: E402
    parse_iterative_sentence_response,
)
from attribute_first.prompting.highlights import get_highlighted_doc  # noqa: E402
from attribute_first.prompting.templates import make_demo  # noqa: E402


class LegacyClusteringParserTests(unittest.TestCase):
    PROMPT = (
        "The highlighted spans are:\n"
        "1. Alpha evidence\n"
        "2. Beta evidence\n"
        "3. Gamma evidence"
    )

    def test_accepts_a_non_empty_exact_partition(self):
        parsed = response_parsers.parse_clustering_response(
            json.dumps(
                [
                    {"cluster": [1, 3]},
                    {"cluster": [2]},
                ]
            ),
            self.PROMPT,
        )

        self.assertEqual(
            parsed["final_output"],
            [
                {"cluster": [1, 3]},
                {"cluster": [2]},
            ],
        )

    def test_rejects_non_partitions_and_non_integer_ids(self):
        invalid_clusters = (
            ("no clusters", []),
            ("empty cluster", [{"cluster": []}, {"cluster": [1, 2, 3]}]),
            ("missing highlight", [{"cluster": [1]}, {"cluster": [3]}]),
            (
                "duplicate highlight",
                [{"cluster": [1, 2]}, {"cluster": [2, 3]}],
            ),
            ("out-of-range highlight", [{"cluster": [1, 2]}, {"cluster": [4]}]),
            (
                "boolean highlight",
                [{"cluster": [1, True]}, {"cluster": [2, 3]}],
            ),
        )

        for label, clusters in invalid_clusters:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    response_parsers.parse_clustering_response(
                        json.dumps(clusters),
                        self.PROMPT,
                    )


class LegacyGenerationInputValidationTests(unittest.TestCase):
    def test_demo_template_without_document_placeholder_is_valid(self):
        rendered, highlights = make_demo(
            item={"docs": []},
            prompt="{INST}\nNo documents are needed.",
            instruction="Follow the instruction.",
        )

        self.assertEqual(
            rendered,
            "Follow the instruction.\nNo documents are needed.",
        )
        self.assertEqual(highlights, [])

    def test_alce_associates_a_citation_adjacent_to_sentence_end(self):
        prompt = (
            "If multiple passages support the sentence, only cite a minimum "
            "sufficient subset of the passages.\n"
            "Document [1]: Alpha evidence."
        )

        parsed = response_parsers.parse_ALCE_response(
            "Claim.[1]",
            prompt,
        )

        self.assertEqual(
            parsed["final_output"],
            [{"sent": "Claim.", "cited_docs": [1]}],
        )

    def test_alce_rejects_document_zero_before_sentence_processing(self):
        prompt = (
            "If multiple passages support the sentence, only cite a minimum "
            "sufficient subset of the passages.\n"
            "Document [1]: Alpha evidence."
        )

        with self.assertRaisesRegex(ValueError, "relevant documents"):
            response_parsers.parse_ALCE_response(
                "Alpha claim [0].",
                prompt,
            )

    def test_highlight_offsets_must_reconstruct_the_declared_text(self):
        highlights = [
            {
                "documentFile": "doc-1",
                "docSentCharIdx": 0,
                "docSpanText": "Gamma",
                "docSpanOffsets": [[0, 5]],
            }
        ]

        with self.assertRaisesRegex(ValueError, "docSpanOffsets"):
            get_highlighted_doc(
                {"doc-1": "Alpha beta."},
                highlights,
                "{HS}",
                "{HE}",
            )

    def test_iterative_parser_rejects_instruction_echo(self):
        with self.assertRaisesRegex(ValueError, "next sentence"):
            parse_iterative_sentence_response(
                "The next sentence should summarize Alpha.",
                "unused",
            )


if __name__ == "__main__":
    unittest.main()
