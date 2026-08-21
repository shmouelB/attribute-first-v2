"""Characterization of the sequential-dialogue module boundaries."""

from pathlib import Path
import sys
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


from attribute_first.application import sequential_dialogue  # noqa: E402
from attribute_first.application import sequential_contracts  # noqa: E402
from attribute_first.application import sequential_instance  # noqa: E402
from attribute_first.application import sequential_pipeline  # noqa: E402
from attribute_first.application import sequential_results  # noqa: E402


class SequentialModuleBoundaryTests(unittest.TestCase):
    def test_compatibility_module_reexports_contract_helpers(self):
        expected = {
            "parse_content_selection_spans": (
                sequential_contracts.parse_content_selection_spans
            ),
            "parse_clusters": sequential_contracts.parse_clusters,
            "parse_fusion_output": (
                sequential_contracts.parse_fusion_output
            ),
            "source_span_metadata": (
                sequential_contracts.source_span_metadata
            ),
            "usage_delta": sequential_contracts.usage_delta,
            "usage_from_attempt_trace": (
                sequential_contracts.usage_from_attempt_trace
            ),
        }

        for name, implementation in expected.items():
            with self.subTest(name=name):
                self.assertIs(
                    getattr(sequential_dialogue, name),
                    implementation,
                )

    def test_compatibility_module_reexports_runtime_objects(self):
        expected = {
            "SequentialInstanceDependencies": (
                sequential_instance.SequentialInstanceDependencies
            ),
            "SequentialDialogueInstanceRunner": (
                sequential_instance.SequentialDialogueInstanceRunner
            ),
            "SequentialPipelineResultAssembler": (
                sequential_results.SequentialPipelineResultAssembler
            ),
            "SequentialPipelineDependencies": (
                sequential_pipeline.SequentialPipelineDependencies
            ),
            "SequentialDialoguePipelineRunner": (
                sequential_pipeline.SequentialDialoguePipelineRunner
            ),
        }

        for name, implementation in expected.items():
            with self.subTest(name=name):
                self.assertIs(
                    getattr(sequential_dialogue, name),
                    implementation,
                )

    def test_protocol_constants_keep_their_public_identity(self):
        self.assertIs(
            sequential_dialogue.SEQUENTIAL_PROTOCOL,
            sequential_contracts.SEQUENTIAL_PROTOCOL,
        )
        self.assertEqual(
            sequential_dialogue.HIGHLIGHT_START,
            "<highlight_start>",
        )
        self.assertEqual(
            sequential_dialogue.HIGHLIGHT_END,
            "<highlight_end>",
        )


if __name__ == "__main__":
    unittest.main()
