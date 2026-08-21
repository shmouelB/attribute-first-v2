"""Offline contracts for the standard-stage object registry."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


import pipeline_converters  # noqa: E402
import prompt_utils  # noqa: E402
import response_parsers  # noqa: E402
import schemas  # noqa: E402
import subtask_specific_utils  # noqa: E402
from attribute_first.application.standard_pipeline import (  # noqa: E402
    StandardPipelineRunner,
)
from attribute_first.compatibility.stage_aliases import (  # noqa: E402
    STAGE_ALIASES,
)
from attribute_first.domain import StageKind  # noqa: E402
from attribute_first.stages.registry import (  # noqa: E402
    LegacyStageRegistryAdapter,
    StageBinding,
    StageRegistry,
)
from attribute_first.stages.standard_registry import (  # noqa: E402
    DEFAULT_STAGE_REGISTRY,
)
from utils import SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS  # noqa: E402


class StageRegistryTests(unittest.TestCase):
    def test_registry_resolves_legacy_aliases_to_canonical_kinds(self):
        expected = {
            "CS": StageKind.CONTENT_SELECTION,
            "content_selection": StageKind.CONTENT_SELECTION,
            "AH": StageKind.CONTEXT_AUGMENTATION,
            "ambiguity_highlight": StageKind.CONTEXT_AUGMENTATION,
            "FiC": StageKind.FUSION_IN_CONTEXT,
            "fusion_in_context": StageKind.FUSION_IN_CONTEXT,
            "topic_outline_fusion": StageKind.FUSION_IN_CONTEXT,
            "topic_cluster_fusion": StageKind.FUSION_IN_CONTEXT,
            "FiC_v2": StageKind.FUSION_IN_CONTEXT,
            "clustering": StageKind.CLUSTERING,
            "e2e_only_setting": StageKind.END_TO_END,
            "ALCE": StageKind.ALCE,
        }

        for alias, kind in expected.items():
            with self.subTest(alias=alias):
                protocol = DEFAULT_STAGE_REGISTRY.resolve(
                    alias,
                    structured_output=False,
                )
                self.assertEqual(protocol.kind, kind)
                self.assertEqual(
                    protocol.canonical_name,
                    kind.value,
                )

    def test_every_public_standard_alias_resolves_its_prompt_contract(self):
        expected_prompt_names = {
            "cs": "content_selection",
            "content_selection": "content_selection",
            "ah": "ambiguity_highlight",
            "ambiguity_highlight": "ambiguity_highlight",
            "context_augmentation": "ambiguity_highlight",
            "clustering": "clustering",
            "fic": "FiC",
            "fusion_in_context": "FiC",
            "topic_outline_fusion": "topic_outline_fusion",
            "topic_cluster_fusion": "topic_cluster_fusion",
            "fic_v2": "FiC_v2",
            "fusion_in_context_v2": "FiC_v2",
            "e2e_only_setting": "e2e_only_setting",
            "end_to_end": "e2e_only_setting",
            "alce": "ALCE",
        }
        registered_aliases = {
            alias
            for alias, kind in STAGE_ALIASES.items()
            if kind is not StageKind.REORDERING
        }
        self.assertEqual(
            set(expected_prompt_names),
            registered_aliases,
        )

        for alias, prompt_name in expected_prompt_names.items():
            with self.subTest(alias=alias):
                protocol = DEFAULT_STAGE_REGISTRY.resolve(alias)
                self.assertEqual(
                    protocol.prompt_subtask_name,
                    prompt_name,
                )

    def test_structured_binding_selects_parser_converter_and_schema_together(self):
        protocol = DEFAULT_STAGE_REGISTRY.resolve(
            "content_selection",
            structured_output=True,
        )

        self.assertIs(
            protocol.parser,
            response_parsers.parse_content_selection_structured_response,
        )
        self.assertIs(
            protocol.converter,
            pipeline_converters
            .convert_content_selection_results_to_pipeline_format,
        )
        self.assertIs(
            protocol.response_schema,
            schemas.CONTENT_SELECTION_SCHEMA,
        )
        self.assertEqual(
            protocol.schema_name,
            "SUBTASK_SCHEMAS.content_selection",
        )

    def test_schema_selection_preserves_historical_alias_behavior(self):
        without_schema = DEFAULT_STAGE_REGISTRY.resolve(
            "topic_outline_fusion",
            structured_output=True,
        )
        with_schema = DEFAULT_STAGE_REGISTRY.resolve(
            "FiC",
            structured_output=True,
        )

        self.assertIsNone(without_schema.response_schema)
        self.assertIs(
            with_schema.response_schema,
            schemas.FIC_COT_SCHEMA,
        )

    def test_short_structured_aliases_keep_their_canonical_schema(self):
        expected_schemas = {
            "CS": schemas.CONTENT_SELECTION_SCHEMA,
            "AH": schemas.AMBIGUITY_HIGHLIGHT_SCHEMA,
        }

        for alias, expected_schema in expected_schemas.items():
            with self.subTest(alias=alias):
                protocol = DEFAULT_STAGE_REGISTRY.resolve(
                    alias,
                    structured_output=True,
                )
                self.assertIs(protocol.response_schema, expected_schema)

    def test_legacy_adapter_falls_back_to_the_canonical_schema(self):
        parser = lambda value: value
        converter = lambda value, source: (value, source)

        def resolver(name, *, structured_output=False):
            return parser, converter

        adapter = LegacyStageRegistryAdapter(
            DEFAULT_STAGE_REGISTRY,
            resolver,
            schemas.SUBTASK_SCHEMAS,
        )

        for alias, expected_schema in (
            ("CS", schemas.CONTENT_SELECTION_SCHEMA),
            ("AH", schemas.AMBIGUITY_HIGHLIGHT_SCHEMA),
        ):
            with self.subTest(alias=alias):
                protocol = adapter.resolve(
                    alias,
                    structured_output=True,
                )
                self.assertIs(protocol.response_schema, expected_schema)

    def test_binding_rejects_unsupported_structured_mode(self):
        binding = StageBinding(
            kind=StageKind.CLUSTERING,
            parser=lambda value: value,
            converter=lambda value, source: (value, source),
            prompt_subtask_name="clustering",
        )

        with self.assertRaisesRegex(
            ValueError,
            "clustering.*structured output",
        ):
            binding.resolve("clustering", structured_output=True)

    def test_unstructured_binding_never_leaks_a_response_schema(self):
        protocol = DEFAULT_STAGE_REGISTRY.resolve(
            "FiC",
            structured_output=False,
        )

        self.assertIs(protocol.parser, response_parsers.parse_FiC_response)
        self.assertIsNone(protocol.response_schema)
        self.assertIsNone(protocol.schema_name)

    def test_compatibility_facade_delegates_to_the_registry(self):
        for structured_output in (False, True):
            with self.subTest(structured_output=structured_output):
                protocol = DEFAULT_STAGE_REGISTRY.resolve(
                    "ambiguity_highlight",
                    structured_output=structured_output,
                )
                self.assertEqual(
                    subtask_specific_utils.get_subtask_funcs(
                        "ambiguity_highlight",
                        structured_output=structured_output,
                    ),
                    (protocol.parser, protocol.converter),
                )

    def test_every_historical_dispatch_keeps_its_callable_pair(self):
        cases = (
            (
                "FiC",
                response_parsers.parse_FiC_response,
                response_parsers.parse_FiC_structured_response,
                pipeline_converters
                .convert_FiC_CoT_results_to_pipeline_format,
            ),
            (
                "content_selection",
                response_parsers.parse_content_selection_response,
                response_parsers
                .parse_content_selection_structured_response,
                pipeline_converters
                .convert_content_selection_results_to_pipeline_format,
            ),
            (
                "clustering",
                response_parsers.parse_clustering_response,
                None,
                pipeline_converters
                .convert_clustering_results_to_pipeline_format,
            ),
            (
                "e2e_only_setting",
                response_parsers.parse_e2e_only_setting_response,
                None,
                pipeline_converters
                .convert_e2e_only_setting_to_pipeline_format,
            ),
            (
                "ALCE",
                response_parsers.parse_ALCE_response,
                None,
                pipeline_converters.convert_ALCE_to_pipeline_format,
            ),
            (
                "ambiguity_highlight",
                response_parsers.parse_ambiguity_highlight_response,
                response_parsers
                .parse_ambiguity_highlight_structured_response,
                pipeline_converters
                .convert_ambiguity_highlight_results_to_pipeline_format,
            ),
            (
                "topic_outline_fusion",
                response_parsers.parse_FiC_response,
                response_parsers.parse_FiC_structured_response,
                pipeline_converters
                .convert_FiC_CoT_results_to_pipeline_format,
            ),
            (
                "topic_cluster_fusion",
                response_parsers.parse_FiC_response,
                response_parsers.parse_FiC_structured_response,
                pipeline_converters
                .convert_FiC_CoT_results_to_pipeline_format,
            ),
            (
                "FiC_v2",
                response_parsers.parse_FiC_response,
                response_parsers.parse_FiC_structured_response,
                pipeline_converters
                .convert_FiC_CoT_results_to_pipeline_format,
            ),
        )

        for name, plain_parser, structured_parser, converter in cases:
            for structured, expected_parser in (
                (False, plain_parser),
                (True, structured_parser),
            ):
                with self.subTest(name=name, structured=structured):
                    if expected_parser is None:
                        with self.assertRaisesRegex(
                            ValueError,
                            "does not support structured output",
                        ):
                            subtask_specific_utils.get_subtask_funcs(
                                name,
                                structured_output=structured,
                            )
                        continue
                    observed = (
                        subtask_specific_utils.get_subtask_funcs(
                            name,
                            structured_output=structured,
                        )
                    )
                    self.assertEqual(
                        observed,
                        (expected_parser, converter),
                    )

    def test_registry_rejects_alias_collisions(self):
        parser = lambda value: value
        converter = lambda value, source: (value, source)
        first = StageBinding(
            kind=StageKind.CONTENT_SELECTION,
            aliases=("shared",),
            parser=parser,
            converter=converter,
            prompt_subtask_name="content_selection",
        )
        second = StageBinding(
            kind=StageKind.CLUSTERING,
            aliases=("SHARED",),
            parser=parser,
            converter=converter,
            prompt_subtask_name="clustering",
        )

        with self.assertRaisesRegex(ValueError, "alias.*shared.*collision"):
            StageRegistry((first, second))

    def test_binding_rejects_schema_alias_typo(self):
        with self.assertRaisesRegex(
            ValueError,
            "schema_aliases must be declared",
        ):
            StageBinding(
                kind=StageKind.CONTENT_SELECTION,
                parser=lambda value: value,
                converter=lambda value, source: (value, source),
                prompt_subtask_name="content_selection",
                structured_parser=lambda value: value,
                response_schema={"type": "object"},
                schema_name="schema.content_selection",
                schema_aliases=("content_seletion",),
            )

    def test_unknown_stage_keeps_the_public_failure_message(self):
        with self.assertRaisesRegex(
            Exception,
            "unknown_stage is not yet supported",
        ):
            subtask_specific_utils.get_subtask_funcs("unknown_stage")


class StandardRunnerRegistryTests(unittest.TestCase):
    def test_runner_consumes_one_resolved_binding(self):
        protocol = mock.Mock()
        protocol.prompt_subtask_name = "content_selection"
        registry = mock.Mock()
        registry.resolve.return_value = protocol
        prompt_details = object()
        dependencies = SimpleNamespace(
            stage_registry=registry,
            subtasks_without_given_highlights=(),
            get_subtask_prompt_structures=mock.Mock(
                return_value=prompt_details
            ),
        )
        runner = StandardPipelineRunner(dependencies)
        args = SimpleNamespace(
            subtask="CS",
            setting="MDS",
            CoT=False,
            always_with_question=False,
            cut_surplus=False,
            prct_surplus=None,
        )

        resolved, observed_prompt_details = runner._subtask_protocol(
            args,
            {"prompt": "definition"},
            True,
        )

        self.assertIs(resolved, protocol)
        self.assertIs(observed_prompt_details, prompt_details)
        registry.resolve.assert_called_once_with(
            "CS",
            structured_output=True,
        )

    def test_prepare_honors_every_public_alias_and_preserves_provenance(self):
        prompt_dict = json.loads(
            (EXPERIMENT_ROOT / "prompts" / "MDS.json").read_text(
                encoding="utf-8"
            )
        )
        expected_prompt_names = {
            "cs": "content_selection",
            "content_selection": "content_selection",
            "ah": "ambiguity_highlight",
            "ambiguity_highlight": "ambiguity_highlight",
            "context_augmentation": "ambiguity_highlight",
            "clustering": "clustering",
            "fic": "FiC",
            "fusion_in_context": "FiC",
            "topic_outline_fusion": "topic_outline_fusion",
            "topic_cluster_fusion": "topic_cluster_fusion",
            "fic_v2": "FiC_v2",
            "fusion_in_context_v2": "FiC_v2",
            "e2e_only_setting": "e2e_only_setting",
            "end_to_end": "e2e_only_setting",
            "alce": "ALCE",
        }

        for alias, prompt_name in expected_prompt_names.items():
            with self.subTest(alias=alias):
                artifact_store = SimpleNamespace(
                    write_json=mock.Mock()
                )
                construct_prompts = mock.Mock(
                    return_value=([], {}, {}, {})
                )
                dependencies = SimpleNamespace(
                    subtasks_without_given_highlights=(
                        SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS
                    ),
                    effective_generation_settings=lambda args: (0, 0.1),
                    load_rerun_source=lambda args, outdir: None,
                    get_environment_flags=lambda: {},
                    get_data=lambda args: (prompt_dict, []),
                    stage_registry=DEFAULT_STAGE_REGISTRY,
                    get_subtask_prompt_structures=(
                        prompt_utils.get_subtask_prompt_structures
                    ),
                    construct_prompts=construct_prompts,
                    get_token_counter=lambda *args: {},
                    artifact_store=artifact_store,
                )
                args = SimpleNamespace(
                    subtask=alias,
                    setting="MDS",
                    split="dev",
                    outdir=f"/offline/{alias}",
                    CoT=True,
                    merge_cross_sents_highlights=False,
                    cut_surplus=False,
                    prct_surplus=None,
                    always_with_question=False,
                    structured_output=False,
                    debugging=False,
                    model_name="models/test",
                    prompt_token_budget=30000,
                    seed=7,
                )

                with mock.patch(
                    "attribute_first.application.standard_pipeline."
                    "OutputDirectoryClaim.claim",
                    return_value=Path(args.outdir),
                ):
                    state = StandardPipelineRunner(dependencies)._prepare(
                        args
                    )

                self.assertEqual(args.subtask, alias)
                self.assertEqual(
                    state.stage_protocol.prompt_subtask_name,
                    prompt_name,
                )
                snapshot = artifact_store.write_json.call_args.args[1]
                self.assertEqual(snapshot["subtask"], alias)
                self.assertEqual(
                    construct_prompts.call_args.kwargs["no_highlights"],
                    (
                        prompt_name
                        in SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS
                    ),
                )


if __name__ == "__main__":
    unittest.main()
