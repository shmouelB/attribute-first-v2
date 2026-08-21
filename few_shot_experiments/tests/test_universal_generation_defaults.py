"""Behavioral contract for the supervisor-approved generation defaults."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_full_pipeline  # noqa: E402
import run_iterative_sentence_generation  # noqa: E402
import run_script  # noqa: E402
from attribute_first.application.dialogue_preparation import (  # noqa: E402
    DialoguePlanBuilder,
)
from attribute_first.application.standard_pipeline import (  # noqa: E402
    StandardPipelineRunner,
)
from attribute_first.stages import StageConfigContract  # noqa: E402


EXPECTED_MODEL = "models/gemini-3-flash-preview"


class GenerationCliDefaultTests(unittest.TestCase):
    def test_single_stage_defaults_to_structured_roles_and_gemini_three(self):
        args = run_script._argument_parser().parse_args([])

        self.assertEqual(args.model_name, EXPECTED_MODEL)
        self.assertIs(args.structured_output, True)
        self.assertIs(args.use_roles, True)

    def test_full_pipeline_defaults_to_structured_roles_and_gemini_three(self):
        args = run_full_pipeline._build_argparser().parse_args(
            ["--config-file", "pipeline.json"]
        )

        self.assertEqual(args.model_name, EXPECTED_MODEL)
        self.assertIs(args.structured_output, True)
        self.assertIs(args.use_roles, True)

    def test_legacy_iterative_cli_no_longer_defaults_to_an_old_model(self):
        self.assertEqual(
            run_iterative_sentence_generation.MODEL_DEFAULT,
            EXPECTED_MODEL,
        )
        source = (
            EXPERIMENT_ROOT / "run_iterative_sentence_generation.py"
        ).read_text(encoding="utf-8")
        self.assertIn("default=MODEL_DEFAULT", source)

    def test_single_stage_keeps_explicit_legacy_opt_outs(self):
        args = run_script._argument_parser().parse_args(
            ["--no-structured-output", "--no-roles"]
        )

        self.assertIs(args.structured_output, False)
        self.assertIs(args.use_roles, False)

    def test_full_pipeline_keeps_explicit_legacy_opt_outs(self):
        args = run_full_pipeline._build_argparser().parse_args(
            [
                "--config-file",
                "pipeline.json",
                "--no-structured-output",
                "--no-roles",
            ]
        )

        self.assertIs(args.structured_output, False)
        self.assertIs(args.use_roles, False)


class EffectiveRoleTransportTests(unittest.TestCase):
    @staticmethod
    def _observe_single_stage(args, *, environment=None):
        observed = {}

        def observe(effective_args):
            observed["roles"] = run_script.get_af_environment_flags()[
                "AF_USE_ROLES"
            ]
            observed["args"] = effective_args
            return "observed"

        with (
            mock.patch.dict(
                os.environ,
                environment or {},
                clear=True,
            ),
            mock.patch.object(
                run_script,
                "_run_with_effective_environment",
                side_effect=observe,
            ),
        ):
            result = run_script.main(args)
        return result, observed

    def test_single_stage_default_activates_real_role_transport(self):
        args = run_script._argument_parser().parse_args(
            [
                "--split",
                "test",
                "--setting",
                "MDS",
                "--subtask",
                "content_selection",
            ]
        )

        result, observed = self._observe_single_stage(args)

        self.assertEqual(result, "observed")
        self.assertIs(observed["roles"], True)

    def test_single_stage_role_opt_out_deactivates_real_role_transport(self):
        args = run_script._argument_parser().parse_args(
            [
                "--split",
                "test",
                "--setting",
                "MDS",
                "--subtask",
                "content_selection",
                "--no-roles",
            ]
        )

        _, observed = self._observe_single_stage(args)

        self.assertIs(observed["roles"], False)

    def test_existing_false_environment_flag_remains_an_explicit_opt_out(self):
        args = run_script._argument_parser().parse_args(
            [
                "--split",
                "test",
                "--setting",
                "MDS",
                "--subtask",
                "content_selection",
            ]
        )

        _, observed = self._observe_single_stage(
            args,
            environment={"AF_USE_ROLES": "false"},
        )

        self.assertIs(observed["roles"], False)

    def test_explicit_stage_config_remains_authoritative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "stage.json"
            config_path.write_text(
                json.dumps(
                    {
                        "split": "test",
                        "setting": "MDS",
                        "subtask": "content_selection",
                        "model_name": "models/gemini-pro-latest",
                        "structured_output": False,
                        "protocol": {
                            "environment_flags": {
                                "AF_USE_ROLES": False,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = run_script._argument_parser().parse_args(
                ["--config-file", str(config_path)]
            )

            _, observed = self._observe_single_stage(args)

        self.assertIs(observed["roles"], False)
        self.assertIs(observed["args"].structured_output, False)
        self.assertEqual(
            observed["args"].model_name,
            "models/gemini-pro-latest",
        )

    @staticmethod
    def _write_dialogue_pipeline(root, *, explicit_roles=None):
        stage_paths = []
        for stage_name in ("content_selection", "fusion_in_context"):
            stage_path = root / f"{stage_name}.json"
            payload = {"model_name": EXPECTED_MODEL}
            if explicit_roles is not None:
                payload["protocol"] = {
                    "environment_flags": {
                        "AF_USE_ROLES": explicit_roles,
                    }
                }
            stage_path.write_text(json.dumps(payload), encoding="utf-8")
            stage_paths.append(stage_path)
        pipeline_path = root / "pipeline.json"
        pipeline_path.write_text(
            json.dumps(
                [
                    {
                        "subtask": "content_selection",
                        "config_file": str(stage_paths[0]),
                    },
                    {
                        "subtask": "fusion_in_context",
                        "config_file": str(stage_paths[1]),
                    },
                ]
            ),
            encoding="utf-8",
        )
        return pipeline_path

    @staticmethod
    def _observe_full_dialogue(args):
        def observe(_runner, effective_args):
            full_configs = json.loads(
                Path(effective_args.config_file).read_text(encoding="utf-8")
            )
            with run_full_pipeline.dialogue_protocol_environment(full_configs):
                return run_full_pipeline.get_af_environment_flags()[
                    "AF_USE_ROLES"
                ]

        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                run_full_pipeline.PipelineApplicationRunner,
                "run",
                new=observe,
            ),
        ):
            return run_full_pipeline.main(args)

    def test_dialogue_pipeline_default_activates_real_role_transport(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = self._write_dialogue_pipeline(Path(tmpdir))
            args = run_full_pipeline._build_argparser().parse_args(
                [
                    "--config-file",
                    str(pipeline_path),
                    "--dialogue-mode",
                ]
            )

            observed_roles = self._observe_full_dialogue(args)

        self.assertIs(observed_roles, True)

    def test_dialogue_pipeline_explicit_protocol_remains_authoritative(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_path = self._write_dialogue_pipeline(
                Path(tmpdir),
                explicit_roles=False,
            )
            args = run_full_pipeline._build_argparser().parse_args(
                [
                    "--config-file",
                    str(pipeline_path),
                    "--dialogue-mode",
                ]
            )

            observed_roles = self._observe_full_dialogue(args)

        self.assertIs(observed_roles, False)


class ProgrammaticStageDefaultTests(unittest.TestCase):
    def test_standard_runner_defaults_missing_structured_flag_to_json(self):
        registry = mock.Mock()
        registry.resolve.return_value = SimpleNamespace(
            prompt_subtask_name="content_selection"
        )
        dependencies = SimpleNamespace(
            subtasks_without_given_highlights=("content_selection",),
            effective_generation_settings=lambda _args: (0, 0.0),
            load_rerun_source=lambda _args, _outdir: None,
            get_environment_flags=lambda: {},
            get_data=lambda _args: ({}, []),
            stage_registry=registry,
            get_subtask_prompt_structures=mock.Mock(return_value={}),
            construct_prompts=mock.Mock(return_value=([], {}, {}, {})),
            get_token_counter=lambda *_args: {},
            artifact_store=SimpleNamespace(write_json=mock.Mock()),
        )
        args = SimpleNamespace(
            subtask="content_selection",
            setting="MDS",
            split="test",
            outdir="/offline/default-structured",
            CoT=False,
            merge_cross_sents_highlights=False,
            cut_surplus=False,
            prct_surplus=None,
            always_with_question=False,
            debugging=False,
            model_name=EXPECTED_MODEL,
            prompt_token_budget=30000,
            seed=7,
        )

        with mock.patch(
            "attribute_first.application.standard_pipeline."
            "OutputDirectoryClaim.claim",
            return_value=Path(args.outdir),
        ):
            StandardPipelineRunner(dependencies)._prepare(args)

        registry.resolve.assert_called_once_with(
            "content_selection",
            structured_output=True,
        )

    def test_dialogue_builder_defaults_missing_structured_flag_to_json(self):
        schema = object()
        structures = mock.Mock(return_value={})
        selectors = mock.Mock(
            return_value=(lambda value: value, lambda value: value)
        )
        dependencies = SimpleNamespace(
            get_token_counter=lambda *_args: {},
            load_subtask_prompt_dict=lambda _args: {},
            get_subtask_prompt_structures=structures,
            get_data=lambda _args: (None, []),
            construct_prompts=mock.Mock(return_value=([], {}, {}, {})),
            get_subtask_funcs=selectors,
            subtask_schemas={"content_selection": schema},
        )
        args = SimpleNamespace(
            model_name=EXPECTED_MODEL,
            setting="MDS",
            CoT=False,
        )

        stage, *_ = DialoguePlanBuilder(
            dependencies
        )._prepare_content_selection(args, no_demos=True)

        self.assertIs(stage.schema, schema)
        self.assertIs(
            structures.call_args.kwargs["structured_output"],
            True,
        )
        selectors.assert_called_once_with(
            "content_selection",
            structured_output=True,
        )

    def test_stage_contract_uses_the_same_universal_defaults(self):
        contract = StageConfigContract.from_mapping(
            {"subtask": "content_selection"},
            declared_subtask="content_selection",
            strict=False,
        )

        self.assertEqual(contract.model_name, EXPECTED_MODEL)
        self.assertIs(contract.structured_output, True)

    def test_explicit_legacy_stage_values_remain_supported(self):
        contract = StageConfigContract.from_mapping(
            {
                "subtask": "content_selection",
                "model_name": "models/gemini-pro-latest",
                "structured_output": False,
            },
            declared_subtask="content_selection",
            strict=False,
        )

        self.assertEqual(
            contract.model_name,
            "models/gemini-pro-latest",
        )
        self.assertIs(contract.structured_output, False)

    def test_controlled_stage_contracts_stay_explicit_and_valid(self):
        paths = sorted(
            (
                EXPERIMENT_ROOT
                / "configs"
                / "controlled"
                / "test"
            ).glob("*/stages/*.json")
        )
        self.assertEqual(len(paths), 12)

        for path in paths:
            with self.subTest(path=path):
                config = json.loads(path.read_text(encoding="utf-8"))
                contract = StageConfigContract.from_mapping(
                    config,
                    declared_subtask=config["subtask"],
                    strict=True,
                )
                self.assertEqual(contract.model_name, EXPECTED_MODEL)
                self.assertIs(contract.structured_output, True)
                self.assertIs(
                    config["protocol"]["environment_flags"][
                        "AF_USE_ROLES"
                    ],
                    True,
                )


if __name__ == "__main__":
    unittest.main()
