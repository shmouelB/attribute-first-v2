"""Fail-fast contracts for full-pipeline stage composition."""

import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


from attribute_first.application.pipeline_application import (  # noqa: E402
    PipelineApplicationRunner,
)
from attribute_first.application.protocol import (  # noqa: E402
    dialogue_protocol_environment,
)


class PipelineApplicationValidationTests(unittest.TestCase):
    @staticmethod
    def _runner():
        return PipelineApplicationRunner(
            SimpleNamespace(
                config_protocol_environment=mock.Mock(),
                main_func=mock.Mock(),
                iterative_sent_gen_main=mock.Mock(),
                persist_pipeline_provenance=mock.Mock(),
                dialogue_protocol_environment=mock.Mock(),
                run_dialogue_pipeline=mock.Mock(),
                persist_pipeline_token_usage=mock.Mock(),
                persist_pipeline_response_metadata=mock.Mock(),
                run_subtask=mock.Mock(),
                log_stage_health=mock.Mock(),
            )
        )

    @staticmethod
    def _args(config_file, outdir, *, dialogue_mode=False):
        return SimpleNamespace(
            config_file=str(config_file),
            outdir=str(outdir),
            dialogue_mode=dialogue_mode,
            indir_alignments=None,
        )

    @staticmethod
    def _write_config(path, stages):
        Path(path).write_text(json.dumps(stages), encoding="utf-8")

    @staticmethod
    def _valid_stage_config(subtask):
        return {
            "split": "dev",
            "setting": "MDS",
            "subtask": subtask,
            "model_name": "models/gemini-3-flash-preview",
            "prompt_token_budget": 30000,
            "dialogue_history_token_budget": 200000,
            "n_demos": 1,
            "num_retries": 3,
            "temperature": 0.1,
            "structured_output": True,
            "output_max_length": 8192,
        }

    def _write_valid_direct_pipeline(self, root, first_config):
        content_selection = root / "content-selection.json"
        fusion = root / "fusion.json"
        self._write_config(content_selection, first_config)
        self._write_config(
            fusion,
            self._valid_stage_config("FiC"),
        )
        pipeline = root / "pipeline.json"
        self._write_config(
            pipeline,
            [
                {
                    "subtask": "content_selection",
                    "config_file": str(content_selection),
                },
                {
                    "subtask": "fusion_in_context",
                    "config_file": str(fusion),
                },
            ],
        )
        return pipeline

    def _materialize_repository_pipeline(self, root, relative_path):
        source_path = EXPERIMENT_ROOT / relative_path
        stages = json.loads(source_path.read_text(encoding="utf-8"))
        for stage in stages:
            stage["config_file"] = str(
                (EXPERIMENT_ROOT / stage["config_file"]).resolve()
            )
        destination = root / source_path.name
        self._write_config(destination, stages)
        return destination

    def test_duplicate_stage_is_rejected_before_output_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "pipeline.json"
            outdir = root / "out"
            self._write_config(
                config,
                [
                    {"subtask": "content_selection"},
                    {"subtask": "fusion_in_context"},
                    {"subtask": "fusion_in_context"},
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "duplicate|exact stage sequence",
            ):
                self._runner().run(self._args(config, outdir))

            self.assertFalse(outdir.exists())

    def test_stage_order_is_validated_before_output_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "pipeline.json"
            outdir = root / "out"
            self._write_config(
                config,
                [
                    {"subtask": "fusion_in_context"},
                    {"subtask": "content_selection"},
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "exact stage sequence",
            ):
                self._runner().run(self._args(config, outdir))

            self.assertFalse(outdir.exists())

    def test_dialogue_rejects_non_dialogue_fusion_before_output_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "pipeline.json"
            outdir = root / "out"
            self._write_config(
                config,
                [
                    {"subtask": "content_selection"},
                    {"subtask": "topic_outline_fusion"},
                ],
            )

            with self.assertRaisesRegex(
                ValueError,
                "dialogue.*fusion_in_context",
            ):
                self._runner().run(
                    self._args(config, outdir, dialogue_mode=True)
                )

            self.assertFalse(outdir.exists())

    def test_dialogue_rejects_mixed_models_before_claim_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            content_selection_config = root / "content-selection.json"
            ambiguity_config = root / "ambiguity-highlight.json"
            fusion_config = root / "fusion.json"
            self._write_config(
                content_selection_config,
                {
                    "split": "dev",
                    "setting": "MDS",
                    "model_name": "models/gemini-test",
                },
            )
            self._write_config(
                ambiguity_config,
                {
                    "split": "dev",
                    "setting": "MDS",
                    "model_name": "models/another-model",
                },
            )
            self._write_config(
                fusion_config,
                {
                    "split": "dev",
                    "setting": "MDS",
                    "model_name": "models/gemini-test",
                },
            )
            config = root / "pipeline.json"
            outdir = root / "out"
            self._write_config(
                config,
                [
                    {
                        "subtask": "content_selection",
                        "config_file": str(content_selection_config),
                    },
                    {
                        "subtask": "ambiguity_highlight",
                        "config_file": str(ambiguity_config),
                    },
                    {
                        "subtask": "fusion_in_context",
                        "config_file": str(fusion_config),
                    },
                ],
            )
            runner = self._runner()
            runner._dependencies.dialogue_protocol_environment = (
                lambda full_configs: dialogue_protocol_environment(
                    full_configs,
                    validate_protocol_environment_flags=lambda value: (
                        value or {}
                    ),
                    protocol_environment=lambda value: nullcontext(value),
                )
            )

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                side_effect=AssertionError("output was claimed"),
            ) as claim:
                with self.assertRaisesRegex(
                    ValueError,
                    "same model_name",
                ):
                    runner.run(
                        self._args(
                            config,
                            outdir,
                            dialogue_mode=True,
                        )
                    )

            claim.assert_not_called()
            dependencies = runner._dependencies
            dependencies.persist_pipeline_provenance.assert_not_called()
            dependencies.run_dialogue_pipeline.assert_not_called()
            dependencies.persist_pipeline_token_usage.assert_not_called()
            dependencies.persist_pipeline_response_metadata.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_stage_config_subtask_mismatch_is_rejected_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._write_valid_direct_pipeline(
                root,
                self._valid_stage_config("FiC"),
            )
            outdir = root / "out"
            runner = self._runner()

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                side_effect=AssertionError("output was claimed"),
            ) as claim:
                with self.assertRaisesRegex(
                    ValueError,
                    "subtask|stage kind",
                ):
                    runner.run(self._args(pipeline, outdir))

            claim.assert_not_called()
            dependencies = runner._dependencies
            dependencies.persist_pipeline_provenance.assert_not_called()
            dependencies.run_subtask.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_invalid_typed_stage_fields_are_rejected_before_claim(self):
        invalid_cases = (
            ("model_name", "", "model"),
            ("model_name", "gemini-3-flash-preview", "model"),
            ("model_name", 3, "model"),
            ("temperature", True, "temperature"),
            ("temperature", 2.1, "temperature"),
            ("output_max_length", True, "output"),
            ("output_max_length", 0, "output"),
            ("n_demos", True, "demonstration|n_demos"),
            ("n_demos", -1, "demonstration|n_demos"),
            ("structured_output", "true", "structured_output"),
            ("prompt_token_budget", True, "prompt|stage_prompt"),
            ("prompt_token_budget", 0, "prompt|stage_prompt"),
            (
                "dialogue_history_token_budget",
                10,
                "dialogue history|dialogue_history",
            ),
            ("num_retries", True, "attempt|num_retries"),
            ("num_retries", 0, "attempt|num_retries"),
        )
        for field, invalid_value, message in invalid_cases:
            with self.subTest(field=field, value=invalid_value):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    stage = self._valid_stage_config(
                        "content_selection"
                    )
                    stage[field] = invalid_value
                    pipeline = self._write_valid_direct_pipeline(
                        root,
                        stage,
                    )
                    outdir = root / "out"
                    runner = self._runner()

                    with mock.patch(
                        "attribute_first.application."
                        "pipeline_application.OutputDirectoryClaim.claim",
                        side_effect=AssertionError("output was claimed"),
                    ) as claim:
                        with self.assertRaisesRegex(
                            ValueError,
                            message,
                        ):
                            runner.run(self._args(pipeline, outdir))

                    claim.assert_not_called()
                    dependencies = runner._dependencies
                    dependencies.persist_pipeline_provenance.assert_not_called()
                    dependencies.run_subtask.assert_not_called()
                    self.assertFalse(outdir.exists())

    def test_controlled_stage_contract_requires_explicit_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = self._valid_stage_config("content_selection")
            stage.pop("output_max_length")
            pipeline = self._write_valid_direct_pipeline(root, stage)
            outdir = root / "out"
            args = self._args(pipeline, outdir)
            args.canonical_cell_id = "mds.direct_fs_independent"
            runner = self._runner()

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                side_effect=AssertionError("output was claimed"),
            ) as claim:
                with self.assertRaisesRegex(
                    ValueError,
                    "controlled.*missing.*output_max_length",
                ):
                    runner.run(args)

            claim.assert_not_called()
            runner._dependencies.run_subtask.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_controlled_stage_rejects_unknown_split_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._valid_stage_config("content_selection")
            first["split"] = "train"
            pipeline = self._write_valid_direct_pipeline(root, first)
            fusion = root / "fusion.json"
            second = json.loads(fusion.read_text(encoding="utf-8"))
            second["split"] = "train"
            self._write_config(fusion, second)
            outdir = root / "out"
            args = self._args(pipeline, outdir)
            args.canonical_cell_id = "mds.direct_fs_independent"
            runner = self._runner()

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                side_effect=AssertionError("output was claimed"),
            ) as claim:
                with self.assertRaisesRegex(ValueError, "split"):
                    runner.run(args)

            claim.assert_not_called()
            runner._dependencies.run_subtask.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_controlled_stage_rejects_unknown_setting_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = self._valid_stage_config("content_selection")
            first["setting"] = "QA"
            pipeline = self._write_valid_direct_pipeline(root, first)
            fusion = root / "fusion.json"
            second = json.loads(fusion.read_text(encoding="utf-8"))
            second["setting"] = "QA"
            self._write_config(fusion, second)
            outdir = root / "out"
            args = self._args(pipeline, outdir)
            args.canonical_cell_id = "mds.direct_fs_independent"
            runner = self._runner()

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                side_effect=AssertionError("output was claimed"),
            ) as claim:
                with self.assertRaisesRegex(ValueError, "setting"):
                    runner.run(args)

            claim.assert_not_called()
            runner._dependencies.run_subtask.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_controlled_stage_rejects_mixed_models_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._write_valid_direct_pipeline(
                root,
                self._valid_stage_config("content_selection"),
            )
            fusion = root / "fusion.json"
            second = json.loads(fusion.read_text(encoding="utf-8"))
            second["model_name"] = "models/gemini-another-model"
            self._write_config(fusion, second)
            outdir = root / "out"
            args = self._args(pipeline, outdir)
            args.canonical_cell_id = "mds.direct_fs_independent"
            runner = self._runner()

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                side_effect=AssertionError("output was claimed"),
            ) as claim:
                with self.assertRaisesRegex(ValueError, "same model_name"):
                    runner.run(args)

            claim.assert_not_called()
            runner._dependencies.run_subtask.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_legacy_independent_pipeline_preserves_mixed_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._write_valid_direct_pipeline(
                root,
                self._valid_stage_config("content_selection"),
            )
            fusion = root / "fusion.json"
            second = json.loads(fusion.read_text(encoding="utf-8"))
            second["model_name"] = "models/gemini-another-model"
            self._write_config(fusion, second)
            outdir = root / "out"
            runner = self._runner()

            with mock.patch(
                "attribute_first.application.pipeline_application."
                "OutputDirectoryClaim.claim",
                return_value=outdir,
            ):
                runner.run(self._args(pipeline, outdir))

            dependencies = runner._dependencies
            self.assertEqual(dependencies.run_subtask.call_count, 2)
            dependencies.persist_pipeline_provenance.assert_called_once()

    def test_repository_controlled_pipeline_passes_strict_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline = self._materialize_repository_pipeline(
                root,
                (
                    "configs/controlled/test/MDS/pipelines/"
                    "direct_few_shot_context_augmented.json"
                ),
            )
            outdir = root / "out"
            args = self._args(pipeline, outdir)
            args.canonical_cell_id = (
                "mds.direct_fs_independent.context_augmented"
            )
            runner = self._runner()

            with mock.patch(
                "attribute_first.application."
                "pipeline_application.OutputDirectoryClaim.claim",
                return_value=outdir,
            ):
                runner.run(args)

            dependencies = runner._dependencies
            self.assertEqual(
                dependencies.run_subtask.call_count,
                3,
            )
            dependencies.persist_pipeline_provenance.assert_called_once()
            dependencies.run_dialogue_pipeline.assert_not_called()

    def test_legacy_pipeline_configs_keep_their_effective_defaults(self):
        cases = (
            ("configs/test/MDS/full_pipeline.json", 3),
            ("configs/test/MDS/full_CoT_pipeline.json", 2),
            ("configs/dev/MDS/full_topic_outline_pipeline.json", 2),
            ("configs/dev/MDS/full_topic_cluster_pipeline.json", 2),
            ("configs/dev/LFQA/full_CoT_pipeline_v2.json", 2),
        )
        for relative_path, expected_stage_calls in cases:
            with self.subTest(pipeline=relative_path):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    pipeline = self._materialize_repository_pipeline(
                        root,
                        relative_path,
                    )
                    outdir = root / "out"
                    runner = self._runner()

                    with mock.patch(
                        "attribute_first.application."
                        "pipeline_application.OutputDirectoryClaim.claim",
                        return_value=outdir,
                    ):
                        runner.run(self._args(pipeline, outdir))

                    self.assertEqual(
                        runner._dependencies.run_subtask.call_count,
                        expected_stage_calls,
                    )
                    dependencies = runner._dependencies
                    dependencies.persist_pipeline_provenance.assert_called_once()
                    dependencies.run_dialogue_pipeline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
