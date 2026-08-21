"""Regression tests for post-generation standard-stage failures."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from attribute_first.application.standard_pipeline import (  # noqa: E402
    StandardPipelineRunner,
)


class StandardPipelineRecoveryTests(unittest.TestCase):
    @staticmethod
    def _state():
        return SimpleNamespace(
            outdir="/claimed/stage",
            rerun_context=None,
            used_demos=[{"unique_id": "demo"}],
            existing_results={},
            additional_data={"u1": {}},
            all_alignments=[{"unique_id": "u1", "response": "Gold."}],
            args=SimpleNamespace(
                subtask="content_selection",
                model_name="models/test",
            ),
        )

    @staticmethod
    def _runner(*, converter):
        events = []

        def write_json(path, _payload):
            events.append(("write_json", path))

        def save_results(outdir, _used_demos, _results, _pipeline):
            events.append(("save_results", outdir))

        artifact_store = SimpleNamespace(
            write_json=mock.Mock(side_effect=write_json),
            write_jsonl=mock.Mock(),
        )
        save_results = mock.Mock(side_effect=save_results)
        dependencies = SimpleNamespace(
            artifact_store=artifact_store,
            save_results=save_results,
            remove_pipeline_artifact=mock.Mock(),
            get_token_usage=lambda: {
                "calls": 1,
                "prompt": 10,
                "completion": 2,
                "cached": 0,
            },
            events=events,
        )
        runner = StandardPipelineRunner(dependencies)
        state = StandardPipelineRecoveryTests._state()
        final_results = {
            "u1": {
                "gold_summary": "Gold.",
                "final_output": {"Document [1]": ["Evidence"]},
                "attempt_trace": [
                    {
                        "status": "parsed",
                        "raw_response": '{"highlights":[]}',
                    }
                ],
            }
        }
        runner._prepare = mock.Mock(return_value=state)
        runner._generate = mock.Mock(return_value={"u1": {}})
        runner.result_assembler = SimpleNamespace(
            assemble=mock.Mock(return_value=final_results)
        )
        runner._convert = converter
        return runner, dependencies, state, final_results

    def test_conversion_failure_keeps_generation_evidence_and_usage(self):
        converter = mock.Mock(
            side_effect=ValueError("ambiguous source occurrence")
        )
        runner, dependencies, state, final_results = self._runner(
            converter=converter
        )

        with self.assertRaisesRegex(ValueError, "ambiguous source"):
            runner.run(SimpleNamespace())

        dependencies.save_results.assert_called_once_with(
            state.outdir,
            state.used_demos,
            final_results,
            None,
        )
        self.assertEqual(
            dependencies.artifact_store.write_json.call_args_list,
            [
                mock.call(f"{state.outdir}/results.json", final_results),
                mock.call(
                    f"{state.outdir}/token_usage.json",
                    {
                        "calls": 1,
                        "prompt": 10,
                        "completion": 2,
                        "cached": 0,
                        "subtask": "content_selection",
                        "model": "models/test",
                    },
                ),
            ],
        )
        self.assertEqual(
            dependencies.events[:3],
            [
                ("write_json", f"{state.outdir}/results.json"),
                ("write_json", f"{state.outdir}/token_usage.json"),
                ("save_results", state.outdir),
            ],
        )
        dependencies.artifact_store.write_jsonl.assert_not_called()

    def test_supporting_artifact_failure_keeps_minimal_checkpoint(self):
        converter = mock.Mock(return_value=[{"unique_id": "u1"}])
        runner, dependencies, state, final_results = self._runner(
            converter=converter
        )
        dependencies.save_results.side_effect = OSError("csv failed")

        with self.assertRaisesRegex(OSError, "csv failed"):
            runner.run(SimpleNamespace())

        self.assertEqual(
            dependencies.artifact_store.write_json.call_args_list,
            [
                mock.call(f"{state.outdir}/results.json", final_results),
                mock.call(
                    f"{state.outdir}/token_usage.json",
                    mock.ANY,
                ),
            ],
        )
        converter.assert_not_called()
        dependencies.artifact_store.write_jsonl.assert_not_called()

    def test_success_publishes_conversion_without_rewriting_raw_results(self):
        pipeline_results = [{"unique_id": "u1"}]
        converter = mock.Mock(return_value=pipeline_results)
        runner, dependencies, state, final_results = self._runner(
            converter=converter
        )

        runner.run(SimpleNamespace())

        dependencies.save_results.assert_called_once_with(
            state.outdir,
            state.used_demos,
            final_results,
            None,
        )
        dependencies.artifact_store.write_jsonl.assert_called_once_with(
            f"{state.outdir}/pipeline_format_results.json",
            pipeline_results,
        )


if __name__ == "__main__":
    unittest.main()
