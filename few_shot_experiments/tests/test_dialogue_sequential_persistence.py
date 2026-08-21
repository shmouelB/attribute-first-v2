import json
import inspect
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

import run_dialogue_sequential as dialogue


class _ChatSession:
    def __init__(self):
        self.history = []


class SequentialDialogueObjectTests(unittest.TestCase):
    def test_instance_runner_uses_injected_chat_boundary_and_rolls_back_fusion(self):
        from attribute_first.application.sequential_dialogue import (
            SequentialDialogueInstanceRunner,
            SequentialInstanceDependencies,
        )
        from attribute_first.infrastructure import CallableDialogueGateway
        from attribute_first.runtime import (
            IncompleteGenerationError,
            ensure_parseable_finish_reason,
        )

        session = _ChatSession()
        messages = []
        evidence = {"usage": None, "metadata": None}
        responses = iter(
            (
                json.dumps(
                    {
                        "highlights": [
                            {"doc_id": "1", "span_text": "Alpha"},
                            {"doc_id": "1", "span_text": "Beta"},
                        ]
                    }
                ),
                json.dumps({"clusters": [[1], [2]]}),
                json.dumps(
                    {
                        "sentence_text": "Alpha sentence.",
                        "highlight_ids": [1],
                    }
                ),
                json.dumps(
                    {
                        "sentence_text": "Beta sentence.",
                        "highlight_ids": [2],
                    }
                ),
            )
        )

        def send_message(active_session, message, **_kwargs):
            raw = next(responses)
            messages.append(message)
            evidence["usage"] = {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "cached_content_token_count": 0,
            }
            evidence["metadata"] = {
                "provider_response_received": True,
                "model_version": "fixture",
                "finish_reason": "STOP",
                "prompt_block_reason": None,
            }
            active_session.history.extend(
                [
                    {"role": "user", "parts": [message]},
                    {"role": "model", "parts": [raw]},
                ]
            )
            return raw

        def reset_evidence():
            evidence["usage"] = None
            evidence["metadata"] = None

        runner = SequentialDialogueInstanceRunner(
            SequentialInstanceDependencies(
                dialogue_gateway=CallableDialogueGateway(
                    create_chat=lambda _model, **_kwargs: session,
                    send_message=send_message,
                ),
                reset_last_call_usage=reset_evidence,
                get_last_call_usage=lambda: evidence["usage"],
                get_last_call_metadata=lambda: evidence["metadata"],
                ensure_parseable_finish_reason=(
                    ensure_parseable_finish_reason
                ),
                stable_value_sha256=lambda _value: "fixture-sha256",
                incomplete_generation_error=IncompleteGenerationError,
                time_module=SimpleNamespace(sleep=lambda _seconds: None),
                content_selection_schema={"name": "cs"},
                clustering_schema={"name": "clustering"},
                sentence_fusion_schema={"name": "fusion"},
            )
        )

        result, delta = runner.run(
            {
                "documents": [
                    {
                        "documentFile": "doc-a",
                        "rawDocumentText": "Alpha Beta",
                    }
                ]
            },
            "CS prompt",
            "Cluster instruction",
            "Fusion instruction",
            "models/test",
            num_retries=1,
        )

        self.assertEqual(
            result["final_output"],
            "Alpha sentence. Beta sentence.",
        )
        self.assertEqual(
            delta,
            {
                "prompt": 4,
                "completion": 4,
                "cached": 0,
                "calls": 4,
            },
        )
        self.assertEqual(len(session.history), 4)
        self.assertEqual(len(messages), 4)
        self.assertNotIn("Alpha sentence.", messages[3])
        self.assertNotIn(messages[2], messages[3])


class DialogueSequentialPersistenceTests(unittest.TestCase):
    def test_cli_declares_three_retries_by_default(self):
        args = dialogue._argument_parser().parse_args(
            ["--setting", "MDS", "-o", "new-output"]
        )

        self.assertEqual(args.num_retries, 3)

    def test_max_examples_is_rejected_before_loading_or_claiming(self):
        get_data = mock.Mock()
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "out"
            args = SimpleNamespace(
                setting="MDS",
                outdir=str(outdir),
                max_examples=1,
                num_retries=3,
            )
            with mock.patch.object(dialogue, "get_data", get_data):
                with self.assertRaisesRegex(
                    ValueError,
                    "max-examples.*fixed population",
                ):
                    dialogue.main(args)

            get_data.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_retry_count_must_be_a_positive_integer_before_loading(self):
        for invalid in (0, -1, True, 1.5, "3"):
            with self.subTest(invalid=invalid):
                get_data = mock.Mock()
                args = SimpleNamespace(
                    setting="MDS",
                    outdir="unused",
                    max_examples=None,
                    num_retries=invalid,
                )
                with mock.patch.object(dialogue, "get_data", get_data):
                    with self.assertRaisesRegex(
                        ValueError,
                        "num-retries.*positive integer",
                    ):
                        dialogue.main(args)
                get_data.assert_not_called()

    def test_missing_prompt_is_rejected_before_any_dialogue_call(self):
        sources = [
            {"unique_id": "u1", "response": "Gold one."},
            {"unique_id": "u2", "response": "Gold two."},
        ]
        run_instance = mock.Mock()

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "out"
            args = SimpleNamespace(
                setting="MDS",
                split="test",
                indir_alignments=None,
                indir_prompt=None,
                outdir=str(outdir),
                model="models/test",
                n_demos=1,
                concurrency=1,
                max_examples=None,
                seed=20260728,
            )
            with mock.patch.object(
                dialogue,
                "get_data",
                return_value=(
                    {
                        "instruction-clustering": "cluster",
                        "instruction-next-cluster-fusion": "fuse",
                    },
                    sources,
                ),
            ), mock.patch.object(
                dialogue,
                "get_subtask_prompt_structures",
                return_value={},
            ), mock.patch.object(
                dialogue,
                "construct_prompts",
                return_value=([], {"u1": "CS"}, {}, {}),
            ), mock.patch.object(
                dialogue,
                "get_token_counter",
                return_value={},
            ), mock.patch.object(
                dialogue,
                "run_instance",
                run_instance,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "fixed-population coverage",
                ):
                    dialogue.main(args)

            run_instance.assert_not_called()
            self.assertFalse(outdir.exists())

    def test_conversion_failure_saves_raw_results_and_exits_nonzero(self):
        from attribute_first.application.sequential_dialogue import (
            SequentialDialoguePipelineRunner,
            SequentialPipelineDependencies,
        )

        save_results = mock.Mock()
        artifact_store = mock.Mock()
        runner = SequentialDialoguePipelineRunner(
            SequentialPipelineDependencies(
                get_data=mock.Mock(),
                get_prompt_structures=mock.Mock(),
                construct_prompts=mock.Mock(),
                get_token_counter=mock.Mock(),
                reset_token_usage=mock.Mock(),
                get_token_usage=lambda: {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "calls": 0,
                },
                run_instance=mock.Mock(),
                save_results=save_results,
                get_environment_flags=lambda: {},
                artifact_store=artifact_store,
                build_pipeline_results=mock.Mock(
                    side_effect=ValueError("invalid conversion")
                ),
            )
        )
        results = {
            "u1": {
                "final_output": "Generated.",
                "alignments": [],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp)
            with self.assertRaisesRegex(
                RuntimeError,
                "pipeline conversion failed",
            ):
                runner._persist(
                    args=SimpleNamespace(
                        setting="MDS",
                        model="models/test",
                    ),
                    outdir=outdir,
                    alignments=[{"unique_id": "u1"}],
                    used_demos=[],
                    results=results,
                )

        save_results.assert_called_once_with(
            str(outdir),
            [],
            results,
            None,
        )

    def test_facade_uses_explicit_ports_and_preserves_public_signatures(self):
        from attribute_first.ports import ArtifactStore, DialogueGateway

        instance_dependencies = dialogue._instance_dependencies()
        pipeline_dependencies = dialogue._pipeline_dependencies()

        self.assertIsInstance(
            instance_dependencies.dialogue_gateway,
            DialogueGateway,
        )
        self.assertFalse(hasattr(instance_dependencies, "send_chat_message"))
        self.assertIsInstance(
            pipeline_dependencies.artifact_store,
            ArtifactStore,
        )
        self.assertFalse(hasattr(pipeline_dependencies, "write_json"))
        self.assertEqual(
            str(inspect.signature(dialogue.run_instance)),
            "(inst, cs_prompt, clustering_instr, fusion_instr, "
            "model, num_retries=3)",
        )
        self.assertEqual(str(inspect.signature(dialogue.main)), "(args)")

    def test_worker_exception_keeps_structured_protocol_trace(self):
        from attribute_first.application.sequential_dialogue import (
            SequentialDialoguePipelineRunner,
            SequentialPipelineDependencies,
        )

        def fail_instance(*_args, **_kwargs):
            raise RuntimeError("fixture failure")

        runner = SequentialDialoguePipelineRunner(
            SequentialPipelineDependencies(
                get_data=lambda _args: ({}, []),
                get_prompt_structures=lambda *_args, **_kwargs: {},
                construct_prompts=lambda *_args, **_kwargs: ([], {}, {}, {}),
                get_token_counter=lambda *_args, **_kwargs: {},
                reset_token_usage=lambda: None,
                get_token_usage=lambda: {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "calls": 0,
                },
                run_instance=fail_instance,
                save_results=lambda *_args, **_kwargs: None,
                get_environment_flags=lambda: {},
                artifact_store=dialogue._legacy_artifact_store(),
                build_pipeline_results=lambda *_args, **_kwargs: [],
            )
        )

        results = runner._execute(
            args=SimpleNamespace(concurrency=1, model="models/test"),
            alignments=[{"unique_id": "u1"}],
            items=[("u1", "CS")],
            gold={"u1": "Gold."},
            clustering_instruction="Cluster",
            fusion_instruction="Fuse",
        )

        trace = results["u1"]["protocol_trace"]
        self.assertEqual(trace["content_selection"], [])
        self.assertEqual(trace["clustering"], [])
        self.assertEqual(trace["fusion"], [])
        self.assertEqual(
            trace["runtime_error"],
            "RuntimeError: fixture failure",
        )

    def test_worker_programming_error_aborts_without_terminal_model_row(self):
        from attribute_first.application.sequential_dialogue import (
            SequentialDialoguePipelineRunner,
            SequentialPipelineDependencies,
        )

        runner = SequentialDialoguePipelineRunner(
            SequentialPipelineDependencies(
                get_data=lambda _args: ({}, []),
                get_prompt_structures=lambda *_args, **_kwargs: {},
                construct_prompts=lambda *_args, **_kwargs: ([], {}, {}, {}),
                get_token_counter=lambda *_args, **_kwargs: {},
                reset_token_usage=lambda: None,
                get_token_usage=lambda: {},
                run_instance=mock.Mock(
                    side_effect=KeyError("programming defect")
                ),
                save_results=lambda *_args, **_kwargs: None,
                get_environment_flags=lambda: {},
                artifact_store=dialogue._legacy_artifact_store(),
                build_pipeline_results=lambda *_args, **_kwargs: [],
            )
        )

        with self.assertRaisesRegex(KeyError, "programming defect"):
            runner._execute(
                args=SimpleNamespace(
                    concurrency=1,
                    model="models/test",
                ),
                alignments=[{"unique_id": "u1"}],
                items=[("u1", "CS")],
                gold={"u1": "Gold."},
                clustering_instruction="Cluster",
                fusion_instruction="Fuse",
            )

    def test_run_instance_preserves_source_span_metadata_for_evaluation(self):
        source_highlight = {
            "documentFile": "doc-a",
            "docSpanText": "Alpha source",
            "docSpanOffsets": [[0, 12]],
            "docSentCharIdx": 0,
            "docSentText": "Alpha source.",
            "sent_idx": 0,
        }
        instance = {
            "unique_id": "u1",
            "documents": [
                {
                    "documentFile": "doc-a",
                    "rawDocumentText": "Alpha source. More context.",
                    "documentText": ["Alpha source.", " More context."],
                    "docSentCharIdxToSentIdx": [0, 13],
                }
            ],
            "set_of_highlights_in_context": [source_highlight],
        }
        responses = iter(
            (
                json.dumps(
                    {
                        "highlights": [
                            {"doc_id": "1", "span_text": "Alpha source"}
                        ]
                    }
                ),
                json.dumps({"clusters": [[1]]}),
                json.dumps(
                    {
                        "sentence_text": "Alpha summary.",
                        "highlight_ids": [1],
                    }
                ),
            )
        )

        with mock.patch.object(
            dialogue, "create_chat_session", return_value=_ChatSession()
        ), mock.patch.object(
            dialogue, "gemini_chat_call", side_effect=lambda *args, **kwargs: next(responses)
        ), mock.patch.object(
            dialogue,
            "get_last_call_usage",
            return_value={
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "cached_content_token_count": 0,
            },
        ), mock.patch.object(
            dialogue,
            "get_last_call_metadata",
            return_value={
                "provider_response_received": True,
                "model_version": "fixture",
                "finish_reason": "STOP",
                "prompt_block_reason": None,
            },
        ), mock.patch.object(
            dialogue,
            "get_token_usage",
            return_value={"prompt": 0, "completion": 0, "cached": 0, "calls": 0},
        ):
            result, _ = dialogue.run_instance(
                instance,
                "CS prompt",
                "cluster",
                "fuse",
                "models/test",
                num_retries=1,
            )

        persisted_span = result["alignments"][0]["highlight_spans"][0]
        self.assertEqual(persisted_span, source_highlight)

    def test_main_persists_evaluable_pipeline_protocol_and_token_usage(self):
        source_highlight = {
            "documentFile": "doc-a",
            "docSpanText": "Alpha source",
            "docSpanOffsets": [[0, 12]],
            "docSentCharIdx": 0,
            "docSentText": "Alpha source.",
            "sent_idx": 0,
        }
        source_instances = [
            {
                "unique_id": "u1",
                "topic": "topic-one",
                "query": "What happened?",
                "documents": [
                    {
                        "documentFile": "doc-a",
                        "rawDocumentText": "Alpha source.",
                        "source_title": "Source A",
                    }
                ],
                "response": "Gold one.",
                "set_of_highlights_in_context": [source_highlight],
                "source_metadata": {"collection": "fixture"},
            },
            {
                "unique_id": "u2",
                "topic": "topic-two",
                "query": "Why?",
                "documents": [
                    {
                        "documentFile": "doc-b",
                        "rawDocumentText": "Beta source.",
                        "source_title": "Source B",
                    }
                ],
                "response": "Gold two.",
                "set_of_highlights_in_context": [],
                "source_metadata": {"collection": "fixture"},
            },
        ]
        generated = {
            "u1": {
                "final_output": "Alpha summary.",
                "alignments": [
                    {
                        "sent_id": 1,
                        "sent_text": "Alpha summary.",
                        "highlights": [1],
                        "highlight_spans": [source_highlight],
                    }
                ],
                "protocol_trace": {"content_selection_raw": "raw"},
            },
            "u2": {
                "final_output": "ERROR - no CS spans",
                "alignments": [],
                "protocol_trace": {"content_selection_raw": "empty"},
            },
        }
        usage = {"prompt": 101, "completion": 17, "cached": 23, "calls": 6}
        environment_flags = {
            "AF_CONTEXT_CACHE": False,
            "AF_DIALOGUE_NO_DEMOS": False,
            "AF_DOCS_FIRST": False,
            "AF_MARK_CONTEXT": False,
            "AF_USE_ROLES": True,
        }

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "out"
            args = SimpleNamespace(
                setting="MDS",
                split="test",
                indir_alignments="source.jsonl",
                indir_prompt="prompts.json",
                outdir=str(outdir),
                model="models/test",
                n_demos=3,
                concurrency=1,
                max_examples=None,
                num_retries=4,
                seed=20260728,
            )

            def fake_run_instance(instance, *args, **kwargs):
                return generated[instance["unique_id"]], {
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "calls": 0,
                }

            with mock.patch.object(
                dialogue,
                "get_data",
                return_value=(
                    {
                        "instruction-clustering": "cluster {HS} {HE}",
                        "instruction-next-cluster-fusion": "fuse {HS} {HE}",
                    },
                    source_instances,
                ),
            ), mock.patch.object(
                dialogue, "get_subtask_prompt_structures", return_value={}
            ), mock.patch.object(
                dialogue,
                "construct_prompts",
                return_value=(
                    [{"unique_id": "demo-1"}],
                    {"u1": "CS one", "u2": "CS two"},
                    {},
                    {},
                ),
            ), mock.patch.object(
                dialogue, "get_token_counter", return_value={}
            ), mock.patch.object(
                dialogue,
                "run_instance",
                side_effect=fake_run_instance,
            ) as run_instance_mock, mock.patch.object(
                dialogue, "reset_token_usage"
            ), mock.patch.object(
                dialogue, "get_token_usage", return_value=dict(usage)
            ), mock.patch.object(
                dialogue.utils,
                "get_af_environment_flags",
                return_value=environment_flags,
            ):
                dialogue.main(args)

            self.assertEqual(run_instance_mock.call_count, 2)
            self.assertTrue(
                all(
                    call.kwargs == {"num_retries": 4}
                    for call in run_instance_mock.call_args_list
                )
            )
            with open(outdir / "pipeline_format_results.json", encoding="utf-8") as stream:
                pipeline_rows = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual([row["unique_id"] for row in pipeline_rows], ["u1", "u2"])

            successful = pipeline_rows[0]
            self.assertEqual(successful["topic"], "topic-one")
            self.assertEqual(successful["query"], "What happened?")
            self.assertEqual(successful["documents"], source_instances[0]["documents"])
            self.assertEqual(
                successful["source_metadata"],
                source_instances[0]["source_metadata"],
            )
            self.assertEqual(successful["response"], "Alpha summary.")
            self.assertEqual(
                successful["set_of_highlights_in_context"][0],
                {
                    **source_highlight,
                    "scuSentCharIdx": 0,
                    "scuSentence": "Alpha summary.",
                },
            )

            failed = pipeline_rows[1]
            self.assertEqual(failed["unique_id"], "u2")
            self.assertEqual(failed["topic"], "topic-two")
            self.assertEqual(failed["documents"], source_instances[1]["documents"])
            self.assertEqual(failed["response"], "ERROR - no CS spans")
            self.assertEqual(failed["set_of_highlights_in_context"], [])
            self.assertEqual(failed["skipped_reason"], "model_error")

            saved_args = json.loads(
                (outdir / "args.json").read_text(encoding="utf-8")
            )
            self.assertEqual(saved_args["setting"], "MDS")
            self.assertEqual(saved_args["split"], "test")
            self.assertEqual(saved_args["model"], "models/test")
            self.assertEqual(saved_args["n_demos"], 3)
            self.assertEqual(saved_args["num_retries"], 4)
            self.assertEqual(saved_args["seed"], 20260728)
            self.assertEqual(saved_args["environment_flags"], environment_flags)
            self.assertEqual(
                saved_args["protocol"]["turns"],
                ["content_selection", "clustering", "sentence_fusion_per_cluster"],
            )
            self.assertEqual(
                saved_args["protocol"]["fusion_history_policy"],
                "rollback_after_each_cluster",
            )

            saved_usage = json.loads(
                (outdir / "token_usage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                saved_usage,
                {
                    **usage,
                    "subtask": "dialogue_sequential",
                    "model": "models/test",
                },
            )

    def test_main_refuses_to_overwrite_previous_results(self):
        source = {
            "unique_id": "u1",
            "documents": [],
            "response": "Gold.",
            "set_of_highlights_in_context": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "out"
            outdir.mkdir()
            results_path = outdir / "results.json"
            previous_bytes = b'{"previous": true}\n'
            results_path.write_bytes(previous_bytes)
            args = SimpleNamespace(
                setting="MDS",
                split="test",
                indir_alignments=None,
                indir_prompt=None,
                outdir=str(outdir),
                model="models/test",
                n_demos=1,
                concurrency=1,
                max_examples=None,
                num_retries=3,
                seed=20260728,
            )
            run_instance = mock.Mock()

            with mock.patch.object(
                dialogue,
                "get_data",
                return_value=(
                    {
                        "instruction-clustering": "cluster",
                        "instruction-next-cluster-fusion": "fuse",
                    },
                    [source],
                ),
            ), mock.patch.object(
                dialogue, "get_subtask_prompt_structures", return_value={}
            ), mock.patch.object(
                dialogue,
                "construct_prompts",
                return_value=([], {"u1": "CS"}, {}, {}),
            ), mock.patch.object(
                dialogue, "get_token_counter", return_value={}
            ), mock.patch.object(
                dialogue,
                "run_instance",
                run_instance,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "new or empty|non-empty",
                ):
                    dialogue.main(args)

            run_instance.assert_not_called()
            self.assertEqual(results_path.read_bytes(), previous_bytes)


if __name__ == "__main__":
    unittest.main()
