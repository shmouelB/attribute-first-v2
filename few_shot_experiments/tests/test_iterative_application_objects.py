import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from attribute_first.application.iterative_sentence_generation import (  # noqa: E402
    IterativeApplicationDependencies,
    IterativeExecutionDependencies,
    IterativeGenerationExecutor,
    IterativePersistenceDependencies,
    IterativePromptBuilder,
    IterativePromptDependencies,
    IterativeResultPersister,
    IterativeRunContext,
    IterativeRunContextFactory,
    IterativeSentenceGenerationApplication,
    parse_iterative_sentence_response,
)
from attribute_first.runtime import (  # noqa: E402
    AttemptDependencies,
    AttemptExecutor,
    AttemptPolicy,
)
from attribute_first.application.iterative_application import (  # noqa: E402
    IterativeApplicationDependencies as ExtractedApplicationDependencies,
    IterativeRunContext as ExtractedRunContext,
    IterativeRunContextFactory as ExtractedRunContextFactory,
    IterativeSentenceGenerationApplication as ExtractedApplication,
)


class _DeterministicRng:
    def choice(self, population, size, replace=False):
        del population, replace
        return list(range(size))


class _RecordingRng:
    def __init__(self):
        self.calls = []

    def choice(self, population, size, replace=False):
        self.calls.append((population, size, replace))
        return list(range(size))


class IterativeApplicationModuleBoundaryTests(unittest.TestCase):
    def test_legacy_module_reexports_the_extracted_application_types(self):
        self.assertIs(
            IterativeApplicationDependencies,
            ExtractedApplicationDependencies,
        )
        self.assertIs(IterativeRunContext, ExtractedRunContext)
        self.assertIs(
            IterativeRunContextFactory,
            ExtractedRunContextFactory,
        )
        self.assertIs(
            IterativeSentenceGenerationApplication,
            ExtractedApplication,
        )

    def test_iterative_modules_stay_below_the_repository_size_limit(self):
        application_root = (
            EXPERIMENT_ROOT / "attribute_first" / "application"
        )
        for name in (
            "iterative_application.py",
            "iterative_sentence_generation.py",
        ):
            with self.subTest(name=name):
                line_count = len(
                    (application_root / name)
                    .read_text(encoding="utf-8")
                    .splitlines()
                )
                self.assertLess(line_count, 1000)


class IterativePromptBuilderTests(unittest.TestCase):
    def test_zero_shot_prompt_resolves_protocol_without_a_demo(self):
        builder = IterativePromptBuilder(
            IterativePromptDependencies(
                extract_highlights=mock.Mock(),
                get_highlighted_doc=mock.Mock(),
                make_demo=mock.Mock(),
                remove_after_last_highlight=mock.Mock(),
            )
        )
        prompt_dict = {
            "instruction-next-cluster-fusion": "base instruction",
            "demo_prompt_next_cluster_fusion": "base template",
            "instruction-next-cluster-fusion-with-question": (
                "question instruction"
            ),
            "demo_prompt_next_cluster_fusion_with_question": (
                "question template"
            ),
            "highlight_start_tkn": "<HS>",
            "highlight_end_tkn": "<HE>",
        }
        token_counter = SimpleNamespace(
            token_count=lambda _prompt: 1
        )

        with mock.patch.object(
            builder,
            "construct_non_demo_part",
            return_value=("target prompt", []),
        ) as construct_target:
            prompt, evidence = builder.construct_prompt(
                prompt_dict=prompt_dict,
                alignments=[],
                curr_docs={},
                used_demos=[],
                curr_cluster_ind=0,
                prefix="",
                merge_cross_sents_highlights=False,
                tkn_counter={
                    "tkn_counter": token_counter,
                    "tkn_max_limit": 100,
                },
                always_with_question=True,
                instance_question="What happened?",
            )

        self.assertEqual(prompt, "target prompt")
        self.assertEqual(evidence["highlighted_docs"], [])
        target_args = construct_target.call_args.args
        self.assertEqual(target_args[7], "question instruction")
        self.assertEqual(target_args[8], "question template")


class IterativeGenerationExecutorTests(unittest.TestCase):
    @staticmethod
    def _instance(sentence_indices):
        return {
            "unique_id": "example-1",
            "documents": [
                {
                    "documentFile": "doc-1",
                    "rawDocumentText": "Alpha source.",
                }
            ],
            "set_of_highlights_in_context": [
                {
                    "documentFile": "doc-1",
                    "scuSentCharIdx": sentence_index,
                    "scuSentence": f"Sentence {sentence_index}.",
                    "docSpanText": "Alpha",
                }
                for sentence_index in sentence_indices
            ],
        }

    def test_successful_sentence_is_appended_with_its_generation_evidence(self):
        model_calls = []

        def construct_prompt(**kwargs):
            return (
                "current prompt",
                {
                    "curr_alignments": kwargs["alignments"],
                    "curr_prefix": kwargs["prefix"],
                },
            )

        def prompt_model(**kwargs):
            model_calls.append(kwargs)
            unique_id = next(iter(kwargs["prompts"]))
            return {
                unique_id: {
                    "final_output": "Generated sentence.",
                    "full_model_response": "Generated sentence.",
                    "attempt_trace": [{"attempt": 1, "status": "parsed"}],
                }
            }

        executor = IterativeGenerationExecutor(
            IterativeExecutionDependencies(
                construct_prompt=construct_prompt,
                prompt_model=prompt_model,
                parse_response=lambda response, _prompt: {
                    "final_output": response
                },
                progress=lambda rows: rows,
                log_info=lambda _message: None,
            )
        )
        instance = self._instance([0])

        results = executor.generate(
            alignments_dict=[instance],
            prompt_dict={"demos": [{"id": "demo-1"}]},
            used_demos=[{"id": "demo-1"}],
            model_name="models/test",
            num_retries=2,
            debugging=False,
            n_demos=1,
            num_demo_changes=0,
            temperature=0.0,
            merge_cross_sents_highlights=False,
            tkn_counter={
                "tkn_counter": object(),
                "tkn_max_limit": 1000,
            },
            rng=_DeterministicRng(),
        )

        self.assertEqual(
            results["example-1"]["generated_summary_sents"],
            ["Generated sentence."],
        )
        history = results["example-1"]["generation_history"]
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["curr_prefix"], "")
        self.assertEqual(history[0]["updated_demos"], [])
        self.assertEqual(model_calls[0]["reset_usage"], False)

    def test_clusters_are_sorted_and_successes_form_the_next_prefix(self):
        prompt_calls = []

        def construct_prompt(**kwargs):
            sentence_index = kwargs["alignments"][0]["scuSentCharIdx"]
            prompt_calls.append(
                (
                    kwargs["curr_cluster_ind"],
                    sentence_index,
                    kwargs["prefix"],
                )
            )
            return (
                f"cluster-{sentence_index}",
                {"curr_alignments": kwargs["alignments"]},
            )

        def prompt_model(**kwargs):
            unique_id = next(iter(kwargs["prompts"]))
            sentence_index = kwargs["prompts"][unique_id].split("-")[1]
            output = f"Generated {sentence_index}."
            return {
                unique_id: {
                    "final_output": output,
                    "full_model_response": output,
                }
            }

        executor = IterativeGenerationExecutor(
            IterativeExecutionDependencies(
                construct_prompt=construct_prompt,
                prompt_model=prompt_model,
                parse_response=parse_iterative_sentence_response,
                progress=lambda rows: rows,
                log_info=lambda _message: None,
            )
        )
        results = executor.generate(
            alignments_dict=[self._instance([9, 0, 9])],
            prompt_dict={"demos": [{"id": "demo-1"}]},
            used_demos=[{"id": "demo-1"}],
            model_name="models/test",
            num_retries=1,
            debugging=False,
            n_demos=1,
            num_demo_changes=0,
            temperature=0.0,
            merge_cross_sents_highlights=False,
            tkn_counter={
                "tkn_counter": object(),
                "tkn_max_limit": 1000,
            },
            rng=_DeterministicRng(),
        )

        self.assertEqual(
            prompt_calls,
            [
                (0, 0, ""),
                (1, 9, "Generated 0."),
            ],
        )
        self.assertEqual(
            results["example-1"]["generated_summary_sents"],
            ["Generated 0.", "Generated 9."],
        )

    def test_terminal_error_advances_rng_and_stops_later_clusters(self):
        model_calls = []
        rng = _RecordingRng()

        def construct_prompt(**kwargs):
            return (
                f"cluster-{kwargs['curr_cluster_ind']}",
                {"curr_alignments": kwargs["alignments"]},
            )

        def prompt_model(**kwargs):
            model_calls.append(kwargs)
            unique_id = next(iter(kwargs["prompts"]))
            return {
                unique_id: {
                    "final_output": "ERROR - exhausted",
                    "full_model_response": "ERROR - exhausted",
                }
            }

        executor = IterativeGenerationExecutor(
            IterativeExecutionDependencies(
                construct_prompt=construct_prompt,
                prompt_model=prompt_model,
                parse_response=parse_iterative_sentence_response,
                progress=lambda rows: rows,
                log_info=lambda _message: None,
            )
        )
        results = executor.generate(
            alignments_dict=[self._instance([0, 20])],
            prompt_dict={
                "demos": [{"id": "demo-1"}, {"id": "demo-2"}]
            },
            used_demos=[{"id": "demo-1"}],
            model_name="models/test",
            num_retries=1,
            debugging=False,
            n_demos=1,
            num_demo_changes=1,
            temperature=0.0,
            merge_cross_sents_highlights=False,
            tkn_counter={
                "tkn_counter": object(),
                "tkn_max_limit": 1000,
            },
            rng=rng,
        )

        self.assertEqual(len(model_calls), 2)
        self.assertEqual(rng.calls, [(2, 1, False)])
        result = results["example-1"]
        self.assertEqual(
            result["generated_summary_sents"],
            ["ERROR - exhausted"],
        )
        self.assertEqual(len(result["generation_history"]), 1)
        self.assertEqual(
            result["generation_history"][0]["updated_demos"],
            [],
        )

    def test_parser_accepts_attempt_executor_keyword_protocol(self):
        self.assertEqual(
            parse_iterative_sentence_response(
                response="Answer: Generated sentence.",
                prompt="ignored prompt",
            )["final_output"],
            "Generated sentence.",
        )

    def test_empty_sentence_responses_retry_then_end_as_error(self):
        invalid_responses = ("", "   ", "Answer:", "Answer: \n")
        for response in invalid_responses:
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    parse_iterative_sentence_response(
                        response=response,
                        prompt="ignored prompt",
                    )

        responses = iter(("", "Answer:"))
        sleep = mock.Mock()
        executor = AttemptExecutor(
            AttemptDependencies(
                invoke=lambda: next(responses),
                parse=parse_iterative_sentence_response,
                reset_evidence=lambda: None,
                last_usage=lambda: None,
                last_metadata=lambda: {"finish_reason": "STOP"},
                ensure_parseable=lambda _metadata: None,
                fingerprint=lambda _value: "unused",
                sleep=sleep,
            )
        )

        result = executor.execute(
            prompt="Generate the next sentence.",
            application_request={"transport": "offline-test"},
            policy=AttemptPolicy(
                model_name="models/test",
                output_max_length=128,
                num_retries=2,
                temperature=0.0,
            ),
        )

        self.assertTrue(result["final_output"].startswith("ERROR -"))
        self.assertEqual(len(result["attempt_trace"]), 2)
        self.assertTrue(
            all(
                attempt["failure_phase"] == "parse"
                for attempt in result["attempt_trace"]
            )
        )
        sleep.assert_called_once_with(1)

    def test_generation_preserves_every_input_uid_without_hidden_exclusions(
        self,
    ):
        input_ids = [
            "test59",
            "test62",
            "test63",
            "test67",
            "test91",
            "ordinary-id",
        ]
        instances = []
        for unique_id in input_ids:
            instance = self._instance([0])
            instance["unique_id"] = unique_id
            instances.append(instance)

        def prompt_model(**kwargs):
            unique_id = next(iter(kwargs["prompts"]))
            output = (
                "ERROR - exhausted"
                if unique_id == "test62"
                else f"Generated {unique_id}."
            )
            return {
                unique_id: {
                    "final_output": output,
                    "full_model_response": output,
                }
            }

        executor = IterativeGenerationExecutor(
            IterativeExecutionDependencies(
                construct_prompt=lambda **_kwargs: (
                    "prompt",
                    {"curr_alignments": []},
                ),
                prompt_model=prompt_model,
                parse_response=parse_iterative_sentence_response,
                progress=lambda rows: rows,
                log_info=lambda _message: None,
            )
        )

        results = executor.generate(
            alignments_dict=instances,
            prompt_dict={"demos": [{"id": "demo-1"}]},
            used_demos=[{"id": "demo-1"}],
            model_name="models/test",
            num_retries=1,
            debugging=False,
            n_demos=1,
            num_demo_changes=0,
            temperature=0.0,
            merge_cross_sents_highlights=False,
            tkn_counter={
                "tkn_counter": object(),
                "tkn_max_limit": 1000,
            },
            rng=_DeterministicRng(),
        )

        self.assertEqual(list(results), input_ids)
        self.assertEqual(len(results), len(instances))
        self.assertEqual(
            results["test62"]["generated_summary_sents"],
            ["ERROR - exhausted"],
        )


class IterativeResultPersisterTests(unittest.TestCase):
    def test_persist_writes_results_and_annotated_aggregate_usage(self):
        saved = []
        written = []
        dependencies = IterativePersistenceDependencies(
            save_results=lambda *args: saved.append(args),
            get_token_usage=lambda: {
                "prompt": 24,
                "completion": 5,
                "cached": 12,
                "calls": 2,
            },
            write_json=lambda path, value: written.append((path, value)),
        )
        persister = IterativeResultPersister(dependencies)

        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "run"
            persister.persist(
                outdir=str(outdir),
                used_demos=[{"id": "demo-1"}],
                final_results={
                    "example-1": {
                        "final_output": "Generated sentence."
                    }
                },
                pipeline_format_results=[],
                model_name="models/test",
            )

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0][0], str(outdir))
        self.assertEqual(
            written,
            [
                (
                    str(outdir / "token_usage.json"),
                    {
                        "prompt": 24,
                        "completion": 5,
                        "cached": 12,
                        "calls": 2,
                        "subtask": "iterative_sentence_generation",
                        "model": "models/test",
                    },
                )
            ],
        )


class IterativeSentenceGenerationApplicationTests(unittest.TestCase):
    @staticmethod
    def _write_json(path, value):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(value, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _args(*, outdir, rerun=False, rerun_path=None):
        return SimpleNamespace(
            config_file=None,
            setting="MDS",
            split="test",
            outdir=str(outdir) if outdir is not None else None,
            rerun=rerun,
            rerun_path=(
                str(rerun_path) if rerun_path is not None else None
            ),
            rerun_n_demos=None,
            rerun_temperature=None,
            no_prefix=False,
            merge_cross_sents_highlights=False,
            n_demos=1,
            model_name="models/test",
            debugging=False,
            num_retries=1,
            num_demo_changes=0,
            temperature=0.0,
            always_with_question=False,
            cut_surplus=False,
            seed=20260728,
            prompt_token_budget=1000,
        )

    def _application(
        self,
        *,
        alignments,
        generate,
        convert_results,
        saved_runs,
        get_data=None,
        artifact_sha256=None,
        get_token_counter=None,
    ):
        def save_results(
            outdir,
            used_demos,
            final_results,
            pipeline_format_results,
        ):
            saved_runs.append(str(outdir))
            self._write_json(
                Path(outdir) / "used_demonstrations.json",
                used_demos,
            )
            self._write_json(
                Path(outdir) / "results.json",
                final_results,
            )
            if pipeline_format_results is not None:
                self._write_json(
                    Path(outdir) / "pipeline_format_results.json",
                    pipeline_format_results,
                )

        persister = IterativeResultPersister(
            IterativePersistenceDependencies(
                save_results=save_results,
                get_token_usage=lambda: {},
                write_json=self._write_json,
            )
        )
        return IterativeSentenceGenerationApplication(
            IterativeApplicationDependencies(
                update_args=lambda args: args,
                get_data=(
                    get_data
                    or (
                        lambda _args: (
                            {"demos": [{"id": "demo-1"}]},
                            alignments,
                        )
                    )
                ),
                generate=generate,
                get_token_counter=(
                    get_token_counter
                    or (
                        lambda *_args: {
                            "tkn_counter": object(),
                            "tkn_max_limit": 1000,
                        }
                    )
                ),
                reset_token_usage=lambda: None,
                convert_results=convert_results,
                persister=persister,
                rng_factory=lambda _seed: _DeterministicRng(),
                log_info=lambda _message: None,
                artifact_sha256=(
                    artifact_sha256
                    or (
                        lambda path: hashlib.sha256(
                            Path(path).read_bytes()
                        ).hexdigest()
                    )
                ),
            )
        )

    def _write_rerun_parent(self, root, results):
        parent = Path(root) / "parent"
        self._write_json(
            parent / "args.json",
            {
                "model_name": "models/original",
                "n_demos": 1,
                "debugging": False,
                "num_retries": 1,
                "num_demo_changes": 0,
                "temperature": 0.7,
                "merge_cross_sents_highlights": False,
                "always_with_question": False,
                "no_prefix": False,
            },
        )
        self._write_json(
            parent / "used_demonstrations.json",
            [{"id": "demo-1"}],
        )
        self._write_json(parent / "results.json", results)
        return parent

    def test_rerun_is_append_only_and_records_parent_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = root / "parent"
            derived = root / "derived"
            self._write_json(
                parent / "args.json",
                {
                    "model_name": "models/original",
                    "n_demos": 1,
                    "debugging": False,
                    "num_retries": 1,
                    "num_demo_changes": 0,
                    "temperature": 0.7,
                    "cut_surplus": True,
                    "prompt_token_budget": 777,
                    "merge_cross_sents_highlights": False,
                    "always_with_question": False,
                    "no_prefix": False,
                },
            )
            self._write_json(
                parent / "used_demonstrations.json",
                [{"id": "demo-1"}],
            )
            self._write_json(
                parent / "results.json",
                {
                    "retry": {
                        "generated_summary_sents": [
                            "ERROR - exhausted"
                        ]
                    },
                    "complete": {
                        "generated_summary_sents": ["Already complete."]
                    },
                },
            )
            parent_before = {
                path.name: path.read_bytes()
                for path in parent.iterdir()
            }
            saved_runs = []
            generated = mock.Mock(
                return_value={
                    "retry": {
                        "generated_summary_sents": ["Fixed."],
                        "generation_history": [],
                    }
                }
            )
            get_token_counter = mock.Mock(
                return_value={
                    "tkn_counter": object(),
                    "tkn_max_limit": 777,
                }
            )
            application = self._application(
                alignments=[
                    {"unique_id": "retry", "response": "Gold retry."},
                    {
                        "unique_id": "complete",
                        "response": "Gold complete.",
                    },
                ],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=saved_runs,
                get_token_counter=get_token_counter,
            )

            rerun_args = self._args(
                outdir=derived,
                rerun=True,
                rerun_path=parent,
            )
            rerun_args.rerun_n_demos = 0
            rerun_args.rerun_temperature = 0.0
            application.run(rerun_args)

            parent_after = {
                path.name: path.read_bytes()
                for path in parent.iterdir()
            }
            self.assertEqual(parent_after, parent_before)
            self.assertEqual(saved_runs, [str(derived.resolve())])
            provenance = json.loads(
                (derived / "rerun_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            parent_results = parent / "results.json"
            self.assertEqual(
                provenance["parent"]["results_path"],
                str(parent_results.resolve()),
            )
            self.assertEqual(
                provenance["parent"]["sha256"],
                hashlib.sha256(parent_results.read_bytes()).hexdigest(),
            )
            self.assertEqual(provenance["retried_ids"], ["retry"])
            self.assertTrue((derived / "args.json").is_file())
            self.assertEqual(generated.call_args.kwargs["n_demos"], 0)
            self.assertEqual(
                generated.call_args.kwargs["used_demos"],
                [],
            )
            self.assertEqual(
                generated.call_args.kwargs["temperature"],
                0.0,
            )
            self.assertIs(
                generated.call_args.kwargs["cut_surplus"],
                True,
            )
            get_token_counter.assert_called_once_with(
                "models/original",
                777,
            )
            args_snapshot = json.loads(
                (derived / "args.json").read_text(encoding="utf-8")
            )
            self.assertEqual(args_snapshot["n_demos"], 0)
            self.assertEqual(args_snapshot["temperature"], 0.0)
            self.assertIs(args_snapshot["cut_surplus"], True)
            self.assertEqual(args_snapshot["prompt_token_budget"], 777)
            self.assertEqual(
                json.loads(
                    (
                        derived / "used_demonstrations.json"
                    ).read_text(encoding="utf-8")
                ),
                [],
            )

    def test_rerun_demo_override_resamples_exactly_and_deterministically(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._write_rerun_parent(
                root,
                {
                    "retry": {
                        "generated_summary_sents": [
                            "ERROR - exhausted"
                        ]
                    }
                },
            )
            derived = root / "derived"
            demos = [
                {"id": "demo-0"},
                {"id": "demo-1"},
                {"id": "demo-2"},
            ]
            generated = mock.Mock(
                return_value={
                    "retry": {
                        "generated_summary_sents": ["Fixed."],
                        "generation_history": [],
                    }
                }
            )
            application = self._application(
                alignments=[
                    {"unique_id": "retry", "response": "Gold."}
                ],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=[],
                get_data=lambda _args: (
                    {"demos": demos},
                    [{"unique_id": "retry", "response": "Gold."}],
                ),
            )
            args = self._args(
                outdir=derived,
                rerun=True,
                rerun_path=parent,
            )
            args.rerun_n_demos = 2

            application.run(args)

            expected = demos[:2]
            self.assertEqual(
                generated.call_args.kwargs["used_demos"],
                expected,
            )
            self.assertEqual(
                json.loads(
                    (
                        derived / "used_demonstrations.json"
                    ).read_text(encoding="utf-8")
                ),
                expected,
            )

    def test_impossible_rerun_demo_override_fails_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._write_rerun_parent(
                root,
                {
                    "retry": {
                        "generated_summary_sents": ["ERROR - retry"]
                    }
                },
            )
            derived = root / "unclaimed"
            generated = mock.Mock()
            application = self._application(
                alignments=[
                    {"unique_id": "retry", "response": "Gold."}
                ],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=[],
            )
            args = self._args(
                outdir=derived,
                rerun=True,
                rerun_path=parent,
            )
            args.rerun_n_demos = 2

            with self.assertRaisesRegex(
                ValueError,
                "not enough demonstrations",
            ):
                application.run(args)

            self.assertFalse(derived.exists())
            generated.assert_not_called()

    def test_missing_rerun_error_uid_fails_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._write_rerun_parent(
                root,
                {
                    "missing": {
                        "generated_summary_sents": ["ERROR - retry"]
                    }
                },
            )
            derived = root / "unclaimed"
            generated = mock.Mock()
            application = self._application(
                alignments=[
                    {"unique_id": "present", "response": "Gold."}
                ],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=[],
            )

            with self.assertRaisesRegex(
                ValueError,
                "ERROR parent UIDs.*missing",
            ):
                application.run(
                    self._args(
                        outdir=derived,
                        rerun=True,
                        rerun_path=parent,
                    )
                )

            self.assertFalse(derived.exists())
            generated.assert_not_called()

    def test_changed_rerun_parent_fails_before_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._write_rerun_parent(
                root,
                {
                    "retry": {
                        "generated_summary_sents": ["ERROR - retry"]
                    }
                },
            )
            derived = root / "unclaimed"
            observed_hashes = iter(("before", "after"))
            generated = mock.Mock()
            application = self._application(
                alignments=[
                    {"unique_id": "retry", "response": "Gold."}
                ],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=[],
                artifact_sha256=lambda _path: next(observed_hashes),
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "parent changed",
            ):
                application.run(
                    self._args(
                        outdir=derived,
                        rerun=True,
                        rerun_path=parent,
                    )
                )

            self.assertFalse(derived.exists())
            generated.assert_not_called()

    def test_invalid_fresh_context_does_not_claim_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "unclaimed"
            application = self._application(
                alignments=[],
                generate=mock.Mock(),
                convert_results=lambda *_args: [],
                saved_runs=[],
                get_data=lambda _args: ({"demos": "invalid"}, []),
            )

            with self.assertRaisesRegex(
                ValueError,
                "demos list",
            ):
                application.run(self._args(outdir=outdir))

            self.assertFalse(outdir.exists())

    def test_invalid_fresh_controls_fail_before_claim_or_generation(self):
        invalid_controls = (
            ("num_retries", 0, "num_retries"),
            ("num_retries", True, "num_retries"),
            ("num_demo_changes", -1, "num_demo_changes"),
            ("num_demo_changes", True, "num_demo_changes"),
            ("temperature", True, "temperature"),
            ("temperature", "warm", "temperature"),
            ("temperature", 3, "temperature"),
            ("seed", True, "seed"),
            ("seed", 1.5, "seed"),
            ("prompt_token_budget", 0, "prompt_token_budget"),
            ("prompt_token_budget", True, "prompt_token_budget"),
        )
        for field, value, error_pattern in invalid_controls:
            with self.subTest(field=field, value=value):
                with tempfile.TemporaryDirectory() as tmp:
                    outdir = Path(tmp) / "unclaimed"
                    generated = mock.Mock()
                    data_loader = mock.Mock(
                        return_value=(
                            {"demos": [{"id": "demo-1"}]},
                            [
                                {
                                    "unique_id": "example",
                                    "response": "Gold.",
                                }
                            ],
                        )
                    )
                    application = self._application(
                        alignments=[],
                        generate=generated,
                        convert_results=lambda *_args: [],
                        saved_runs=[],
                        get_data=data_loader,
                    )
                    args = self._args(outdir=outdir)
                    setattr(args, field, value)

                    with self.assertRaisesRegex(
                        ValueError,
                        error_pattern,
                    ):
                        application.run(args)

                    self.assertFalse(outdir.exists())
                    generated.assert_not_called()
                    data_loader.assert_not_called()

    def test_rerun_protocol_and_demo_count_fail_before_claim(self):
        invalid_parent_controls = (
            ("num_retries", 0, "num_retries"),
            ("num_demo_changes", -1, "num_demo_changes"),
            ("temperature", True, "temperature"),
        )
        for field, value, error_pattern in invalid_parent_controls:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parent = self._write_rerun_parent(
                    root,
                    {
                        "retry": {
                            "generated_summary_sents": ["ERROR - retry"]
                        }
                    },
                )
                parent_args_path = parent / "args.json"
                parent_args = json.loads(
                    parent_args_path.read_text(encoding="utf-8")
                )
                parent_args[field] = value
                self._write_json(parent_args_path, parent_args)
                outdir = root / "unclaimed"
                generated = mock.Mock()
                application = self._application(
                    alignments=[
                        {"unique_id": "retry", "response": "Gold."}
                    ],
                    generate=generated,
                    convert_results=lambda *_args: [],
                    saved_runs=[],
                )

                with self.assertRaisesRegex(ValueError, error_pattern):
                    application.run(
                        self._args(
                            outdir=outdir,
                            rerun=True,
                            rerun_path=parent,
                        )
                    )

                self.assertFalse(outdir.exists())
                generated.assert_not_called()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent = self._write_rerun_parent(
                root,
                {
                    "retry": {
                        "generated_summary_sents": ["ERROR - retry"]
                    }
                },
            )
            parent_args_path = parent / "args.json"
            parent_args = json.loads(
                parent_args_path.read_text(encoding="utf-8")
            )
            parent_args["n_demos"] = 2
            self._write_json(parent_args_path, parent_args)
            outdir = root / "unclaimed"
            generated = mock.Mock()
            application = self._application(
                alignments=[
                    {"unique_id": "retry", "response": "Gold."}
                ],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=[],
            )

            with self.assertRaisesRegex(
                ValueError,
                "demonstration|n_demos",
            ):
                application.run(
                    self._args(
                        outdir=outdir,
                        rerun=True,
                        rerun_path=parent,
                    )
                )

            self.assertFalse(outdir.exists())
            generated.assert_not_called()

    def test_invalid_rerun_parent_does_not_claim_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outdir = root / "unclaimed"
            application = self._application(
                alignments=[],
                generate=mock.Mock(),
                convert_results=lambda *_args: [],
                saved_runs=[],
            )

            with self.assertRaisesRegex(
                ValueError,
                "rerun path is not a run directory",
            ):
                application.run(
                    self._args(
                        outdir=outdir,
                        rerun=True,
                        rerun_path=root / "missing-parent",
                    )
                )

            self.assertFalse(outdir.exists())

    def test_conversion_failure_is_persisted_but_reported_as_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "run"
            saved_runs = []
            application = self._application(
                alignments=[
                    {"unique_id": "example", "response": "Gold."}
                ],
                generate=lambda **_kwargs: {
                    "example": {
                        "generated_summary_sents": ["Generated."],
                        "generation_history": [],
                    }
                },
                convert_results=mock.Mock(
                    side_effect=ValueError("invalid pipeline conversion")
                ),
                saved_runs=saved_runs,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "pipeline conversion failed",
            ):
                application.run(self._args(outdir=outdir))

            self.assertEqual(saved_runs, [str(outdir.resolve())])
            self.assertTrue((outdir / "results.json").is_file())
            self.assertFalse(
                (outdir / "pipeline_format_results.json").exists()
            )

    def test_non_empty_output_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            outdir = Path(tmp) / "run"
            outdir.mkdir()
            (outdir / "sentinel.txt").write_text(
                "existing run",
                encoding="utf-8",
            )
            generated = mock.Mock()
            application = self._application(
                alignments=[],
                generate=generated,
                convert_results=lambda *_args: [],
                saved_runs=[],
            )

            with self.assertRaisesRegex(
                ValueError,
                "new or empty|non-empty",
            ):
                application.run(self._args(outdir=outdir))

            generated.assert_not_called()


if __name__ == "__main__":
    unittest.main()
