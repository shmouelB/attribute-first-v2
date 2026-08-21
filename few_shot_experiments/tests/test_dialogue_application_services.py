"""Characterization tests for decomposed dialogue application services."""

from collections.abc import Sequence
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


from attribute_first.application.dialogue_persistence import (  # noqa: E402
    DialogueResultPersister,
)
from attribute_first.application.dialogue_content_selection import (  # noqa: E402
    DialogueContentSelectionService,
)
from attribute_first.application.dialogue_fusion import (  # noqa: E402
    DialogueFusionService,
)
from attribute_first.application.dialogue_preparation import (  # noqa: E402
    DialoguePlanBuilder,
)
from attribute_first.application.dialogue_sessions import (  # noqa: E402
    DialogueSessionService,
)
from attribute_first.application.dialogue_shared_content_selection import (  # noqa: E402
    DialogueContentSelectionCheckpointService,
)
from attribute_first.application.dialogue_state import (  # noqa: E402
    DialogueInstanceState,
    DialogueStage,
)
from attribute_first.application.dialogue_turns import (  # noqa: E402
    jsonable_dialogue_value,
)
from attribute_first.application.dialogue_stage_prompts import (  # noqa: E402
    DialogueStagePromptBuilder,
    PreparedDialogueStage,
)
from attribute_first.ports import ChatRequest  # noqa: E402


class _ProviderRepeatedParts(Sequence):
    """Minimal stand-in for the Gemini SDK's repeated-part container."""

    def __init__(self, values):
        self._values = tuple(values)

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return len(self._values)


class DialogueTraceSerializationTests(unittest.TestCase):
    def test_provider_repeated_parts_remain_a_json_array(self):
        history = [
            SimpleNamespace(
                role="user",
                parts=_ProviderRepeatedParts(
                    [
                        SimpleNamespace(text="first part"),
                        SimpleNamespace(text="second part"),
                    ]
                ),
            )
        ]

        self.assertEqual(
            jsonable_dialogue_value(history),
            [
                {
                    "role": "user",
                    "parts": ["first part", "second part"],
                }
            ],
        )


class DialogueSessionServiceTests(unittest.TestCase):
    def test_cache_session_sends_only_the_uncached_target_suffix(self):
        session = SimpleNamespace(history=[])
        gateway = SimpleNamespace(
            create_chat=mock.Mock(return_value=session)
        )
        dependencies = SimpleNamespace(
            dialogue_gateway=gateway,
            stable_value_sha256=lambda value: f"sha:{value}",
        )
        service = DialogueSessionService(dependencies)
        cache = object()
        state = SimpleNamespace(
            plan=SimpleNamespace(model_name="models/test"),
            cache_state=SimpleNamespace(
                cache=cache,
                prefix="SHARED PREFIX\n",
                role_tails={},
            ),
        )
        instance = DialogueInstanceState(
            uid="u1",
            content_selection_prompt="SHARED PREFIX\nNEW TARGET",
            role_payload=None,
        )

        turn = service.start(state, instance)

        self.assertEqual(turn, "NEW TARGET")
        self.assertIs(instance.session, session)
        self.assertTrue(instance.cache_bound)
        self.assertEqual(
            instance.protocol["cs_live_message_sha256"],
            "sha:NEW TARGET",
        )
        gateway.create_chat.assert_called_once_with(
            ChatRequest(
                model_name="models/test",
                cached_content=cache,
                system_instruction=None,
                history=(),
            )
        )


class DialoguePlanBuilderTests(unittest.TestCase):
    def test_shared_cs_plan_does_not_build_or_count_cs_prompts(self):
        with tempfile.TemporaryDirectory() as temporary:
            used_demos = Path(temporary) / "used_demonstrations.json"
            used_demos.write_text(
                '[{"unique_id": "demo-1"}]\n',
                encoding="utf-8",
            )
            reference = SimpleNamespace(
                snapshot_for=lambda name: (
                    used_demos
                    if name == "used_demonstrations.json"
                    else None
                )
            )
            alignments = [{"unique_id": "u1"}, {"unique_id": "u2"}]
            stage_args = {
                "content_selection": SimpleNamespace(
                    subtask="content_selection",
                    model_name="models/test",
                    temperature=0.3,
                    num_retries=2,
                    n_demos=1,
                    structured_output=True,
                ),
                "ambiguity_highlight": SimpleNamespace(
                    subtask="ambiguity_highlight",
                    model_name="models/test",
                    n_demos=1,
                    structured_output=True,
                ),
                "fusion_in_context": SimpleNamespace(
                    subtask="FiC",
                    model_name="models/test",
                    n_demos=1,
                    structured_output=True,
                ),
            }
            dependencies = SimpleNamespace(
                build_subtask_args=lambda configs, name, original, **kwargs: (
                    stage_args[name]
                ),
                update_args=lambda value: value,
                env_flag=lambda name: name == "AF_USE_ROLES",
                get_data=mock.Mock(return_value=({}, alignments)),
                get_token_counter=mock.Mock(
                    side_effect=AssertionError(
                        "shared CS must not create a token counter"
                    )
                ),
                construct_prompts=mock.Mock(
                    side_effect=AssertionError(
                        "shared CS must not construct prompts"
                    )
                ),
                get_subtask_funcs=mock.Mock(
                    side_effect=AssertionError(
                        "shared CS must not prepare a parser"
                    )
                ),
                subtask_schemas={"content_selection": {"type": "object"}},
                stable_value_sha256=lambda value: f"sha:{value!r}",
                get_af_environment_flags=lambda: {},
                system_instruction="system",
            )
            builder = DialoguePlanBuilder(dependencies)
            ah_stage = DialogueStage(
                name="ambiguity_highlight",
                args=stage_args["ambiguity_highlight"],
                token_counter=None,
                prompt_dict={},
                structures={},
                parse_fn=None,
                pipeline_fn=None,
                schema={},
            )
            fic_stage = DialogueStage(
                name="fusion_in_context",
                args=stage_args["fusion_in_context"],
                token_counter=None,
                prompt_dict={},
                structures={},
                parse_fn=None,
                pipeline_fn=mock.Mock(),
                schema={},
            )
            args = SimpleNamespace(
                indir_alignments=None,
                _shared_content_selection_reference=reference,
            )
            full_configs = [
                {"subtask": "content_selection"},
                {"subtask": "ambiguity_highlight"},
                {"subtask": "fusion_in_context"},
            ]

            with mock.patch.object(
                builder,
                "_prepare_ambiguity_highlight",
                return_value=ah_stage,
            ), mock.patch.object(
                builder,
                "_prepare_fusion",
                return_value=(fic_stage, []),
            ):
                plan = builder.build(
                    args=args,
                    full_configs=full_configs,
                    original_args_dict={},
                    outdir=str(Path(temporary) / "out"),
                    intermediate_outdir=str(
                        Path(temporary) / "out" / "intermediate"
                    ),
                )

        self.assertEqual(
            set(plan.content_selection_prompts),
            {"u1", "u2"},
        )
        self.assertEqual(
            plan.initial_content_selection_demos,
            [{"unique_id": "demo-1"}],
        )
        dependencies.get_token_counter.assert_not_called()
        dependencies.construct_prompts.assert_not_called()
        dependencies.get_subtask_funcs.assert_not_called()


class DialogueResultPersisterTests(unittest.TestCase):
    def _state(self):
        fusion_converter = mock.Mock(
            return_value=[{"unique_id": "u2"}, {"unique_id": "u1"}]
        )
        plan = SimpleNamespace(
            alignments=[
                {"unique_id": "u2"},
                {"unique_id": "u1"},
            ],
            has_ambiguity_highlight=False,
            content_selection_outdir="intermediate/content_selection",
            ambiguity_highlight_outdir=None,
            final_outdir="final",
            model_name="models/test",
            fusion=SimpleNamespace(
                pipeline_fn=fusion_converter,
                args=SimpleNamespace(structured_output=True),
            ),
        )
        state = SimpleNamespace(
            plan=plan,
            content_selection_demos=[{"demo": "cs"}],
            ambiguity_highlight_demos=[],
            fusion_demos=[{"demo": "fic"}],
            content_selection_results={"u1": {"r": 1}, "u2": {"r": 2}},
            ambiguity_highlight_results={},
            fusion_results={"u1": {"f": 1}, "u2": {"f": 2}},
            content_selection_rows={
                "u1": {"unique_id": "u1", "stage": "cs"},
                "u2": {"unique_id": "u2", "stage": "cs"},
            },
            ambiguity_highlight_rows={},
            fusion_source_rows={
                "u1": {"unique_id": "u1", "stage": "source"},
                "u2": {"unique_id": "u2", "stage": "source"},
            },
            args_snapshot={
                "dialogue_role_contract": {},
            },
            call_records=[{"unique_id": "u1"}],
        )
        return state, fusion_converter

    def test_save_results_preserves_population_order_for_all_pipeline_rows(self):
        coverage_calls = []
        save_results = mock.Mock()
        dependencies = SimpleNamespace(
            assert_uid_coverage=lambda label, values, order: (
                coverage_calls.append((label, tuple(values), tuple(order)))
            ),
            save_results=save_results,
        )
        state, fusion_converter = self._state()

        DialogueResultPersister(dependencies).save_results(state)

        source_order = ("u2", "u1")
        self.assertTrue(
            all(call[2] == source_order for call in coverage_calls)
        )
        first_save = save_results.call_args_list[0]
        self.assertEqual(
            first_save.kwargs["pipeline_format_results"],
            [
                state.content_selection_rows["u2"],
                state.content_selection_rows["u1"],
            ],
        )
        fusion_converter.assert_called_once_with(
            state.fusion_results,
            [
                state.fusion_source_rows["u2"],
                state.fusion_source_rows["u1"],
            ],
            structured_output=True,
        )
        self.assertEqual(
            save_results.call_args_list[-1].kwargs[
                "pipeline_format_results"
            ],
            fusion_converter.return_value,
        )

    def test_conversion_failure_keeps_raw_fusion_results(self):
        state, fusion_converter = self._state()
        fusion_converter.side_effect = ValueError(
            "structured FiC coverage mismatch"
        )
        save_results = mock.Mock()
        dependencies = SimpleNamespace(
            assert_uid_coverage=lambda *_args: None,
            save_results=save_results,
        )

        with self.assertRaisesRegex(
            ValueError,
            "coverage mismatch",
        ):
            DialogueResultPersister(dependencies).save_results(state)

        save_results.assert_any_call(
            state.plan.final_outdir,
            state.fusion_demos,
            state.fusion_results,
            pipeline_format_results=None,
        )

    def test_runtime_artifacts_keep_cache_calls_usage_and_demo_hashes_together(self):
        artifact_store = SimpleNamespace(
            write_json=mock.Mock(),
            write_jsonl=mock.Mock(),
        )
        dependencies = SimpleNamespace(
            stable_value_sha256=lambda value: f"sha:{len(value)}",
            artifact_store=artifact_store,
            get_token_usage=lambda: {
                "calls": 3,
                "prompt": 100,
                "completion": 20,
                "cached": 40,
            },
        )
        state, _ = self._state()
        cache_trace = {"requested": True, "effective": True}

        DialogueResultPersister(
            dependencies
        ).persist_runtime_artifacts(state, cache_trace)

        role_contract = state.args_snapshot["dialogue_role_contract"]
        self.assertEqual(
            role_contract["demonstration_sets"],
            {
                "content_selection": {
                    "count": 1,
                    "sha256": "sha:1",
                },
                "ambiguity_highlight": {
                    "count": 0,
                    "sha256": "sha:0",
                },
                "fusion_in_context": {
                    "count": 1,
                    "sha256": "sha:1",
                },
            },
        )
        self.assertEqual(
            state.args_snapshot["dialogue_cache_trace"],
            cache_trace,
        )
        artifact_store.write_jsonl.assert_called_once_with(
            str(Path("final") / "dialogue_calls.jsonl"),
            state.call_records,
        )
        usage_call = artifact_store.write_json.call_args_list[-1]
        self.assertEqual(
            usage_call.args,
            (
                str(Path("final") / "token_usage.json"),
                {
                    "calls": 3,
                    "prompt": 100,
                    "completion": 20,
                    "cached": 40,
                    "subtask": "dialogue_pipeline",
                    "model": "models/test",
                },
            ),
        )

    def test_runtime_artifacts_propagate_token_usage_write_failure(self):
        def write_json(path, payload):
            del payload
            if path == str(Path("final") / "token_usage.json"):
                raise OSError("token usage disk failure")

        artifact_store = SimpleNamespace(
            write_json=mock.Mock(side_effect=write_json),
            write_jsonl=mock.Mock(),
        )
        dependencies = SimpleNamespace(
            stable_value_sha256=lambda value: f"sha:{len(value)}",
            artifact_store=artifact_store,
            get_token_usage=lambda: {
                "calls": 3,
                "prompt": 100,
                "completion": 20,
                "cached": 40,
            },
        )
        state, _ = self._state()

        with self.assertRaisesRegex(
            OSError,
            "token usage disk failure",
        ):
            DialogueResultPersister(
                dependencies
            ).persist_runtime_artifacts(
                state,
                {"requested": False, "effective": False},
            )

        artifact_store.write_jsonl.assert_called_once_with(
            str(Path("final") / "dialogue_calls.jsonl"),
            state.call_records,
        )


class DialogueFusionValidationTests(unittest.TestCase):
    def test_reused_content_selection_failure_does_not_claim_fusion_trace(
        self,
    ):
        budget_trace = {
            "prompt_token_budget": 30_000,
            "transport_scope": "constructed_stage_prompt",
        }
        dependencies = SimpleNamespace(
            with_gold_summary=lambda result, _row: result,
        )
        state = SimpleNamespace(
            plan=SimpleNamespace(has_ambiguity_highlight=False),
            content_selection_results={
                "u1": {
                    "final_output": "ERROR - dialogue CS failed",
                    "prompt_budget_trace": budget_trace,
                }
            },
            fusion_source_rows={},
            fusion_results={},
        )
        instance = DialogueInstanceState(
            uid="u1",
            content_selection_prompt="",
            role_payload=None,
        )

        DialogueContentSelectionCheckpointService(
            dependencies,
            session_service=mock.Mock(),
        )._seed_failure(
            state,
            instance,
            {"unique_id": "u1"},
            {"unique_id": "u1"},
        )

        self.assertNotIn(
            "prompt_budget_trace",
            state.fusion_results["u1"],
        )

    def test_content_selection_failure_does_not_claim_a_fusion_budget_trace(
        self,
    ):
        budget_trace = {
            "prompt_token_budget": 30_000,
            "transport_scope": "constructed_stage_prompt",
        }
        dependencies = SimpleNamespace(
            single_pipeline_row=lambda *_args: {"unique_id": "u1"},
            with_gold_summary=lambda result, _row: result,
        )
        state = SimpleNamespace(
            plan=SimpleNamespace(
                content_selection_additional={
                    "u1": {"prompt_budget_trace": budget_trace}
                },
                content_selection=SimpleNamespace(
                    pipeline_fn=mock.Mock()
                ),
                has_ambiguity_highlight=False,
            ),
            content_selection_results={},
            content_selection_rows={},
            ambiguity_highlight_results={},
            ambiguity_highlight_rows={},
            fusion_source_rows={},
            fusion_results={},
        )
        instance = DialogueInstanceState(
            uid="u1",
            content_selection_prompt="CS",
            role_payload=None,
        )

        DialogueContentSelectionService(
            dependencies,
            session_service=mock.Mock(),
        ).record_failure(
            state,
            instance,
            {"unique_id": "u1"},
        )

        self.assertEqual(
            state.content_selection_results["u1"]["prompt_budget_trace"],
            budget_trace,
        )
        self.assertNotIn(
            "prompt_budget_trace",
            state.fusion_results["u1"],
        )

    def test_fusion_failure_keeps_its_constructed_prompt_budget_trace(self):
        budget_trace = {
            "prompt_token_budget": 30_000,
            "transport_scope": "constructed_stage_prompt",
        }
        prepared = PreparedDialogueStage(
            demos=[],
            validation_prompt="validation prompt",
            additional={
                "u1": {
                    "highlighted_docs": [],
                    "prompt_budget_trace": budget_trace,
                }
            },
            role_messages={},
            live_demo_count=0,
        )
        dependencies = SimpleNamespace(
            fic_highlight_registry=lambda _additional: "1. highlight",
            stable_value_sha256=lambda value: f"sha:{value!r}",
            dialogue_turn=mock.Mock(return_value=(None, None)),
            with_gold_summary=lambda result, _row: result,
        )
        stage = SimpleNamespace(
            continuation="ONLY THE NEW FUSION TASK",
            parse_fn=mock.Mock(),
            schema={"type": "object"},
            args=SimpleNamespace(
                num_retries=2,
                temperature=0.0,
                output_max_length=8192,
                model_name="models/test",
            ),
        )
        state = SimpleNamespace(
            plan=SimpleNamespace(
                fusion=stage,
                num_retries=2,
                temperature=0.0,
            ),
            call_records=[],
            fusion_results={},
        )
        instance = DialogueInstanceState(
            uid="u1",
            content_selection_prompt="CS",
            role_payload=None,
            session=object(),
        )

        DialogueFusionService(
            dependencies,
            SimpleNamespace(build=mock.Mock(return_value=prepared)),
            SimpleNamespace(inject=mock.Mock()),
        ).run(
            state,
            instance,
            {"unique_id": "u1"},
            {"unique_id": "u1"},
        )

        self.assertEqual(
            state.fusion_results["u1"]["prompt_budget_trace"],
            budget_trace,
        )

    def test_shared_role_transport_keeps_downstream_demo_count(self):
        plan = SimpleNamespace(no_demos=False)
        instance = DialogueInstanceState(
            uid="u1",
            content_selection_prompt="shared",
            role_payload=None,
            uses_roles=True,
        )
        stage = SimpleNamespace(args=SimpleNamespace(n_demos=4))

        self.assertEqual(
            DialogueStagePromptBuilder._live_demo_count(
                plan,
                instance,
                stage,
            ),
            4,
        )

    def test_fusion_parser_receives_full_validation_prompt_not_new_turn(self):
        validation_prompt = (
            "The highlighted spans are:\n"
            "1. First source span\n"
            "2. Second source span"
        )
        prepared = PreparedDialogueStage(
            demos=[{"demo": 1}, {"demo": 2}],
            validation_prompt=validation_prompt,
            additional={"u1": {"highlighted_docs": []}},
            role_messages={"u1": []},
            live_demo_count=2,
        )
        dialogue_turn = mock.Mock(
            return_value=(
                {
                    "final_output": "Summary.",
                    "alignments": [
                        {
                            "sent_id": 1,
                            "sent_text": "Summary.",
                            "highlights": [1, 2],
                        }
                    ],
                },
                '{"sentences":[]}',
            )
        )
        dependencies = SimpleNamespace(
            fic_highlight_registry=lambda _additional: (
                "The highlighted spans are:\n"
                "1. First source span\n"
                "2. Second source span"
            ),
            stable_value_sha256=lambda value: f"sha:{value!r}",
            dialogue_turn=dialogue_turn,
            with_gold_summary=lambda result, _row: result,
        )
        stage = SimpleNamespace(
            continuation="ONLY THE NEW FUSION TASK",
            parse_fn=mock.Mock(),
            schema={"type": "object"},
            args=SimpleNamespace(
                num_retries=2,
                temperature=0.0,
                output_max_length=8192,
                model_name="models/test",
            ),
        )
        state = SimpleNamespace(
            plan=SimpleNamespace(
                fusion=stage,
                num_retries=2,
                temperature=0.0,
            ),
            call_records=[],
            fusion_results={},
        )
        instance = DialogueInstanceState(
            uid="u1",
            content_selection_prompt="CS",
            role_payload=None,
            uses_roles=True,
            session=object(),
        )
        demonstration_service = SimpleNamespace(
            inject=mock.Mock(),
        )
        service = DialogueFusionService(
            dependencies,
            SimpleNamespace(build=mock.Mock(return_value=prepared)),
            demonstration_service,
        )

        service.run(
            state,
            instance,
            {"unique_id": "u1"},
            {"unique_id": "u1"},
        )

        self.assertEqual(
            dialogue_turn.call_args.args[3],
            validation_prompt,
        )
        application_message = dialogue_turn.call_args.args[1]
        self.assertIn("ONLY THE NEW FUSION TASK", application_message)
        self.assertIn(
            "### CANONICAL HIGHLIGHT IDS FOR THIS TURN ###",
            application_message,
        )
        self.assertIn("2. Second source span", application_message)
        demonstration_service.inject.assert_called_once_with(
            state=state,
            instance=instance,
            stage_name="fusion_in_context",
            demos=prepared.demos,
            role_messages=prepared.role_messages,
            demo_count=2,
        )


if __name__ == "__main__":
    unittest.main()
