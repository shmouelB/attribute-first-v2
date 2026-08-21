import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


def _load_core_utils():
    spec = importlib.util.spec_from_file_location(
        "dialogue_protocol_core_utils",
        EXPERIMENT_ROOT / "utils.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_full_pipeline(core_utils):
    run_script_stub = types.ModuleType("run_script")
    run_script_stub.main = lambda args: None
    iterative_stub = types.ModuleType("run_iterative_sentence_generation")
    iterative_stub.main = lambda args: None
    subtask_stub = types.ModuleType("subtask_specific_utils")
    for name in (
        "get_data",
        "get_subtask_funcs",
        "get_subtask_prompt_structures",
        "construct_non_demo_part",
        "construct_prompts",
        "parse_ambiguity_highlight_response",
        "convert_ambiguity_highlight_results_to_pipeline_format",
        "parse_FiC_response",
        "convert_FiC_CoT_results_to_pipeline_format",
    ):
        setattr(subtask_stub, name, lambda *args, **kwargs: None)

    spec = importlib.util.spec_from_file_location(
        "dialogue_protocol_run_full_pipeline",
        EXPERIMENT_ROOT / "run_full_pipeline.py",
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "utils": core_utils,
            "run_script": run_script_stub,
            "run_iterative_sentence_generation": iterative_stub,
            "subtask_specific_utils": subtask_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _load_run_dialogue_sequential(core_utils):
    prompt_stub = types.ModuleType("prompt_utils")
    prompt_stub.get_data = lambda args: None
    prompt_stub.get_subtask_prompt_structures = lambda *args, **kwargs: None
    prompt_stub.construct_prompts = lambda *args, **kwargs: None

    spec = importlib.util.spec_from_file_location(
        "dialogue_protocol_run_dialogue_sequential",
        EXPERIMENT_ROOT / "run_dialogue_sequential.py",
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"utils": core_utils, "prompt_utils": prompt_stub},
    ):
        spec.loader.exec_module(module)
    return module


CORE_UTILS = _load_core_utils()
RUN_FULL_PIPELINE = _load_run_full_pipeline(CORE_UTILS)
RUN_DIALOGUE_SEQUENTIAL = _load_run_dialogue_sequential(CORE_UTILS)

SCHEMAS_SPEC = importlib.util.spec_from_file_location(
    "dialogue_protocol_schemas",
    EXPERIMENT_ROOT / "schemas.py",
)
SCHEMAS = importlib.util.module_from_spec(SCHEMAS_SPEC)
SCHEMAS_SPEC.loader.exec_module(SCHEMAS)


def _record_provider_stop():
    CORE_UTILS._record_usage(
        SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=1,
                candidates_token_count=1,
                cached_content_token_count=0,
                total_token_count=2,
            ),
            model_version="gemini-test-001",
            candidates=[SimpleNamespace(finish_reason=1)],
            prompt_feedback=SimpleNamespace(block_reason=0),
        )
    )


class _StatefulChat:
    """Provider-boundary fake: the SDK session, not application code, owns history."""

    def __init__(self):
        self.history = []
        self.initial_history = []
        self.application_messages = []
        self.cached_content = None
        self.system_instruction = None

    def record_exchange(self, message, raw_response):
        self.application_messages.append(message)
        self.history.extend(
            [
                {"role": "user", "parts": [message]},
                {"role": "model", "parts": [raw_response]},
            ]
        )


def _message_text(message):
    """Normalize the Gemini-compatible message shapes used at the provider boundary."""
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        return _message_text(message.get("parts", []))
    if isinstance(message, (list, tuple)):
        return "".join(_message_text(part) for part in message)
    return str(message)


class DialogueProtocolContractTests(unittest.TestCase):
    def _write_stage_config(
        self,
        directory,
        filename,
        subtask,
        *,
        num_retries=1,
    ):
        config = {
            "split": "test",
            "setting": "MDS",
            "subtask": subtask,
            "model_name": "models/gemini-test",
            "n_demos": 1,
            "num_retries": num_retries,
            "temperature": 0.0,
            "structured_output": True,
            "output_max_length": 128,
            "CoT": subtask == "FiC",
            "always_with_question": False,
            "debugging": False,
            "merge_cross_sents_highlights": False,
            "cut_surplus": False,
            "prct_surplus": None,
        }
        path = Path(directory) / filename
        path.write_text(json.dumps(config), encoding="utf-8")
        return path

    def _run_structured_full_dialogue(
        self,
        *,
        use_roles=False,
        context_cache=False,
        fail_first_cached_turn=False,
        fail_ah_turn=False,
        fail_cs_parse=False,
        fail_cs_converter=False,
        no_demos=False,
    ):
        selector_calls = []
        parser_calls = []
        chat_calls = []
        prompt_construction_calls = []
        fic_conversion_kwargs = []
        sessions = []
        saved_results = []
        json_writes = []
        jsonl_writes = []
        cached_failures_remaining = [1 if fail_first_cached_turn else 0]
        source_instances = [
            {
                "unique_id": uid,
                "response": f"Gold {uid}",
                "set_of_highlights_in_context": [],
            }
            for uid in (("u1", "u2") if context_cache else ("u1",))
        ]
        uids = [instance["unique_id"] for instance in source_instances]
        demo_history = [
            {"role": "user", "parts": ["CS_DEMO_USER"]},
            {"role": "model", "parts": ["CS_DEMO_MODEL"]},
        ]
        ah_demo_history = [
            {"role": "user", "parts": ["AH_DEMO_USER"]},
            {"role": "model", "parts": ["AH_DEMO_MODEL"]},
        ]
        fic_demo_history = [
            {"role": "user", "parts": ["FIC_DEMO_USER"]},
            {"role": "model", "parts": ["FIC_DEMO_MODEL"]},
        ]
        cache_prefix = "CS_SHARED_DEMO_PREFIX\n"
        cache_target_header = "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###\n"
        fake_cache = SimpleNamespace(
            usage_metadata=SimpleNamespace(total_token_count=17),
            delete=mock.Mock(),
        )
        cache_create = mock.Mock(return_value=fake_cache)

        def parser_for(stage, mode):
            def parse(raw_response, prompt):
                parser_calls.append((stage, mode))
                if stage == "content_selection":
                    if fail_cs_parse:
                        return None
                    return {
                        "final_output": {
                            "Document [1]": ["Alpha"]
                        },
                        "alignments": [],
                    }
                if stage == "ambiguity_highlight":
                    if fail_ah_turn:
                        return None
                    return {"final_output": "AH parsed", "alignments": []}
                return {
                    "final_output": "FiC parsed",
                    "alignments": [
                        {
                            "sent_id": 1,
                            "sent_text": "FiC parsed",
                            "highlights": [1],
                        }
                    ],
                }

            return parse

        def convert_content_selection(results, source):
            if fail_cs_converter:
                raise RuntimeError("simulated CS converter failure")
            converted = [
                dict(instance, from_live_cs=True)
                for instance in source
                if instance["unique_id"] in results
            ]
            return converted

        def convert_ambiguity(results, source):
            return [
                dict(instance, from_live_ah=True)
                for instance in source
                if instance["unique_id"] in results
            ]

        def convert_fic(results, source, **kwargs):
            fic_conversion_kwargs.append(dict(kwargs))
            return list(source)

        converters = {
            "content_selection": convert_content_selection,
            "ambiguity_highlight": convert_ambiguity,
            "FiC": convert_fic,
        }

        def select_subtask(stage, structured_output=False):
            selector_calls.append((stage, structured_output))
            mode = "structured" if structured_output else "text"
            return parser_for(stage, mode), converters[stage]

        def construct_prompts(**kwargs):
            prompt_source = kwargs["prompt_dict"]
            prompt_construction_calls.append(
                {
                    "stage": next(
                        name
                        for name, candidate in prompt_dicts.items()
                        if candidate is prompt_source
                    ),
                    "n_demos": kwargs["n_demos"],
                    "alignments": json.loads(
                        json.dumps(kwargs["alignments_dict"])
                    ),
                }
            )
            if kwargs["no_highlights"]:
                cs_demo_history = (
                    list(demo_history) if kwargs["n_demos"] else []
                )
                role_messages = {
                    uid: {
                        "system": "CS_SYSTEM_INSTRUCTION",
                        "contents": cs_demo_history
                        + [
                            {
                                "role": "user",
                                "parts": [f"CS_TARGET_ONLY_{uid}"],
                            }
                        ],
                    }
                    for uid in uids
                }
                if context_cache:
                    prompts = {
                        uid: (
                            f"{cache_prefix}{cache_target_header}"
                            f"CS_FLAT_TARGET_{uid}"
                        )
                        for uid in uids
                    }
                else:
                    prompts = {"u1": "CS_FULL_APPLICATION_PROMPT"}
                return (
                    ([{"demo": "cs"}] if kwargs["n_demos"] else []),
                    prompts,
                    {
                        uid: {"non_highlighted_docs": []}
                        for uid in uids
                    },
                    role_messages if use_roles else {},
                )
            if kwargs["n_demos"]:
                if prompt_source is prompt_dicts["ambiguity_highlight"]:
                    stage_system = "AH_SYSTEM_INSTRUCTION"
                    stage_history = ah_demo_history
                else:
                    stage_system = "FIC_SYSTEM_INSTRUCTION"
                    stage_history = fic_demo_history
                stage_uids = [
                    row["unique_id"]
                    for row in kwargs["alignments_dict"]
                ]
                stage_roles = {
                    uid: {
                        "system": stage_system,
                        "contents": list(stage_history)
                        + [
                            {
                                "role": "user",
                                "parts": [f"IGNORED_STAGE_TARGET_{uid}"],
                            }
                        ],
                    }
                    for uid in stage_uids
                }
                if use_roles:
                    if prompt_source is prompt_dicts["ambiguity_highlight"]:
                        prompts = {
                            uid: (
                                "AH_VALIDATION_PROMPT\n"
                                "Document [1]: Alpha"
                            )
                            for uid in stage_uids
                        }
                        additional = {
                            uid: {"non_highlighted_docs": []}
                            for uid in stage_uids
                        }
                    else:
                        prompts = {
                            uid: (
                                "FIC_VALIDATION_PROMPT\n"
                                "The highlighted spans are:\n"
                                "1. Document [1]: Alpha"
                            )
                            for uid in stage_uids
                        }
                        additional = {
                            uid: {
                                "highlights": [["Alpha"]],
                                "highlighted_docs": [],
                            }
                            for uid in stage_uids
                        }
                    return (
                        [{"demo": "fic"}],
                        prompts,
                        additional,
                        stage_roles,
                    )
                return (
                    [{"demo": "fic"}],
                    {
                        uid: (
                            "FIC_DEMO_BLOCK\n"
                            "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###\n"
                            "target"
                        )
                        for uid in stage_uids
                    },
                    {},
                    stage_roles if use_roles else {},
                )
            if prompt_source is prompt_dicts["ambiguity_highlight"]:
                return (
                    [],
                    {
                        uid: (
                            "AH_VALIDATION_PROMPT\n"
                            "Document [1]: Alpha"
                        )
                        for uid in uids
                    },
                    {
                        uid: {"non_highlighted_docs": []}
                        for uid in uids
                    },
                    {},
                )
            if prompt_source is prompt_dicts["FiC"]:
                return (
                    [],
                    {
                        uid: (
                            "FIC_VALIDATION_PROMPT\n"
                            "The highlighted spans are:\n"
                            "1. Document [1]: Alpha"
                        )
                        for uid in uids
                    },
                    {
                        uid: {
                            "highlights": [["Alpha"]],
                            "highlighted_docs": [],
                        }
                        for uid in uids
                    },
                    {},
                )
            return (
                [],
                {uid: "unused" for uid in uids},
                {uid: {} for uid in uids},
                {},
            )

        def create_chat_session(
            model_name,
            cached_content=None,
            system_instruction=None,
            history=None,
        ):
            session = _StatefulChat()
            session.model_name = model_name
            session.cached_content = cached_content
            session.system_instruction = system_instruction
            session.initial_history = list(history or [])
            session.history = list(history or [])
            sessions.append(session)
            return session

        def chat_call(
            session,
            message,
            output_max_length=4096,
            temperature=0,
            response_schema=None,
        ):
            message_text = _message_text(message)
            if (
                message_text == "CS_FULL_APPLICATION_PROMPT"
                or "CS_FLAT_TARGET_" in message_text
                or "CS_TARGET_ONLY_" in message_text
            ):
                stage = "content_selection"
            elif "AH_ONLY_INSTRUCTION" in message_text:
                stage = "ambiguity_highlight"
            else:
                stage = "FiC"
            chat_calls.append(
                {
                    "stage": stage,
                    "session": session,
                    "message": message,
                    "response_schema": response_schema,
                    "history_before_call": json.loads(
                        json.dumps(session.history)
                    ),
                }
            )
            if (
                stage == "content_selection"
                and session.cached_content is fake_cache
                and cached_failures_remaining[0]
            ):
                cached_failures_remaining[0] -= 1
                raise RuntimeError("simulated cached CS transport failure")
            CORE_UTILS._record_usage(
                SimpleNamespace(
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=1,
                        candidates_token_count=1,
                        cached_content_token_count=0,
                        total_token_count=2,
                    ),
                    model_version="gemini-test-001",
                    candidates=[SimpleNamespace(finish_reason=1)],
                    prompt_feedback=SimpleNamespace(block_reason=0),
                )
            )
            raw_response = f"{stage} raw response"
            session.record_exchange(message, raw_response)
            return raw_response

        def save_results(
            outdir,
            used_demos,
            final_results,
            pipeline_format_results=None,
        ):
            saved_results.append(
                {
                    "outdir": outdir,
                    "results": json.loads(json.dumps(final_results)),
                    "pipeline": json.loads(json.dumps(pipeline_format_results)),
                }
            )

        def record_json(path, value, **_kwargs):
            json_writes.append(
                (str(path), json.loads(json.dumps(value)))
            )

        def record_jsonl(path, values, **_kwargs):
            jsonl_writes.append(
                (str(path), json.loads(json.dumps(values)))
            )

        prompt_dicts = {
            "content_selection": {},
            "ambiguity_highlight": {
                "instruction-ambiguity-highlight": "AH_ONLY_INSTRUCTION",
                "answer_ambiguity_highlight_prompt": "",
                "answer_ambiguity_highlight_format": "",
            },
            "FiC": {
                "instruction-FiC-CoT": "FIC_ONLY_INSTRUCTION",
                "answer_FiC-CoT_prompt": "",
                "dialogue-FiC-CoT-format-example": "",
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            retry_budget = 2 if fail_first_cached_turn else 1
            cs_config = self._write_stage_config(
                tmp,
                "content-selection.json",
                "content_selection",
                num_retries=retry_budget,
            )
            ah_config = self._write_stage_config(
                tmp,
                "ambiguity-highlight.json",
                "ambiguity_highlight",
                num_retries=retry_budget,
            )
            fic_config = self._write_stage_config(
                tmp,
                "fic.json",
                "FiC",
                num_retries=retry_budget,
            )
            full_configs = [
                {
                    "subtask": "content_selection",
                    "config_file": str(cs_config),
                },
                {
                    "subtask": "ambiguity_highlight",
                    "config_file": str(ah_config),
                },
                {
                    "subtask": "fusion_in_context",
                    "config_file": str(fic_config),
                },
            ]
            with mock.patch.object(
                RUN_FULL_PIPELINE,
                "_load_subtask_prompt_dict",
                side_effect=lambda args: prompt_dicts[args.subtask],
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "get_subtask_prompt_structures",
                return_value={},
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "get_data",
                return_value=({}, source_instances),
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "construct_prompts",
                side_effect=construct_prompts,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "get_token_counter",
                return_value={},
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "get_subtask_funcs",
                side_effect=select_subtask,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "parse_ambiguity_highlight_response",
                side_effect=parser_for("ambiguity_highlight", "text"),
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "convert_ambiguity_highlight_results_to_pipeline_format",
                side_effect=converters["ambiguity_highlight"],
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "create_chat_session",
                side_effect=create_chat_session,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "gemini_chat_call",
                side_effect=chat_call,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "save_results",
                side_effect=save_results,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "atomic_write_json",
                side_effect=record_json,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "atomic_write_jsonl",
                side_effect=record_jsonl,
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "get_token_usage",
                return_value={
                    "prompt": 0,
                    "completion": 0,
                    "cached": 0,
                    "calls": 3,
                },
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "env_flag",
                side_effect=lambda name: (
                    use_roles
                    if name == "AF_USE_ROLES"
                    else context_cache
                    if name == "AF_CONTEXT_CACHE"
                    else no_demos
                    if name == "AF_DIALOGUE_NO_DEMOS"
                    else False
                ),
            ), mock.patch.object(
                RUN_FULL_PIPELINE,
                "tqdm",
                side_effect=lambda values: values,
            ), mock.patch(
                "google.generativeai.caching.CachedContent.create",
                side_effect=cache_create,
            ), mock.patch.dict(
                sys.modules,
                {"utils": CORE_UTILS},
            ):
                RUN_FULL_PIPELINE.run_dialogue_pipeline(
                    SimpleNamespace(indir_alignments=None),
                    full_configs,
                    {},
                    str(Path(tmp) / "out"),
                    str(Path(tmp) / "out" / "intermediate"),
                )

        return {
            "selector_calls": selector_calls,
            "parser_calls": parser_calls,
            "chat_calls": chat_calls,
            "prompt_construction_calls": prompt_construction_calls,
            "fic_conversion_kwargs": fic_conversion_kwargs,
            "sessions": sessions,
            "demo_history_by_stage": {
                "content_selection": demo_history,
                "ambiguity_highlight": ah_demo_history,
                "fusion_in_context": fic_demo_history,
            },
            "fake_cache": fake_cache,
            "cache_create": cache_create,
            "saved_results": saved_results,
            "json_writes": json_writes,
            "jsonl_writes": jsonl_writes,
        }

    def test_structured_dialogue_forwards_schema_on_every_stage(self):
        observed = self._run_structured_full_dialogue()

        self.assertEqual(
            [
                (call["stage"], call["response_schema"])
                for call in observed["chat_calls"]
            ],
            [
                ("content_selection", SCHEMAS.CONTENT_SELECTION_SCHEMA),
                ("ambiguity_highlight", SCHEMAS.AMBIGUITY_HIGHLIGHT_SCHEMA),
                ("FiC", SCHEMAS.FIC_COT_SCHEMA),
            ],
        )

    def test_structured_dialogue_selects_parser_from_each_stage_config(self):
        observed = self._run_structured_full_dialogue()

        self.assertEqual(
            observed["selector_calls"],
            [
                ("content_selection", True),
                ("ambiguity_highlight", True),
                ("FiC", True),
            ],
        )
        self.assertEqual(
            observed["parser_calls"],
            [
                ("content_selection", "structured"),
                ("ambiguity_highlight", "structured"),
                ("FiC", "structured"),
            ],
        )
        self.assertEqual(
            observed["fic_conversion_kwargs"],
            [{"structured_output": True}],
        )

    def test_dialogue_runners_reuse_one_chat_without_resending_prior_prompts(self):
        full = self._run_structured_full_dialogue()

        self.assertEqual(len(full["sessions"]), 1)
        full_session = full["sessions"][0]
        self.assertTrue(
            all(call["session"] is full_session for call in full["chat_calls"])
        )
        self.assertEqual(len(full_session.history), 6)
        self.assertEqual(
            full_session.application_messages[0],
            "CS_FULL_APPLICATION_PROMPT",
        )
        self.assertNotIn(
            "CS_FULL_APPLICATION_PROMPT",
            full_session.application_messages[1],
        )
        self.assertNotIn(
            "CS_FULL_APPLICATION_PROMPT",
            full_session.application_messages[2],
        )
        self.assertNotIn(
            full_session.application_messages[1],
            full_session.application_messages[2],
        )

        sequential_sessions = []

        def create_sequential_session(model_name):
            session = _StatefulChat()
            sequential_sessions.append(session)
            return session

        def sequential_chat_call(
            session,
            message,
            output_max_length=4096,
            temperature=0,
            response_schema=None,
        ):
            call_number = len(session.application_messages)
            raw_responses = (
                json.dumps(
                    {
                        "highlights": [
                            {"doc_id": "1", "span_text": "Alpha"}
                        ]
                    }
                ),
                json.dumps({"clusters": [[1]]}),
                json.dumps(
                    {
                        "sentence_text": "Alpha.",
                        "highlight_ids": [1],
                    }
                ),
            )
            raw_response = raw_responses[call_number]
            _record_provider_stop()
            session.record_exchange(message, raw_response)
            return raw_response

        with mock.patch.object(
            RUN_DIALOGUE_SEQUENTIAL,
            "create_chat_session",
            side_effect=create_sequential_session,
        ), mock.patch.object(
            RUN_DIALOGUE_SEQUENTIAL,
            "gemini_chat_call",
            side_effect=sequential_chat_call,
        ):
            result, _ = RUN_DIALOGUE_SEQUENTIAL.run_instance(
                {
                    "documents": [
                        {
                            "documentFile": "doc-a",
                            "rawDocumentText": "Alpha",
                        }
                    ]
                },
                "CS_FULL_APPLICATION_PROMPT",
                "CLUSTER_ONLY_INSTRUCTION",
                "FUSE_ONLY_INSTRUCTION",
                "models/gemini-test",
                num_retries=1,
            )

        self.assertEqual(result["final_output"], "Alpha.")
        self.assertEqual(len(sequential_sessions), 1)
        sequential_session = sequential_sessions[0]
        self.assertEqual(len(sequential_session.application_messages), 3)
        self.assertNotIn(
            "CS_FULL_APPLICATION_PROMPT",
            sequential_session.application_messages[1],
        )
        self.assertNotIn(
            "CS_FULL_APPLICATION_PROMPT",
            sequential_session.application_messages[2],
        )
        self.assertNotIn(
            sequential_session.application_messages[1],
            sequential_session.application_messages[2],
        )

    def test_roles_dialogue_seeds_cs_contract_and_sends_only_target_at_turn_one(self):
        observed = self._run_structured_full_dialogue(use_roles=True)

        self.assertEqual(len(observed["sessions"]), 1)
        session = observed["sessions"][0]
        self.assertEqual(
            session.system_instruction,
            RUN_FULL_PIPELINE.DIALOGUE_SYSTEM_INSTRUCTION,
        )
        self.assertNotEqual(
            session.system_instruction,
            "CS_SYSTEM_INSTRUCTION",
        )
        initial_history_text = "\n".join(
            _message_text(message) for message in session.initial_history
        )
        self.assertIn("CS_SYSTEM_INSTRUCTION", initial_history_text)
        self.assertNotIn("AH_SYSTEM_INSTRUCTION", initial_history_text)
        self.assertNotIn("FIC_SYSTEM_INSTRUCTION", initial_history_text)
        self.assertEqual(len(session.application_messages), 3)
        self.assertEqual(
            _message_text(session.application_messages[0]),
            "### LIVE INSTANCE — CONTENT_SELECTION ###\n"
            "CS_SYSTEM_INSTRUCTION\n\nCS_TARGET_ONLY_u1",
        )
        self.assertIn(
            "AH_ONLY_INSTRUCTION",
            _message_text(session.application_messages[1]),
        )
        self.assertIn(
            "FIC_ONLY_INSTRUCTION",
            _message_text(session.application_messages[2]),
        )
        self.assertNotIn(
            "### LIVE STATE FROM CONTENT_SELECTION ###",
            _message_text(session.application_messages[1]),
        )
        self.assertNotIn(
            "Document [1]: Alpha",
            _message_text(session.application_messages[1]),
        )
        self.assertIn(
            "The highlighted spans are:",
            _message_text(session.application_messages[2]),
        )
        self.assertIn(
            "Document [1]: Alpha",
            _message_text(session.application_messages[2]),
        )
        self.assertNotIn(
            "CS_FULL_APPLICATION_PROMPT",
            _message_text(session.application_messages[2]),
        )
        sent_text = "\n".join(
            _message_text(message) for message in session.application_messages
        )
        self.assertNotIn("CS_FULL_APPLICATION_PROMPT", sent_text)
        for demo_text in (
            "CS_DEMO_USER",
            "CS_DEMO_MODEL",
            "AH_DEMO_USER",
            "AH_DEMO_MODEL",
            "FIC_DEMO_USER",
            "FIC_DEMO_MODEL",
        ):
            self.assertNotIn(demo_text, sent_text)
        self.assertTrue(
            all(call["session"] is session for call in observed["chat_calls"])
        )
        history_before_cs, history_before_ah, history_before_fic = (
            "\n".join(
                _message_text(message)
                for message in call["history_before_call"]
            )
            for call in observed["chat_calls"]
        )
        self.assertEqual(history_before_cs.count("CS_DEMO_USER"), 1)
        self.assertNotIn("AH_DEMO_USER", history_before_cs)
        self.assertNotIn("FIC_DEMO_USER", history_before_cs)
        self.assertNotIn("CS_DEMO_USER", history_before_ah)
        self.assertIn("content_selection raw response", history_before_ah)
        self.assertEqual(history_before_ah.count("AH_DEMO_USER"), 1)
        self.assertNotIn("FIC_DEMO_USER", history_before_ah)
        self.assertNotIn("CS_DEMO_USER", history_before_fic)
        self.assertNotIn("AH_DEMO_USER", history_before_fic)
        self.assertIn("content_selection raw response", history_before_fic)
        self.assertIn("ambiguity_highlight raw response", history_before_fic)
        self.assertEqual(history_before_fic.count("FIC_DEMO_USER"), 1)

    def test_jit_stage_demos_are_selected_from_live_intermediate_rows(self):
        observed = self._run_structured_full_dialogue(use_roles=True)

        few_shot_calls = [
            call
            for call in observed["prompt_construction_calls"]
            if call["stage"] in {"ambiguity_highlight", "FiC"}
            and call["n_demos"]
        ]
        self.assertEqual(
            [call["stage"] for call in few_shot_calls],
            ["ambiguity_highlight", "FiC"],
        )
        self.assertTrue(
            few_shot_calls[0]["alignments"][0]["from_live_cs"]
        )
        self.assertTrue(
            few_shot_calls[1]["alignments"][0]["from_live_ah"]
        )

    def test_roles_cache_failure_recreates_equivalent_non_cached_session(self):
        observed = self._run_structured_full_dialogue(
            use_roles=True,
            context_cache=True,
            fail_first_cached_turn=True,
        )

        cached_sessions = [
            session
            for session in observed["sessions"]
            if session.cached_content is observed["fake_cache"]
        ]
        non_cached_sessions = [
            session
            for session in observed["sessions"]
            if session.cached_content is None
        ]
        self.assertGreaterEqual(len(cached_sessions), 1)
        self.assertEqual(len(non_cached_sessions), 2)

        first_cached = cached_sessions[0]
        fallback = next(
            session
            for session in non_cached_sessions
            if any(
                "CS_DEMO_USER" in _message_text(message)
                for message in session.initial_history
            )
        )
        detached_continuation = next(
            session
            for session in non_cached_sessions
            if session is not fallback
        )
        self.assertIsNone(first_cached.system_instruction)
        self.assertEqual(first_cached.initial_history, [])
        self.assertEqual(
            fallback.system_instruction,
            RUN_FULL_PIPELINE.DIALOGUE_SYSTEM_INSTRUCTION,
        )
        fallback_history_text = "\n".join(
            _message_text(message) for message in fallback.initial_history
        )
        self.assertIn("CS_SYSTEM_INSTRUCTION", fallback_history_text)
        self.assertNotIn("AH_SYSTEM_INSTRUCTION", fallback_history_text)
        self.assertNotIn("FIC_SYSTEM_INSTRUCTION", fallback_history_text)
        detached_history_text = "\n".join(
            _message_text(message)
            for message in detached_continuation.initial_history
        )
        self.assertIn(
            "content_selection raw response",
            detached_history_text,
        )
        self.assertNotIn("CS_DEMO_USER", detached_history_text)
        self.assertEqual(
            len(detached_continuation.application_messages),
            2,
        )
        self.assertIn(
            "AH_ONLY_INSTRUCTION",
            _message_text(
                detached_continuation.application_messages[0]
            ),
        )
        self.assertIn(
            "FIC_ONLY_INSTRUCTION",
            _message_text(
                detached_continuation.application_messages[1]
            ),
        )
        cache_kwargs = observed["cache_create"].call_args.kwargs
        self.assertEqual(
            cache_kwargs["system_instruction"],
            fallback.system_instruction,
        )
        self.assertEqual(
            cache_kwargs["contents"],
            fallback.initial_history,
        )

        failed_attempt = next(
            call
            for call in observed["chat_calls"]
            if call["session"] is first_cached
            and call["stage"] == "content_selection"
        )
        expected_first_turn = (
            "### LIVE INSTANCE — CONTENT_SELECTION ###\n"
            "CS_SYSTEM_INSTRUCTION\n\nCS_TARGET_ONLY_u1"
        )
        self.assertEqual(
            _message_text(failed_attempt["message"]),
            expected_first_turn,
        )
        self.assertEqual(
            _message_text(fallback.application_messages[0]),
            expected_first_turn,
        )
        self.assertEqual(len(fallback.application_messages), 3)
        self.assertEqual(
            len(
                [
                    call
                    for call in observed["chat_calls"]
                    if call["stage"] == "content_selection"
                    and "u1" in _message_text(call["message"])
                ]
            ),
            2,
            "cache fallback must stay within the declared retry budget",
        )
        self.assertIn(
            "AH_ONLY_INSTRUCTION",
            _message_text(fallback.application_messages[1]),
        )
        self.assertIn(
            "FIC_ONLY_INSTRUCTION",
            _message_text(fallback.application_messages[2]),
        )
        fallback_text = "\n".join(
            _message_text(message) for message in fallback.application_messages
        )
        self.assertNotIn("CS_SHARED_DEMO_PREFIX", fallback_text)
        self.assertNotIn("CS_FLAT_TARGET_u1", fallback_text)
        for demo_text in (
            "CS_DEMO_USER",
            "CS_DEMO_MODEL",
            "AH_DEMO_USER",
            "AH_DEMO_MODEL",
            "FIC_DEMO_USER",
            "FIC_DEMO_MODEL",
        ):
            self.assertNotIn(demo_text, fallback_text)

        final_args = [
            value
            for path, value in observed["json_writes"]
            if path.endswith("args.json")
        ][-1]
        cache_trace = final_args["dialogue_cache_trace"]
        self.assertTrue(cache_trace["requested"])
        self.assertTrue(cache_trace["created"])
        self.assertTrue(cache_trace["delete_attempted"])
        self.assertTrue(cache_trace["deleted"])
        self.assertGreater(cache_trace["bound_calls"], 0)

        call_records = [
            values
            for path, values in observed["jsonl_writes"]
            if path.endswith("dialogue_calls.jsonl")
        ][0]
        self.assertEqual(len(call_records), len(observed["chat_calls"]))
        fallback_cs_records = [
            record
            for record in call_records
            if record["unique_id"] == "u1"
            and record["stage"] == "content_selection"
        ]
        self.assertEqual(
            [record["attempt"] for record in fallback_cs_records],
            [1, 2],
            "cache fallback must continue absolute attempt numbering",
        )
        self.assertEqual(
            [record["num_retries"] for record in fallback_cs_records],
            [2, 2],
            "every fallback record must retain the declared absolute limit",
        )
        fallback_record = fallback_cs_records[1]
        self.assertTrue(fallback_record["cache_fallback"])
        self.assertEqual(
            fallback_record["session_reset_reason"],
            "cache_transport_fallback",
        )
        self.assertEqual(
            fallback_record["session_reset_before"],
            fallback_record["local_history_before"],
        )
        final_results = observed["saved_results"][-1]["results"]
        self.assertEqual(
            final_results["u1"]["dialogue_protocol_trace"][
                "cs_cache_fallback_message_sha256"
            ],
            RUN_FULL_PIPELINE.stable_value_sha256(
                fallback_record["application_message"]
            ),
        )
        for record in call_records:
            for required in (
                "unique_id",
                "stage",
                "application_message",
                "local_history_before",
                "local_history_after",
                "usage",
                "status",
            ):
                self.assertIn(required, record)
            if record.get("failure_phase"):
                self.assertIn("error", record)
            else:
                self.assertIn("raw_response", record)

    def test_cached_parse_failure_does_not_get_non_cached_best_of_retry(self):
        observed = self._run_structured_full_dialogue(
            use_roles=True,
            context_cache=True,
            fail_cs_parse=True,
        )

        self.assertEqual(
            [
                session
                for session in observed["sessions"]
                if session.cached_content is None
            ],
            [],
        )
        self.assertTrue(
            all(
                session.cached_content is observed["fake_cache"]
                for session in observed["sessions"]
            )
        )
        final_results = observed["saved_results"][-1]["results"]
        self.assertTrue(
            all(
                result["final_output"]
                == "ERROR - dialogue CS failed"
                for result in final_results.values()
            )
        )

    def test_dialogue_no_demos_disables_every_stage_demo(self):
        observed = self._run_structured_full_dialogue(
            use_roles=True,
            no_demos=True,
        )

        session = observed["sessions"][0]
        self.assertEqual(session.initial_history, [])
        all_call_history = "\n".join(
            _message_text(message)
            for call in observed["chat_calls"]
            for message in call["history_before_call"]
        )
        for demo_text in (
            "CS_DEMO_USER",
            "CS_DEMO_MODEL",
            "AH_DEMO_USER",
            "AH_DEMO_MODEL",
            "FIC_DEMO_USER",
            "FIC_DEMO_MODEL",
        ):
            self.assertNotIn(demo_text, all_call_history)
        self.assertEqual(len(session.application_messages), 3)

    def test_sequential_dialogue_structures_every_model_turn(self):
        session = _StatefulChat()
        schemas_seen = []
        responses = iter(
            (
                json.dumps(
                    {
                        "highlights": [
                            {"doc_id": "1", "span_text": "Alpha"}
                        ]
                    }
                ),
                json.dumps({"clusters": [[1]]}),
                json.dumps(
                    {
                        "sentence_text": "Alpha.",
                        "highlight_ids": [1],
                    }
                ),
            )
        )

        def chat_call(
            active_session,
            message,
            output_max_length=4096,
            temperature=0,
            response_schema=None,
        ):
            schemas_seen.append(response_schema)
            raw = next(responses)
            _record_provider_stop()
            active_session.record_exchange(message, raw)
            return raw

        with mock.patch.object(
            RUN_DIALOGUE_SEQUENTIAL,
            "create_chat_session",
            return_value=session,
        ), mock.patch.object(
            RUN_DIALOGUE_SEQUENTIAL,
            "gemini_chat_call",
            side_effect=chat_call,
        ):
            result, _ = RUN_DIALOGUE_SEQUENTIAL.run_instance(
                {
                    "documents": [
                        {
                            "documentFile": "doc-a",
                            "rawDocumentText": "Alpha",
                        }
                    ]
                },
                "CS prompt",
                "Cluster instruction",
                "Fusion instruction",
                "models/gemini-test",
                num_retries=1,
            )

        self.assertEqual(result["final_output"], "Alpha.")
        self.assertEqual(
            schemas_seen,
            [
                SCHEMAS.CONTENT_SELECTION_SCHEMA,
                SCHEMAS.CLUSTERING_SCHEMA,
                RUN_DIALOGUE_SEQUENTIAL.SENTENCE_FUSION_SCHEMA,
            ],
        )

    def test_sequential_dialogue_rejects_fusion_attribution_mismatch(self):
        session = _StatefulChat()
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
                json.dumps({"clusters": [[1, 2]]}),
                json.dumps(
                    {
                        "sentence_text": "Alpha only.",
                        "highlight_ids": [1],
                    }
                ),
            )
        )

        def chat_call(active_session, message, **kwargs):
            raw = next(responses)
            _record_provider_stop()
            active_session.record_exchange(message, raw)
            return raw

        with mock.patch.object(
            RUN_DIALOGUE_SEQUENTIAL,
            "create_chat_session",
            return_value=session,
        ), mock.patch.object(
            RUN_DIALOGUE_SEQUENTIAL,
            "gemini_chat_call",
            side_effect=chat_call,
        ):
            result, _ = RUN_DIALOGUE_SEQUENTIAL.run_instance(
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
                "models/gemini-test",
                num_retries=1,
            )

        self.assertTrue(result["final_output"].startswith("ERROR"))
        self.assertEqual(result["alignments"], [])

    def test_full_dialogue_stops_before_fic_when_ambiguity_highlight_fails(self):
        observed = self._run_structured_full_dialogue(fail_ah_turn=True)

        self.assertEqual(
            [call["stage"] for call in observed["chat_calls"]],
            ["content_selection", "ambiguity_highlight"],
        )
        final_results = observed["saved_results"][-1]["results"]
        self.assertEqual(
            final_results["u1"]["final_output"],
            "ERROR - dialogue AH failed",
        )
        self.assertEqual(final_results["u1"]["alignments"], [])
        self.assertEqual(
            list(final_results["u1"]["dialogue_attempt_trace"]),
            [
                "content_selection",
                "ambiguity_highlight",
                "fusion_in_context",
            ],
        )
        self.assertEqual(
            final_results["u1"]["dialogue_attempt_trace"][
                "fusion_in_context"
            ],
            [],
        )

    def test_unexpected_instance_exception_fails_the_dialogue_run(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "simulated CS converter failure",
        ):
            self._run_structured_full_dialogue(
                fail_cs_converter=True,
            )


if __name__ == "__main__":
    unittest.main()
