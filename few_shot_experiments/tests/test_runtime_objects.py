"""Object contracts for the generation runtime.

These tests use only in-memory fakes.  Importing and exercising the application
runtime must never configure or call an external model provider.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))


class RuntimeObjectTests(unittest.TestCase):
    def test_output_directory_claim_is_atomic_and_scopes_children(self):
        from concurrent.futures import ThreadPoolExecutor
        from attribute_first.artifacts import OutputDirectoryClaim

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"

            def claim():
                return OutputDirectoryClaim.claim(
                    run_dir,
                    owner="offline-test",
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(claim) for _ in range(2)]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(("ok", future.result()))
                except ValueError as exc:
                    outcomes.append(("error", str(exc)))

            self.assertEqual(
                sum(kind == "ok" for kind, _ in outcomes),
                1,
            )
            self.assertEqual(
                sum(kind == "error" for kind, _ in outcomes),
                1,
            )
            child = OutputDirectoryClaim.prepare_child(
                run_dir / "stages" / "content_selection",
                owner_root=run_dir,
            )
            self.assertTrue(child.is_dir())
            with self.assertRaisesRegex(ValueError, "escapes"):
                OutputDirectoryClaim.prepare_child(
                    Path(tmp) / "outside",
                    owner_root=run_dir,
                )

    def test_uppercase_gpt_id_routes_to_openai(self):
        import utils

        with mock.patch.object(
            utils,
            "openai_call",
            return_value="accepted",
        ) as openai_call, mock.patch.object(
            utils,
            "gemini_call",
        ) as gemini_call, mock.patch.object(
            utils,
            "ensure_parseable_finish_reason",
            return_value=None,
        ):
            result = utils.model_call_wrapper(
                prompt="new task",
                model_name="GPT-offline-test",
                parse_response_fn=lambda *, response, prompt: {
                    "final_output": response,
                },
                num_retries=1,
            )

        self.assertEqual(result["final_output"], "accepted")
        openai_call.assert_called_once()
        gemini_call.assert_not_called()

    def test_unknown_model_provider_is_rejected_before_a_call(self):
        import utils

        with mock.patch.object(
            utils,
            "openai_call",
        ) as openai_call, mock.patch.object(
            utils,
            "gemini_call",
        ) as gemini_call:
            with self.assertRaisesRegex(
                ValueError,
                "unsupported provider model ID",
            ):
                utils.model_call_wrapper(
                    prompt="new task",
                    model_name="unclassified-model",
                    parse_response_fn=lambda **_kwargs: {},
                    num_retries=1,
                )

        openai_call.assert_not_called()
        gemini_call.assert_not_called()

    def test_openai_wrapper_rejects_structured_transport_before_call(self):
        import utils

        with mock.patch.object(
            utils,
            "openai_call",
        ) as openai_call:
            with self.assertRaisesRegex(
                ValueError,
                "does not support request field.*response_schema",
            ):
                utils.model_call_wrapper(
                    prompt="new task",
                    model_name="gpt-offline-test",
                    parse_response_fn=lambda **_kwargs: {},
                    response_schema={"type": "object"},
                    num_retries=1,
                )

        openai_call.assert_not_called()

    def test_openai_wrapper_records_parseable_finish_and_usage(self):
        import utils

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content="  accepted  "),
                )
            ],
            model="gpt-offline-test",
            usage=SimpleNamespace(
                prompt_tokens=7,
                completion_tokens=2,
                total_tokens=9,
                prompt_tokens_details=SimpleNamespace(cached_tokens=3),
            ),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock.Mock(return_value=response)
                )
            )
        )
        utils.reset_token_usage()

        with mock.patch.object(utils, "openai", client):
            result = utils.model_call_wrapper(
                prompt="new task",
                model_name="gpt-offline-test",
                parse_response_fn=lambda *, response, prompt: {
                    "final_output": response,
                },
                num_retries=1,
            )

        self.assertEqual(result["final_output"], "accepted")
        attempt = result["attempt_trace"][0]
        self.assertEqual(attempt["status"], "parsed")
        self.assertEqual(
            attempt["response_metadata"],
            {
                "provider_response_received": True,
                "model_version": "gpt-offline-test",
                "finish_reason": "STOP",
                "prompt_block_reason": None,
            },
        )
        self.assertEqual(
            attempt["usage"],
            {
                "prompt_token_count": 7,
                "candidates_token_count": 2,
                "cached_content_token_count": 3,
                "total_token_count": 9,
            },
        )

    def test_openai_length_finish_is_never_passed_to_the_parser(self):
        import utils

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(content="partial"),
                )
            ],
            model="gpt-offline-test",
            usage=SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                prompt_tokens_details=None,
            ),
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock.Mock(return_value=response)
                )
            )
        )
        parser = mock.Mock(return_value={"final_output": "must not run"})

        with mock.patch.object(utils, "openai", client):
            result = utils.model_call_wrapper(
                prompt="new task",
                model_name="gpt-offline-test",
                parse_response_fn=parser,
                num_retries=1,
            )

        parser.assert_not_called()
        self.assertTrue(result["final_output"].startswith("ERROR"))
        attempt = result["attempt_trace"][0]
        self.assertEqual(
            attempt["response_metadata"]["finish_reason"],
            "MAX_TOKENS",
        )
        self.assertEqual(attempt["failure_phase"], "generation")

    def test_attempt_executor_retries_parse_failures_with_full_evidence(self):
        from attribute_first.runtime import (
            AttemptDependencies,
            AttemptExecutor,
            AttemptPolicy,
        )

        responses = iter(("invalid", "valid"))
        current = {"usage": None, "metadata": None}

        def invoke():
            current["usage"] = {
                "prompt_token_count": 1,
                "candidates_token_count": 1,
                "cached_content_token_count": 0,
            }
            current["metadata"] = {"finish_reason": "STOP"}
            return next(responses)

        def parse(*, response, prompt):
            if response == "invalid":
                raise ValueError("schema mismatch")
            return {"final_output": f"{prompt}:{response}"}

        executor = AttemptExecutor(
            AttemptDependencies(
                invoke=invoke,
                parse=parse,
                reset_evidence=lambda: current.update(
                    usage=None,
                    metadata=None,
                ),
                last_usage=lambda: current["usage"],
                last_metadata=lambda: current["metadata"],
                ensure_parseable=lambda metadata: None,
                fingerprint=lambda value: f"hash:{value}",
                sleep=lambda _seconds: None,
            )
        )

        result = executor.execute(
            prompt="prompt",
            policy=AttemptPolicy(
                model_name="models/fake",
                output_max_length=10,
                num_retries=2,
                temperature=0,
            ),
            application_request={"transport": "flat"},
            response_schema={"type": "object"},
        )

        self.assertEqual(result["final_output"], "prompt:valid")
        self.assertEqual(
            [attempt["status"] for attempt in result["attempt_trace"]],
            ["error", "parsed"],
        )
        self.assertEqual(
            result["attempt_trace"][0]["failure_phase"],
            "parse",
        )
        self.assertEqual(
            result["attempt_trace"][0]["response_schema_sha256"],
            "hash:{'type': 'object'}",
        )

    def test_attempt_executor_propagates_programming_errors_without_retry(self):
        from attribute_first.runtime import (
            AttemptDependencies,
            AttemptExecutor,
            AttemptPolicy,
        )

        for error_type in (KeyError, AttributeError, TypeError):
            with self.subTest(error_type=error_type.__name__):
                invoke = mock.Mock(return_value="provider response")
                parse = mock.Mock(
                    side_effect=error_type("programming defect")
                )
                sleep = mock.Mock()
                executor = AttemptExecutor(
                    AttemptDependencies(
                        invoke=invoke,
                        parse=parse,
                        reset_evidence=lambda: None,
                        last_usage=lambda: {},
                        last_metadata=lambda: {
                            "finish_reason": "STOP"
                        },
                        ensure_parseable=lambda _metadata: None,
                        fingerprint=lambda _value: "sha",
                        sleep=sleep,
                    )
                )

                with self.assertRaises(error_type):
                    executor.execute(
                        prompt="prompt",
                        policy=AttemptPolicy(
                            model_name="models/fake",
                            output_max_length=10,
                            num_retries=3,
                            temperature=0,
                        ),
                        application_request={"transport": "flat"},
                    )
                invoke.assert_called_once()
                parse.assert_called_once()
                sleep.assert_not_called()

    def test_attempt_policy_rejects_bool_and_out_of_range_temperature(self):
        from attribute_first.runtime import AttemptPolicy

        for temperature in (True, False, -0.01, 2.01):
            with self.subTest(temperature=temperature):
                with self.assertRaisesRegex(
                    ValueError,
                    "temperature",
                ):
                    AttemptPolicy(
                        model_name="models/fake",
                        output_max_length=10,
                        num_retries=1,
                        temperature=temperature,
                    )

    def test_attempt_executor_retries_provider_429(self):
        from attribute_first.runtime import (
            AttemptDependencies,
            AttemptExecutor,
            AttemptPolicy,
        )

        invoke = mock.Mock(
            side_effect=(
                RuntimeError("429 resource exhausted"),
                "valid",
            )
        )
        sleep = mock.Mock()
        executor = AttemptExecutor(
            AttemptDependencies(
                invoke=invoke,
                parse=lambda **_kwargs: {"final_output": "ok"},
                reset_evidence=lambda: None,
                last_usage=lambda: {},
                last_metadata=lambda: {"finish_reason": "STOP"},
                ensure_parseable=lambda _metadata: None,
                fingerprint=lambda _value: "sha",
                sleep=sleep,
            )
        )

        result = executor.execute(
            prompt="prompt",
            policy=AttemptPolicy(
                model_name="models/fake",
                output_max_length=10,
                num_retries=2,
                temperature=0,
            ),
            application_request={"transport": "flat"},
        )

        self.assertEqual(result["final_output"], "ok")
        sleep.assert_called_once_with(60)

    def test_dialogue_turn_propagates_programming_errors_without_retry(self):
        from attribute_first.application.dialogue_turns import (
            DialogueTurnDependencies,
            DialogueTurnExecutor,
        )

        for phase in ("transport", "parse"):
            for error_type in (KeyError, AttributeError, TypeError):
                with self.subTest(
                    phase=phase,
                    error_type=error_type.__name__,
                ):
                    gateway = SimpleNamespace(
                        send_message=mock.Mock(
                            side_effect=(
                                error_type("programming defect")
                                if phase == "transport"
                                else None
                            ),
                            return_value="provider response",
                        )
                    )
                    if phase == "transport":
                        gateway.send_message = mock.Mock(
                            side_effect=error_type(
                                "programming defect"
                            )
                        )
                    parser = mock.Mock(
                        side_effect=(
                            error_type("programming defect")
                            if phase == "parse"
                            else None
                        ),
                        return_value={"final_output": "ok"},
                    )
                    sleep = mock.Mock()
                    executor = DialogueTurnExecutor(
                        DialogueTurnDependencies(
                            dialogue_gateway=gateway,
                            reset_last_call_usage=lambda: None,
                            get_last_call_usage=lambda: {},
                            get_last_call_metadata=lambda: {
                                "finish_reason": "STOP"
                            },
                            ensure_parseable_finish_reason=(
                                lambda _metadata: None
                            ),
                            stable_value_sha256=lambda _value: "sha",
                            incomplete_generation_error=ValueError,
                            time_module=SimpleNamespace(sleep=sleep),
                        )
                    )

                    with self.assertRaises(error_type):
                        executor.execute(
                            SimpleNamespace(history=[]),
                            "new task",
                            parser,
                            "validation prompt",
                            num_retries=3,
                            temperature=0,
                        )
                    gateway.send_message.assert_called_once()
                    sleep.assert_not_called()

    def test_usage_recorders_fail_fast_when_provider_usage_is_missing(self):
        import utils

        gemini_response = SimpleNamespace(
            candidates=[],
            prompt_feedback=None,
            model_version="gemini-test",
            usage_metadata=None,
        )
        openai_response = SimpleNamespace(
            choices=[],
            model="openai-test",
            usage=None,
        )

        for recorder, response in (
            (utils._record_usage, gemini_response),
            (utils._record_openai_usage, openai_response),
        ):
            with self.subTest(recorder=recorder.__name__):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "mandatory usage metadata",
                ):
                    recorder(response)
                self.assertIsNone(utils.get_last_call_usage())
                self.assertTrue(
                    utils.get_last_call_metadata()[
                        "provider_response_received"
                    ]
                )

    def test_usage_recorder_does_not_swallow_invalid_counter_types(self):
        import utils

        response = SimpleNamespace(
            candidates=[],
            prompt_feedback=None,
            model_version="gemini-test",
            usage_metadata=SimpleNamespace(
                prompt_token_count="not-an-integer",
                candidates_token_count=1,
                cached_content_token_count=0,
                total_token_count=1,
            ),
        )

        with self.assertRaises(ValueError):
            utils._record_usage(response)

    def test_usage_ledger_keeps_totals_and_thread_local_evidence(self):
        from attribute_first.runtime import UsageLedger

        ledger = UsageLedger()
        ledger.record(
            prompt=11,
            completion=3,
            cached=5,
            provider_total=14,
            metadata={
                "provider_response_received": True,
                "finish_reason": "STOP",
            },
        )
        ledger.record(
            prompt=7,
            completion=2,
            cached=0,
            provider_total=19,
            metadata={
                "provider_response_received": True,
                "finish_reason": "STOP",
            },
        )

        self.assertEqual(
            ledger.snapshot(),
            {
                "prompt": 18,
                "completion": 5,
                "cached": 5,
                "calls": 2,
                "provider_total": 33,
                "provider_total_calls": 2,
            },
        )
        self.assertEqual(
            ledger.last_usage(),
            {
                "prompt_token_count": 7,
                "candidates_token_count": 2,
                "cached_content_token_count": 0,
                "total_token_count": 19,
            },
        )
        self.assertEqual(
            ledger.last_metadata()["finish_reason"],
            "STOP",
        )

        ledger.reset()
        self.assertEqual(
            ledger.snapshot(),
            {
                "prompt": 0,
                "completion": 0,
                "cached": 0,
                "calls": 0,
                "provider_total": 0,
                "provider_total_calls": 0,
            },
        )
        self.assertIsNone(ledger.last_usage())
        self.assertIsNone(ledger.last_metadata())

    def test_protocol_environment_restores_the_exact_process_state(self):
        from attribute_first.runtime import ProtocolEnvironment

        policy = ProtocolEnvironment(
            allowed_flags=("AF_CONTEXT_CACHE", "AF_USE_ROLES")
        )
        original = os.environ.get("AF_CONTEXT_CACHE")
        os.environ["AF_CONTEXT_CACHE"] = "false"
        try:
            with policy.apply(
                {
                    "environment_flags": {
                        "AF_CONTEXT_CACHE": False,
                        "AF_USE_ROLES": True,
                    }
                }
            ) as effective:
                self.assertEqual(
                    effective,
                    {
                        "AF_CONTEXT_CACHE": False,
                        "AF_USE_ROLES": True,
                    },
                )
                self.assertEqual(os.environ["AF_USE_ROLES"], "true")
            self.assertNotIn("AF_USE_ROLES", os.environ)
            self.assertEqual(os.environ["AF_CONTEXT_CACHE"], "false")
        finally:
            if original is None:
                os.environ.pop("AF_CONTEXT_CACHE", None)
            else:
                os.environ["AF_CONTEXT_CACHE"] = original

    def test_model_gateway_is_a_structural_port(self):
        from attribute_first.ports import (
            DialogueGateway,
            GenerationGateway,
            ModelGateway,
        )

        class FakeGateway:
            def generate(self, request):
                return "offline"

            def create_chat(self, request):
                return {"history": [], "request": request}

            def send_message(self, chat, request):
                chat["history"].append(request)
                return "offline"

        gateway = FakeGateway()
        self.assertIsInstance(gateway, GenerationGateway)
        self.assertIsInstance(gateway, DialogueGateway)
        self.assertIsInstance(gateway, ModelGateway)
        self.assertEqual(gateway.generate({"prompt": "x"}), "offline")

    def test_openai_implements_only_the_supported_generation_port(self):
        from attribute_first.infrastructure import OpenAIGateway
        from attribute_first.ports import (
            DialogueGateway,
            GenerationGateway,
            ModelGateway,
        )

        gateway = OpenAIGateway(SimpleNamespace())

        self.assertIsInstance(gateway, GenerationGateway)
        self.assertNotIsInstance(gateway, DialogueGateway)
        self.assertNotIsInstance(gateway, ModelGateway)

    def test_openai_gateway_uses_the_declared_system_instruction(self):
        from attribute_first.infrastructure import OpenAIGateway
        from attribute_first.ports import GenerationRequest

        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  generated  ")
                )
            ]
        )
        create = mock.Mock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)
            )
        )
        recorded = []
        resets = []
        gateway = OpenAIGateway(
            client,
            record_usage=recorded.append,
            reset_evidence=lambda: resets.append(True),
        )

        result = gateway.generate(
            GenerationRequest(
                model_name="gpt-test",
                prompt="new task",
                output_max_length=321,
                temperature=0.25,
                system_instruction="Follow the scientific protocol.",
            )
        )

        self.assertEqual(result, "generated")
        self.assertEqual(recorded, [response])
        self.assertEqual(resets, [True])
        create.assert_called_once_with(
            model="gpt-test",
            messages=[
                {
                    "role": "system",
                    "content": "Follow the scientific protocol.",
                },
                {"role": "user", "content": "new task"},
            ],
            max_tokens=321,
            temperature=0.25,
        )

    def test_openai_gateway_rejects_fields_it_cannot_honor(self):
        from attribute_first.infrastructure import OpenAIGateway
        from attribute_first.ports import GenerationRequest

        create = mock.Mock()
        gateway = OpenAIGateway(
            SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(create=create)
                )
            )
        )
        unsupported_requests = (
            GenerationRequest(
                model_name="gpt-test",
                prompt="task",
                response_schema={"type": "object"},
            ),
            GenerationRequest(
                model_name="gpt-test",
                prompt="task",
                model_override=object(),
            ),
            GenerationRequest(
                model_name="gpt-test",
                prompt="task",
                contents=[{"role": "user", "parts": ["task"]}],
            ),
        )

        for request in unsupported_requests:
            with self.subTest(request=request):
                with self.assertRaisesRegex(
                    ValueError,
                    "does not support request field",
                ):
                    gateway.generate(request)
        create.assert_not_called()

    def test_application_dependencies_use_runtime_ports(self):
        from attribute_first.application import (
            DialoguePipelineDependencies,
            StandardPipelineDependencies,
        )
        from attribute_first.application.planned_pipeline import (
            PlannedPipelineDependencies,
        )
        from attribute_first.application.dialogue_turns import (
            DialogueTurnDependencies,
        )
        from attribute_first.stages.planned import (
            StageExecutionDependencies,
        )
        from attribute_first.ports import (
            ArtifactStore,
            BatchGenerationGateway,
            DialogueGateway,
            GenerationGateway,
        )

        self.assertIs(
            StandardPipelineDependencies.__annotations__[
                "generation_gateway"
            ],
            BatchGenerationGateway,
        )
        self.assertIs(
            StandardPipelineDependencies.__annotations__["artifact_store"],
            ArtifactStore,
        )
        self.assertIs(
            DialoguePipelineDependencies.__annotations__[
                "dialogue_gateway"
            ],
            DialogueGateway,
        )
        self.assertIs(
            DialoguePipelineDependencies.__annotations__["artifact_store"],
            ArtifactStore,
        )
        self.assertIs(
            DialogueTurnDependencies.__annotations__["dialogue_gateway"],
            DialogueGateway,
        )
        self.assertIs(
            PlannedPipelineDependencies.__annotations__["artifact_store"],
            ArtifactStore,
        )
        self.assertIs(
            StageExecutionDependencies.__annotations__[
                "generation_gateway"
            ],
            GenerationGateway,
        )

    def test_gemini_gateway_owns_provider_transport_and_model_reuse(self):
        from attribute_first.infrastructure import GeminiGateway
        from attribute_first.ports import GenerationRequest

        response = SimpleNamespace(text="  generated  ", prompt_feedback="")
        model = SimpleNamespace(
            generate_content=mock.Mock(return_value=response)
        )
        sdk = SimpleNamespace(
            GenerativeModel=mock.Mock(return_value=model)
        )
        recorded = []
        gateway = GeminiGateway(
            sdk=sdk,
            content_types=SimpleNamespace(),
            normalize_model_name=lambda name: f"normalized/{name}",
            record_usage=recorded.append,
            ensure_parseable=lambda: None,
            reset_evidence=lambda: None,
            last_metadata=lambda: {"finish_reason": "STOP"},
            safety_settings=[],
        )

        request = GenerationRequest(
            model_name="fake",
            prompt="new task",
            output_max_length=123,
            response_schema={"type": "object"},
        )
        self.assertEqual(gateway.generate(request), "generated")
        self.assertEqual(gateway.generate(request), "generated")

        sdk.GenerativeModel.assert_called_once_with("normalized/fake")
        self.assertEqual(recorded, [response, response])
        call = model.generate_content.call_args
        self.assertEqual(call.args, ("new task",))
        self.assertEqual(
            call.kwargs["generation_config"],
            {
                "temperature": 0,
                "max_output_tokens": 123,
                "response_mime_type": "application/json",
                "response_schema": {"type": "object"},
            },
        )

    def test_gemini_gateway_returns_raw_text_before_finish_classification(self):
        from attribute_first.infrastructure import GeminiGateway
        from attribute_first.ports import GenerationRequest
        from attribute_first.runtime import (
            AttemptDependencies,
            AttemptExecutor,
            AttemptPolicy,
            ensure_parseable_finish_reason,
        )

        provider_responses = iter(
            (
                SimpleNamespace(
                    text="truncated raw JSON",
                    finish_reason="MAX_TOKENS",
                    prompt_feedback="",
                ),
                SimpleNamespace(
                    text="valid raw JSON",
                    finish_reason="STOP",
                    prompt_feedback="",
                ),
            )
        )
        model = SimpleNamespace(
            generate_content=mock.Mock(
                side_effect=lambda *_args, **_kwargs: next(
                    provider_responses
                )
            )
        )
        current = {"usage": None, "metadata": None}

        def record(response):
            current["usage"] = {
                "prompt_token_count": 4,
                "candidates_token_count": 2,
                "cached_content_token_count": 0,
            }
            current["metadata"] = {
                "finish_reason": response.finish_reason,
            }

        gateway = GeminiGateway(
            sdk=SimpleNamespace(
                GenerativeModel=mock.Mock(return_value=model)
            ),
            content_types=SimpleNamespace(),
            normalize_model_name=lambda name: name,
            record_usage=record,
            ensure_parseable=mock.Mock(
                side_effect=AssertionError(
                    "transport must not classify finish reasons"
                )
            ),
            reset_evidence=lambda: current.update(
                usage=None,
                metadata=None,
            ),
            last_metadata=lambda: current["metadata"],
            safety_settings=[],
        )
        request = GenerationRequest(
            model_name="models/test",
            prompt="prompt",
        )
        executor = AttemptExecutor(
            AttemptDependencies(
                invoke=lambda: gateway.generate(request),
                parse=lambda **kwargs: {
                    "final_output": kwargs["response"]
                },
                reset_evidence=lambda: current.update(
                    usage=None,
                    metadata=None,
                ),
                last_usage=lambda: current["usage"],
                last_metadata=lambda: current["metadata"],
                ensure_parseable=ensure_parseable_finish_reason,
                fingerprint=lambda value: str(value),
                sleep=lambda _seconds: None,
            )
        )

        result = executor.execute(
            prompt="prompt",
            policy=AttemptPolicy(
                model_name="models/test",
                output_max_length=10,
                num_retries=2,
                temperature=0,
            ),
            application_request={"transport": "flat"},
        )

        attempts = result["attempt_trace"]
        self.assertEqual(
            [attempt["raw_response"] for attempt in attempts],
            ["truncated raw JSON", "valid raw JSON"],
        )
        self.assertEqual(attempts[0]["failure_phase"], "generation")
        self.assertEqual(attempts[1]["status"], "parsed")
        self.assertTrue(all(attempt["usage"] for attempt in attempts))

    def test_gemini_gateway_sends_only_new_task_through_chat_boundary(self):
        from attribute_first.infrastructure import GeminiGateway
        from attribute_first.ports import ChatTurnRequest

        metadata = {"finish_reason": "STOP"}
        response = SimpleNamespace(text=" accepted ", prompt_feedback="")
        model = SimpleNamespace(
            generate_content=mock.Mock(return_value=response)
        )
        content = SimpleNamespace(role="", parts=["new task"])
        chat = SimpleNamespace(
            history=[SimpleNamespace(role="model", parts=["prior answer"])],
            model=model,
            _check_response=mock.Mock(),
        )
        gateway = GeminiGateway(
            sdk=SimpleNamespace(),
            content_types=SimpleNamespace(
                to_content=mock.Mock(return_value=content)
            ),
            normalize_model_name=lambda name: name,
            record_usage=lambda _response: None,
            ensure_parseable=lambda: None,
            reset_evidence=lambda: None,
            last_metadata=lambda: metadata,
            safety_settings=[],
        )

        result = gateway.send_message(
            chat,
            ChatTurnRequest(message="new task"),
        )

        self.assertEqual(result, "accepted")
        sent = model.generate_content.call_args.kwargs["contents"]
        self.assertEqual(sent, [chat.history[0], content])
        self.assertEqual(chat._last_sent, content)
        self.assertEqual(chat._last_received, response)

    def test_dialogue_pipeline_always_finalizes_created_cache(self):
        from attribute_first.application import DialoguePipelineRunner

        plan = SimpleNamespace(
            content_selection_prompts={"u1": "target"},
            dialogue_role_messages={},
            roles_requested=False,
            model_name="models/test",
            initial_args_snapshot={},
            initial_content_selection_demos=[],
            initial_fusion_demos=[],
        )
        artifact_store = SimpleNamespace(write_json=mock.Mock())
        dependencies = SimpleNamespace(
            artifact_store=artifact_store,
            reset_token_usage=mock.Mock(),
            env_flag=mock.Mock(return_value=True),
            caching_module=SimpleNamespace(),
            normalize_model_name=lambda value: value,
            context_cache_target="target",
            tqdm=lambda values: values,
        )
        runner = DialoguePipelineRunner(dependencies)
        cache_state = object()
        runner._plan_builder = SimpleNamespace(
            build=mock.Mock(return_value=plan)
        )
        runner._cache_manager = SimpleNamespace(
            create=mock.Mock(return_value=cache_state),
            finalize=mock.Mock(return_value={"deleted": True}),
        )
        runner._run_instance = mock.Mock()
        runner._save_results = mock.Mock(
            side_effect=RuntimeError("simulated persistence failure")
        )
        runner._persist_runtime_artifacts = mock.Mock()

        with self.assertRaisesRegex(
            RuntimeError,
            "simulated persistence failure",
        ):
            runner.run(
                SimpleNamespace(),
                [],
                {},
                "out",
                "out/intermediate",
            )

        runner._cache_manager.finalize.assert_called_once_with(
            cache_state,
            mock.ANY,
        )
        runner._persist_runtime_artifacts.assert_called_once_with(
            mock.ANY,
            {"deleted": True},
        )

    def test_standard_pipeline_fails_if_token_usage_cannot_be_persisted(self):
        from attribute_first.application.standard_pipeline import (
            StandardPipelineRunner,
        )

        artifact_store = SimpleNamespace(
            write_json=mock.Mock(
                side_effect=OSError("simulated token evidence failure")
            )
        )
        runner = StandardPipelineRunner(
            SimpleNamespace(
                get_token_usage=lambda: {
                    "calls": 1,
                    "prompt": 10,
                    "completion": 2,
                    "cached": 0,
                },
                artifact_store=artifact_store,
            )
        )
        state = SimpleNamespace(
            outdir="claimed-output",
            args=SimpleNamespace(
                subtask="FiC",
                model_name="models/test",
            ),
        )

        with self.assertRaisesRegex(
            OSError,
            "token evidence failure",
        ):
            runner._write_token_usage(state)

    def test_dialogue_pipeline_propagates_unexpected_programming_error(self):
        from attribute_first.application.dialogue_pipeline import (
            DialoguePipelineRunner,
        )

        runner = object.__new__(DialoguePipelineRunner)
        runner._dependencies = SimpleNamespace(
            stable_value_sha256=lambda value: f"sha:{value}"
        )
        runner._session_service = SimpleNamespace(
            start=mock.Mock(
                side_effect=KeyError("simulated dialogue invariant")
            )
        )
        state = SimpleNamespace(
            plan=SimpleNamespace(
                alignments=[{"unique_id": "u1"}],
                dialogue_role_messages={},
                roles_requested=False,
            )
        )

        with self.assertRaisesRegex(
            KeyError,
            "dialogue invariant",
        ):
            runner._run_instance(state, "u1", "live prompt")

    def test_planned_pipeline_propagates_unexpected_programming_error(self):
        from attribute_first.application.planned_pipeline import (
            PlannedPipelineRunner,
        )

        runner = PlannedPipelineRunner(
            SimpleNamespace(
                run_instance=mock.Mock(
                    side_effect=KeyError("simulated planned invariant")
                ),
                terminal_result=mock.Mock(
                    return_value={
                        "plan_metadata": {"stage_traces": {}}
                    }
                ),
                empty_stage_traces=mock.Mock(return_value={}),
                trace_usage_summary=mock.Mock(return_value={}),
            )
        )
        context = SimpleNamespace(
            args=SimpleNamespace(model="models/test", setting="MDS"),
            protocol_sha256="sha:protocol",
        )

        with self.assertRaisesRegex(
            KeyError,
            "planned invariant",
        ):
            runner._work(
                {"unique_id": "u1"},
                context,
                {"u1": "gold"},
            )

    def test_json_artifact_store_writes_atomic_json_and_jsonl(self):
        from attribute_first.infrastructure import JsonArtifactStore

        with tempfile.TemporaryDirectory() as temporary_directory:
            store = JsonArtifactStore(Path(temporary_directory))
            store.write_json("nested/value.json", {"answer": 42})
            store.write_jsonl(
                "nested/rows.jsonl",
                ({"id": "a"}, {"id": "b"}),
            )

            self.assertEqual(
                store.read_json("nested/value.json"),
                {"answer": 42},
            )
            self.assertEqual(
                [
                    json.loads(line)
                    for line in (
                        Path(temporary_directory) / "nested/rows.jsonl"
                    ).read_text(encoding="utf-8").splitlines()
                ],
                [{"id": "a"}, {"id": "b"}],
            )

    def test_json_artifact_store_rejects_paths_outside_its_root(self):
        from attribute_first.infrastructure import JsonArtifactStore

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            store_root = temporary_root / "store"
            outside_root = temporary_root / "outside"
            outside_root.mkdir()
            store = JsonArtifactStore(store_root)

            for unsafe_path in (
                outside_root / "absolute.json",
                Path("..") / "outside" / "parent.json",
            ):
                with self.subTest(path=unsafe_path):
                    with self.assertRaisesRegex(
                        ValueError,
                        "must stay below",
                    ):
                        store.write_json(unsafe_path, {"unsafe": True})

            (store_root / "linked").symlink_to(
                outside_root,
                target_is_directory=True,
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                store.write_json(
                    "linked/symlink.json",
                    {"unsafe": True},
                )
            self.assertEqual(list(outside_root.iterdir()), [])

    def test_json_artifact_store_cleans_temporary_file_on_replace_failure(self):
        from attribute_first.infrastructure import JsonArtifactStore

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            store = JsonArtifactStore(root)
            with mock.patch(
                "attribute_first.infrastructure.json_artifact_store."
                "os.replace",
                side_effect=OSError("simulated replace failure"),
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "simulated replace failure",
                ):
                    store.write_json("nested/value.json", {"answer": 42})

            destination = root / "nested" / "value.json"
            self.assertFalse(destination.exists())
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
