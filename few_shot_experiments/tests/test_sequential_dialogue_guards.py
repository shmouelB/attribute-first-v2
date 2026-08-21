import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from attribute_first.application import sequential_dialogue as sequential
from attribute_first.runtime import (
    IncompleteGenerationError,
    ensure_parseable_finish_reason,
)


STOP_METADATA = {
    "provider_response_received": True,
    "model_version": "fixture-model",
    "finish_reason": "STOP",
    "prompt_block_reason": None,
}


class _ThreadEvidence:
    def __init__(self):
        self._local = threading.local()

    def reset(self):
        self._local.usage = None
        self._local.metadata = None

    def record(self, usage, metadata):
        self._local.usage = usage
        self._local.metadata = metadata

    def usage(self):
        return self._local.usage

    def metadata(self):
        return self._local.metadata


class _ScriptedSession:
    def __init__(self, model_name, responses):
        self.model_name = model_name
        self.responses = list(responses)
        self.history = []


class _ScriptedDialogueGateway:
    def __init__(self, scripts, evidence, barrier=None):
        self._scripts = scripts
        self._evidence = evidence
        self._barrier = barrier
        self.sessions = []
        self.requests = []
        self._lock = threading.Lock()

    def create_chat(self, request):
        session = _ScriptedSession(
            request.model_name,
            self._scripts[request.model_name],
        )
        with self._lock:
            self.sessions.append(session)
        return session

    def send_message(self, chat, request):
        response = chat.responses.pop(0)
        if self._barrier is not None:
            self._barrier.wait(timeout=5)
        self._evidence.record(
            response["usage"],
            response.get("metadata", STOP_METADATA),
        )
        chat.history.extend(
            [
                {"role": "user", "parts": [request.message]},
                {"role": "model", "parts": [response["raw"]]},
            ]
        )
        with self._lock:
            self.requests.append((chat.model_name, request))
        return response["raw"]


def _response(raw, prompt=1, completion=1, cached=0, metadata=None):
    return {
        "raw": raw,
        "usage": {
            "prompt_token_count": prompt,
            "candidates_token_count": completion,
            "cached_content_token_count": cached,
            "total_token_count": prompt + completion,
        },
        "metadata": metadata or dict(STOP_METADATA),
    }


def _instance():
    return {
        "unique_id": "fixture",
        "documents": [
            {
                "documentFile": "doc-a",
                "rawDocumentText": "Alpha only.",
                "documentText": ["Alpha only."],
                "docSentCharIdxToSentIdx": [0],
            },
            {
                "documentFile": "doc-b",
                "rawDocumentText": "Beta only.",
                "documentText": ["Beta only."],
                "docSentCharIdxToSentIdx": [0],
            },
        ],
        "set_of_highlights_in_context": [],
    }


def _dependencies(gateway, evidence, *, parse_content_selection=None, sleeps=None):
    return sequential.SequentialInstanceDependencies(
        dialogue_gateway=gateway,
        reset_last_call_usage=evidence.reset,
        get_last_call_usage=evidence.usage,
        get_last_call_metadata=evidence.metadata,
        ensure_parseable_finish_reason=ensure_parseable_finish_reason,
        stable_value_sha256=lambda _value: "fixture-sha256",
        incomplete_generation_error=IncompleteGenerationError,
        time_module=SimpleNamespace(
            sleep=(
                sleeps
                if sleeps is not None
                else []
            ).append,
        ),
        content_selection_schema={"name": "content-selection"},
        clustering_schema={"name": "clustering"},
        sentence_fusion_schema={"name": "sentence-fusion"},
        parse_content_selection=(
            parse_content_selection
            or sequential.parse_content_selection_spans
        ),
    )


class SequentialDialogueGuardTests(unittest.TestCase):
    def test_parsers_accept_only_contract_json(self):
        self.assertEqual(
            sequential.parse_content_selection_spans(
                "Document [1]: Alpha"
            ),
            [],
        )
        self.assertEqual(
            sequential.parse_content_selection_spans(
                json.dumps(
                    {
                        "highlights": [
                            {"doc_id": 1, "span_text": "Alpha"}
                        ]
                    }
                )
            ),
            [],
        )
        self.assertEqual(sequential.parse_clusters("[[1]]", 1), [])
        self.assertEqual(
            sequential.parse_clusters(
                'Here is the answer: {"clusters":[[1]]}',
                1,
            ),
            [],
        )
        self.assertEqual(
            sequential.parse_clusters(
                json.dumps({"clusters": [[True]]}),
                1,
            ),
            [],
        )

    def test_every_stage_retries_invalid_contract_and_traces_provider_evidence(self):
        evidence = _ThreadEvidence()
        scripts = {
            "models/test": [
                _response(
                    json.dumps(
                        {
                            "highlights": [
                                {"doc_id": "1", "span_text": "Beta"}
                            ]
                        }
                    )
                ),
                _response(
                    json.dumps(
                        {
                            "highlights": [
                                {"doc_id": "2", "span_text": "Beta"}
                            ]
                        }
                    )
                ),
                _response("[[1]]"),
                _response(json.dumps({"clusters": [[1]]})),
                _response(
                    json.dumps(
                        {
                            "sentence_text": "Beta summary.",
                            "highlight_ids": [True],
                        }
                    )
                ),
                _response(
                    json.dumps(
                        {
                            "sentence_text": "Beta summary.",
                            "highlight_ids": [1],
                        }
                    )
                ),
            ]
        }
        sleeps = []
        gateway = _ScriptedDialogueGateway(scripts, evidence)
        runner = sequential.SequentialDialogueInstanceRunner(
            _dependencies(
                gateway,
                evidence,
                sleeps=sleeps,
            )
        )

        result, usage = runner.run(
            _instance(),
            "Select",
            "Cluster",
            "Fuse",
            "models/test",
            num_retries=2,
        )

        self.assertEqual(result["final_output"], "Beta summary.")
        self.assertEqual(usage["calls"], 6)
        self.assertEqual(
            result["alignments"][0]["highlight_spans"][0][
                "docSpanOffsets"
            ],
            [[0, 4]],
        )
        trace = result["protocol_trace"]
        for stage in ("content_selection", "clustering", "fusion"):
            self.assertEqual(len(trace[stage]), 2)
            self.assertEqual(trace[stage][0]["status"], "error")
            self.assertEqual(trace[stage][1]["status"], "parsed")
            for attempt in trace[stage]:
                self.assertIsInstance(attempt["usage"], dict)
                self.assertEqual(
                    attempt["response_metadata"]["finish_reason"],
                    "STOP",
                )
        self.assertEqual(len(gateway.sessions[0].history), 4)
        self.assertEqual(sleeps, [1, 1, 1])

    def test_max_tokens_is_retried_without_invoking_content_selection_parser(self):
        evidence = _ThreadEvidence()
        parser_calls = []

        def parse_content_selection(raw):
            parser_calls.append(raw)
            return sequential.parse_content_selection_spans(raw)

        valid_cs = json.dumps(
            {
                "highlights": [
                    {"doc_id": "1", "span_text": "Alpha"}
                ]
            }
        )
        scripts = {
            "models/test": [
                _response(
                    valid_cs,
                    prompt=10,
                    metadata={
                        **STOP_METADATA,
                        "finish_reason": "MAX_TOKENS",
                    },
                ),
                _response(valid_cs, prompt=20),
                _response(
                    json.dumps({"clusters": [[1]]}),
                    prompt=30,
                ),
                _response(
                    json.dumps(
                        {
                            "sentence_text": "Alpha summary.",
                            "highlight_ids": [1],
                        }
                    ),
                    prompt=40,
                ),
            ]
        }
        gateway = _ScriptedDialogueGateway(scripts, evidence)
        runner = sequential.SequentialDialogueInstanceRunner(
            _dependencies(
                gateway,
                evidence,
                parse_content_selection=parse_content_selection,
            )
        )

        result, usage = runner.run(
            _instance(),
            "Select",
            "Cluster",
            "Fuse",
            "models/test",
            num_retries=2,
        )

        self.assertEqual(result["final_output"], "Alpha summary.")
        self.assertEqual(parser_calls, [valid_cs])
        self.assertEqual(usage["prompt"], 100)
        self.assertEqual(usage["calls"], 4)
        first_attempt = result["protocol_trace"]["content_selection"][0]
        self.assertEqual(first_attempt["failure_phase"], "generation")
        self.assertIn("MAX_TOKENS", first_attempt["error"])

    def test_missing_exact_document_span_never_produces_null_offsets(self):
        with self.assertRaisesRegex(ValueError, "exact span"):
            sequential.source_span_metadata(
                _instance(),
                "doc-a",
                "Beta",
            )

    def test_repeated_span_text_requires_unambiguous_source_offsets(self):
        instance = _instance()
        instance["documents"][0]["rawDocumentText"] = (
            "Alpha first. Alpha second."
        )
        instance["documents"][0]["documentText"] = [
            "Alpha first.",
            " Alpha second.",
        ]
        instance["documents"][0]["docSentCharIdxToSentIdx"] = [0, 12]

        with self.assertRaisesRegex(ValueError, "ambiguous|multiple"):
            sequential.source_span_metadata(
                instance,
                "doc-a",
                "Alpha",
            )

    def test_source_metadata_discards_mixed_valid_and_parasite_offsets(self):
        instance = _instance()
        instance["set_of_highlights_in_context"] = [
            {
                "documentFile": "doc-a",
                "docSpanText": "Alpha",
                "docSpanOffsets": [[0, 5], [6, 10]],
                "untrusted_marker": True,
            }
        ]

        metadata = sequential.source_span_metadata(
            instance,
            "doc-a",
            "Alpha",
        )

        self.assertEqual(metadata["docSpanOffsets"], [[0, 5]])
        self.assertNotIn("untrusted_marker", metadata)

    def test_per_instance_usage_isolated_under_concurrent_dialogues(self):
        evidence = _ThreadEvidence()
        barrier = threading.Barrier(2)

        def script(prompt_tokens, text):
            return [
                _response(
                    json.dumps(
                        {
                            "highlights": [
                                {"doc_id": "1", "span_text": "Alpha"}
                            ]
                        }
                    ),
                    prompt=prompt_tokens,
                ),
                _response(
                    json.dumps({"clusters": [[1]]}),
                    prompt=prompt_tokens,
                ),
                _response(
                    json.dumps(
                        {
                            "sentence_text": text,
                            "highlight_ids": [1],
                        }
                    ),
                    prompt=prompt_tokens,
                ),
            ]

        gateway = _ScriptedDialogueGateway(
            {
                "models/a": script(2, "A summary."),
                "models/b": script(20, "B summary."),
            },
            evidence,
            barrier=barrier,
        )
        runner = sequential.SequentialDialogueInstanceRunner(
            _dependencies(gateway, evidence)
        )

        def run(model_name):
            return runner.run(
                _instance(),
                "Select",
                "Cluster",
                "Fuse",
                model_name,
                num_retries=1,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(run, "models/a")
            future_b = executor.submit(run, "models/b")
            _, usage_a = future_a.result(timeout=10)
            _, usage_b = future_b.result(timeout=10)

        self.assertEqual(
            usage_a,
            {"prompt": 6, "completion": 3, "cached": 0, "calls": 3},
        )
        self.assertEqual(
            usage_b,
            {
                "prompt": 60,
                "completion": 3,
                "cached": 0,
                "calls": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
