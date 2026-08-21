"""Downstream clustering, ordering, and fusion for derived variants."""

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping

from ..ports import GenerationGateway, GenerationRequest
from ..runtime.retry_policy import DEFAULT_RETRY_DELAY_POLICY


PROMPT_BUDGET_SCOPE = "provider_prompt_usage"
CONTROLLED_STAGE_PROTOCOLS = {
    "MDS": {
        "clustering": {
            "temperature": 0.0,
            "num_retries": 5,
            "prompt_token_budget": 30000,
            "max_output_tokens": 8192,
        },
        "reorder": {
            "temperature": 0.0,
            "num_retries": 5,
            "prompt_token_budget": 30000,
            "max_output_tokens": 2048,
        },
        "fusion": {
            "temperature": 0.1,
            "num_retries": 5,
            "prompt_token_budget": 30000,
            "max_output_tokens": 16384,
        },
    },
    "LFQA": {
        "clustering": {
            "temperature": 0.0,
            "num_retries": 5,
            "prompt_token_budget": 30000,
            "max_output_tokens": 8192,
        },
        "reorder": {
            "temperature": 0.0,
            "num_retries": 5,
            "prompt_token_budget": 30000,
            "max_output_tokens": 2048,
        },
        "fusion": {
            "temperature": 0.3,
            "num_retries": 5,
            "prompt_token_budget": 30000,
            "max_output_tokens": 8192,
        },
    },
}
CLUSTER_INSTR = (
    "You are given a numbered list of highlighted spans selected from source "
    "documents. Group the highlights that should be fused into the SAME "
    "summary sentence. Each group (cluster) becomes one sentence. Return JSON "
    "with a 'clusters' array; each cluster is a list of highlight numbers, "
    'e.g. {"clusters":[[1,2],[3],[4,5]]}. Cover every highlight exactly once.'
)
REORDER_INSTR = (
    "You are given clusters of highlights; each cluster will become one "
    "summary sentence, and they are currently in an arbitrary order. Reorder "
    "the clusters so the resulting summary flows coherently and groups "
    "related topics together (avoid jumping between topics). Return JSON with "
    "an 'order' array of the original cluster numbers, e.g. "
    '{"order":[3,1,2]}.'
)
FUSION_INSTR = (
    "You are given numbered highlighted spans and a SENTENCE PLAN (an ordered "
    "list of clusters; each cluster lists the highlight numbers to fuse into "
    "that sentence). Write the summary by producing ONE sentence per cluster, "
    "IN THE GIVEN ORDER, fusing all and only that cluster's highlights. Output "
    "JSON per the schema: sentences[] with sentence_id (1-based, in order), "
    "sentence_text, and highlight_ids (the cluster's highlight numbers)."
)
LFQA_CLUSTER_INSTR = (
    CLUSTER_INSTR
    + " Use the question to decide which evidence belongs in the same answer "
    "sentence."
)
LFQA_REORDER_INSTR = (
    REORDER_INSTR
    + " Use the question to order the clusters as a direct, coherent answer."
)
LFQA_FUSION_INSTR = (
    "You are given a question, numbered highlighted evidence spans, and a "
    "SENTENCE PLAN (an ordered list of clusters; each cluster lists the "
    "highlight numbers to fuse into that answer sentence). Answer the question "
    "by producing ONE sentence per cluster, IN THE GIVEN ORDER, fusing all and "
    "only that cluster's highlights. Output JSON per the schema: sentences[] "
    "with sentence_id (1-based, in order), sentence_text, and highlight_ids "
    "(the cluster's highlight numbers)."
)


@dataclass(frozen=True)
class ProtocolDefinition:
    """Immutable inputs needed to materialize an effective protocol."""

    stage_protocols: Mapping[str, Mapping[str, Mapping[str, Any]]]
    response_schemas: Mapping[str, Any]
    prompt_budget_scope: str = PROMPT_BUDGET_SCOPE


class ProtocolFactory:
    """Build the complete, hash-ready downstream treatment declaration."""

    def __init__(self, definition, stable_value_sha256):
        self.definition = definition
        self.stable_value_sha256 = stable_value_sha256

    @staticmethod
    def _instructions(setting):
        if setting == "LFQA":
            return {
                "clustering": LFQA_CLUSTER_INSTR,
                "reorder": LFQA_REORDER_INSTR,
                "fusion": LFQA_FUSION_INSTR,
            }
        return {
            "clustering": CLUSTER_INSTR,
            "reorder": REORDER_INSTR,
            "fusion": FUSION_INSTR,
        }

    def build(self, model, *, setting="MDS"):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if setting not in self.definition.stage_protocols:
            raise ValueError(
                "setting must be one of "
                + ", ".join(sorted(self.definition.stage_protocols))
            )
        parameters = self.definition.stage_protocols[setting]
        instructions = self._instructions(setting)
        return {
            "schema_version": 2,
            "setting": setting,
            "model": model,
            "transport": "independent_gemini_role_calls",
            "dialogue": False,
            "explicit_context_cache": False,
            "prompt_budget_scope": self.definition.prompt_budget_scope,
            "randomness": {
                "downstream_uses_demonstrations": False,
                "demonstration_selection_seed": None,
                "provider_generation_seed": None,
                "provider_generation_seed_supported": False,
                "stage_temperatures": {
                    stage: values["temperature"]
                    for stage, values in parameters.items()
                },
            },
            "retry_semantics": (
                "repeat_identical_stage_request_then_terminal_error"
            ),
            "retry_backoff": {
                "rate_limit_error_marker": "429",
                "rate_limit_seconds": 60,
                "other_error_seconds": 1,
                "applies_to_failure_phases": [
                    "transport",
                    "provider_response",
                    "generation",
                    "parse",
                ],
                "sleep_after_final_attempt": False,
            },
            "fallback_policy": "terminal_error",
            "stage_order": ["clustering", "reorder", "fusion"],
            "stage_parameters": deepcopy(parameters),
            "system_instructions": {
                stage: {
                    "value": instruction,
                    "sha256": self.stable_value_sha256(instruction),
                }
                for stage, instruction in instructions.items()
            },
            "response_schemas": {
                stage: {
                    "value": schema,
                    "sha256": self.stable_value_sha256(schema),
                }
                for stage, schema in self.definition.response_schemas.items()
            },
        }


def parse_clusters(output, highlight_count):
    """Return clusters only when every highlight is covered exactly once."""
    try:
        raw_object = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return []
    if (
        not isinstance(raw_object, dict)
        or set(raw_object) != {"clusters"}
    ):
        return []
    clusters = raw_object["clusters"]
    if (
        not isinstance(clusters, list)
        or not clusters
        or highlight_count < 1
    ):
        return []
    if any(
        not isinstance(cluster, list) or not cluster
        for cluster in clusters
    ):
        return []
    flattened = [
        highlight_id
        for cluster in clusters
        for highlight_id in cluster
    ]
    if any(type(highlight_id) is not int for highlight_id in flattened):
        return []
    if sorted(flattened) != list(range(1, highlight_count + 1)):
        return []
    return clusters


def parse_reorder(output, cluster_count):
    """Return an order only when it is a complete cluster permutation."""
    try:
        raw = json.loads(output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(raw, dict) or set(raw) != {"order"}:
        return None
    order = raw["order"]
    if not isinstance(order, list) or any(
        type(value) is not int for value in order
    ):
        return None
    if sorted(order) != list(range(1, cluster_count + 1)):
        return None
    return order


def validate_fusion_plan(parsed, clusters):
    """Prove a one-to-one ordered mapping between clusters and sentences."""
    alignments = parsed.get("alignments") if isinstance(parsed, dict) else None
    if not isinstance(alignments, list) or len(alignments) != len(clusters):
        raise ValueError(
            "fusion plan mismatch: one attributed sentence is required per "
            "cluster"
        )
    for sentence_id, (alignment, cluster) in enumerate(
        zip(alignments, clusters),
        1,
    ):
        if alignment.get("sent_id") != sentence_id:
            raise ValueError(
                "fusion plan mismatch: sentence IDs must follow cluster order"
            )
        actual = alignment.get("highlights")
        if (
            not isinstance(actual, list)
            or len(actual) != len(set(actual))
            or sorted(actual) != sorted(cluster)
        ):
            raise ValueError(
                f"fusion plan mismatch at sentence {sentence_id}: "
                f"expected cluster {cluster}, got {actual}"
            )
        if not str(alignment.get("sent_text", "")).strip():
            raise ValueError(
                f"fusion plan mismatch: sentence {sentence_id} is empty"
            )
    return True


@dataclass(frozen=True)
class StageRequest:
    """One complete schema-bound provider request."""

    stage: str
    system_instruction: str
    user_message: str
    model: str
    response_schema: Any
    max_output_tokens: int
    prompt_token_budget: int
    prompt_budget_scope: str
    temperature: float
    num_retries: int
    parser: Callable[[str], Any]


@dataclass(frozen=True)
class StageExecutionDependencies:
    """Mutable boundaries injected into a deterministic stage executor."""

    generation_gateway: GenerationGateway
    stable_value_sha256: Callable[[Any], str]
    reset_last_call_usage: Callable[[], None]
    get_last_call_usage: Callable[[], Any]
    get_last_call_metadata: Callable[[], Any]
    ensure_parseable_finish_reason: Callable[[Any], None]
    incomplete_generation_error: type
    sleep: Callable[[float], None]
    prompt_budget_scope: str = PROMPT_BUDGET_SCOPE


class StageExecutor:
    """Execute identical attempts while recording every call boundary."""

    def __init__(self, dependencies):
        self.dependencies = dependencies

    def _attempt_record(self, request, attempt_number, role_payload):
        stable_hash = self.dependencies.stable_value_sha256
        return {
            "attempt": attempt_number,
            "stage": request.stage,
            "application_role_payload": deepcopy(role_payload),
            "application_role_payload_sha256": stable_hash(role_payload),
            "response_schema": deepcopy(request.response_schema),
            "response_schema_sha256": stable_hash(
                request.response_schema
            ),
            "effective": {
                "model": request.model,
                "temperature": request.temperature,
                "num_retries": request.num_retries,
                "max_output_tokens": request.max_output_tokens,
                "prompt_token_budget": request.prompt_token_budget,
                "prompt_budget_scope": request.prompt_budget_scope,
                "downstream_demonstration_count": 0,
                "demonstration_selection_seed": None,
                "provider_generation_seed": None,
                "provider_generation_seed_supported": False,
            },
            "raw_response": None,
            "usage": None,
            "parser_outcome": {"status": "not_run"},
        }

    @staticmethod
    def _retry_seconds(error):
        return DEFAULT_RETRY_DELAY_POLICY.delay_seconds(error)

    def execute(self, request):
        if request.prompt_budget_scope != self.dependencies.prompt_budget_scope:
            raise ValueError(
                "derived prompt budget scope must be provider_prompt_usage"
            )
        attempts = []
        role_payload = {
            "system_instruction": request.system_instruction,
            "contents": [
                {"role": "user", "parts": [request.user_message]}
            ],
        }
        for attempt_number in range(1, request.num_retries + 1):
            attempt = self._attempt_record(
                request,
                attempt_number,
                role_payload,
            )
            self.dependencies.reset_last_call_usage()
            try:
                raw_response = self.dependencies.generation_gateway.generate(
                    GenerationRequest(
                        model_name=request.model,
                        prompt=request.user_message,
                        output_max_length=request.max_output_tokens,
                        temperature=request.temperature,
                        response_schema=request.response_schema,
                        contents=[
                            {
                                "role": "user",
                                "parts": [request.user_message],
                            }
                        ],
                        system_instruction=request.system_instruction,
                    )
                )
                attempt["raw_response"] = raw_response
                attempt["usage"] = self.dependencies.get_last_call_usage()
                attempt[
                    "response_metadata"
                ] = self.dependencies.get_last_call_metadata()
                usage = attempt["usage"]
                if (
                    isinstance(usage, dict)
                    and usage.get("prompt_token_count", 0)
                    >= request.prompt_token_budget
                ):
                    attempt["parser_outcome"] = {
                        "status": "error",
                        "error": (
                            "PromptBudgetExceeded: provider prompt usage "
                            f"{usage['prompt_token_count']} is not below "
                            f"{request.prompt_token_budget}"
                        ),
                    }
                    attempt["status"] = "error"
                    attempt["failure_phase"] = "protocol_budget"
                    attempt["error"] = attempt["parser_outcome"]["error"]
                    attempts.append(attempt)
                    return None, attempts
            except (KeyError, AttributeError, TypeError):
                raise
            except Exception as exc:
                attempt["usage"] = self.dependencies.get_last_call_usage()
                metadata = self.dependencies.get_last_call_metadata()
                attempt["response_metadata"] = metadata
                attempt["status"] = "error"
                attempt["failure_phase"] = (
                    "provider_response"
                    if isinstance(metadata, dict)
                    else "transport"
                )
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                if attempt_number < request.num_retries:
                    wait_seconds = self._retry_seconds(exc)
                    attempt["retry_backoff_seconds"] = wait_seconds
                attempts.append(attempt)
                if attempt_number < request.num_retries:
                    self.dependencies.sleep(wait_seconds)
                continue

            try:
                self.dependencies.ensure_parseable_finish_reason(
                    attempt["response_metadata"]
                )
                parsed = request.parser(raw_response)
                attempt["parser_outcome"] = {
                    "status": "parsed",
                    "value": deepcopy(parsed),
                }
                attempt["status"] = "parsed"
                attempts.append(attempt)
                return parsed, attempts
            except (
                self.dependencies.incomplete_generation_error,
                ValueError,
            ) as exc:
                attempt["parser_outcome"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                attempt["status"] = "error"
                attempt["failure_phase"] = (
                    "generation"
                    if isinstance(
                        exc,
                        self.dependencies.incomplete_generation_error,
                    )
                    else "parse"
                )
                attempt["error"] = f"{type(exc).__name__}: {exc}"
                if attempt_number < request.num_retries:
                    wait_seconds = self._retry_seconds(exc)
                    attempt["retry_backoff_seconds"] = wait_seconds
                attempts.append(attempt)
                if attempt_number < request.num_retries:
                    self.dependencies.sleep(wait_seconds)
        return None, attempts


def terminal_result(stage, message, stage_traces, **metadata):
    """Represent a per-example failure without dropping its population row."""
    plan_metadata = {
        "fallback_policy": "terminal_error",
        "terminal_stage": stage,
        "stage_traces": stage_traces,
        "clusters_initial": metadata.pop("clusters_initial", None),
        "reorder_order": metadata.pop("reorder_order", None),
        "clusters_final": metadata.pop("clusters_final", None),
        **metadata,
    }
    plan_metadata["fusion_attempts"] = stage_traces.get("fusion", [])
    return {
        "final_output": f"ERROR - {stage}: {message}",
        "alignments": [],
        "plan_metadata": plan_metadata,
    }


def source_upstream_failure(instance):
    """Return an explicit upstream failure or ``None`` for runnable input."""
    skipped_reason = instance.get("skipped_reason")
    upstream_error = instance.get("upstream_error")
    has_explicit_error = (
        isinstance(upstream_error, str)
        and upstream_error.strip().startswith("ERROR")
    )
    if not skipped_reason and not has_explicit_error:
        return None
    if not has_explicit_error:
        upstream_error = f"ERROR - upstream stage skipped: {skipped_reason}"
    return {
        "skipped_reason": str(skipped_reason or "upstream_error"),
        "upstream_error": upstream_error,
    }


@dataclass(frozen=True)
class PlannedInstanceDependencies:
    """Collaborators used by one derived-instance pipeline."""

    effective_protocol: Callable[..., dict]
    stable_value_sha256: Callable[[Any], str]
    execute_stage: Callable[..., tuple[Any, list]]
    parse_clusters: Callable[[str, int], list]
    parse_reorder: Callable[[str, int], Any]
    parse_structured_fusion: Callable[..., dict]
    validate_fusion_plan: Callable[[dict, list], bool]
    terminal_result: Callable[..., dict]
    source_upstream_failure: Callable[[dict], Any]
    clustering_schema: Any
    reorder_schema: Any
    fusion_schema: Any


class PlannedInstanceRunner:
    """Run clustering, reorder, and fusion for one validated source row."""

    def __init__(self, dependencies):
        self.dependencies = dependencies

    def _terminal(self, stage, message, traces, protocol_sha256, **metadata):
        return self.dependencies.terminal_result(
            stage,
            message,
            traces,
            protocol_sha256=protocol_sha256,
            **metadata,
        )

    def _input_context(self, instance, setting):
        highlights = instance.get("set_of_highlights_in_context", []) or []
        if not highlights:
            raise ValueError("no highlights")
        if any(not isinstance(highlight, dict) for highlight in highlights):
            raise TypeError("every highlight must be an object")
        texts = [highlight.get("docSpanText", "") for highlight in highlights]
        if any(
            not isinstance(text, str) or not text.strip() for text in texts
        ):
            raise TypeError(
                "every highlight must have non-empty docSpanText"
            )
        question_prefix = ""
        if setting == "LFQA":
            question = instance.get("query")
            if not isinstance(question, str) or not question.strip():
                raise LookupError("LFQA instance has no non-empty query")
            question_prefix = f"Question:\n{question.strip()}\n\n"
        highlight_list = "\n".join(
            f"{index + 1}. {text}" for index, text in enumerate(texts)
        )
        return highlights, texts, highlight_list, question_prefix

    def _stage(self, name, protocol, user_message, model, schema, parser):
        parameters = protocol["stage_parameters"][name]
        return self.dependencies.execute_stage(
            stage=name,
            system_instruction=protocol["system_instructions"][name]["value"],
            user_message=user_message,
            model=model,
            response_schema=schema,
            max_output_tokens=parameters["max_output_tokens"],
            prompt_token_budget=parameters["prompt_token_budget"],
            prompt_budget_scope=protocol["prompt_budget_scope"],
            temperature=parameters["temperature"],
            num_retries=parameters["num_retries"],
            parser=parser,
        )

    def _fusion_parser(self, raw_response, fusion_user, clusters):
        try:
            structured = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                "fusion response must be a JSON object matching the response "
                "schema"
            ) from exc
        if not isinstance(structured, dict):
            raise ValueError("fusion response must be a JSON object")
        sentences = structured.get("sentences")
        if not isinstance(sentences, list):
            raise ValueError("fusion response must contain a sentences array")
        if len(sentences) != len(clusters):
            raise ValueError(
                "fusion response must contain exactly one sentence per cluster"
            )
        for index, (sentence, cluster) in enumerate(
            zip(sentences, clusters),
            start=1,
        ):
            self._validate_sentence(sentence, cluster, index)
        parsed = self.dependencies.parse_structured_fusion(
            raw_response,
            prompt=fusion_user,
        )
        self.dependencies.validate_fusion_plan(parsed, clusters)
        return parsed

    @staticmethod
    def _validate_sentence(sentence, cluster, index):
        if not isinstance(sentence, dict):
            raise ValueError(f"fusion sentence {index} must be an object")
        if type(sentence.get("sentence_id")) is not int:
            raise ValueError(
                f"fusion sentence {index} sentence_id must be an integer"
            )
        if sentence["sentence_id"] != index:
            raise ValueError(
                "fusion sentence IDs must follow the cluster order"
            )
        sentence_text = sentence.get("sentence_text")
        if not isinstance(sentence_text, str) or not sentence_text.strip():
            raise ValueError(
                f"fusion sentence {index} sentence_text must be a non-empty "
                "string"
            )
        highlight_ids = sentence.get("highlight_ids")
        if not isinstance(highlight_ids, list) or any(
            type(highlight_id) is not int for highlight_id in highlight_ids
        ):
            raise ValueError(
                f"fusion sentence {index} highlight_ids must be an integer "
                "array"
            )
        if (
            len(highlight_ids) != len(set(highlight_ids))
            or sorted(highlight_ids) != sorted(cluster)
        ):
            raise ValueError(
                f"fusion sentence {index} must cite exactly cluster {cluster}"
            )

    def run(self, instance, model, *, setting="MDS"):
        protocol = self.dependencies.effective_protocol(
            model,
            setting=setting,
        )
        protocol_sha256 = self.dependencies.stable_value_sha256(protocol)
        traces = empty_stage_traces()
        upstream_failure = self.dependencies.source_upstream_failure(instance)
        if upstream_failure is not None:
            result = self._terminal(
                "upstream",
                upstream_failure["upstream_error"]
                .removeprefix("ERROR - ")
                .strip(),
                traces,
                protocol_sha256,
            )
            result["upstream_skipped_reason"] = upstream_failure[
                "skipped_reason"
            ]
            return result

        try:
            highlights, texts, highlight_list, question_prefix = (
                self._input_context(instance, setting)
            )
        except (ValueError, TypeError, LookupError) as exc:
            return self._terminal(
                "input",
                str(exc),
                traces,
                protocol_sha256,
            )

        clustering_user = (
            f"{question_prefix}The highlighted spans are:\n{highlight_list}"
        )

        def clustering_parser(raw_response):
            clusters = self.dependencies.parse_clusters(
                raw_response,
                len(texts),
            )
            if not clusters:
                raise ValueError(
                    "clustering response must cover every highlight exactly "
                    "once"
                )
            return clusters

        clusters, traces["clustering"] = self._stage(
            "clustering",
            protocol,
            clustering_user,
            model,
            self.dependencies.clustering_schema,
            clustering_parser,
        )
        if not clusters:
            attempts = protocol["stage_parameters"]["clustering"][
                "num_retries"
            ]
            return self._terminal(
                "clustering",
                f"failed after {attempts} identical attempt(s)",
                traces,
                protocol_sha256,
            )
        clusters_initial = deepcopy(clusters)

        cluster_block = "\n".join(
            f"Cluster {index + 1}: highlights {cluster}"
            for index, cluster in enumerate(clusters)
        )
        reorder_user = (
            f"{question_prefix}The highlighted spans are:\n{highlight_list}"
            f"\n\nCLUSTERS TO REORDER:\n{cluster_block}"
        )

        def reorder_parser(raw_response):
            order = self.dependencies.parse_reorder(
                raw_response,
                len(clusters),
            )
            if order is None:
                raise ValueError(
                    "reorder response must be a permutation of every "
                    "cluster ID"
                )
            return order

        order, traces["reorder"] = self._stage(
            "reorder",
            protocol,
            reorder_user,
            model,
            self.dependencies.reorder_schema,
            reorder_parser,
        )
        if order is None:
            attempts = protocol["stage_parameters"]["reorder"]["num_retries"]
            return self._terminal(
                "reorder",
                f"failed after {attempts} identical attempt(s)",
                traces,
                protocol_sha256,
                clusters_initial=clusters_initial,
            )
        clusters = [clusters[index - 1] for index in order]

        plan = "\n".join(
            f"sentence {index + 1}: highlights {cluster}"
            for index, cluster in enumerate(clusters)
        )
        fusion_user = (
            f"{question_prefix}The highlighted spans are:\n{highlight_list}"
            f"\n\nSENTENCE PLAN (in order):\n{plan}"
        )
        fusion_parser = lambda raw: self._fusion_parser(  # noqa: E731
            raw,
            fusion_user,
            clusters,
        )
        parsed, traces["fusion"] = self._stage(
            "fusion",
            protocol,
            fusion_user,
            model,
            self.dependencies.fusion_schema,
            fusion_parser,
        )
        if not parsed or not parsed.get("alignments"):
            attempts = protocol["stage_parameters"]["fusion"]["num_retries"]
            return self._terminal(
                "fusion",
                f"failed after {attempts} identical attempt(s)",
                traces,
                protocol_sha256,
                clusters_initial=clusters_initial,
                reorder_order=order,
                clusters_final=deepcopy(clusters),
            )

        for alignment in parsed["alignments"]:
            alignment["highlight_spans"] = [
                highlights[index - 1]
                for index in alignment.get("highlights", [])
                if 1 <= index <= len(highlights)
            ]
        plan_metadata = {
            "fallback_policy": "terminal_error",
            "terminal_stage": None,
            "protocol_sha256": protocol_sha256,
            "stage_traces": traces,
            "clustering_raw_output": traces["clustering"][-1][
                "raw_response"
            ],
            "clusters_initial": clusters_initial,
            "clustering_fallback": False,
            "reorder_raw_output": traces["reorder"][-1]["raw_response"],
            "reorder_order": order,
            "reorder_fallback": False,
            "clusters_final": deepcopy(clusters),
            "fusion_attempts": traces["fusion"],
        }
        return {
            "final_output": parsed["final_output"],
            "alignments": parsed["alignments"],
            "full_model_response": parsed.get("full_model_response"),
            "n_clusters": len(clusters),
            "plan_metadata": plan_metadata,
        }


def empty_stage_traces():
    return {"clustering": [], "reorder": [], "fusion": []}


def trace_usage_summary(stage_traces):
    """Summarize provider counters stored at every downstream boundary."""
    summary = {
        "attempt_count": 0,
        "response_count": 0,
        "usage_record_count": 0,
        "prompt_token_count": 0,
        "candidates_token_count": 0,
        "cached_content_token_count": 0,
        "total_token_count": 0,
        "total_token_count_record_count": 0,
    }
    if not isinstance(stage_traces, dict):
        return summary
    for attempts in stage_traces.values():
        if not isinstance(attempts, list):
            continue
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            summary["attempt_count"] += 1
            if attempt.get("raw_response") is not None:
                summary["response_count"] += 1
            usage = attempt.get("usage")
            if not isinstance(usage, dict):
                continue
            summary["usage_record_count"] += 1
            for counter in (
                "prompt_token_count",
                "candidates_token_count",
                "cached_content_token_count",
            ):
                value = usage.get(counter)
                if type(value) is int:
                    summary[counter] += value
            provider_total = usage.get("total_token_count")
            if type(provider_total) is int:
                summary["total_token_count"] += provider_total
                summary["total_token_count_record_count"] += 1
    return summary


def all_results_trace_usage(results):
    """Aggregate the per-result summaries without reading global counters."""
    aggregate = trace_usage_summary({})
    for result in results.values():
        result_usage = result.get("usage_summary")
        if not isinstance(result_usage, dict):
            continue
        for counter in aggregate:
            value = result_usage.get(counter)
            if type(value) is int:
                aggregate[counter] += value
    return aggregate
