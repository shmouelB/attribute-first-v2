"""Transactional per-instance execution for sequential dialogue."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Mapping

from .dialogue_turns import DialogueTurnDependencies, DialogueTurnExecutor
from .sequential_contracts import (
    concrete_offsets,
    parse_clusters,
    parse_content_selection_spans,
    parse_fusion_output,
    source_span_metadata,
    usage_from_attempt_trace,
)
from ..ports import ChatRequest, DialogueGateway


@dataclass(frozen=True)
class SequentialInstanceDependencies:
    """Provider and parsing boundaries for one sequential conversation."""

    dialogue_gateway: DialogueGateway
    reset_last_call_usage: Callable[[], None]
    get_last_call_usage: Callable[[], Mapping[str, object] | None]
    get_last_call_metadata: Callable[[], Mapping[str, object] | None]
    ensure_parseable_finish_reason: Callable[[object], None]
    stable_value_sha256: Callable[[object], str]
    incomplete_generation_error: type[Exception]
    time_module: object
    content_selection_schema: object
    clustering_schema: object
    sentence_fusion_schema: object
    parse_content_selection: Callable[
        [str],
        list[tuple[str, str]],
    ] = parse_content_selection_spans
    parse_clustering: Callable[[str, int], list[list[int]]] = parse_clusters
    source_metadata: Callable[
        [Mapping[str, object], object, str],
        dict[str, object],
    ] = source_span_metadata


class SequentialDialogueInstanceRunner:
    """Execute CS, clustering, and rollback-isolated fusion."""

    def __init__(self, dependencies: SequentialInstanceDependencies) -> None:
        self._dependencies = dependencies
        self._turn_executor = DialogueTurnExecutor(
            DialogueTurnDependencies(
                dialogue_gateway=dependencies.dialogue_gateway,
                reset_last_call_usage=dependencies.reset_last_call_usage,
                get_last_call_usage=dependencies.get_last_call_usage,
                get_last_call_metadata=dependencies.get_last_call_metadata,
                ensure_parseable_finish_reason=(
                    dependencies.ensure_parseable_finish_reason
                ),
                stable_value_sha256=dependencies.stable_value_sha256,
                incomplete_generation_error=(
                    dependencies.incomplete_generation_error
                ),
                time_module=dependencies.time_module,
            )
        )

    def run(
        self,
        instance: Mapping[str, object],
        content_selection_prompt: str,
        clustering_instruction: str,
        fusion_instruction: str,
        model_name: str,
        num_retries: int = 3,
    ) -> tuple[dict[str, object], dict[str, int]]:
        """Run one transactional conversation and return exact usage."""

        session = self._dependencies.dialogue_gateway.create_chat(
            ChatRequest(model_name=model_name)
        )
        trace: dict[str, object] = {
            "content_selection_raw": None,
            "clustering_raw": None,
            "content_selection": [],
            "clustering": [],
            "fusion": [],
        }

        content_selection, raw = self._turn_executor.execute(
            session,
            content_selection_prompt,
            self._content_selection_parser(instance),
            content_selection_prompt,
            num_retries,
            0,
            response_schema=self._dependencies.content_selection_schema,
            output_max_length=8192,
            model_name=model_name,
            attempt_trace=trace["content_selection"],
        )
        trace["content_selection_raw"] = self._latest_raw(
            trace["content_selection"],
            raw,
        )
        if content_selection is None:
            return self._error("ERROR - no CS spans", trace)
        spans = content_selection["spans"]

        numbered_spans = "\n".join(
            f"{index}. {span['span_text']}"
            for index, span in enumerate(spans, start=1)
        )
        clustering_message = (
            "Now cluster the highlights you selected. "
            + clustering_instruction
            + f"\n\nThe highlighted spans are:\n{numbered_spans}\n\n"
            'Return JSON with a clusters array, e.g. '
            '{"clusters":[[1,2],[3]]}.'
        )
        clustering, raw = self._turn_executor.execute(
            session,
            clustering_message,
            self._clustering_parser(len(spans)),
            clustering_message,
            num_retries,
            0,
            response_schema=self._dependencies.clustering_schema,
            output_max_length=8192,
            model_name=model_name,
            attempt_trace=trace["clustering"],
        )
        trace["clustering_raw"] = self._latest_raw(
            trace["clustering"],
            raw,
        )
        if clustering is None:
            return self._error(
                "ERROR - invalid clustering output",
                trace,
            )
        clusters = clustering["clusters"]

        alignments: list[dict[str, object]] = []
        for sentence_id, cluster in enumerate(clusters, start=1):
            sentence = self._fuse_cluster(
                session=session,
                cluster=cluster,
                spans=spans,
                fusion_instruction=fusion_instruction,
                num_retries=num_retries,
                trace=trace,
                model_name=model_name,
                sentence_id=sentence_id,
            )
            if sentence is None:
                return self._error(
                    f"ERROR - fusion failed for cluster {sentence_id}",
                    trace,
                )
            alignments.append(
                {
                    "sent_id": sentence_id,
                    "sent_text": sentence,
                    "highlights": cluster,
                    "highlight_spans": self._attributed_spans(
                        spans,
                        cluster,
                    ),
                }
            )

        final_output = " ".join(
            str(alignment["sent_text"]) for alignment in alignments
        )
        result = {
            "final_output": final_output or "ERROR - no fusion output",
            "alignments": alignments,
            "protocol_trace": trace,
        }
        return result, usage_from_attempt_trace(trace)

    def _content_selection_parser(
        self,
        instance: Mapping[str, object],
    ) -> Callable[[str, str], dict[str, object]]:
        def parse(raw: str, _prompt: str) -> dict[str, object]:
            parsed_spans = self._dependencies.parse_content_selection(raw)
            if not parsed_spans:
                raise ValueError(
                    "content selection violates the JSON contract"
                )
            return {
                "spans": self._validated_spans(
                    instance,
                    parsed_spans,
                )
            }

        return parse

    def _validated_spans(
        self,
        instance: Mapping[str, object],
        spans: list[tuple[str, str]],
    ) -> list[dict[str, object]]:
        documents = instance.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(
                "content selection has no source documents"
            )

        validated: list[dict[str, object]] = []
        for document_id, span_text in spans:
            if (
                type(document_id) is not str
                or not document_id.isascii()
                or not document_id.isdecimal()
                or str(int(document_id)) != document_id
            ):
                raise ValueError(
                    f"invalid content-selection doc_id {document_id!r}"
                )
            document_index = int(document_id) - 1
            if not 0 <= document_index < len(documents):
                raise ValueError(
                    f"content-selection doc_id {document_id!r} "
                    "is outside the current instance"
                )
            document = documents[document_index]
            if not isinstance(document, Mapping):
                raise ValueError(
                    f"source document {document_id!r} is invalid"
                )
            raw_document = document.get("rawDocumentText")
            if (
                type(raw_document) is not str
                or raw_document.find(span_text) < 0
            ):
                raise ValueError(
                    f"exact span for doc_id {document_id!r} is absent "
                    "from that document"
                )
            document_file = document.get(
                "documentFile",
                document_id,
            )
            metadata = self._dependencies.source_metadata(
                instance,
                document_file,
                span_text,
            )
            if (
                not isinstance(metadata, dict)
                or not concrete_offsets(
                    metadata.get("docSpanOffsets"),
                    raw_document,
                    span_text,
                )
            ):
                raise ValueError(
                    f"exact span for doc_id {document_id!r} has no "
                    "concrete document offsets"
                )
            validated.append(
                {
                    "doc_id": document_id,
                    "span_text": span_text,
                    "document_file": document_file,
                    "source_metadata": deepcopy(metadata),
                }
            )
        return validated

    def _clustering_parser(
        self,
        span_count: int,
    ) -> Callable[[str, str], dict[str, object]]:
        def parse(raw: str, _prompt: str) -> dict[str, object]:
            clusters = self._dependencies.parse_clustering(
                raw,
                span_count,
            )
            if not clusters:
                raise ValueError("clustering violates the JSON contract")
            return {"clusters": clusters}

        return parse

    def _fuse_cluster(
        self,
        *,
        session: object,
        cluster: list[int],
        spans: list[dict[str, object]],
        fusion_instruction: str,
        num_retries: int,
        trace: dict[str, object],
        model_name: str,
        sentence_id: int,
    ) -> str | None:
        base_history_length = len(session.history)
        cluster_texts = "; ".join(
            str(spans[index - 1]["span_text"])
            for index in cluster
        )
        message = (
            fusion_instruction
            + f"\n\nFuse ONLY highlights {cluster} into ONE sentence. "
            f"Their texts: {cluster_texts}\n"
            "Prefix: (none — do NOT refer to or continue from any earlier "
            "sentence).\n"
            'Return JSON: {"sentence_text": <the sentence>, '
            f'"highlight_ids": {cluster}' + "}."
        )
        trace_start = len(trace["fusion"])

        def parse(raw: str, _prompt: str) -> dict[str, object]:
            parsed = parse_fusion_output(raw, cluster)
            if parsed is None:
                raise ValueError(
                    f"fusion violates expected highlight_ids {cluster}"
                )
            return parsed

        try:
            parsed, _ = self._turn_executor.execute(
                session,
                message,
                parse,
                message,
                num_retries,
                0,
                response_schema=self._dependencies.sentence_fusion_schema,
                output_max_length=8192,
                model_name=model_name,
                attempt_trace=trace["fusion"],
            )
        finally:
            if len(session.history) > base_history_length:
                session.history = list(
                    session.history[:base_history_length]
                )
        for attempt in trace["fusion"][trace_start:]:
            attempt["cluster_id"] = sentence_id
            attempt["expected_highlight_ids"] = list(cluster)
        if parsed is None:
            return None
        return str(parsed["sentence_text"])

    @staticmethod
    def _attributed_spans(
        spans: list[dict[str, object]],
        cluster: list[int],
    ) -> list[dict[str, object]]:
        return [
            deepcopy(spans[index - 1]["source_metadata"])
            for index in cluster
        ]

    @staticmethod
    def _latest_raw(
        attempts: object,
        parsed_raw: str | None,
    ) -> str | None:
        if parsed_raw is not None:
            return parsed_raw
        if not isinstance(attempts, list):
            return None
        for attempt in reversed(attempts):
            raw = attempt.get("raw_response")
            if isinstance(raw, str):
                return raw
        return None

    @staticmethod
    def _error(
        message: str,
        trace: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, int]]:
        return {
            "final_output": message,
            "alignments": [],
            "protocol_trace": trace,
        }, usage_from_attempt_trace(trace)


__all__ = [
    "SequentialDialogueInstanceRunner",
    "SequentialInstanceDependencies",
]
