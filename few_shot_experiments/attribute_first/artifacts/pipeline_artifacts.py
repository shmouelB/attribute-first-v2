"""Pipeline artifact aggregation, metadata, and immutable provenance."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .standard_provenance import (
    StandardPipelineProvenanceRepository,
)
from .shared_content_runtime import SharedContentSelectionRuntime
from .dialogue_usage import DialogueContentSelectionUsageRepository
from .token_usage import PipelineTokenUsageAggregator


@dataclass(frozen=True)
class ArtifactDependencies:
    """Patch-aware writes and domain values supplied by the legacy façade."""

    atomic_write_json: Callable[[Any, Any], None]
    stable_value_sha256: Callable[[Any], str]
    summarize_response_metadata: Callable[[Any], Mapping[str, Any]]
    subtask_schemas: Mapping[str, Any]
    source_file: str
    resolved_input_path: Callable
    load_fixed_population: Callable
    capture_provenance_file: Callable

    @classmethod
    def from_namespace(cls, namespace):
        """Capture current façade values for compatibility with test patches."""
        return cls(
            atomic_write_json=namespace["atomic_write_json"],
            stable_value_sha256=namespace["stable_value_sha256"],
            summarize_response_metadata=namespace[
                "summarize_response_metadata"
            ],
            subtask_schemas=namespace["SUBTASK_SCHEMAS"],
            source_file=namespace["__file__"],
            resolved_input_path=namespace["_resolved_input_path"],
            load_fixed_population=namespace["_load_fixed_population"],
            capture_provenance_file=namespace[
                "_capture_provenance_file"
            ],
        )


class PipelineArtifactService:
    """Compatibility facade for run evidence and standard provenance."""

    def __init__(self, dependencies):
        self._dependencies = dependencies
        self._provenance_repository = (
            StandardPipelineProvenanceRepository(
                dependencies,
                git_source_state=lambda root, specs: (
                    self._git_source_state(root, specs)
                ),
                started_at=lambda outdir: self._started_at(outdir),
            )
        )
        self._token_usage_aggregator = PipelineTokenUsageAggregator(
            dependencies.atomic_write_json
        )
        self._shared_content_selection = SharedContentSelectionRuntime(
            dependencies.atomic_write_json
        )
        self._dialogue_cs_usage = (
            DialogueContentSelectionUsageRepository(
                dependencies.atomic_write_json
            )
        )

    def log_stage_health(self, pipeline_format_path, label):
        """Log a compact health summary for a stage artifact."""
        if not os.path.exists(pipeline_format_path):
            logging.warning(
                f"[health:{label}] MISSING {pipeline_format_path}"
            )
            return
        with open(
            pipeline_format_path,
            "r",
            encoding="utf-8",
        ) as source:
            text = source.read().strip()
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                if "unique_id" in payload:
                    items = [payload]
                else:
                    items = list(payload.values())
            else:
                items = [payload]
        except json.JSONDecodeError:
            items = []
            for line in text.splitlines():
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        items = [item for item in items if isinstance(item, dict)]
        instance_count = len(items)
        highlight_count = 0
        none_document = 0
        empty_instances = 0
        none_scu = 0
        for item in items:
            highlights = item.get("set_of_highlights_in_context") or []
            if not highlights:
                empty_instances += 1
            for highlight in highlights:
                highlight_count += 1
                if highlight.get("documentFile") is None:
                    none_document += 1
                if highlight.get("scuSentence") is None:
                    none_scu += 1
        logging.info(
            f"[health:{label}] instances={instance_count} "
            f"highlights={highlight_count} "
            f"empty_instances={empty_instances} "
            f"None_documentFile={none_document} "
            f"None_scuSentence={none_scu}"
        )

    def persist_token_usage(self, outdir):
        """Persist one whole-pipeline usage total and stage breakdown."""
        return self._token_usage_aggregator.persist(outdir)

    def persist_response_metadata(self, outdir, dialogue_mode):
        """Persist backend and termination evidence for the whole cell."""
        root = Path(outdir)
        attempts = []
        if dialogue_mode:
            call_path = root / "dialogue_calls.jsonl"
            if not call_path.is_file():
                raise FileNotFoundError("dialogue_calls.jsonl is missing")
            for line in call_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if line.strip():
                    attempts.append(json.loads(line))
        else:
            result_paths = [root / "results.json"]
            intermediate = root / "itermediate_results"
            if intermediate.is_dir():
                result_paths.extend(
                    sorted(intermediate.glob("*/results.json"))
                )
            for result_path in result_paths:
                if not result_path.is_file():
                    continue
                results = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                if not isinstance(results, dict):
                    raise ValueError(
                        f"{result_path}: results must be an object"
                    )
                for result in results.values():
                    if not isinstance(result, dict):
                        continue
                    trace = result.get("attempt_trace")
                    if isinstance(trace, list):
                        attempts.extend(trace)

        summary = self._dependencies.summarize_response_metadata(attempts)
        self._dependencies.atomic_write_json(
            root / "response_metadata.json",
            summary,
        )

        args_path = root / "args.json"
        args_snapshot = json.loads(
            args_path.read_text(encoding="utf-8")
        )
        if not isinstance(args_snapshot, dict):
            raise ValueError("args.json must be an object")
        args_snapshot["observed_response_metadata"] = summary
        self._dependencies.atomic_write_json(args_path, args_snapshot)

        provenance_path = root / "pipeline_provenance.json"
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        if not isinstance(provenance, dict):
            raise ValueError("pipeline_provenance.json must be an object")
        provenance["observed_response_metadata"] = summary
        provenance["run"]["completed_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._dependencies.atomic_write_json(
            provenance_path,
            provenance,
        )
        return summary

    @staticmethod
    def resolved_input_path(config, args, key, default):
        return StandardPipelineProvenanceRepository.resolved_input_path(
            config,
            args,
            key,
            default,
        )

    @staticmethod
    def load_fixed_population(input_path, max_examples, payload=None):
        return (
            StandardPipelineProvenanceRepository.load_fixed_population(
                input_path,
                max_examples,
                payload=payload,
            )
        )

    @staticmethod
    def capture_provenance_file(source_path, outdir, relative_path):
        return (
            StandardPipelineProvenanceRepository.capture_provenance_file(
                source_path,
                outdir,
                relative_path,
            )
        )

    def persist_provenance(self, args, full_configs, outdir):
        return self._provenance_repository.persist(
            args,
            full_configs,
            outdir,
        )

    def prepare_shared_content_selection(self, args, outdir):
        """Validate and snapshot a requested producer before generation."""
        source_root = getattr(
            args,
            "shared_content_selection_source",
            None,
        )
        if source_root is None:
            return None
        return self._shared_content_selection.prepare(
            source_root=source_root,
            consumer_root=outdir,
        )

    def persist_dialogue_content_selection_usage(self, outdir):
        """Publish a shareable CS-only ledger after aggregate accounting."""
        return self._dialogue_cs_usage.persist(outdir)

    @staticmethod
    def _git_source_state(repository_root, source_specs):
        return StandardPipelineProvenanceRepository.git_source_state(
            repository_root,
            source_specs,
        )

    @staticmethod
    def _started_at(outdir):
        return StandardPipelineProvenanceRepository.started_at(outdir)


__all__ = [
    "ArtifactDependencies",
    "PipelineArtifactService",
]
