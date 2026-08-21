"""Runtime equivalence and provenance for shared content selection."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Callable

from .shared_content_selection import (
    PRODUCER_PROVENANCE_NAME,
    SOURCE_BUNDLE_DRIFT_RELATIONS,
    SharedContentSelectionReference,
    SharedContentSelectionReferenceError,
    SharedContentSelectionRepository,
)


class ContentSelectionEquivalence:
    """Build the exact provider-affecting contract for one CS stage."""

    @classmethod
    def from_provenance(cls, provenance: object) -> dict[str, object]:
        if not isinstance(provenance, dict):
            raise SharedContentSelectionReferenceError(
                "pipeline provenance must be an object"
            )
        run = cls._mapping(provenance, "run")
        population = cls._mapping(provenance, "input")
        source = cls._mapping(provenance, "source")
        runtime = cls._mapping(provenance, "runtime")
        stages = provenance.get("stages")
        if not isinstance(stages, list):
            raise SharedContentSelectionReferenceError(
                "pipeline provenance stages must be an array"
            )
        content_stages = [
            stage
            for stage in stages
            if (
                isinstance(stage, dict)
                and stage.get("subtask") == "content_selection"
            )
        ]
        if len(content_stages) != 1:
            raise SharedContentSelectionReferenceError(
                "pipeline provenance must contain one content-selection "
                "stage"
            )
        stage = content_stages[0]
        config_file = cls._mapping(stage, "config_file")
        factors = cls._mapping(run, "factors")
        prompt_inputs = provenance.get("prompt_inputs")
        if not isinstance(prompt_inputs, list) or not prompt_inputs:
            raise SharedContentSelectionReferenceError(
                "pipeline provenance prompt inputs are missing"
            )
        prompt_hashes = sorted(
            cls._sha256(record, "prompt input")
            for record in prompt_inputs
        )
        return {
            "schema_version": 1,
            "stage": "content_selection",
            "population": {
                "setting": cls._string(run, "setting"),
                "split": cls._string(run, "split"),
                "dataset_sha256": cls._sha256(
                    population,
                    "input population",
                ),
                "count": population.get("count"),
                "unique_ids_sha256": cls._sha256_field(
                    population,
                    "unique_ids_sha256",
                    "input unique IDs",
                ),
                "max_examples": population.get("max_examples"),
            },
            "request": {
                "config_sha256": cls._sha256(
                    config_file,
                    "content-selection config",
                ),
                "effective_config": deepcopy(
                    stage.get("effective_config")
                ),
                "response_schema": deepcopy(
                    stage.get("effective_response_schema")
                ),
                "randomness": deepcopy(
                    stage.get("effective_randomness")
                ),
                "prompt_sha256": prompt_hashes,
                "demonstration_mode": cls._string(
                    factors,
                    "demonstrations",
                ),
                "transport": cls._string(run, "transport"),
                "protocol": deepcopy(provenance.get("protocol")),
            },
            "implementation": {
                "source_bundle_sha256": cls._sha256_field(
                    source,
                    "bundle_sha256",
                    "generation source bundle",
                ),
                "runtime": deepcopy(runtime),
            },
        }

    @staticmethod
    def digest(value: object) -> str:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def compatible_for_reuse(
        cls,
        producer: dict,
        consumer: dict,
        *,
        producer_canonical_id: object,
        consumer_canonical_id: object,
    ) -> bool:
        if producer == consumer:
            return True
        if (
            producer_canonical_id,
            consumer_canonical_id,
        ) not in SOURCE_BUNDLE_DRIFT_RELATIONS:
            return False
        return cls._without_source_bundle(
            producer
        ) == cls._without_source_bundle(consumer)

    @staticmethod
    def _without_source_bundle(value: dict) -> dict:
        normalized = deepcopy(value)
        implementation = normalized.get("implementation")
        if isinstance(implementation, dict):
            implementation.pop("source_bundle_sha256", None)
        return normalized

    @classmethod
    def generated_execution(cls, provenance: dict) -> dict:
        equivalence = cls.from_provenance(provenance)
        digest = cls.digest(equivalence)
        return {
            "schema_version": 1,
            "mode": "generated",
            "execution_id": digest,
            "equivalence": {
                "value": equivalence,
                "sha256": digest,
            },
        }

    @staticmethod
    def _mapping(parent: dict, key: str) -> dict:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise SharedContentSelectionReferenceError(
                f"pipeline provenance {key} must be an object"
            )
        return value

    @staticmethod
    def _string(parent: dict, key: str) -> str:
        value = parent.get(key)
        if not isinstance(value, str) or not value:
            raise SharedContentSelectionReferenceError(
                f"pipeline provenance {key} must be non-empty"
            )
        return value

    @classmethod
    def _sha256(cls, record: dict, label: str) -> str:
        return cls._sha256_field(record, "sha256", label)

    @staticmethod
    def _sha256_field(
        record: dict,
        key: str,
        label: str,
    ) -> str:
        value = record.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise SharedContentSelectionReferenceError(
                f"{label} has no valid SHA-256"
            )
        return value


class SharedContentSelectionRuntime:
    """Validate equivalence, snapshot CS, and mark reused provenance."""

    def __init__(
        self,
        atomic_write_json: Callable[[Path, object], None],
    ) -> None:
        self._write_json = atomic_write_json
        self._references = SharedContentSelectionRepository(
            atomic_write_json=atomic_write_json
        )

    def prepare(
        self,
        *,
        source_root: str | Path,
        consumer_root: str | Path,
    ) -> SharedContentSelectionReference:
        source = Path(source_root).expanduser().resolve()
        consumer = Path(consumer_root).expanduser().resolve()
        source_provenance = self._load_provenance(source)
        consumer_provenance = self._load_provenance(consumer)
        source_run = source_provenance["run"]
        consumer_run = consumer_provenance["run"]
        if (
            not isinstance(source_run.get("completed_at_utc"), str)
            or not source_run["completed_at_utc"].strip()
        ):
            raise SharedContentSelectionReferenceError(
                "content-selection producer is not complete"
            )
        producer_id = source_run.get("canonical_cell_id")
        consumer_id = consumer_run.get("canonical_cell_id")
        source_equivalence = ContentSelectionEquivalence.from_provenance(
            source_provenance
        )
        consumer_equivalence = (
            ContentSelectionEquivalence.from_provenance(
                consumer_provenance
            )
        )
        if not ContentSelectionEquivalence.compatible_for_reuse(
            source_equivalence,
            consumer_equivalence,
            producer_canonical_id=producer_id,
            consumer_canonical_id=consumer_id,
        ):
            raise SharedContentSelectionReferenceError(
                "content-selection source is not exactly equivalent to "
                "the consumer request"
            )
        digest = ContentSelectionEquivalence.digest(source_equivalence)
        self._validate_generated_execution(source_provenance, digest)
        reference = self._references.persist(
            source_root=source,
            consumer_root=consumer,
            producer_canonical_id=producer_id,
            consumer_canonical_id=consumer_id,
            equivalence_sha256=digest,
            equivalence=source_equivalence,
        )
        source_provenance = deepcopy(
            reference.producer_provenance.value
        )
        source_equivalence = ContentSelectionEquivalence.from_provenance(
            source_provenance
        )
        digest = ContentSelectionEquivalence.digest(source_equivalence)
        if (
            digest != reference.equivalence_sha256
            or not ContentSelectionEquivalence.compatible_for_reuse(
                source_equivalence,
                consumer_equivalence,
                producer_canonical_id=producer_id,
                consumer_canonical_id=consumer_id,
            )
        ):
            raise SharedContentSelectionReferenceError(
                "immutable producer provenance is not exactly equivalent "
                "to the consumer request"
            )
        self._validate_generated_execution(source_provenance, digest)
        content_stage = self._content_stage(consumer_provenance)
        provenance_artifact = reference.artifacts[
            PRODUCER_PROVENANCE_NAME
        ]
        content_stage["execution"] = {
            "schema_version": 1,
            "mode": "reused",
            "execution_id": digest,
            "equivalence": {
                "value": source_equivalence,
                "sha256": digest,
            },
            "producer": {
                "canonical_cell_id": reference.producer_canonical_id,
                "source_root": str(reference.source_root),
                "pipeline_provenance": {
                    "path": str(provenance_artifact.source_path),
                    "snapshot_path": provenance_artifact.snapshot_path
                    .relative_to(reference.consumer_root)
                    .as_posix(),
                    "sha256": provenance_artifact.sha256,
                },
            },
            "artifacts": {
                name: {
                    "source_path": str(artifact.source_path),
                    "snapshot_path": str(
                        artifact.snapshot_path.relative_to(
                            reference.consumer_root
                        )
                    ),
                    "sha256": artifact.sha256,
                }
                for name, artifact in reference.artifacts.items()
                if name != PRODUCER_PROVENANCE_NAME
            },
        }
        self._write_json(
            consumer / "pipeline_provenance.json",
            consumer_provenance,
        )
        return reference

    @staticmethod
    def _load_provenance(root: Path) -> dict:
        path = root / "pipeline_provenance.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SharedContentSelectionReferenceError(
                f"invalid pipeline provenance at {path}: {exc}"
            ) from exc
        if not isinstance(value, dict) or not isinstance(
            value.get("run"),
            dict,
        ):
            raise SharedContentSelectionReferenceError(
                f"invalid pipeline provenance at {path}"
            )
        return value

    @classmethod
    def _validate_generated_execution(
        cls,
        provenance: dict,
        expected_digest: str,
    ) -> None:
        execution = cls._content_stage(provenance).get("execution")
        expected = ContentSelectionEquivalence.generated_execution(
            provenance
        )
        if (
            execution != expected
            or expected.get("execution_id") != expected_digest
        ):
            raise SharedContentSelectionReferenceError(
                "producer content-selection execution provenance is stale"
            )

    @staticmethod
    def _content_stage(provenance: dict) -> dict:
        stages = [
            stage
            for stage in provenance.get("stages", [])
            if (
                isinstance(stage, dict)
                and stage.get("subtask") == "content_selection"
            )
        ]
        if len(stages) != 1:
            raise SharedContentSelectionReferenceError(
                "pipeline provenance must contain one content-selection "
                "stage"
            )
        return stages[0]


__all__ = [
    "ContentSelectionEquivalence",
    "SharedContentSelectionRuntime",
]
