"""Immutable snapshots for content-selection reuse between catalog cells."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping


MANIFEST_NAME = "shared_content_selection.json"
STAGE_NAME = "content_selection"
PRODUCER_PROVENANCE_NAME = "pipeline_provenance.json"
SNAPSHOT_DIRECTORY = Path(
    "provenance_snapshot/shared_stages/content_selection"
)
REQUIRED_STAGE_ARTIFACT_NAMES = (
    "results.json",
    "pipeline_format_results.json",
    "used_demonstrations.json",
    "token_usage.json",
)
REQUIRED_ARTIFACT_NAMES = (
    *REQUIRED_STAGE_ARTIFACT_NAMES,
    PRODUCER_PROVENANCE_NAME,
)
OPTIONAL_ARTIFACT_NAMES = ("dialogue_checkpoint.json",)
SAFE_CANONICAL_RELATIONS = frozenset(
    (
        f"{setting}.direct_{demonstrations}_{transport}",
        (
            f"{setting}.direct_{demonstrations}_"
            f"context_augmented_{transport}"
        ),
    )
    for setting in ("mds", "lfqa")
    for demonstrations, transport in (
        ("zs", "independent"),
        ("fs", "independent"),
        ("fs", "dialogue"),
    )
)
SAFE_SELF_REUSE_CANONICAL_IDS = frozenset(
    {"lfqa.planned_fs_context_augmented_dialogue"}
)
SOURCE_BUNDLE_DRIFT_RELATIONS = frozenset(
    {
        (
            "lfqa.direct_fs_dialogue",
            "lfqa.planned_fs_context_augmented_dialogue",
        ),
        (
            "lfqa.planned_fs_context_augmented_dialogue",
            "lfqa.planned_fs_context_augmented_dialogue",
        ),
    }
)


class SharedContentSelectionReferenceError(ValueError):
    """A shared-stage manifest is unsafe, incomplete, or inconsistent."""


@dataclass(frozen=True)
class SharedContentSelectionArtifact:
    """One source artifact and its immutable consumer-side snapshot."""

    name: str
    source_path: Path
    snapshot_path: Path
    sha256: str


@dataclass(frozen=True)
class ProducerProvenanceSnapshot:
    """Validated immutable evidence for the physical CS execution."""

    artifact: SharedContentSelectionArtifact
    value: Mapping[str, object]
    canonical_cell_id: str
    completed_at_utc: str
    equivalence_sha256: str


@dataclass(frozen=True)
class SharedContentSelectionReference:
    """Validated relation between one physical CS run and one consumer."""

    consumer_root: Path
    source_root: Path
    producer_canonical_id: str
    consumer_canonical_id: str
    equivalence_sha256: str
    artifacts: Mapping[str, SharedContentSelectionArtifact]
    producer_provenance: ProducerProvenanceSnapshot

    def snapshot_for(self, name: str) -> Path:
        """Return a validated snapshot path by artifact name."""
        try:
            return self.artifacts[name].snapshot_path
        except KeyError as exc:
            raise KeyError(
                f"shared content-selection artifact is missing: {name}"
            ) from exc


class SharedContentSelectionRepository:
    """Persist and load the immutable shared-CS reference contract."""

    def __init__(
        self,
        atomic_write_json: Callable[[Path, object], None] | None = None,
    ) -> None:
        self._atomic_write_json = atomic_write_json

    def persist(
        self,
        *,
        source_root: str | Path,
        consumer_root: str | Path,
        producer_canonical_id: str,
        consumer_canonical_id: str,
        equivalence_sha256: str,
        artifacts: (
            Mapping[str, str | Path]
            | Iterable[str | Path]
            | None
        ) = None,
        equivalence: object | None = None,
    ) -> SharedContentSelectionReference:
        """Snapshot source bytes once, then publish an immutable manifest."""
        source = self._existing_directory(source_root, "source root")
        consumer = Path(consumer_root).expanduser().resolve()
        consumer.mkdir(parents=True, exist_ok=True)
        self._validate_identity(
            producer_canonical_id,
            consumer_canonical_id,
            equivalence_sha256,
        )
        self._reject_local_stage(consumer)
        selected = self._selected_artifacts(source, artifacts)
        records = [
            self._snapshot_artifact(consumer, name, path)
            for name, path in selected.items()
        ]
        manifest = {
            "schema_version": 1,
            "stage": STAGE_NAME,
            "producer_canonical_id": producer_canonical_id,
            "consumer_canonical_id": consumer_canonical_id,
            "source_root": str(source),
            "equivalence_sha256": equivalence_sha256,
            "artifacts": records,
        }
        if equivalence is not None:
            if _stable_value_sha256(equivalence) != equivalence_sha256:
                raise SharedContentSelectionReferenceError(
                    "equivalence value does not match equivalence_sha256"
                )
            manifest["equivalence"] = {
                "value": equivalence,
                "sha256": equivalence_sha256,
            }
        self._persist_manifest(consumer / MANIFEST_NAME, manifest)
        return self.load(
            consumer,
            expected_source_root=source,
        )

    def load(
        self,
        consumer_root: str | Path,
        *,
        expected_source_root: str | Path | None = None,
    ) -> SharedContentSelectionReference:
        """Load and fully validate one consumer-side shared-CS manifest."""
        consumer = self._existing_directory(
            consumer_root,
            "consumer root",
        )
        manifest_path = consumer / MANIFEST_NAME
        if manifest_path.is_symlink():
            raise SharedContentSelectionReferenceError(
                f"{MANIFEST_NAME} cannot be a symbolic link"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise SharedContentSelectionReferenceError(
                f"missing {MANIFEST_NAME}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise SharedContentSelectionReferenceError(
                f"invalid {MANIFEST_NAME}: {exc}"
            ) from exc
        if not isinstance(manifest, dict):
            raise SharedContentSelectionReferenceError(
                f"{MANIFEST_NAME} must be a JSON object"
            )
        self._validate_manifest_header(manifest)
        source = self._canonical_manifest_root(
            manifest.get("source_root")
        )
        if expected_source_root is not None:
            expected = Path(expected_source_root).expanduser().resolve()
            if source != expected:
                raise SharedContentSelectionReferenceError(
                    "shared content-selection source root is unexpected"
                )
        self._reject_local_stage(consumer)
        artifact_records = self._validated_artifact_records(
            consumer,
            source,
            manifest.get("artifacts"),
        )
        self._validate_equivalence(manifest)
        producer_provenance = self._validated_producer_provenance(
            artifact_records[PRODUCER_PROVENANCE_NAME],
            manifest,
        )
        return SharedContentSelectionReference(
            consumer_root=consumer,
            source_root=source,
            producer_canonical_id=manifest["producer_canonical_id"],
            consumer_canonical_id=manifest["consumer_canonical_id"],
            equivalence_sha256=manifest["equivalence_sha256"],
            artifacts=artifact_records,
            producer_provenance=producer_provenance,
        )

    @staticmethod
    def _existing_directory(value: str | Path, label: str) -> Path:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise SharedContentSelectionReferenceError(
                f"{label} is not an existing directory: {path}"
            )
        return path

    @staticmethod
    def _validate_identity(
        producer: object,
        consumer: object,
        equivalence_sha256: object,
    ) -> None:
        is_safe_relation = (
            (producer, consumer) in SAFE_CANONICAL_RELATIONS
            or (producer, consumer) in SOURCE_BUNDLE_DRIFT_RELATIONS
        )
        is_safe_recovery = (
            producer == consumer
            and producer in SAFE_SELF_REUSE_CANONICAL_IDS
        )
        if not (is_safe_relation or is_safe_recovery):
            raise SharedContentSelectionReferenceError(
                "producer/consumer canonical IDs are not a safe "
                "content-selection relation"
            )
        if not _is_sha256(equivalence_sha256):
            raise SharedContentSelectionReferenceError(
                "equivalence_sha256 must be a lowercase SHA-256 digest"
            )

    def _validate_manifest_header(self, manifest: dict) -> None:
        if manifest.get("schema_version") != 1:
            raise SharedContentSelectionReferenceError(
                f"{MANIFEST_NAME} must use schema_version 1"
            )
        if manifest.get("stage") != STAGE_NAME:
            raise SharedContentSelectionReferenceError(
                f"{MANIFEST_NAME}.stage must be {STAGE_NAME!r}"
            )
        self._validate_identity(
            manifest.get("producer_canonical_id"),
            manifest.get("consumer_canonical_id"),
            manifest.get("equivalence_sha256"),
        )

    @staticmethod
    def _canonical_manifest_root(value: object) -> Path:
        if not isinstance(value, str) or not value:
            raise SharedContentSelectionReferenceError(
                "source_root must be a non-empty absolute path"
            )
        recorded = Path(value).expanduser()
        if not recorded.is_absolute() or recorded != recorded.resolve():
            raise SharedContentSelectionReferenceError(
                "source_root must be a canonical absolute path"
            )
        if not recorded.is_dir():
            raise SharedContentSelectionReferenceError(
                f"source_root is missing: {recorded}"
            )
        return recorded

    @staticmethod
    def _reject_local_stage(consumer: Path) -> None:
        local_stage = (
            consumer / "itermediate_results" / STAGE_NAME
        )
        if local_stage.exists() or local_stage.is_symlink():
            raise SharedContentSelectionReferenceError(
                "a reused content-selection stage cannot also exist "
                "locally"
            )

    def _selected_artifacts(
        self,
        source: Path,
        requested: (
            Mapping[str, str | Path]
            | Iterable[str | Path]
            | None
        ),
    ) -> dict[str, Path]:
        stage_root = source / "itermediate_results" / STAGE_NAME
        if requested is None:
            selected = {
                name: stage_root / name
                for name in REQUIRED_STAGE_ARTIFACT_NAMES
            }
            selected[PRODUCER_PROVENANCE_NAME] = (
                source / PRODUCER_PROVENANCE_NAME
            )
            selected.update(
                {
                    name: stage_root / name
                    for name in OPTIONAL_ARTIFACT_NAMES
                    if (stage_root / name).is_file()
                }
            )
        elif isinstance(requested, Mapping):
            selected = {
                name: self._source_artifact_path(
                    source,
                    stage_root,
                    name,
                    value,
                )
                for name, value in requested.items()
            }
        else:
            paths = [
                self._source_artifact_path(
                    source,
                    stage_root,
                    Path(value).name,
                    value,
                )
                for value in requested
            ]
            selected = {path.name: path for path in paths}
            if len(selected) != len(paths):
                raise SharedContentSelectionReferenceError(
                    "shared artifact names must be unique"
                )
        missing = set(REQUIRED_ARTIFACT_NAMES) - set(selected)
        if missing:
            raise SharedContentSelectionReferenceError(
                "missing required shared artifacts: "
                + ", ".join(sorted(missing))
            )
        for name, path in selected.items():
            self._validate_artifact_name(name)
            if path.name != name or not path.is_file():
                raise SharedContentSelectionReferenceError(
                    f"shared source artifact is missing: {path}"
                )
            try:
                path.resolve().relative_to(source)
            except ValueError as exc:
                raise SharedContentSelectionReferenceError(
                    f"shared source artifact escapes source root: {path}"
                ) from exc
        return dict(sorted(selected.items()))

    @staticmethod
    def _source_artifact_path(
        source: Path,
        stage_root: Path,
        name: str,
        value: str | Path,
    ) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            base = (
                source
                if name == PRODUCER_PROVENANCE_NAME
                else stage_root
            )
            path = base / path
        return path.resolve()

    @staticmethod
    def _validate_artifact_name(name: object) -> None:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in {".", ".."}
        ):
            raise SharedContentSelectionReferenceError(
                "shared artifact name must be one safe file name"
            )

    def _snapshot_artifact(
        self,
        consumer: Path,
        name: str,
        source_path: Path,
    ) -> dict[str, str]:
        payload = source_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        snapshot_relative = SNAPSHOT_DIRECTORY / name
        snapshot = consumer / snapshot_relative
        _persist_immutable_bytes(snapshot, payload)
        if _artifact_sha256(source_path) != digest:
            raise SharedContentSelectionReferenceError(
                f"source artifact changed while snapshotting: {source_path}"
            )
        return {
            "name": name,
            "source_path": str(source_path),
            "snapshot_path": snapshot_relative.as_posix(),
            "sha256": digest,
        }

    def _persist_manifest(self, path: Path, manifest: dict) -> None:
        if path.is_symlink():
            raise FileExistsError(
                f"refusing symbolic-link shared manifest: {path}"
            )
        if path.exists():
            try:
                current = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FileExistsError(
                    f"refusing to replace invalid shared manifest: {path}"
                ) from exc
            if current != manifest:
                raise FileExistsError(
                    f"refusing to replace shared manifest: {path}"
                )
            return
        if self._atomic_write_json is not None:
            self._atomic_write_json(path, manifest)
        else:
            payload = (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            _persist_immutable_bytes(path, payload)

    def _validated_artifact_records(
        self,
        consumer: Path,
        source: Path,
        raw_records: object,
    ) -> dict[str, SharedContentSelectionArtifact]:
        if not isinstance(raw_records, list) or not raw_records:
            raise SharedContentSelectionReferenceError(
                "artifacts must be a non-empty list"
            )
        records = {}
        for index, raw in enumerate(raw_records):
            artifact = self._validated_artifact_record(
                consumer,
                source,
                raw,
                index,
            )
            if artifact.name in records:
                raise SharedContentSelectionReferenceError(
                    f"duplicate shared artifact name: {artifact.name}"
                )
            records[artifact.name] = artifact
        missing = set(REQUIRED_ARTIFACT_NAMES) - set(records)
        if missing:
            raise SharedContentSelectionReferenceError(
                "missing required shared artifacts: "
                + ", ".join(sorted(missing))
            )
        return records

    def _validated_artifact_record(
        self,
        consumer: Path,
        source: Path,
        raw: object,
        index: int,
    ) -> SharedContentSelectionArtifact:
        label = f"artifacts[{index}]"
        if not isinstance(raw, dict):
            raise SharedContentSelectionReferenceError(
                f"{label} must be an object"
            )
        name = raw.get("name")
        self._validate_artifact_name(name)
        source_path = self._validated_source_path(
            source,
            raw.get("source_path"),
            name,
            label,
        )
        snapshot_path = self._validated_snapshot_path(
            consumer,
            raw.get("snapshot_path"),
            name,
            label,
        )
        digest = raw.get("sha256")
        if not _is_sha256(digest):
            raise SharedContentSelectionReferenceError(
                f"{label}.sha256 is invalid"
            )
        for kind, path in (
            ("source", source_path),
            ("snapshot", snapshot_path),
        ):
            if not path.is_file():
                raise SharedContentSelectionReferenceError(
                    f"{label} {kind} is missing: {path}"
                )
            if _artifact_sha256(path) != digest:
                raise SharedContentSelectionReferenceError(
                    f"{name} {kind} fingerprint mismatch"
                )
        return SharedContentSelectionArtifact(
            name=name,
            source_path=source_path,
            snapshot_path=snapshot_path,
            sha256=digest,
        )

    @staticmethod
    def _validated_source_path(
        source: Path,
        value: object,
        name: str,
        label: str,
    ) -> Path:
        if not isinstance(value, str) or not value:
            raise SharedContentSelectionReferenceError(
                f"{label}.source_path must be non-empty"
            )
        recorded = Path(value).expanduser()
        resolved = recorded.resolve()
        if (
            not recorded.is_absolute()
            or recorded != resolved
            or resolved.name != name
        ):
            raise SharedContentSelectionReferenceError(
                f"{label}.source_path must be canonical and match name"
            )
        try:
            resolved.relative_to(source)
        except ValueError as exc:
            raise SharedContentSelectionReferenceError(
                f"{label}.source_path escapes source_root"
            ) from exc
        return resolved

    @staticmethod
    def _validated_snapshot_path(
        consumer: Path,
        value: object,
        name: str,
        label: str,
    ) -> Path:
        if not isinstance(value, str) or not value:
            raise SharedContentSelectionReferenceError(
                f"{label}.snapshot_path must be non-empty"
            )
        relative = Path(value)
        expected = SNAPSHOT_DIRECTORY / name
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative != expected
        ):
            raise SharedContentSelectionReferenceError(
                f"{label}.snapshot_path is unsafe"
            )
        lexical = (consumer / relative).absolute()
        resolved = lexical.resolve()
        if lexical != resolved:
            raise SharedContentSelectionReferenceError(
                f"{label}.snapshot_path traverses a symbolic link"
            )
        return resolved

    @staticmethod
    def _validate_equivalence(manifest: dict) -> None:
        record = manifest.get("equivalence")
        if (
            not isinstance(record, dict)
            or set(record) != {"value", "sha256"}
            or record.get("sha256")
            != manifest["equivalence_sha256"]
            or _stable_value_sha256(record.get("value"))
            != manifest["equivalence_sha256"]
        ):
            raise SharedContentSelectionReferenceError(
                "equivalence record does not match equivalence_sha256"
            )

    @classmethod
    def _validated_producer_provenance(
        cls,
        artifact: SharedContentSelectionArtifact,
        manifest: dict,
    ) -> ProducerProvenanceSnapshot:
        try:
            value = json.loads(
                artifact.snapshot_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise SharedContentSelectionReferenceError(
                "producer pipeline_provenance.json snapshot is invalid"
            ) from exc
        if not isinstance(value, dict):
            raise SharedContentSelectionReferenceError(
                "producer pipeline_provenance.json must be an object"
            )
        run = value.get("run")
        if (
            not isinstance(run, dict)
            or run.get("canonical_cell_id")
            != manifest["producer_canonical_id"]
        ):
            raise SharedContentSelectionReferenceError(
                "producer pipeline provenance canonical ID conflicts with "
                "the shared reference"
            )
        completed_at = run.get("completed_at_utc")
        if not cls._is_completed_timestamp(completed_at):
            raise SharedContentSelectionReferenceError(
                "content-selection producer is not complete"
            )
        stages = [
            stage
            for stage in value.get("stages", [])
            if (
                isinstance(stage, dict)
                and stage.get("subtask") == STAGE_NAME
            )
        ]
        if len(stages) != 1:
            raise SharedContentSelectionReferenceError(
                "producer pipeline provenance must contain exactly one "
                "content-selection stage"
            )
        execution = stages[0].get("execution")
        expected_equivalence = manifest["equivalence"]
        expected_digest = manifest["equivalence_sha256"]
        if (
            not isinstance(execution, dict)
            or execution.get("schema_version") != 1
            or execution.get("mode") != "generated"
            or execution.get("execution_id") != expected_digest
            or execution.get("equivalence") != expected_equivalence
        ):
            raise SharedContentSelectionReferenceError(
                "producer content-selection execution provenance is stale"
            )
        return ProducerProvenanceSnapshot(
            artifact=artifact,
            value=value,
            canonical_cell_id=manifest["producer_canonical_id"],
            completed_at_utc=completed_at,
            equivalence_sha256=expected_digest,
        )

    @staticmethod
    def _is_completed_timestamp(value: object) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        try:
            timestamp = datetime.fromisoformat(
                value.replace("Z", "+00:00")
            )
        except ValueError:
            return False
        return timestamp.tzinfo is not None


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _stable_value_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _persist_immutable_bytes(destination: Path, payload: bytes) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise FileExistsError(
            f"refusing symbolic-link immutable snapshot: {destination}"
        )
    if destination.exists():
        if (
            not destination.is_file()
            or _artifact_sha256(destination) != digest
        ):
            raise FileExistsError(
                f"refusing to replace immutable snapshot: {destination}"
            )
        return
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    if _artifact_sha256(destination) != digest:
        raise RuntimeError(
            f"immutable snapshot fingerprint mismatch: {destination}"
        )


__all__ = [
    "MANIFEST_NAME",
    "OPTIONAL_ARTIFACT_NAMES",
    "PRODUCER_PROVENANCE_NAME",
    "REQUIRED_ARTIFACT_NAMES",
    "REQUIRED_STAGE_ARTIFACT_NAMES",
    "SAFE_CANONICAL_RELATIONS",
    "SNAPSHOT_DIRECTORY",
    "ProducerProvenanceSnapshot",
    "SharedContentSelectionArtifact",
    "SharedContentSelectionReference",
    "SharedContentSelectionReferenceError",
    "SharedContentSelectionRepository",
]
