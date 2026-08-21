"""Immutable source and upstream provenance for controlled derived runs."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import tempfile

from .source_catalog import DERIVED_SOURCE_FILE_NAMES


EXPECTED_UPSTREAM_PIPELINE_CONFIGS = {
    "coherence": (
        "configs/controlled/test/{setting}/pipelines/direct_few_shot.json"
    ),
    "mega": (
        "configs/controlled/test/{setting}/pipelines/"
        "direct_zero_shot_context_augmented.json"
    ),
    "planned_zero_shot_without_context": (
        "configs/controlled/test/{setting}/pipelines/"
        "direct_zero_shot.json"
    ),
    "planned_few_shot_context_augmented_independent": (
        "configs/controlled/test/{setting}/pipelines/"
        "direct_few_shot_context_augmented.json"
    ),
}
LEGACY_UPSTREAM_PIPELINE_CONFIGS = {
    "coherence": "configs/test/{setting}/fullcot_g3flash_pipeline.json",
    "mega": (
        "configs/test/{setting}/decontext_zeroshot_g3flash_pipeline.json"
    ),
    "planned_zero_shot_without_context": (
        "configs/test/{setting}/fullcot_zeroshot_g3flash_pipeline.json"
    ),
    "planned_few_shot_context_augmented_independent": (
        "configs/test/{setting}/decontext_g3flash_pipeline.json"
    ),
}
SOURCE_FILE_NAMES = DERIVED_SOURCE_FILE_NAMES
DEPENDENCY_MANIFEST_NAMES = (
    "requirements.txt",
    "requirements-lock-py311-macos-arm64.txt",
)


@dataclass(frozen=True)
class ProvenanceDependencies:
    """Hashing and environment boundaries used by provenance services."""

    stable_value_sha256: object
    artifact_sha256: object
    get_environment_flags: object


class ProvenanceRepository:
    """Load, validate, and archive provenance inputs."""

    def __init__(
        self,
        experiment_root,
        dependencies,
        derived_variants,
        source_file_names=SOURCE_FILE_NAMES,
        dependency_manifest_names=DEPENDENCY_MANIFEST_NAMES,
    ):
        self.experiment_root = Path(experiment_root).resolve()
        self.repository_root = self.experiment_root.parent
        self.dependencies = dependencies
        self.derived_variants = derived_variants
        self.source_file_names = tuple(source_file_names)
        self.dependency_manifest_names = tuple(dependency_manifest_names)

    @staticmethod
    def prepare_output_directory(path):
        """Create and exclusively claim a new or empty output directory."""
        outdir = Path(path).expanduser().resolve()
        if outdir.exists():
            if not outdir.is_dir():
                raise ValueError(f"outdir is not a directory: {outdir}")
            if any(outdir.iterdir()):
                raise ValueError(
                    "outdir must be new or empty to remain append-safe; "
                    f"refusing non-empty directory: {outdir}"
                )
        else:
            outdir.mkdir(parents=True, exist_ok=False)
        claim_path = outdir / ".controlled_run_claim"
        try:
            claim_fd = os.open(
                claim_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError as exc:
            raise ValueError(
                f"outdir is already claimed by another run: {outdir}"
            ) from exc
        with os.fdopen(claim_fd, "w", encoding="utf-8") as claim_file:
            claim_file.write("controlled-derived-run-v1\n")
            claim_file.flush()
            os.fsync(claim_file.fileno())
        return outdir

    @staticmethod
    def load_upstream_provenance(args, input_path):
        """Load the explicit or nearest upstream pipeline provenance."""
        configured = getattr(args, "upstream_provenance", None)
        if configured:
            candidate = Path(configured).expanduser().resolve()
        else:
            candidate = None
            for parent in input_path.parents:
                possible = parent / "pipeline_provenance.json"
                if possible.is_file():
                    candidate = possible
                    break
        if candidate is None:
            raise ValueError(
                "upstream pipeline provenance is required for a controlled "
                "derived variant"
            )
        if not candidate.is_file():
            raise FileNotFoundError(
                f"upstream pipeline provenance not found: {candidate}"
            )
        payload = candidate.read_bytes()
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"upstream pipeline provenance is invalid JSON: {candidate}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                "upstream pipeline provenance must be an object"
            )
        return {
            "path": candidate,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "payload": payload,
            "value": value,
        }

    def _pipeline_path(self, setting, variant, mapping):
        relative = mapping[variant].format(setting=setting)
        return (self.experiment_root / relative).resolve()

    def _load_contract_path(self, pipeline_path):
        if not pipeline_path.is_file():
            raise FileNotFoundError(
                f"expected upstream pipeline config is missing: "
                f"{pipeline_path}"
            )
        pipeline_payload = pipeline_path.read_bytes()
        try:
            pipeline = json.loads(pipeline_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"expected upstream pipeline config is invalid: "
                f"{pipeline_path}"
            ) from exc
        if not isinstance(pipeline, list) or not pipeline:
            raise ValueError(
                "expected upstream pipeline config must be a non-empty list: "
                f"{pipeline_path}"
            )

        stages = {}
        for stage in pipeline:
            if not isinstance(stage, dict):
                raise ValueError(
                    f"expected upstream pipeline stage is invalid: {stage!r}"
                )
            stage_name = stage.get("subtask")
            configured_path = stage.get("config_file")
            if not isinstance(stage_name, str) or not isinstance(
                configured_path,
                str,
            ):
                raise ValueError(
                    "expected upstream pipeline stage lacks "
                    "subtask/config_file"
                )
            config_path = Path(configured_path).expanduser()
            if not config_path.is_absolute():
                config_path = (self.experiment_root / config_path).resolve()
            try:
                config_payload = config_path.read_bytes()
                effective_config = json.loads(
                    config_payload.decode("utf-8")
                )
            except (
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                raise ValueError(
                    f"expected upstream stage config is invalid: {config_path}"
                ) from exc
            stages[stage_name] = {
                "path": config_path,
                "sha256": hashlib.sha256(config_payload).hexdigest(),
                "payload": config_payload,
                "effective_config": effective_config,
            }
        return {
            "pipeline_config": {
                "path": pipeline_path,
                "sha256": hashlib.sha256(pipeline_payload).hexdigest(),
                "payload": pipeline_payload,
            },
            "stages": stages,
        }

    def expected_upstream_contract(self, setting, variant):
        """Return the canonical contract plus accepted historical aliases."""
        canonical = self._load_contract_path(
            self._pipeline_path(
                setting,
                variant,
                EXPECTED_UPSTREAM_PIPELINE_CONFIGS,
            )
        )
        legacy_path = self._pipeline_path(
            setting,
            variant,
            LEGACY_UPSTREAM_PIPELINE_CONFIGS,
        )
        compatible = []
        if legacy_path.is_file():
            compatible.append(self._load_contract_path(legacy_path))
        canonical["compatible_contracts"] = compatible
        return canonical

    def resolve_recorded_path(self, value):
        if not isinstance(value, str) or not value:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.experiment_root / path
        return path.resolve()

    def _recorded_stage_maps(self, provenance):
        stage_configs = {}
        stage_records = {}
        for stage in provenance.get("stages", []):
            if not isinstance(stage, dict):
                continue
            name = stage.get("subtask")
            config = stage.get("effective_config")
            if isinstance(name, str) and isinstance(config, dict):
                stage_configs[name] = config
                stage_records[name] = stage
        return stage_configs, stage_records

    def _matches_contract(
        self,
        recorded_pipeline,
        stage_configs,
        stage_records,
        contract,
    ):
        expected_pipeline = contract["pipeline_config"]
        if (
            not isinstance(recorded_pipeline, dict)
            or self.resolve_recorded_path(recorded_pipeline.get("path"))
            != expected_pipeline["path"]
            or recorded_pipeline.get("sha256")
            != expected_pipeline["sha256"]
            or set(stage_records) != set(contract["stages"])
        ):
            return False
        for stage_name, expected_stage in contract["stages"].items():
            recorded_stage = stage_records.get(stage_name)
            recorded_file = (
                recorded_stage.get("config_file")
                if isinstance(recorded_stage, dict)
                else None
            )
            if (
                not isinstance(recorded_file, dict)
                or self.resolve_recorded_path(recorded_file.get("path"))
                != expected_stage["path"]
                or recorded_file.get("sha256") != expected_stage["sha256"]
                or stage_configs.get(stage_name)
                != expected_stage["effective_config"]
            ):
                return False
        return True

    def _validate_run_identity(
        self,
        provenance,
        *,
        setting,
        split,
        input_path,
        expected_input_stage,
    ):
        run = provenance.get("run")
        if not isinstance(run, dict):
            raise ValueError("upstream provenance has no run object")
        if run.get("setting") != setting or run.get("split") != split:
            raise ValueError(
                "upstream provenance setting/split does not match the "
                "derived run"
            )
        if run.get("dialogue_mode") is not False:
            raise ValueError(
                "controlled coherence/mega require an independent upstream run"
            )
        upstream_root = Path(
            run.get("output_dir", "")
        ).expanduser().resolve()
        expected_input_path = (
            upstream_root
            / "itermediate_results"
            / expected_input_stage
            / "pipeline_format_results.json"
        ).resolve()
        if Path(input_path).expanduser().resolve() != expected_input_path:
            raise ValueError(
                f"derived variant must consume the exact upstream "
                f"{expected_input_stage} artifact {expected_input_path}, "
                f"got {Path(input_path).expanduser().resolve()}"
            )

    @staticmethod
    def _validate_population(
        provenance,
        population_reference_sha256,
        population_ids,
    ):
        upstream_input = provenance.get("input")
        if population_reference_sha256 is not None:
            if not isinstance(upstream_input, dict):
                raise ValueError(
                    "upstream provenance has no fixed input population"
                )
            if upstream_input.get("sha256") != population_reference_sha256:
                raise ValueError(
                    "upstream pipeline used a different dataset snapshot"
                )
        if population_ids is None:
            return
        recorded_ids = (
            upstream_input.get("unique_ids")
            if isinstance(upstream_input, dict)
            else None
        )
        if (
            not isinstance(recorded_ids, list)
            or set(recorded_ids) != set(population_ids)
        ):
            raise ValueError(
                "upstream pipeline used a different UID population"
            )

    @staticmethod
    def _validate_treatment(stage_configs, treatment, model):
        stages_to_check = ["content_selection"]
        if treatment["context_augmentation"]:
            stages_to_check.append("ambiguity_highlight")
        demonstration_mode = treatment["demonstration_mode"]
        for stage_name in stages_to_check:
            config = stage_configs.get(stage_name)
            if config is None:
                raise ValueError(
                    f"upstream provenance is missing {stage_name!r}"
                )
            if config.get("model_name") != model:
                raise ValueError(
                    f"upstream {stage_name} model does not match derived model"
                )
            demonstrations = config.get("n_demos")
            if demonstration_mode == "few_shot":
                if type(demonstrations) is not int or demonstrations < 1:
                    raise ValueError(
                        "planned few-shot treatment requires demonstrations "
                        f"for upstream {stage_name}"
                    )
            elif demonstration_mode == "zero_shot":
                if type(demonstrations) is int and demonstrations == 0:
                    continue
                raise ValueError(
                    "planned zero-shot treatment requires no "
                    f"demonstrations for upstream {stage_name}"
                )
            else:
                raise ValueError(
                    "derived treatment has an unknown demonstration mode"
                )

    def validate_upstream_treatment(
        self,
        upstream_snapshot,
        *,
        variant,
        setting,
        split,
        model,
        input_path,
        population_reference_sha256=None,
        population_ids=None,
        expected_contract=None,
    ):
        """Validate treatment and return the exact matched contract."""
        if upstream_snapshot is None:
            raise ValueError(
                "upstream pipeline provenance is required for a controlled "
                "derived variant"
            )
        provenance = upstream_snapshot["value"]
        treatment = self.derived_variants[variant]
        expected_input_stage = treatment["input_stage"]
        self._validate_run_identity(
            provenance,
            setting=setting,
            split=split,
            input_path=input_path,
            expected_input_stage=expected_input_stage,
        )
        self._validate_population(
            provenance,
            population_reference_sha256,
            population_ids,
        )
        if expected_contract is None:
            expected_contract = self.expected_upstream_contract(
                setting,
                variant,
            )
        stage_configs, stage_records = self._recorded_stage_maps(provenance)
        candidates = [
            expected_contract,
            *expected_contract.get("compatible_contracts", []),
        ]
        recorded_pipeline = provenance.get("pipeline_config")
        matched = next(
            (
                candidate
                for candidate in candidates
                if self._matches_contract(
                    recorded_pipeline,
                    stage_configs,
                    stage_records,
                    candidate,
                )
            ),
            None,
        )
        if matched is None:
            raise ValueError(
                "upstream pipeline or stage config does not match the "
                f"controlled {variant} contract"
            )
        if expected_input_stage not in stage_configs:
            raise ValueError(
                "upstream provenance does not contain required "
                f"{expected_input_stage!r} stage for {variant}"
            )
        self._validate_treatment(stage_configs, treatment, model)
        return matched

    def snapshot_payload(self, payload, outdir, relative_path):
        """Persist one immutable byte-exact copy inside the derived run."""
        if not isinstance(payload, bytes):
            raise TypeError("provenance snapshot payload must be bytes")
        relative_path = Path(relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe provenance snapshot path: {relative_path}"
            )
        outdir = Path(outdir).expanduser().resolve()
        destination = outdir / "provenance_snapshot" / relative_path
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (
                self.dependencies.artifact_sha256(destination)
                != expected_sha256
            ):
                raise FileExistsError(
                    "refusing to overwrite a different provenance snapshot: "
                    f"{destination}"
                )
        else:
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
                if (
                    temporary_path is not None
                    and temporary_path.exists()
                ):
                    temporary_path.unlink()
        if (
            self.dependencies.artifact_sha256(destination)
            != expected_sha256
        ):
            raise RuntimeError(
                f"provenance snapshot hash mismatch: {destination}"
            )
        return str(destination.relative_to(outdir))

    def capture_source(self, outdir):
        """Archive every generation source and its live repository state."""
        source_specs = [
            (self.experiment_root / name, f"source/{name}")
            for name in self.source_file_names
        ]
        source_specs.extend(
            (
                self.repository_root / name,
                f"environment/{name}",
            )
            for name in self.dependency_manifest_names
        )
        source_files = []
        for path, snapshot_relative_path in source_specs:
            payload = path.read_bytes()
            source_files.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "snapshot_path": self.snapshot_payload(
                        payload,
                        outdir,
                        snapshot_relative_path,
                    ),
                }
            )
        git_state = self._git_state(
            [path for path, _ in source_specs]
        )
        try:
            sdk_version = importlib_metadata.version(
                "google-generativeai"
            )
        except importlib_metadata.PackageNotFoundError:
            sdk_version = None
        return {
            "files": source_files,
            "bundle_sha256": self.dependencies.stable_value_sha256(
                source_files
            ),
            "git": git_state,
            "runtime": {
                "python_version": platform.python_version(),
                "google_generativeai_version": sdk_version,
            },
        }

    def _git_state(self, source_paths):
        relative_paths = [
            str(path.relative_to(self.repository_root))
            for path in source_paths
        ]
        git_commit = None
        git_dirty = None
        try:
            commit = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repository_root),
                    "rev-parse",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_commit = commit.stdout.strip()
            tracked = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repository_root),
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    *relative_paths,
                ],
                check=False,
                timeout=10,
            )
            untracked = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repository_root),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    *relative_paths,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_dirty = (
                tracked.returncode != 0 or bool(untracked.stdout.strip())
            )
        except (OSError, subprocess.SubprocessError):
            logging.warning("could not capture git source state")
        return {
            "commit": git_commit,
            "dirty": git_dirty,
            "dirty_scope": "generation source files listed above",
        }


class ProvenanceBuilder:
    """Assemble and verify the complete pipeline provenance document."""

    def __init__(self, repository):
        self.repository = repository
        self.dependencies = repository.dependencies

    def _input_provenance(
        self,
        population,
        variant,
        outdir,
    ):
        snapshot = population["input"]
        selected_ids = population["selected_ids"]
        stage = self.repository.derived_variants[variant]["input_stage"]
        stable_hash = self.dependencies.stable_value_sha256
        return {
            "path": str(snapshot["path"]),
            "sha256": snapshot["sha256"],
            "snapshot_path": self.repository.snapshot_payload(
                snapshot["payload"],
                outdir,
                f"upstream/artifact/{stage}-pipeline_format_results.json",
            ),
            "full_count": len(population["full_input_ids"]),
            "full_unique_ids": population["full_input_ids"],
            "full_unique_ids_order_sha256": stable_hash(
                population["full_input_ids"]
            ),
            "unique_ids": selected_ids,
            "unique_ids_order_sha256": stable_hash(selected_ids),
            "unique_ids_set_sha256": stable_hash(sorted(selected_ids)),
            "selected_row_sha256": {
                row["unique_id"]: stable_hash(row)
                for row in population["selected_rows"]
            },
            "order_normalization": deepcopy(
                population["input_order_normalization"]
            ),
            "max_examples": population["max_examples"],
        }

    def _population_reference(self, args, population, outdir):
        snapshot = population["reference"]
        return {
            "path": str(snapshot["path"]),
            "sha256": snapshot["sha256"],
            "snapshot_path": self.repository.snapshot_payload(
                snapshot["payload"],
                outdir,
                f"input/{args.setting}-{args.split}.json",
            ),
            "count": len(population["reference_ids"]),
            "unique_ids": population["reference_ids"],
            "unique_ids_set_sha256": (
                self.dependencies.stable_value_sha256(
                    sorted(population["reference_ids"])
                )
            ),
        }

    def _upstream_contract(self, expected_contract, outdir):
        pipeline = expected_contract["pipeline_config"]
        pipeline_record = {
            "path": str(pipeline["path"]),
            "sha256": pipeline["sha256"],
            "snapshot_path": self.repository.snapshot_payload(
                pipeline["payload"],
                outdir,
                "upstream/config/pipeline_config.json",
            ),
        }
        stage_records = []
        for index, (stage_name, stage) in enumerate(
            expected_contract["stages"].items(),
            start=1,
        ):
            stage_records.append(
                {
                    "subtask": stage_name,
                    "config_file": {
                        "path": str(stage["path"]),
                        "sha256": stage["sha256"],
                        "snapshot_path": self.repository.snapshot_payload(
                            stage["payload"],
                            outdir,
                            (
                                f"upstream/config/stages/{index:02d}-"
                                f"{stage_name}.json"
                            ),
                        ),
                    },
                    "effective_config": deepcopy(
                        stage["effective_config"]
                    ),
                }
            )
        return pipeline_record, stage_records

    def _verify_archive(self, entries, outdir):
        for entry in entries:
            observed = self.dependencies.artifact_sha256(
                Path(outdir) / entry["snapshot_path"]
            )
            if observed != entry["sha256"]:
                raise RuntimeError(
                    "provenance source changed while its snapshot was "
                    f"captured: {entry['path']}"
                )

    def build(
        self,
        args,
        *,
        variant,
        protocol,
        population,
        upstream_snapshot,
        expected_contract,
        outdir,
    ):
        source = self.repository.capture_source(outdir)
        runtime = source.pop("runtime")
        input_record = self._input_provenance(
            population,
            variant,
            outdir,
        )
        population_record = self._population_reference(
            args,
            population,
            outdir,
        )
        upstream_record = {
            "path": str(upstream_snapshot["path"]),
            "sha256": upstream_snapshot["sha256"],
            "snapshot_path": self.repository.snapshot_payload(
                upstream_snapshot["payload"],
                outdir,
                "upstream/pipeline_provenance.json",
            ),
            "value": upstream_snapshot["value"],
        }
        pipeline_record, stage_records = self._upstream_contract(
            expected_contract,
            outdir,
        )
        archived_entries = [
            input_record,
            population_record,
            upstream_record,
            pipeline_record,
            *[stage["config_file"] for stage in stage_records],
            *source["files"],
        ]
        self._verify_archive(archived_entries, outdir)
        snapshot_manifest = [
            {
                "snapshot_path": entry["snapshot_path"],
                "sha256": entry["sha256"],
            }
            for entry in archived_entries
        ]
        snapshot_bundle_sha256 = (
            self.dependencies.stable_value_sha256(snapshot_manifest)
        )
        source["snapshot_bundle_sha256"] = snapshot_bundle_sha256
        stable_hash = self.dependencies.stable_value_sha256
        treatment = deepcopy(
            self.repository.derived_variants[variant]
        )
        upstream_ids = treatment.pop(
            "upstream_canonical_id_by_setting",
            {},
        )
        upstream_canonical_id = upstream_ids.get(args.setting)
        if (
            not isinstance(upstream_canonical_id, str)
            or not upstream_canonical_id
        ):
            raise ValueError(
                "derived treatment has no catalog upstream canonical ID "
                f"for {args.setting!r}"
            )
        canonical_cell_id = (
            f"{args.setting.lower()}."
            f"{treatment['canonical_factor_id']}"
        )
        return {
            "schema_version": 2,
            "variant": {
                "name": variant,
                "cell_id": f"{args.setting}.{variant}",
                "canonical_cell_id": canonical_cell_id,
                **treatment,
                "upstream_canonical_id": upstream_canonical_id,
            },
            "protocol": {
                **deepcopy(protocol),
                "sha256": stable_hash(protocol),
            },
            "input": input_record,
            "population_reference": population_record,
            "upstream_pipeline_provenance": upstream_record,
            "controlled_upstream_contract": {
                "pipeline_config": pipeline_record,
                "stages": stage_records,
            },
            "provenance_snapshot": {
                "root": "provenance_snapshot",
                "entries": snapshot_manifest,
                "bundle_sha256": snapshot_bundle_sha256,
            },
            "source": source,
            "runtime": runtime,
            "run": {
                "cell_id": f"{args.setting}.{variant}",
                "canonical_cell_id": canonical_cell_id,
                "setting": args.setting,
                "split": args.split,
                "output_dir": str(outdir),
                "concurrency": args.concurrency,
                "append_policy": "new_or_empty_directory_only",
                "input_order_is_output_order": True,
                "errors_preserve_population_rows": True,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            "environment_flags": (
                self.dependencies.get_environment_flags()
            ),
        }
