"""Immutable provenance for standard multi-stage pipeline runs."""

from __future__ import annotations

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

from .source_catalog import STANDARD_SOURCE_FILE_NAMES
from .shared_content_runtime import ContentSelectionEquivalence


_PROTOCOL_ENVIRONMENT_NAMES = (
    "AF_CONTEXT_CACHE",
    "AF_DIALOGUE_NO_DEMOS",
    "AF_DOCS_FIRST",
    "AF_MARK_CONTEXT",
    "AF_USE_ROLES",
)


class StandardPipelineProvenanceRepository:
    """Snapshot and assemble the protocol used by a standard pipeline."""

    def __init__(
        self,
        dependencies,
        *,
        git_source_state=None,
        started_at=None,
    ):
        self._dependencies = dependencies
        self._git_source_state_callback = (
            git_source_state or self.git_source_state
        )
        self._started_at_callback = started_at or self.started_at

    @staticmethod
    def resolved_input_path(config, args, key, default):
        """Resolve a stage input as update_args/get_data resolve it."""
        if key in config:
            configured = config[key]
        else:
            configured = getattr(args, key, None)
        return Path(configured or default).expanduser().resolve()

    @staticmethod
    def load_fixed_population(input_path, max_examples, payload=None):
        rows = []
        if payload is None:
            payload = input_path.read_bytes()
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{input_path}: input is not UTF-8: {exc}"
            ) from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"{input_path}:{line_number}: input row must be an object"
                )
            rows.append(row)

        if max_examples is not None:
            if type(max_examples) is not int or max_examples < 1:
                raise ValueError("max_examples must be a positive integer")
            rows = rows[:max_examples]
        if not rows:
            raise ValueError(f"{input_path}: fixed population is empty")

        unique_ids = []
        seen = set()
        for index, row in enumerate(rows, start=1):
            unique_id = row.get("unique_id")
            if not isinstance(unique_id, str) or not unique_id:
                raise ValueError(
                    f"{input_path}: input row {index} has no "
                    "non-empty unique_id"
                )
            if unique_id in seen:
                raise ValueError(
                    f"{input_path}: duplicate unique_id {unique_id!r}"
                )
            seen.add(unique_id)
            unique_ids.append(unique_id)
        return sorted(unique_ids)

    @staticmethod
    def capture_provenance_file(source_path, outdir, relative_path):
        """Hash and snapshot one immutable byte buffer."""
        source_path = Path(source_path).expanduser().resolve()
        relative_path = Path(relative_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"unsafe provenance snapshot path: {relative_path}"
            )
        output_root = Path(outdir).expanduser().resolve()
        destination = (
            output_root / "provenance_snapshot" / relative_path
        )
        payload = source_path.read_bytes()
        expected_sha256 = hashlib.sha256(payload).hexdigest()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if (
                hashlib.sha256(destination.read_bytes()).hexdigest()
                != expected_sha256
            ):
                raise FileExistsError(
                    "refusing to overwrite a different provenance "
                    f"snapshot: {destination}"
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
            hashlib.sha256(destination.read_bytes()).hexdigest()
            != expected_sha256
        ):
            raise RuntimeError(
                f"provenance snapshot hash mismatch: {destination}"
            )
        return (
            {
                "path": str(source_path),
                "sha256": expected_sha256,
                "snapshot_path": str(
                    destination.relative_to(output_root)
                ),
            },
            payload,
        )

    def persist(self, args, full_configs, outdir):
        """Fingerprint the exact protocol and population before generation."""
        dependencies = self._dependencies
        pipeline_config_path = Path(args.config_file).expanduser().resolve()
        pipeline_config, _ = dependencies.capture_provenance_file(
            pipeline_config_path,
            outdir,
            "pipeline_config.json",
        )
        stages = []
        loaded_configs = []
        for stage_index, stage in enumerate(full_configs, start=1):
            config_path = Path(stage["config_file"]).expanduser().resolve()
            config_record, config_payload = (
                dependencies.capture_provenance_file(
                    config_path,
                    outdir,
                    (
                        f"stages/{stage_index:02d}-"
                        f"{stage['subtask']}.json"
                    ),
                )
            )
            try:
                config = json.loads(config_payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{config_path}: invalid stage config: {exc}"
                ) from exc
            if not isinstance(config, dict):
                raise ValueError(
                    f"{config_path}: stage config must be an object"
                )
            loaded_configs.append((stage, config_path, config))
            schema_key = self._response_schema_key(stage, config)
            effective_schema = (
                dependencies.subtask_schemas.get(schema_key)
                if config.get("structured_output") is True
                else None
            )
            stages.append(
                {
                    "subtask": stage["subtask"],
                    "config_file": config_record,
                    "effective_config": config,
                    "effective_response_schema": {
                        "value": effective_schema,
                        "sha256": (
                            dependencies.stable_value_sha256(
                                effective_schema
                            )
                            if effective_schema is not None
                            else None
                        ),
                    },
                    "effective_randomness": {
                        "demonstration_selection_seed": config.get("seed"),
                        "demonstration_selection_algorithm": (
                            "sha256-rank-v1"
                        ),
                        "provider_generation_seed": None,
                        "provider_generation_seed_supported": False,
                        "temperature": config.get("temperature"),
                    },
                }
            )

        if not loaded_configs:
            raise ValueError("pipeline must contain at least one stage")
        split_values = {
            config.get("split") for _, _, config in loaded_configs
        }
        setting_values = {
            config.get("setting") for _, _, config in loaded_configs
        }
        if len(split_values) != 1 or None in split_values:
            raise ValueError(
                "pipeline stage configs must share one explicit split"
            )
        if len(setting_values) != 1 or None in setting_values:
            raise ValueError(
                "pipeline stage configs must share one explicit setting"
            )
        split = next(iter(split_values))
        setting = next(iter(setting_values))
        controlled_identity = self._controlled_identity(
            args,
            setting=setting,
            loaded_configs=loaded_configs,
        )

        content_stage = next(
            (
                config
                for stage, _, config in loaded_configs
                if stage["subtask"] == "content_selection"
            ),
            loaded_configs[0][2],
        )
        input_path = dependencies.resolved_input_path(
            content_stage,
            args,
            "indir_alignments",
            f"../data/{setting}/{split}.json",
        )
        max_examples = (
            content_stage["max_examples"]
            if "max_examples" in content_stage
            else getattr(args, "max_examples", None)
        )
        input_provenance, input_payload = (
            dependencies.capture_provenance_file(
                input_path,
                outdir,
                f"input/{input_path.name}",
            )
        )
        unique_ids = dependencies.load_fixed_population(
            input_path,
            max_examples,
            payload=input_payload,
        )

        prompt_paths = {}
        for _, _, config in loaded_configs:
            prompt_path = dependencies.resolved_input_path(
                config,
                args,
                "indir_prompt",
                f"prompts/{setting}.json",
            )
            prompt_key = str(prompt_path)
            if prompt_key not in prompt_paths:
                prompt_index = len(prompt_paths) + 1
                prompt_paths[prompt_key], _ = (
                    dependencies.capture_provenance_file(
                        prompt_path,
                        outdir,
                        (
                            f"prompts/{prompt_index:02d}-"
                            f"{prompt_path.name}"
                        ),
                    )
                )

        source_root = Path(dependencies.source_file).resolve().parent
        repository_root = source_root.parent
        source_specs = [
            (source_root / source_name, f"source/{source_name}")
            for source_name in STANDARD_SOURCE_FILE_NAMES
        ]
        source_specs.extend(
            [
                (
                    repository_root / "requirements.txt",
                    "environment/requirements.txt",
                ),
                (
                    repository_root
                    / "requirements-lock-py311-macos-arm64.txt",
                    (
                        "environment/"
                        "requirements-lock-py311-macos-arm64.txt"
                    ),
                ),
            ]
        )
        source_files = []
        for source_path, snapshot_relative_path in source_specs:
            source_record, _ = dependencies.capture_provenance_file(
                source_path,
                outdir,
                snapshot_relative_path,
            )
            source_files.append(source_record)
        git_commit, git_dirty = self._git_source_state_callback(
            repository_root,
            source_specs,
        )

        input_provenance.update(
            {
                "count": len(unique_ids),
                "unique_ids": unique_ids,
                "unique_ids_sha256": (
                    dependencies.stable_value_sha256(unique_ids)
                ),
                "max_examples": max_examples,
            }
        )
        ordered_prompt_inputs = [
            prompt_paths[path] for path in sorted(prompt_paths)
        ]
        archived_entries = [
            pipeline_config,
            *[stage["config_file"] for stage in stages],
            input_provenance,
            *ordered_prompt_inputs,
            *source_files,
        ]
        started_at_utc = self._started_at_callback(outdir)
        provenance = {
            "schema_version": 2,
            "pipeline_config": pipeline_config,
            "stages": stages,
            "input": input_provenance,
            "prompt_inputs": ordered_prompt_inputs,
            "source": {
                "files": source_files,
                "bundle_sha256": (
                    dependencies.stable_value_sha256(source_files)
                ),
                "snapshot_bundle_sha256": (
                    dependencies.stable_value_sha256(
                        [
                            {
                                "snapshot_path": entry["snapshot_path"],
                                "sha256": entry["sha256"],
                            }
                            for entry in archived_entries
                        ]
                    )
                ),
                "git": {
                    "commit": git_commit,
                    "dirty": git_dirty,
                    "dirty_scope": (
                        "generation source files listed above"
                    ),
                },
            },
            "run": {
                "split": split,
                "setting": setting,
                "dialogue_mode": bool(
                    getattr(args, "dialogue_mode", False)
                ),
                "transport": (
                    "dialogue"
                    if bool(getattr(args, "dialogue_mode", False))
                    else "independent"
                ),
                "cell_id": Path(outdir).expanduser().resolve().name,
                "output_dir": str(
                    Path(outdir).expanduser().resolve()
                ),
                "started_at_utc": started_at_utc,
                **(controlled_identity or {}),
            },
            "runtime": {
                "python_version": platform.python_version(),
                "google_generativeai_version": (
                    importlib_metadata.version("google-generativeai")
                ),
            },
            "protocol": {
                "environment_flags": {
                    name: os.environ.get(name)
                    for name in _PROTOCOL_ENVIRONMENT_NAMES
                },
                "cli": {
                    "structured_output": getattr(
                        args,
                        "structured_output",
                        None,
                    ),
                    "use_roles": getattr(args, "use_roles", None),
                },
            },
        }
        if controlled_identity is not None:
            content_stage_record = next(
                stage
                for stage in provenance["stages"]
                if stage["subtask"] == "content_selection"
            )
            content_stage_record["execution"] = (
                ContentSelectionEquivalence.generated_execution(
                    provenance
                )
            )
        dependencies.atomic_write_json(
            Path(outdir) / "pipeline_provenance.json",
            provenance,
        )
        return provenance

    @staticmethod
    def _response_schema_key(stage, config):
        protocol = config.get("protocol")
        declared = (
            protocol.get("schema")
            if isinstance(protocol, dict)
            else None
        )
        prefix = "SUBTASK_SCHEMAS."
        if isinstance(declared, str) and declared.startswith(prefix):
            key = declared[len(prefix) :]
            if key:
                return key
        return (
            "FiC"
            if stage["subtask"] in {"FiC", "fusion_in_context"}
            else stage["subtask"]
        )

    @staticmethod
    def _controlled_identity(
        args,
        *,
        setting,
        loaded_configs,
    ):
        declared = {
            "canonical_cell_id": getattr(
                args, "canonical_cell_id", None
            ),
            "generation": getattr(
                args, "generation_strategy", None
            ),
            "demonstrations": getattr(
                args, "demonstration_mode", None
            ),
            "context_augmentation": getattr(
                args, "context_augmentation", None
            ),
            "transport": getattr(args, "transport_mode", None),
        }
        if all(value is None for value in declared.values()):
            return None
        if any(value is None for value in declared.values()):
            raise ValueError(
                "controlled standard identity requires canonical ID and "
                "all four factors"
            )
        planned_dialogue = (
            declared["generation"] == "planned"
            and bool(getattr(args, "planned_dialogue", False))
            and str(setting) == "LFQA"
            and bool(getattr(args, "dialogue_mode", False))
        )
        if declared["generation"] != "direct" and not planned_dialogue:
            raise ValueError(
                "pipeline generation factor must be direct unless it is "
                "the LFQA planned dialogue treatment"
            )
        actual_transport = (
            "dialogue"
            if bool(getattr(args, "dialogue_mode", False))
            else "independent"
        )
        if declared["transport"] != actual_transport:
            raise ValueError(
                "declared transport factor conflicts with dialogue mode"
            )
        stage_sequence = tuple(
            stage["subtask"] for stage, _, _ in loaded_configs
        )
        if planned_dialogue and stage_sequence != (
            "content_selection",
            "ambiguity_highlight",
            "clustering",
            "reorder",
            "fusion_in_context",
        ):
            raise ValueError(
                "LFQA planned dialogue provenance requires the exact "
                "five-stage sequence"
            )
        stage_names = set(stage_sequence)
        actual_context = (
            "enabled"
            if "ambiguity_highlight" in stage_names
            else "disabled"
        )
        if declared["context_augmentation"] != actual_context:
            raise ValueError(
                "declared context factor conflicts with stage composition"
            )
        demo_counts = [
            config.get("n_demos")
            for _, _, config in loaded_configs
        ]
        if any(
            type(count) is not int or count < 0
            for count in demo_counts
        ):
            raise ValueError(
                "controlled stage configs require explicit n_demos"
            )
        actual_demonstrations = (
            "zero_shot"
            if all(count == 0 for count in demo_counts)
            else "few_shot"
        )
        if declared["demonstrations"] != actual_demonstrations:
            raise ValueError(
                "declared demonstration factor conflicts with stage configs"
            )
        short_demo = (
            "fs"
            if declared["demonstrations"] == "few_shot"
            else "zs"
        )
        factor_parts = [declared["generation"], short_demo]
        if declared["context_augmentation"] == "enabled":
            factor_parts.append("context_augmented")
        factor_parts.append(declared["transport"])
        expected_canonical_id = (
            f"{str(setting).lower()}." + "_".join(factor_parts)
        )
        if declared["canonical_cell_id"] != expected_canonical_id:
            raise ValueError(
                "canonical cell ID conflicts with declared factors: "
                f"expected {expected_canonical_id!r}"
            )
        return {
            "canonical_cell_id": declared["canonical_cell_id"],
            "factors": {
                key: declared[key]
                for key in (
                    "generation",
                    "demonstrations",
                    "context_augmentation",
                    "transport",
                )
            },
        }

    @staticmethod
    def git_source_state(repository_root, source_specs):
        git_commit = None
        git_dirty = None
        try:
            commit_result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "rev-parse",
                    "HEAD",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_commit = commit_result.stdout.strip()
            relative_sources = [
                str(path.relative_to(repository_root))
                for path, _ in source_specs
            ]
            tracked_diff = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "diff",
                    "--quiet",
                    "HEAD",
                    "--",
                    *relative_sources,
                ],
                check=False,
                timeout=10,
            )
            untracked = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository_root),
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "--",
                    *relative_sources,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            git_dirty = tracked_diff.returncode != 0 or bool(
                untracked.stdout.strip()
            )
        except (OSError, subprocess.SubprocessError):
            logging.warning("could not capture git source state")
        return git_commit, git_dirty

    @staticmethod
    def started_at(outdir):
        started_at_utc = datetime.now(timezone.utc).isoformat()
        existing_path = Path(outdir) / "pipeline_provenance.json"
        if not existing_path.is_file():
            return started_at_utc
        try:
            existing = json.loads(
                existing_path.read_text(encoding="utf-8")
            )
            existing_started_at = existing.get("run", {}).get(
                "started_at_utc"
            )
            if isinstance(existing_started_at, str) and existing_started_at:
                return existing_started_at
        except (OSError, json.JSONDecodeError, AttributeError):
            raise ValueError(
                f"{existing_path}: invalid existing provenance"
            )
        return started_at_utc


__all__ = ["StandardPipelineProvenanceRepository"]
