"""Rerun and result artifacts for the standard single-stage runner."""

import json
from pathlib import Path

from ..stages.configuration import DEFAULT_GENERATION


class RerunPolicy:
    """Validate immutable-parent, append-only rerun semantics."""

    def __init__(self, artifact_sha256):
        self.artifact_sha256 = artifact_sha256

    @staticmethod
    def effective_generation_settings(args):
        rerun = getattr(args, "rerun", False)
        rerun_n_demos = getattr(args, "rerun_n_demos", None)
        rerun_temperature = getattr(args, "rerun_temperature", None)
        n_demos = (
            rerun_n_demos
            if rerun and rerun_n_demos is not None
            else args.n_demos
        )
        temperature = (
            rerun_temperature
            if rerun and rerun_temperature is not None
            else args.temperature
        )
        return n_demos, temperature

    def load(self, args, outdir):
        """Return a validated parent snapshot or ``None`` for a fresh run."""
        if not getattr(args, "rerun", False):
            return None
        rerun_path = getattr(args, "rerun_path", None)
        if not rerun_path:
            raise ValueError(
                "--rerun requires an explicit --rerun-path to the parent "
                "results.json; the derived --outdir must be distinct"
            )
        source_path = Path(rerun_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"rerun parent results not found: {source_path}"
            )

        parent_outdir = source_path.parent
        derived_outdir = Path(outdir).expanduser().resolve()
        if (
            derived_outdir == parent_outdir
            or parent_outdir in derived_outdir.parents
        ):
            raise ValueError(
                "rerun parent and outdir must be distinct; the derived "
                "outdir cannot be the parent run or a directory inside it"
            )
        if derived_outdir.exists():
            if not derived_outdir.is_dir():
                raise ValueError(
                    f"rerun outdir is not a directory: {derived_outdir}"
                )
            if any(derived_outdir.iterdir()):
                raise ValueError(
                    "rerun outdir must be new or empty to remain append-only: "
                    f"{derived_outdir}"
                )

        with source_path.open("r", encoding="utf-8") as parent_file:
            existing_results = json.load(parent_file)
        if not isinstance(existing_results, dict):
            raise ValueError(
                "rerun parent results.json must contain a JSON object"
            )
        error_ids = sorted(
            str(unique_id)
            for unique_id, result in existing_results.items()
            if isinstance(result, dict)
            and str(result.get("final_output", "")).startswith("ERROR")
        )
        return {
            "source_path": source_path,
            "source_sha256": self.artifact_sha256(source_path),
            "existing_results": existing_results,
            "error_ids": error_ids,
            "derived_outdir": derived_outdir,
        }


class DemonstrationDescriptorFactory:
    """Create stable, non-content-leaking demonstration descriptors."""

    def __init__(self, stable_value_sha256):
        self.stable_value_sha256 = stable_value_sha256

    def build(self, used_demonstrations):
        descriptors = []
        for index, demonstration in enumerate(used_demonstrations):
            demonstration_id = None
            if isinstance(demonstration, dict):
                for key in ("unique_id", "id", "topic"):
                    if demonstration.get(key) is not None:
                        demonstration_id = str(demonstration[key])
                        break
            descriptors.append(
                {
                    "index": index,
                    "id": demonstration_id,
                    "sha256": self.stable_value_sha256(demonstration),
                }
            )
        return descriptors


class RerunProvenanceBuilder:
    """Build a reproducible child-to-parent rerun record."""

    def __init__(
        self,
        stable_value_sha256,
        demonstration_descriptors=None,
    ):
        self.stable_value_sha256 = stable_value_sha256
        self.demonstration_descriptors = (
            demonstration_descriptors
            or DemonstrationDescriptorFactory(stable_value_sha256).build
        )

    def build(
        self,
        rerun_context,
        *,
        prompts,
        role_messages,
        used_demos,
        args,
        effective_n_demos,
        effective_temperature,
        environment_flags,
    ):
        demonstrations = self.demonstration_descriptors(used_demos)
        retried_ids = sorted(str(unique_id) for unique_id in prompts)
        examples = {}
        for unique_id in retried_ids:
            role_payload = role_messages.get(unique_id)
            examples[unique_id] = {
                "prompt_sha256": self.stable_value_sha256(
                    prompts[unique_id]
                ),
                "role_messages_sha256": (
                    self.stable_value_sha256(role_payload)
                    if role_payload is not None
                    else None
                ),
                "demonstrations": [
                    dict(item) for item in demonstrations
                ],
            }
        return {
            "schema_version": 1,
            "parent": {
                "results_path": str(rerun_context["source_path"]),
                "sha256": rerun_context["source_sha256"],
            },
            "derived": {
                "outdir": str(rerun_context["derived_outdir"]),
            },
            "retried_ids": retried_ids,
            "examples": examples,
            "effective": {
                "setting": args.setting,
                "split": args.split,
                "subtask": args.subtask,
                "model_name": args.model_name,
                "n_demos": effective_n_demos,
                "temperature": effective_temperature,
                "num_retries": args.num_retries,
                "structured_output": (
                    DEFAULT_GENERATION.structured_output_for(args)
                ),
                "output_max_length": getattr(
                    args,
                    "output_max_length",
                    4096,
                ),
                "concurrency": getattr(args, "concurrency", 1),
                "environment_flags": environment_flags,
            },
        }


class StandardResultAssembler:
    """Preserve upstream failures and merge generated rows by UID."""

    @staticmethod
    def partition_upstream_failures(alignments):
        failures = {}
        active = []
        for item in alignments:
            skipped_reason = item.get("skipped_reason")
            if not skipped_reason:
                active.append(item)
                continue
            unique_id = str(item.get("unique_id"))
            upstream_error = item.get("upstream_error")
            if (
                not isinstance(upstream_error, str)
                or not upstream_error.strip().startswith("ERROR")
            ):
                upstream_error = (
                    "ERROR - upstream stage skipped: "
                    f"{skipped_reason}"
                )
            failures[unique_id] = {
                "final_output": upstream_error,
                "alignments": [],
                "upstream_skipped_reason": str(skipped_reason),
            }
        return active, failures

    @staticmethod
    def add_upstream_failures(
        responses,
        additional_data,
        upstream_failures,
    ):
        for unique_id, failure in upstream_failures.items():
            responses[unique_id] = failure
            additional_data[unique_id] = {
                "upstream_skipped_reason": failure[
                    "upstream_skipped_reason"
                ]
            }

    @staticmethod
    def assemble(
        existing_results,
        responses,
        additional_data,
        all_alignments,
        args,
    ):
        final_results = dict(existing_results)
        final_results.update({key: {} for key in responses})
        alignments_by_id = {
            str(item["unique_id"]): item for item in all_alignments
        }
        for instance_name, response in responses.items():
            current = final_results[instance_name]
            current.update(additional_data[instance_name])
            current["gold_summary"] = alignments_by_id[
                str(instance_name)
            ]["response"]
            if (
                args.subtask == "FiC"
                and args.CoT
                and "alignments" not in response
            ):
                current["alignments"] = []
            current.update(response)
        return final_results
