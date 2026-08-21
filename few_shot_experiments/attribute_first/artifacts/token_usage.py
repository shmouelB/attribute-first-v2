"""Whole-pipeline token-usage discovery, validation, and persistence."""

import json
from pathlib import Path

from .shared_content_selection import (
    MANIFEST_NAME as SHARED_CONTENT_SELECTION_MANIFEST,
    SharedContentSelectionRepository,
)


class PipelineTokenUsageAggregator:
    """Combine stage ledgers without coupling the artifact facade to layout."""

    _PREFERRED_STAGE_ORDER = (
        "content_selection",
        "ambiguity_highlight",
        "clustering",
    )
    _REQUIRED_COUNTERS = ("prompt", "completion", "cached", "calls")
    _OPTIONAL_COUNTERS = (
        "provider_total",
        "provider_total_calls",
    )

    def __init__(self, atomic_write_json):
        self._atomic_write_json = atomic_write_json

    def persist(self, outdir):
        """Write and return the aggregate for every discovered stage ledger."""
        root = Path(outdir)
        stage_paths = self._discover_stage_paths(root)
        reference = self._shared_reference(root)
        if not stage_paths and reference is None:
            raise FileNotFoundError(
                f"no stage token_usage.json found below {root}"
            )

        aggregate, breakdown = self._aggregate(stage_paths)
        aggregate["stages"] = breakdown
        aggregate["accounting_scope"] = "physical"
        aggregate["reused_stages"] = self._reused_stages(reference)
        aggregate["logical_totals"] = self._logical_totals(
            aggregate,
            aggregate["reused_stages"],
        )
        self._attach_run_identity(root, aggregate)
        self._atomic_write_json(
            root / "pipeline_token_usage.json",
            aggregate,
        )
        return aggregate

    @staticmethod
    def _shared_reference(root):
        manifest_path = root / SHARED_CONTENT_SELECTION_MANIFEST
        if not manifest_path.exists() and not manifest_path.is_symlink():
            return None
        return SharedContentSelectionRepository().load(root)

    def _reused_stages(self, reference):
        if reference is None:
            return {}
        usage_path = reference.snapshot_for("token_usage.json")
        try:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{usage_path}: invalid shared token usage: {exc}"
            ) from exc
        scratch = {
            counter: 0
            for counter in (
                self._REQUIRED_COUNTERS + self._OPTIONAL_COUNTERS
            )
        }
        self._validate_and_add(usage_path, usage, scratch)
        if usage.get("subtask") != "content_selection":
            raise ValueError(
                f"{usage_path}: reused usage must be content_selection"
            )
        return {"content_selection": usage}

    def _logical_totals(self, aggregate, reused_stages):
        counters = self._REQUIRED_COUNTERS + self._OPTIONAL_COUNTERS
        return {
            counter: aggregate[counter]
            + sum(
                usage.get(counter, 0)
                for usage in reused_stages.values()
            )
            for counter in counters
        }

    def _discover_stage_paths(self, root):
        final_usage_path = root / "token_usage.json"
        if final_usage_path.is_file():
            final_usage = json.loads(
                final_usage_path.read_text(encoding="utf-8")
            )
            if final_usage.get("subtask") == "dialogue_pipeline":
                return [("final", final_usage_path)]
        stage_paths = []
        intermediate = root / "itermediate_results"
        if intermediate.is_dir():
            for stage_name in self._PREFERRED_STAGE_ORDER:
                usage_path = intermediate / stage_name / "token_usage.json"
                if usage_path.is_file():
                    stage_paths.append((stage_name, usage_path))
            known = {name for name, _ in stage_paths}
            for usage_path in sorted(
                intermediate.glob("*/token_usage.json")
            ):
                stage_name = usage_path.parent.name
                if stage_name not in known:
                    stage_paths.append((stage_name, usage_path))

        if final_usage_path.is_file():
            stage_paths.append(("final", final_usage_path))
        return stage_paths

    def _aggregate(self, stage_paths):
        counters = self._REQUIRED_COUNTERS + self._OPTIONAL_COUNTERS
        aggregate = {counter: 0 for counter in counters}
        breakdown = {}
        for fallback_name, usage_path in stage_paths:
            usage = json.loads(usage_path.read_text(encoding="utf-8"))
            self._validate_and_add(usage_path, usage, aggregate)
            stage_name = usage.get("subtask") or fallback_name
            if stage_name in breakdown:
                stage_name = fallback_name
            breakdown[stage_name] = usage
        return aggregate, breakdown

    def _validate_and_add(self, usage_path, usage, aggregate):
        for counter in self._REQUIRED_COUNTERS:
            value = self._validated_counter(
                usage_path,
                counter,
                usage.get(counter),
            )
            aggregate[counter] += value
        for counter in self._OPTIONAL_COUNTERS:
            value = self._validated_counter(
                usage_path,
                counter,
                usage.get(counter, 0),
            )
            aggregate[counter] += value
        if usage.get("provider_total_calls", 0) > usage["calls"]:
            raise ValueError(
                f"{usage_path}: provider_total_calls cannot exceed calls"
            )

    @staticmethod
    def _validated_counter(usage_path, counter, value):
        if type(value) is not int or value < 0:
            raise ValueError(
                f"{usage_path}: {counter} must be a non-negative integer"
            )
        return value

    @staticmethod
    def _attach_run_identity(root, aggregate):
        provenance_path = root / "pipeline_provenance.json"
        if not provenance_path.is_file():
            return
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )
        run_identity = provenance.get("run", {})
        if "canonical_cell_id" in run_identity:
            aggregate["canonical_cell_id"] = run_identity[
                "canonical_cell_id"
            ]
            aggregate["factors"] = run_identity.get("factors")


__all__ = ["PipelineTokenUsageAggregator"]
