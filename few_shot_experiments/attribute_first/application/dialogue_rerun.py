"""Append-only recovery support for stateful dialogue pipelines."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


class DialogueRerunService:
    """Hydrate retained parent outcomes before retrying terminal errors."""

    _PLANNED_STAGES = (
        "ambiguity_highlight",
        "clustering",
        "reorder",
    )

    @staticmethod
    def active_ids(context, expected_ids):
        """Return validated parent error IDs in canonical population order."""

        expected = list(expected_ids)
        errors = set(context.get("error_ids", ()))
        unknown = errors - set(expected)
        if unknown:
            raise ValueError(
                "dialogue rerun parent contains unknown ERROR IDs: "
                + ", ".join(sorted(unknown))
            )
        if not errors:
            raise ValueError("dialogue rerun parent contains no ERROR outputs")
        return [unique_id for unique_id in expected if unique_id in errors]

    def hydrate(self, state, shared_entries):
        """Seed all retained UIDs while leaving parent errors unpopulated."""

        context = state.plan.rerun_context
        if context is None:
            return
        expected_ids = [
            str(row["unique_id"]) for row in state.plan.alignments
        ]
        active_ids = set(self.active_ids(context, expected_ids))
        retained_ids = [
            unique_id
            for unique_id in expected_ids
            if unique_id not in active_ids
        ]
        parent_results = context.get("existing_results")
        if not isinstance(parent_results, dict) or set(parent_results) != set(
            expected_ids
        ):
            raise ValueError(
                "dialogue rerun parent results do not cover the canonical "
                "population"
            )
        if not isinstance(shared_entries, dict) or not set(
            retained_ids
        ).issubset(shared_entries):
            raise ValueError(
                "dialogue rerun cannot hydrate retained content selection"
            )

        for unique_id in retained_ids:
            checkpoint = shared_entries[unique_id]
            state.content_selection_results[unique_id] = deepcopy(
                checkpoint["result"]
            )
            state.content_selection_rows[unique_id] = deepcopy(
                checkpoint["pipeline_row"]
            )
            state.fusion_results[unique_id] = deepcopy(
                parent_results[unique_id]
            )

        parent_root = Path(context["source_path"]).parent
        if state.plan.has_ambiguity_highlight:
            self._hydrate_stage(
                state,
                parent_root,
                "ambiguity_highlight",
                retained_ids,
            )
            state.fusion_source_rows.update(
                deepcopy(state.ambiguity_highlight_rows)
            )
        else:
            state.fusion_source_rows.update(
                deepcopy(state.content_selection_rows)
            )
        if state.plan.uses_coherence_planning:
            for stage_name in ("clustering", "reorder"):
                self._hydrate_stage(
                    state,
                    parent_root,
                    stage_name,
                    retained_ids,
                )
        state.call_records.extend(
            record
            for record in self._read_jsonl(
                parent_root / "dialogue_calls.jsonl"
            )
            if record.get("unique_id") in set(retained_ids)
        )

    @classmethod
    def _hydrate_stage(
        cls,
        state,
        parent_root,
        stage_name,
        retained_ids,
    ):
        stage_root = parent_root / "itermediate_results" / stage_name
        results = cls._read_json(stage_root / "results.json")
        rows = {
            str(row["unique_id"]): row
            for row in cls._read_jsonl(
                stage_root / "pipeline_format_results.json"
            )
        }
        retained = set(retained_ids)
        if not retained.issubset(results) or not retained.issubset(rows):
            raise ValueError(
                f"dialogue rerun parent {stage_name} artifacts are incomplete"
            )
        getattr(state, f"{stage_name}_results").update(
            {
                unique_id: deepcopy(results[unique_id])
                for unique_id in retained_ids
            }
        )
        getattr(state, f"{stage_name}_rows").update(
            {
                unique_id: deepcopy(rows[unique_id])
                for unique_id in retained_ids
            }
        )

    @staticmethod
    def _read_json(path):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid dialogue rerun artifact {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"dialogue rerun artifact is not an object: {path}")
        return value

    @staticmethod
    def _read_jsonl(path):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            values = [json.loads(line) for line in lines if line.strip()]
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid dialogue rerun artifact {path}") from exc
        if any(not isinstance(value, dict) for value in values):
            raise ValueError(
                f"dialogue rerun JSONL contains a non-object: {path}"
            )
        return values


__all__ = ["DialogueRerunService"]
