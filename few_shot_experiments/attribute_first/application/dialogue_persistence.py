"""Population-ordered result and protocol persistence for dialogue runs."""

import logging
import os
from pathlib import Path


class DialogueResultPersister:
    """Validate coverage and durably write every dialogue artifact."""

    def __init__(self, dependencies):
        self._dependencies = dependencies

    def save_results(self, state):
        """Persist CS/AH/FiC results in the immutable source order."""

        plan = state.plan
        source_order = [
            instance["unique_id"] for instance in plan.alignments
        ]
        assert_coverage = self._dependencies.assert_uid_coverage
        assert_coverage(
            "content-selection results",
            state.content_selection_results,
            source_order,
        )
        assert_coverage(
            "content-selection pipeline rows",
            state.content_selection_rows,
            source_order,
        )
        assert_coverage(
            "FiC results",
            state.fusion_results,
            source_order,
        )
        assert_coverage(
            "FiC source rows",
            state.fusion_source_rows,
            source_order,
        )
        if plan.has_ambiguity_highlight:
            assert_coverage(
                "ambiguity-highlight results",
                state.ambiguity_highlight_results,
                source_order,
            )
            assert_coverage(
                "ambiguity-highlight pipeline rows",
                state.ambiguity_highlight_rows,
                source_order,
            )
        if getattr(plan, "uses_coherence_planning", False) is True:
            for stage_name in ("clustering", "reorder"):
                assert_coverage(
                    f"{stage_name} results",
                    getattr(state, f"{stage_name}_results"),
                    source_order,
                )
                assert_coverage(
                    f"{stage_name} pipeline rows",
                    getattr(state, f"{stage_name}_rows"),
                    source_order,
                )

        if getattr(
            plan,
            "shared_content_selection_reference",
            None,
        ) is None:
            cs_pipeline = [
                state.content_selection_rows[uid] for uid in source_order
            ]
            self._dependencies.save_results(
                plan.content_selection_outdir,
                state.content_selection_demos,
                state.content_selection_results,
                pipeline_format_results=cs_pipeline,
            )
        if plan.has_ambiguity_highlight:
            ah_pipeline = [
                state.ambiguity_highlight_rows[uid]
                for uid in source_order
            ]
            self._dependencies.save_results(
                plan.ambiguity_highlight_outdir,
                state.ambiguity_highlight_demos,
                state.ambiguity_highlight_results,
                pipeline_format_results=ah_pipeline,
            )
        if getattr(plan, "uses_coherence_planning", False) is True:
            for stage_name, stage_outdir in (
                ("clustering", plan.clustering_outdir),
                ("reorder", plan.reorder_outdir),
            ):
                self._dependencies.save_results(
                    stage_outdir,
                    [],
                    getattr(state, f"{stage_name}_results"),
                    pipeline_format_results=[
                        getattr(state, f"{stage_name}_rows")[uid]
                        for uid in source_order
                    ],
                )
        base_for_fic = [
            state.fusion_source_rows[uid] for uid in source_order
        ]
        if state.fusion_results:
            self._dependencies.save_results(
                plan.final_outdir,
                state.fusion_demos,
                state.fusion_results,
                pipeline_format_results=None,
            )
            fic_pipeline = plan.fusion.pipeline_fn(
                state.fusion_results,
                base_for_fic,
                structured_output=bool(
                    getattr(
                        plan.fusion.args,
                        "structured_output",
                        False,
                    )
                ),
            )
            self._dependencies.save_results(
                plan.final_outdir,
                state.fusion_demos,
                state.fusion_results,
                pipeline_format_results=fic_pipeline,
            )
        else:
            logging.error("[dialogue] No FiC results to save!")

    def persist_runtime_artifacts(self, state, cache_trace):
        """Persist cache, calls, usage, and demonstration evidence."""

        plan = state.plan
        hash_value = self._dependencies.stable_value_sha256
        snapshot = state.args_snapshot
        role_contract = snapshot["dialogue_role_contract"]
        role_contract["demonstration_sets"] = {
            "content_selection": {
                "count": len(state.content_selection_demos),
                "sha256": hash_value(state.content_selection_demos),
            },
            "ambiguity_highlight": {
                "count": len(state.ambiguity_highlight_demos),
                "sha256": hash_value(state.ambiguity_highlight_demos),
            },
            "fusion_in_context": {
                "count": len(state.fusion_demos),
                "sha256": hash_value(state.fusion_demos),
            },
        }
        if getattr(plan, "uses_coherence_planning", False) is True:
            for stage_name in ("clustering", "reorder"):
                role_contract["demonstration_sets"][stage_name] = {
                    "count": 0,
                    "sha256": hash_value([]),
                }
        snapshot["dialogue_cache_trace"] = cache_trace
        self._dependencies.artifact_store.write_json(
            os.path.join(plan.final_outdir, "args.json"),
            snapshot,
        )
        self._dependencies.artifact_store.write_jsonl(
            os.path.join(
                plan.final_outdir,
                "dialogue_calls.jsonl",
            ),
            state.call_records,
        )
        usage = (
            self._usage_from_call_records(state.call_records)
            if getattr(plan, "rerun_context", None) is not None
            else self._dependencies.get_token_usage()
        )
        logging.info(
            "[tokens] dialogue calls=%s prompt=%s completion=%s cached=%s",
            usage["calls"],
            usage["prompt"],
            usage["completion"],
            usage["cached"],
        )
        usage = dict(usage)
        usage["subtask"] = "dialogue_pipeline"
        usage["model"] = plan.model_name
        self._dependencies.artifact_store.write_json(
            os.path.join(
                plan.final_outdir,
                "token_usage.json",
            ),
            usage,
        )
        self._persist_rerun_provenance(state)

    def _persist_rerun_provenance(self, state):
        context = getattr(state.plan, "rerun_context", None)
        if context is None:
            return
        if (
            self._dependencies.artifact_sha256(context["source_path"])
            != context["source_sha256"]
        ):
            raise RuntimeError(
                "dialogue rerun parent changed while the child executed"
            )
        retried_ids = sorted(context["error_ids"])
        self._dependencies.artifact_store.write_json(
            os.path.join(
                state.plan.final_outdir,
                "rerun_provenance.json",
            ),
            {
                "schema_version": 1,
                "mode": "dialogue_terminal_error_retry",
                "parent": {
                    "results_path": str(
                        Path(context["source_path"]).resolve()
                    ),
                    "sha256": context["source_sha256"],
                },
                "derived": {
                    "outdir": str(
                        Path(state.plan.final_outdir).resolve()
                    ),
                },
                "retried_ids": retried_ids,
                "retained_ids": sorted(
                    set(context["existing_results"]) - set(retried_ids)
                ),
                "merge_policy": "replace_parent_errors_by_unique_id",
            },
        )

    @staticmethod
    def _usage_from_call_records(records):
        totals = {
            "prompt": 0,
            "completion": 0,
            "cached": 0,
            "calls": 0,
            "provider_total": 0,
            "provider_total_calls": 0,
        }
        for record in records:
            usage = record.get("usage")
            if not isinstance(usage, dict):
                continue
            totals["prompt"] += usage.get("prompt_token_count", 0)
            totals["completion"] += usage.get(
                "candidates_token_count",
                0,
            )
            totals["cached"] += usage.get(
                "cached_content_token_count",
                0,
            )
            totals["calls"] += 1
            provider_total = usage.get("total_token_count")
            if isinstance(provider_total, int):
                totals["provider_total"] += provider_total
                totals["provider_total_calls"] += 1
        return totals


__all__ = ["DialogueResultPersister"]
