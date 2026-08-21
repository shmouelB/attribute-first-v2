"""Checkpoint and restore shared content selection in dialogue pipelines."""

from __future__ import annotations

from copy import deepcopy
import json
import os

from ..runtime.conversation import Conversation


class DialogueContentSelectionCheckpointService:
    """Persist live CS state and restore it without a provider CS turn."""

    def __init__(self, dependencies, session_service):
        self._dependencies = dependencies
        self._session_service = session_service

    def capture(self, state, instance):
        """Capture the exact post-pruning live exchange for one producer."""

        uid = instance.uid
        history = self._dependencies.jsonable_dialogue_value(
            Conversation.wrap(instance.session).history
        )
        result = deepcopy(state.content_selection_results[uid])
        row = deepcopy(state.content_selection_rows[uid])
        status = (
            "error"
            if (
                isinstance(result.get("final_output"), str)
                and result["final_output"].lstrip().startswith("ERROR")
            )
            else "parsed"
        )
        if status == "parsed":
            self._validate_live_history(history, uid)
        system_instruction = (
            instance.role_payload["system"]
            if instance.role_payload is not None
            else None
        )
        hash_value = self._dependencies.stable_value_sha256
        state.content_selection_checkpoints[uid] = {
            "status": status,
            "system_instruction": system_instruction,
            "history": history,
            "result": result,
            "pipeline_row": row,
            "protocol": deepcopy(instance.protocol),
            "hashes": {
                "system_instruction": hash_value(system_instruction),
                "history": hash_value(history),
                "result": hash_value(result),
                "pipeline_row": hash_value(row),
                "protocol": hash_value(instance.protocol),
            },
        }

    def persist(self, state):
        """Write one producer checkpoint after every UID is captured."""

        plan = state.plan
        if getattr(
            plan,
            "shared_content_selection_reference",
            None,
        ) is not None:
            return
        expected_ids = {
            row["unique_id"] for row in plan.alignments
        }
        if set(state.content_selection_checkpoints) != expected_ids:
            raise RuntimeError(
                "dialogue content-selection checkpoint coverage mismatch"
            )
        payload = {
            "schema_version": 1,
            "stage": "content_selection",
            "producer_canonical_id": state.args_snapshot.get(
                "canonical_cell_id"
            ),
            "transport": "dialogue",
            "model_name": plan.model_name,
            "unique_ids": sorted(expected_ids),
            "dialogues": {
                uid: state.content_selection_checkpoints[uid]
                for uid in sorted(expected_ids)
            },
        }
        self._dependencies.artifact_store.write_json(
            os.path.join(
                plan.content_selection_outdir,
                "dialogue_checkpoint.json",
            ),
            payload,
        )

    def load(self, plan):
        """Load and validate the immutable checkpoint snapshot."""

        reference = getattr(
            plan,
            "shared_content_selection_reference",
            None,
        )
        if reference is None:
            return None
        checkpoint_path = reference.snapshot_for(
            "dialogue_checkpoint.json"
        )
        try:
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid dialogue content-selection checkpoint: {exc}"
            ) from exc
        expected_ids = sorted(
            row["unique_id"] for row in plan.alignments
        )
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version") != 1
            or checkpoint.get("stage") != "content_selection"
            or checkpoint.get("transport") != "dialogue"
            or checkpoint.get("model_name") != plan.model_name
            or checkpoint.get("producer_canonical_id")
            != reference.producer_canonical_id
            or checkpoint.get("unique_ids") != expected_ids
            or set(checkpoint.get("dialogues", {})) != set(expected_ids)
        ):
            raise ValueError(
                "dialogue content-selection checkpoint contract mismatch"
            )
        for uid in expected_ids:
            self._validate_entry(checkpoint["dialogues"][uid], uid)
        return checkpoint["dialogues"]

    def restore(self, state, instance, entry, source_row):
        """Seed one consumer and continue from a fresh local chat session."""

        result = deepcopy(entry["result"])
        row = deepcopy(entry["pipeline_row"])
        uid = instance.uid
        if row.get("unique_id") != uid:
            raise ValueError(
                f"shared dialogue checkpoint UID mismatch for {uid!r}"
            )
        instance.protocol = deepcopy(entry["protocol"])
        instance.protocol.update(
            {
                "cs_execution_mode": "reused",
                "cs_execution_id": (
                    state.plan.shared_content_selection_reference
                    .equivalence_sha256
                ),
                "cs_checkpoint_history_sha256": entry["hashes"][
                    "history"
                ],
            }
        )
        state.content_selection_results[uid] = result
        state.content_selection_rows[uid] = row
        if entry["status"] == "error":
            self._seed_failure(state, instance, row, source_row)
            return False, row
        instance.session = self._session_service.create_chat(
            state.plan.model_name,
            system_instruction=entry["system_instruction"],
            history=entry["history"],
        )
        return True, row

    def _validate_entry(self, entry, uid):
        if (
            not isinstance(entry, dict)
            or entry.get("status") not in {"parsed", "error"}
            or not isinstance(entry.get("history"), list)
            or not isinstance(entry.get("result"), dict)
            or not isinstance(entry.get("pipeline_row"), dict)
            or not isinstance(entry.get("protocol"), dict)
            or not isinstance(entry.get("hashes"), dict)
        ):
            raise ValueError(
                f"invalid dialogue checkpoint entry for {uid!r}"
            )
        hash_value = self._dependencies.stable_value_sha256
        for field in (
            "system_instruction",
            "history",
            "result",
            "pipeline_row",
            "protocol",
        ):
            if entry["hashes"].get(field) != hash_value(entry[field]):
                raise ValueError(
                    f"dialogue checkpoint {uid!r} {field} hash mismatch"
                )
        if entry["status"] == "parsed":
            self._validate_live_history(entry["history"], uid)

    @staticmethod
    def _validate_live_history(history, uid):
        if (
            not isinstance(history, list)
            or len(history) != 2
            or [
                message.get("role")
                if isinstance(message, dict)
                else None
                for message in history
            ]
            != ["user", "model"]
        ):
            raise ValueError(
                f"dialogue checkpoint {uid!r} must contain exactly the "
                "pruned live CS exchange"
            )

    def _seed_failure(self, state, instance, row, source_row):
        plan = state.plan
        uid = instance.uid
        if plan.has_ambiguity_highlight:
            error = {
                "final_output": (
                    "ERROR - upstream content_selection failed"
                ),
                "alignments": [],
                "upstream_skipped_reason": "model_error",
                "dialogue_attempt_trace": deepcopy(instance.trace),
                "dialogue_protocol_trace": deepcopy(instance.protocol),
            }
            state.ambiguity_highlight_results[uid] = error
            ah_row = self._dependencies.single_pipeline_row(
                plan.ambiguity_highlight.pipeline_fn,
                uid,
                error,
                [row],
                "ambiguity_highlight",
            )
            state.ambiguity_highlight_rows[uid] = ah_row
            state.fusion_source_rows[uid] = ah_row
        else:
            state.fusion_source_rows[uid] = row
        final_error = deepcopy(state.content_selection_results[uid])
        final_error.pop("prompt_budget_trace", None)
        final_error["upstream_skipped_reason"] = "model_error"
        final_error["dialogue_attempt_trace"] = deepcopy(instance.trace)
        final_error["dialogue_protocol_trace"] = deepcopy(
            instance.protocol
        )
        state.fusion_results[uid] = self._dependencies.with_gold_summary(
            final_error,
            source_row,
        )


__all__ = ["DialogueContentSelectionCheckpointService"]
