"""Stage-specific usage publication for stateful dialogue producers."""

from __future__ import annotations

import json
from pathlib import Path

from .shared_content_selection import (
    MANIFEST_NAME as SHARED_CONTENT_SELECTION_MANIFEST,
)


class DialogueContentSelectionUsageRepository:
    """Derive a shareable physical CS ledger from dialogue call records."""

    def __init__(self, atomic_write_json):
        self._write_json = atomic_write_json

    def persist(self, outdir):
        root = Path(outdir)
        if (root / SHARED_CONTENT_SELECTION_MANIFEST).is_file():
            return None
        call_path = root / "dialogue_calls.jsonl"
        records = [
            json.loads(line)
            for line in call_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        records = [
            record
            for record in records
            if record.get("stage") == "content_selection"
        ]
        if not records:
            raise ValueError(
                "dialogue producer has no content-selection call records"
            )
        usage = self._empty_usage(records[0].get("model_name"))
        for record in records:
            self._accumulate(record, usage)
        stage_path = (
            root
            / "itermediate_results"
            / "content_selection"
            / "token_usage.json"
        )
        self._write_json(stage_path, usage)
        return usage

    @staticmethod
    def _empty_usage(model):
        return {
            "prompt": 0,
            "completion": 0,
            "cached": 0,
            "calls": 0,
            "provider_total": 0,
            "provider_total_calls": 0,
            "subtask": "content_selection",
            "model": model,
        }

    @staticmethod
    def _accumulate(record, aggregate):
        provider = record.get("usage")
        if provider is None and record.get(
            "failure_phase"
        ) == "transport":
            return
        if not isinstance(provider, dict):
            raise ValueError(
                "dialogue content-selection call has no usage"
            )
        for provider_key, aggregate_key in (
            ("prompt_token_count", "prompt"),
            ("candidates_token_count", "completion"),
            ("cached_content_token_count", "cached"),
        ):
            value = provider.get(provider_key)
            if type(value) is not int or value < 0:
                raise ValueError(
                    "dialogue content-selection usage is invalid"
                )
            aggregate[aggregate_key] += value
        aggregate["calls"] += 1
        provider_total = provider.get("total_token_count")
        if provider_total is None:
            return
        if type(provider_total) is not int or provider_total < 0:
            raise ValueError("dialogue provider total is invalid")
        aggregate["provider_total"] += provider_total
        aggregate["provider_total_calls"] += 1


__all__ = ["DialogueContentSelectionUsageRepository"]
