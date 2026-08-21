"""Scoped token accounting and per-call provider evidence."""

from collections.abc import Callable, MutableMapping
import threading
from typing import Any


_ZERO_TOTALS = {
    "prompt": 0,
    "completion": 0,
    "cached": 0,
    "calls": 0,
    "provider_total": 0,
    "provider_total_calls": 0,
}


class UsageLedger:
    """Own aggregate usage while keeping last-call evidence thread-local.

    ``totals_provider`` exists solely for the legacy ``utils`` facade, whose
    historical tests replace its public ``_TOKEN_USAGE`` mapping. New code
    should rely on the default instance-owned state.
    """

    def __init__(
        self,
        *,
        totals: MutableMapping[str, int] | None = None,
        totals_provider: Callable[[], MutableMapping[str, int]] | None = None,
        lock: object | None = None,
        usage_local: threading.local | None = None,
        metadata_local: threading.local | None = None,
    ) -> None:
        if totals is not None and totals_provider is not None:
            raise ValueError("provide totals or totals_provider, not both")
        owned_totals = totals if totals is not None else dict(_ZERO_TOTALS)
        self._totals_provider = totals_provider or (lambda: owned_totals)
        self._lock = lock or threading.RLock()
        self._usage_local = usage_local or threading.local()
        self._metadata_local = metadata_local or threading.local()
        self._validate_totals(self._totals_provider())

    @staticmethod
    def _validate_totals(totals: MutableMapping[str, int]) -> None:
        missing = sorted(set(_ZERO_TOTALS) - set(totals))
        if missing:
            raise ValueError(
                "usage totals are missing keys: " + ", ".join(missing)
            )

    @staticmethod
    def _count(value: object, field: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
        return value

    def reset(self) -> None:
        """Reset aggregate and current-thread evidence."""

        with self._lock:
            totals = self._totals_provider()
            self._validate_totals(totals)
            for key in _ZERO_TOTALS:
                totals[key] = 0
        self.clear_last()

    def snapshot(self) -> dict[str, int]:
        """Return an immutable copy of aggregate usage."""

        with self._lock:
            totals = self._totals_provider()
            self._validate_totals(totals)
            return {key: int(totals[key]) for key in _ZERO_TOTALS}

    def clear_last(self) -> None:
        """Clear evidence for the current worker thread."""

        self._usage_local.value = None
        self._metadata_local.value = None

    def last_usage(self) -> dict[str, int] | None:
        usage = getattr(self._usage_local, "value", None)
        return dict(usage) if isinstance(usage, dict) else None

    def last_metadata(self) -> dict[str, Any] | None:
        metadata = getattr(self._metadata_local, "value", None)
        return dict(metadata) if isinstance(metadata, dict) else None

    def record(
        self,
        *,
        prompt: int,
        completion: int,
        cached: int,
        metadata: dict[str, Any],
        provider_total: int | None = None,
    ) -> dict[str, int]:
        """Record one provider response and return its exact usage object."""

        prompt = self._count(prompt, "prompt")
        completion = self._count(completion, "completion")
        cached = self._count(cached, "cached")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")

        usage = {
            "prompt_token_count": prompt,
            "candidates_token_count": completion,
            "cached_content_token_count": cached,
        }
        if provider_total is not None:
            provider_total = self._count(
                provider_total,
                "provider_total",
            )
            usage["total_token_count"] = provider_total

        self._usage_local.value = usage
        self._metadata_local.value = dict(metadata)
        with self._lock:
            totals = self._totals_provider()
            self._validate_totals(totals)
            totals["prompt"] += prompt
            totals["completion"] += completion
            totals["cached"] += cached
            totals["calls"] += 1
            if provider_total is not None:
                totals["provider_total"] += provider_total
                totals["provider_total_calls"] += 1
        return dict(usage)

    def record_metadata_only(self, metadata: dict[str, Any]) -> None:
        """Retain provider termination evidence when usage is unavailable."""

        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        self._usage_local.value = None
        self._metadata_local.value = dict(metadata)
