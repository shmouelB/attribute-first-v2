"""Provider-neutral retry delay classification."""

from dataclasses import dataclass


def _status_number(value) -> int | None:
    """Extract an HTTP-like status from integers or enum-style values."""

    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    if type(value) is int:
        return value
    enum_value = getattr(value, "value", None)
    if type(enum_value) is int:
        return enum_value
    if (
        isinstance(enum_value, tuple)
        and enum_value
        and type(enum_value[0]) is int
    ):
        return enum_value[0]
    return None


@dataclass(frozen=True, slots=True)
class RetryDelayPolicy:
    """Choose a long wait for quota errors and a short wait otherwise."""

    rate_limit_seconds: int = 60
    default_seconds: int = 1

    _RATE_LIMIT_MARKERS = (
        "429",
        "resource exhausted",
        "resourceexhausted",
        "rate limit",
        "rate-limit",
        "quota exceeded",
        "too many requests",
    )

    @classmethod
    def is_rate_limited(cls, error: BaseException) -> bool:
        """Recognize structured provider status before textual fallbacks."""

        candidates = [
            getattr(error, "status_code", None),
            getattr(error, "code", None),
        ]
        response = getattr(error, "response", None)
        if response is not None:
            candidates.extend(
                (
                    getattr(response, "status_code", None),
                    getattr(response, "code", None),
                )
            )
        if any(_status_number(candidate) == 429 for candidate in candidates):
            return True

        description = (
            f"{type(error).__name__}: {error}"
        ).casefold()
        return any(
            marker in description
            for marker in cls._RATE_LIMIT_MARKERS
        )

    def delay_seconds(self, error: BaseException) -> int:
        """Return the deterministic retry delay for ``error``."""

        return (
            self.rate_limit_seconds
            if self.is_rate_limited(error)
            else self.default_seconds
        )


DEFAULT_RETRY_DELAY_POLICY = RetryDelayPolicy()


__all__ = [
    "DEFAULT_RETRY_DELAY_POLICY",
    "RetryDelayPolicy",
]
