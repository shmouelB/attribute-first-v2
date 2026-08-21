"""Validated identifiers used at persistence boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import re


_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class RunId:
    """A portable, single-directory identifier for an append-only run."""

    value: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or _RUN_ID_PATTERN.fullmatch(self.value) is None
        ):
            raise ValueError(
                "run id must start with an ASCII letter or digit and "
                "contain only ASCII letters, digits, '.', '_' or '-'"
            )

    @classmethod
    def parse(cls, value: str | "RunId") -> "RunId":
        """Validate external input, preserving an existing value object."""

        if isinstance(value, cls):
            return value
        return cls(value)

    def __str__(self) -> str:
        return self.value
