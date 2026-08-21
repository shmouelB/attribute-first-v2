"""Persistence boundary for generation artifacts."""

from pathlib import Path
from typing import Iterable, Mapping, Protocol, runtime_checkable


@runtime_checkable
class ArtifactStore(Protocol):
    """Structural port implemented by atomic filesystem stores."""

    def write_json(self, path: str | Path, value: object) -> None:
        """Persist one JSON value atomically."""

    def write_jsonl(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, object]],
    ) -> None:
        """Persist JSON objects as newline-delimited records."""

    def read_json(self, path: str | Path) -> object:
        """Read one JSON value."""


__all__ = ["ArtifactStore"]
