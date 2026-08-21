"""Atomic JSON persistence constrained to one explicit root."""

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Callable


class CallableArtifactStore:
    """Adapt patchable legacy persistence functions to ``ArtifactStore``."""

    def __init__(
        self,
        *,
        write_json: Callable[[str | Path, object], None],
        write_jsonl: Callable[[str | Path, Iterable[Mapping]], None],
        read_json: Callable[[str | Path], object] | None = None,
    ) -> None:
        self._write_json = write_json
        self._write_jsonl = write_jsonl
        self._read_json = read_json

    def write_json(self, path: str | Path, value: object) -> None:
        self._write_json(path, value)

    def write_jsonl(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, object]],
    ) -> None:
        self._write_jsonl(path, rows)

    def read_json(self, path: str | Path) -> object:
        if self._read_json is not None:
            return self._read_json(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))


class JsonArtifactStore:
    """Filesystem adapter that rejects paths escaping its root."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str | Path, content: str) -> None:
        destination = self._resolve(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def write_json(
        self,
        path: str | Path,
        value: object,
        *,
        indent: int = 2,
    ) -> None:
        self.write_text(
            path,
            json.dumps(value, ensure_ascii=False, indent=indent) + "\n",
        )

    def write_jsonl(
        self,
        path: str | Path,
        rows: Iterable[Mapping[str, object]],
    ) -> None:
        payload = "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        )
        self.write_text(path, payload)

    def read_json(self, path: str | Path) -> object:
        return json.loads(self._resolve(path).read_text(encoding="utf-8"))

    def _resolve(self, path: str | Path) -> Path:
        relative = Path(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"artifact path must stay below {self.root}: {path}"
            )
        destination = (self.root / relative).resolve()
        try:
            destination.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                f"artifact path escapes {self.root}: {path}"
            ) from exc
        return destination


__all__ = ["CallableArtifactStore", "JsonArtifactStore"]
