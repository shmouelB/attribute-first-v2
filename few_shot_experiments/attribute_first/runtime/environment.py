"""Validated, reversible runtime feature flags."""

from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
import os


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})


class ProtocolEnvironment:
    """Apply protocol-declared flags without leaking process state."""

    def __init__(
        self,
        *,
        allowed_flags: Sequence[str],
        environment: MutableMapping[str, str] | None = None,
    ) -> None:
        normalized = tuple(dict.fromkeys(allowed_flags))
        if not normalized or any(
            not isinstance(name, str) or not name for name in normalized
        ):
            raise ValueError("allowed_flags must contain non-empty strings")
        self._allowed_flags = normalized
        self._allowed_set = frozenset(normalized)
        self._environment = (
            environment if environment is not None else os.environ
        )

    def flag(self, name: str, default: bool = False) -> bool:
        """Parse one shell boolean deterministically."""

        if name not in self._allowed_set:
            raise ValueError(f"unsupported protocol environment flag: {name}")
        raw = self._environment.get(name)
        if raw is None:
            return bool(default)
        normalized = str(raw).strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        raise ValueError(
            f"{name} must be one of "
            f"{sorted(_TRUE_VALUES | _FALSE_VALUES)}, got {raw!r}"
        )

    def snapshot(self) -> dict[str, bool]:
        """Return every allowed flag in stable declaration order."""

        return {name: self.flag(name) for name in self._allowed_flags}

    def declared_flags(
        self,
        protocol: Mapping[str, object] | None,
    ) -> dict[str, bool]:
        """Validate and copy a protocol's explicit flag declaration."""

        if protocol is None:
            return {}
        if not isinstance(protocol, Mapping):
            raise ValueError("protocol must be an object")
        declared = protocol.get("environment_flags", {})
        if not isinstance(declared, Mapping):
            raise ValueError("protocol.environment_flags must be an object")
        unknown = sorted(set(declared) - self._allowed_set)
        if unknown:
            raise ValueError(
                "unsupported protocol environment flag(s): "
                + ", ".join(unknown)
            )
        invalid_types = sorted(
            name for name, value in declared.items() if type(value) is not bool
        )
        if invalid_types:
            raise ValueError(
                "protocol.environment_flags values must be boolean: "
                + ", ".join(invalid_types)
            )
        return {name: bool(value) for name, value in declared.items()}

    @contextmanager
    def apply(
        self,
        protocol: Mapping[str, object] | None,
    ) -> Iterator[dict[str, bool]]:
        """Apply declared flags and restore the exact previous environment."""

        declared = self.declared_flags(protocol)
        previous = {
            name: self._environment[name]
            for name in declared
            if name in self._environment
        }
        absent = set(declared) - set(previous)
        for name, expected in declared.items():
            if name in previous and self.flag(name) != expected:
                raise ValueError(
                    f"{name} conflicts with protocol.environment_flags: "
                    f"environment={previous[name]!r}, declared={expected!r}"
                )

        try:
            for name, value in declared.items():
                self._environment[name] = "true" if value else "false"
            yield self.snapshot()
        finally:
            for name in absent:
                self._environment.pop(name, None)
            for name, value in previous.items():
                self._environment[name] = value
