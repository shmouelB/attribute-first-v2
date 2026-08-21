"""Typed registry for standard generation-stage protocols."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Protocol

from ..compatibility.stage_aliases import (
    aliases_for_stage,
    normalize_stage_name,
)
from ..domain import StageKind


class UnsupportedStandardStageError(Exception):
    """A label has no parser/converter binding in the standard runtime."""

    def __init__(self, requested_name: object) -> None:
        super().__init__(f"{requested_name} is not yet supported")
        self.requested_name = requested_name


@dataclass(frozen=True, slots=True)
class StageBinding:
    """All executable functions and schema metadata for one semantic stage."""

    kind: StageKind
    parser: Callable[..., Any]
    converter: Callable[..., Any]
    prompt_subtask_name: str
    structured_parser: Callable[..., Any] | None = None
    response_schema: Any = None
    schema_name: str | None = None
    schema_aliases: tuple[str, ...] | None = None
    aliases: tuple[str, ...] = ()
    prompt_name_overrides: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, StageKind):
            raise TypeError("kind must be a StageKind")
        if not callable(self.parser):
            raise TypeError("parser must be callable")
        if not callable(self.converter):
            raise TypeError("converter must be callable")
        if (
            not isinstance(self.prompt_subtask_name, str)
            or not self.prompt_subtask_name.strip()
        ):
            raise ValueError(
                "prompt_subtask_name must be a non-empty string"
            )
        if (
            self.structured_parser is not None
            and not callable(self.structured_parser)
        ):
            raise TypeError("structured_parser must be callable")
        if (self.response_schema is None) != (self.schema_name is None):
            raise ValueError(
                "response_schema and schema_name must be declared together"
            )
        if self.response_schema is None and self.schema_aliases is not None:
            raise ValueError(
                "schema_aliases require a response_schema"
            )
        aliases = tuple(self.aliases)
        if any(
            not isinstance(alias, str) or not alias.strip()
            for alias in aliases
        ):
            raise ValueError("aliases must be non-empty strings")
        object.__setattr__(self, "aliases", aliases)
        schema_aliases = (
            None
            if self.schema_aliases is None
            else tuple(
                normalize_stage_name(alias)
                for alias in self.schema_aliases
            )
        )
        if schema_aliases is not None and any(
            not alias for alias in schema_aliases
        ):
            raise ValueError(
                "schema_aliases must be non-empty strings"
            )
        declared_aliases = {
            *aliases_for_stage(self.kind),
            *(
                normalize_stage_name(alias)
                for alias in aliases
            ),
        }
        raw_prompt_overrides = self.prompt_name_overrides or {}
        if not isinstance(raw_prompt_overrides, Mapping):
            raise TypeError("prompt_name_overrides must be a mapping")
        prompt_overrides = {}
        for alias, prompt_name in raw_prompt_overrides.items():
            normalized_alias = normalize_stage_name(alias)
            if normalized_alias not in declared_aliases:
                raise ValueError(
                    "prompt_name_overrides keys must be declared "
                    "stage aliases"
                )
            if (
                not isinstance(prompt_name, str)
                or not prompt_name.strip()
            ):
                raise ValueError(
                    "prompt_name_overrides values must be non-empty "
                    "strings"
                )
            incumbent = prompt_overrides.get(normalized_alias)
            if incumbent is not None and incumbent != prompt_name:
                raise ValueError(
                    "prompt_name_overrides contain conflicting aliases"
                )
            prompt_overrides[normalized_alias] = prompt_name
        object.__setattr__(
            self,
            "prompt_name_overrides",
            MappingProxyType(prompt_overrides),
        )
        if schema_aliases is not None and not set(
            schema_aliases
        ).issubset(declared_aliases):
            raise ValueError(
                "schema_aliases must be declared stage aliases"
            )
        object.__setattr__(self, "schema_aliases", schema_aliases)

    @property
    def canonical_name(self) -> str:
        """Return the readable semantic stage name."""

        return self.kind.value

    def resolve(
        self,
        requested_name: str,
        *,
        structured_output: bool,
    ) -> "ResolvedStageProtocol":
        """Select one coherent parser/converter/schema protocol."""

        if structured_output and self.structured_parser is None:
            raise ValueError(
                f"{self.canonical_name} does not support structured output"
            )
        parser = (
            self.structured_parser
            if structured_output
            else self.parser
        )
        normalized = normalize_stage_name(requested_name)
        schema_enabled = (
            structured_output
            and self.response_schema is not None
            and (
                self.schema_aliases is None
                or normalized in self.schema_aliases
            )
        )
        return ResolvedStageProtocol(
            requested_name=requested_name,
            kind=self.kind,
            canonical_name=self.canonical_name,
            prompt_subtask_name=(
                self.prompt_name_overrides.get(
                    normalized,
                    self.prompt_subtask_name,
                )
            ),
            parser=parser,
            converter=self.converter,
            structured_output=structured_output,
            response_schema=(
                self.response_schema if schema_enabled else None
            ),
            schema_name=self.schema_name if schema_enabled else None,
        )


@dataclass(frozen=True, slots=True)
class ResolvedStageProtocol:
    """One executable stage binding selected for a concrete run."""

    requested_name: str
    kind: StageKind
    canonical_name: str
    prompt_subtask_name: str
    parser: Callable[..., Any]
    converter: Callable[..., Any]
    structured_output: bool
    response_schema: Any
    schema_name: str | None


class StageProtocolRegistry(Protocol):
    """Structural interface consumed by the standard application runner."""

    def resolve(
        self,
        requested_name: object,
        *,
        structured_output: bool = False,
    ) -> ResolvedStageProtocol:
        """Resolve one configured stage into its executable protocol."""


class StageRegistry:
    """Resolve legacy labels into typed executable stage bindings."""

    def __init__(self, bindings: Iterable[StageBinding]) -> None:
        by_kind: dict[StageKind, StageBinding] = {}
        by_alias: dict[str, StageBinding] = {}
        for binding in tuple(bindings):
            if not isinstance(binding, StageBinding):
                raise TypeError("bindings must contain StageBinding values")
            if binding.kind in by_kind:
                raise ValueError(
                    f"duplicate binding for {binding.kind.value}"
                )
            by_kind[binding.kind] = binding
            aliases = (
                *aliases_for_stage(binding.kind),
                *binding.aliases,
            )
            for alias in aliases:
                normalized = normalize_stage_name(alias)
                incumbent = by_alias.get(normalized)
                if incumbent is not None and incumbent.kind is not binding.kind:
                    raise ValueError(
                        f"stage alias {normalized!r} collision between "
                        f"{incumbent.kind.value} and {binding.kind.value}"
                    )
                by_alias[normalized] = binding
        if not by_kind:
            raise ValueError("at least one stage binding is required")
        self._by_kind: Mapping[StageKind, StageBinding] = MappingProxyType(
            by_kind
        )
        self._by_alias: Mapping[str, StageBinding] = MappingProxyType(
            by_alias
        )

    @property
    def bindings(self) -> tuple[StageBinding, ...]:
        """Return each registered binding in declaration order."""

        return tuple(self._by_kind.values())

    def binding_for_kind(self, kind: StageKind) -> StageBinding:
        """Return the executable binding for one semantic stage."""

        try:
            return self._by_kind[kind]
        except (KeyError, TypeError) as exc:
            requested = kind.value if isinstance(kind, StageKind) else kind
            raise UnsupportedStandardStageError(requested) from exc

    def resolve(
        self,
        requested_name: object,
        *,
        structured_output: bool = False,
    ) -> ResolvedStageProtocol:
        """Resolve one public stage label without procedural branching."""

        if type(structured_output) is not bool:
            raise TypeError("structured_output must be boolean")
        normalized = normalize_stage_name(requested_name)
        try:
            binding = self._by_alias[normalized]
        except KeyError as exc:
            raise UnsupportedStandardStageError(requested_name) from exc
        return binding.resolve(
            str(requested_name),
            structured_output=structured_output,
        )


class LegacyStageRegistryAdapter:
    """Adapt an injected legacy resolver to the typed registry contract.

    Production uses ``StageRegistry`` directly. This adapter preserves the
    long-standing monkeypatch/integration seam without putting a callable
    dispatch field back into ``StandardPipelineRunner``.
    """

    def __init__(
        self,
        canonical_registry: StageRegistry,
        resolver: Callable[..., tuple],
        subtask_schemas: Mapping[str, Any],
    ) -> None:
        if not isinstance(canonical_registry, StageRegistry):
            raise TypeError("canonical_registry must be a StageRegistry")
        if not callable(resolver):
            raise TypeError("resolver must be callable")
        self._canonical_registry = canonical_registry
        self._resolver = resolver
        self._subtask_schemas = subtask_schemas

    def resolve(
        self,
        requested_name: object,
        *,
        structured_output: bool = False,
    ) -> ResolvedStageProtocol:
        """Return typed metadata with injected legacy callables."""

        canonical = self._canonical_registry.resolve(
            requested_name,
            structured_output=structured_output,
        )
        parser, converter = self._resolver(
            requested_name,
            structured_output=structured_output,
        )
        response_schema = (
            (
                self._subtask_schemas.get(requested_name)
                or canonical.response_schema
            )
            if structured_output
            else None
        )
        return ResolvedStageProtocol(
            requested_name=str(requested_name),
            kind=canonical.kind,
            canonical_name=canonical.canonical_name,
            prompt_subtask_name=canonical.prompt_subtask_name,
            parser=parser,
            converter=converter,
            structured_output=structured_output,
            response_schema=response_schema,
            schema_name=(
                canonical.schema_name
                if response_schema is not None
                else None
            ),
        )


__all__ = [
    "LegacyStageRegistryAdapter",
    "ResolvedStageProtocol",
    "StageBinding",
    "StageProtocolRegistry",
    "StageRegistry",
    "UnsupportedStandardStageError",
]
