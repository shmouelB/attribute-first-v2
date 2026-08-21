"""Typed preflight contracts for executable stage configuration files."""

from collections.abc import Mapping
from dataclasses import dataclass

from ..compatibility.stage_aliases import resolve_stage_alias
from ..domain import (
    ModelProvider,
    RetryPolicy,
    StageKind,
    TokenBudget,
)


@dataclass(frozen=True, slots=True)
class GenerationDefaults:
    """One source of truth for supervisor-approved generation defaults."""

    model_name: str = "models/gemini-3-flash-preview"
    structured_output: bool = True
    use_roles: bool = True

    def structured_output_for(self, configuration: object) -> bool:
        """Resolve a namespace or mapping without a legacy false fallback."""

        if isinstance(configuration, Mapping):
            value = configuration.get(
                "structured_output",
                self.structured_output,
            )
        else:
            value = getattr(
                configuration,
                "structured_output",
                self.structured_output,
            )
        if type(value) is not bool:
            raise ValueError("structured_output must be boolean")
        return value

    def protocol_with_role_default(
        self,
        protocol: Mapping[str, object] | None,
        *,
        use_roles: bool | None = None,
    ) -> dict[str, object]:
        """Add the role default without overriding an explicit protocol."""

        if protocol is None:
            effective_protocol: dict[str, object] = {}
        elif isinstance(protocol, Mapping):
            effective_protocol = dict(protocol)
        else:
            raise ValueError("protocol must be an object")

        requested_roles = self.use_roles if use_roles is None else use_roles
        if type(requested_roles) is not bool:
            raise ValueError("use_roles must be boolean")

        declared_flags = effective_protocol.get("environment_flags", {})
        if not isinstance(declared_flags, Mapping):
            raise ValueError(
                "protocol.environment_flags must be an object"
            )
        effective_flags = dict(declared_flags)
        effective_flags.setdefault("AF_USE_ROLES", requested_roles)
        effective_protocol["environment_flags"] = effective_flags
        if (
            effective_flags["AF_USE_ROLES"]
            and "prompt_transport" not in effective_protocol
        ):
            effective_protocol[
                "prompt_transport"
            ] = "system-user-model-roles"
        return effective_protocol


DEFAULT_GENERATION = GenerationDefaults()


@dataclass(frozen=True, slots=True)
class StageConfigContract:
    """Validated runtime fields shared by execution and provenance."""

    declared_subtask: str
    configured_subtask: str
    kind: StageKind
    model_name: str
    model_provider: ModelProvider
    demonstration_count: int
    retry_policy: RetryPolicy
    token_budget: TokenBudget
    temperature: int | float
    output_token_limit: int
    structured_output: bool

    @classmethod
    def from_mapping(
        cls,
        config: object,
        *,
        declared_subtask: object,
        defaults: Mapping[str, object] | None = None,
        strict: bool = True,
        require_budget_contract: bool = False,
    ) -> "StageConfigContract":
        """Validate one config without mutating it or touching providers."""

        if not isinstance(config, Mapping):
            raise ValueError("stage config must be an object")
        if type(strict) is not bool:
            raise TypeError("strict must be boolean")
        if type(require_budget_contract) is not bool:
            raise TypeError(
                "require_budget_contract must be boolean"
            )
        effective_defaults = defaults or {}
        if not isinstance(effective_defaults, Mapping):
            raise TypeError("defaults must be a mapping")

        required_fields = (
            "subtask",
            "model_name",
            "prompt_token_budget",
            "dialogue_history_token_budget",
            "n_demos",
            "num_retries",
            "temperature",
            "structured_output",
            "output_max_length",
        )
        required = (
            required_fields
            if strict
            else (
                (
                    "prompt_token_budget",
                    "dialogue_history_token_budget",
                    "num_retries",
                )
                if require_budget_contract
                else ()
            )
        )
        missing = [
            field for field in required if field not in config
        ]
        if missing:
            qualifier = "controlled " if strict else ""
            raise ValueError(
                f"{qualifier}stage config is missing explicit fields: "
                + ", ".join(missing)
            )

        def effective_value(name, fallback=None):
            if name in config:
                return config[name]
            return effective_defaults.get(name, fallback)

        try:
            declared_kind = resolve_stage_alias(declared_subtask)
        except ValueError as exc:
            raise ValueError(
                f"unsupported declared subtask {declared_subtask!r}"
            ) from exc

        configured_subtask = effective_value(
            "subtask",
            declared_subtask,
        )
        try:
            configured_kind = resolve_stage_alias(configured_subtask)
        except ValueError as exc:
            raise ValueError(
                "stage config subtask must be a supported non-empty alias"
            ) from exc
        if configured_kind is not declared_kind:
            raise ValueError(
                "pipeline stage kind conflicts with its config: "
                f"{declared_subtask!r} resolves to {declared_kind.value}, "
                f"but {configured_subtask!r} resolves to "
                f"{configured_kind.value}"
            )

        model_name = effective_value(
            "model_name",
            DEFAULT_GENERATION.model_name,
        )
        try:
            model_provider = ModelProvider.from_model_id(model_name)
        except ValueError as exc:
            raise ValueError(
                "model_name must be a non-empty full provider model ID"
            ) from exc

        demonstration_count = effective_value("n_demos", 2)
        if (
            type(demonstration_count) is not int
            or demonstration_count < 0
        ):
            raise ValueError(
                "n_demos must be a non-negative integer"
            )

        try:
            retry_policy = RetryPolicy(
                max_attempts=effective_value("num_retries", 1)
            )
        except ValueError as exc:
            raise ValueError(f"invalid num_retries: {exc}") from exc
        prompt_budget = effective_value(
            "prompt_token_budget",
            30000,
        )
        try:
            token_budget = TokenBudget(
                stage_prompt_tokens=prompt_budget,
                dialogue_history_prompt_tokens=effective_value(
                    "dialogue_history_token_budget",
                    prompt_budget,
                ),
            )
        except ValueError as exc:
            raise ValueError(f"invalid token budget: {exc}") from exc

        temperature = effective_value("temperature", 0.2)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")

        output_token_limit = effective_value(
            "output_max_length",
            4096,
        )
        if (
            type(output_token_limit) is not int
            or output_token_limit <= 0
        ):
            raise ValueError(
                "output_max_length must be a positive integer"
            )

        structured_output = effective_value(
            "structured_output",
            DEFAULT_GENERATION.structured_output,
        )
        if type(structured_output) is not bool:
            raise ValueError("structured_output must be boolean")

        return cls(
            declared_subtask=str(declared_subtask),
            configured_subtask=str(configured_subtask),
            kind=declared_kind,
            model_name=model_name,
            model_provider=model_provider,
            demonstration_count=demonstration_count,
            retry_policy=retry_policy,
            token_budget=token_budget,
            temperature=temperature,
            output_token_limit=output_token_limit,
            structured_output=structured_output,
        )


__all__ = [
    "DEFAULT_GENERATION",
    "GenerationDefaults",
    "StageConfigContract",
]
