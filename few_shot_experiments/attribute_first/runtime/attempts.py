"""Provider-independent retry and parsing orchestration."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .retry_policy import DEFAULT_RETRY_DELAY_POLICY


class IncompleteGenerationError(ValueError):
    """A provider response is not safe to pass to a parser."""


PARSEABLE_FINISH_REASONS = frozenset(
    {"FINISH_REASON_UNSPECIFIED", "STOP"}
)


def ensure_parseable_finish_reason(
    metadata: Mapping[str, object] | None,
) -> None:
    """Reject partial or blocked provider terminations."""

    finish_reason = (
        metadata.get("finish_reason")
        if isinstance(metadata, Mapping)
        else None
    )
    if finish_reason not in PARSEABLE_FINISH_REASONS:
        raise IncompleteGenerationError(
            "provider stopped generation with "
            f"finish_reason={finish_reason}"
        )


@dataclass(frozen=True)
class AttemptPolicy:
    """Stable request settings persisted in each attempt record."""

    model_name: str
    output_max_length: int
    num_retries: int
    temperature: float

    def __post_init__(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name:
            raise ValueError("model_name must be a non-empty string")
        if type(self.output_max_length) is not int or self.output_max_length < 1:
            raise ValueError("output_max_length must be a positive integer")
        if type(self.num_retries) is not int or self.num_retries < 1:
            raise ValueError("num_retries must be a positive integer")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError(
                "temperature must be numeric, not boolean, and in [0, 2]"
            )

    def evidence(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "output_max_length": self.output_max_length,
            "num_retries": self.num_retries,
        }


@dataclass(frozen=True)
class AttemptDependencies:
    """Side effects injected into :class:`AttemptExecutor`."""

    invoke: Callable[[], str]
    parse: Callable[..., dict[str, object]]
    reset_evidence: Callable[[], None]
    last_usage: Callable[[], dict[str, int] | None]
    last_metadata: Callable[[], dict[str, object] | None]
    ensure_parseable: Callable[[Mapping[str, object] | None], None]
    fingerprint: Callable[[object], str]
    sleep: Callable[[float], None]
    incomplete_error: type[Exception] = IncompleteGenerationError


class AttemptExecutor:
    """Execute retryable generation calls and preserve every outcome."""

    def __init__(self, dependencies: AttemptDependencies) -> None:
        self._dependencies = dependencies

    def execute(
        self,
        *,
        prompt: str,
        policy: AttemptPolicy,
        application_request: Mapping[str, object],
        response_schema: object = None,
    ) -> dict[str, object]:
        attempts: list[dict[str, object]] = []
        parsed_response: dict[str, object] | None = None
        last_response: str | None = None
        error_message = "generation produced no parseable response"

        for attempt_number in range(1, policy.num_retries + 1):
            attempt = self._attempt_record(
                attempt_number=attempt_number,
                policy=policy,
                application_request=application_request,
                response_schema=response_schema,
            )
            response_received = False
            try:
                self._dependencies.reset_evidence()
                response = self._dependencies.invoke()
                response_received = True
                last_response = response
                attempt["raw_response"] = response
                attempt["usage"] = self._dependencies.last_usage()
                attempt["response_metadata"] = (
                    self._dependencies.last_metadata()
                )
                self._dependencies.ensure_parseable(
                    attempt["response_metadata"]
                )
                parsed_response = self._dependencies.parse(
                    response=response,
                    prompt=prompt,
                )
                if not isinstance(parsed_response, dict):
                    raise TypeError("parser must return a result object")
                parsed_response["prompt"] = prompt
                attempt["status"] = "parsed"
                attempts.append(attempt)
                parsed_response["attempt_trace"] = attempts
                return parsed_response
            except (KeyError, AttributeError, TypeError):
                raise
            except Exception as exception:
                self._record_failure(
                    attempt,
                    exception=exception,
                    response_received=response_received,
                )
                attempts.append(attempt)
                print(f"{exception}. Retrying...")
                error_message = str(exception)
                if attempt_number < policy.num_retries:
                    wait_seconds = (
                        DEFAULT_RETRY_DELAY_POLICY.delay_seconds(exception)
                    )
                    self._dependencies.sleep(wait_seconds)

        return {
            "final_output": f"ERROR - {error_message}",
            "full_model_response": last_response or "",
            "prompt": prompt,
            "attempt_trace": attempts,
        }

    def _attempt_record(
        self,
        *,
        attempt_number: int,
        policy: AttemptPolicy,
        application_request: Mapping[str, object],
        response_schema: object,
    ) -> dict[str, object]:
        return {
            "attempt": attempt_number,
            "application_request": dict(application_request),
            "effective_request": policy.evidence(),
            "response_schema_sha256": (
                self._dependencies.fingerprint(response_schema)
                if response_schema is not None
                else None
            ),
        }

    def _record_failure(
        self,
        attempt: dict[str, object],
        *,
        exception: Exception,
        response_received: bool,
    ) -> None:
        attempt.setdefault("usage", self._dependencies.last_usage())
        attempt.setdefault(
            "response_metadata",
            self._dependencies.last_metadata(),
        )
        attempt["status"] = "error"
        if isinstance(exception, self._dependencies.incomplete_error):
            failure_phase = "generation"
        elif response_received:
            failure_phase = "parse"
        elif isinstance(attempt.get("response_metadata"), dict):
            failure_phase = "provider_response"
        else:
            failure_phase = "transport"
        attempt["failure_phase"] = failure_phase
        attempt["error"] = (
            f"{type(exception).__name__}: {exception}"
        )
