"""Provider-independent model requests used by application runners."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One stateless generation request independent of an SDK."""

    model_name: str
    prompt: str
    output_max_length: int = 2048
    temperature: float = 0
    response_schema: object | None = None
    model_override: object | None = None
    contents: object | None = None
    system_instruction: str | None = None


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """Parameters needed to create one stateful conversation."""

    model_name: str
    cached_content: object | None = None
    system_instruction: str | None = None
    history: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatTurnRequest:
    """One new user task appended to an existing conversation."""

    message: object
    output_max_length: int = 4096
    temperature: float = 0
    response_schema: object | None = None


@runtime_checkable
class GenerationGateway(Protocol):
    """Port implemented by providers supporting independent generation."""

    def generate(self, request: GenerationRequest) -> str:
        """Generate one independent response."""


@runtime_checkable
class DialogueGateway(Protocol):
    """Port implemented by providers supporting stateful dialogue."""

    def create_chat(self, request: ChatRequest) -> object:
        """Create a stateful conversation."""

    def send_message(
        self,
        chat: object,
        request: ChatTurnRequest,
    ) -> str:
        """Append one user request to a stateful conversation."""


@runtime_checkable
class ModelGateway(GenerationGateway, DialogueGateway, Protocol):
    """Combined port for providers supporting both transport modes."""


@dataclass(frozen=True, slots=True)
class BatchGenerationRequest:
    """One legacy-compatible batch submitted through an application port."""

    prompts: Mapping[str, str]
    model_name: str
    parse_response: Callable[..., dict[str, object]]
    output_max_length: int
    num_retries: int
    temperature: float
    response_schema: object | None
    concurrency: int
    role_messages: Mapping[str, object]


@runtime_checkable
class BatchGenerationGateway(Protocol):
    """Port for standard-stage batch generation and concurrency."""

    def generate_batch(
        self,
        request: BatchGenerationRequest,
    ) -> dict[str, dict[str, object]]:
        """Generate and parse every keyed prompt."""


__all__ = [
    "BatchGenerationGateway",
    "BatchGenerationRequest",
    "ChatRequest",
    "ChatTurnRequest",
    "DialogueGateway",
    "GenerationGateway",
    "GenerationRequest",
    "ModelGateway",
]
