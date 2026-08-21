"""Concrete provider adapters behind the generation port."""

from collections.abc import Callable, MutableMapping, Sequence

from ..ports import (
    BatchGenerationRequest,
    ChatRequest,
    ChatTurnRequest,
    GenerationRequest,
)


class CallableGenerationGateway:
    """Adapt one patchable generation callable to the typed port."""

    def __init__(
        self,
        generate: Callable[[GenerationRequest], str],
    ) -> None:
        self._generate = generate

    def generate(self, request: GenerationRequest) -> str:
        return self._generate(request)


class CallableDialogueGateway:
    """Adapt legacy chat callables without leaking them into application code."""

    def __init__(
        self,
        *,
        create_chat: Callable[..., object],
        send_message: Callable[..., str],
    ) -> None:
        self._create_chat = create_chat
        self._send_message = send_message

    def create_chat(self, request: ChatRequest) -> object:
        return self._create_chat(
            request.model_name,
            cached_content=request.cached_content,
            system_instruction=request.system_instruction,
            history=list(request.history),
        )

    def send_message(
        self,
        chat: object,
        request: ChatTurnRequest,
    ) -> str:
        return self._send_message(
            chat,
            request.message,
            output_max_length=request.output_max_length,
            temperature=request.temperature,
            response_schema=request.response_schema,
        )


class CallableBatchGenerationGateway:
    """Adapt the historical batch helper to an explicit application port."""

    def __init__(self, generate_batch: Callable[..., dict]) -> None:
        self._generate_batch = generate_batch

    def generate_batch(
        self,
        request: BatchGenerationRequest,
    ) -> dict[str, dict[str, object]]:
        return self._generate_batch(
            prompts=request.prompts,
            model_name=request.model_name,
            parse_response_fn=request.parse_response,
            output_max_length=request.output_max_length,
            num_retries=request.num_retries,
            temperature=request.temperature,
            response_schema=request.response_schema,
            concurrency=request.concurrency,
            role_messages=request.role_messages,
        )


class GeminiGateway:
    """Adapt the pinned Gemini SDK to provider-independent requests."""

    def __init__(
        self,
        *,
        sdk: object,
        content_types: object,
        normalize_model_name: Callable[[str], str],
        record_usage: Callable[[object], object],
        ensure_parseable: Callable[[], None],
        reset_evidence: Callable[[], None],
        last_metadata: Callable[[], dict[str, object] | None],
        safety_settings: Sequence[dict[str, str]],
        flat_model: object | None = None,
        flat_model_name: str | None = None,
        role_models: MutableMapping[tuple[str, str], object] | None = None,
    ) -> None:
        self._sdk = sdk
        self._content_types = content_types
        self._normalize_model_name = normalize_model_name
        self._record_usage = record_usage
        # Kept as an accepted constructor dependency for compatibility.
        # Finish-reason classification belongs to the retry executors, after
        # they have captured the provider's raw text, usage, and metadata.
        self._ensure_parseable = ensure_parseable
        self._reset_evidence = reset_evidence
        self._last_metadata = last_metadata
        self._safety_settings = list(safety_settings)
        self.flat_model = flat_model
        self.flat_model_name = flat_model_name
        self._role_models = role_models if role_models is not None else {}

    def create_chat(self, request: ChatRequest) -> object:
        history = list(request.history)
        if request.cached_content is not None:
            if request.system_instruction is not None or history:
                raise ValueError(
                    "cached dialogue context already owns its system "
                    "instruction and history; do not provide them twice"
                )
            model = self._sdk.GenerativeModel.from_cached_content(
                cached_content=request.cached_content
            )
        else:
            model = self._sdk.GenerativeModel(
                self._normalize_model_name(request.model_name),
                system_instruction=request.system_instruction,
            )
        return model.start_chat(history=history)

    def send_message(
        self,
        chat: object,
        request: ChatTurnRequest,
    ) -> str:
        generation_config = self._generation_config(
            output_max_length=request.output_max_length,
            temperature=request.temperature,
            response_schema=request.response_schema,
        )
        self._reset_evidence()
        content = self._content_types.to_content(request.message)
        if not content.role:
            content.role = "user"
        history = list(chat.history)
        history.append(content)
        response = chat.model.generate_content(
            contents=history,
            generation_config=generation_config,
            safety_settings=self._safety_settings,
        )
        self._record_usage(response)
        chat._check_response(response=response, stream=False)
        response_text = self._response_text(response)
        if (self._last_metadata() or {}).get("finish_reason") != "MAX_TOKENS":
            chat._last_sent = content
            chat._last_received = response
        return response_text

    def generate(self, request: GenerationRequest) -> str:
        self._reset_evidence()
        model = self._model_for(request)
        response = model.generate_content(
            (
                request.contents
                if request.contents is not None
                else request.prompt
            ),
            generation_config=self._generation_config(
                output_max_length=request.output_max_length,
                temperature=request.temperature,
                response_schema=request.response_schema,
            ),
            safety_settings=self._safety_settings,
        )
        self._record_usage(response)
        return self._response_text(response)

    def _model_for(self, request: GenerationRequest) -> object:
        if request.model_override is not None:
            return request.model_override
        normalized = self._normalize_model_name(request.model_name)
        if request.contents is not None:
            key = (normalized, request.system_instruction or "")
            model = self._role_models.get(key)
            if model is None:
                model = self._sdk.GenerativeModel(
                    normalized,
                    system_instruction=request.system_instruction,
                )
                self._role_models[key] = model
            return model
        if self.flat_model is None or self.flat_model_name != normalized:
            self.flat_model = self._sdk.GenerativeModel(normalized)
            self.flat_model_name = normalized
        return self.flat_model

    @staticmethod
    def _generation_config(
        *,
        output_max_length: int,
        temperature: float,
        response_schema: object | None,
    ) -> dict[str, object]:
        config: dict[str, object] = {
            "temperature": temperature,
            "max_output_tokens": output_max_length,
        }
        if response_schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = response_schema
        return config

    @staticmethod
    def _response_text(response: object) -> str:
        try:
            return str(response.text).strip()
        except Exception as exc:
            feedback = getattr(response, "prompt_feedback", "")
            raise Exception(str(exc) + str(feedback)) from exc


class OpenAIGateway:
    """Adapt the supported OpenAI chat-completions subset to generation."""

    def __init__(
        self,
        client: object,
        *,
        record_usage: Callable[[object], object] = lambda _response: None,
        reset_evidence: Callable[[], None] = lambda: None,
    ) -> None:
        self._client = client
        self._record_usage = record_usage
        self._reset_evidence = reset_evidence

    def generate(self, request: GenerationRequest) -> str:
        self._reject_unsupported_fields(request)
        self._reset_evidence()
        response = self._client.chat.completions.create(
            model=request.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        request.system_instruction
                        or "You are a helpful assistant."
                    ),
                },
                {"role": "user", "content": request.prompt},
            ],
            max_tokens=request.output_max_length,
            temperature=request.temperature,
        )
        self._record_usage(response)
        return response.choices[0].message.content.strip()

    @staticmethod
    def _reject_unsupported_fields(request: GenerationRequest) -> None:
        unsupported = tuple(
            name
            for name, value in (
                ("response_schema", request.response_schema),
                ("model_override", request.model_override),
                ("contents", request.contents),
            )
            if value is not None
        )
        if unsupported:
            raise ValueError(
                "OpenAI generation does not support request field(s): "
                + ", ".join(unsupported)
            )

__all__ = [
    "CallableBatchGenerationGateway",
    "CallableDialogueGateway",
    "CallableGenerationGateway",
    "GeminiGateway",
    "OpenAIGateway",
]
