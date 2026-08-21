import json
import argparse
import hashlib
import os
from contextlib import contextmanager
from pathlib import Path as _Path

# Load .env from repo root if present (never committed — see .gitignore).
_env_file = _Path(__file__).parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import logging
import tempfile
import threading
import time
import openai
import google.generativeai as genai
from google.api_core import exceptions as google_api_exceptions
from google.generativeai import protos
from google.generativeai.types import content_types

try:
    from attribute_first.infrastructure import (
        GeminiGateway,
        JsonArtifactStore,
        OpenAIGateway,
    )
    from attribute_first.domain import ModelProvider
    from attribute_first.ports import (
        ChatRequest,
        ChatTurnRequest,
        GenerationRequest,
    )
    from attribute_first.prompting import highlights as _highlight_helpers
    from attribute_first.prompting import templates as _template_helpers
    from attribute_first.runtime import (
        AttemptDependencies,
        AttemptExecutor,
        AttemptPolicy,
        IncompleteGenerationError,
        ProtocolEnvironment,
        UsageLedger,
        ensure_parseable_finish_reason as _ensure_parseable_finish_reason,
    )
    from attribute_first.runtime import system_resources as _system_resources
except ModuleNotFoundError as exc:
    if exc.name != "attribute_first":
        raise
    from few_shot_experiments.attribute_first.infrastructure import (
        GeminiGateway,
        JsonArtifactStore,
        OpenAIGateway,
    )
    from few_shot_experiments.attribute_first.domain import ModelProvider
    from few_shot_experiments.attribute_first.ports import (
        ChatRequest,
        ChatTurnRequest,
        GenerationRequest,
    )
    from few_shot_experiments.attribute_first.prompting import (
        highlights as _highlight_helpers,
    )
    from few_shot_experiments.attribute_first.prompting import (
        templates as _template_helpers,
    )
    from few_shot_experiments.attribute_first.runtime import (
        AttemptDependencies,
        AttemptExecutor,
        AttemptPolicy,
        IncompleteGenerationError,
        ProtocolEnvironment,
        UsageLedger,
        ensure_parseable_finish_reason as _ensure_parseable_finish_reason,
    )
    from few_shot_experiments.attribute_first.runtime import (
        system_resources as _system_resources,
    )


SPAN_SEP = _highlight_helpers.SPAN_SEP
SENT_SEP = _highlight_helpers.SENT_SEP
highlight_sep_strip = _highlight_helpers.highlight_sep_strip
find_substring_indices = _highlight_helpers.find_substring_indices
longest_common_subsequence = _highlight_helpers.longest_common_subsequence
rmv_txt_after_last_highlight = _highlight_helpers.rmv_txt_after_last_highlight
rmv_spaces_and_punct = _highlight_helpers.rmv_spaces_and_punct
remove_spaces_and_punctuation = (
    _highlight_helpers.remove_spaces_and_punctuation
)
find_substring = _highlight_helpers.find_substring
get_consecutive_subspans = _highlight_helpers.get_consecutive_subspans
merge_spans = _highlight_helpers.merge_spans
add_highlights = _highlight_helpers.add_highlights
extract_highlights = _highlight_helpers.extract_highlights
get_highlighted_doc = _highlight_helpers.get_highlighted_doc
_merge_doc_spans = _highlight_helpers._merge_doc_spans
add_highlights_typed = _highlight_helpers.add_highlights_typed
get_highlighted_doc_two_sets = (
    _highlight_helpers.get_highlighted_doc_two_sets
)

one_doc_fusion_prompt = _template_helpers.one_doc_fusion_prompt
make_highlights_fusion_prompt = (
    _template_helpers.make_highlights_fusion_prompt
)
make_clustering_prompt = _template_helpers.make_clustering_prompt
make_content_selection_prompt = (
    _template_helpers.make_content_selection_prompt
)
make_highlights_listing_prompt = (
    _template_helpers.make_highlights_listing_prompt
)
make_doc_prompt = _template_helpers.make_doc_prompt
make_ALCE_prompt = _template_helpers.make_ALCE_prompt
make_demo = _template_helpers.make_demo

get_max_memory = _system_resources.get_max_memory
# from IPython.display import display
# from IPython.display import Markdown

openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
genai_model = None
_genai_model_name = None

# --- Lightweight token-usage accounting (IMPROVEMENTS 4.4) ---
# Accumulates Gemini usage_metadata across calls so we can measure the token cost of a stage
# and the savings from context caching / dialogue. Reset per prompt_model run.
_TOKEN_USAGE = {
    "prompt": 0,
    "completion": 0,
    "cached": 0,
    "calls": 0,
    "provider_total": 0,
    "provider_total_calls": 0,
}
_TOKEN_USAGE_LOCK = threading.Lock()
_LAST_CALL_USAGE = threading.local()
_LAST_CALL_METADATA = threading.local()

AF_ENV_FLAGS = (
    "AF_CONTEXT_CACHE",
    "AF_DIALOGUE_NO_DEMOS",
    "AF_DOCS_FIRST",
    "AF_MARK_CONTEXT",
    "AF_USE_ROLES",
)
_TRUE_ENV_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})
_PROTOCOL_ENVIRONMENT = ProtocolEnvironment(
    allowed_flags=AF_ENV_FLAGS,
    environment=os.environ,
)
_USAGE_LEDGER = UsageLedger(
    totals_provider=lambda: _TOKEN_USAGE,
    lock=_TOKEN_USAGE_LOCK,
    usage_local=_LAST_CALL_USAGE,
    metadata_local=_LAST_CALL_METADATA,
)


def env_flag(name, default=False):
    """Return a reproducible boolean for an environment feature flag.

    Shell strings such as ``"0"`` and ``"false"`` must not enable an
    experiment merely because they are non-empty.
    """
    return _PROTOCOL_ENVIRONMENT.flag(name, default)


def get_af_environment_flags():
    """Snapshot every prompt/runtime AF flag for run provenance."""
    return _PROTOCOL_ENVIRONMENT.snapshot()


def validate_protocol_environment_flags(protocol):
    """Validate and return a config's declared runtime feature flags."""
    return _PROTOCOL_ENVIRONMENT.declared_flags(protocol)


@contextmanager
def protocol_environment(protocol):
    """Apply declared flags for one stage and restore the exact prior environment."""
    with _PROTOCOL_ENVIRONMENT.apply(protocol) as effective:
        yield effective


@contextmanager
def config_protocol_environment(config_file):
    """Apply the protocol flags declared by one stage config file."""
    if not config_file:
        with protocol_environment(None) as effective:
            yield effective
        return
    with open(config_file, "r", encoding="utf-8") as source:
        config = json.load(source)
    with protocol_environment(config.get("protocol")) as effective:
        yield effective


def reset_token_usage():
    _USAGE_LEDGER.reset()


def get_token_usage():
    return _USAGE_LEDGER.snapshot()


def reset_last_call_usage():
    """Clear provider evidence for the current worker thread before one call."""
    _USAGE_LEDGER.clear_last()


def get_last_call_usage():
    """Return the exact provider usage metadata from the current thread's call."""
    return _USAGE_LEDGER.last_usage()


def get_last_call_metadata():
    """Return the exact backend and termination metadata for the last response."""
    return _USAGE_LEDGER.last_metadata()


def _enum_name(value, enum_type):
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    try:
        return enum_type(value).name
    except (TypeError, ValueError):
        return str(value)


def _record_usage(response):
    reset_last_call_usage()
    candidates = getattr(response, "candidates", None) or []
    candidate = candidates[0] if candidates else None
    prompt_feedback = getattr(response, "prompt_feedback", None)
    metadata = {
        "provider_response_received": True,
        "model_version": (
            str(getattr(response, "model_version", "") or "") or None
        ),
        "finish_reason": _enum_name(
            getattr(candidate, "finish_reason", None),
            protos.Candidate.FinishReason,
        ),
        "prompt_block_reason": _enum_name(
            getattr(prompt_feedback, "block_reason", None),
            protos.GenerateContentResponse.PromptFeedback.BlockReason,
        ),
    }
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        _USAGE_LEDGER.record_metadata_only(metadata)
        raise RuntimeError(
            "provider response is missing mandatory usage metadata"
        )
    prompt = int(getattr(usage, "prompt_token_count", 0) or 0)
    completion = int(
        getattr(usage, "candidates_token_count", 0) or 0
    )
    cached = int(
        getattr(usage, "cached_content_token_count", 0) or 0
    )
    provider_total = getattr(usage, "total_token_count", None)
    return _USAGE_LEDGER.record(
        prompt=prompt,
        completion=completion,
        cached=cached,
        metadata=metadata,
        provider_total=(
            int(provider_total or 0)
            if provider_total is not None
            else None
        ),
    )


def _record_openai_usage(response):
    """Normalize OpenAI completion evidence into the shared usage ledger."""

    reset_last_call_usage()
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    raw_finish_reason = getattr(choice, "finish_reason", None)
    finish_reason = (
        {
            "stop": "STOP",
            "length": "MAX_TOKENS",
        }.get(
            str(raw_finish_reason).casefold(),
            str(raw_finish_reason).upper(),
        )
        if raw_finish_reason is not None
        else None
    )
    metadata = {
        "provider_response_received": True,
        "model_version": (
            str(getattr(response, "model", "") or "") or None
        ),
        "finish_reason": finish_reason,
        "prompt_block_reason": None,
    }
    usage = getattr(response, "usage", None)
    if usage is None:
        _USAGE_LEDGER.record_metadata_only(metadata)
        raise RuntimeError(
            "provider response is missing mandatory usage metadata"
        )
    prompt_details = getattr(
        usage,
        "prompt_tokens_details",
        None,
    )
    cached = int(
        getattr(prompt_details, "cached_tokens", 0) or 0
    )
    provider_total = getattr(usage, "total_tokens", None)
    return _USAGE_LEDGER.record(
        prompt=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion=int(
            getattr(usage, "completion_tokens", 0) or 0
        ),
        cached=cached,
        metadata=metadata,
        provider_total=(
            int(provider_total)
            if provider_total is not None
            else None
        ),
    )


def ensure_parseable_finish_reason(metadata=None):
    """Reject partial/blocked provider terminations before any parser runs."""
    if metadata is None:
        metadata = get_last_call_metadata()
    _ensure_parseable_finish_reason(metadata)


def summarize_response_metadata(attempts):
    """Aggregate auditable backend/termination evidence from attempt records."""
    summary = {
        "schema_version": 1,
        "provider_responses": 0,
        "model_versions": [],
        "finish_reason_counts": {},
        "prompt_block_reason_counts": {},
        "responses_missing_model_version": 0,
        "responses_missing_usage": 0,
        "max_tokens_responses": 0,
        "parsed_max_tokens_responses": 0,
    }
    model_versions = set()
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        metadata = attempt.get("response_metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("provider_response_received") is not True
        ):
            continue
        summary["provider_responses"] += 1
        model_version = metadata.get("model_version")
        if isinstance(model_version, str) and model_version:
            model_versions.add(model_version)
        else:
            summary["responses_missing_model_version"] += 1
        if not isinstance(attempt.get("usage"), dict):
            summary["responses_missing_usage"] += 1
        finish_reason = metadata.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            counts = summary["finish_reason_counts"]
            counts[finish_reason] = counts.get(finish_reason, 0) + 1
        block_reason = metadata.get("prompt_block_reason")
        if isinstance(block_reason, str) and block_reason:
            counts = summary["prompt_block_reason_counts"]
            counts[block_reason] = counts.get(block_reason, 0) + 1
        if finish_reason == "MAX_TOKENS":
            summary["max_tokens_responses"] += 1
            if attempt.get("status") == "parsed":
                summary["parsed_max_tokens_responses"] += 1
    summary["model_versions"] = sorted(model_versions)
    summary["finish_reason_counts"] = dict(
        sorted(summary["finish_reason_counts"].items())
    )
    summary["prompt_block_reason_counts"] = dict(
        sorted(summary["prompt_block_reason_counts"].items())
    )
    return summary

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

SUBTASK_WITHOUT_GIVEN_HIGHLIGHTS = ["content_selection", "e2e_only_setting", "ALCE", "iterative_blue_print"]

class TokenCounter:
    _MAX_ATTEMPTS = 3
    _TRANSIENT_ERRORS = (
        google_api_exceptions.DeadlineExceeded,
        google_api_exceptions.ServiceUnavailable,
        google_api_exceptions.InternalServerError,
        google_api_exceptions.TooManyRequests,
    )

    def __init__(self, model_name: str):
        """
        model_name: name of model that is being prompted
        """
        # normalise "gemini-pro" -> "models/gemini-pro-latest"
        self.model_name = _normalize_model_name(model_name)

        # accept all Gemini model ids: models/gemini-...
        if self.model_name.startswith("models/gemini"):
            self.model = genai.GenerativeModel(self.model_name)
        else:
            raise NotImplementedError(
                f"Token counter for {model_name} not supported yet."
            )

    def token_count(self, prompt):
        if self.model_name.startswith("models/gemini"):
            for attempt in range(1, self._MAX_ATTEMPTS + 1):
                try:
                    return self.model.count_tokens(prompt).total_tokens
                except self._TRANSIENT_ERRORS:
                    if attempt == self._MAX_ATTEMPTS:
                        raise
                    delay_seconds = float(2 ** (attempt - 1))
                    logging.warning(
                        "Transient Gemini count_tokens failure; "
                        "retrying attempt %d/%d in %.1fs",
                        attempt + 1,
                        self._MAX_ATTEMPTS,
                        delay_seconds,
                    )
                    time.sleep(delay_seconds)
        raise NotImplementedError(f"token_count for {self.model_name} not supported yet.")

def update_args(args):
    """update args with arguments from config file"""
    with open(args.config_file, 'r') as f1:
        updated_args = json.loads(f1.read())
    validate_protocol_environment_flags(updated_args.get("protocol"))
    additional_args = {key:value for key,value in args.__dict__.items() if not key in updated_args.keys()}
    if set(additional_args).intersection(updated_args):
        raise ValueError("config and command-line arguments contain overlapping keys")
    updated_args.update(additional_args)
    return argparse.Namespace(**updated_args)


def _normalize_model_name(model_name: str) -> str:
    # Backward compat: the original repo used "gemini-pro" (Gemini 1.0).
    # The current google-generativeai SDK expects fully-qualified ids like "models/gemini-pro-latest".
    if model_name == "gemini-pro":
        return "models/gemini-pro-latest"
    return model_name


def get_token_counter(model_name, prompt_token_budget=None):
    model_name = _normalize_model_name(model_name)
    if (
        type(prompt_token_budget) is not int
        or prompt_token_budget < 1
    ):
        raise ValueError(
            "prompt_token_budget must be an explicit positive integer"
        )

    # Gemini family (google-generativeai)
    if model_name.startswith("models/gemini"):
        return {
            "tkn_counter": TokenCounter(model_name=model_name),
            "tkn_max_limit": prompt_token_budget,
        }

    raise Exception(
        f"not supported yet for {model_name}. Please add a token counter and max limit for {model_name}"
    )


def _artifact_store_destination(path):
    destination = _Path(path).expanduser()
    if not destination.is_absolute():
        destination = (_Path.cwd() / destination).resolve()
    else:
        destination = destination.resolve()
    return JsonArtifactStore(destination.parent), destination.name


def atomic_write_text(path, content):
    """Replace a text artifact atomically within its destination directory."""
    store, relative_path = _artifact_store_destination(path)
    store.write_text(relative_path, content)


def atomic_write_json(path, value, *, indent=2):
    store, relative_path = _artifact_store_destination(path)
    store.write_json(relative_path, value, indent=indent)


def atomic_write_jsonl(path, values):
    store, relative_path = _artifact_store_destination(path)
    store.write_jsonl(relative_path, values)


def artifact_sha256(path):
    """Return the SHA-256 of an artifact's exact bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_value_sha256(value):
    """Fingerprint text directly and structured values as canonical JSON."""
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def remove_pipeline_artifact(outdir):
    """Remove a pipeline artifact that cannot be attributed to the current run."""
    pipeline_path = _Path(outdir) / "pipeline_format_results.json"
    if pipeline_path.exists():
        pipeline_path.unlink()


def save_results(outdir, used_demos, final_results, pipeline_format_results=None):
    atomic_write_json(os.path.join(outdir, "used_demonstrations.json"), used_demos)
    atomic_write_json(os.path.join(outdir, "results.json"), final_results)

    # A missing conversion result belongs to this run. Remove any older pipeline
    # artifact so downstream stages cannot mistake stale data for current output.
    pipeline_path = _Path(outdir) / "pipeline_format_results.json"
    if pipeline_format_results is not None:
        atomic_write_jsonl(pipeline_path, pipeline_format_results)
    else:
        remove_pipeline_artifact(outdir)

    df_style_data = [{'instance': key, **value} for key, value in final_results.items()]
    df_style_data = [{key:json.dumps(value) if type(value) in [list,dict] else value for key,value in instance.items()} for instance in df_style_data]
    final_results_dataframe = pd.DataFrame(df_style_data)
    csv_path = _Path(outdir) / "results.csv"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=csv_path.parent,
        prefix=f".{csv_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as csv_file:
        csv_temp_path = _Path(csv_file.name)
        final_results_dataframe.to_csv(csv_file, index=False)
    try:
        os.replace(csv_temp_path, csv_path)
    finally:
        if csv_temp_path.exists():
            csv_temp_path.unlink()

_DIALOGUE_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


def _gemini_gateway():
    """Build an SDK adapter from the facade's patchable dependencies."""
    return GeminiGateway(
        sdk=genai,
        content_types=content_types,
        normalize_model_name=_normalize_model_name,
        record_usage=_record_usage,
        ensure_parseable=ensure_parseable_finish_reason,
        reset_evidence=reset_last_call_usage,
        last_metadata=get_last_call_metadata,
        safety_settings=_DIALOGUE_SAFETY_SETTINGS,
        flat_model=genai_model,
        flat_model_name=_genai_model_name,
        role_models=_role_model_cache,
    )


def create_chat_session(
    model_name: str,
    cached_content=None,
    system_instruction=None,
    history=None,
):
    """Create a new Gemini ChatSession for dialogue-mode pipelines.

    If cached_content is given, the session is bound to a CachedContent (shared demo prefix),
    so the prefix is uploaded once and reused across all instances' chats instead of being
    re-sent in every Turn 1. A non-cached role dialogue can instead provide one
    system instruction and an initial user/model demonstration history."""
    return _gemini_gateway().create_chat(
        ChatRequest(
            model_name=model_name,
            cached_content=cached_content,
            system_instruction=system_instruction,
            history=tuple(history or ()),
        )
    )

def gemini_chat_call(chat_session, message: str, output_max_length: int = 4096, temperature: float = 0,
                     response_schema: dict = None):
    """Send one turn while retaining response evidence before SDK validation.

    This mirrors ``ChatSession.send_message`` for the pinned 0.8.6 SDK, but
    records usage/backend/finish metadata before ``_check_response`` can raise.
    Only a complete response is committed to the chat history.
    """
    return _gemini_gateway().send_message(
        chat_session,
        ChatTurnRequest(
            message=message,
            output_max_length=output_max_length,
            temperature=temperature,
            response_schema=response_schema,
        ),
    )

_role_model_cache = {}


def gemini_call(prompt, model_name, output_max_length: int = 2048, temperature: int = 0,
                response_schema: dict = None, model_override=None,
                contents=None, system_instruction=None):
    # Roles path (AF_USE_ROLES): send a system_instruction + a list of user/model turns
    # (few-shot demos as real conversation turns) instead of one flat concatenated string.
    # This matches how the models are trained. `prompt` (the flat string) is still returned to
    # the parser upstream so sentinel/doc extraction keeps working unchanged.
    global genai_model, _genai_model_name
    gateway = _gemini_gateway()
    try:
        return gateway.generate(
            GenerationRequest(
                model_name=model_name,
                prompt=prompt,
                output_max_length=output_max_length,
                temperature=temperature,
                response_schema=response_schema,
                model_override=model_override,
                contents=contents,
                system_instruction=system_instruction,
            )
        )
    finally:
        genai_model = gateway.flat_model
        _genai_model_name = gateway.flat_model_name

def openai_call(prompt, model_name, output_max_length: int = 2048, temperature: int = 0):
    return OpenAIGateway(
        openai,
        record_usage=_record_openai_usage,
        reset_evidence=reset_last_call_usage,
    ).generate(
        GenerationRequest(
            model_name=model_name,
            prompt=prompt,
            output_max_length=output_max_length,
            temperature=temperature,
        )
    )

def model_call_wrapper(prompt: str, model_name: str, parse_response_fn, output_max_length : int = 4096, num_retries: int = 5, temperature: int = 0, response_schema: dict = None, send_prompt: str = None, model_override=None, contents=None, system_instruction=None):
    # send_prompt: what is actually sent to the model (e.g. the per-instance tail when the
    # shared demo prefix lives in a CachedContent). parsing/storage still use the full `prompt`
    # so parsers that inspect the prompt (sentinels, doc extraction) keep working.
    provider = ModelProvider.from_model_id(model_name)
    if provider is ModelProvider.OPENAI:
        unsupported = tuple(
            name
            for name, value in (
                ("response_schema", response_schema),
                ("model_override", model_override),
                ("contents", contents),
            )
            if value is not None
        )
        if unsupported:
            raise ValueError(
                "OpenAI compatibility path does not support request "
                "field(s): "
                + ", ".join(unsupported)
            )
    call_func = (
        openai_call
        if provider is ModelProvider.OPENAI
        else gemini_call
    )
    prompt_to_send = send_prompt if send_prompt is not None else prompt
    if contents is not None:
        application_request = {
            "transport": "roles",
            "system_instruction": system_instruction,
            "contents": contents,
            "cache_bound": model_override is not None,
        }
    elif send_prompt is not None:
        application_request = {
            "transport": "flat",
            "sent_prompt": prompt_to_send,
            "cache_bound": model_override is not None,
        }
    else:
        application_request = {
            "transport": "flat",
            "prompt_source": "result.prompt",
            "cache_bound": model_override is not None,
        }

    def invoke():
        call_kwargs = {
            "prompt": prompt_to_send,
            "model_name": model_name,
            "output_max_length": output_max_length,
            "temperature": temperature,
        }
        if call_func is gemini_call:
            if response_schema is not None:
                call_kwargs["response_schema"] = response_schema
            if model_override is not None:
                call_kwargs["model_override"] = model_override
            if contents is not None:
                call_kwargs["contents"] = contents
                call_kwargs["system_instruction"] = system_instruction
        return call_func(**call_kwargs)

    executor = AttemptExecutor(
        AttemptDependencies(
            invoke=invoke,
            parse=parse_response_fn,
            reset_evidence=reset_last_call_usage,
            last_usage=get_last_call_usage,
            last_metadata=get_last_call_metadata,
            ensure_parseable=ensure_parseable_finish_reason,
            fingerprint=stable_value_sha256,
            sleep=time.sleep,
            incomplete_error=IncompleteGenerationError,
        )
    )
    return executor.execute(
        prompt=prompt,
        policy=AttemptPolicy(
            model_name=model_name,
            output_max_length=output_max_length,
            num_retries=num_retries,
            temperature=temperature,
        ),
        application_request=application_request,
        response_schema=response_schema,
    )

_CONTEXT_CACHE_TARGET = "### TARGET DOCUMENTS (ANSWER ONLY THESE) ###\n"


def _build_context_cache(prompts, model_name):
    """If AF_CONTEXT_CACHE is set, cache the shared demo prefix (identical across all instance
    prompts of a stage) in a Gemini CachedContent so it is uploaded once and reused, instead
    of being re-sent with every instance. Returns (cache, cached_model, prefix) or (None,..).

    Splits each prompt at the TARGET header: everything before it is the shared demo prefix
    (cacheable); everything from it on is the per-instance tail (docs + instruction).
    """
    if (
        not env_flag("AF_CONTEXT_CACHE")
        or ModelProvider.from_model_id(model_name)
        is ModelProvider.OPENAI
        or len(prompts) < 2
    ):
        return None, None, None
    vals = list(prompts.values())
    if not all(_CONTEXT_CACHE_TARGET in p for p in vals):
        logging.warning("[context-cache] disabled: not all prompts contain the TARGET header")
        return None, None, None
    prefix = vals[0].split(_CONTEXT_CACHE_TARGET)[0]
    if not all(p.split(_CONTEXT_CACHE_TARGET)[0] == prefix for p in vals) or not prefix.strip():
        logging.warning("[context-cache] disabled: shared prefix not identical across instances")
        return None, None, None
    try:
        from google.generativeai import caching
        import datetime
        cache = caching.CachedContent.create(
            model=_normalize_model_name(model_name),
            contents=[prefix],
            ttl=datetime.timedelta(minutes=20),
        )
        cached_model = genai.GenerativeModel.from_cached_content(cached_content=cache)
        toks = getattr(getattr(cache, "usage_metadata", None), "total_token_count", None)
        logging.info(f"[context-cache] cached shared prefix ({toks} tokens) once; "
                     f"sending only per-instance tails for {len(prompts)} instances")
        return cache, cached_model, prefix
    except Exception as e:
        logging.warning(f"[context-cache] disabled (create failed): {e}")
        return None, None, None


def _build_role_context_cache(role_messages, model_name):
    """Cache the shared system instruction and role-typed demonstrations.

    The final user turn is instance-specific and remains outside the cache.
    All earlier turns must be identical across instances so caching cannot
    silently change the experimental prompt.
    """
    if (
        not env_flag("AF_CONTEXT_CACHE")
        or ModelProvider.from_model_id(model_name)
        is ModelProvider.OPENAI
        or len(role_messages) < 2
    ):
        return None, None, {}

    payloads = list(role_messages.values())
    if any(
        not isinstance(payload, dict)
        or not isinstance(payload.get("system"), str)
        or not isinstance(payload.get("contents"), list)
        or not payload["contents"]
        or not isinstance(payload["contents"][-1], dict)
        or payload["contents"][-1].get("role") != "user"
        for payload in payloads
    ):
        logging.warning(
            "[context-cache+roles] disabled: invalid role payload contract"
        )
        return None, None, {}

    shared_system = payloads[0]["system"]
    shared_turns = payloads[0]["contents"][:-1]
    if not shared_turns:
        logging.info(
            "[context-cache+roles] disabled: zero-shot payload has no shared "
            "demonstration turns"
        )
        return None, None, {}
    if any(
        payload["system"] != shared_system
        or payload["contents"][:-1] != shared_turns
        for payload in payloads[1:]
    ):
        logging.warning(
            "[context-cache+roles] disabled: role prefixes differ across "
            "instances"
        )
        return None, None, {}

    try:
        from google.generativeai import caching
        import datetime

        cache = caching.CachedContent.create(
            model=_normalize_model_name(model_name),
            system_instruction=shared_system,
            contents=shared_turns,
            ttl=datetime.timedelta(
                minutes=max(60, 2 * len(role_messages))
            ),
        )
        cached_model = genai.GenerativeModel.from_cached_content(
            cached_content=cache
        )
        tails = {
            instance_id: payload["contents"][-1]
            for instance_id, payload in role_messages.items()
        }
        tokens = getattr(
            getattr(cache, "usage_metadata", None), "total_token_count", None
        )
        logging.info(
            "[context-cache+roles] cached system + demonstration turns "
            f"({tokens} tokens) once for {len(role_messages)} instances"
        )
        return cache, cached_model, tails
    except Exception as exc:
        logging.warning(
            f"[context-cache+roles] disabled (create failed): {exc}"
        )
        return None, None, {}


def prompt_model(prompts: Dict, model_name: str, parse_response_fn, output_max_length : int = 4096, num_retries: int = 5, verbose: bool = True, temperature: int = 0, response_schema: dict = None, concurrency: int = 1, role_messages: Dict = None, reset_usage: bool = True):
    prompts_tpls = [(inst_name, prompt) for inst_name,prompt in prompts.items()]
    results = dict()
    if reset_usage:
        reset_token_usage()

    requested_roles = env_flag("AF_USE_ROLES")
    if requested_roles and prompts and not role_messages:
        raise ValueError(
            "AF_USE_ROLES is enabled but prompt construction produced no "
            "role payloads"
        )
    use_roles = requested_roles or bool(role_messages)
    if use_roles:
        missing_role_ids = sorted(set(prompts) - set(role_messages))
        if missing_role_ids:
            raise ValueError(
                "role transport is missing payloads for: "
                + ", ".join(missing_role_ids)
            )
        cache, cached_model, role_tails = _build_role_context_cache(
            role_messages, model_name
        )
        prefix = None
        logging.info(
            f"[roles] sending {len(prompts_tpls)} instances as "
            "system+user/model turns"
        )
    else:
        role_tails = {}
        cache, cached_model, prefix = _build_context_cache(prompts, model_name)

    def _run_one(item):
        inst_name, prompt = item
        send_prompt = None
        if cached_model is not None and prefix is not None and prompt.startswith(prefix):
            # send only the tail; the cached prefix is served from the CachedContent
            send_prompt = prompt[len(prefix):]
        rm = (role_messages or {}).get(inst_name)
        cached_role_tail = role_tails.get(inst_name)
        if cached_role_tail is not None:
            contents = [cached_role_tail]
            system_instruction = None
        else:
            contents = rm["contents"] if rm else None
            system_instruction = rm["system"] if rm else None
        res = model_call_wrapper(prompt=prompt,
                                 model_name=model_name,
                                 parse_response_fn=parse_response_fn,
                                 output_max_length=output_max_length,
                                 num_retries=num_retries,
                                 temperature=temperature,
                                 response_schema=response_schema,
                                 send_prompt=send_prompt,
                                 model_override=cached_model,
                                 contents=contents,
                                 system_instruction=system_instruction)
        return inst_name, res

    try:
        if concurrency and concurrency > 1 and len(prompts_tpls) > 1:
            # Concurrent path (IMPROVEMENTS 1.2): the model calls are I/O-bound, so a bounded
            # thread pool cuts wall-clock ~Nx without changing results. Bounded to respect rate
            # limits. gemini_call's generate_content is stateless per call → thread-safe.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            logging.info(f"[concurrency] running {len(prompts_tpls)} instances with {concurrency} workers")
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = [ex.submit(_run_one, it) for it in prompts_tpls]
                completed = as_completed(futs)
                completed = tqdm(completed, total=len(futs)) if verbose else completed
                for f in completed:
                    inst_name, res = f.result()
                    results[inst_name] = res
        else:
            iterator = tqdm(prompts_tpls) if verbose else prompts_tpls
            for item in iterator:
                inst_name, res = _run_one(item)
                results[inst_name] = res
    finally:
        if cache is not None:
            try:
                cache.delete()
                logging.info("[context-cache] deleted cache")
            except Exception as e:
                logging.warning(f"[context-cache] cache delete failed: {e}")
    u = get_token_usage()
    logging.info(f"[tokens] calls={u['calls']} prompt={u['prompt']} completion={u['completion']} cached={u['cached']}")
    return results
