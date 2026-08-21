"""Shared-prefix cache lifecycle for dialogue pipelines."""

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class DialogueCacheState:
    """Mutable lifecycle state for one optional provider cache."""

    requested: bool
    cache: Any = None
    prefix: str | None = None
    role_tails: dict = field(default_factory=dict)
    trace: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.trace:
            self.trace = {
                "requested": self.requested,
                "created": False,
                "identifier": None,
                "creation_error": None,
                "creation_token_count": None,
                "bound_calls": 0,
                "effective_calls": 0,
                "provider_cached_token_count": 0,
                "delete_attempted": False,
                "deleted": None,
                "delete_error": None,
            }


class DialogueCacheManager:
    """Create, bind, measure, and delete one shared dialogue cache."""

    def __init__(
        self,
        *,
        caching_module,
        normalize_model_name,
        context_cache_target,
    ):
        self._caching = caching_module
        self._normalize_model_name = normalize_model_name
        self._context_cache_target = context_cache_target

    def create(
        self,
        *,
        requested,
        cs_prompts,
        role_messages,
        roles_requested,
        model_name,
    ):
        state = DialogueCacheState(requested=requested)
        if not requested or len(cs_prompts) <= 1:
            return state
        if roles_requested:
            self._create_role_cache(
                state,
                role_messages=role_messages,
                model_name=model_name,
                instance_count=len(cs_prompts),
            )
        else:
            self._create_flat_cache(
                state,
                cs_prompts=cs_prompts,
                model_name=model_name,
            )
        return state

    def _create_role_cache(
        self,
        state,
        *,
        role_messages,
        model_name,
        instance_count,
    ):
        payloads = list(role_messages.values())
        shared_system = payloads[0]["system"]
        shared_history = payloads[0]["contents"][:-1]
        prefix_is_shared = (
            bool(shared_history)
            and all(
                payload["system"] == shared_system
                and payload["contents"][:-1] == shared_history
                and payload["contents"][-1].get("role") == "user"
                for payload in payloads
            )
        )
        if not prefix_is_shared:
            logging.warning(
                "[dialogue+cache+roles] disabled: role prefixes are not "
                "identical or contain no CS demonstrations"
            )
            return
        try:
            ttl_minutes = max(120, 3 * instance_count)
            state.cache = self._caching.CachedContent.create(
                model=self._normalize_model_name(model_name),
                system_instruction=shared_system,
                contents=shared_history,
                ttl=datetime.timedelta(minutes=ttl_minutes),
            )
            state.role_tails = {
                uid: payload["contents"][-1]
                for uid, payload in role_messages.items()
            }
            tokens = getattr(
                getattr(state.cache, "usage_metadata", None),
                "total_token_count",
                None,
            )
            self._record_created(state, tokens)
            logging.info(
                "[dialogue+cache+roles] cached dialogue system + "
                f"CS demonstrations ({tokens} tokens) once for "
                f"{instance_count} chat sessions"
            )
        except Exception as exc:
            state.trace["creation_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            logging.warning(
                f"[dialogue+cache+roles] disabled (create failed): {exc}"
            )

    def _create_flat_cache(
        self,
        state,
        *,
        cs_prompts,
        model_name,
    ):
        prompts = list(cs_prompts.values())
        if not all(
            self._context_cache_target in prompt
            for prompt in prompts
        ):
            return
        candidate = prompts[0].split(self._context_cache_target)[0]
        if (
            not candidate.strip()
            or not all(
                prompt.split(self._context_cache_target)[0] == candidate
                for prompt in prompts
            )
        ):
            return
        try:
            ttl_minutes = max(120, 3 * len(cs_prompts))
            state.cache = self._caching.CachedContent.create(
                model=self._normalize_model_name(model_name),
                contents=[candidate],
                ttl=datetime.timedelta(minutes=ttl_minutes),
            )
            state.prefix = candidate
            tokens = getattr(
                getattr(state.cache, "usage_metadata", None),
                "total_token_count",
                None,
            )
            self._record_created(state, tokens)
            logging.info(
                "[dialogue+cache] cached shared CS prefix "
                f"({tokens} tokens) once for {len(cs_prompts)} "
                "chat sessions"
            )
        except Exception as exc:
            state.trace["creation_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            logging.warning(
                f"[dialogue+cache] disabled (create failed): {exc}"
            )

    @staticmethod
    def _record_created(state, tokens):
        state.trace.update(
            {
                "created": True,
                "identifier": str(
                    getattr(state.cache, "name", None)
                    or getattr(state.cache, "id", None)
                    or "<provider-did-not-expose-id>"
                ),
                "creation_token_count": tokens,
            }
        )

    def finalize(self, state, call_records):
        if state.cache is not None:
            state.trace["delete_attempted"] = True
            try:
                state.cache.delete()
                state.trace["deleted"] = True
                logging.info("[dialogue+cache] deleted cache")
            except Exception as exc:
                state.trace["deleted"] = False
                state.trace["delete_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                logging.warning(
                    f"[dialogue+cache] cache delete failed: {exc}"
                )
        state.trace["bound_calls"] = sum(
            bool(record.get("cache_bound"))
            for record in call_records
        )
        state.trace["effective_calls"] = sum(
            bool(record.get("cache_bound"))
            and int(
                (record.get("usage") or {}).get(
                    "cached_content_token_count",
                    0,
                )
                or 0
            )
            > 0
            for record in call_records
        )
        state.trace["provider_cached_token_count"] = sum(
            int(
                (record.get("usage") or {}).get(
                    "cached_content_token_count",
                    0,
                )
                or 0
            )
            for record in call_records
        )
        return state.trace


__all__ = [
    "DialogueCacheManager",
    "DialogueCacheState",
]
