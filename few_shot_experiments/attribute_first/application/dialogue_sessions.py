"""Dialogue-session creation and cache-fallback policy."""

from copy import deepcopy
import logging

from ..ports import ChatRequest
from ..runtime.conversation import Conversation


class DialogueSessionService:
    """Own provider chat sessions without executing semantic stages."""

    def __init__(self, dependencies):
        self._dependencies = dependencies

    def create_chat(
        self,
        model_name,
        *,
        cached_content=None,
        system_instruction=None,
        history=None,
    ):
        """Create one provider session through the dialogue gateway."""

        return self._dependencies.dialogue_gateway.create_chat(
            ChatRequest(
                model_name=model_name,
                cached_content=cached_content,
                system_instruction=system_instruction,
                history=tuple(history or ()),
            )
        )

    def start(self, state, instance):
        """Start one instance and return only its first live task."""

        plan = state.plan
        cache_state = state.cache_state
        role_payload = instance.role_payload
        if role_payload is not None:
            role_target = role_payload["contents"][-1]
            turn = role_target["parts"]
            instance.content_selection_demo_history = deepcopy(
                role_payload["contents"][:-1]
            )
            if (
                cache_state.cache is not None
                and instance.uid in cache_state.role_tails
            ):
                instance.session = self.create_chat(
                    plan.model_name,
                    cached_content=cache_state.cache,
                )
                instance.cache_bound = True
            else:
                instance.session = self.create_chat(
                    plan.model_name,
                    system_instruction=role_payload["system"],
                    history=role_payload["contents"][:-1],
                )
        else:
            instance.session = self.create_chat(
                plan.model_name,
                cached_content=cache_state.cache,
            )
            instance.cache_bound = cache_state.cache is not None
            prefix = cache_state.prefix
            turn = (
                instance.content_selection_prompt[len(prefix):]
                if (
                    cache_state.cache is not None
                    and prefix
                    and instance.content_selection_prompt.startswith(prefix)
                )
                else instance.content_selection_prompt
            )
        instance.protocol["cs_live_message_sha256"] = (
            self._dependencies.stable_value_sha256(turn)
        )
        return turn

    def remove_completed_content_selection_demos(
        self,
        state,
        instance,
    ):
        """Keep the live CS exchange but drop its now-obsolete examples."""

        demos = instance.content_selection_demo_history
        if instance.role_payload is None:
            if instance.cache_bound:
                live_history = (
                    self._dependencies.jsonable_dialogue_value(
                        Conversation.wrap(instance.session).history
                    )
                )
                if len(live_history) < 2:
                    raise ValueError(
                        "cached content-selection session has no live "
                        "exchange to checkpoint"
                    )
                live_history = live_history[-2:]
                live_history[0] = {
                    "role": "user",
                    "parts": [instance.content_selection_prompt],
                }
                instance.session = self.create_chat(
                    state.plan.model_name,
                    history=live_history,
                )
                instance.cache_bound = False
            return
        conversation = Conversation.wrap(instance.session)
        live_history = conversation.history
        if instance.cache_bound:
            instance.session = self.create_chat(
                state.plan.model_name,
                system_instruction=instance.role_payload["system"],
                history=live_history,
            )
            instance.cache_bound = False
        elif demos:
            count = len(demos)
            observed = self._dependencies.jsonable_dialogue_value(
                live_history[:count]
            )
            expected = self._dependencies.jsonable_dialogue_value(demos)
            if (
                self._dependencies.stable_value_sha256(observed)
                != self._dependencies.stable_value_sha256(expected)
            ):
                raise ValueError(
                    "content-selection demonstration history drifted "
                    "before pruning"
                )
            conversation.reset(live_history[count:])
        instance.protocol.update(
            {
                "cs_previous_task_demos_removed": True,
                "cs_removed_demo_history_sha256": (
                    self._dependencies.stable_value_sha256(demos)
                ),
            }
        )

    def retry_content_selection_without_cache(
        self,
        state,
        instance,
        remaining_retries,
    ):
        """Reset a failed cached session and retry the complete CS prompt."""

        plan = state.plan
        stage = plan.content_selection
        logging.warning(
            "[dialogue] cached CS transport failed for %s — "
            "retrying without cache",
            instance.uid,
        )
        if instance.role_payload is not None:
            instance.session = self.create_chat(
                plan.model_name,
                system_instruction=instance.role_payload["system"],
                history=instance.role_payload["contents"][:-1],
            )
            retry_message = instance.role_payload[
                "contents"
            ][-1]["parts"]
        else:
            instance.session = self.create_chat(plan.model_name)
            retry_message = instance.content_selection_prompt
        instance.cache_bound = False
        fallback_history = self._dependencies.jsonable_dialogue_value(
            Conversation.wrap(instance.session).history
        )
        instance.protocol["cs_cache_fallback_message_sha256"] = (
            self._dependencies.stable_value_sha256(retry_message)
        )
        return self._dependencies.dialogue_turn(
            instance.session,
            retry_message,
            stage.parse_fn,
            instance.content_selection_prompt,
            remaining_retries,
            plan.temperature,
            response_schema=stage.schema,
            output_max_length=getattr(
                stage.args,
                "output_max_length",
                4096,
            ),
            model_name=stage.args.model_name,
            attempt_trace=instance.trace["content_selection"],
            call_records=state.call_records,
            call_context={
                "unique_id": instance.uid,
                "stage": "content_selection",
                "cache_bound": False,
                "cache_fallback": True,
                "session_reset_reason": "cache_transport_fallback",
                "session_reset_before": fallback_history,
            },
        )


__all__ = ["DialogueSessionService"]
