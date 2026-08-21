"""Transactional ownership of provider conversation history."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterable


class Conversation:
    """Encapsulate one provider session and its mutable local history."""

    def __init__(self, provider_session: object) -> None:
        if isinstance(provider_session, Conversation):
            provider_session = provider_session.provider_session
        if not hasattr(provider_session, "history"):
            raise TypeError(
                "provider conversation must expose a history collection"
            )
        self._provider_session = provider_session

    @classmethod
    def wrap(cls, value: object) -> "Conversation":
        """Return ``value`` as a conversation without nested wrappers."""

        return value if isinstance(value, cls) else cls(value)

    @property
    def provider_session(self) -> object:
        """Return the opaque session consumed by the provider gateway."""

        return self._provider_session

    @property
    def history(self) -> list[Any]:
        """Return a snapshot of the current provider-managed history."""

        return list(self._provider_session.history)

    def begin(self) -> "ConversationTransaction":
        """Open a transaction that can restore the exact current history."""

        return ConversationTransaction(
            conversation=self,
            original_history=tuple(self._provider_session.history),
        )

    def append_history(self, messages: Iterable[object]) -> None:
        """Append owned copies of application-supplied history messages."""

        additions = deepcopy(list(messages))
        if additions:
            self._provider_session.history = self.history + additions

    def reset(self, history: Iterable[object] = ()) -> None:
        """Replace provider history at an explicit session boundary."""

        self._provider_session.history = list(history)


@dataclass(slots=True)
class ConversationTransaction:
    """One retry attempt over a conversation's local mutable history."""

    conversation: Conversation
    original_history: tuple[object, ...]
    _closed: bool = field(default=False, init=False, repr=False)

    def commit(self) -> None:
        """Keep the provider history produced by a successful turn."""

        self._closed = True

    def rollback(self) -> None:
        """Restore the exact pre-attempt history once."""

        if self._closed:
            return
        self.conversation.reset(self.original_history)
        self._closed = True


__all__ = ["Conversation", "ConversationTransaction"]
