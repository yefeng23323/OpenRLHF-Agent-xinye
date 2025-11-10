"""Conversation buffer that keeps chat state tidy."""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping

from openrlhf_agent.core import Message


class Conversation:
    """Stores chat messages and knows how to render them."""

    def __init__(self) -> None:
        self._messages: List[Message] = []

    def reset(self, *, system_prompt: str) -> None:
        """Start a fresh transcript using the provided system prompt."""

        self._messages = [Message(role="system", content=system_prompt)]

    def extend(self, messages: Iterable[Message | Mapping[str, Any]]) -> None:
        """Append a list of historical messages."""

        for message in messages:
            if isinstance(message, Message):
                self._messages.append(message)
            elif isinstance(message, Mapping):
                self._messages.append(Message(**message))

    def append(self, message: Message) -> None:
        """Append a message and return it for convenience."""

        self._messages.append(message)

    @property
    def messages(self) -> List[dict]:
        """Expose a shallow copy for inspection or debugging."""

        return [message.model_dump(exclude_none=True) for message in self._messages]


__all__ = ["Conversation"]
