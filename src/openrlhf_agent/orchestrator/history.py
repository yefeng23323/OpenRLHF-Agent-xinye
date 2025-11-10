"""Conversation buffer that keeps chat state tidy."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from openrlhf_agent.chat_protocol import ChatProtocol
from openrlhf_agent.core import Message


class Conversation:
    """Stores chat messages and knows how to render them."""

    def __init__(self) -> None:
        self._messages: List[Message] = []

    # ------------------------------------------------------------------ lifecycle

    @staticmethod
    def _coerce(message: Message | Mapping[str, Any]) -> Message:
        if isinstance(message, Message):
            return message
        if isinstance(message, Mapping):
            return Message(**message)
        raise TypeError(f"Unsupported message type: {type(message)!r}")

    def reset(self, *, system_prompt: str) -> None:
        """Start a fresh transcript using the provided system prompt."""

        self._messages = [Message(role="system", content=system_prompt)]

    def extend(self, messages: Iterable[Message | Mapping[str, Any]]) -> None:
        """Append a list of historical messages."""

        for message in messages:
            self._messages.append(self._coerce(message))

    # -------------------------------------------------------------------- mutators

    def append(self, message: Message) -> Message:
        """Append a message and return it for convenience."""

        self._messages.append(message)
        return message

    def add_tool(self, content: str) -> Message:
        """Record a tool response as a `tool` role message."""

        return self.append(Message(role="tool", content=content))

    # --------------------------------------------------------------------- exports

    def payload(self) -> List[Dict[str, Any]]:
        """Return provider-ready dictionaries of the transcript."""

        return [message.model_dump(exclude_none=True) for message in self._messages]

    def render_prompt(
        self,
        protocol: ChatProtocol,
        *,
        tools_manifest: Optional[Sequence[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """Render the transcript into whatever prompt the protocol expects."""

        return protocol.render_messages(
            messages=self.payload(),
            tools_manifest=tools_manifest,
            add_generation_prompt=add_generation_prompt,
        )

    @property
    def messages(self) -> List[Message]:
        """Expose a shallow copy for inspection or debugging."""

        return list(self._messages)


# Backwards compatible aliases
ConversationBuffer = Conversation
ChatHistory = Conversation


__all__ = ["Conversation", "ConversationBuffer", "ChatHistory"]
