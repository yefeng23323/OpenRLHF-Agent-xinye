"""Factory helpers for chat protocols."""

from typing import Dict, Optional, Type

from openrlhf_agent.chat_protocol.base import ChatProtocol
from openrlhf_agent.chat_protocol.qwen3_instruct import Qwen3InstructProtocol
from openrlhf_agent.chat_protocol.qwen3_thinking import Qwen3ThinkingProtocol


_DEFAULT_PROTOCOL = "qwen3_instruct"
_PROTOCOL_REGISTRY: Dict[str, Type[ChatProtocol]] = {
    "qwen3_instruct": Qwen3InstructProtocol,
    "qwen3_thinking": Qwen3ThinkingProtocol,
}


def make_chat_protocol(name: Optional[str] = None) -> ChatProtocol:
    """Return a chat protocol instance by name."""

    resolved_name = (name or _DEFAULT_PROTOCOL).lower()
    try:
        protocol_cls = _PROTOCOL_REGISTRY[resolved_name]
    except KeyError as exc:
        raise ValueError(f"Unknown chat protocol '{name}'.") from exc

    return protocol_cls()


__all__ = ["ChatProtocol", "Qwen3InstructProtocol", "Qwen3ThinkingProtocol", "make_chat_protocol"]
