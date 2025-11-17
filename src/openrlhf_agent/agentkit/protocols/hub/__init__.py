"""Built-in chat protocol implementations."""

from .qwen3_instruct import Qwen3InstructProtocol
from .qwen3_thinking import Qwen3ThinkingProtocol

__all__ = ["Qwen3InstructProtocol", "Qwen3ThinkingProtocol"]
