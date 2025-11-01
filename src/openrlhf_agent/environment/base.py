from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from openrlhf_agent.utils.types import ToolResult, ToolCall


class Environment(ABC):
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return a system prompt string for the agent."""
        raise NotImplementedError

    @abstractmethod
    def tools_manifest(self) -> List[Dict[str, Any]]:
        """Return a list of tool specifications (OpenAI compatible)."""
        raise NotImplementedError

    @abstractmethod
    def execute_tool(self, name: str, args: Dict[str, Any], context: Dict[str, Any] = {}) -> ToolResult:
        """Execute the tool by name with arguments and optional context."""
        raise NotImplementedError

    @abstractmethod
    def reward_hook(self, event: Dict[str, Any]) -> float:
        """Compute reward given an event emitted during interaction."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state before a new episode."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: List[ToolCall], label: Optional[str] = None, runtime: bool = False) -> Tuple[str, float, bool]:
        """Apply actions and return (next_observation, reward, terminated)."""
        raise NotImplementedError
